"""2.8.0 spatial SEGS diagnostics: exact geometry, merge-rule replay and the
contract that the diagnostics never change an image or hold a model.

Synthetic SEGS only; nothing detects, segments or samples.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_segs_spatial_diagnostic_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import numpy as np  # noqa: E402
import torch  # noqa: E402

diag = importlib.import_module(f"{TEST_PACKAGE}.utils.segs_diagnostic")
native = importlib.import_module(f"{TEST_PACKAGE}.engine.native_detailer")
stable = importlib.import_module(f"{TEST_PACKAGE}.utils.stable_segs")
merge_mod = importlib.import_module(f"{TEST_PACKAGE}.utils.segs_branch_merge")
advanced = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_detailer_advanced_multimodel")
smart = importlib.import_module(f"{TEST_PACKAGE}.nodes.image.smart_detailer")
boundary = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_module_boundary_cache")
persistent = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_persistent_cache")

CMK_TEST_TIER = "T0"
H, W = 96, 128
SEGSPaste = native.SEGSPaste


def disc_mask(height, width, cy, cx, radius):
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2).float()[None]


def seg(bbox, crop_region, *, radius=None, value=1.0, mask=None):
    x1, y1, x2, y2 = crop_region
    h, w = y2 - y1, x2 - x1
    if mask is None:
        cy = (bbox[1] + bbox[3]) / 2 - y1
        cx = (bbox[0] + bbox[2]) / 2 - x1
        mask = disc_mask(h, w, cy, cx, radius if radius is not None else min(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2)
    crop = torch.full((1, h, w, 3), float(value))
    return native.CMKSEG(crop, mask, 0.9, tuple(crop_region), tuple(bbox), "x", None, None)


def source_image():
    torch.manual_seed(3)
    return torch.rand(1, H, W, 3)


def stable_branch(source, items):
    segs = ((W, H), items)
    image = SEGSPaste.doit(image=source, segs=segs, feather=5, alpha=255)[0]
    return stable.make_stable_segs(segs, source, image, coverage_segs=segs), image


def stable_branch_sized(source, items):
    segs = ((source.shape[2], source.shape[1]), items)
    image = SEGSPaste.doit(image=source, segs=segs, feather=5, alpha=255)[0]
    return stable.make_stable_segs(segs, source, image, coverage_segs=segs), image


def contains_object(value, kinds, seen=None):
    seen = seen if seen is not None else set()
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, kinds):
        return True
    if isinstance(value, dict):
        return any(contains_object(v, kinds, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_object(v, kinds, seen) for v in value)
    return False


def allowed_leaves(value):
    """Diagnostics may only carry image data, numbers and text."""
    if isinstance(value, dict):
        return all(allowed_leaves(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(allowed_leaves(v) for v in value)
    return value is None or isinstance(value, (str, int, float, bool, np.ndarray, torch.Tensor))


class DetailerGeometry(unittest.TestCase):
    def setUp(self):
        # Detection box 20x20, sampling ROI = crop factor 3 around it.
        self.segment = seg((50, 30, 70, 50), (30, 10, 90, 70), radius=8)
        self.segs = ((W, H), [self.segment])

    def test_detection_roi_and_pasteback_differ(self):
        self.assertEqual(diag.bbox_rect(self.segment, W, H), (50, 30, 70, 50))
        self.assertEqual(diag.roi_rect(self.segment, W, H), (30, 10, 90, 70))
        roi = diag.roi_support(self.segs, H, W)
        self.assertEqual(int(roi.sum()), 60 * 60)
        raw = diag.projected_mask(self.segs, H, W, feather=0)
        paste = diag.projected_mask(self.segs, H, W, feather=5)
        # Raw mask: the disc only; pasteback: feathered beyond it, never outside the ROI.
        self.assertEqual(int((raw > 0).sum()), int(disc_mask(60, 60, 30, 30, 8).sum()))
        self.assertGreater(int((paste > 0).sum()), int((raw > 0).sum()))
        self.assertLess(int((paste > 0).sum()), int(roi.sum()))
        self.assertFalse(bool((paste[~roi] > 0).any()))
        self.assertTrue(bool(((paste > 0) & (paste < 1)).any()), "feathering must be visible")

    def test_pasteback_weight_equals_segspaste(self):
        second = seg((40, 40, 60, 60), (25, 25, 85, 80), radius=12)
        for alpha in (255, 128):
            segs = ((W, H), [self.segment, second])
            pasted = SEGSPaste.doit(image=torch.zeros(1, H, W, 3), segs=segs, feather=5, alpha=alpha)[0]
            weight = diag.projected_mask(segs, H, W, feather=5, alpha=alpha)
            self.assertLess(float((pasted[0, :, :, 0] - weight).abs().max()), 1e-6)

    def test_mask_projection_resizes_like_the_paste(self):
        # cropped_mask with a different size than the ROI is resized by SEGSPaste.
        small = seg((50, 30, 70, 50), (30, 10, 90, 70), mask=disc_mask(30, 30, 15, 15, 4))
        segs = ((W, H), [small])
        pasted = SEGSPaste.doit(image=torch.zeros(1, H, W, 3), segs=segs, feather=5, alpha=255)[0]
        self.assertLess(float((pasted[0, :, :, 0] - diag.projected_mask(segs, H, W, feather=5)).abs().max()), 1e-6)

    def test_spatial_stages_are_inserted_once_after_detection(self):
        payload = smart.make_diagnostic_payload(
            title="t", node="n",
            stages=[{"title": f"0{i} {name}", "image": source_image()} for i, name in
                    enumerate(("Source", "Detection", "Detailer", "Difference", "Final"), start=1)],
        )
        result = diag.with_spatial_stages(payload, source_image(), self.segs)
        titles = [stage["title"] for stage in result["stages"]]
        self.assertEqual(titles, ["01 Source", "02 Detection", "03 Sampling ROI", "04 Mask", "05 Pasteback",
                                  "06 Detailer", "07 Difference", "08 Final"])
        self.assertEqual(len(result["preview"]), 8)
        self.assertIs(diag.with_spatial_stages(result, source_image(), self.segs), result)
        self.assertTrue(allowed_leaves(result))
        # Colors: ROI frame orange (dashed), detection frame white.

    def test_roi_and_detection_frames_are_distinct(self):
        # Large canvas so the legend does not cover the probed frame pixels.
        k = 5
        big = torch.rand(1, H * k, W * k, 3)
        segment = seg((50 * k, 30 * k, 70 * k, 50 * k), (30 * k, 10 * k, 90 * k, 70 * k), radius=8 * k)
        stages = diag.detailer_spatial_stages(big, ((W * k, H * k), [segment]))
        roi_image = np.clip(stages[0]["image"][0].numpy() * 255, 0, 255).round().astype(np.uint8)
        # ROI right edge (dashed, first dash), detection bottom edge (solid).
        self.assertEqual(tuple(int(v) for v in roi_image[70 * k - 40, 90 * k - 1]), diag.ROI_COLOR)
        self.assertEqual(tuple(int(v) for v in roi_image[50 * k - 1, 60 * k]), diag.DETECTION_COLOR)
        # Inside the ROI but outside the detection box: original pixels, not dimmed.
        expected = np.clip(big[0, 60 * k, 35 * k].numpy() * 255, 0, 255).round().astype(np.uint8)
        self.assertTrue(np.array_equal(roi_image[60 * k, 35 * k], expected))


MASKED = merge_mod.merge_segs_collections_masked


class MaskedMerge(unittest.TestCase):
    """merge_segs_collections_masked: alpha blend by the effective pasteback mask."""

    def setUp(self):
        self.source = source_image()

    def branches(self, *specs):
        return [stable_branch(self.source, items) for items in specs]

    def test_non_overlapping_masks_keep_both_results(self):
        (d1, img1), (d2, img2) = self.branches(
            [seg((8, 50, 24, 66), (0, 34, 40, 82), value=0.1)],
            [seg((92, 20, 108, 36), (76, 4, 124, 52), value=0.9)])
        merged = MASKED(self.source, [d1, d2])
        for segs, image in ((d1, img1), (d2, img2)):
            area = diag.projected_mask(segs, H, W, feather=5) > 0
            self.assertTrue(torch.equal(merged[0][area], image[0][area]))
        untouched = ~(diag.projected_mask(d1, H, W, feather=5) > 0) & ~(diag.projected_mask(d2, H, W, feather=5) > 0)
        self.assertTrue(torch.equal(merged[0][untouched], self.source[0][untouched]))

    def test_large_overlapping_rois_with_separate_masks_do_not_overwrite(self):
        # Face D2 and hand D3: the ROIs overlap widely, the masks do not.
        (d2, img2), (d3, img3) = self.branches(
            [seg((54, 20, 78, 44), (30, 0, 102, 64), radius=10, value=0.5)],
            [seg((80, 50, 96, 66), (56, 26, 120, 90), radius=6, value=0.9)])
        w2, w3 = (diag.projected_mask(d, H, W, feather=5) for d in (d2, d3))
        self.assertTrue(bool((diag.roi_support(d2, H, W) & diag.roi_support(d3, H, W)).any()))
        self.assertFalse(bool(((w2 > 0) & (w3 > 0)).any()))
        merged = MASKED(self.source, [None, d2, d3])
        self.assertTrue(torch.equal(merged[0][w2 > 0], img2[0][w2 > 0]), "face fully kept")
        self.assertTrue(torch.equal(merged[0][w3 > 0], img3[0][w3 > 0]))
        rest = (w2 == 0) & (w3 == 0)
        self.assertTrue(torch.equal(merged[0][rest], self.source[0][rest]), "ROI pixels outside masks stay source")
        # The legacy rectangle merge (CMK SEGS CONCAT) is unchanged: D3's ROI replaces face pixels.
        legacy = merge_mod.merge_segs_collections(self.source, [None, d2, d3])
        lost = (w2 > 0) & diag.roi_support(d3, H, W)
        self.assertTrue(bool(lost.any()))
        self.assertTrue(torch.equal(legacy[0][lost], img3[0][lost]))

    def test_overlapping_masks_blend_in_slot_order(self):
        specs = ([seg((40, 30, 70, 60), (25, 15, 85, 75), radius=16, value=0.2)],
                 [seg((55, 35, 85, 65), (40, 20, 100, 80), radius=16, value=0.6)],
                 [seg((50, 45, 80, 75), (35, 30, 95, 90), radius=16, value=1.0)])
        branches = self.branches(*specs)
        collections = [b[0] for b in branches]
        merged = MASKED(self.source, collections)
        weights = diag.merge_weights(collections, H, W)
        shares, source_share = diag.alpha_shares(weights, H, W)
        expected = self.source[0] * source_share[..., None]
        for slot, value in ((1, 0.2), (2, 0.6), (3, 1.0)):
            expected = expected + value * shares[slot][..., None]
        self.assertLess(float((merged[0] - expected).abs().max()), 1e-5)
        total = source_share + sum(shares.values())
        self.assertLess(float((total - 1).abs().max()), 1e-6)
        overlap = (weights[2] > 0) & (weights[3] > 0)
        self.assertTrue(bool(overlap.any()))

    def test_single_branch_reproduces_its_detailer_image_including_feather(self):
        (d2, img2), = self.branches([seg((54, 20, 78, 44), (30, 0, 102, 64), radius=10, value=0.5)])
        merged = MASKED(self.source, [None, d2, None])
        self.assertTrue(torch.equal(merged, img2))
        weight = diag.projected_mask(d2, H, W, feather=5)
        self.assertTrue(bool(((weight > 0) & (weight < 1)).any()), "feather edge present")

    def test_several_segs_in_one_detailer_apply_alpha_once(self):
        items = [seg((40, 30, 60, 50), (20, 10, 80, 70), radius=12, value=0.3),
                 seg((50, 35, 70, 55), (30, 15, 90, 75), radius=12, value=0.8)]
        (d1, img1), = self.branches(items)
        self.assertTrue(torch.equal(MASKED(self.source, [d1]), img1))
        # On top of an earlier branch the result is exactly the native paste.
        (d0, img0), = self.branches([seg((44, 30, 64, 50), (24, 10, 84, 70), radius=14, value=0.0)])
        merged = MASKED(self.source, [d0, d1])
        expected = SEGSPaste.doit(image=img0, segs=d1, feather=5, alpha=255)[0]
        self.assertTrue(torch.equal(merged, expected))

    def test_disabled_and_empty_branches_change_nothing(self):
        (d1, img1), = self.branches([seg((8, 50, 24, 66), (0, 34, 40, 82), value=0.1)])
        empty = stable.make_stable_segs(((W, H), []), self.source, self.source)
        self.assertTrue(torch.equal(MASKED(self.source, [None, ((W, H), []), empty]), self.source))
        self.assertTrue(torch.equal(MASKED(self.source, [d1, None, ((W, H), [])]), img1))

    def test_segs_that_do_not_reproduce_their_image_are_rejected(self):
        (d1, img1), = self.branches([seg((8, 50, 24, 66), (0, 34, 40, 82), value=0.1)])
        forged = stable.make_stable_segs(((W, H), list(d1[1])), self.source, img1.clamp(0, 0.5))
        with self.assertRaisesRegex(RuntimeError, "do not reproduce"):
            MASKED(self.source, [forged])


class MergeNode(unittest.TestCase):
    def setUp(self):
        self.source = source_image()
        self.d1, _ = stable_branch(self.source, [seg((8, 50, 24, 66), (0, 34, 40, 82), value=0.1)])
        self.d3, _ = stable_branch(self.source, [seg((92, 46, 108, 62), (76, 30, 124, 78), value=0.9)])
        families = ("SDXL", "None", "Z-Image Turbo")
        self.prompt = {"p": {"class_type": "CMKDetailerPrepareAdvancedMultiModelPipe",
                             "inputs": {f"model_family_{s}": f for s, f in zip((1, 2, 3), families)}},
                       "m": {"class_type": "CMKDetailerMergeMultiModel", "inputs": {}}}
        for slot, node_id in ((1, "d1"), (2, "d2"), (3, "d3")):
            self.prompt[node_id] = {"class_type": "CMK_SmartDetailerMultiModelPipe", "inputs": {"DETAILER": ["p", slot - 1]}}
            self.prompt["m"]["inputs"][f"SEGS {slot}"] = [node_id, 1]

    def run_merge(self):
        sentinel = {"type": "CMK_DIAGNOSTIC", "title": "single detailer", "stages": []}
        return advanced.CMKDetailerMergeMultiModel().merge(
            IMAGE=self.source, prompt=self.prompt, unique_id="m",
            **{"SEGS 1": self.d1, "SEGS 3": self.d3, "diagnostic 1": sentinel, "diagnostic 3": sentinel})

    def test_node_uses_the_mask_merge(self):
        image, _log, _visual, _diag = self.run_merge()
        self.assertTrue(torch.equal(image, MASKED(self.source, [self.d1, ((W, H), []), self.d3])))

    def test_own_four_stage_diagnostic_with_disabled_slot(self):
        image, _log, _visual, diagnostic = self.run_merge()
        self.assertEqual([stage["title"] for stage in diagnostic["stages"]],
                         ["01 · INPUT SEGS", "02 · OVERLAP", "03 · COMPOSITE", "04 · FINAL"])
        self.assertNotIn("single detailer", str(diagnostic))
        self.assertEqual(diagnostic["metadata"]["families"], {"D1": "sdxl", "D2": "none", "D3": "z_image_turbo"})
        self.assertEqual(diagnostic["metrics"]["share_px"]["D2"], 0)
        weights = diag.merge_weights([self.d1, None, self.d3], H, W)
        self.assertAlmostEqual(diagnostic["metrics"]["share_px"]["D1"], round(float(weights[1].sum()), 1), places=1)
        final = diagnostic["stages"][3]["image"]
        self.assertTrue(np.array_equal(final, np.clip(image[0].numpy() * 255, 0, 255).round().astype(np.uint8)))
        self.assertTrue(allowed_leaves(diagnostic))

    def test_composite_colors_are_the_exact_alpha_shares(self):
        k = 6
        big = torch.nn.functional.interpolate(self.source.movedim(-1, 1), scale_factor=k).movedim(1, -1)
        scaled = lambda box: tuple(v * k for v in box)
        specs = (((40, 30, 70, 60), (25, 15, 85, 75)), ((55, 35, 85, 65), (40, 20, 100, 80)), ((50, 45, 80, 75), (35, 30, 95, 90)))
        collections = [stable_branch_sized(big, [seg(scaled(b), scaled(r), radius=16 * k, value=0.5)])[0] for b, r in specs]
        result = MASKED(big, collections)
        stages, metrics = diag.merge_diagnostic_stages(big, result, collections, {1: "sdxl", 2: "z_image_turbo", 3: "sdxl"})
        composite = stages[2]["image"][0].numpy()
        weights = diag.merge_weights(collections, big.shape[1], big.shape[2])
        shares, source_share = diag.alpha_shares(weights, big.shape[1], big.shape[2])
        gray = big[0].mean(dim=-1).numpy() * 255 * 0.4
        expected = gray * source_share.numpy()
        expected = expected[..., None].repeat(3, axis=2)
        for slot, share in shares.items():
            expected = expected + np.asarray(diag.SLOT_COLORS[slot], dtype=np.float32) * share.numpy()[..., None]
        region = (slice(40 * k, 90 * k), slice(60 * k, 100 * k))  # below the legend
        # uint8 rounding of the source and of the stage image only.
        self.assertLess(float(np.abs(composite[region] * 255 - expected[region]).max()), 1.5)
        overlap = (weights[2] > 0) & (weights[3] > 0)
        self.assertEqual(metrics["mask_overlap_px"]["D2xD3"], int(overlap.sum()))
        self.assertGreater(metrics["reduced_by_later_px"]["D2"], 0)

    def test_no_model_objects_in_diagnostics(self):
        class Model:  # stands for any model/patcher object
            pass
        _image, _log, _visual, diagnostic = self.run_merge()
        self.assertFalse(contains_object(diagnostic, (Model,)))
        self.assertTrue(allowed_leaves(diagnostic))


class BoundaryKey(unittest.TestCase):
    def prompt(self, merge_class):
        return {"img": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
                "m": {"class_type": merge_class, "inputs": {"IMAGE": ["img", 0], "feather": 5, "alpha": 255}},
                "b": {"class_type": "CMKDetailerBoundaryCache", "inputs": {"IMAGE": ["m", 0], "LOG": ["m", 1]}}}

    def test_merge_rule_is_part_of_the_advanced_key_only(self):
        core = boundary._DetailerBoundaryCore()
        advanced_prompt = self.prompt("CMKDetailerMergeMultiModel")
        key = core._cache_key(advanced_prompt, "b")[0]
        original = advanced.CMKDetailerMergeMultiModel.MERGE_SEMANTICS
        self.addCleanup(setattr, advanced.CMKDetailerMergeMultiModel, "MERGE_SEMANTICS", original)
        advanced.CMKDetailerMergeMultiModel.MERGE_SEMANTICS = "rectangle_v0"
        self.assertNotEqual(core._cache_key(dict(advanced_prompt), "b")[0], key, "old merge rule must not hit")
        legacy_prompt = self.prompt("CMK_SEGSConcate")
        expected = persistent.build_node_fingerprint(
            legacy_prompt, "b", ("CMKDetailerBoundaryCache",), core._SCHEMA,
            exclude_inputs=boundary.VISUAL_INPUT_NAMES + boundary.PASSTHROUGH_INPUT_NAMES, include_node_identity=True)
        self.assertEqual(core._cache_key(dict(legacy_prompt), "b"), expected, "other Detailer keys unchanged")


class CachedDetailerDiagnostic(unittest.TestCase):
    def test_cache_hit_completes_old_diagnostic_and_keeps_image(self):
        source = source_image()
        branch, image = stable_branch(source, [seg((50, 30, 70, 50), (30, 10, 90, 70), radius=8)])
        old = smart.make_diagnostic_payload(
            title="Smart Detailer", node="CMK Smart Detailer",
            stages=[{"title": "01 Source", "image": source}, {"title": "02 Detection", "image": source},
                    {"title": "03 Detailer", "image": image}],
        )
        payload = {"segs_detected": branch, "segs_proceed": branch, "image_proceed": image,
                   "diagnostic": old, "log_block": "", "enabled": True}
        originals = (smart.load_pickle, smart.write_status)
        self.addCleanup(lambda: (setattr(smart, "load_pickle", originals[0]), setattr(smart, "write_status", originals[1])))
        smart.load_pickle = lambda scope, key: payload
        smart.write_status = lambda *a, **k: None
        node = smart.CMK_SmartDetailerPipe()
        node._publish_dependency_revision = lambda *a, **k: None
        result = node._load_cached_result("k" * 64, True, source, unique_id="7")
        self.assertIs(result[2], image, "cached image returned unchanged")
        self.assertIs(result[6], image)
        titles = [stage["title"] for block in result[5].get("groups", []) for stage in block.get("stages", [])] \
            or [stage["title"] for stage in result[5]["stages"]]
        for name in ("Sampling ROI", "Mask", "Pasteback"):
            self.assertTrue(any(title.endswith(name) for title in titles), (name, titles))


if __name__ == "__main__":
    unittest.main()
