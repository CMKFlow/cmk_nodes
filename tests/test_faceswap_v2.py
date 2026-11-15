"""FaceSwap v2 (2.8): MODE control and the 2.8 diagnostic of the Standard and
Advanced modules, against the unchanged FaceSwap Image -Pipe-.

The real executor (selection, BLEND, quality profile, SEGS, LOG, VISUAL), the
real SEGS CONCAT merge and the diagnostics run; only face detection and the
swap inference are replaced (the "swap" inverts the target inside a feathered
face ellipse and returns that ellipse as paste mask).  These are dummy runs,
not real FaceSwap runs.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_faceswap_v2_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import numpy as np  # noqa: E402
import torch  # noqa: E402

v2 = importlib.import_module(f"{TEST_PACKAGE}.nodes.swap.face_swap_v2")
legacy = importlib.import_module(f"{TEST_PACKAGE}.nodes.swap.face_swap")
concat = importlib.import_module(f"{TEST_PACKAGE}.nodes.image.segs_concate")
detector = importlib.import_module(f"{TEST_PACKAGE}.engine.detector_engine")
swap_engine = importlib.import_module(f"{TEST_PACKAGE}.engine.swap_selected_engine")
output = importlib.import_module(f"{TEST_PACKAGE}.utils.comfy_preview_output")
segs_diagnostic = importlib.import_module(f"{TEST_PACKAGE}.utils.segs_diagnostic")
prepare_card = importlib.import_module(f"{TEST_PACKAGE}.utils.prepare_card")
visual = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_visual")
boundary = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_module_boundary_cache")
mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")

CMK_TEST_TIER = "T0"
H, W = 320, 480
LEFT, CENTER, RIGHT = [40, 90, 120, 190], [200, 80, 290, 200], [340, 70, 450, 210]
FACES = [{"bbox": LEFT}, {"bbox": CENTER}, {"bbox": RIGHT}]
torch.manual_seed(5)
TARGET = ((torch.rand(1, H, W, 3) * 0.6 + 0.2) * 255).round() / 255   # 8-bit like a loaded image
SOURCE = torch.rand(1, 96, 96, 3)
PROCESS = {"content_guard_enabled": False, "source_model_family": "sdxl"}


def face(bbox, score=0.9):
    x1, y1, x2, y2 = bbox
    kps = [[x1 + (x2 - x1) * fx, y1 + (y2 - y1) * fy] for fx, fy in ((.3, .4), (.7, .4), (.5, .6), (.35, .8), (.65, .8))]
    return {"bbox": list(map(float, bbox)), "det_score": score, "kps": kps}


def ellipse(bbox):
    x1, y1, x2, y2 = bbox
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy, rx, ry = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 0.6, (y2 - y1) * 0.6
    distance = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    return np.clip((1.15 - distance) / 0.3, 0.0, 1.0)


def values(**overrides):
    result = {"IMAGE_TARGET": TARGET, "IMAGE_SOURCE": SOURCE, "PROCESS": PROCESS, "GLOBAL ENABLE": True, "ENABLE": True,
              "SWAP MODEL": "fake.onnx", "DETECT MODEL": "fake", "TARGET FACE": "Center", "SOURCE FACE": "Largest",
              "PATCH ENHANCER": False, "QUALITY PROFILE": "Automatic", "POST RESTORE": False,
              "POST RESTORE MODEL": "none", "unique_id": "7"}
    result.update(overrides)
    return result


def rgb(stage):
    value = stage["image"]
    value = value[0] if getattr(value, "ndim", 0) == 4 else value
    value = torch.as_tensor(np.asarray(value)).float()
    return value / 255.0 if value.max() > 1.0 else value


def visible(payload):
    return [step["title"] for step in output._preview_steps_from_payload(payload)]


def card_fields(card):
    return {label.strip(): value.strip() for label, _sep, value in
            (line.partition(" : ") for line in card["summary"].splitlines()) if _sep and label.strip()}


class Fakes(unittest.TestCase):
    def setUp(self):
        def patch(owner, name, value):
            original = getattr(owner, name)
            self.addCleanup(setattr, owner, name, original)
            setattr(owner, name, value)

        self.swaps = []
        swaps = self.swaps
        patch(detector.CMKDetectorEngine, "detect_image",
              lambda self, image, settings: [face(item["bbox"]) for item in FACES] if image.shape[0] == H else [face([10, 10, 80, 80])])

        def swap(self, *, target_rgb, source_rgb, source_selected_face, target_selected_face, settings, content_guard_enabled=True):
            bbox = target_selected_face["face"]["bbox"]
            swaps.append([int(v) for v in bbox])
            mask = ellipse(bbox)
            raw = target_rgb.astype(np.float32) * (1 - mask[..., None]) + (255 - target_rgb.astype(np.float32)) * mask[..., None]
            return raw.clip(0, 255).astype(np.uint8), mask

        patch(swap_engine.CMKSelectedSwapEngine, "swap_selected_with_mask", swap)


class ModeAndRegistration(unittest.TestCase):
    def test_registration(self):
        for name in ("CMKFaceSwapImagePipeV2", "CMKFaceSwapSlotPipeV2", "CMKFaceSwapAdvancedDiagnosticV2",
                     "CMKFaceSwapModeEnable"):
            self.assertIn(name, mappings.NODE_CLASS_MAPPINGS)
            self.assertIn(name, mappings.NODE_DISPLAY_NAME_MAPPINGS)
        for name in ("CMKFaceSwapImagePipeV2", "CMKFaceSwapSlotPipeV2"):
            node = mappings.NODE_CLASS_MAPPINGS[name]
            self.assertEqual(node.RETURN_NAMES, legacy.CMKFaceSwapImagePipe.RETURN_NAMES)
            self.assertEqual(node.INPUT_TYPES.__func__, legacy.CMKFaceSwapImagePipe.INPUT_TYPES.__func__,
                             "same inputs: GLOBAL ENABLE is fed by CMK FaceSwap · Mode")
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKFaceSwapImagePipe"], legacy.CMKFaceSwapImagePipe)

    def test_mode_reuses_the_facerebuild_mode(self):
        node = v2.CMKFaceSwapModeEnable
        self.assertEqual(node.INPUT_TYPES()["required"]["faceswap_mode"], (["ENABLE", "DISABLE"], {"default": "ENABLE"}))
        self.assertEqual(node.read("ENABLE"), (True,))
        self.assertEqual(node.read("DISABLE"), (False,))
        self.assertEqual((node.RETURN_TYPES, node.RETURN_NAMES), (("BOOLEAN",), ("ENABLE",)))

    def test_shared_lookups_know_the_v2_executors(self):
        for name in ("CMKFaceSwapImagePipeV2", "CMKFaceSwapSlotPipeV2"):
            self.assertTrue(visual._matches_live_type(name, "CMKFaceSwapImagePipe"), "SEGS CONCAT publishes the provider")
        self.assertFalse(visual._matches_live_type("CMKFaceProcessPipe", "CMKFaceSwapImagePipe"))

    def test_static_bypass_detection(self):
        def prompt(enable):
            return {
                "1": {"class_type": "CMKFaceSwapSlotPipeV2", "inputs": {"IMAGE_TARGET": ["0", 0], "GLOBAL ENABLE": ["9", 0],
                                                                         "ENABLE": enable}},
                "9": {"class_type": "CMKFaceSwapModeEnable", "inputs": {"faceswap_mode": "ENABLE"}},
                "5": {"class_type": "CMKFaceSwapBoundaryCache", "inputs": {"IMAGE": ["1", 0], "LOG": ["1", 2]}},
            }
        self.assertTrue(boundary._faceswap_disabled_state(prompt(False), "5")[0], "every slot locally off")
        self.assertFalse(boundary._faceswap_disabled_state(prompt(True), "5")[0])


class Standard(Fakes):
    def run_both(self, **overrides):
        return (legacy.CMKFaceSwapImagePipe().run_pipe(**values(**overrides)),
                v2.CMKFaceSwapImagePipeV2().run_pipe(**values(**overrides)))

    def test_processing_is_unchanged(self):
        old, new = self.run_both()
        self.assertTrue(torch.equal(old[0], new[0]), "same IMAGE")
        self.assertTrue(torch.equal(old[1].cmk_branch_mask, new[1].cmk_branch_mask), "same SEGS")
        self.assertEqual(old[2], new[2], "same LOG")
        self.assertEqual([(p["module_type"], p["stage_key"]) for p in old[3]["providers"]],
                         [(p["module_type"], p["stage_key"]) for p in new[3]["providers"]], "same VISUAL provider")
        self.assertEqual(self.swaps, [CENTER, CENTER])

    def test_legacy_diagnostic_is_unchanged(self):
        old, _new = self.run_both()
        self.assertEqual(old[4]["title"], "CMK Flow · FaceSwap")
        group = old[4]["metadata"]["diagnostic_groups"][0]
        self.assertEqual([stage["title"] for stage in group["stages"]],
                         ["01 Target", "02 Source", "03 Swap Result", "04 Difference", "05 Final"])

    def test_stages(self):
        _old, new = self.run_both()
        self.assertEqual(new[4]["title"], "CMK Flow · FaceSwap")
        prepare, process = new[4]["metadata"]["diagnostic_groups"]
        self.assertEqual((prepare["node"], process["node"]), ("CMK FaceSwap v2 · Prepare", "CMK FaceSwap v2"))
        self.assertEqual(prepare["title"], "FaceSwap · Prepare")
        self.assertEqual(prepare["metadata"][output.SUMMARY_CARD_LAYOUT], output.SUMMARY_CARD_FOOTER)
        self.assertEqual(visible(prepare), [], "Prepare is a text card")
        fields = card_fields(prepare)
        self.assertEqual(fields["TARGET"], "Center · 1 of 3 faces")
        self.assertEqual(fields["FACE"], "90 × 120 px · score 90%")
        self.assertEqual(fields["SOURCE"], "Largest · 1 face")
        self.assertEqual(fields["PROFILE"], "Automatic → SDXL")
        self.assertEqual(fields["PASTE"], "blend 1.00 · feather 22")
        self.assertEqual(fields["CROP"], "factor 1.5 · dilation 0")
        self.assertEqual(fields["RESTORE"], "OFF")
        titles = [stage["title"] for stage in process["stages"]]
        self.assertEqual(titles, ["01 Swap Face", "02 Paste Mask", "03 Pasteback", "04 Difference"])
        self.assertEqual(visible(process), titles, "all spatial stages are rendered")

    def test_geometry_is_the_run_geometry(self):
        _old, new = self.run_both()
        _prepare, process = new[4]["metadata"]["diagnostic_groups"]
        support = new[1].cmk_branch_mask[0, ..., 0].numpy()
        self.assertTrue(np.array_equal(support, ellipse(CENTER) > 0), "paste support = paste mask x BLEND > 0")
        swap = rgb(process["stages"][0])
        white = torch.tensor(segs_diagnostic.DETECTION_COLOR, dtype=torch.float32) / 255.0
        x1, y1, x2, y2 = CENTER
        self.assertTrue(torch.allclose(swap[y1 + 1, (x1 + x2) // 2], white, atol=0.01), "target face box at the face")
        rows, cols = np.flatnonzero(support.any(axis=1)), np.flatnonzero(support.any(axis=0))
        self.assertIn("region", process["stages"][0]["subtitle"])
        self.assertIn(f"region {cols[-1] + 1 - cols[0]} × {rows[-1] + 1 - rows[0]}", process["stages"][0]["subtitle"])
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        self.assertTrue(torch.allclose(rgb(process["stages"][2])[cy, cx], new[0][0, cy, cx], atol=0.01),
                        "inside the paste mask: the swapped result")
        self.assertIn(f"{int(support.sum())} px", process["stages"][1]["subtitle"])

    def test_post_restore_and_profile_in_the_card(self):
        legacy_module = legacy
        original = legacy_module.RestoreFaceAdvanced
        self.addCleanup(setattr, legacy_module, "RestoreFaceAdvanced", original)
        legacy_module.RestoreFaceAdvanced = lambda: types.SimpleNamespace(execute=lambda **kw: (kw["image"],))
        new = v2.CMKFaceSwapImagePipeV2().run_pipe(**values(**{"POST RESTORE": True, "POST RESTORE MODEL": "GFPGANv1.4.pth",
                                                               "PROCESS": {**PROCESS, "source_model_family": "z_image_turbo"}}))
        prepare, process = new[4]["metadata"]["diagnostic_groups"]
        fields = card_fields(prepare)
        self.assertEqual(fields["RESTORE"], "GFPGANv1.4.pth · visibility 1.00")
        self.assertEqual(fields["PROFILE"], "Automatic → ZIT")
        self.assertEqual(fields["PASTE"], "blend 0.70 · feather 30")
        self.assertEqual(fields["CROP"], "factor 1.3 · dilation 0")
        self.assertIn("restore GFPGANv1.4.pth", process["stages"][2]["subtitle"])

    def test_no_difference_without_change(self):
        swap_engine.CMKSelectedSwapEngine.swap_selected_with_mask = (
            lambda self, *, target_rgb, settings, **kw: (target_rgb.copy(), ellipse(CENTER)))
        new = v2.CMKFaceSwapImagePipeV2().run_pipe(**values())
        titles = [stage["title"] for stage in new[4]["metadata"]["diagnostic_groups"][-1]["stages"]]
        self.assertEqual(titles, ["01 Swap Face", "02 Paste Mask", "03 Pasteback"])

    def test_disabled_run_is_a_text_card(self):
        new = v2.CMKFaceSwapImagePipeV2().run_pipe(**values(ENABLE=False, IMAGE_SOURCE=None))
        self.assertIs(new[0], TARGET)
        self.assertEqual(self.swaps, [])
        card = new[4]
        self.assertEqual(card["title"], "FaceSwap · Bypass")
        self.assertEqual(visible(new[4]), [], "no placeholder image")
        self.assertIn("IMAGE TARGET PASSTHROUGH", card["summary"])


class Advanced(Fakes):
    """The Advanced subgraph: three slot executors chained F1 → F2 → F3 on the
    target, SEGS CONCAT with the SEGS in the subgraph's input order
    (segs = F1, segs_2 = F3, segs_3 = F2), then the v2 diagnostic."""

    def run_slots(self, slots=(("Leftmost", False), ("Center", True), ("Rightmost", True)), node=None):
        node = node or v2.CMKFaceSwapSlotPipeV2
        chain, results = None, []
        for target, enable in slots:
            result = node().run_pipe(**values(**{"TARGET FACE": target, "ENABLE": enable, "opt_diagnostic": chain,
                                                 "IMAGE_SOURCE": SOURCE if enable else None}))
            chain = result[4]
            results.append(result)
        segs = [results[0][1], results[2][1], results[1][1]]
        merged = concat.CMK_SEGSConcate().concate(TARGET, segs[0], segs_2=segs[1], segs_3=segs[2])[0]
        return results, merged, segs

    def diagnose(self, results, merged, segs):
        return v2.CMKFaceSwapAdvancedDiagnosticV2().diagnose(
            TARGET, results[-1][4], segs=segs[0], segs_2=segs[1], segs_3=segs[2], **{"IMAGE MERGED": merged})[0]

    def test_processing_is_unchanged(self):
        old, old_merged, _ = self.run_slots(node=legacy.CMKFaceSwapImagePipe)
        new, new_merged, _ = self.run_slots()
        self.assertTrue(torch.equal(old_merged, new_merged), "same merged IMAGE")
        self.assertEqual(old[-1][2], new[-1][2], "same LOG")
        for before, after in zip(old, new):
            self.assertTrue(torch.equal(before[0], after[0]))

    def test_stages_follow_the_merge(self):
        results, merged, segs = self.run_slots()
        payload = self.diagnose(results, merged, segs)
        self.assertEqual(payload["title"], "CMK Flow · FaceSwap · Advanced")
        *cards, process = payload["metadata"]["diagnostic_groups"]
        self.assertEqual(process["node"], "CMK FaceSwap v2 · Advanced")
        self.assertEqual(process["metadata"]["order"], [3, 2], "SEGS CONCAT input order: F3 (segs_2) before F2 (segs_3)")
        self.assertEqual([card["title"] for card in cards], ["FaceSwap · Prepare", "Slot 2 · Prepare", "Slot 3 · Prepare"],
                         "module card, then the active slots; no card for the disabled slot")
        module = card_fields(cards[0])
        self.assertEqual(module["SLOTS"], "F1 · Leftmost · OFF")
        self.assertIn("F2 · Center · ON", cards[0]["summary"])
        self.assertEqual(module["MERGE"], "F3 → F2 (SEGS CONCAT order, later on top)")
        self.assertTrue(card_fields(cards[1])["TARGET"].startswith("Center"))
        self.assertTrue(card_fields(cards[2])["TARGET"].startswith("Rightmost"))
        self.assertEqual([visible(card) for card in cards], [[], [], []])
        titles = [stage["title"] for stage in process["stages"]]
        self.assertEqual(titles, ["01 Target Faces", "02 Paste Masks", "03 Merge", "04 Difference"],
                         "no Overlap stage: the paste masks do not overlap")
        self.assertEqual(visible(process), titles)
        self.assertEqual(process["stages"][0]["subtitle"], "F3 rightmost · F2 center")

    def test_merge_view_is_the_merged_image(self):
        results, merged, segs = self.run_slots()
        process = self.diagnose(results, merged, segs)["metadata"]["diagnostic_groups"][-1]
        x1, y1, x2, y2 = RIGHT
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        self.assertTrue(torch.allclose(rgb(process["stages"][2])[cy, cx], merged[0, cy, cx], atol=0.01))

    def test_overlap_only_when_paste_masks_overlap(self):
        global FACES
        saved = FACES
        FACES = [{"bbox": LEFT}, {"bbox": [200, 80, 300, 200]}, {"bbox": [270, 85, 370, 205]}]
        self.addCleanup(lambda: globals().__setitem__("FACES", saved))
        results, merged, segs = self.run_slots()
        process = self.diagnose(results, merged, segs)["metadata"]["diagnostic_groups"][-1]
        titles = [stage["title"] for stage in process["stages"]]
        self.assertEqual(titles[3], "04 Overlap")
        self.assertTrue(process["stages"][3]["subtitle"].endswith("px · later slot on top"))

    def test_no_active_slot(self):
        results, merged, segs = self.run_slots(slots=(("Leftmost", False), ("Center", False), ("Rightmost", False)))
        self.assertTrue(torch.equal(merged, TARGET))
        payload = self.diagnose(results, merged, segs)
        card = payload
        self.assertEqual(card["title"], "FaceSwap · Prepare")
        self.assertEqual(card_fields(card)["MERGE"], "no active slot")
        self.assertEqual(visible(payload), [], "text card only")


class Rendering(Fakes):
    """The rendered Preview Render timeline: Prepare cards do not resize the image cards."""

    def check(self, payload):
        *cards, process = payload["metadata"]["diagnostic_groups"]
        for card in cards:
            lines = card["summary"].splitlines()
            self.assertLessEqual(len(lines), prepare_card.CARD_ROWS)
            for line in lines:
                self.assertLessEqual(len(line), prepare_card.CARD_COLUMNS, line)
        alone = output.render_preview_panel(process, caption="Details")
        timeline = output.render_preview_panel(payload, caption="Details")
        self.assertEqual(timeline.shape[0], alone.shape[0], "the image cards keep their height")

    def test_standard(self):
        self.check(v2.CMKFaceSwapImagePipeV2().run_pipe(**values())[4])

    def test_advanced(self):
        case = Advanced()
        results, merged, segs = case.run_slots()
        self.check(case.diagnose(results, merged, segs))


if __name__ == "__main__":
    unittest.main()
