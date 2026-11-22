"""Regional layout fidelity: joint mask profile, native AREA crop, cache revision.

Model-free.  The native part runs ComfyUI's own conditioning code
(ConditioningSetAreaPercentage, ConditioningSetMask, area/mask resolution and
_calc_cond_batch) against a recording stub model, so the UNet input shapes of
every regional pass are observed directly.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
import types
import unittest
import uuid
from unittest.mock import patch

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_regional_edge_weighting"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

weighting = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_regional_weighting")
persistent = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_persistent_cache")
refiner_cache = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_refiner_boundary_cache")
instantid_conditioning = importlib.import_module(f"{TEST_PACKAGE}.pipe.instantid.cmk_instantid_conditioning")

import comfy.samplers as samplers  # noqa: E402
import nodes  # noqa: E402


# The three neighbouring regions of "CMK 2.8 · Text2Image · Identity 3" (1344x768).
REFERENCE = [
    {"x": 0.030386214343660303, "y": 0.31661799159967896, "width": 0.25349900290229654,
     "height": 0.6819645840660431, "strength": 2.0},
    {"x": 0.2905306227352141, "y": 0.21378355688591333, "width": 0.2899562833511512,
     "height": 0.7862164431140867, "strength": 1.4},
    {"x": 0.5961140489129182, "y": 0.3040214198356601, "width": 0.3560135182650426,
     "height": 0.6959785801643399, "strength": 1.4},
]
LATENT_W, LATENT_H = 1344 // 8, 768 // 8


def box_distance(box, px, py):
    x0, y0, x1, y1 = box
    dx = max(x0 - px, 0, px - (x1 - 1))
    dy = max(y0 - py, 0, py - (y1 - 1))
    return (dx * dx + dy * dy) ** 0.5


def comfy_area(region):
    """The integer AREA ComfyUI derives from the user's percentage rectangle."""
    cond = {"area": ("percentage", region["height"], region["width"], region["y"], region["x"])}
    conds = [cond]
    samplers.resolve_areas_and_cond_masks_multidim(conds, (LATENT_H, LATENT_W), "cpu")
    return conds[0]["area"]


def native_conditioning(region, profile):
    cond = [[torch.zeros(1, 77, 2048), {"pooled_output": torch.zeros(1, 1280)}]]
    x, y, width, height = profile.crop_percent
    cond = nodes.ConditioningSetAreaPercentage().append(cond, width, height, x, y, region["strength"])[0]
    return nodes.ConditioningSetMask().append(cond, profile.mask, "default", 1.0)[0]


def sampler_cond(metadata):
    cond = dict(metadata)
    cond["model_conds"] = {}
    cond["uuid"] = uuid.uuid4()
    return cond


class Patcher:
    def prepare_state(self, *args, **kwargs):
        pass

    def get_free_memory(self, *args):
        return 1 << 40

    def apply_hooks(self, hooks=None):
        return {}


class ReferenceProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = weighting.build_region_weights(REFERENCE, LATENT_W, LATENT_H)
        cls.masks = torch.stack([profile.mask[0] for profile in cls.profiles])

    def test_rectangles_are_rasterised_exactly_like_comfyui(self):
        for region, profile in zip(REFERENCE, self.profiles):
            height, width, y, x = comfy_area(region)
            self.assertEqual(profile.box, (x, y, x + width, y + height))
        self.assertEqual([p.box for p in self.profiles], [(5, 30, 48, 95), (49, 21, 98, 96), (100, 29, 160, 96)])

    def test_user_rectangles_keep_full_weight_up_to_the_seam_cells(self):
        for profile in self.profiles:
            self.assertGreaterEqual(profile.min_weight, 0.75)
            self.assertGreater(profile.mean_weight, 0.99)

    def test_foreign_share_is_limited_to_the_outermost_seam_cells(self):
        boxes = [p.box for p in self.profiles]
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            for py in range(y0, y1):
                for px in range(x0, x1):
                    foreign = float(self.masks[:, py, px].sum() - self.masks[index, py, px])
                    self.assertLessEqual(foreign, 0.25 + 1e-6)
                    nearest = min(box_distance(b, px, py) for j, b in enumerate(boxes) if j != index)
                    if nearest > weighting.SHARED_CROSSFADE:
                        self.assertEqual(float(self.masks[index, py, px]), 1.0)
                        self.assertEqual(foreign, 0.0)

    def test_left_and_middle_seam_crossfades_over_three_columns(self):
        row = 60
        self.assertEqual([round(float(v), 2) for v in self.masks[0, row, 46:51]], [1.0, 0.75, 0.5, 0.25, 0.0])
        self.assertEqual([round(float(v), 2) for v in self.masks[1, row, 46:51]], [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_middle_and_right_regions_do_not_enter_each_other(self):
        row = 60
        self.assertEqual(float(self.masks[1, row, 97]), 1.0)
        self.assertEqual(float(self.masks[2, row, 100]), 1.0)
        self.assertAlmostEqual(float(self.masks[1:, row, 98].sum()), 1.0, places=5)
        self.assertAlmostEqual(float(self.masks[1:, row, 99].sum()), 1.0, places=5)

    def test_free_space_ramp_lies_outside_the_rectangle(self):
        self.assertEqual([round(float(v), 2) for v in self.masks[0, 60, 0:5]], [0.0, 0.2, 0.4, 0.6, 0.8])
        self.assertEqual([round(float(v), 2) for v in self.masks[1, 16:21, 70]], [0.0, 0.2, 0.4, 0.6, 0.8])
        boxes = [p.box for p in self.profiles]
        for py in range(LATENT_H):
            for px in range(LATENT_W):
                if all(box_distance(b, px, py) > 0 for b in boxes):
                    self.assertLessEqual(float(self.masks[:, py, px].sum()), 1.0 + 1e-6)

    def test_crop_covers_the_weighted_support_plus_context(self):
        expected = [(0, 24, 52, 96), (43, 15, 104, 96), (96, 23, 166, 96)]
        self.assertEqual([p.crop for p in self.profiles], expected)
        for profile in self.profiles:
            cx0, cy0, cx1, cy1 = profile.crop
            outside = profile.mask[0].clone()
            outside[cy0:cy1, cx0:cx1] = 0
            self.assertEqual(float(outside.sum()), 0.0)

    def test_user_geometry_is_not_written(self):
        before = [dict(region) for region in REFERENCE]
        weighting.build_region_weights(REFERENCE, LATENT_W, LATENT_H)
        self.assertEqual(REFERENCE, before)


class GeometryEdgeCases(unittest.TestCase):
    def test_touching_rectangles_stay_below_the_foreign_limit(self):
        regions = [
            {"x": 0.0, "y": 0.0, "width": 0.5, "height": 1.0},
            {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0},
        ]
        profiles = weighting.build_region_weights(regions, 64, 32)
        left, right = profiles[0].mask[0], profiles[1].mask[0]
        self.assertAlmostEqual(float(right[10, 31]), 0.25, places=5)
        self.assertEqual(float(right[10, 29]), 0.0)
        self.assertTrue(torch.allclose(left + right, torch.ones_like(left)))
        self.assertLessEqual(max(p.max_foreign_weight for p in profiles), 0.25 + 1e-6)

    def test_narrow_rectangle_between_two_touching_neighbours(self):
        regions = [
            {"x": 0.0, "y": 0.0, "width": 0.45, "height": 1.0},
            {"x": 0.45, "y": 0.0, "width": 0.05, "height": 1.0},
            {"x": 0.5, "y": 0.0, "width": 0.5, "height": 1.0},
        ]
        profiles = weighting.build_region_weights(regions, 40, 16)
        middle = profiles[1]
        self.assertGreaterEqual(middle.min_weight, 0.75 - 1e-6)
        self.assertLessEqual(middle.max_foreign_weight, 0.25 + 1e-6)

    def test_user_overlap_keeps_both_regions_at_full_weight(self):
        regions = [
            {"x": 0.0, "y": 0.0, "width": 0.6, "height": 1.0},
            {"x": 0.4, "y": 0.0, "width": 0.6, "height": 1.0},
        ]
        a, b = (p.mask[0] for p in weighting.build_region_weights(regions, 50, 10))
        self.assertEqual(float(a[5, 25]), 1.0)
        self.assertEqual(float(b[5, 25]), 1.0)

    def test_full_frame_region_uses_the_whole_latent_without_ramp(self):
        profile = weighting.build_region_weights([{"x": 0, "y": 0, "width": 1, "height": 1}], 32, 24)[0]
        self.assertTrue(torch.equal(profile.mask, torch.ones(1, 24, 32)))
        self.assertEqual(profile.crop, (0, 0, 32, 24))

    def test_no_regions_builds_nothing(self):
        self.assertEqual(weighting.build_region_weights([], 32, 24), [])


class NativeSamplingGeometry(unittest.TestCase):
    """ConditioningSetAreaPercentage + ConditioningSetMask through ComfyUI's sampler path."""

    @classmethod
    def setUpClass(cls):
        cls.profiles = weighting.build_region_weights(REFERENCE, LATENT_W, LATENT_H)
        metadata = [native_conditioning(r, p)[0][1] for r, p in zip(REFERENCE, cls.profiles)]
        cls.conds = [sampler_cond({})] + [sampler_cond(m) for m in metadata]
        samplers.resolve_areas_and_cond_masks_multidim(cls.conds, (LATENT_H, LATENT_W), "cpu")

    def test_metadata_carries_crop_mask_and_unchanged_strength(self):
        for region, profile, cond in zip(REFERENCE, self.profiles, self.conds[1:]):
            cx0, cy0, cx1, cy1 = profile.crop
            self.assertEqual(cond["area"], (cy1 - cy0, cx1 - cx0, cy0, cx0))
            self.assertEqual(tuple(cond["mask"].shape), (1, LATENT_H, LATENT_W))
            self.assertFalse(cond["set_area_to_bounds"])
            self.assertEqual(cond["strength"], region["strength"])
            self.assertEqual(cond["mask_strength"], 1.0)

    def test_each_region_runs_the_unet_on_its_bounded_crop_only(self):
        shapes = []
        model = types.SimpleNamespace(
            current_patcher=Patcher(),
            memory_required=lambda shape, cond_shapes=None: 0,
            apply_model=lambda x, t, **c: (shapes.append(tuple(x.shape)), torch.ones_like(x))[1],
        )
        x = torch.zeros(1, 4, LATENT_H, LATENT_W)
        samplers._calc_cond_batch(model, [self.conds], x, torch.tensor([500.0]), {})
        full_frame = (1, 4, LATENT_H, LATENT_W)
        crops = sorted((1, 4, p.crop[3] - p.crop[1], p.crop[2] - p.crop[0]) for p in self.profiles)
        self.assertEqual(shapes.count(full_frame), 1)  # the global conditioning only
        self.assertEqual(sorted(s for s in shapes if s != full_frame), crops)

    def test_native_edge_falloff_is_replaced_by_the_mask_profile(self):
        x = torch.zeros(1, 4, LATENT_H, LATENT_W)
        for region, profile, cond in zip(REFERENCE, self.profiles, self.conds[1:]):
            result = samplers.get_area_and_mult(cond, x, torch.tensor([500.0]))
            cx0, cy0, cx1, cy1 = profile.crop
            expected = profile.mask[0, cy0:cy1, cx0:cx1] * region["strength"]
            self.assertTrue(torch.allclose(result.mult[0, 0], expected))
            x0, y0, x1, y1 = profile.box
            rim = result.mult[0, 0, y0 - cy0, x0 - cx0:x1 - cx0]
            self.assertGreaterEqual(float(rim.min()), 0.75 * region["strength"] - 1e-6)

    def test_uncond_copy_keeps_the_same_crop_and_mask(self):
        negative = [sampler_cond({})]
        for cond in self.conds[1:]:
            samplers.create_cond_with_same_area_if_none(negative, cond)
        self.assertEqual(len(negative), 4)
        for copy, cond in zip(negative[1:], self.conds[1:]):
            self.assertEqual(copy["area"], cond["area"])
            self.assertIs(copy["mask"], cond["mask"])

    def test_instantid_hybrid_still_receives_semantics_only(self):
        source = [native_conditioning(r, p)[0] for r, p in zip(REFERENCE, self.profiles)]
        result = instantid_conditioning._hybrid_positive_conditioning(
            source, instantid_mode="HYBRID", zero_pass_mode=False,
        )
        for (tensor, metadata), (source_tensor, _) in zip(result, source):
            self.assertIs(tensor, source_tensor)
            self.assertTrue({"area", "mask", "mask_strength", "strength", "set_area_to_bounds"}.isdisjoint(metadata))


def chain_prompt(with_regional: bool) -> dict:
    prompt = {
        "1": {"class_type": "CMKPipeCreateImage", "inputs": {"prompt": "station"}},
        "3": {"class_type": "CMKSamplerPrepareSDXLPipe", "inputs": {"PROCESS": ["1", 0], "steps": 25}},
        "4": {"class_type": "CMKKSamplerPipe", "inputs": {"SAMPLER": ["3", 0]}},
    }
    if with_regional:
        prompt["2"] = {"class_type": "CMKRegionalLayoutSDXL",
                       "inputs": {"PROCESS": ["1", 0], "regions_json": "{\"regions\":[]}"}}
        prompt["3"]["inputs"]["PROCESS"] = ["2", 0]
    return prompt


class CacheRevision(unittest.TestCase):
    def fingerprint(self, prompt):
        persistent._FINGERPRINT_MEMO_PROMPT_ID = None
        return persistent.build_node_fingerprint(prompt, "4", ("CMKKSamplerPipe",), "test-schema")[0]

    def refiner_node_hash(self, prompt):
        return refiner_cache._canonical_node(prompt, "3", {}, set())["sha256"]

    def test_regional_graphs_get_new_keys_in_both_canonicalisers(self):
        prompt = chain_prompt(True)
        current = (self.fingerprint(prompt), self.refiner_node_hash(prompt))
        with patch.dict(persistent.NODE_BEHAVIOUR_REVISIONS, clear=True):
            legacy = (self.fingerprint(prompt), self.refiner_node_hash(prompt))
        self.assertNotEqual(current[0], legacy[0])
        self.assertNotEqual(current[1], legacy[1])

    def test_graphs_without_regional_nodes_keep_their_keys(self):
        prompt = chain_prompt(False)
        current = (self.fingerprint(prompt), self.refiner_node_hash(prompt))
        with patch.dict(persistent.NODE_BEHAVIOUR_REVISIONS, clear=True):
            legacy = (self.fingerprint(prompt), self.refiner_node_hash(prompt))
        self.assertEqual(current, legacy)

    def test_both_regional_classes_carry_the_profile_revision(self):
        regional = {name: revision for name, revision in persistent.NODE_BEHAVIOUR_REVISIONS.items()
                    if name.startswith("CMKRegional")}
        self.assertEqual(regional, {
            "CMKRegionalLayoutSDXL": weighting.PROFILE_VERSION,
            "CMKRegionalConditioningSDXL": weighting.PROFILE_VERSION,
        })
        self.assertIs(refiner_cache.NODE_BEHAVIOUR_REVISIONS, persistent.NODE_BEHAVIOUR_REVISIONS)

    def test_every_graph_canonicaliser_embeds_the_revision(self):
        """A new canonicaliser without the revision would let stale results survive."""
        found = {}
        for path in (PACKAGE_ROOT / "pipe").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "_canonical_node":
                    found[path.name] = "NODE_BEHAVIOUR_REVISIONS" in ast.unparse(node)
        self.assertEqual(found, {"cmk_persistent_cache.py": True, "cmk_refiner_boundary_cache.py": True})


if __name__ == "__main__":
    unittest.main()
