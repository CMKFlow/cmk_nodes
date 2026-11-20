"""07 Regional Layout: CONDITIONING BLEND steps 0-6.

Model-free.  Step 0 must be the native ComfyUI percentage AREA used before
cmk-regional-edge-true-v1, step 6 the unchanged MASK profile, steps 1-5 the
discrete mixes.  The full spatial weight of every region (own weight,
neighbour crossfade, free-space ramp, overlaps, image border) and every UNet
crop is compared cell by cell against an independent per-cell reference: the
native AREA part comes from ComfyUI's own get_area_and_mult, the profile part
from the written specification below.
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import sys
import types
import unittest
import uuid

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_regional_conditioning_blend"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

layout = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_regional_layout")
weighting = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_regional_weighting")
prepare = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_sampler_prepare")
persistent = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_persistent_cache")
refiner_cache = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_refiner_boundary_cache")
instantid_conditioning = importlib.import_module(f"{TEST_PACKAGE}.pipe.instantid.cmk_instantid_conditioning")

import comfy.samplers as samplers  # noqa: E402
import nodes  # noqa: E402


# Specification of the steps: ramp, context margin, crossfade (cells), foreign share cap.
STEP_TABLE = {
    1: (1, 0, 0, 0.25 / 6), 2: (1, 1, 1, 0.25 * 2 / 6), 3: (2, 1, 1, 0.25 * 3 / 6),
    4: (3, 1, 1, 0.25 * 4 / 6), 5: (3, 2, 2, 0.25 * 5 / 6), 6: (4, 2, 2, 0.25),
}

W, H = 1344, 768
IDENTITY_3 = [  # the three neighbouring regions of "CMK 2.8 · Text2Image · Identity 3", unequal strengths
    {"x": 0.030386214343660303, "y": 0.31661799159967896, "width": 0.25349900290229654, "height": 0.6819645840660431, "strength": 2.0},
    {"x": 0.5961140489129182, "y": 0.3040214198356601, "width": 0.3560135182650426, "height": 0.6959785801643399, "strength": 1.4},
    {"x": 0.2905306227352141, "y": 0.21378355688591333, "width": 0.2899562833511512, "height": 0.7862164431140867, "strength": 1.4},
]
SCENARIOS = {
    "identity 3": (1344, 768, IDENTITY_3),
    "one person at the bottom border": (1024, 1024, [{"x": 0.067, "y": 0.5, "width": 0.25, "height": 0.5, "strength": 1.0}]),
    "image corners and edges": (1024, 1024, [
        {"x": 0.0, "y": 0.0, "width": 0.3, "height": 0.3, "strength": 1.2},        # top-left corner
        {"x": 0.0, "y": 0.75, "width": 1.0, "height": 0.25, "strength": 0.8},      # full width at the bottom
        {"x": 0.95, "y": 0.4, "width": 0.03, "height": 0.2, "strength": 1.5},      # narrow, 2-3 cells from the right
    ]),
    "touching and overlapping boxes": (1344, 768, [
        {"x": 0.10, "y": 0.2, "width": 0.25, "height": 0.6, "strength": 1.0},
        {"x": 0.35, "y": 0.2, "width": 0.25, "height": 0.6, "strength": 1.6},      # touches the first (gap 0)
        {"x": 0.55, "y": 0.3, "width": 0.25, "height": 0.5, "strength": 0.9},      # overlaps the second
    ]),
    "full frame": (1024, 1024, [{"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0, "strength": 1.0}]),
}


def native_area(region, lw, lh):
    """(box, full-frame weight) of ComfyUI's own AREA path, strength 1."""
    cond = nodes.ConditioningSetAreaPercentage().append(
        [[torch.zeros(1, 77, 2048), {}]], region["width"], region["height"], region["x"], region["y"], 1.0)[0][0][1]
    conds = [dict(cond, model_conds={}, uuid=uuid.uuid4())]
    samplers.resolve_areas_and_cond_masks_multidim(conds, (lh, lw), "cpu")
    result = samplers.get_area_and_mult(conds[0], torch.zeros(1, 4, lh, lw), torch.tensor([500.0]))
    h, w, y, x = result.area
    full = torch.zeros(lh, lw)
    full[y:y + h, x:x + w] = result.mult[0, 0]
    return (x, y, x + w, y + h), full


def distance(box, px, py):
    x0, y0, x1, y1 = box
    dx = max(x0 - px, 0, px - (x1 - 1))
    dy = max(y0 - py, 0, py - (y1 - 1))
    return math.sqrt(dx * dx + dy * dy)


def reference(regions, lw, lh, step):
    """Per-cell specification of the blend: weights [R, H, W] and crops."""
    ramp, margin, crossfade, cap = STEP_TABLE[step]
    natives = [native_area(region, lw, lh) for region in regions]
    boxes = [box for box, _ in natives]
    count = len(boxes)
    profile = torch.zeros(count, lh, lw, dtype=torch.float64)
    for py in range(lh):
        for px in range(lw):
            d = [distance(box, px, py) for box in boxes]
            inside = [value == 0 for value in d]
            if sum(inside) >= 2:
                values = [1.0 if flag else 0.0 for flag in inside]
            elif sum(inside) == 1:
                owner = inside.index(True)
                foreign = [0.0 if i == owner or d[i] > crossfade else 1.0 - d[i] / (crossfade + 1) for i in range(count)]
                total = sum(foreign)
                scale = min(1.0, (cap / (1.0 - cap)) / max(total, 1e-12))
                foreign = [value * scale for value in foreign]
                foreign[owner] = 1.0
                norm = sum(foreign)
                values = [value / norm for value in foreign]
            else:
                ramps = [max(0.0, 1.0 - value / (ramp + 1)) for value in d]
                norm = max(1.0, sum(ramps))
                values = [value / norm for value in ramps]
            profile[:, py, px] = torch.tensor(values, dtype=torch.float64)
    mix = step / 6
    weights = torch.stack([(1 - mix) * native.double() + mix * profile[i] for i, (_, native) in enumerate(natives)])
    crops = []
    for index in range(count):
        ys, xs = torch.nonzero(weights[index] > 0, as_tuple=True)
        crops.append((max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
                      min(lw, int(xs.max()) + 1 + margin), min(lh, int(ys.max()) + 1 + margin)))
    return boxes, weights, crops


class StepParameters(unittest.TestCase):
    def test_parameters_follow_the_step_table(self):
        self.assertEqual(weighting.BLEND_STEPS, 6)
        for step, expected in STEP_TABLE.items():
            ramp, margin, crossfade, cap = weighting.blend_parameters(step)
            self.assertEqual((ramp, margin, crossfade), expected[:3])
            self.assertAlmostEqual(cap, expected[3], places=12)

    def test_step_0_has_no_profile(self):
        with self.assertRaises(ValueError):
            weighting.build_region_weights(IDENTITY_3, W // 8, H // 8, 0)


class FullSpatialProfiles(unittest.TestCase):
    """Every cell of every region and every crop, steps 1-6, all scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.cases = []
        for name, (width, height, regions) in SCENARIOS.items():
            lw, lh = width // 8, height // 8
            for step in range(1, 7):
                cls.cases.append((name, step, regions, lw, lh,
                                  weighting.build_region_weights(regions, lw, lh, step), reference(regions, lw, lh, step)))

    def test_weights_match_the_specification_cell_by_cell(self):
        for name, step, _, _, _, profiles, (boxes, weights, _) in self.cases:
            for index, profile in enumerate(profiles):
                with self.subTest(scenario=name, step=step, region=index):
                    self.assertEqual(profile.box, boxes[index])
                    self.assertTrue(torch.allclose(profile.mask[0].double(), weights[index], atol=1e-6, rtol=0),
                                    float((profile.mask[0].double() - weights[index]).abs().max()))

    def test_crops_match_the_specification_exactly(self):
        for name, step, _, _, _, profiles, (_, _, crops) in self.cases:
            with self.subTest(scenario=name, step=step):
                self.assertEqual([p.crop for p in profiles], crops)

    def test_crops_grow_monotonically_and_stay_inside_the_mask_crop(self):
        by_case = {}
        for name, step, _, _, _, profiles, _ in self.cases:
            by_case.setdefault(name, {})[step] = [p.crop for p in profiles]
        for name, steps in by_case.items():
            for index in range(len(steps[6])):
                previous = steps[1][index]
                for step in range(2, 7):
                    crop = steps[step][index]
                    with self.subTest(scenario=name, step=step, region=index):
                        self.assertLessEqual(crop[0], previous[0]); self.assertLessEqual(crop[1], previous[1])
                        self.assertGreaterEqual(crop[2], previous[2]); self.assertGreaterEqual(crop[3], previous[3])
                    previous = crop

    def test_free_sides_grow_one_cell_per_step(self):
        # one person, box x9-41 y64-128 on 128x128: top, left and right are free, the bottom is the border
        steps = {step: p[0].crop for name, step, _, _, _, p, _ in self.cases if name == "one person at the bottom border"}
        for step, (x0, y0, x1, y1) in steps.items():
            self.assertEqual((x0, y0, x1, y1), (9 - step, 64 - step, 41 + step, 128))

    def test_neighbour_crossfade_and_single_owner_sum(self):
        for name, step, regions, lw, lh, profiles, (boxes, _, _) in self.cases:
            stack = torch.stack([p.mask[0] for p in profiles])
            owners = torch.zeros(lh, lw, dtype=torch.int64)
            for x0, y0, x1, y1 in boxes:
                owners[y0:y1, x0:x1] += 1
            single = owners == 1
            with self.subTest(scenario=name, step=step):
                self.assertLessEqual(float(stack.sum(dim=0)[single].max()), 1.0 + 1e-6)
                for index, (x0, y0, x1, y1) in enumerate(boxes):
                    alone = single[y0:y1, x0:x1]
                    for other in range(len(boxes)):
                        if other != index and alone.any():
                            self.assertLessEqual(float(stack[other, y0:y1, x0:x1][alone].max()), STEP_TABLE[step][3] + 1e-6)

    def test_step_6_is_the_unchanged_edge_true_profile(self):
        for name, (width, height, regions) in SCENARIOS.items():
            default = weighting.build_region_weights(regions, width // 8, height // 8)
            explicit = weighting.build_region_weights(regions, width // 8, height // 8, 6)
            for a, b in zip(default, explicit):
                self.assertTrue(torch.equal(a.mask, b.mask))
                self.assertEqual(a.crop, b.crop)


def legacy_layout(**extra):
    return json.dumps({
        "version": 1, "width": W, "height": H, "strength_offset": 0.5,
        "regions": [
            {"id": f"r{i}", "name": f"Region {i}", "prompt": f"woman {i}", "strength": s, "end": e,
             "box": {key: region[key] for key in ("x", "y", "width", "height")}}
            for i, (region, s, e) in enumerate(zip(IDENTITY_3, (1.5, 0.9, 0.9), (1.0, 1.0, 0.8)), 1)
        ],
        **extra,
    })


def regions(value):
    return layout.parse_layout(value, W, H)


class Prepare(prepare.CMKSamplerPrepareSDXLPipe):
    def __init__(self):
        self.encoded = []

    def _encode_sdxl_plus(self, clip, width, height, size_cond_factor, text):
        self.encoded.append(text)
        return [[torch.full((1, 77, 2048), float(len(self.encoded))), {"pooled_output": torch.zeros(1, 1280)}]]


def apply(region_list):
    node = Prepare()
    base = [[torch.zeros(1, 77, 2048), {"pooled_output": torch.zeros(1, 1280)}]]
    combined, translations, profiles = node._apply_regional_conditioning(base, None, W, H, 4, region_list)
    return node, combined, translations, profiles


def head_area_reference(region_list):
    """The regional path of the committed CMK 2.8 state (before cmk-regional-edge-true-v1)."""
    out = []
    for region in region_list:
        cond = [[torch.zeros(1, 77, 2048), {}]]
        cond = nodes.ConditioningSetAreaPercentage().append(
            cond, region["width"], region["height"], region["x"], region["y"], region["strength"])[0]
        cond = nodes.ConditioningSetTimestepRange().set_range(cond, region["start"], region["end"])[0]
        out.append(cond[0][1])
    return out


def without_pooled(metadata):
    return {key: value for key, value in metadata.items() if key != "pooled_output"}


class LayoutBlend(unittest.TestCase):
    def test_migration_of_older_layouts(self):
        self.assertEqual({r["conditioning_blend"] for r in regions(legacy_layout())}, {6})
        self.assertEqual({r["conditioning_blend"] for r in regions(legacy_layout(conditioning_mode="MASK"))}, {6})
        self.assertEqual({r["conditioning_blend"] for r in regions(legacy_layout(conditioning_mode="area"))}, {0})
        self.assertEqual({r["conditioning_blend"] for r in regions(legacy_layout(conditioning_blend=3, conditioning_mode="AREA"))}, {3})

    def test_invalid_steps_are_clear_errors(self):
        for value in (7, -1, 2.5, "3", True, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "integer 0-6"):
                regions(legacy_layout(conditioning_blend=value))
        with self.assertRaisesRegex(ValueError, "conditioning mode"):
            regions(legacy_layout(conditioning_mode="FEATHER"))

    def test_frontend_serialisation_round_trip(self):
        # Exactly what serializeLayout() writes (tests/test_regional_layout_frontend.mjs asserts the same string).
        saved = ('{"version":1,"width":1024,"height":1024,"strength_offset":0,"conditioning_blend":3,'
                 '"regions":[{"id":"r","name":"Region 1","prompt":"person","strength":1,"end":1,'
                 '"box":{"x":0.25,"y":0.25,"width":0.25,"height":0.5}}]}')
        self.assertEqual(layout.parse_layout(saved, 1024, 1024)[0]["conditioning_blend"], 3)

    def test_all_steps_carry_identical_user_values(self):
        strip = lambda items: [{k: v for k, v in r.items() if k != "conditioning_blend"} for r in items]
        base = strip(regions(legacy_layout(conditioning_blend=0)))
        for step in range(1, 7):
            self.assertEqual(strip(regions(legacy_layout(conditioning_blend=step))), base)
        self.assertEqual([r["strength"] for r in regions(legacy_layout())], [2.0, 1.4, 1.4])
        self.assertEqual([r["end"] for r in regions(legacy_layout())], [1.0, 1.0, 0.8])

    def test_node_log_names_the_step(self):
        expected = {0: "BLEND 0/6 · AREA", 3: "BLEND 3/6 · AREA 50 % / MASK 50 %", 6: "BLEND 6/6 · MASK"}
        for step, label in expected.items():
            _, _, log, _, diagnostic = layout.CMKRegionalLayoutSDXL().configure(
                {"width": W, "height": H}, None, {"blocks": []}, W, H, legacy_layout(conditioning_blend=step))
            self.assertIn(f"CONDITIONING    : {label}", log["blocks"][-1]["lines"])
            self.assertEqual(diagnostic["metadata"]["conditioning_blend"], step)
        self.assertEqual(weighting.blend_label(2), "BLEND 2/6 · AREA 67 % / MASK 33 %")


class SamplerConditioning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regions = {step: regions(legacy_layout(conditioning_blend=step)) for step in range(7)}
        cls.runs = {step: apply(cls.regions[step]) for step in range(7)}

    def test_step_0_is_exactly_the_previous_native_path(self):
        _, combined, _, profiles = self.runs[0]
        self.assertEqual(profiles, [])
        for (_, metadata), expected in zip(combined[1:], head_area_reference(self.regions[0])):
            self.assertEqual(without_pooled(metadata), expected)

    def test_area_layout_of_the_earlier_selector_is_step_0(self):
        _, combined, _, _ = apply(regions(legacy_layout(conditioning_mode="AREA")))
        for (_, a), (_, b) in zip(combined[1:], self.runs[0][1][1:]):
            self.assertEqual(without_pooled(a), without_pooled(b))

    def test_step_6_is_exactly_the_previous_mask_path(self):
        _, combined, _, profiles = self.runs[6]
        expected = weighting.build_region_weights(self.regions[6], W // 8, H // 8)
        self.assertEqual([index for index, _, _ in profiles], [1, 2, 3])
        for (_, metadata), reference_profile, region in zip(combined[1:], expected, self.regions[6]):
            x, y, width, height = reference_profile.crop_percent
            self.assertEqual(set(without_pooled(metadata)),
                             {"area", "strength", "set_area_to_bounds", "mask", "mask_strength", "start_percent", "end_percent"})
            self.assertEqual(metadata["area"], ("percentage", height, width, y, x))
            self.assertTrue(torch.equal(metadata["mask"], reference_profile.mask))
            self.assertEqual((metadata["strength"], metadata["mask_strength"]), (region["strength"], 1.0))
        for layout_json in (legacy_layout(), legacy_layout(conditioning_mode="MASK")):
            _, legacy, _, _ = apply(regions(layout_json))
            for (_, a), (_, b) in zip(legacy[1:], combined[1:]):
                self.assertEqual(a["area"], b["area"])
                self.assertTrue(torch.equal(a["mask"], b["mask"]))

    def test_intermediate_steps_carry_their_profiles(self):
        for step in range(1, 6):
            _, combined, _, profiles = self.runs[step]
            expected = weighting.build_region_weights(self.regions[step], W // 8, H // 8, step)
            self.assertEqual([s for _, _, s in profiles], [step] * 3)
            for (_, metadata), reference_profile in zip(combined[1:], expected):
                x, y, width, height = reference_profile.crop_percent
                self.assertEqual(metadata["area"], ("percentage", height, width, y, x))
                self.assertTrue(torch.equal(metadata["mask"], reference_profile.mask))

    def test_same_prompts_strengths_and_timing_in_every_step(self):
        base_node, base, base_tr, _ = self.runs[0]
        for step in range(1, 7):
            node, combined, translations, _ = self.runs[step]
            self.assertEqual(node.encoded, base_node.encoded)
            self.assertEqual([t.text for _, t in translations], [t.text for _, t in base_tr])
            for (a_tensor, a), (b_tensor, b) in zip(combined[1:], base[1:]):
                self.assertTrue(torch.equal(a_tensor, b_tensor))
                self.assertEqual((a["strength"], a["start_percent"], a["end_percent"]),
                                 (b["strength"], b["start_percent"], b["end_percent"]))

    def test_one_unet_pass_per_region_with_the_stepped_crop_and_weight(self):
        dims = (H // 8, W // 8)
        for step, (_, combined, _, profiles) in self.runs.items():
            conds = [dict(metadata, model_conds={}, uuid=uuid.uuid4()) for _, metadata in combined]
            samplers.resolve_areas_and_cond_masks_multidim(conds, dims, "cpu")
            shapes = []
            model = types.SimpleNamespace(
                current_patcher=types.SimpleNamespace(
                    prepare_state=lambda *a, **k: None, get_free_memory=lambda *a, **k: 1 << 40,
                    apply_hooks=lambda *a, **k: {}, prepare_hook_patches_current_keyframe=lambda *a, **k: None),
                memory_required=lambda shape, cond_shapes=None: 0,
                apply_model=lambda x, t, **c: (shapes.extend([tuple(x.shape[2:])] * x.shape[0]), torch.ones_like(x))[1],
            )
            samplers._calc_cond_batch(model, [conds], torch.zeros(1, 4, *dims), torch.tensor([500.0]), {})
            if step == 0:
                expected = [native_area(r, *dims[::-1])[0] for r in self.regions[0]]
            else:
                expected = [profile.crop for _, profile, _ in profiles]
            with self.subTest(step=step):
                self.assertEqual(shapes.count(dims), 1)
                self.assertEqual(sorted(s for s in shapes if s != dims),
                                 sorted((y1 - y0, x1 - x0) for x0, y0, x1, y1 in expected))
                for cond, region, index in zip(conds[1:], self.regions[step], range(3)):
                    mult = samplers.get_area_and_mult(cond, torch.zeros(1, 4, *dims), torch.tensor([500.0])).mult[0, 0]
                    x0, y0, x1, y1 = expected[index]
                    if step == 0:
                        weight = native_area(region, *dims[::-1])[1][y0:y1, x0:x1]
                    else:
                        weight = profiles[index][1].mask[0, y0:y1, x0:x1]
                    self.assertTrue(torch.allclose(mult, weight * region["strength"], atol=1e-6))

    def test_mixed_chain_groups_profiles_by_step(self):
        chained = self.regions[3][:2] + [dict(self.regions[6][2])]
        chained[2].pop("conditioning_blend")      # CMKRegionalConditioningSDXL regions carry no step
        _, combined, _, profiles = apply(chained)
        self.assertEqual([(index, step) for index, _, step in profiles], [(1, 3), (2, 3), (3, 6)])
        alone = weighting.build_region_weights(chained[2:], W // 8, H // 8)[0]
        pair = weighting.build_region_weights(chained[:2], W // 8, H // 8, 3)
        self.assertTrue(torch.equal(combined[3][1]["mask"], alone.mask))
        self.assertTrue(torch.equal(combined[1][1]["mask"], pair[0].mask))

    def test_instantid_hybrid_strips_spatial_data_in_every_step(self):
        for step, (_, combined, _, _) in self.runs.items():
            stripped = instantid_conditioning._hybrid_positive_conditioning(
                combined, instantid_mode="HYBRID", zero_pass_mode=False)
            for (tensor, metadata), (source, _) in zip(stripped, combined):
                with self.subTest(step=step):
                    self.assertIs(tensor, source)
                    self.assertTrue(set(instantid_conditioning._REGIONAL_PROMPT_KEYS).isdisjoint(metadata))


def chain_prompt(regions_json):
    return {
        "1": {"class_type": "CMKPipeCreateImage", "inputs": {"prompt": "station"}},
        "2": {"class_type": "CMKRegionalLayoutSDXL",
              "inputs": {"PROCESS": ["1", 0], "width": W, "height": H, "regions_json": regions_json}},
        "3": {"class_type": "CMKSamplerPrepareSDXLPipe", "inputs": {"PROCESS": ["2", 0], "steps": 25}},
        "4": {"class_type": "CMKKSamplerPipe", "inputs": {"SAMPLER": ["3", 0]}},
    }


class CacheKeys(unittest.TestCase):
    def keys(self, regions_json):
        prompt = chain_prompt(regions_json)
        persistent._FINGERPRINT_MEMO_PROMPT_ID = None
        return (persistent.build_node_fingerprint(prompt, "4", ("CMKKSamplerPipe",), "test-schema")[0],
                refiner_cache._canonical_node(prompt, "3", {}, set())["sha256"])

    def test_every_step_has_its_own_keys_in_both_canonicalisers(self):
        keys = [self.keys(legacy_layout(conditioning_blend=step)) for step in range(7)]
        self.assertEqual(len({k[0] for k in keys}), 7)
        self.assertEqual(len({k[1] for k in keys}), 7)
        self.assertEqual(keys[3], self.keys(legacy_layout(conditioning_blend=3)))


if __name__ == "__main__":
    unittest.main()
