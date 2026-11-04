from __future__ import annotations

import ast
import importlib.util
import logging
from pathlib import Path
import unittest

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE_ROOT / "pipe" / "instantid" / "cmk_instantid_conditioning.py"
COMFY_ROOT = PACKAGE_ROOT.parents[1]

spec = importlib.util.spec_from_file_location("_cmk_instantid_conditioning_test", MODULE_PATH)
conditioning_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conditioning_module)


class InstantIDSemanticConditioningTests(unittest.TestCase):
    def test_hybrid_globalizes_and_detaches_external_controlnet_without_mutating_source(self):
        global_tensor = object()
        region_a_tensor = object()
        region_b_tensor = object()
        pooled_global = object()
        pooled_a = object()
        pooled_b = object()
        control = object()
        source = [
            [global_tensor, {"pooled_output": pooled_global, "custom": "global"}],
            [
                region_a_tensor,
                {
                    "area": ("percentage", 0.5, 0.5, 0.0, 0.0),
                    "strength": 1.5,
                    "set_area_to_bounds": False,
                    "start_percent": 0.0,
                    "end_percent": 0.8,
                    "pooled_output": pooled_a,
                    "control": control,
                    "control_apply_to_uncond": False,
                    "custom": "region-a",
                },
            ],
            [
                region_b_tensor,
                {
                    "area": ("percentage", 0.5, 0.5, 0.0, 0.5),
                    "strength": 0.5,
                    "set_area_to_bounds": False,
                    "start_percent": 0.1,
                    "end_percent": 1.0,
                    "pooled_output": pooled_b,
                    "custom": "region-b",
                },
            ],
        ]

        result = conditioning_module._hybrid_positive_conditioning(
            source,
            instantid_mode="HYBRID",
            zero_pass_mode=False,
        )

        self.assertIsNot(result, source)
        self.assertEqual(len(result), 3)
        self.assertEqual([entry[0] for entry in result], [global_tensor, region_a_tensor, region_b_tensor])
        for entry in result:
            self.assertTrue(set(conditioning_module._REGIONAL_PROMPT_KEYS).isdisjoint(entry[1]))
            self.assertTrue(set(conditioning_module._EXTERNAL_CONTROLNET_KEYS).isdisjoint(entry[1]))
        self.assertIs(result[0][1]["pooled_output"], pooled_global)
        self.assertIs(result[1][1]["pooled_output"], pooled_a)
        self.assertIs(result[2][1]["pooled_output"], pooled_b)
        self.assertEqual([entry[1]["custom"] for entry in result], ["global", "region-a", "region-b"])
        self.assertIn("area", source[1][1])
        self.assertEqual(source[1][1]["strength"], 1.5)
        self.assertEqual(source[1][1]["end_percent"], 0.8)
        self.assertIs(source[1][1]["control"], control)
        self.assertFalse(source[1][1]["control_apply_to_uncond"])

    def test_hybrid_inpaint_uses_the_same_globalization(self):
        source = [[object(), {"area": (1, 1, 0, 0), "strength": 2.0}]]
        result = conditioning_module._hybrid_positive_conditioning(
            source,
            instantid_mode="HYBRID INPAINT",
            zero_pass_mode=False,
        )
        self.assertEqual(result[0][1], {})

    def test_non_hybrid_modes_are_unchanged(self):
        source = [[object(), {"area": (1, 1, 0, 0), "strength": 2.0}]]
        cases = (
            {"instantid_mode": "ZERO-PASS", "zero_pass_mode": True},
            {"instantid_mode": "OTHER", "zero_pass_mode": False},
        )
        for values in cases:
            with self.subTest(**values):
                result = conditioning_module._hybrid_positive_conditioning(source, **values)
                self.assertIs(result, source)

    def test_existing_reference_blending_keeps_all_globalized_entries(self):
        nodes_source = (COMFY_ROOT / "nodes.py").read_text(encoding="utf-8")
        nodes_tree = ast.parse(nodes_source)
        average_class = next(
            node
            for node in nodes_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ConditioningAverage"
        )
        namespace = {"logging": logging, "torch": torch}
        exec(compile(ast.Module(body=[average_class], type_ignores=[]), "nodes.py", "exec"), namespace)
        ConditioningAverage = namespace["ConditioningAverage"]

        prepared = [
            [torch.full((1, 2, 3), 1.0), {"pooled_output": torch.full((1, 3), 1.0)}],
            [
                torch.full((1, 2, 3), 2.0),
                {
                    "area": ("percentage", 0.5, 0.5, 0.0, 0.0),
                    "strength": 1.5,
                    "set_area_to_bounds": False,
                    "start_percent": 0.0,
                    "end_percent": 0.8,
                    "pooled_output": torch.full((1, 3), 2.0),
                },
            ],
        ]
        reference = [[
            torch.full((1, 2, 3), 10.0),
            {"pooled_output": torch.full((1, 3), 10.0)},
        ]]
        globalized = conditioning_module._hybrid_positive_conditioning(
            prepared,
            instantid_mode="HYBRID",
            zero_pass_mode=False,
        )

        result = ConditioningAverage().addWeighted(globalized, reference, 0.25)[0]

        self.assertEqual(len(result), 2)
        torch.testing.assert_close(result[0][0], torch.full((1, 2, 3), 7.75))
        torch.testing.assert_close(result[1][0], torch.full((1, 2, 3), 8.0))
        self.assertTrue(set(conditioning_module._REGIONAL_PROMPT_KEYS).isdisjoint(result[1][1]))

    def test_15s_and_15a_detach_layout_metadata_before_reference_blending(self):
        for relative in (
            "pipe/instantid/cmk_instantid_sampler.py",
            "pipe/instantid/cmk_instantid_regional_advanced.py",
        ):
            source = (PACKAGE_ROOT / relative).read_text(encoding="utf-8")
            globalize_positive = source.index("positive = _hybrid_positive_conditioning(")
            globalize_negative = source.index("negative = _hybrid_positive_conditioning(")
            blending = source.index('if str(conditioning_mode) == "InstantID reference":')
            self.assertLess(globalize_positive, blending)
            self.assertLess(globalize_negative, blending)
            self.assertEqual(source.count("positive = _hybrid_positive_conditioning("), 1)
            self.assertEqual(source.count("negative = _hybrid_positive_conditioning("), 1)
            self.assertIn("SDXL layout only · stopped at handoff", source)

    def test_internal_instantid_controlnet_is_applied_after_external_detachment(self):
        sampler_source = (
            PACKAGE_ROOT / "pipe/instantid/cmk_instantid_sampler.py"
        ).read_text(encoding="utf-8")
        regional_source = (
            PACKAGE_ROOT / "pipe/instantid/cmk_instantid_regional_advanced.py"
        ).read_text(encoding="utf-8")
        self.assertLess(
            sampler_source.index("negative = _hybrid_positive_conditioning("),
            sampler_source.index('("CMKInternalApplyInstantIDAdvanced",)'),
        )
        self.assertLess(
            regional_source.index("negative = _hybrid_positive_conditioning("),
            regional_source.index("positive_out, negative_out = _apply_regional_controlnet("),
        )

    def test_15a_identity_masks_and_keypoint_branches_remain_separate(self):
        source = (
            PACKAGE_ROOT / "pipe/instantid/cmk_instantid_regional_advanced.py"
        ).read_text(encoding="utf-8")
        self.assertIn("mask=mask_a", source)
        self.assertIn("mask=mask_b", source)
        self.assertIn("branches = [(hint_a, mask_a, embeds_a, pose_strength_a)]", source)
        self.assertIn("branches.append((hint_b, mask_b, embeds_b, pose_strength_b))", source)
        self.assertIn("_apply_regional_controlnet(\n                control_net,\n                branches, positive, negative", source)


if __name__ == "__main__":
    unittest.main()
