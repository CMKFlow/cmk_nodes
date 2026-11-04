from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import types
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_regional_layout_test"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

log_pipe = types.ModuleType(f"{TEST_PACKAGE}.pipe.cmk_log_pipe")
log_pipe.cmk_clean_text = lambda value: str(value or "").strip()
log_pipe.cmk_add_block = lambda log, *_args: log
sys.modules[log_pipe.__name__] = log_pipe
visual = types.ModuleType(f"{TEST_PACKAGE}.pipe.cmk_visual")
visual.normalize_visual = lambda value: value or {}
sys.modules[visual.__name__] = visual
diagnostic = types.ModuleType(f"{TEST_PACKAGE}.utils.cmk_diagnostic")
diagnostic.make_diagnostic_payload = lambda **kwargs: kwargs
sys.modules[diagnostic.__name__] = diagnostic

module = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_regional_layout")
legacy_module = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_regional_conditioning")


class RegionalLayoutTests(unittest.TestCase):
    def test_more_than_three_regions_keep_order_and_values(self):
        payload = {"regions": [
            {"id": f"r{i}", "name": f"Area {i}", "prompt": f"prompt {i}", "strength": i / 4, "box": {"x": i / 20, "y": 0.1, "width": 0.2, "height": 0.3}}
            for i in range(1, 6)
        ]}
        regions = module.parse_layout(json.dumps(payload), 1344, 768)
        self.assertEqual([region["id"] for region in regions], ["r1", "r2", "r3", "r4", "r5"])
        self.assertEqual(regions[3]["prompt"], "prompt 4")
        self.assertEqual(regions[3]["strength"], 1.0)
        self.assertEqual(regions[0]["source"], "LAYOUT 1344x768")

    def test_geometry_is_normalized_and_empty_prompts_do_not_execute(self):
        payload = {"regions": [
            {"id": "empty", "prompt": "", "box": {"x": 0, "y": 0, "width": 1, "height": 1}},
            {"id": "active", "prompt": "subject", "box": {"x": 0.9, "y": -1, "width": 0.4, "height": 2}, "strength": 9, "end": -1},
        ]}
        regions = module.parse_layout(json.dumps(payload), 1024, 1024)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["x"], 0.6)
        self.assertEqual(regions[0]["y"], 0.0)
        self.assertEqual(regions[0]["height"], 1.0)
        self.assertEqual(regions[0]["strength"], 2.0)
        self.assertEqual(regions[0]["end"], 0.0)

    def test_process_contract_appends_without_mutating_input(self):
        original = {"width": 1344, "height": 768, "regional_conditioning": [{"prompt": "legacy"}]}
        payload = json.dumps({"regions": [{"id": "new", "name": "New", "prompt": "new prompt", "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}}]})
        process, image, log, visual_pipe, diagnostic_payload = module.CMKRegionalLayoutSDXL().configure(
            original, "image", "log", 512, 512, payload
        )
        self.assertIsNot(process, original)
        self.assertEqual(len(original["regional_conditioning"]), 1)
        self.assertEqual([item["prompt"] for item in process["regional_conditioning"]], ["legacy", "new prompt"])
        self.assertEqual(process["regional_conditioning_count"], 2)
        self.assertEqual(process["regional_conditioning"][1]["strength"], 1.0)
        self.assertEqual(process["regional_conditioning"][1]["source"], "LAYOUT 1344x768")
        self.assertEqual((process["width"], process["height"]), (1344, 768))
        self.assertEqual((image, log, visual_pipe), ("image", "log", {}))
        self.assertEqual(diagnostic_payload["metadata"]["active_regions"], 1)
        self.assertEqual(diagnostic_payload["title"], "07 Regional Layout")
        self.assertEqual(diagnostic_payload["node"], "CMK Flow · 07 Regional Layout")

    def test_empty_layout_returns_the_identical_process_unchanged(self):
        original = {"width": 1152, "height": 832, "prompt_pos": "global"}
        process, image, log, *_ = module.CMKRegionalLayoutSDXL().configure(
            original, "image", "log", 512, 512,
            json.dumps({"width": 512, "height": 512, "regions": []}),
        )
        self.assertIs(process, original)
        self.assertEqual(process, {"width": 1152, "height": 832, "prompt_pos": "global"})
        self.assertEqual((image, log), ("image", "log"))

    def test_process_dimensions_fall_back_to_target_dimensions(self):
        self.assertEqual(module.process_dimensions({"target_width": 896, "target_height": 1152}), (896, 1152))

    def test_strength_offset_is_additive_and_not_limited_to_two(self):
        for offset, expected in ((0.0, 1.5), (1.0, 2.5), (2.0, 3.5)):
            payload = {"strength_offset": offset, "regions": [{
                "prompt": "subject", "strength": 1.5,
                "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
            }]}
            regions = module.parse_layout(payload, 1024, 1024)
            self.assertEqual(payload["regions"][0]["strength"], 1.5)
            self.assertEqual(regions[0]["strength"], expected)

    def test_missing_strength_offset_defaults_to_zero(self):
        regions = module.parse_layout({"regions": [{"prompt": "subject", "strength": 0.5}]}, 1024, 1024)
        self.assertEqual(regions[0]["strength"], 0.5)

    def test_legacy_and_layout_have_equal_sampler_effective_fields(self):
        legacy, *_ = legacy_module.CMKRegionalConditioningSDXL().configure(
            {}, "image", "log", region_1_position="CUSTOM", region_1_x=0.1,
            region_1_y=0.2, region_1_width=0.3, region_1_height=0.4,
            region_1_area_prompt="same prompt", region_1_strength=1.25,
            region_1_end=0.75,
        )
        layout, *_ = module.CMKRegionalLayoutSDXL().configure(
            {"width": 1152, "height": 832}, "image", "log", 512, 512,
            json.dumps({"selected_id": "ignored", "strength_offset": 0.0, "regions": [{
                "id": "r1", "name": "Left", "prompt": "same prompt",
                "strength": 1.25, "end": 0.75,
                "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
            }]}),
        )
        effective = ("prompt", "x", "y", "width", "height", "strength", "start", "end")
        self.assertEqual(
            {key: legacy["regional_conditioning"][0][key] for key in effective},
            {key: layout["regional_conditioning"][0][key] for key in effective},
        )

    def test_legacy_source_is_untouched(self):
        legacy = (PACKAGE_ROOT / "pipe" / "cmk_regional_conditioning.py").read_text(encoding="utf-8")
        self.assertNotIn("CMKRegionalLayoutSDXL", legacy)
        self.assertNotIn("regions_json", legacy)

    def test_new_node_registration_and_legacy_execution(self):
        mappings = (PACKAGE_ROOT / "cmk_mappings.py").read_text(encoding="utf-8")
        self.assertIn('"CMKRegionalLayoutSDXL": CMKRegionalLayoutSDXL', mappings)
        self.assertIn('"CMKRegionalLayoutSDXL": "CMK Flow · 07 Regional Layout"', mappings)
        self.assertNotIn('"CMKRegionalLayoutSDXL": "CMK Flow · 02 Regional Layout SDXL"', mappings)
        layout_inputs = module.CMKRegionalLayoutSDXL.INPUT_TYPES()
        self.assertEqual(
            list(layout_inputs["required"]),
            ["PROCESS", "IMAGE", "LOG", "width", "height", "regions_json"],
        )
        self.assertEqual(list(layout_inputs["optional"]), ["VISUAL"])
        self.assertEqual(module.CMKRegionalLayoutSDXL.RETURN_TYPES[:3], (
            "CMK_PROCESS_SDXL", "IMAGE", "CMK_LOG_PIPE",
        ))
        legacy_inputs = legacy_module.CMKRegionalConditioningSDXL.INPUT_TYPES()
        self.assertEqual(list(legacy_inputs["required"])[:3], ["PROCESS", "IMAGE", "LOG"])
        self.assertEqual(list(legacy_inputs["optional"]), ["VISUAL"])
        process, *_ = legacy_module.CMKRegionalConditioningSDXL().configure(
            {}, "image", "log", region_1_position="CUSTOM", region_1_x=0.1,
            region_1_y=0.2, region_1_width=0.3, region_1_height=0.4,
            region_1_area_prompt="legacy prompt", region_1_strength=1.25,
            region_1_end=0.75,
        )
        self.assertEqual(process["regional_conditioning"][0]["prompt"], "legacy prompt")
        self.assertEqual(process["regional_conditioning"][0]["strength"], 1.25)


if __name__ == "__main__":
    unittest.main()
