import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CombinedControlNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            ROOT / "pipe" / "controlnet" / "cmk_combined_controlnet_prepare.py"
        )
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_is_a_real_registered_custom_node(self):
        mappings = (ROOT / "cmk_mappings.py").read_text(encoding="utf-8")
        self.assertIn("class CMKCombinedControlNetPreparePipe", self.source)
        self.assertIn(
            '"CMKCombinedControlNetPreparePipe": CMKCombinedControlNetPreparePipe',
            mappings,
        )
        self.assertIn(
            '"CMKCombinedControlNetPreparePipe": "CMK Flow · 05 Combined ControlNet (optional)"',
            mappings,
        )

    def test_has_one_shared_image_input_and_output(self):
        self.assertEqual(self.source.count('"IMAGE": ("IMAGE",)'), 1)
        self.assertIn(
            'RETURN_NAMES = ("PROCESS SDXL", "PROCESS ZIT", "IMAGE", "LOG", "diagnostic")',
            self.source,
        )

    def test_family_specific_settings_are_advanced_and_labelled(self):
        for label in (
            "sdxl · controlnet_model",
            "sdxl · preprocessor",
            "sdxl · start_percent",
            "sdxl · end_percent",
            "sdxl · invert_hint",
            "zit · low_threshold",
            "zit · high_threshold",
            "zit · model_patch",
        ):
            self.assertIn(label, self.source)
        self.assertGreaterEqual(self.source.count('"advanced": True'), 8)

    def test_shared_defaults_and_picker_migration_are_explicit(self):
        self.assertIn(
            '{"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}',
            self.source,
        )
        self.assertIn(
            '{"default": 768, "min": 64, "max": 8192, "step": 8}',
            self.source,
        )
        frontend = (
            ROOT / "web" / "js" / "cmk_controlnet_prepare_preview.js"
        ).read_text(encoding="utf-8")
        self.assertIn("normalizeCombinedPickerValue", frontend)
        self.assertIn("CMK_COMBINED_CONTROLNET_TYPES.has(nodeData.name)", frontend)

    def test_disabled_node_previews_the_selected_source_not_the_placeholder(self):
        self.assertIn("def _combined_preview_ui", self.source)
        self.assertIn(
            '_tensor_image_to_temp_ui(preview_image, "cmk_controlnet_reference")',
            self.source,
        )

    def test_packaged_reference_stays_inside_the_extension(self):
        backend = (ROOT / "nodes" / "controlnet" / "controlnet.py").read_text(
            encoding="utf-8"
        )
        frontend = (
            ROOT / "web" / "js" / "cmk_controlnet_prepare_preview.js"
        ).read_text(encoding="utf-8")
        self.assertIn("CMK Package · controlnet_reference.png", backend)
        self.assertIn("assets\" / \"references", backend)
        self.assertIn("/cmk/reference-assets/controlnet_reference.png", frontend)
        self.assertTrue(
            (ROOT / "assets" / "references" / "controlnet_reference.png").is_file()
        )

    def test_metadata_is_valid_and_lists_both_families(self):
        metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]["CMKCombinedControlNetPreparePipe"]
        self.assertEqual(metadata["compatibility"], ["SDXL", "Z-Image Turbo"])


if __name__ == "__main__":
    unittest.main()
