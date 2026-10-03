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
            '"CMKCombinedControlNetPreparePipe": "CMK Flow · 05 ControlNet Combined"',
            mappings,
        )

    def test_has_one_shared_image_input_and_output(self):
        self.assertEqual(self.source.count('"IMAGE": ("IMAGE",)'), 1)
        self.assertIn(
            '"PROCESS SDXL", "PROCESS ZIT", "IMAGE", "LOG", "VISUAL", "diagnostic",',
            self.source,
        )

    def test_hybrid_prepares_only_the_sdxl_family_branch(self):
        self.assertNotIn("requires exactly one active model family", self.source)
        self.assertIn("if not active_sdxl and not active_zit:", self.source)
        self.assertIn('process_sdxl.get("hybrid_mode", False)', self.source)
        self.assertIn("prepare_zit = active_zit and not hybrid_mode", self.source)
        self.assertIn("if active_sdxl:", self.source)
        self.assertIn("if prepare_zit:", self.source)
        self.assertIn("prepared_sdxl, prepared_zit, image, log", self.source)

    def test_hybrid_gate_keeps_zit_on_the_bypass_contract(self):
        gates = (ROOT / "pipe" / "cmk_family_result.py").read_text(encoding="utf-8")
        marker = "class CMKCombinedControlNetBypassGate"
        section = gates[gates.index(marker):gates.index("class CMKSamplerBypassGate")]
        self.assertIn('inputs.get("PROCESS SDXL ACTIVE")', section)
        self.assertIn('inputs.get("PROCESS ZIT BYPASS")', section)
        self.assertIn('result_label = "HYBRID SDXL ACTIVE / ZIT BYPASS"', section)

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
        self.assertIn("_tensor_image_to_temp_ui(", self.source)
        self.assertIn('preview_image, "cmk_controlnet_reference"', self.source)

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
        self.assertEqual(metadata["compatibility"], ["SDXL", "Z-Image Turbo", "Hybrid"])


if __name__ == "__main__":
    unittest.main()
