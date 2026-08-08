import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ZITControlNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepare_source = (
            ROOT / "pipe" / "controlnet" / "cmk_zit_controlnet_prepare.py"
        ).read_text(encoding="utf-8")
        cls.sampler_source = (ROOT / "pipe" / "cmk_z_image_turbo.py").read_text(
            encoding="utf-8"
        )
        cls.prepare_tree = ast.parse(cls.prepare_source)

    def test_public_node_is_zit_typed_and_optional_by_default(self):
        self.assertIn('"PROCESS": ("CMK_PROCESS_Z_IMAGE",)', self.prepare_source)
        self.assertIn(
            'RETURN_NAMES = ("PROCESS", "IMAGE", "LOG", "diagnostic")',
            self.prepare_source,
        )
        self.assertIn('"ENABLE": ("BOOLEAN", {"default": False})', self.prepare_source)
        self.assertNotIn("CMK_PROCESS_SDXL", self.prepare_source)

    def test_authoritative_image_is_forwarded_for_zit_inpaint(self):
        self.assertIn(
            'bool(PROCESS.get("boolean_inpaint_mode", False))',
            self.prepare_source,
        )
        self.assertIn(
            '"result": (process, kwargs.get("IMAGE"), log, diagnostic)',
            self.prepare_source,
        )
        self.assertIn(
            '"Unchanged authoritative IMAGE for the following ZIT module."',
            self.prepare_source,
        )

    def test_ui_follows_sdxl_controlnet_source_pattern(self):
        for field in (
            '"IMAGE SOURCE"',
            '"REFERENCE IMAGE"',
            '"APPLY MASK"',
            '"STRENGTH"',
        ):
            self.assertIn(field, self.prepare_source)
        mapping = (ROOT / "cmk_mappings.py").read_text(encoding="utf-8")
        self.assertIn(
            '"CMKZITControlNetPreparePipe": '
            '"CMK Flow · 05 ControlNet ZIT (optional)"',
            mapping,
        )

    def test_prepare_uses_registered_native_reference_nodes(self):
        self.assertIn('("ImageScaleToMaxDimension",)', self.prepare_source)
        self.assertIn('("Canny",)', self.prepare_source)
        self.assertNotIn("cv2", self.prepare_source)

    def test_patch_load_and_apply_stay_inside_gated_zit_sampler(self):
        self.assertNotIn('("ModelPatchLoader",)', self.prepare_source)
        self.assertIn('_call_node_kwargs(("ModelPatchLoader",)', self.sampler_source)
        self.assertIn(
            '("QwenImageDiffsynthControlnet", "ZImageFunControlnet")',
            self.sampler_source,
        )
        control_guard = self.sampler_source.index("if controlnet_enabled:")
        patch_load = self.sampler_source.index('("ModelPatchLoader",)')
        model_sampling = self.sampler_source.index('("ModelSamplingAuraFlow",)')
        self.assertLess(control_guard, patch_load)
        self.assertLess(patch_load, model_sampling)

    def test_text2image_path_remains_the_default(self):
        self.assertIn(
            'controlnet_enabled = bool(PROCESS.get("boolean_controlnet_enable", False))',
            self.sampler_source,
        )
        self.assertIn('else ("ControlNet" if controlnet_enabled else "Text2Image")', self.sampler_source)

    def test_inpaint_path_uses_native_conditioning_and_zit_patch(self):
        self.assertIn('("InpaintModelConditioning",)', self.sampler_source)
        self.assertIn('("ZImageFunControlnet",)', self.sampler_source)
        self.assertIn("inpaint_image=IMAGE", self.sampler_source)
        self.assertIn("mask=mask", self.sampler_source)
        self.assertIn('return ["IMAGE"]', self.sampler_source)

    def test_flow_browser_metadata_is_registered(self):
        metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]["CMKZITControlNetPreparePipe"]
        self.assertEqual(metadata["recommendedAfter"], ["10 KSampler Z-Image Turbo"])

    def test_zit_reference_picker_has_its_own_widget_normalization(self):
        source = (
            ROOT / "web" / "js" / "cmk_controlnet_prepare_preview.js"
        ).read_text(encoding="utf-8")
        self.assertIn("function normalizeZITPickerValue", source)
        self.assertIn(
            'if (nodeData.name === "CMKZITControlNetPreparePipe")',
            source,
        )
        self.assertIn("normalized.splice(pickerIndex, 0, null)", source)
        self.assertIn('typeof normalized[8] === "number"', source)
        self.assertIn(
            'normalized[8] = "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
