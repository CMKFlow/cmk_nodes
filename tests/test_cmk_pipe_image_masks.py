import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_module():
    root = Path(__file__).resolve().parents[1]
    for name in ("cmk_nodes", "cmk_nodes.pipe", "cmk_nodes.utils"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    log_module = types.ModuleType("cmk_nodes.pipe.cmk_log_pipe")
    log_module.cmk_add_block = lambda incoming, *_args, **_kwargs: dict(incoming or {})
    log_module.cmk_bool = lambda value: str(bool(value))
    log_module.cmk_clean_text = lambda value: str(value or "").strip()
    sys.modules[log_module.__name__] = log_module

    diagnostic_module = types.ModuleType("cmk_nodes.utils.cmk_diagnostic")
    diagnostic_module.make_diagnostic_payload = lambda **values: values
    sys.modules[diagnostic_module.__name__] = diagnostic_module

    visual_module = types.ModuleType("cmk_nodes.pipe.cmk_visual")
    visual_module.empty_visual = lambda: {
        "type": "CMK_VISUAL_PIPE", "version": 1, "providers": []
    }
    def register_provider(visual, **provider):
        result = dict(visual)
        result["providers"] = list(visual.get("providers", [])) + [provider]
        return result
    visual_module.register_provider = register_provider
    sys.modules[visual_module.__name__] = visual_module

    comfy_module = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")

    def common_upscale(samples, width, height, method, _crop):
        interpolation = "nearest" if method in {"nearest", "nearest-exact"} else "bilinear"
        options = {} if interpolation == "nearest" else {"align_corners": False}
        return F.interpolate(
            samples,
            size=(height, width),
            mode=interpolation,
            **options,
        )

    comfy_utils.common_upscale = common_upscale
    comfy_module.utils = comfy_utils
    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.utils"] = comfy_utils

    spec = importlib.util.spec_from_file_location(
        "cmk_nodes.pipe.cmk_pipe_image",
        root / "pipe" / "cmk_pipe_image.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CreateImageMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    @staticmethod
    def result(payload):
        return payload["result"] if isinstance(payload, dict) else payload

    def test_fit_keeps_image_and_mask_geometry_aligned(self):
        image = torch.zeros((1, 2, 4, 3))
        image[:, :, :2, 0] = 1
        image[:, :, 2:, 1] = 1
        mask = torch.zeros((1, 2, 4))
        mask[:, :, 2:] = 1

        fitted, fitted_mask, uncovered = self.module.prepare_image_and_mask(
            image, mask, 8, 8, "nearest", "Fit", "Top"
        )

        self.assertEqual(tuple(fitted.shape), (1, 8, 8, 3))
        self.assertTrue(torch.all(uncovered[:, :4, :] == 0))
        self.assertTrue(torch.all(uncovered[:, 4:, :] == 1))
        self.assertTrue(torch.all(fitted_mask[:, :4, :4] == 0))
        self.assertTrue(torch.all(fitted_mask[:, :4, 4:] == 1))

    def test_feathered_fill_has_soft_edge_but_generation_mask_stays_binary(self):
        mask = torch.zeros((1, 32, 32))
        mask[:, 16:, :] = 1
        generation_mask = self.module.expand_mask_tensor(mask, 4)
        fill_mask = self.module.feather_mask_tensor(generation_mask, 2)

        self.assertTrue(torch.all((generation_mask == 0) | (generation_mask == 1)))
        self.assertGreater(float(fill_mask[:, 10:16, :].max()), 0.0)
        self.assertLess(float(fill_mask[:, 10:16, :].min()), 1.0)
        self.assertTrue(torch.all(generation_mask[:, 12:, :] == 1))

    def test_optional_positive_prompt_is_appended_after_primary_prompt(self):
        result = self.module.CMKPipeCreateImage().create_image(
            **{
                "GLOBAL PROMPT POS": "primary",
                "opt_prompt_pos": "additional",
                "PROMPT NEG": "",
                "INPAINT_MODE": "Text2Image",
            }
        )
        pipe = self.result(result)[0]
        self.assertEqual(pipe["prompt_pos"], "primary\nadditional")
        self.assertEqual(pipe["prompt_pos_primary"], "primary")
        self.assertEqual(pipe["opt_prompt_pos"], "additional")

    def test_z_image_family_preserves_inpaint_image_and_mask_contract(self):
        image = torch.zeros((1, 16, 16, 3))
        mask = torch.zeros((1, 16, 16))
        mask[:, 4:12, 4:12] = 1
        result = self.module.CMKPipeCreateImage().create_image(
            **{
                "PROMPT POS": "photo",
                "PROMPT NEG": "",
                "INPAINT_MODE": "Inpaint",
                "model_family": "Z-Image Turbo",
                "resolution": "1024x1024",
                "IMAGE": image,
                "MASK": mask,
                "FILENAME": "zit-inpaint.png",
            }
        )
        result = self.result(result)
        self.assertFalse(result[0]["family_active"])
        pipe = result[1]
        self.assertTrue(pipe["family_active"])
        self.assertEqual(pipe["model_family"], "z_image_turbo")
        self.assertEqual(pipe["generation_mode"], "inpaint")
        self.assertTrue(pipe["boolean_inpaint_mode"])
        self.assertEqual(tuple(pipe["mask"].shape), (1, 1024, 1024))
        self.assertEqual(tuple(result[2].shape), (1, 1024, 1024, 3))

    def test_z_image_rejects_low_resolution_but_sdxl_keeps_it(self):
        normalize = self.module.normalize_resolution_for_family
        self.assertEqual(normalize("512x768", "z_image_turbo"), "1024x1024")
        self.assertEqual(
            normalize("512x768", "z_image_turbo", inpaint_mode=True),
            "512x768",
        )
        self.assertEqual(
            normalize("768x512", "z_image_turbo", inpaint_mode=True),
            "768x512",
        )
        self.assertEqual(normalize("1344x768", "z_image_turbo"), "1344x768")
        self.assertEqual(normalize("512x768", "sdxl"), "512x768")

    def test_z_image_inpaint_applies_the_selected_safe_preset(self):
        image = torch.zeros((1, 900, 1400, 3))
        mask = torch.ones((1, 900, 1400))
        result = self.result(self.module.CMKPipeCreateImage().create_image(**{
            "PROMPT POS": "photo",
            "INPAINT_MODE": "Inpaint",
            "model_family": "Z-Image Turbo",
            "resolution": "768x512",
            "IMAGE": image,
            "MASK": mask,
            "FILENAME": "zit-inpaint.png",
        }))
        pipe = result[1]
        self.assertEqual(pipe["resolution"], "768x512")
        self.assertEqual((pipe["width"], pipe["height"]), (768, 512))
        self.assertEqual(tuple(result[2].shape), (1, 512, 768, 3))
        self.assertEqual(tuple(pipe["mask"].shape), (1, 512, 768))

    def test_visible_family_tabs_drive_backend_when_technical_widget_is_hidden(self):
        result = self.module.CMKPipeCreateImage().create_image(
            **{
                "PROMPT POS": "photo",
                "MODEL FAMILY TABS": "Z-Image Turbo",
                "resolution": "1024x1024",
            }
        )
        result = self.result(result)
        self.assertFalse(result[0]["family_active"])
        pipe = result[1]
        self.assertTrue(pipe["family_active"])
        self.assertEqual(pipe["model_family"], "z_image_turbo")
        self.assertEqual(pipe["generation_mode"], "text2image")

    def test_public_optional_input_names_are_self_explanatory(self):
        optional = self.module.CMKPipeCreateImage.INPUT_TYPES()["optional"]
        self.assertEqual(self.module.CMKPipeCreateImage.INPUT_TYPES()["required"], {})
        self.assertIn("GLOBAL PROMPT POS", optional)
        self.assertIn("PROMPT NEG", optional)
        self.assertIn("INPAINT_MODE", optional)
        self.assertIn("upscale_method", optional)
        self.assertIn("device", optional)
        self.assertIn("FILENAME", optional)
        self.assertIn("LORA STACK", optional)
        self.assertIn("ACTIVE LORAS", optional)
        self.assertIn("ADDITIONAL PROMPT", optional)

    def test_model_family_outputs_are_mechanically_separated(self):
        node = self.module.CMKPipeCreateImage
        self.assertEqual(
            node.RETURN_TYPES[:2],
            ("CMK_PROCESS_SDXL", "CMK_PROCESS_Z_IMAGE"),
        )
        self.assertEqual(
            node.RETURN_NAMES[:2],
            ("PROCESS SDXL", "PROCESS ZIT"),
        )

        sdxl = self.result(node().create_image(**{"GLOBAL PROMPT POS": "photo"}))
        self.assertIsInstance(sdxl[0], dict)
        self.assertTrue(sdxl[0]["family_active"])
        self.assertIsInstance(sdxl[1], dict)
        self.assertFalse(sdxl[1]["family_active"])

    def test_remove_preserves_prompts_but_bypasses_loras(self):
        image = torch.zeros((1, 16, 16, 3))
        mask = torch.zeros((1, 16, 16))
        mask[:, 4:12, 4:12] = 1

        pipe, *_ = self.result(self.module.CMKPipeCreateImage().create_image(
            **{
                "GLOBAL PROMPT POS": "futuristic botanical observatory",
                "PROMPT NEG": "low quality",
                "INPAINT_MODE": "Inpaint",
                "process_mode": "Remove Object",
                "resolution": "512x512",
                "IMAGE": image,
                "MASK": mask,
                "FILENAME": "test.png",
                "ACTIVE LORAS": "must be ignored",
            }
        ))

        self.assertEqual(pipe["prompt_pos"], "futuristic botanical observatory")
        self.assertEqual(pipe["prompt_neg"], "low quality")
        self.assertEqual(pipe["active_loras"], "")
        self.assertEqual(pipe["fill_masked_area"], "noise")
        self.assertFalse(pipe["remove_isolated"])
        self.assertIsNone(pipe["remove_result_image"])

    def test_remove_sends_noise_filled_image_and_preserves_mask_exterior(self):
        image = torch.full((1, 16, 16, 3), 0.25)
        mask = torch.zeros((1, 16, 16))
        mask[:, 4:12, 4:12] = 1
        payload = self.module.CMKPipeCreateImage().create_image(
            **{
                "INPAINT_MODE": "Inpaint",
                "process_mode": "Remove Object",
                "resolution": "512x512",
                "IMAGE": image,
                "MASK": mask,
                "FILENAME": "remove-preview.png",
                "unique_id": "node-01",
            }
        )
        result = self.result(payload)

        process = result[0]
        image_output = result[2]
        visual_image = result[4]["providers"][0]["channels"]["result"]
        self.assertGreater(float(process["mask"].sum()), 0.0)
        self.assertFalse(torch.allclose(image_output, torch.full_like(image_output, 0.25)))
        exterior = process["mask_fill"] <= 0
        self.assertTrue(torch.allclose(image_output[exterior], torch.full_like(image_output[exterior], 0.25)))
        self.assertTrue(torch.allclose(visual_image, image_output))
        self.assertEqual(payload["ui"], {"images": []})

    def test_text2image_keeps_visual_empty(self):
        payload = self.module.CMKPipeCreateImage().create_image(
            **{"GLOBAL PROMPT POS": "photo", "INPAINT_MODE": "Text2Image"}
        )
        result = self.result(payload)
        self.assertEqual(result[4]["providers"], [])
        self.assertEqual(payload["ui"], {"images": []})

    def test_inpaint_publishes_early_visualizer_stage_without_own_preview(self):
        image = torch.zeros((1, 16, 16, 3))
        mask = torch.ones((1, 16, 16))
        original_preview = self.module.image_node_preview
        self.module.image_node_preview = lambda _image: {
            "images": [{"filename": "stage.png", "subfolder": "", "type": "temp"}]
        }
        try:
            payload = self.module.CMKPipeCreateImage().create_image(**{
                "INPAINT_MODE": "Inpaint",
                "resolution": "512x512",
                "IMAGE": image,
                "MASK": mask,
                "FILENAME": "inpaint.png",
                "unique_id": "node-01",
            })
        finally:
            self.module.image_node_preview = original_preview

        stage = json.loads(payload["ui"]["cmk_visual_stage"][0])
        self.assertEqual(stage["module_label"], "Inpaint Preparation")
        self.assertEqual(stage["sequence"], 1)
        self.assertEqual(stage["stage_key"], "sdxl.input.inpaint")
        self.assertEqual(stage["channels"], [{"name": "result", "image_index": 0}])
        self.assertEqual(payload["ui"]["images"], [])
        self.assertEqual(payload["ui"]["cmk_visual_stage_images"][0]["filename"], "stage.png")


if __name__ == "__main__":
    unittest.main()
