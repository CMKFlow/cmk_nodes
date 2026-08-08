import importlib.util
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
                "PROMPT POS": "primary",
                "opt_prompt_pos": "additional",
                "PROMPT NEG": "",
                "INPAINT_MODE": "Text2Image",
            }
        )
        pipe = result[0]
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
                "resolution": "512x512",
                "IMAGE": image,
                "MASK": mask,
                "FILENAME": "zit-inpaint.png",
            }
        )
        self.assertFalse(result[0]["family_active"])
        pipe = result[1]
        self.assertTrue(pipe["family_active"])
        self.assertEqual(pipe["model_family"], "z_image_turbo")
        self.assertEqual(pipe["generation_mode"], "inpaint")
        self.assertTrue(pipe["boolean_inpaint_mode"])
        self.assertEqual(tuple(pipe["mask"].shape), (1, 512, 512))
        self.assertEqual(tuple(result[2].shape), (1, 512, 512, 3))

    def test_visible_family_tabs_drive_backend_when_technical_widget_is_hidden(self):
        result = self.module.CMKPipeCreateImage().create_image(
            **{
                "PROMPT POS": "photo",
                "MODEL FAMILY TABS": "Z-Image Turbo",
                "resolution": "1024x1024",
            }
        )
        self.assertFalse(result[0]["family_active"])
        pipe = result[1]
        self.assertTrue(pipe["family_active"])
        self.assertEqual(pipe["model_family"], "z_image_turbo")
        self.assertEqual(pipe["generation_mode"], "text2image")

    def test_public_optional_input_names_are_self_explanatory(self):
        optional = self.module.CMKPipeCreateImage.INPUT_TYPES()["optional"]
        self.assertEqual(self.module.CMKPipeCreateImage.INPUT_TYPES()["required"], {})
        self.assertIn("PROMPT POS", optional)
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

        sdxl = node().create_image(**{"PROMPT POS": "photo"})
        self.assertIsInstance(sdxl[0], dict)
        self.assertTrue(sdxl[0]["family_active"])
        self.assertIsInstance(sdxl[1], dict)
        self.assertFalse(sdxl[1]["family_active"])

    def test_remove_uses_prompt_free_diffusion_instead_of_lama_bypass(self):
        image = torch.zeros((1, 16, 16, 3))
        mask = torch.zeros((1, 16, 16))
        mask[:, 4:12, 4:12] = 1

        pipe, *_ = self.module.CMKPipeCreateImage().create_image(
            **{
                "PROMPT POS": "must be ignored",
                "PROMPT NEG": "must also be ignored",
                "INPAINT_MODE": "Inpaint",
                "process_mode": "Remove Object",
                "resolution": "512x512",
                "IMAGE": image,
                "MASK": mask,
                "FILENAME": "test.png",
                "ACTIVE LORAS": "must be ignored",
            }
        )

        self.assertEqual(pipe["prompt_pos"], "")
        self.assertEqual(pipe["prompt_neg"], "")
        self.assertEqual(pipe["active_loras"], "")
        self.assertEqual(pipe["fill_masked_area"], "noise")
        self.assertFalse(pipe["remove_isolated"])
        self.assertIsNone(pipe["remove_result_image"])


if __name__ == "__main__":
    unittest.main()
