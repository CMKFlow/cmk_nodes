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


if __name__ == "__main__":
    unittest.main()
