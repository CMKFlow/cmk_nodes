"""2.7.1 CMK PostProcess Model: one node, MODEL SDXL + MODEL ZIT specifications.

Pins the node contract: the settings of both single sources, two independent
lazy specifications identical to those of the single sources, no loading in
the source itself, and the unchanged registration of the single sources.
Consumer behaviour (only the requested family loads, ZIT -> SDXL release) is
proven with the PromptExecutor harness.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_postprocess_model_source_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

source = importlib.import_module(f"{TEST_PACKAGE}.pipe.loaders.postprocess_model_source")
sdxl_loader = importlib.import_module(f"{TEST_PACKAGE}.pipe.loaders.checkpoint_vae_loader")
zit_loader = importlib.import_module(f"{TEST_PACKAGE}.pipe.loaders.z_image_turbo_loader")
mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")

CMK_TEST_TIER = "T0"
NODE = source.CMKPostProcessModelSourcePipe
SDXL = dict(ckpt_name="juggernautXL_ragnarok.safetensors", vae_name="ClearVAE_V2.2.safetensors", checkpoint_vae=True)
ZIT = dict(diffusion_model="z_image_turbo_bf16.safetensors", text_encoder="qwen_3_4b.safetensors",
           vae="ae.safetensors", weight_dtype="fp8_e4m3fn", clip_device="cpu")


class Contract(unittest.TestCase):
    def test_inputs_are_those_of_both_single_sources(self):
        required = NODE.INPUT_TYPES()["required"]
        sdxl = sdxl_loader.CMKPostProcessModelPipe.INPUT_TYPES()["required"]
        zit = zit_loader.CMKZImageTurboPostProcessModelPipe.INPUT_TYPES()["required"]
        self.assertEqual(list(required), [*sdxl, *zit])
        self.assertEqual(list(required), [*source.SDXL_INPUTS, *source.ZIT_INPUTS])
        for name, spec in {**sdxl, **zit}.items():
            self.assertEqual(required[name], spec, name)
        self.assertIn(sdxl_loader.POSTPROCESS_MODEL_NONE, required["ckpt_name"][0])

    def test_outputs(self):
        self.assertEqual(NODE.RETURN_TYPES, ("CMK_MODEL_PIPE", "CMK_MODEL_PIPE"))
        self.assertEqual(NODE.RETURN_NAMES, ("MODEL SDXL", "MODEL ZIT"))

    def test_specifications_equal_the_single_sources(self):
        model_sdxl, model_zit = NODE().load(**SDXL, **ZIT)
        self.assertEqual(model_sdxl, sdxl_loader.CMKPostProcessModelPipe().load(**SDXL)[0])
        self.assertEqual(model_zit, zit_loader.CMKZImageTurboPostProcessModelPipe().load(**ZIT)[0])
        self.assertTrue(sdxl_loader.is_postprocess_model_spec(model_sdxl))
        self.assertTrue(zit_loader.is_zit_postprocess_model_spec(model_zit))
        self.assertFalse(model_sdxl["materialized"] or model_zit["materialized"])

    def test_none_checkpoint_keeps_zit_configured(self):
        model_sdxl, model_zit = NODE().load(**{**SDXL, "ckpt_name": "None"}, **ZIT)
        self.assertTrue(sdxl_loader.is_postprocess_model_disabled(model_sdxl))
        self.assertEqual(model_zit["diffusion_model_name"], ZIT["diffusion_model"])

    def test_source_loads_nothing(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("the source must not load models")

        saved = (sdxl_loader.load_checkpoint_vae_resources, zit_loader._call_node_kwargs)
        sdxl_loader.load_checkpoint_vae_resources = forbidden
        zit_loader._call_node_kwargs = forbidden
        try:
            model_sdxl, model_zit = NODE().load(**SDXL, **ZIT)
        finally:
            sdxl_loader.load_checkpoint_vae_resources, zit_loader._call_node_kwargs = saved
        self.assertIsNone(model_sdxl.get("model"))
        self.assertIsNone(model_zit.get("model"))
        self.assertEqual(zit_loader._ZIT_RESOURCE_CACHE, {})


class Registration(unittest.TestCase):
    def test_new_and_single_sources_registered(self):
        classes, names = mappings.NODE_CLASS_MAPPINGS, mappings.NODE_DISPLAY_NAME_MAPPINGS
        self.assertIs(classes["CMKPostProcessModelSourcePipe"], NODE)
        self.assertIs(classes["CMKPostProcessModelPipe"], sdxl_loader.CMKPostProcessModelPipe)
        self.assertIs(classes["CMKZImageTurboPostProcessModelPipe"], zit_loader.CMKZImageTurboPostProcessModelPipe)
        self.assertEqual(names["CMKPostProcessModelSourcePipe"], "CMK PostProcess Model")
        self.assertEqual(names["CMKPostProcessModelPipe"], "CMK PostProcess Model SDXL")
        self.assertEqual(NODE.CATEGORY, sdxl_loader.CMKPostProcessModelPipe.CATEGORY)

    def test_frontend_sections(self):
        js = (PACKAGE_ROOT / "web" / "js" / "cmk_postprocess_model_source_ui.js").read_text(encoding="utf-8")
        self.assertIn('{ title: "SDXL", first: "ckpt_name" }', js)
        self.assertIn('{ title: "Z-IMAGE TURBO", first: "diffusion_model" }', js)
        self.assertIn("data-widget-name", js)


if __name__ == "__main__":
    unittest.main()
