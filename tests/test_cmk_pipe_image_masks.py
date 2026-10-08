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

    def test_outpaint_overlap_and_fill_feather_do_not_modify_source_mask(self):
        source = torch.zeros((1, 24, 24))
        source[:, 8:16, 8:16] = 1
        uncovered = torch.zeros_like(source)
        uncovered[:, :, :3] = 1

        generation, fill = self.module.combine_inpaint_and_outpaint_masks(
            source,
            uncovered,
            outpaint_on=True,
            outpaint_overlap=2,
            feather_outpaint_fill=True,
        )

        self.assertTrue(torch.equal(generation[:, 8:16, 8:16], source[:, 8:16, 8:16]))
        self.assertTrue(torch.equal(fill[:, 8:16, 8:16], source[:, 8:16, 8:16]))
        self.assertEqual(float(generation[:, 7, 8:16].max()), 0.0)
        self.assertEqual(float(fill[:, 7, 8:16].max()), 0.0)
        self.assertTrue(torch.all(fill[generation == 0] == 0))

    def test_outpaint_overlap_still_grows_uncovered_canvas(self):
        source = torch.zeros((1, 16, 16))
        source[:, 6:10, 6:10] = 1
        uncovered = torch.zeros_like(source)
        uncovered[:, :, :2] = 1

        generation, _fill = self.module.combine_inpaint_and_outpaint_masks(
            source,
            uncovered,
            outpaint_on=True,
            outpaint_overlap=3,
            feather_outpaint_fill=True,
        )

        self.assertTrue(torch.all(generation[:, :, :5] == 1))
        self.assertEqual(float(generation[:, :, 5].max()), 0.0)

    def test_selected_lora_trigger_words_are_appended_after_primary_prompt(self):
        result = self.module.CMKPipeCreateImage().create_image(
            **{
                "GLOBAL PROMPT POS": "primary",
                "LoRA SDXL": {
                    "family": "sdxl",
                    "lora_stack": [],
                    "active_loras": "",
                    "trigger_words": "additional",
                },
                "PROMPT NEG": "",
                "INPAINT_MODE": "Text2Image",
            }
        )
        pipe = self.result(result)[0]
        self.assertEqual(pipe["prompt_pos"], "primary\nadditional")
        self.assertEqual(pipe["prompt_pos_primary"], "primary")
        self.assertEqual(pipe["opt_prompt_pos"], "additional")

    def test_z_image_family_preserves_inpaint_image_and_mask_contract(self):
        image = torch.full((1, 16, 16, 3), 0.25)
        image[:, 4:12, 4:12, :] = 0.9
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
        source = pipe["inpaint_source_image"]
        prepared = result[2]
        self.assertEqual(tuple(source.shape), tuple(prepared.shape))
        exterior = pipe["mask"] <= 0
        interior = pipe["mask"] >= 1
        self.assertTrue(torch.allclose(source[exterior], prepared[exterior]))
        self.assertFalse(torch.allclose(source[interior], prepared[interior]))

    def test_z_image_inpaint_ignores_hidden_outpaint_and_guided_mode_values(self):
        image = torch.zeros((1, 16, 16, 3))
        mask = torch.zeros((1, 16, 16))
        mask[:, 4:12, 4:12] = 1

        def create(outpaint_on):
            return self.result(self.module.CMKPipeCreateImage().create_image(**{
                "PROMPT POS": "repair",
                "INPAINT_MODE": "Inpaint",
                "model_family": "Z-Image Turbo",
                "process_mode": "Extend Image",
                "outpaint_on": outpaint_on,
                "outpaint_overlap": 32,
                "resolution": "512x512",
                "IMAGE": image,
                "MASK": mask,
                "FILENAME": "zit-inpaint.png",
            }))

        stale = create(True)
        clean = create(False)
        for process in stale[:2]:
            self.assertFalse(process["outpaint_on"])
            self.assertEqual(process["inpaint_process_mode"], "custom")
        self.assertFalse(stale[5]["metadata"]["outpaint_on"])
        self.assertEqual(stale[5]["metadata"]["inpaint_process_mode"], "custom")
        self.assertTrue(torch.equal(stale[1]["mask"], clean[1]["mask"]))
        self.assertTrue(torch.equal(stale[1]["mask_fill"], clean[1]["mask_fill"]))

    def test_sdxl_and_hybrid_keep_explicit_outpaint_contract(self):
        image = torch.zeros((1, 16, 16, 3))
        mask = torch.zeros((1, 16, 16))
        mask[:, 4:12, 4:12] = 1
        for family in ("SDXL", "Hybrid"):
            result = self.result(self.module.CMKPipeCreateImage().create_image(**{
                "PROMPT POS": "extend",
                "INPAINT_MODE": "Inpaint",
                "model_family": family,
                "process_mode": "Extend Image",
                "outpaint_on": True,
                "resolution": "512x512",
                "IMAGE": image,
                "MASK": mask,
                "FILENAME": "inpaint.png",
            }))
            self.assertTrue(result[0]["outpaint_on"], family)
            self.assertEqual(result[0]["inpaint_process_mode"], "extend", family)

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
        self.assertEqual(optional["LoRA SDXL"][0], "CMK_LORA_SDXL_PIPE")
        self.assertEqual(optional["LoRA ZIT"][0], "CMK_LORA_ZIT_PIPE")
        self.assertNotIn("LORA STACK", optional)
        self.assertNotIn("ACTIVE LORAS", optional)
        self.assertNotIn("ADDITIONAL PROMPT", optional)

    def test_zit_live_preview_defaults_on_and_can_be_disabled(self):
        optional = self.module.CMKPipeCreateImage.INPUT_TYPES()["optional"]
        self.assertTrue(optional["live_preview_zit"][1]["default"])

        default_result = self.result(
            self.module.CMKPipeCreateImage().create_image(
                **{"MODEL FAMILY TABS": "Z-Image Turbo"}
            )
        )
        disabled_result = self.result(
            self.module.CMKPipeCreateImage().create_image(
                **{
                    "MODEL FAMILY TABS": "Z-Image Turbo",
                    "live_preview_zit": False,
                }
            )
        )
        self.assertTrue(default_result[1]["live_preview_zit"])
        self.assertFalse(disabled_result[1]["live_preview_zit"])

    def test_only_selected_family_lora_bundle_is_applied(self):
        sdxl_bundle = {
            "family": "sdxl",
            "lora_stack": [("sdxl.safetensors", 0.8, 0.7)],
            "active_loras": "sdxl.safetensors",
            "trigger_words": "sdxl trigger",
        }
        zit_bundle = {
            "family": "z_image_turbo",
            "lora_stack": [("zit.safetensors", 1.0, 1.0)],
            "active_loras": "zit.safetensors",
            "trigger_words": "zit trigger",
        }
        result = self.result(self.module.CMKPipeCreateImage().create_image(**{
            "GLOBAL PROMPT POS": "base",
            "model_family": "Z-Image Turbo",
            "LoRA SDXL": sdxl_bundle,
            "LoRA ZIT": zit_bundle,
        }))
        pipe = result[1]
        self.assertEqual(pipe["lora_stack"], zit_bundle["lora_stack"])
        self.assertEqual(pipe["active_loras"], "zit.safetensors")
        self.assertEqual(pipe["prompt_pos"], "base\nzit trigger")

    def test_hybrid_activates_both_stages_with_separate_lora_bundles(self):
        sdxl_bundle = {
            "family": "sdxl", "lora_stack": [("sdxl", 1.0, 1.0)],
            "active_loras": "sdxl", "trigger_words": "sdxl trigger",
        }
        zit_bundle = {
            "family": "z_image_turbo", "lora_stack": [("zit", 1.0, 1.0)],
            "active_loras": "zit", "trigger_words": "zit trigger",
        }
        result = self.result(self.module.CMKPipeCreateImage().create_image(**{
            "GLOBAL PROMPT POS": "base",
            "model_family": "Hybrid",
            "LoRA SDXL": sdxl_bundle,
            "LoRA ZIT": zit_bundle,
            "HYBRID BALANCE": 2,
        }))
        sdxl, zit = result[:2]
        self.assertTrue(sdxl["family_active"])
        self.assertTrue(zit["family_active"])
        self.assertTrue(sdxl["hybrid_mode"])
        self.assertEqual("base\nsdxl trigger", sdxl["prompt_pos"])
        self.assertEqual("base\nzit trigger", zit["prompt_pos"])
        self.assertEqual(2, zit["hybrid_balance"])
        self.assertEqual(85, zit["hybrid_sdxl_handoff"])
        self.assertEqual(0.20, zit["hybrid_zit_denoise"])

    def test_hybrid_inpaint_patches_only_sdxl_branch(self):
        image = torch.zeros((1, 16, 16, 3))
        mask = torch.zeros((1, 16, 16))
        mask[:, 4:12, 4:12] = 1
        result = self.result(self.module.CMKPipeCreateImage().create_image(**{
            "GLOBAL PROMPT POS": "repair",
            "INPAINT_MODE": "Inpaint",
            "model_family": "Hybrid",
            "resolution": "512x512",
            "IMAGE": image,
            "MASK": mask,
            "FILENAME": "hybrid-inpaint.png",
        }))
        sdxl, zit = result[:2]
        self.assertTrue(sdxl["family_active"])
        self.assertTrue(sdxl["boolean_inpaint_mode"])
        self.assertTrue(sdxl["hybrid_inpaint_mode"])
        self.assertTrue(zit["family_active"])
        self.assertFalse(zit["boolean_inpaint_mode"])
        self.assertTrue(zit["hybrid_inpaint_mode"])
        self.assertEqual("hybrid_finish", zit["inpaint_process_mode"])

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
                "LoRA SDXL": {
                    "family": "sdxl",
                    "lora_stack": [("must-be-ignored.safetensors", 1.0, 1.0)],
                    "active_loras": "must be ignored",
                    "trigger_words": "",
                },
            }
        ))

        self.assertEqual(pipe["prompt_pos"], "futuristic botanical observatory")
        self.assertEqual(pipe["prompt_neg"], "low quality")
        self.assertEqual(pipe["active_loras"], "")
        self.assertEqual(pipe["fill_masked_area"], "noise")
        self.assertNotIn("remove_isolated", pipe)
        self.assertNotIn("remove_result_image", pipe)
        self.assertNotIn("lama", self.module.MASKED_AREA_FILL)

    def test_hybrid_remove_without_user_prompt_supplies_both_family_pipes(self):
        image = torch.zeros((1, 16, 16, 3))
        mask = torch.zeros((1, 16, 16))
        mask[:, 4:12, 4:12] = 1

        result = self.result(self.module.CMKPipeCreateImage().create_image(**{
            "GLOBAL PROMPT POS": "",
            "INPAINT_MODE": "Inpaint",
            "process_mode": "Remove Object",
            "model_family": "Hybrid",
            "resolution": "512x512",
            "IMAGE": image,
            "MASK": mask,
            "FILENAME": "hybrid-remove.png",
        }))

        for family_pipe in result[:2]:
            self.assertEqual(family_pipe["prompt_pos"], "")
            self.assertIn("surrounding scene", family_pipe["effective_prompt_pos"])
            self.assertIn("foreground subject", family_pipe["effective_prompt_neg"])
            self.assertEqual(family_pipe["prompt_source"], "INTERNAL REMOVE GUIDANCE")

    def test_hybrid_extend_without_user_prompt_supplies_both_family_pipes(self):
        image = torch.zeros((1, 16, 16, 3))

        result = self.result(self.module.CMKPipeCreateImage().create_image(**{
            "GLOBAL PROMPT POS": "",
            "PROMPT NEG": "low quality",
            "INPAINT_MODE": "Inpaint",
            "process_mode": "Extend Image",
            "model_family": "Hybrid",
            "resolution": "512x512",
            "IMAGE": image,
            "FILENAME": "hybrid-extend.png",
        }))

        for family_pipe in result[:2]:
            self.assertEqual(family_pipe["prompt_pos"], "")
            self.assertIn("beyond its original boundaries", family_pipe["effective_prompt_pos"])
            self.assertEqual(family_pipe["effective_prompt_neg"], "low quality")
            self.assertEqual(family_pipe["prompt_source"], "INTERNAL EXTEND GUIDANCE")

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
