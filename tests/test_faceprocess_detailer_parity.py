"""FaceProcess 2.8: same model family, same operation, same parameter semantics
as the Multi-Model Detailer.

The FaceProcess branch materialises through the real CMK Detailer Prepare
Multi-Model; only weights, SAM, CLIP encode and model patches are stubbed and
recorded.  Checked: sampling entry -> denoise, prompt and seed resolution with
and without a first pass, SDXL LoRA and LoRA ZIT paths.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_faceprocess_parity_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import torch  # noqa: E402

mm = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_faceprocess_multimodel")
smart = importlib.import_module(f"{TEST_PACKAGE}.nodes.image.smart_detailer")
detailer_mm = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_detailer_prepare_multimodel")
detailer_sdxl = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_detailer_prepare")
native = importlib.import_module(f"{TEST_PACKAGE}.engine.native_detailer")

CMK_TEST_TIER = "T0"
IMAGE = torch.zeros(1, 64, 64, 3)
SDXL_SPEC = {"model_family": "sdxl", "ckpt_name": "ck.safetensors", "materialized": False}
ZIT_SPEC = {"model_family": "z_image_turbo", "diffusion_model_name": "z.safetensors", "text_encoder_name": "q",
            "vae_name": "ae", "weight_dtype": "default", "clip_device": "default", "materialized": False}
IMAGE_SOURCE = {"result_contract": "family_neutral", "source_model_family": "image"}
SDXL_PASS = {"result_contract": "family_neutral", "source_model_family": "sdxl", "seed": 7,
             "prompt_pos": "first pass", "prompt_neg": "bad", "lora_syntax": "<lora:sdxl_pass.safetensors:1>"}
ZIT_PASS = {"result_contract": "family_neutral", "source_model_family": "z_image_turbo", "seed": 11,
            "prompt_pos": "zit pass", "lora_syntax": "<lora:zit_pass.safetensors:1>"}


def settings(**overrides):
    values = {
        "sam_model_name": "sam.pth", "sam_device_mode": "CPU", "use_prompt_lora_from_sampler": True,
        "use_lora_from_1st_pass": False, "use_1st_pass_sampling": True, "lora_name": "local_sdxl.safetensors",
        "strength_model": 1.0, "strength_clip": 1.0, "prompt_pos": "local portrait", "prompt_neg": "local bad",
        "steps": 30, "cfg": 3.0, "sampling_entry": 95.0, "zit_steps": 8, "zit_cfg": 1.0, "zit_sampling_entry": 85.0,
        "sampler": "euler", "scheduler": "simple", "zit_sampler": "res_multistep", "zit_scheduler": "simple",
        "stop_at_clip_layer": -2, "pag_scale": 2.5, "sampling": "v_prediction", "zsnr": True, "zit_model_shift": 3.0,
        "freeu_enabled": True, "freeu_b1": 1.3, "freeu_b2": 1.4, "freeu_s1": 0.9, "freeu_s2": 0.2,
    }
    values.update(overrides)
    return values


class Parity(unittest.TestCase):
    def setUp(self):
        self.events = []
        record = self.events.append

        class LoRA:
            def load_loras(self, model, clip, opt_lora_syntax="", opt_lora_stack=None):
                record(("lora_text", opt_lora_syntax))
                return model, clip, "", opt_lora_syntax

        patches = [
            (detailer_sdxl, "resolve_postprocess_model", lambda model: {**model, "model": "SDXL", "clip": "CLIP", "vae": "VAE"}),
            (detailer_sdxl, "CMKLoRATextLoader", LoRA),
            (detailer_mm, "CMKLoRATextLoader", LoRA),
            (native.CMKSAMLoader, "load", lambda self, name, mode="CPU": ("SAM",)),
            (detailer_sdxl.CMKDetailerPreparePipe, "_encode", staticmethod(lambda clip, text: (record(("encode", text)), [["cond", {}]])[1])),
            (detailer_mm, "resolve_zit_postprocess_model", lambda model: {**model, "model": "ZIT", "vae": "ZVAE", "clip": None}),
            (detailer_mm, "ensure_zit_text_encoder", lambda model: ("ZCLIP", None)),
            (detailer_mm, "evict_zit_text_encoder", lambda model: "evicted"),
            (detailer_mm, "release_zit_resources", lambda reason: record(("release_zit", reason))),
            (detailer_mm, "_call_node", lambda names, *args: (args[0],) if names[0] == "ModelSamplingAuraFlow" else ([["zero", {}]],)),
            (detailer_mm, "_validate_conditioning", lambda value, name: value),
        ]
        # No translation service: prompts must arrive exactly as resolved.
        untranslated = lambda text: types.SimpleNamespace(text=text, status="disabled", log_line=lambda label: "")
        patches += [(detailer_sdxl, "translate_prompt", untranslated), (detailer_mm, "translate_prompt", untranslated)]
        helper = detailer_sdxl.CMKSamplerPrepareSDXLPipe
        patches += [(helper, "_clip_set_last_layer", lambda self, clip, layer: clip),
                    (helper, "_apply_single_lora", lambda self, model, clip, name, sm, sc: (record(("lora_local", name)), (model, clip, name))[1]),
                    (helper, "_apply_pag", lambda self, model, *a: model), (helper, "_apply_sampling", lambda self, model, *a: model),
                    (helper, "_apply_freeu", lambda self, model, *a: model)]
        for owner, name, value in patches:
            original = owner.__dict__[name] if isinstance(owner, type) and name in owner.__dict__ else getattr(owner, name)
            self.addCleanup(setattr, owner, name, original)
            setattr(owner, name, value)

    def branch(self, mode, process, **extra):
        face = mm.CMKFaceProcessPrepareMultiModelPipe().prepare(IMAGE=IMAGE, process_mode=mode, PROCESS=process, **settings(**extra))[0]
        return mm.CMKFaceProcessMultiModelPipe()._materialize_face(face)

    def kinds(self, kind):
        return [value for name, value in self.events if name == kind]

    def test_sampling_entry_uses_the_detailer_conversion(self):
        self.assertIs(mm.detailer_entry_to_denoise, smart._sampling_entry_to_denoise)
        self.assertAlmostEqual(mm.detailer_entry_to_denoise(95.0), 0.05)
        self.assertAlmostEqual(mm.detailer_entry_to_denoise(85.0), 0.15)
        self.assertEqual(self.branch("SDXL", IMAGE_SOURCE, sampling_entry=92.0, **{"MODEL SDXL": SDXL_SPEC})["face_sampling_entry"], 92.0)
        self.assertEqual(self.branch("Z-Image Turbo", IMAGE_SOURCE, zit_sampling_entry=88.0, **{"MODEL ZIT": ZIT_SPEC})["face_sampling_entry"], 88.0)

    def test_sdxl_image_source_uses_the_local_postprocess_path(self):
        pipe = self.branch("SDXL", IMAGE_SOURCE, **{"MODEL SDXL": SDXL_SPEC})
        self.assertEqual(self.kinds("encode"), ["local portrait", "local bad"])
        self.assertEqual(pipe["face_seed"], 0)
        self.assertEqual(self.kinds("lora_local"), ["local_sdxl.safetensors"])
        self.assertEqual(self.kinds("lora_text"), [])
        self.assertEqual(self.kinds("release_zit"), ["ZIT -> SDXL Detailer"], "guarded ZIT release before SDXL")
        self.assertFalse(pipe["face_use_prompt_from_1st_pass"])

    def test_sdxl_first_pass_prompt_seed_and_lora(self):
        pipe = self.branch("SDXL", SDXL_PASS, use_lora_from_1st_pass=True, **{"MODEL SDXL": SDXL_SPEC})
        self.assertEqual(self.kinds("encode"), ["first pass", "bad"])
        self.assertEqual(pipe["face_seed"], 7)
        self.assertEqual(self.kinds("lora_text"), ["<lora:sdxl_pass.safetensors:1>"])
        self.assertEqual(self.kinds("lora_local"), [], "first-pass LoRA replaces the local one")
        self.assertTrue(pipe["face_use_prompt_from_1st_pass"])

    def test_sdxl_does_not_take_a_zit_first_pass_lora(self):
        self.branch("SDXL", ZIT_PASS, use_lora_from_1st_pass=True, **{"MODEL SDXL": SDXL_SPEC})
        self.assertEqual(self.kinds("lora_text"), [])
        self.assertEqual(self.kinds("lora_local"), ["local_sdxl.safetensors"], "family-safe local LoRA")

    def test_zit_image_source_with_lora_zit(self):
        bundle = {"family": "z_image_turbo", "active_loras": "<lora:bundle.safetensors:0.8>", "lora_stack": None}
        pipe = self.branch("Z-Image Turbo", IMAGE_SOURCE, **{"MODEL ZIT": ZIT_SPEC, "LoRA ZIT": bundle})
        self.assertEqual(self.kinds("encode"), ["local portrait"])
        self.assertEqual(self.kinds("lora_text"), ["<lora:bundle.safetensors:0.8>"])
        self.assertEqual(self.kinds("lora_local"), [], "no SDXL LoRA on ZIT")
        self.assertEqual(pipe["face_seed"], 0)
        self.assertEqual(self.kinds("release_zit"), [])

    def test_zit_first_pass_lora_and_prompt(self):
        bundle = {"family": "z_image_turbo", "active_loras": "<lora:bundle.safetensors:0.8>"}
        pipe = self.branch("Z-Image Turbo", ZIT_PASS, use_lora_from_1st_pass=True, **{"MODEL ZIT": ZIT_SPEC, "LoRA ZIT": bundle})
        self.assertEqual(self.kinds("encode"), ["zit pass"])
        self.assertEqual(self.kinds("lora_text"), ["<lora:zit_pass.safetensors:1>"], "first-pass ZIT LoRA has priority")
        self.assertEqual(pipe["face_seed"], 11)

    def test_zit_ignores_an_sdxl_first_pass_lora(self):
        pipe = self.branch("Z-Image Turbo", SDXL_PASS, use_lora_from_1st_pass=True, **{"MODEL ZIT": ZIT_SPEC})
        self.assertEqual(self.kinds("lora_text"), [""], "no LoRA: SDXL first pass, no LoRA ZIT")
        self.assertEqual(self.kinds("encode"), ["first pass"], "the prompt is family neutral")
        self.assertEqual(pipe["face_seed"], 7)


class DetailerStepEntry(unittest.TestCase):
    def test_detailer_samples_with_the_family_entry(self):
        captured = []

        class StubDetailer:
            def doit(self, image, segs, denoise, **_kwargs):
                captured.append(denoise)
                return (segs,)

        original = mm.SEGSDetailer
        self.addCleanup(setattr, mm, "SEGSDetailer", original)
        mm.SEGSDetailer = StubDetailer
        node = mm.CMKFaceProcessMultiModelPipe()
        seg = native.CMKSEG(IMAGE[:, 8:40, 8:40, :], torch.ones(1, 32, 32), 0.9, (8, 8, 40, 40), (12, 12, 36, 36), "face", None, 0)
        node._detect_segs = lambda image, **kwargs: ((64, 64), [seg])
        node._merge_transport = lambda image, sel, segs, en, log, diag, **kw: (image, sel, segs, en, log, None, diag)
        for entry, expected in ((95.0, 0.05), (85.0, 0.15), (None, 0.2)):
            captured.clear()
            node._run_detailer({"face_mode": "z_image_turbo", "face_image": IMAGE, "face_sampling_entry": entry},
                               "Largest", {"detect_model": "bbox/face/face_yolov8m.pt", "refine_mode": "Off"},
                               opt_log=None, VISUAL=None, opt_diagnostic=None, unique_id="e", cache_key=None, cache_detail="")
            self.assertAlmostEqual(captured[0], expected, places=6)


if __name__ == "__main__":
    unittest.main()
