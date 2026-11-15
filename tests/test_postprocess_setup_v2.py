"""CMK Flow · 30 PostProcess Setup v2 (2.8): SDXL and ZIT PostProcess models.

The generation completion of the Setup runs for real (``_finish``: PROCESS,
LOG, SDXL model specification); only the completion preflight, the release of
the generation models and the completion cache are replaced.  Nothing loads a
model.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_postprocess_setup_v2_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import torch  # noqa: E402

family = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_family_result")
v2 = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_postprocess_setup_v2")
source = importlib.import_module(f"{TEST_PACKAGE}.pipe.loaders.postprocess_model_source")
mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")

CMK_TEST_TIER = "T0"
SDXL = ("sdxl_post.safetensors", "post.vae", False)
ZIT = {"postprocess_zit_model": "zit_post.safetensors", "postprocess_zit_text_encoder": "qwen.safetensors",
       "postprocess_zit_vae": "zit_ae.safetensors", "postprocess_zit_weight_dtype": "fp8_e4m3fn",
       "postprocess_zit_clip_device": "cpu"}
IMAGE = torch.zeros(1, 8, 8, 3)


def generation(family_name):
    suffix = "SDXL" if family_name == "sdxl" else "ZIT"
    other = "ZIT" if suffix == "SDXL" else "SDXL"
    return {
        f"PROCESS {suffix}": {"model_family": family_name, "family_active": True, "seed": 7},
        f"PROCESS {other}": {"family_active": False},
        f"MODEL {suffix}": {"model_role": "generation", "model_family": family_name},
        f"IMAGE {suffix}": IMAGE,
        f"LOG {suffix}": {"type": "CMK_LOG_PIPE", "blocks": []},
        f"VISUAL {suffix}": {"type": "CMK_VISUAL_PIPE", "version": 1, "providers": []},
        "prompt": {}, "unique_id": "30",
    }


class Completion(unittest.TestCase):
    """Preflight, generation release and completion store replaced."""

    def setUp(self):
        self.released = []
        contracts = {"sdxl": family._SDXL_COMPLETION, "z_image_turbo": family._ZIT_COMPLETION}

        def preflight(cls, _contracts, _prompt, _unique_id):
            return {"contract": contracts[self.family], "key": "k" * 64, "cached": None, "detail": "test", "graph": None}

        def release(family_name, model, process):
            self.released.append(family_name)
            return {"generation_unload": "UNLOADED", "unload_requested": True, "cleared_generation_entries": 1}, None

        for target, value in (("_resolve_completion_preflight", preflight),
                              ("_release_completed_generation_resources", release),
                              ("_completion_finalization_evidence", lambda *a: (None, "test"))):
            patcher = patch.object(family, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def both(self, family_name):
        self.family = family_name
        legacy = family.CMKPostProcessSetupPipe().merge(*SDXL, **generation(family_name))
        new = v2.CMKPostProcessSetupV2Pipe().merge(*SDXL, **ZIT, **generation(family_name))
        return legacy, new

    def test_sdxl_outputs_are_the_legacy_outputs(self):
        for family_name in ("sdxl", "z_image_turbo"):
            with self.subTest(family=family_name):
                legacy, new = self.both(family_name)
                self.assertEqual(len(new), 6)
                self.assertEqual(new[1], legacy[0], "MODEL SDXL: the legacy PostProcess MODEL")
                self.assertIs(new[3], legacy[2])
                self.assertEqual({k: v for k, v in new[2].items() if k != "postprocess_zit_model"}, legacy[1])
                self.assertEqual(new[4]["blocks"][:-1], legacy[3]["blocks"], "LOG: legacy blocks + ZIT block")
                self.assertEqual(new[5], legacy[4])

    def test_zit_model_is_the_postprocess_model_source_spec(self):
        for family_name in ("sdxl", "z_image_turbo"):
            with self.subTest(family=family_name):
                _legacy, new = self.both(family_name)
                expected = source.CMKPostProcessModelSourcePipe().load(*SDXL, *ZIT.values())
                zit = dict(new[0])
                self.assertEqual(zit.pop("source_model_family"), family_name)
                self.assertEqual(zit.pop("generation_family"), family_name)
                self.assertEqual(zit, expected[1], "the ZIT spec of CMK PostProcess Model")
                self.assertEqual(new[1]["model_family"], "sdxl")
                self.assertEqual(new[0]["model_family"], "z_image_turbo")
                self.assertFalse(new[0]["materialized"])
                self.assertFalse(new[1]["materialized"])
                self.assertEqual(new[2]["postprocess_zit_model"], "zit_post.safetensors")
                block = new[4]["blocks"][-1]
                self.assertEqual(block["title"], "PostProcess Model ZIT")
                self.assertIn("MODEL LOAD      : DEFERRED UNTIL REQUIRED", block["lines"])
                self.assertEqual(self.released[-1], family_name, "generation released exactly as legacy")

    def test_restore_from_the_completion_cache(self):
        self.family = "sdxl"
        cached = {"model_descriptor": {"model_role": "generation", "model_family": "sdxl"},
                  "process": {"model_family": "sdxl", "family_active": True},
                  "image": IMAGE, "log": {"type": "CMK_LOG_PIPE", "blocks": []}, "visual": None}
        with patch.object(family, "_resolve_completion_preflight", lambda *a: {
                "contract": family._SDXL_COMPLETION, "key": "k" * 64, "cached": cached, "detail": "hit", "graph": None}):
            new = v2.CMKPostProcessSetupV2Pipe().merge(*SDXL, **ZIT)
        self.assertEqual(new[0]["diffusion_model_name"], "zit_post.safetensors")
        self.assertEqual(new[1]["ckpt_name"], "sdxl_post.safetensors")
        self.assertIs(new[3], IMAGE)

    def test_stop_after_handoff_preview(self):
        process = {"model_family": "sdxl", "family_active": True, "stop_after_handoff": True,
                   "hybrid_mode": False, "instantid_enabled": True}
        inputs = {"PROCESS SDXL": process, "MODEL SDXL": {"model_family": "sdxl"}, "IMAGE SDXL": IMAGE,
                  "LOG SDXL": {"type": "CMK_LOG_PIPE", "blocks": []}, "VISUAL SDXL": None}
        with patch.object(family, "stop_after_handoff_active", lambda value: value is process):
            legacy = family.CMKPostProcessSetupPipe().merge(*SDXL, **inputs)
            new = v2.CMKPostProcessSetupV2Pipe().merge(*SDXL, **ZIT, **inputs)
        self.assertEqual(new[2]["result_contract"], "preview_handoff")
        self.assertEqual(new[2], legacy[1])
        self.assertEqual(new[4], legacy[3])
        self.assertEqual(new[0]["model_family"], "z_image_turbo")


class Contract(unittest.TestCase):
    def test_registration_and_ports(self):
        node = mappings.NODE_CLASS_MAPPINGS["CMKPostProcessSetupV2Pipe"]
        self.assertIs(node, v2.CMKPostProcessSetupV2Pipe)
        self.assertEqual(mappings.NODE_DISPLAY_NAME_MAPPINGS["CMKPostProcessSetupV2Pipe"], "CMK Flow · 30 PostProcess Setup v2")
        self.assertEqual(node.RETURN_NAMES, ("MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL"))
        self.assertEqual(node.RETURN_TYPES, ("CMK_MODEL_PIPE", "CMK_MODEL_PIPE", "CMK_RESULT_PROCESS", "IMAGE",
                                             "CMK_LOG_PIPE", "CMK_VISUAL_PIPE"))
        self.assertEqual(family.CMKPostProcessSetupPipe.RETURN_NAMES, ("MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL"),
                         "the legacy Setup keeps its outputs")

    def test_inputs_extend_the_legacy_setup(self):
        legacy = family.CMKPostProcessSetupPipe.INPUT_TYPES()
        new = v2.CMKPostProcessSetupV2Pipe.INPUT_TYPES()
        self.assertEqual(new["optional"], legacy["optional"], "same generation inputs")
        self.assertEqual(list(new["required"])[:3], list(legacy["required"]), "the SDXL settings first, unchanged")
        for name in legacy["required"]:
            self.assertEqual(new["required"][name], legacy["required"][name])
        zit = source.CMKPostProcessModelSourcePipe.INPUT_TYPES()["required"]
        for name, source_name in v2.ZIT_WIDGETS.items():
            self.assertEqual(new["required"][name][0], zit[source_name][0], "choices of CMK PostProcess Model")
            self.assertTrue(new["required"][name][1]["advanced"])
        self.assertNotIn("postprocess_zit_model", family.CMKPostProcessSetupPipe.INPUT_TYPES()["required"])

    def test_visualizer_sees_the_v2_completion(self):
        prompt = {"30": {"class_type": "CMKPostProcessSetupV2Pipe", "inputs": {}},
                  "v": {"class_type": "CMKVisualizer", "inputs": {"IMAGE": ["30", 3]}}}
        self.assertTrue(family._visualizer_upstream_contains_postprocess(prompt, {"IMAGE": ["30", 3]}))


if __name__ == "__main__":
    unittest.main()
