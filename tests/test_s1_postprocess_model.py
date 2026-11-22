"""H-2 — standalone PostProcess workflows load model resources only when an
active consumer needs them.

``CMK Load Image`` passes ``MODEL SDXL (opt)`` through and has no lazy
input, so whatever feeds it runs on every prompt.  ``CMKCheckpointVAELoaderPipe``
materialises model/clip/vae immediately; the new ``CMKPostProcessModelPipe``
returns the ``make_postprocess_model_spec`` contract that the PostProcess
Boundary already uses, and the first real consumer materialises it
(``resolve_postprocess_model``).

Legacy: two fixtures exported before H-2 (``*_legacy_checkpoint``) pin the
eager loading of the old wiring.  (The migration check of the CMK 2.6
standalone showcase workflows ended with their removal from the 2.8 package.)
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from harness.comfy_runtime import boot
from harness.support import GLOBAL_ENABLE, ScenarioCase

CMK_TEST_TIER = "T1"
TESTS = Path(__file__).resolve().parent
CHECKPOINT_FIXTURES = ("detailer_std", "detailer_adv", "faceprocess_std", "faceprocess_adv",
                       "facerebuild_std", "facerebuild_adv", "maskdetailer")
LOAD = "checkpoint.load_resources"
RESOLVE = "prepare.resolve_postprocess_model"


def _model_source(fixture):
    prompt = json.loads((TESTS / "fixtures" / "api_prompts" / f"{fixture}.json").read_text(encoding="utf-8"))
    return prompt["1"]["inputs"]["MODEL SDXL (opt)"][0], prompt


class LegacyCheckpointLoadsEagerly(ScenarioCase):
    """The old wiring keeps its behaviour: the checkpoint loads on every prompt."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bypass = cls.run_case("run_fixture", fixture="detailer_std_legacy_checkpoint",
                                  widget_overrides={"8417.detailer_global_enable": False})
        cls.active = cls.run_case("run_fixture", fixture="detailer_std_legacy_checkpoint")
        cls.pending = cls.run_case("run_fixture", fixture="maskdetailer_legacy_checkpoint")

    def test_source_is_the_checkpoint_loader(self):
        for fixture in ("detailer_std_legacy_checkpoint", "maskdetailer_legacy_checkpoint"):
            source, prompt = _model_source(fixture)
            self.assertEqual(prompt[source]["class_type"], "CMKCheckpointVAELoaderPipe")

    def test_bypass_loads_the_checkpoint(self):
        run = self.bypass["runs"][0]
        self.assertTrue(run["succeeded"], run["errors"])
        self.assertEqual(run["seams"]["substituted_calls"], {LOAD: 1})

    def test_active_loads_before_prepare(self):
        run = self.active["runs"][0]
        self.assertEqual(run["seams"]["substituted_calls"], {LOAD: 1})
        self.assertEqual(run["seams"]["stopped_calls"], {RESOLVE: 1})

    def test_pending_mask_detailer_loads_the_checkpoint(self):
        run = self.pending["runs"][0]
        self.assertTrue(run["succeeded"], run["errors"])
        self.assertEqual(run["seams"]["substituted_calls"], {LOAD: 1})


class DeferredMaterialisation(ScenarioCase):
    """Canonical fixtures (built from the current model source)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bypass = {}
        for fixture in CHECKPOINT_FIXTURES:
            result = cls.run_case("run_fixture", fixture=fixture,
                                  widget_overrides={f"8417.{GLOBAL_ENABLE[fixture]}": False})
            cls.bypass[fixture] = None if result is None else result["runs"][0]
        cls.active = cls.run_case("run_fixture", fixture="detailer_std")
        cls.pending = cls.run_case("run_fixture", fixture="maskdetailer")

    def test_model_source_is_the_postprocess_model(self):
        for fixture in CHECKPOINT_FIXTURES:
            source, prompt = _model_source(fixture)
            self.assertEqual(prompt[source]["class_type"], "CMKPostProcessModelPipe", fixture)

    def test_bypass_loads_nothing(self):
        for fixture, run in self.bypass.items():
            self.assertTrue(run["succeeded"], run["errors"])
            self.assertEqual(run["seams"]["substituted_calls"], {}, fixture)

    def test_active_materialises_at_the_first_consumer(self):
        run = self.active["runs"][0]
        self.assertEqual(run["seams"]["substituted_calls"], {})
        self.assertEqual(run["seams"]["stopped_calls"], {RESOLVE: 1})
        self.assertEqual([e["node_type"] for e in run["errors"]], ["CMKDetailerPreparePipe"])

    def test_pending_mask_detailer_loads_nothing(self):
        run = self.pending["runs"][0]
        self.assertTrue(run["succeeded"], run["errors"])
        self.assertEqual(run["seams"]["substituted_calls"], {})


class PostProcessModelNode(unittest.TestCase):
    """The node itself, in-process with the checkpoint loading substituted."""

    @classmethod
    def setUpClass(cls):
        from harness.seams import Seams, fake_checkpoint_resources

        cls.root = Path(tempfile.mkdtemp(prefix="cmk-s1-ppmodel-"))
        runtime = boot(cls.root)
        cls.loader = runtime.cmk_module("pipe.loaders.checkpoint_vae_loader")
        cls.seams = Seams()
        cls.seams.substitute(runtime.cmk_module("loader.checkpoint_vae_loader"), "load_checkpoint_vae_resources",
                             "checkpoint.load_resources", fake_checkpoint_resources, rebind=True)

    @classmethod
    def tearDownClass(cls):
        cls.seams.uninstall()
        shutil.rmtree(cls.root, ignore_errors=True)

    def _node(self):
        return self.loader.CMKPostProcessModelPipe()

    def test_returns_the_postprocess_spec_without_loading(self):
        self.seams.reset_counters()
        (model,) = self._node().load("ckpt.safetensors", "vae.safetensors", True)
        self.assertEqual(model, self.loader.make_postprocess_model_spec("ckpt.safetensors", "vae.safetensors", True))
        self.assertEqual(dict(self.seams.substituted_calls), {})

    def test_resolution_materialises_once_and_visualizer_unloads(self):
        self.seams.reset_counters()
        (spec,) = self._node().load("lifecycle.safetensors", "vae.safetensors", True)
        self.assertEqual(self.loader.unload_model_pipe(spec), "NOT MATERIALIZED")
        first = self.loader.resolve_postprocess_model(spec)
        second = self.loader.resolve_postprocess_model(spec)
        self.assertTrue(first["materialized"])
        self.assertIs(first["model"], second["model"])
        self.assertEqual(dict(self.seams.substituted_calls), {"checkpoint.load_resources": 1})
        self.assertNotEqual(self.loader.unload_model_pipe(spec), "NOT MATERIALIZED")

    def test_none_is_selectable_and_fails_clearly_at_a_consumer(self):
        choices = self.loader.CMKPostProcessModelPipe.INPUT_TYPES()["required"]["ckpt_name"][0]
        self.assertEqual(choices[-1], "None")
        (spec,) = self._node().load("None", "vae.safetensors", True)
        self.assertTrue(self.loader.is_postprocess_model_disabled(spec))
        self.assertEqual(self.loader.unload_model_pipe(spec), "NOT CONFIGURED")
        with self.assertRaisesRegex(ValueError, "CMK PostProcess Model.*None"):
            self.loader.resolve_postprocess_model(spec)


if __name__ == "__main__":
    unittest.main()
