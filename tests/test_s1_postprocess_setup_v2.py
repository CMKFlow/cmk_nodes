"""PostProcess Setup v2 behind a real generation graph (SDXL/ZIT Hybrid and SDXL).

The same flows run with CMK Flow · 30 PostProcess Setup and with v2: the
generation completion, IMAGE, PROCESS and the SDXL PostProcess MODEL at the
Visualizer are unchanged; v2 adds the ZIT PostProcess MODEL.  Neither
PostProcess model is loaded.

Model-free: install_generation_seams fakes CLIP, VAE and sampling.
"""

from __future__ import annotations

import unittest

from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
SDXL_ONLY = {"model_family": "SDXL"}


def without(view, *keys):
    return {key: value for key, value in (view or {}).items() if key not in keys}


class SetupV2(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.runs = {}
        if cls.paths is None:
            return  # setUp skips every test without the model-path YAML
        for name, start in (("hybrid", {}), ("sdxl", SDXL_ONLY)):
            for version in (True, "v2"):
                cls.runs[name, version] = cls.run_case(
                    "hybrid_stop_sequence", stops=[False], start_inputs=[start], postprocess_tail=version,
                )["runs"][0]

    def test_runs_succeed(self):
        for key, run in self.runs.items():
            with self.subTest(case=key):
                self.assertTrue(run["succeeded"], run["errors"])

    def test_generation_and_sdxl_path_are_unchanged(self):
        for name in ("hybrid", "sdxl"):
            with self.subTest(case=name):
                legacy, new = self.runs[name, True], self.runs[name, "v2"]
                self.assertEqual(new["sampling"], legacy["sampling"])
                self.assertEqual(new["setup_image"], legacy["setup_image"])
                self.assertEqual(new["visualizer_model"], legacy["visualizer_model"], "MODEL SDXL = legacy MODEL")
                self.assertEqual(without(new["setup_process"], "postprocess_zit_model"), legacy["setup_process"])
                self.assertEqual(new["seams"]["substituted_calls"].get("checkpoint.load_resources"),
                                 legacy["seams"]["substituted_calls"].get("checkpoint.load_resources"),
                                 "no additional SDXL load")

    def test_zit_postprocess_model(self):
        expected_family = {"hybrid": "z_image_turbo", "sdxl": "sdxl"}
        for name in ("hybrid", "sdxl"):
            with self.subTest(case=name):
                run = self.runs[name, "v2"]
                zit = run["setup_zit_model"]
                self.assertEqual((zit["model_role"], zit["model_family"]), ("postprocess", "z_image_turbo"))
                self.assertFalse(zit["materialized"])
                self.assertEqual(zit["diffusion_model_name"], "z_image_turbo_bf16.safetensors")
                self.assertEqual(zit["source_model_family"], expected_family[name])
                self.assertEqual(run["visualizer_model"]["model_family"], "sdxl")
                self.assertIsNone(self.runs[name, True]["setup_zit_model"])


if __name__ == "__main__":
    unittest.main()
