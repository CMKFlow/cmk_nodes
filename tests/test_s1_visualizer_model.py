"""MODEL at the Visualizer behind PostProcess Setup and a module MODEL forward.

MODEL runs through CMKModelForwardPipe, a node of its own, so it reaches the
Visualizer only when the Visualizer asks for it.  Behind PostProcess Setup the
Visualizer did not ask: SAVE stopped in CMKResultUnpackPipe after a complete
generation, and the PostProcess spec was never released.  The Visualizer now
requests MODEL whenever it is connected.

Model-free: install_generation_seams fakes CLIP, VAE and sampling.
"""

from __future__ import annotations

import unittest

from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
SAVE = {"SAVE ENABLED": True}
SDXL_ONLY = {"model_family": "SDXL"}
SAVE_TO_ZIT = "image"


def sdxl(runs):
    return [run["sampling"]["sdxl"] for run in runs]


def zit(runs):
    return [run["sampling"]["zit"] for run in runs]


class FinishToVisualizer(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tail = {"postprocess_tail": True}
        cls.saved = cls.run_case("hybrid_stop_sequence", stops=[False], visualizer_inputs=SAVE, **tail)
        cls.unsaved = cls.run_case("hybrid_stop_sequence", stops=[False], **tail)
        cls.reused = cls.run_case("hybrid_stop_sequence", stops=[False, False], visualizer_inputs=SAVE,
                                  mode="ram_pressure", evict_between_runs=True, **tail)
        cls.after_stop = cls.run_case("hybrid_stop_sequence", stops=[True, False], visualizer_inputs=SAVE, **tail)
        cls.sdxl = cls.run_case("hybrid_stop_sequence", stops=[False], visualizer_inputs=SAVE,
                                start_inputs=[SDXL_ONLY], **tail)
        cls.direct = cls.run_case("hybrid_stop_sequence", stops=[False, False], visualizer_inputs=SAVE)

    def test_finish_saves(self):
        run = self.saved["runs"][0]
        self.assertTrue(run["succeeded"], run["errors"])
        self.assertEqual((run["sampling"]["sdxl"], run["sampling"]["zit"]), (1, 1))
        self.assertEqual(run["function_calls"].get("CMKVisualizer"), 1)
        self.assertEqual(len([name for name in run["new_files"] if name.endswith(".png")]), 1)

    def test_model_is_the_postprocess_spec(self):
        for result in (self.saved, self.unsaved, self.sdxl):
            run = result["runs"][0]
            self.assertTrue(run["succeeded"], run["errors"])
            self.assertEqual(run["function_calls"].get("CMKModelForwardPipe"), 1)
            self.assertEqual(run["visualizer_model"]["model_role"], "postprocess")

    def test_postprocess_spec_is_not_loaded_by_the_request(self):
        run = self.saved["runs"][0]
        self.assertEqual(run["seams"]["substituted_calls"].get("checkpoint.load_resources"), 1)  # SDXL first pass

    def test_save_changes_nothing_else(self):
        saved, unsaved = self.saved["runs"][0], self.unsaved["runs"][0]
        self.assertEqual(saved["visual"], unsaved["visual"])
        self.assertEqual(saved["sampling"], unsaved["sampling"])
        self.assertEqual(unsaved["new_files"], [] if isinstance(unsaved["new_files"], list) else {})

    def test_image_is_unchanged_by_save(self):
        saved, unsaved = self.saved["runs"][0], self.unsaved["runs"][0]
        self.assertIsNotNone(saved["setup_image"])
        self.assertEqual(saved["setup_image"], unsaved["setup_image"])

    def test_zit_boundary_hit_saves_without_sampling(self):
        runs = self.reused["runs"]
        self.assertTrue(all(run["succeeded"] for run in runs), [run["errors"] for run in runs])
        self.assertEqual(zit(runs), [1, 0])
        self.assertEqual(sdxl(runs), [1, 0])
        self.assertEqual(len([name for name in runs[1]["new_files"] if name.endswith(".png")]), 1)

    def test_finish_from_the_stored_handoff(self):
        runs = self.after_stop["runs"]
        self.assertTrue(all(run["succeeded"] for run in runs), [run["errors"] for run in runs])
        self.assertEqual(sdxl(runs), [1, 0])
        self.assertEqual(zit(runs), [0, 1])

    def test_sdxl_only_path(self):
        run = self.sdxl["runs"][0]
        self.assertEqual(run["sampling"]["zit"], 0)
        self.assertEqual(run["sampling"]["sdxl"], 1)
        self.assertEqual(len([name for name in run["new_files"] if name.endswith(".png")]), 1)

    def test_direct_visualizer_keeps_the_released_model(self):
        first, second = self.direct["runs"]
        self.assertTrue(first["succeeded"] and second["succeeded"], first["errors"] + second["errors"])
        self.assertEqual(zit(self.direct["runs"]), [1, 0])
        self.assertEqual(second["function_calls"].get("CMKModelForwardPipe"), None)


if __name__ == "__main__":
    unittest.main()
