"""S1-T6 §3.D — Upscale & Save: IMAGE pass-through checked separately from saving.

The upscale engine is substituted by a deterministic 2x nearest-neighbour
copy; files are written into the harness' isolated output directory only.
OD-8 and T-5 are fixed by H-4 (definition 1.4.0); the pre-fix legacy
definition is pinned in test_s1_upscale_save_legacy_reference.py.
"""

from __future__ import annotations

import unittest

from harness.support import ScenarioCase, gate_of

CMK_TEST_TIER = "T1"


def _shape(output):
    return output["shape"]


class UpscaleSave(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.current = cls.run_case(
            "run_fixture", fixture="upscale_save", runs=2,
            inspect={"gate": f"{gate_of('upscale_save')}:1", "loader": "1:2"},
        )
        cls.legacy = cls.run_case(
            "run_fixture", fixture="upscale_save_legacy_fullflow", runs=1,
            inspect={"gate": f"{gate_of('upscale_save_legacy_fullflow')}:1", "loader": "1:2"},
        )
        workdir = cls.workdir("restart")
        first = cls.run_case("run_fixture", workdir=workdir, fixture="upscale_save")
        second = cls.run_case("run_fixture", workdir=workdir, fixture="upscale_save")
        cls.restart = None if first is None else first["runs"] + second["runs"]
        cls.off = cls.run_case("run_fixture", fixture="upscale_save",
                               widget_overrides={"8417.SAVE ENABLED": False})

    # -- IMAGE pass-through (OD-8) -----------------------------------------
    def test_image_out_equals_image_in(self):
        cold = self.current["runs"][0]
        self.assertEqual(cold["outputs"]["gate"], cold["outputs"]["loader"])

    # -- saving --------------------------------------------------------------
    def test_saved_copy_is_upscaled(self):
        cold = self.current["runs"][0]
        loader = _shape(cold["outputs"]["loader"])
        self.assertEqual(list(cold["new_files"].values()), [[loader[2] * 2, loader[1] * 2]])

    def test_save_disabled_writes_nothing(self):
        run = self.off["runs"][0]
        self.assertTrue(run["succeeded"], run["errors"])
        self.assertEqual(run["new_files"], {})

    def test_repeated_save_writes_a_new_file(self):
        # T-5 / OD-7: every executed save job writes a file.
        warm = self.current["runs"][1]
        self.assertEqual([name.rsplit("_", 1)[-1] for name in warm["new_files"]], ["00002.png"])
        self.assertEqual(list(warm["new_file_hashes"].values()), list(self.current["runs"][0]["new_file_hashes"].values()))

    def test_save_after_restart_writes_next_counter(self):
        first, second = self.restart
        self.assertEqual([name.rsplit("_", 1)[-1] for name in first["new_files"]], ["00001.png"])
        self.assertEqual([name.rsplit("_", 1)[-1] for name in second["new_files"]], ["00002.png"])

    def test_warm_save_runs_no_upstream_compute(self):
        warm = self.current["runs"][1]
        self.assertNotIn("CMKImageLoadAndResizePipe", warm["function_calls"])
        self.assertEqual(warm["seams"]["substituted_calls"], {})

    # -- legacy semantics (K-3, Option A) ---------------------------------
    def test_legacy_and_canonical_save_the_same_upscaled_copy(self):
        legacy = self.legacy["runs"][0]
        current = self.current["runs"][0]
        self.assertEqual(list(legacy["new_file_hashes"].values()), list(current["new_file_hashes"].values()))
        self.assertEqual(list(legacy["new_files"].values()), list(current["new_files"].values()))
        # Only the chain image differs: legacy passes the upscaled copy on.
        self.assertEqual(_shape(legacy["outputs"]["gate"]), [1, 1664, 2304, 3])
        self.assertEqual(current["outputs"]["gate"], current["outputs"]["loader"])


if __name__ == "__main__":
    unittest.main()
