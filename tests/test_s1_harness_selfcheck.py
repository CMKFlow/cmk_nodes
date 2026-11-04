"""S1 §3.0 — the harness must observe execution without changing it."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from harness.restart import run_scenario

CMK_TEST_TIER = "T1"


class HarnessSelfCheck(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="cmk-s1-"))

    def tearDown(self):
        # Only the private harness work directory is removed, never a real cache.
        shutil.rmtree(self.root, ignore_errors=True)

    def _comparable(self, result):
        return [
            (run["succeeded"], run["executed"], run["cached"], run["boundary_status"], run["image_hash"])
            for run in result["runs"]
        ]

    def test_harness_probe_neutrality(self):
        plain = run_scenario("selfcheck_chain", self.root / "plain", probes=False)
        probed = run_scenario("selfcheck_chain", self.root / "probed", probes=True)
        self.assertEqual(self._comparable(plain), self._comparable(probed))
        calls = probed["probes"]["function_calls"]
        # The probes must actually observe the cold run (no silent no-op probes).
        for class_type in ("CMKImageLoadAndResizePipe", "CMKUpscaleSaveBoundaryCache", "CMKVisualizer"):
            self.assertGreaterEqual(len(calls.get(class_type, [])), 1, class_type)
        self.assertTrue(probed["probes"]["lazy_requests"].get("CMKUpscaleSaveBoundaryCache"))
        self.assertTrue(probed["probes"]["statuses"])

    def test_harness_warm_run_uses_comfyui_cache(self):
        result = run_scenario("selfcheck_chain", self.root / "warm", probes=True)
        cold, warm = result["runs"]
        self.assertEqual(cold["cached"], [])
        # Loader and boundary come from the ComfyUI cache; the Visualizer is an
        # OUTPUT_NODE with IS_CHANGED = nan and must run again on every prompt.
        self.assertEqual(sorted(warm["cached"]), ["1", "2"])
        self.assertEqual(warm["entered"], ["3"])
        self.assertEqual(
            len(result["probes"]["function_calls"]["CMKVisualizer"]), 2,
            "Visualizer FUNCTION runs once per prompt",
        )
        self.assertEqual(len(result["probes"]["function_calls"]["CMKUpscaleSaveBoundaryCache"]), 1)
        self.assertEqual(cold["image_hash"], warm["image_hash"])

    def test_harness_restart_isolation(self):
        workdir = self.root / "restart"
        first = run_scenario("selfcheck_chain", workdir, runs=1)
        second = run_scenario("selfcheck_chain", workdir, runs=1)
        # A fresh process has no ComfyUI output cache ...
        self.assertEqual(second["runs"][0]["cached"], [])
        self.assertIn("2", second["runs"][0]["executed"])
        # ... while the private work directory survives the restart.
        self.assertTrue((workdir / "temp" / "cmk" / "upscale_save_boundary").is_dir())
        self.assertEqual(first["runs"][0]["image_hash"], second["runs"][0]["image_hash"])

    def test_harness_ram_pressure_mode_runs(self):
        result = run_scenario("selfcheck_chain", self.root / "ram", mode="ram_pressure")
        self.assertTrue(all(run["succeeded"] for run in result["runs"]))
        self.assertEqual(sorted(result["runs"][1]["cached"]), ["1", "2"])


if __name__ == "__main__":
    unittest.main()
