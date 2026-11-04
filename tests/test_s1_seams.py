"""S1-T6 — the engine substitutions must not change what ComfyUI observes.

Checks, each in fresh processes: node-class surface and ComfyUI cache
signatures with and without the standard seams (no execution), lazy requests,
execution order, ComfyUI cache reuse and output hashes on the model-free
self-check chain, and the separation of original vs. substituted calls
(including the H-2 checkpoint-loader substitution).
"""

from __future__ import annotations

import unittest

from harness.support import ScenarioCase, gate_of

CMK_TEST_TIER = "T1"


class SeamNeutrality(ScenarioCase):
    def _neutrality(self, fixture):
        result = self.run_case("seam_neutrality", fixture=fixture)
        self.assertTrue(result["surface_equal"], result["surface_diff"])
        self.assertTrue(result["surface_restored"])
        self.assertTrue(result["keys_equal"], result["keys_diff"])
        return result

    def test_detailer_std_surface_and_cache_keys_unchanged(self):
        result = self._neutrality("detailer_std")
        # Visual Compare and Visualizer return IS_CHANGED = nan: their keys are
        # unstable by design even without seams and are reported, not compared.
        self.assertEqual(result["volatile_nodes"], ["8417:7708", "8418"])
        # 8 since OD-2: CMK Model Forward feeds the public MODEL output.
        self.assertEqual(result["compared_nodes"], 8)

    def test_faceswap_std_surface_and_cache_keys_unchanged(self):
        result = self._neutrality("faceswap_std")
        self.assertEqual(result["volatile_nodes"], ["8417:7855", "8418"])

    def test_lazy_order_and_cache_reuse_unchanged_on_selfcheck_chain(self):
        plain = self.run_case("selfcheck_chain", probes=True)
        seamed = self.run_case("selfcheck_chain", probes=True, seams=True)
        project = lambda r: [
            (run["executed"], run["cached"], run["boundary_status"], run["image_hash"]) for run in r["runs"]
        ]
        self.assertEqual(project(plain), project(seamed))
        self.assertEqual(plain["probes"]["lazy_requests"], seamed["probes"]["lazy_requests"])
        self.assertEqual(plain["probes"]["function_calls"], seamed["probes"]["function_calls"])


class CounterSeparation(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bypass = cls.run_case(
            "run_fixture", fixture="detailer_std",
            widget_overrides={"8417.detailer_global_enable": False},
        )

    def test_substituted_seams_never_call_originals(self):
        seams = self.bypass["runs"][0]["seams"]
        self.assertEqual(seams["original_calls"], {})
        self.assertEqual(seams["tripped"], [])

    def test_h2_bypass_loads_no_checkpoint(self):
        # Since H-2 the fixture's model source is CMK PostProcess Model.
        run = self.bypass["runs"][0]
        self.assertEqual(run["seams"]["substituted_calls"], {})
        self.assertNotIn("CMKCheckpointVAELoaderPipe", run["function_calls"])


if __name__ == "__main__":
    unittest.main()
