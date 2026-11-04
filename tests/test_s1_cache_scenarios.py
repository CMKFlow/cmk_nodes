"""S1-T6 §3.B — cache behaviour on the fixtures: classic vs ram_pressure,
warm run vs forced eviction vs fresh process (simulated restart).

Counters are kept apart: node FUNCTION calls, substituted engine calls,
boundary statuses (persistent CMK cache) and ComfyUI cache reuse
(``execution_cached``).

H-1 (shared diagnostic tensors made every store fail) is fixed in the
boundary serializer; the persistent-HIT tests below are its regression guard.
H-3 / F-03 are fixed (the boundary publishes the module's VISUAL providers)
and H-4 is fixed (Upscale & Save 1.4.0); both are guarded below.
"""

from __future__ import annotations

import unittest

from harness.support import ScenarioCase, gate_of

CMK_TEST_TIER = "T1"
COMPUTE = {"faceswap_std": "CMKFaceSwapImagePipe", "upscale_save": "CMK_SmartUpscalerPipe"}
ENGINE = {"faceswap_std": "faceswap.swap", "upscale_save": "upscale.run"}
BOUNDARY_SCOPE = {"faceswap_std": "faceswap_boundary", "upscale_save": "upscale_save_boundary"}
PERSISTENT = ("ram_evict", "restart_classic", "restart_ram")


def _statuses(run, scope):
    return [status for s, status, _key in run["statuses"] if s == scope]


class CacheModes(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in COMPUTE:
            inspect = {"gate": f"{gate_of(fixture)}:1"}
            common = dict(fixture=fixture, inspect=inspect)
            cls.r[(fixture, "classic")] = cls.run_case("run_fixture", runs=2, mode="classic", **common)
            cls.r[(fixture, "ram")] = cls.run_case("run_fixture", runs=2, mode="ram_pressure", **common)
            cls.r[(fixture, "ram_evict")] = cls.run_case(
                "run_fixture", runs=2, mode="ram_pressure", evict_between_runs=True, **common)
            for label, mode in (("restart_classic", "classic"), ("restart_ram", "ram_pressure")):
                workdir = cls.workdir(f"{label}_{fixture}")
                first = cls.run_case("run_fixture", workdir=workdir, mode=mode, **common)
                second = cls.run_case("run_fixture", workdir=workdir, mode=mode, **common)
                cls.r[(fixture, label)] = None if first is None else {"runs": first["runs"] + second["runs"]}

    def _runs(self, fixture, variant):
        return self.r[(fixture, variant)]["runs"]

    def test_cold_run_computes_once_and_stores(self):
        for fixture in COMPUTE:
            for variant in ("classic", "ram") + PERSISTENT:
                with self.subTest(fixture=fixture, variant=variant):
                    cold = self._runs(fixture, variant)[0]
                    self.assertTrue(cold["succeeded"], cold["errors"])
                    self.assertEqual(cold["function_calls"].get(COMPUTE[fixture]), 1)
                    self.assertEqual(cold["seams"]["substituted_calls"].get(ENGINE[fixture]), 1)
                    self.assertEqual(cold["cached"], [])
                    # H-1 regression: STORE_FAILED here means shared diagnostic tensors broke the store again.
                    self.assertEqual(_statuses(cold, BOUNDARY_SCOPE[fixture]), ["MISS_STORED"])

    def test_warm_run_reuses_comfyui_cache_in_both_modes(self):
        for fixture in COMPUTE:
            for variant in ("classic", "ram"):
                with self.subTest(fixture=fixture, variant=variant):
                    warm = self._runs(fixture, variant)[1]
                    self.assertNotIn(COMPUTE[fixture], warm["function_calls"])
                    self.assertEqual(warm["seams"]["substituted_calls"], {})
                    self.assertTrue(warm["cached"])
                    self.assertEqual(_statuses(warm, BOUNDARY_SCOPE[fixture]), [])

    def test_ram_pressure_without_eviction_matches_classic(self):
        for fixture in COMPUTE:
            with self.subTest(fixture=fixture):
                classic = [sorted(run["cached"]) for run in self._runs(fixture, "classic")]
                ram = [sorted(run["cached"]) for run in self._runs(fixture, "ram")]
                self.assertEqual(classic, ram)

    def test_persistent_boundary_hit_after_eviction_and_restart(self):
        """The result comes from the CMK disk cache, not from ComfyUI's cache,
        and IMAGE/VISUAL equal the cold run pixel for pixel."""
        for fixture in COMPUTE:
            for variant in PERSISTENT:
                with self.subTest(fixture=fixture, variant=variant):
                    cold, again = self._runs(fixture, variant)
                    self.assertTrue(again["succeeded"], again["errors"])
                    statuses = _statuses(again, BOUNDARY_SCOPE[fixture])
                    self.assertEqual(statuses[-1], "HIT")
                    self.assertEqual(set(statuses), {"HIT_READY", "HIT"})
                    self.assertEqual(again["cached"], [])
                    self.assertEqual(again["outputs"]["gate"], cold["outputs"]["gate"])
                    self.assertEqual(again["visual"], cold["visual"])

    def test_faceswap_hit_runs_no_compute(self):
        # F-03 regression: VISUAL ACTIVE comes from the boundary, so a HIT
        # publishes the provider without pulling the compute node.
        for variant in PERSISTENT:
            with self.subTest(variant=variant):
                again = self._runs("faceswap_std", variant)[1]
                self.assertNotIn("CMKFaceSwapImagePipe", again["function_calls"])
                self.assertEqual(again["seams"]["substituted_calls"], {})

    def test_faceswap_hit_publishes_provider(self):
        # H-3 regression: the provider reaches Visualizer and Visual Compare.
        for variant in PERSISTENT:
            with self.subTest(variant=variant):
                again = self._runs("faceswap_std", variant)[1]
                self.assertEqual([p["module_type"] for p in again["visual"]["8418"]], ["CMKFaceSwapImagePipe"])
                self.assertEqual(again["visual"]["8418"], again["visual"]["8417:7855"])

    def test_upscale_save_hit_runs_no_upscaler_but_saves_again(self):
        """H-4 regression: the save endpoint reads the boundary result, so a
        HIT neither upscales nor unloads models, yet the save job still runs."""
        for variant in PERSISTENT:
            with self.subTest(variant=variant):
                cold, again = self._runs("upscale_save", variant)
                self.assertNotIn("CMK_SmartUpscalerPipe", again["function_calls"])
                self.assertEqual(again["seams"]["substituted_calls"], {})
                self.assertEqual(again["seams"]["watched_calls"], {})
                self.assertEqual(again["function_calls"].get("CMK_SaveProjectImage"), 1)
                self.assertEqual(list(again["new_file_hashes"].values()), list(cold["new_file_hashes"].values()))


class FileIdentity(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        common = dict(fixture="faceswap_std", boundary_class="CMKFaceSwapBoundaryCache")
        cls.original = cls.run_case("boundary_key", **common)
        cls.replaced = cls.run_case("boundary_key", input_source="cmk_s1_input_replacement.png", **common)

    def test_baseline_m7_same_name_new_content_keeps_boundary_key(self):
        self.assertEqual(self.original["key"], self.replaced["key"])

    @unittest.expectedFailure  # M7
    def test_same_name_new_content_changes_boundary_key(self):
        self.assertNotEqual(self.original["key"], self.replaced["key"])


if __name__ == "__main__":
    unittest.main()
