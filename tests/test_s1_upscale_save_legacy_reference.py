"""K-3 Option A / A10-L — the embedded CMK 2.6 Full Flow Upscale & Save
definition is frozen.  Its observable behaviour is pinned to values recorded
on 2026-11-04 before the H-4 correction; any change to the shared node
classes that alters this legacy wiring fails here.

Recorded with the T1 seams (2x nearest-neighbour upscale) on
fixtures/images/cmk_s1_input.png.  File names carry a date folder, so only
the counter suffix and pixels are compared.
"""

from __future__ import annotations

from harness.support import ScenarioCase, gate_of

CMK_TEST_TIER = "T1"
FIXTURE = "upscale_save_legacy_fullflow"
INSTANCE = "8444"

INPUT_HASH = "1c6a37a62460f893802d57f1703c8e10ea339c4286aa55df4ab974ea209a608f"
UPSCALED_HASH = "e0af3743fab63c1eba3fb17b23730cb56014f775c02057dec48d52ae82ef0c0f"
UPSCALED_SHAPE = [1, 1664, 2304, 3]
INPUT_SHAPE = [1, 832, 1152, 3]
SAVED_PIXELS = "95601310177c9b573cd584493b5ee0ae4fb93b3e63467635eeb13ee178a62606"
BEFORE_PIXELS = "d6ae4c3d6e2c25663cae2587768edccbf319b977f633376e39ad3a1a380c5443"
PROVIDER = {
    "provider_id": "cmk-CMK-SmartUpscalerPipe-8444-8370",
    "status": "completed",
    "module_type": "CMK_SmartUpscalerPipe",
    "channels": ["before", "after"],
    "channel_hashes": {"before": BEFORE_PIXELS, "after": SAVED_PIXELS},
}
VISUAL = {"8418": [PROVIDER], f"{INSTANCE}:8374": [PROVIDER]}
FULL_RUN_CALLS = {
    "CMKImageLoadAndResizePipe": 1, "CMK_SmartUpscalerPipe": 1, "CMK_SaveProjectImage": 1,
    "CMKUpscaleSaveBoundaryCache": 1, "CMKModuleBypassGate": 1, "CMKVisualizer": 2, "CMKVisualCompare": 1,
}
FULL_RUN_SEAMS = {"upscale.load_model": 1, "upscale.run": 1}


def _counters(run):
    return [name.rsplit("_", 1)[-1] for name in run["new_file_hashes"]]


class LegacyFullFlowReference(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        inspect = {"gate": f"{gate_of(FIXTURE)}:1", "loader": "1:2"}
        common = dict(fixture=FIXTURE, inspect=inspect)
        cls.classic = cls.run_case("run_fixture", runs=2, **common)
        cls.evict = cls.run_case("run_fixture", runs=2, mode="ram_pressure", evict_between_runs=True, **common)
        workdir = cls.workdir("restart")
        first = cls.run_case("run_fixture", workdir=workdir, **common)
        second = cls.run_case("run_fixture", workdir=workdir, **common)
        cls.restart = None if first is None else first["runs"] + second["runs"]
        cls.bypass = cls.run_case("run_fixture", widget_overrides={f"{INSTANCE}.SAVE ENABLED": False}, **common)

    def _assert_full_run(self, run, statuses, counter):
        self.assertTrue(run["succeeded"], run["errors"])
        self.assertEqual([s[1] for s in run["statuses"]], statuses)
        self.assertEqual(run["function_calls"], FULL_RUN_CALLS)
        self.assertEqual(run["seams"]["substituted_calls"], FULL_RUN_SEAMS)
        self.assertEqual(run["seams"]["watched_calls"], {"lifecycle.unload_all_models": 1})
        self.assertEqual(run["cached"], [])
        self.assertEqual(run["outputs"]["gate"], {"hash": UPSCALED_HASH, "shape": UPSCALED_SHAPE})
        self.assertEqual(run["outputs"]["loader"], {"hash": INPUT_HASH, "shape": INPUT_SHAPE})
        self.assertEqual(list(run["new_file_hashes"].values()), [SAVED_PIXELS])
        self.assertEqual(list(run["new_files"].values()), [[2304, 1664]])
        self.assertEqual(_counters(run), [counter])
        self.assertEqual(run["visual"], VISUAL)

    def test_cold_run(self):
        self._assert_full_run(self.classic["runs"][0], ["MISS_STORED"], "00001.png")

    def test_warm_run_reuses_everything_and_saves_nothing(self):
        warm = self.classic["runs"][1]
        self.assertEqual(warm["function_calls"], {"CMKVisualizer": 2, "CMKVisualCompare": 1})
        self.assertEqual(warm["seams"]["substituted_calls"], {})
        self.assertEqual(warm["seams"]["watched_calls"], {})
        self.assertEqual(len(warm["cached"]), 5)
        self.assertEqual(warm["statuses"], [])
        self.assertEqual(warm["new_file_hashes"], {})
        self.assertEqual(warm["outputs"]["gate"], {"hash": UPSCALED_HASH, "shape": UPSCALED_SHAPE})
        self.assertEqual(warm["visual"], VISUAL)

    def test_eviction_hit_still_upscales_and_saves(self):
        self._assert_full_run(self.evict["runs"][0], ["MISS_STORED"], "00001.png")
        self._assert_full_run(self.evict["runs"][1], ["HIT_READY", "HIT"], "00002.png")

    def test_restart_hit_still_upscales_and_saves(self):
        self._assert_full_run(self.restart[0], ["MISS_STORED"], "00001.png")
        self._assert_full_run(self.restart[1], ["HIT_READY", "HIT"], "00002.png")

    def test_bypass(self):
        run = self.bypass["runs"][0]
        self.assertTrue(run["succeeded"], run["errors"])
        self.assertEqual(run["function_calls"], {
            "CMKImageLoadAndResizePipe": 1, "CMK_SmartUpscalerPipe": 1, "CMK_SaveProjectImage": 1,
            "CMKModuleBypassGate": 1, "CMKVisualizer": 1, "CMKVisualCompare": 1,
        })
        self.assertEqual(run["seams"]["substituted_calls"], {})
        self.assertEqual(run["seams"]["watched_calls"], {})
        self.assertEqual(run["statuses"], [])
        self.assertEqual(run["new_file_hashes"], {})
        self.assertEqual(run["outputs"]["gate"], {"hash": INPUT_HASH, "shape": INPUT_SHAPE})
        self.assertEqual(run["visual"], {"8418": [], f"{INSTANCE}:8374": []})
