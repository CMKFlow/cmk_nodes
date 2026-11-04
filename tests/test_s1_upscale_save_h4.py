"""H-4 — CMK Flow · Upscale & Save: IMAGE pass-through, upscaling and saving
as three separate responsibilities (canonical definition 1.4.0, fixture
``upscale_save``).

Before the fix two independent edges pulled the upscaler although the
boundary was a HIT: the OUTPUT_NODE ``CMK_SaveProjectImage`` in front of the
boundary (non-lazy inputs) and the gate's ``VISUAL ACTIVE`` taken from the
upscaler.  The tests below are the regression guard for both; the frozen
pre-fix definition is pinned in test_s1_upscale_save_legacy_reference.py.
"""

from __future__ import annotations

import unittest

from harness.scenarios import inner_ids, load_api_prompt
from harness.support import ScenarioCase, gate_of

CMK_TEST_TIER = "T1"
FIXTURE = "upscale_save"
UPSCALER, BOUNDARY, GATE = "8417:8370", "8417:8372", gate_of(FIXTURE)
INPUT_HASH = "1c6a37a62460f893802d57f1703c8e10ea339c4286aa55df4ab974ea209a608f"
INPUT_SHAPE = [1, 832, 1152, 3]
UPSCALED_PIXELS = "95601310177c9b573cd584493b5ee0ae4fb93b3e63467635eeb13ee178a62606"
BEFORE_PIXELS = "d6ae4c3d6e2c25663cae2587768edccbf319b977f633376e39ad3a1a380c5443"
PERSISTENT = ("evict", "restart")


def _counters(run):
    return [name.rsplit("_", 1)[-1] for name in run["new_file_hashes"]]


class UpscaleSaveResponsibilities(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        inspect = {"gate": f"{GATE}:1", "loader": "1:2"}
        common = dict(fixture=FIXTURE, inspect=inspect)
        cls.classic = cls.run_case("run_fixture", runs=2, **common)
        cls.persistent = {
            "evict": cls.run_case("run_fixture", runs=2, mode="ram_pressure", evict_between_runs=True, **common),
        }
        workdir = cls.workdir("restart")
        first = cls.run_case("run_fixture", workdir=workdir, **common)
        second = cls.run_case("run_fixture", workdir=workdir, **common)
        cls.persistent["restart"] = None if first is None else {"runs": first["runs"] + second["runs"]}
        cls.bypass = cls.run_case("run_fixture", runs=2, widget_overrides={"8417.SAVE ENABLED": False}, **common)
        cls.local_off = cls.run_case("run_fixture", overrides={f"{UPSCALER}.enable": False}, **common)
        cls.endpoint = inner_ids(FIXTURE, "CMKSaveEndpoint")[0]
        cls.provider = inner_ids(FIXTURE, "CMKVisualProvider")[0]

    def _hit(self, variant):
        return self.persistent[variant]["runs"][1]

    # graph: no edge from the save job or the upscaler into the result path --
    def test_save_endpoint_is_a_leaf_and_visual_follows_the_boundary(self):
        prompt = load_api_prompt(FIXTURE)
        self.assertNotIn("CMK_SaveProjectImage", {node["class_type"] for node in prompt.values()})
        consumed = {value[0] for node in prompt.values() for value in node["inputs"].values()
                    if isinstance(value, list) and len(value) == 2}
        self.assertNotIn(self.endpoint, consumed)
        self.assertEqual(prompt[self.endpoint]["inputs"]["IMAGE"], [BOUNDARY, 1])
        self.assertEqual(prompt[BOUNDARY]["inputs"]["IMAGE"], [UPSCALER, 0])
        self.assertEqual(prompt[GATE]["inputs"]["IMAGE ACTIVE"], prompt[GATE]["inputs"]["IMAGE BYPASS"])
        self.assertEqual(prompt[GATE]["inputs"]["VISUAL ACTIVE"], [self.provider, 0])
        self.assertEqual(prompt[self.provider]["inputs"]["IMAGE"], [BOUNDARY, 1])

    # boundary HIT: no upscale, no unload, but every executed save writes ------
    def test_hit_runs_no_upscale_and_no_unload(self):
        for variant in PERSISTENT:
            with self.subTest(variant=variant):
                hit = self._hit(variant)
                self.assertEqual([s[1] for s in hit["statuses"]][-1], "HIT")
                self.assertNotIn("CMK_SmartUpscalerPipe", hit["function_calls"])
                self.assertEqual(hit["seams"]["substituted_calls"], {})
                self.assertEqual(hit["seams"]["watched_calls"], {})

    def test_every_executed_save_after_hit_writes_one_new_file_of_the_boundary_result(self):
        for variant in PERSISTENT:
            with self.subTest(variant=variant):
                cold, hit = self.persistent[variant]["runs"]
                self.assertEqual(_counters(cold), ["00001.png"])
                self.assertEqual(_counters(hit), ["00002.png"])
                self.assertEqual(list(hit["new_file_hashes"].values()), [UPSCALED_PIXELS])
                self.assertEqual(hit["visual"]["8418"][0]["channel_hashes"]["result"], UPSCALED_PIXELS)

    def test_warm_run_writes_exactly_one_new_file(self):
        warm = self.classic["runs"][1]
        self.assertEqual(_counters(warm), ["00002.png"])
        self.assertEqual(list(warm["new_file_hashes"].values()), [UPSCALED_PIXELS])

    # bypass: the endpoint is entered once as a no-op and requests nothing ----
    def test_bypass_runs_no_upscaler_boundary_or_save(self):
        for index, run in enumerate(self.bypass["runs"]):
            with self.subTest(run=index):
                self.assertTrue(run["succeeded"], run["errors"])
                for name in ("CMK_SmartUpscalerPipe", "CMKUpscaleSaveBoundaryCache", "CMK_SaveProjectImage",
                             "CMKVisualProvider"):
                    self.assertNotIn(name, run["function_calls"])
                self.assertEqual(run["new_file_hashes"], {})
                self.assertEqual(run["statuses"], [])
                self.assertEqual(run["seams"]["substituted_calls"], {})
                self.assertEqual(run["seams"]["watched_calls"], {})
                self.assertEqual(run["outputs"]["gate"], {"hash": INPUT_HASH, "shape": INPUT_SHAPE})

    def test_bypass_endpoint_requests_no_input_and_is_cached_afterwards(self):
        first, second = self.bypass["runs"]
        self.assertEqual(first["function_calls"].get("CMKSaveEndpoint"), 1)
        self.assertEqual([item[1] for item in first["lazy_requests"]["CMKSaveEndpoint"]], [[]])
        self.assertNotIn("CMKSaveEndpoint", second["function_calls"])
        self.assertIn(self.endpoint, second["cached"])

    # IMAGE OUT == IMAGE IN ---------------------------------------------------
    def test_image_out_equals_in_on_miss_hit_warm_and_local_upscaler_off(self):
        runs = (*self.classic["runs"], self._hit("evict"), self._hit("restart"), self.local_off["runs"][0])
        for index, run in enumerate(runs):
            with self.subTest(run=index):
                self.assertTrue(run["succeeded"], run["errors"])
                self.assertEqual(run["outputs"]["gate"], {"hash": INPUT_HASH, "shape": INPUT_SHAPE})
        self.assertEqual(self.local_off["runs"][0]["seams"]["substituted_calls"], {})

    # the endpoint's per-prompt IS_CHANGED invalidates nothing else -----------
    def test_warm_run_reuses_all_processing_nodes(self):
        warm = self.classic["runs"][1]
        self.assertEqual(set(warm["function_calls"]),
                         {"CMKVisualizer", "CMKVisualCompare", "CMKSaveEndpoint", "CMK_SaveProjectImage"})
        self.assertEqual(warm["function_calls"]["CMKSaveEndpoint"], 1)
        self.assertLessEqual({"1", UPSCALER, BOUNDARY, GATE, self.provider}, set(warm["cached"]))
        self.assertNotIn(self.endpoint, warm["cached"])
        self.assertEqual(warm["seams"]["substituted_calls"], {})
        self.assertEqual(warm["seams"]["watched_calls"], {})

    # VISUAL shows the before/result pair --------------------------------------
    def test_visual_shows_one_before_result_pair(self):
        for run in (self.classic["runs"][0], self.classic["runs"][1], self._hit("evict"), self._hit("restart")):
            for node in ("8418", "8417:8374"):
                with self.subTest(node=node):
                    providers = run["visual"][node]
                    self.assertEqual(len(providers), 1)
                    self.assertEqual(providers[0]["module_type"], "CMKVisualProvider")
                    self.assertEqual(providers[0]["status"], "completed")
                    self.assertEqual(providers[0]["channel_hashes"], {"result": UPSCALED_PIXELS, "before": BEFORE_PIXELS})


if __name__ == "__main__":
    unittest.main()
