"""FaceProcess 2.8 Advanced (F5) with the real PromptExecutor: resource order
RESTORE -> ZIT -> SDXL, exact layer composition D1 -> D2 -> D3, lazy
materialisation, branch and boundary HIT/MISS per branch.

Model-free seams as in test_s1_faceprocess28_f2.py; every sequence runs in
its own work directory with its own persistent caches.
"""

from __future__ import annotations

import unittest

from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
RZS = ("Restore", "Z-Image Turbo", "SDXL")
PREPARE = "8417:7845"


def statuses(run, scope):
    return [status for s, status, _key in run["statuses"] if s == scope and status in ("HIT", "MISS_STORED")]


def order(run):
    return [(event["slot"], event["mode"]) for event in run["branches"]]


def loads(run):
    seams = run["seams"]
    count = lambda name: seams["original_calls"].get(name, 0) + seams["substituted_calls"].get(name, 0)
    return {"checkpoint": count("checkpoint.load_resources"), "zit": len(run["zit_load"]), "sam": count("prepare.sam_load"),
            "restore": count("restore.backend")}


NO_LOADS = {"checkpoint": 0, "zit": 0, "sam": 0, "restore": 0}


class AdvancedSequence(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        steps = [
            {"modes": RZS},                                                      # 0 cold
            {"modes": ("SDXL", "Z-Image Turbo", "Restore"), "restart": True},     # 1 other slot positions
            {"modes": RZS, "restart": True},                                     # 2 back: branch HITs (the boundary keeps its latest entry)
            {"modes": RZS, "restart": True},                                     # 3 restart: boundary HIT
            {"modes": RZS, "drop_scopes": ["faceprocess_boundary"]},             # 4 branch HITs
            {"modes": RZS, "overrides": {f"{PREPARE}.zit_steps_2": 6}},          # 5 only D2
            {"modes": RZS, "overrides": {f"{PREPARE}.zit_steps_2": 6, f"{PREPARE}.refine_mode_1": "Detail"}},  # 6 only D1
            {"modes": ("None", "Restore", "None")},                              # 7
            {"modes": ("None", "None", "None")},                                 # 8 module off
            {"modes": RZS, "enable": "DISABLE"},                                 # 9 global DISABLE
        ]
        result = cls.run_case("faceprocess28_advanced_sequence", steps=steps)
        cls.runs = None if result is None else result["runs"]

    def test_all_runs_succeed(self):
        for index, run in enumerate(self.runs):
            with self.subTest(run=index):
                self.assertTrue(run["succeeded"], run["errors"])
                self.assertEqual(run["seams"]["tripped"], [])
                self.assertEqual(run["inlined"], 3, "each branch mode arrives as literal")

    def test_resource_order_restore_zit_sdxl(self):
        self.assertEqual(order(self.runs[0]), [(1, "restore"), (2, "z_image_turbo"), (3, "sdxl")])
        self.assertEqual(order(self.runs[1])[:3], [(3, "restore"), (2, "z_image_turbo"), (1, "sdxl")])
        for run in (self.runs[0], self.runs[1]):
            per_mode = {event["mode"]: event for event in run["branches"]}
            self.assertEqual((per_mode["restore"]["zit"], per_mode["restore"]["checkpoint"]), (0, 0), "RESTORE loads no model")
            self.assertEqual(per_mode["sdxl"]["zit"], 0)

    def test_composition_is_exact(self):
        for index, run in enumerate(self.runs):
            if "composition_deviation" in run:
                with self.subTest(run=index):
                    self.assertEqual(run["composition_deviation"], 0.0)

    def test_cold_run_materialises_per_branch(self):
        cold = self.runs[0]
        self.assertEqual(statuses(cold, "faceprocess_branch"), ["MISS_STORED"] * 3)
        self.assertEqual(statuses(cold, "faceprocess_boundary")[-1], "MISS_STORED")
        self.assertEqual(loads(cold), {"checkpoint": 1, "zit": 3, "sam": 2, "restore": 1})
        self.assertIsNotNone(cold["outputs"]["image"])
        self.assertNotEqual(cold["outputs"]["image"], cold["outputs"]["source"])
        self.assertEqual(cold["outputs"]["image"], cold["outputs"]["merge"])
        for index in range(4):
            self.assertIsNotNone(cold["outputs"][f"preview {index}"], "merge and slot previews render")

    def test_same_branch_settings_are_reused_across_positions(self):
        self.assertEqual(self.runs[1]["branch_keys"]["2"], self.runs[0]["branch_keys"]["2"])
        self.assertEqual(statuses(self.runs[1], "faceprocess_branch").count("HIT"), 1, "the ZIT branch D2")

    def test_switching_back_reuses_the_branches(self):
        run = self.runs[2]
        self.assertEqual(statuses(run, "faceprocess_branch"), ["HIT"] * 3)
        self.assertEqual(loads(run), NO_LOADS)
        self.assertEqual(run["outputs"]["image"], self.runs[0]["outputs"]["image"])

    def test_restart_boundary_hit_loads_nothing(self):
        run = self.runs[3]
        self.assertEqual(statuses(run, "faceprocess_boundary")[-1], "HIT")
        self.assertEqual(loads(run), NO_LOADS)
        self.assertEqual(run["outputs"]["image"], self.runs[0]["outputs"]["image"])

    def test_branch_hits_without_boundary(self):
        run = self.runs[4]
        self.assertEqual(statuses(run, "faceprocess_branch"), ["HIT"] * 3)
        self.assertEqual(statuses(run, "faceprocess_boundary")[-1], "MISS_STORED")
        self.assertEqual(loads(run), NO_LOADS)
        self.assertEqual(run["outputs"]["image"], self.runs[0]["outputs"]["image"])

    def test_a_branch_setting_invalidates_only_its_branch(self):
        zit, refine = self.runs[5], self.runs[6]
        self.assertEqual(statuses(zit, "faceprocess_branch"), ["HIT", "MISS_STORED", "HIT"])
        self.assertEqual(loads(zit), {"checkpoint": 0, "zit": 3, "sam": 1, "restore": 0})
        self.assertEqual(statuses(refine, "faceprocess_branch"), ["MISS_STORED", "HIT", "HIT"])
        self.assertEqual(loads(refine)["zit"] + loads(refine)["checkpoint"], 0, "Refine of a RESTORE branch loads no model")
        self.assertNotEqual(refine["outputs"]["image"], zit["outputs"]["image"])

    def test_none_branches_and_module_bypass(self):
        single = self.runs[7]
        self.assertEqual([event["slot"] for event in single["branches"]][0], 2, "the active branch first, NONE last")
        self.assertEqual(loads(single)["zit"] + loads(single)["checkpoint"], 0)
        for run in (self.runs[8], self.runs[9]):
            self.assertEqual(run["outputs"]["image"], run["outputs"]["source"], "module bypass passes the input on")
            self.assertEqual(loads(run), NO_LOADS)
            self.assertEqual(statuses(run, "faceprocess_branch"), [])


if __name__ == "__main__":
    unittest.main()
