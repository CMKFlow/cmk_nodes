"""FaceProcess 2.8 F2 with the real PromptExecutor: deferred Prepare, lazy
model sources, NONE bypass, RESTORE without diffusion model, branch and
boundary HIT/MISS.

Model-free seams (fake checkpoint, SAM, CLIP encode, FaceProcess compute and
Z-Image Turbo loaders); the materialisation calls are counted.  Each sequence
runs in its own work directory with its own persistent caches.
"""

from __future__ import annotations

import unittest

from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
PREPARE, EXECUTE, BOUNDARY = "8417:7845", "8417:7842", "8417:7844"
WORK = {"CMKFaceProcessPrepareMultiModelPipe", "CMKFaceProcessMultiModelPipe", "CMKFaceBoundaryCache",
        "CMKFaceProcessPreparePipe", "CMKFaceProcessPipe"}


def calls(run, name):
    return run["function_calls"].get(name, 0)


def seam(run, name):
    seams = run["seams"]
    return seams["original_calls"].get(name, 0) + seams["substituted_calls"].get(name, 0)


def prepare_requests(run):
    requested = run["lazy_requests"].get("CMKFaceProcessPrepareMultiModelPipe", [])
    return sorted({name for _uid, names in requested for name in names})


def compute(run):
    """FaceProcess compute runs of the 2.8 branch (one YOLO detection per run);
    the legacy compute path is never used."""
    return seam(run, "faceprocess.run") + seam(run, "branch.yolo")


def statuses(run, scope):
    return [status for s, status, _key in run["statuses"] if s == scope]


def loads(run):
    return {"sdxl_resolve": seam(run, "prepare.resolve_postprocess_model"),
            "checkpoint": seam(run, "checkpoint.load_resources"), "sam": seam(run, "prepare.sam_load"),
            "encode": seam(run, "prepare.clip_encode"), "zit": len(run["zit_load"])}


NO_LOADS = {"sdxl_resolve": 0, "checkpoint": 0, "sam": 0, "encode": 0, "zit": 0}


class Modes(ScenarioCase):
    """Cold run and a restart (persistent caches kept) per mode."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.runs = {}
        for mode in ("None", "Restore", "SDXL", "Z-Image Turbo"):
            result = cls.run_case("faceprocess28_sequence", steps=[{"mode": mode}, {"mode": mode, "restart": True}])
            cls.runs[mode] = None if result is None else result["runs"]

    def test_all_runs_succeed(self):
        for mode, runs in self.runs.items():
            for index, run in enumerate(runs):
                with self.subTest(mode=mode, run=index):
                    self.assertTrue(run["succeeded"], run["errors"])
                    self.assertEqual(run["seams"]["tripped"], [])
                    # PROCESS MODE reaches the Prepare as literal selection; the ENABLE links stay links.
                    self.assertEqual(run["inlined"], 1)

    def test_none_is_the_complete_bypass(self):
        for run in self.runs["None"]:
            self.assertEqual({name for name in run["function_calls"] if name in WORK}, set(), "no module node runs")
            self.assertEqual(loads(run), NO_LOADS)
            self.assertEqual(compute(run), 0, "no detection, restore, detailer or refine")
            self.assertEqual(run["branch_revision"], "disabled")
            self.assertIsNotNone(run["outputs"]["image"], "the gate passes the module input on")

    def test_restore_requests_and_loads_no_diffusion_model(self):
        cold = self.runs["Restore"][0]
        self.assertEqual(prepare_requests(cold), ["IMAGE", "LOG"], "no MODEL ZIT, MODEL SDXL or PROCESS")
        self.assertEqual(loads(cold), NO_LOADS, "no SDXL, ZIT, SAM or conditioning")
        self.assertEqual(seam(cold, "faceprocess.run"), 0, "RESTORE no longer runs the legacy path")
        self.assertEqual((seam(cold, "branch.yolo"), seam(cold, "restore.insightface"), seam(cold, "restore.backend")), (1, 1, 1))
        self.assertEqual(seam(cold, "detailer.segs"), 0)
        self.assertEqual(statuses(cold, "faceprocess_branch")[-1], "MISS_STORED")
        self.assertNotEqual(cold["outputs"]["image"], self.runs["None"][0]["outputs"]["image"], "the assigned face is restored")

    def test_sdxl_requests_and_materialises_only_sdxl(self):
        cold = self.runs["SDXL"][0]
        self.assertEqual(prepare_requests(cold), ["IMAGE", "LOG", "MODEL SDXL", "PROCESS"])
        self.assertEqual(loads(cold), {"sdxl_resolve": 1, "checkpoint": 1, "sam": 1, "encode": 2, "zit": 0})
        self.assertEqual(calls(cold, "CMKDetailerPrepareMultiModelPipe"), 1, "the Multi-Model Detailer preparation runs in the branch")
        self.assertEqual(calls(cold, "CMKFaceProcessPreparePipe"), 0)
        self.assertEqual((seam(cold, "faceprocess.run"), seam(cold, "branch.yolo"), seam(cold, "detailer.segs")), (0, 1, 1),
                         "SDXL runs the 2.8 detailer step (F4 layers), not the legacy compute")
        self.assertEqual(seam(cold, "restore.insightface"), 0, "no InsightFace in the detailer path")

    def test_zit_requests_and_materialises_only_zit(self):
        cold = self.runs["Z-Image Turbo"][0]
        self.assertEqual(prepare_requests(cold), ["IMAGE", "LOG", "MODEL ZIT", "PROCESS"])
        self.assertEqual(loads(cold), {"sdxl_resolve": 0, "checkpoint": 0, "sam": 1, "encode": 1, "zit": 3})
        self.assertEqual(calls(cold, "CMKDetailerPrepareMultiModelPipe"), 1)

    def test_prepare_itself_loads_nothing(self):
        # Every load happens inside the branch run: with the branch skipped
        # (restart -> boundary HIT) nothing is loaded although MODEL arrives.
        for mode in ("Restore", "SDXL", "Z-Image Turbo"):
            warm = self.runs[mode][1]
            with self.subTest(mode=mode):
                self.assertEqual(statuses(warm, "faceprocess_boundary")[-1], "HIT")
                self.assertEqual(loads(warm), NO_LOADS)
                self.assertEqual(calls(warm, "CMKFaceProcessMultiModelPipe"), 0)
                self.assertEqual(calls(warm, "CMKFaceProcessPrepareMultiModelPipe"), 0)
                self.assertEqual(warm["outputs"]["image"], self.runs[mode][0]["outputs"]["image"])


class RestoreCaches(ScenarioCase):
    """One work directory: branch HIT without boundary, irrelevant settings,
    a relevant change and the switch to NONE."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        steps = [
            {"mode": "Restore"},
            {"mode": "Restore", "restart": True, "drop_scopes": ["faceprocess_boundary"]},
            {"mode": "Restore", "overrides": {f"{PREPARE}.zit_steps": 12, f"{PREPARE}.steps": 40,
                                               f"{PREPARE}.prompt_pos": "other", f"{PREPARE}.sampling_entry": 70.0,
                                               f"{PREPARE}.zit_sampling_entry": 60.0, f"{EXECUTE}.detail_paste_feather": 5}},
            {"mode": "Restore", "overrides": {f"{EXECUTE}.refine_mode": "Smooth"}},
            {"mode": "None", "overrides": {f"{EXECUTE}.refine_mode": "Smooth"}},
        ]
        result = cls.run_case("faceprocess28_sequence", steps=steps)
        cls.runs = None if result is None else result["runs"]

    def test_all_runs_succeed(self):
        for run in self.runs:
            self.assertTrue(run["succeeded"], run["errors"])

    def test_branch_hit_without_boundary_materialises_nothing(self):
        run = self.runs[1]
        self.assertEqual(statuses(run, "faceprocess_branch")[-1], "HIT")
        self.assertEqual(compute(run), 0)
        self.assertEqual(loads(run), NO_LOADS)
        self.assertEqual(statuses(run, "faceprocess_boundary")[-1], "MISS_STORED")
        self.assertEqual(run["branch_key"], self.runs[0]["branch_key"])
        self.assertEqual(run["outputs"]["image"], self.runs[0]["outputs"]["image"])

    def test_settings_of_other_modes_do_not_invalidate(self):
        run = self.runs[2]
        self.assertEqual(statuses(run, "faceprocess_boundary")[-1], "HIT", "SDXL/ZIT/detail settings are not RESTORE identity")
        self.assertEqual(compute(run), 0)

    def test_refine_change_recomputes_the_branch(self):
        run = self.runs[3]
        self.assertEqual(statuses(run, "faceprocess_branch")[-1], "MISS_STORED")
        self.assertNotEqual(run["branch_key"], self.runs[0]["branch_key"])
        self.assertEqual(compute(run), 1)
        self.assertEqual(loads(run), NO_LOADS)

    def test_none_bypasses_refine_too(self):
        run = self.runs[4]
        self.assertEqual(compute(run), 0)
        self.assertEqual({name for name in run["function_calls"] if name in WORK}, set())
        self.assertEqual(run["branch_revision"], "disabled")


class DiffusionCaches(ScenarioCase):
    """SDXL: settings of the other family stay out of the key; a branch HIT
    never materialises SDXL; a relevant change does."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        steps = [
            {"mode": "SDXL"},
            {"mode": "SDXL", "overrides": {f"{PREPARE}.zit_steps": 12, f"{EXECUTE}.restore_visibility": 0.5}},
            {"mode": "SDXL", "restart": True, "drop_scopes": ["faceprocess_boundary"]},
            {"mode": "SDXL", "overrides": {f"{PREPARE}.steps": 40}},
            {"mode": "Z-Image Turbo"},
            {"mode": "Z-Image Turbo", "overrides": {f"{PREPARE}.steps": 50, f"{PREPARE}.zit_steps": 8}},
        ]
        result = cls.run_case("faceprocess28_sequence", steps=steps)
        cls.runs = None if result is None else result["runs"]

    def test_all_runs_succeed(self):
        for run in self.runs:
            self.assertTrue(run["succeeded"], run["errors"])

    def test_zit_and_restore_settings_are_not_sdxl_identity(self):
        run = self.runs[1]
        self.assertEqual(statuses(run, "faceprocess_boundary")[-1], "HIT")
        self.assertEqual(loads(run), NO_LOADS)

    def test_branch_hit_does_not_materialise_sdxl(self):
        run = self.runs[2]
        self.assertEqual(statuses(run, "faceprocess_branch")[-1], "HIT")
        self.assertEqual(loads(run), NO_LOADS)

    def test_sdxl_setting_change_materialises_sdxl(self):
        run = self.runs[3]
        self.assertEqual(statuses(run, "faceprocess_branch")[-1], "MISS_STORED")
        self.assertEqual(loads(run)["sdxl_resolve"], 1)
        self.assertEqual(loads(run)["zit"], 0)

    def test_mode_switch_and_sdxl_settings_under_zit(self):
        self.assertEqual(loads(self.runs[4])["zit"], 3)
        self.assertNotEqual(self.runs[4]["branch_key"], self.runs[3]["branch_key"], "the mode is part of the key")
        run = self.runs[5]
        self.assertEqual(statuses(run, "faceprocess_boundary")[-1], "HIT", "SDXL settings are not ZIT identity")
        self.assertEqual(loads(run), NO_LOADS)


if __name__ == "__main__":
    unittest.main()
