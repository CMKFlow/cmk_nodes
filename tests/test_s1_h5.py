"""H-5 — FaceProcess boundary passthrough and diagnostic.

Two defects, both already solved for the Detailer by K-5:
1. The gate took DIAGNOSTIC ACTIVE from the last CMKFaceProcessPipe; its
   non-lazy opt_log / opt_diagnostic chain pulled every branch and Prepare
   (with resolve_postprocess_model) on a boundary HIT.
2. Global ON with every local branch OFF made the boundary return
   ExecutionBlocker for MODEL/IMAGE/LOG; gate, Visual Compare, Visualizer and
   all downstream modules were silently blocked (contrary to ARCHITECTURE.md
   "FaceProcess-Boundary: deaktivierter Modul-Passthrough").

CMK Face Boundary Cache now uses the shared _PassthroughSources mixin
(IMAGE IN / LOG IN / diagnostic, diagnostic output after VISUAL).  Keys and
manifests are unchanged (contracts/boundary_keys_pre_h5.json); embedded
pre-H-5 definitions (``*_legacy_pre_h5``) keep their behaviour.
Active-path matrix and toggles: test_s1_k5b.  MODEL-only: test_s1_k5_od2.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from harness.scenarios import inner_ids
from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
CONTRACTS = Path(__file__).resolve().parent / "contracts"
PINNED = json.loads((CONTRACTS / "boundary_keys_pre_h5.json").read_text(encoding="utf-8"))["fixtures"]
FIXTURES = ("faceprocess_std", "faceprocess_adv")
LEGACY = ("faceprocess_std_legacy_pre_h5", "faceprocess_adv_legacy_pre_h5")
SCOPE = "faceprocess_boundary"
RESOLVE = "prepare.resolve_postprocess_model"


def _gate(fixture):
    return inner_ids(fixture, "CMKModuleBypassGate")[0]


def _inspect(fixture):
    gate = _gate(fixture)
    return {"image": f"{gate}:1", "log": f"{gate}:2", "diagnostic": f"{gate}:3",
            "loader": "1:2", "loader_log": "1:3"}


def _local(fixture, active):
    return {f"{node}.enable": index in active
            for index, node in enumerate(inner_ids(fixture, "CMKFaceProcessPipe"), start=1)}


def _statuses(run):
    return [status for scope, status, _key in run["statuses"] if scope == SCOPE]


def _work(run):
    return {name: run["function_calls"][name] for name in ("CMKFaceProcessPreparePipe", "CMKFaceProcessPipe")
            if name in run["function_calls"]}


class BoundaryKeys(ScenarioCase):
    def test_keys_and_manifests_byte_identical(self):
        for fixture, pinned in PINNED.items():
            with self.subTest(fixture=fixture):
                result = self.run_case("boundary_key", fixture=fixture, boundary_class="CMKFaceBoundaryCache")
                self.assertEqual(result["key"], pinned["key"])
                self.assertEqual(result["dependencies"], pinned["deps"])


class Passthrough(ScenarioCase):
    """Global ON, all local branches OFF: unchanged image, input LOG plus a
    status block, lightweight diagnostic, unchanged VISUAL, nothing blocked."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in FIXTURES:
            result = cls.run_case("run_fixture", fixture=fixture, overrides=_local(fixture, ()),
                                  fake_postprocess=True, upstream_visual=True, inspect=_inspect(fixture))
            cls.r[fixture] = None if result is None else result["runs"][0]

    def test_no_prepare_compute_or_resolution(self):
        for fixture, run in self.r.items():
            with self.subTest(fixture=fixture):
                self.assertTrue(run["succeeded"], run["errors"])
                self.assertEqual(_work(run), {})
                self.assertEqual(run["seams"]["original_calls"], {})
                self.assertEqual(run["seams"]["substituted_calls"], {})

    def test_image_byte_identical_and_not_blocked(self):
        for fixture, run in self.r.items():
            self.assertEqual(run["outputs"]["image"], run["outputs"]["loader"], fixture)

    def test_log_keeps_input_and_adds_status_block(self):
        for fixture, run in self.r.items():
            with self.subTest(fixture=fixture):
                blocks = run["outputs"]["log"]["blocks"]
                self.assertEqual(blocks[:-1], run["outputs"]["loader_log"]["blocks"])
                self.assertEqual(blocks[-1][0], "FaceProcess Boundary")
                self.assertIn("STATUS          : DISABLED PASSTHROUGH", blocks[-1][1])

    def test_lightweight_diagnostic(self):
        for fixture, run in self.r.items():
            self.assertEqual(run["outputs"]["diagnostic"],
                             {"type": "diagnostic", "title": "FaceProcess Boundary", "mode": "disabled / passthrough"},
                             fixture)

    def test_visual_provider_kept_and_visualizer_runs(self):
        for fixture, run in self.r.items():
            with self.subTest(fixture=fixture):
                self.assertEqual([p["module_type"] for p in run["visual"]["8418"]], ["CMKVisualProvider"])
                self.assertIn("CMKVisualizer", run["function_calls"])

    def test_status(self):
        for fixture, run in self.r.items():
            self.assertEqual(_statuses(run)[-1], "DISABLED_PASSTHROUGH", fixture)
            self.assertNotIn("DISABLED_ACTIVE_BRANCH_BLOCKED", _statuses(run), fixture)


class GlobalOff(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in FIXTURES:
            result = cls.run_case("run_fixture", fixture=fixture, widget_overrides={"8417.face_global_enable": False},
                                  fake_postprocess=True, upstream_visual=True, inspect=_inspect(fixture))
            cls.r[fixture] = None if result is None else result["runs"][0]

    def test_bypass_without_module_work(self):
        for fixture, run in self.r.items():
            with self.subTest(fixture=fixture):
                self.assertTrue(run["succeeded"], run["errors"])
                self.assertEqual(_work(run), {})
                self.assertNotIn("CMKFaceBoundaryCache", run["function_calls"])
                self.assertEqual(run["outputs"]["image"], run["outputs"]["loader"])
                self.assertEqual([p["module_type"] for p in run["visual"]["8418"]], ["CMKVisualProvider"])
                self.assertIn("CMKVisualizer", run["function_calls"])


class ActiveSingleAndMultiple(ScenarioCase):
    """Cold MISS stores; HIT after eviction and after a restart restores IMAGE,
    LOG, diagnostic and VISUAL from the boundary without Prepare, branch nodes
    or model resolution."""

    CASES = {("faceprocess_std", "single"): (1,), ("faceprocess_adv", "single"): (2,),
             ("faceprocess_adv", "multiple"): (1, 2, 3)}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for (fixture, label), active in cls.CASES.items():
            common = dict(fixture=fixture, overrides=_local(fixture, active), fake_postprocess=True,
                          upstream_visual=True, inspect=_inspect(fixture))
            evict = cls.run_case("run_fixture", runs=2, mode="ram_pressure", evict_between_runs=True, **common)
            workdir = cls.workdir(f"restart_{fixture}_{label}")
            first = cls.run_case("run_fixture", workdir=workdir, **common)
            second = cls.run_case("run_fixture", workdir=workdir, **common)
            cls.r[(fixture, label, "evict")] = None if evict is None else evict["runs"]
            cls.r[(fixture, label, "restart")] = None if first is None else first["runs"] + second["runs"]

    def test_cold_miss_stores(self):
        for (fixture, label, variant), (cold, _hit) in self.r.items():
            with self.subTest(fixture=fixture, label=label, variant=variant):
                self.assertTrue(cold["succeeded"], cold["errors"])
                self.assertEqual(_statuses(cold), ["MISS_STORED"])
                self.assertEqual(cold["seams"]["original_calls"], {RESOLVE: 1})
                self.assertEqual(cold["seams"]["substituted_calls"].get("faceprocess.run"),
                                 len(self.CASES[(fixture, label)]))

    def test_hit_restores_everything_without_work(self):
        for (fixture, label, variant), (cold, hit) in self.r.items():
            with self.subTest(fixture=fixture, label=label, variant=variant):
                self.assertTrue(hit["succeeded"], hit["errors"])
                self.assertEqual(_statuses(hit)[-1], "HIT")
                self.assertEqual(_work(hit), {})
                self.assertEqual(hit["seams"]["original_calls"], {})
                self.assertEqual(hit["seams"]["substituted_calls"], {})
                self.assertEqual(hit["outputs"], cold["outputs"])
                self.assertEqual(hit["visual"], cold["visual"])
                self.assertNotEqual(hit["outputs"]["diagnostic"]["title"], "Module Result")


class LegacyPreH5(ScenarioCase):
    """Embedded pre-H-5 definitions keep their behaviour: blocked passthrough,
    HIT pulls Prepare and the branches through the compute diagnostic."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in LEGACY:
            single = (1,) if fixture.startswith("faceprocess_std") else (2,)
            off = cls.run_case("run_fixture", fixture=fixture, overrides=_local(fixture, ()),
                               fake_postprocess=True, inspect=_inspect(fixture))
            hit = cls.run_case("run_fixture", fixture=fixture, overrides=_local(fixture, single), runs=2,
                               mode="ram_pressure", evict_between_runs=True, fake_postprocess=True, inspect=_inspect(fixture))
            cls.r[fixture] = (None if off is None else off["runs"][0], None if hit is None else hit["runs"])

    def test_legacy_passthrough_still_blocks(self):
        for fixture, (off, _hit) in self.r.items():
            with self.subTest(fixture=fixture):
                self.assertTrue(off["succeeded"])
                self.assertIn("DISABLED_ACTIVE_BRANCH_BLOCKED", _statuses(off))
                self.assertEqual(off["outputs"]["image"], {"type": "ExecutionBlocker"})
                self.assertNotIn("CMKVisualizer", off["function_calls"])

    def test_legacy_hit_still_pulls_prepare(self):
        for fixture, (_off, (cold, hit)) in self.r.items():
            with self.subTest(fixture=fixture):
                self.assertEqual(_statuses(cold), ["MISS_STORED"])
                self.assertEqual(_statuses(hit)[-1], "HIT")
                self.assertEqual(hit["function_calls"].get("CMKFaceProcessPreparePipe"), 1)
                self.assertEqual(hit["seams"]["original_calls"], {RESOLVE: 1})


class EntryWithoutDiagnostic(ScenarioCase):
    """D2: an entry stored without a persisted diagnostic (pre-H-5 definition,
    same key) is invalidated once by the existing mixin check, then HITs."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.runs = None
        if cls.paths is None:
            return
        workdir = cls.workdir("migration")
        common = dict(overrides=_local("faceprocess_std", (1,)), fake_postprocess=True)
        legacy = cls.run_case("run_fixture", workdir=workdir, fixture="faceprocess_std_legacy_pre_h5", **common)["runs"][0]
        diagnostic_files = sorted(p.name for p in (Path(workdir) / "temp" / "cmk" / SCOPE).glob("*.diagnostic.*"))
        first = cls.run_case("run_fixture", workdir=workdir, fixture="faceprocess_std",
                             inspect=_inspect("faceprocess_std"), **common)["runs"][0]
        second = cls.run_case("run_fixture", workdir=workdir, fixture="faceprocess_std",
                              inspect=_inspect("faceprocess_std"), **common)["runs"][0]
        cls.runs = (legacy, diagnostic_files, first, second)

    def test_one_time_invalidation_then_hit(self):
        legacy, diagnostic_files, first, second = self.runs
        self.assertEqual(_statuses(legacy), ["MISS_STORED"])
        self.assertEqual(diagnostic_files, [])
        self.assertEqual(_statuses(first)[-1], "MISS_STORED")
        self.assertEqual(first["seams"]["original_calls"], {RESOLVE: 1})
        self.assertEqual(_statuses(second)[-1], "HIT")
        self.assertEqual(_work(second), {})
        self.assertEqual(second["outputs"], first["outputs"])


if __name__ == "__main__":
    unittest.main()
