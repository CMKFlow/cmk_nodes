"""K5-B — persistent Detailer / FaceProcess boundaries.

The boundary dependency manifest must look up each branch revision under
exactly the key the branch node publishes it.  Before K5-B the manifest used
its own schema constants (detailer v6, faceprocess v4) without the branch
input exclusions, so the manifest was never complete and the boundaries never
stored.

Since K5-B the manifest takes scope and key from the branch class
(``dependency_cache_key``); the Detailer HIT takes MODEL from the current input
instead of a session copy.  The dynamic tests use the model-free seams
(``fake_postprocess``: E9, D5).

K5-C: a branch switched off in the prompt (``enable`` literally False) reports
the manifest revision "disabled" through ``dependency_revision`` of its branch
class; Advanced modules with mixed branch settings now store and HIT.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from harness.scenarios import inner_ids
from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
CONTRACTS = Path(__file__).resolve().parent / "contracts"
PINNED = json.loads((CONTRACTS / "branch_keys_pre_k5b.json").read_text(encoding="utf-8"))["fixtures"]
MANIFEST_FIXTURES = ("detailer_std", "detailer_adv", "faceprocess_std", "faceprocess_adv",
                     "chain_postprocess", "detailer_std_legacy_pre_k5", "detailer_adv_legacy_pre_k5")


class ManifestKeys(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.rows = []
        for fixture in MANIFEST_FIXTURES:
            result = cls.run_case("manifest_keys", fixture=fixture)
            cls.rows += [] if result is None else [{**row, "fixture": fixture} for row in result["rows"]]

    def test_every_manifest_fixture_has_branches(self):
        self.assertEqual({row["fixture"] for row in self.rows}, set(MANIFEST_FIXTURES))
        self.assertEqual({row["class_type"] for row in self.rows}, {"CMK_SmartDetailerPipe", "CMKFaceProcessPipe"})

    def test_manifest_uses_the_branch_scope(self):
        for row in self.rows:
            self.assertEqual(row["scope"], row["branch_scope"], row)

    def test_manifest_key_equals_published_branch_key(self):
        for row in self.rows:
            with self.subTest(fixture=row["fixture"], node=row["node_id"]):
                self.assertEqual(row["manifest_key"], row["published_key"])

    def test_published_branch_keys_unchanged(self):
        # Recorded at 92b7a67 before K5-B; branch caches stay valid.
        self.assertEqual(sum(len(nodes) for nodes in PINNED.values()), len(self.rows))
        for row in self.rows:
            pinned = PINNED[row["fixture"]][row["node_id"]]
            self.assertEqual((row["class_type"], row["published_key"]), (pinned["class_type"], pinned["published_key"]), row)

    def test_baseline_pre_k5b_manifest_keys_differed(self):
        for row in self.rows:
            self.assertNotEqual(PINNED[row["fixture"]][row["node_id"]]["manifest_key_pre_k5b"], row["published_key"], row)


def _statuses(run, scope):
    return [status for s, status, _key in run["statuses"] if s == scope]


def _inspect(fixture, gate_class="CMKModuleBypassGate"):
    gate = inner_ids(fixture, gate_class)[0]
    return {"image": f"{gate}:1", "log": f"{gate}:2", "diagnostic": f"{gate}:3"}


class BranchRevisionInvalidates(ScenarioCase):
    """The manifest's purpose: the boundary entry is reused only while every
    branch revision equals the stored one.

    FaceProcess: a recomputed branch (pickle and revision removed) writes a
    new revision token -> STALE_DEPENDENCIES -> MISS_STORED.
    Detailer: the branch revision is deterministic (its artifact key); a
    tampered revision is detected (STALE_DEPENDENCIES), the branch HIT
    re-publishes the original revision and the boundary is reused unchanged."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.runs = {}
        if cls.paths is None:
            return
        for fixture, scope, action in (("faceprocess_std", "faceprocess_branch", "recompute"),
                                       ("detailer_std", "detailer_branch", "tamper")):
            common = dict(fixture=fixture, fake_postprocess=True, upstream_visual=True, inspect=_inspect(fixture))
            workdir = cls.workdir(f"revision_{fixture}")
            cold = cls.run_case("run_fixture", workdir=workdir, **common)["runs"][0]
            key = next(k for sc, status, k in cold["statuses"] if sc == scope and status == "MISS_STORED")
            branch_dir = Path(workdir) / "temp" / "cmk" / scope
            revision = branch_dir / f"{key}.rev"
            found = revision.is_file()
            if action == "recompute":
                for path in branch_dir.glob(f"{key}.*"):
                    path.unlink()
            else:
                revision.write_text("cmk-s1-k5b-tampered", encoding="utf-8")
            again = cls.run_case("run_fixture", workdir=workdir, **common)["runs"][0]
            third = cls.run_case("run_fixture", workdir=workdir, **common)["runs"][0]
            cls.runs[fixture] = (found, cold, again, third)

    def test_revision_file_is_the_runtime_branch_key(self):
        for fixture, (found, *_runs) in self.runs.items():
            self.assertTrue(found, fixture)

    def test_recomputed_faceprocess_branch_invalidates_the_boundary_once(self):
        _found, cold, again, third = self.runs["faceprocess_std"]
        statuses = _statuses(again, "faceprocess_boundary")
        self.assertEqual(_statuses(cold, "faceprocess_boundary"), ["MISS_STORED"])
        self.assertIn("STALE_DEPENDENCIES", statuses)
        self.assertEqual(statuses[-1], "MISS_STORED")
        self.assertEqual(again["seams"]["substituted_calls"].get("faceprocess.run"), 1)
        self.assertEqual(_statuses(third, "faceprocess_boundary")[-1], "HIT")
        self.assertEqual(again["outputs"]["image"], cold["outputs"]["image"])

    def test_tampered_detailer_revision_is_detected_and_restored(self):
        _found, cold, again, third = self.runs["detailer_std"]
        statuses = _statuses(again, "detailer_boundary")
        self.assertIn("STALE_DEPENDENCIES", statuses)
        self.assertEqual(statuses[-1], "HIT")
        self.assertNotIn("detailer.run", again["seams"]["substituted_calls"])
        self.assertEqual(_statuses(third, "detailer_boundary")[-1], "HIT")
        self.assertEqual(again["outputs"], cold["outputs"])


class FaceProcessPersistence(ScenarioCase):
    """FaceProcess boundary stores and HITs after eviction and restart.  Since
    H-5 the gate takes DIAGNOSTIC ACTIVE from the boundary, so a HIT runs no
    Prepare, no branch node and no model resolution.  Advanced: see
    AdvancedMatrix."""

    FIXTURES = ("faceprocess_std",)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in cls.FIXTURES:
            common = dict(fixture=fixture, fake_postprocess=True, upstream_visual=True, inspect=_inspect(fixture))
            evict = cls.run_case("run_fixture", runs=2, mode="ram_pressure", evict_between_runs=True, **common)
            workdir = cls.workdir(f"restart_{fixture}")
            first = cls.run_case("run_fixture", workdir=workdir, **common)
            second = cls.run_case("run_fixture", workdir=workdir, **common)
            cls.r[(fixture, "evict")] = None if evict is None else evict["runs"]
            cls.r[(fixture, "restart")] = None if first is None else first["runs"] + second["runs"]

    def test_miss_computes_and_stores(self):
        for (fixture, variant), (cold, _hit) in self.r.items():
            with self.subTest(fixture=fixture, variant=variant):
                self.assertTrue(cold["succeeded"], cold["errors"])
                self.assertEqual(_statuses(cold, "faceprocess_boundary"), ["MISS_STORED"])
                self.assertEqual(cold["function_calls"].get("CMKFaceProcessPreparePipe"), 1)
                self.assertEqual(cold["seams"]["substituted_calls"].get("faceprocess.run"), 1)

    def test_hit_after_eviction_and_restart_reuses_the_boundary_result(self):
        for (fixture, variant), (cold, hit) in self.r.items():
            with self.subTest(fixture=fixture, variant=variant):
                self.assertTrue(hit["succeeded"], hit["errors"])
                self.assertEqual(_statuses(hit, "faceprocess_boundary")[-1], "HIT")
                self.assertNotIn("faceprocess.run", hit["seams"]["substituted_calls"])
                self.assertEqual(hit["outputs"]["image"], cold["outputs"]["image"])
                self.assertEqual(hit["outputs"]["log"], cold["outputs"]["log"])
                self.assertEqual(hit["visual"], cold["visual"])

    def test_hit_runs_no_prepare_compute_or_resolution(self):
        for (fixture, variant), (cold, hit) in self.r.items():
            with self.subTest(fixture=fixture, variant=variant):
                self.assertNotIn("CMKFaceProcessPreparePipe", hit["function_calls"])
                self.assertNotIn("CMKFaceProcessPipe", hit["function_calls"])
                self.assertEqual(hit["seams"]["original_calls"], {})
                self.assertEqual(hit["outputs"]["diagnostic"], cold["outputs"]["diagnostic"])


ADVANCED = {
    # fixture: (branch class, boundary scope, prepare class, engine seam)
    "detailer_adv": ("CMK_SmartDetailerPipe", "detailer_boundary", "CMKDetailerPreparePipe", "detailer.run"),
    "faceprocess_adv": ("CMKFaceProcessPipe", "faceprocess_boundary", "CMKFaceProcessPreparePipe", "faceprocess.run"),
}
SETTINGS = {"all_on": (1, 2, 3), "only_1": (1,), "only_2": (2,), "only_3": (3,), "two_1_3": (1, 3)}


def _enable(fixture, active):
    nodes = inner_ids(fixture, ADVANCED[fixture][0])
    return {f"{node}.enable": index in active for index, node in enumerate(nodes, start=1)}


class AdvancedManifestContract(ScenarioCase):
    """K5-C static contract: a branch switched off in the prompt reports the
    marker "disabled"; an active branch reports its published revision (none
    before the first run => manifest incomplete).  Keys stay byte-identical."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in ADVANCED:
            for setting, active in {**SETTINGS, "all_off": ()}.items():
                result = cls.run_case("manifest_keys", fixture=fixture, overrides=_enable(fixture, active))
                cls.r[(fixture, setting)] = None if result is None else result["rows"]

    def test_disabled_marker_and_missing_active_revision(self):
        for (fixture, setting), rows in self.r.items():
            with self.subTest(fixture=fixture, setting=setting):
                for row in rows:
                    self.assertEqual(row["revision"], "disabled" if row["enable"] is False else None, row)
                any_active = any(row["enable"] is not False for row in rows)
                self.assertEqual(rows[0]["complete"], not any_active)

    def test_branch_keys_unchanged(self):
        # Default setting (only branch 2) equals the pre-K5-B record byte for byte.
        for fixture in ADVANCED:
            for row in self.r[(fixture, "only_2")]:
                self.assertEqual(row["published_key"], PINNED[fixture][row["node_id"]]["published_key"])
                self.assertEqual(row["manifest_key"], row["published_key"])

    def test_enable_toggle_changes_only_the_toggled_branch_key(self):
        for fixture in ADVANCED:
            default = {row["node_id"]: row["published_key"] for row in self.r[(fixture, "only_2")]}
            every = {row["node_id"]: row["published_key"] for row in self.r[(fixture, "all_on")]}
            first, second, third = inner_ids(fixture, ADVANCED[fixture][0])
            self.assertNotEqual(default[first], every[first], fixture)
            self.assertEqual(default[second], every[second], fixture)
            self.assertNotEqual(default[third], every[third], fixture)

    def test_boundary_key_identity(self):
        # Detailer: the boundary fingerprint contains the branch widgets; FaceProcess
        # excludes MODEL/IMAGE/LOG, so its invalidation comes from the manifest alone.
        detailer = {self.r[("detailer_adv", s)][0]["boundary_key"] for s in SETTINGS}
        faceprocess = {self.r[("faceprocess_adv", s)][0]["boundary_key"] for s in SETTINGS}
        self.assertEqual(len(detailer), len(SETTINGS))
        self.assertEqual(len(faceprocess), 1)


class AdvancedMatrix(ScenarioCase):
    """K5-C runtime matrix (model-free seams): cold MISS stores, the boundary
    HITs after eviction and after a restart with kept cache files, results are
    identical.  HIT: no Prepare, no branch node, no model resolution (FaceProcess
    since H-5: DIAGNOSTIC ACTIVE comes from the boundary)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in ADVANCED:
            for setting, active in SETTINGS.items():
                common = dict(fixture=fixture, fake_postprocess=True, upstream_visual=True,
                              overrides=_enable(fixture, active), inspect=_inspect(fixture))
                evict = cls.run_case("run_fixture", runs=2, mode="ram_pressure", evict_between_runs=True, **common)
                workdir = cls.workdir(f"restart_{fixture}_{setting}")
                first = cls.run_case("run_fixture", workdir=workdir, **common)
                second = cls.run_case("run_fixture", workdir=workdir, **common)
                cls.r[(fixture, setting, "evict")] = None if evict is None else evict["runs"]
                cls.r[(fixture, setting, "restart")] = None if first is None else first["runs"] + second["runs"]

    def _cases(self):
        for (fixture, setting, variant), runs in self.r.items():
            yield fixture, setting, variant, runs[0], runs[1]

    def test_cold_run_computes_active_branches_and_stores(self):
        for fixture, setting, variant, cold, _hit in self._cases():
            with self.subTest(fixture=fixture, setting=setting, variant=variant):
                _branch, scope, prepare, engine = ADVANCED[fixture]
                self.assertTrue(cold["succeeded"], cold["errors"])
                self.assertEqual(_statuses(cold, scope), ["MISS_STORED"])
                self.assertEqual(cold["function_calls"].get(prepare), 1)
                self.assertEqual(cold["seams"]["substituted_calls"].get(engine), len(SETTINGS[setting]))

    def test_hit_after_eviction_and_restart(self):
        for fixture, setting, variant, cold, hit in self._cases():
            with self.subTest(fixture=fixture, setting=setting, variant=variant):
                _branch, scope, _prepare, engine = ADVANCED[fixture]
                self.assertTrue(hit["succeeded"], hit["errors"])
                self.assertEqual(_statuses(hit, scope)[-1], "HIT")
                self.assertNotIn(engine, hit["seams"]["substituted_calls"])
                self.assertEqual(hit["outputs"], cold["outputs"])
                self.assertEqual(hit["visual"], cold["visual"])

    def test_hit_runs_no_prepare_compute_or_resolution(self):
        for fixture, setting, variant, _cold, hit in self._cases():
            branch, _scope, prepare, _engine = ADVANCED[fixture]
            with self.subTest(fixture=fixture, setting=setting, variant=variant):
                self.assertNotIn(prepare, hit["function_calls"])
                self.assertNotIn(branch, hit["function_calls"])
                self.assertEqual(hit["seams"]["original_calls"], {})


class AdvancedAllOff(ScenarioCase):
    """All local branches OFF: disabled passthrough for Detailer (K-5) and
    FaceProcess (H-5) -- the manifest is never consulted."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in ADVANCED:
            result = cls.run_case("run_fixture", fixture=fixture, fake_postprocess=True,
                                  overrides=_enable(fixture, ()), inspect=_inspect(fixture))
            cls.r[fixture] = None if result is None else result["runs"][0]

    def test_all_off_unchanged(self):
        detailer, faceprocess = self.r["detailer_adv"], self.r["faceprocess_adv"]
        self.assertEqual(_statuses(detailer, "detailer_boundary")[-1], "DISABLED_PASSTHROUGH")
        self.assertEqual(_statuses(faceprocess, "faceprocess_boundary")[-1], "DISABLED_PASSTHROUGH")
        for run in (detailer, faceprocess):
            self.assertEqual(run["seams"]["original_calls"], {})
            self.assertEqual(run["seams"]["substituted_calls"], {})


class AdvancedToggles(ScenarioCase):
    """OFF -> ON and ON -> OFF across restarts in one cache directory.
    Detailer: the boundary key changes with every toggle (MISS); FaceProcess:
    the boundary key stays, the manifest changes (STALE_DEPENDENCIES -> MISS).
    Only newly enabled branches compute; the others reuse their branch cache.
    A boundary keeps only its latest entry per node, so returning to an
    earlier setting recomputes the boundary once from the branch caches."""

    STEPS = (("only_2", (2,)), ("on_1", (1, 2)), ("off_1", (2,)), ("repeat", (2,)))

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in ADVANCED:
            workdir = cls.workdir(f"toggle_{fixture}")
            for step, active in cls.STEPS:
                result = cls.run_case("run_fixture", workdir=workdir, fixture=fixture, fake_postprocess=True,
                                      upstream_visual=True, overrides=_enable(fixture, active), inspect=_inspect(fixture))
                cls.r[(fixture, step)] = None if result is None else result["runs"][0]

    def test_off_to_on_computes_only_the_new_branch(self):
        for fixture, (_b, scope, _p, engine) in ADVANCED.items():
            with self.subTest(fixture=fixture):
                run = self.r[(fixture, "on_1")]
                self.assertTrue(run["succeeded"], run["errors"])
                self.assertEqual(_statuses(run, scope)[-1], "MISS_STORED")
                self.assertEqual(run["seams"]["substituted_calls"].get(engine), 1)
                if fixture == "faceprocess_adv":
                    self.assertIn("STALE_DEPENDENCIES", _statuses(run, scope))
                else:
                    self.assertNotIn("STALE_DEPENDENCIES", _statuses(run, scope))

    def test_on_to_off_recomputes_the_boundary_from_branch_caches(self):
        for fixture, (_b, scope, _p, engine) in ADVANCED.items():
            with self.subTest(fixture=fixture):
                run = self.r[(fixture, "off_1")]
                self.assertEqual(_statuses(run, scope)[-1], "MISS_STORED")
                self.assertNotIn(engine, run["seams"]["substituted_calls"])
                self.assertEqual(run["outputs"], self.r[(fixture, "only_2")]["outputs"])
                if fixture == "faceprocess_adv":
                    self.assertIn("STALE_DEPENDENCIES", _statuses(run, scope))

    def test_unchanged_setting_hits_again(self):
        for fixture, (_b, scope, _p, engine) in ADVANCED.items():
            with self.subTest(fixture=fixture):
                run = self.r[(fixture, "repeat")]
                self.assertEqual(_statuses(run, scope)[-1], "HIT")
                self.assertNotIn(engine, run["seams"]["substituted_calls"])
                self.assertEqual(run["outputs"], self.r[(fixture, "only_2")]["outputs"])
                self.assertEqual(run["visual"], self.r[(fixture, "only_2")]["visual"])


if __name__ == "__main__":
    unittest.main()
