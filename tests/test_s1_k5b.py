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
    """FaceProcess boundary stores and HITs after eviction and restart.  The
    gate still requests DIAGNOSTIC ACTIVE from the compute node (left
    untouched, part of the later H-5 correction), so Prepare and compute run
    on a HIT; the boundary result is reused.  Advanced: see K5C."""

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

    def test_baseline_h5_diagnostic_line_still_pulls_prepare_and_compute(self):
        for (fixture, variant), (_cold, hit) in self.r.items():
            with self.subTest(fixture=fixture, variant=variant):
                self.assertEqual(hit["function_calls"].get("CMKFaceProcessPreparePipe"), 1)
                self.assertIn("CMKFaceProcessPipe", hit["function_calls"])

    @unittest.expectedFailure  # H-5 scope: FaceProcess DIAGNOSTIC ACTIVE comes from the compute node
    def test_hit_runs_no_prepare(self):
        for (_fixture, _variant), (_cold, hit) in self.r.items():
            self.assertNotIn("CMKFaceProcessPreparePipe", hit["function_calls"])


class K5C(ScenarioCase):
    """Pre-existing finding K5-C (not corrected): a locally disabled branch
    stores no pickle and publishes no revision, so the manifest of an
    Advanced module with any branch OFF (the default: 2 of 3) is never
    complete and its boundary never stores."""

    FIXTURES = {"detailer_adv": "detailer_boundary", "faceprocess_adv": "faceprocess_boundary"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in cls.FIXTURES:
            result = cls.run_case("run_fixture", runs=2, mode="ram_pressure", evict_between_runs=True,
                                  fixture=fixture, fake_postprocess=True, inspect=_inspect(fixture))
            cls.r[fixture] = None if result is None else result["runs"]

    def test_baseline_k5c_advanced_with_disabled_branch_never_stores(self):
        for fixture, scope in self.FIXTURES.items():
            with self.subTest(fixture=fixture):
                cold, again = self.r[fixture]
                self.assertTrue(cold["succeeded"], cold["errors"])
                self.assertEqual(_statuses(cold, scope), [])
                self.assertEqual(_statuses(again, scope), [])

    @unittest.expectedFailure  # K5-C: disabled branches publish no revision
    def test_advanced_boundary_hits_after_eviction(self):
        for fixture, scope in self.FIXTURES.items():
            self.assertEqual(_statuses(self.r[fixture][1], scope)[-1:], ["HIT"])


if __name__ == "__main__":
    unittest.main()
