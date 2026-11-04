"""FaceRebuild — persistent boundary HITs without the MODEL session (F1-F4).

Before: a HIT required ``_FACE_REBUILD_SESSION[key] = (MODEL, diagnostic)``;
after eviction the session survived, after a restart the boundary always
recomputed although IMAGE, LOG and VISUAL were on disk.

Now (variant C, no rewiring, keys byte-identical): the diagnostic is stored
next to the entry and a HIT needs disk only.  MODEL: canonical definitions
(OD-2) never read the boundary MODEL output, so a HIT requests nothing; legacy
definitions whose gate reads ``boundary.MODEL`` get the session copy or, after
a restart, the requested input (which pulls the compute node, as before).

Model-free seam: ``fake_postprocess`` replaces ``_rebuild`` of both compute
classes (counted as ``facerebuild.run``); model resolution stays real and is
counted.  MODEL-only requests: test_s1_k5_od2.ModelOnly.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from harness.scenarios import inner_ids
from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
CONTRACTS = Path(__file__).resolve().parent / "contracts"
PINNED = json.loads((CONTRACTS / "boundary_keys_pre_facerebuild.json").read_text(encoding="utf-8"))["fixtures"]
SCOPE = "face_rebuild_boundary"
ENGINE = "facerebuild.run"
RESOLVE = "prepare.resolve_postprocess_model"
COMPUTE = {"facerebuild_std": "CMKInstantIDFaceRebuildSDXL", "facerebuild_adv": "CMKInstantIDFaceRebuildAdvancedSDXL"}


def _compute_class(fixture):
    return COMPUTE["facerebuild_adv" if fixture.startswith("facerebuild_adv") else "facerebuild_std"]


def _inspect(fixture):
    gate = inner_ids(fixture, "CMKModuleBypassGate")[0]
    boundary = inner_ids(fixture, "CMKFaceRebuildBoundaryCache")[0]
    return {"model": f"{gate}:0", "boundary_model": f"{boundary}:0", "image": f"{gate}:1", "log": f"{gate}:2",
            "diagnostic": f"{gate}:3", "loader": "1:2"}


def _slots(fixture, active):
    node = inner_ids(fixture, _compute_class(fixture))[0]
    return {f"{node}.FACE {index} ENABLE": index in active for index in (1, 2, 3)}


def _statuses(run):
    return [status for scope, status, _key in run["statuses"] if scope == SCOPE]


def _result(run):
    return {key: run["outputs"][key] for key in ("image", "log", "diagnostic")}, run["visual"]


class BoundaryKeys(ScenarioCase):
    def test_keys_byte_identical(self):
        for fixture, pinned in PINNED.items():
            with self.subTest(fixture=fixture):
                result = self.run_case("boundary_key", fixture=fixture, boundary_class="CMKFaceRebuildBoundaryCache")
                self.assertEqual(result["key"], pinned["key"])


class CanonicalPersistence(ScenarioCase):
    """Standard and Advanced (1, 2, 3 slots, none): cold MISS stores; same
    process reuses ComfyUI's cache; after eviction and after a restart the
    boundary HITs with 0 compute and 0 model resolution, results identical."""

    CASES = {("facerebuild_std", "one"): None, ("facerebuild_adv", "one"): (1,), ("facerebuild_adv", "two"): (1, 2),
             ("facerebuild_adv", "three"): (1, 2, 3), ("facerebuild_adv", "none"): ()}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for (fixture, label), active in cls.CASES.items():
            overrides = {} if active is None else _slots(fixture, active)
            common = dict(fixture=fixture, overrides=overrides, fake_postprocess=True, upstream_visual=True,
                          inspect=_inspect(fixture))
            same = cls.run_case("run_fixture", runs=2, **common)
            evict = cls.run_case("run_fixture", runs=2, mode="ram_pressure", evict_between_runs=True, **common)
            workdir = cls.workdir(f"restart_{fixture}_{label}")
            first = cls.run_case("run_fixture", workdir=workdir, **common)
            second = cls.run_case("run_fixture", workdir=workdir, **common)
            cls.r[(fixture, label, "same")] = None if same is None else same["runs"]
            cls.r[(fixture, label, "evict")] = None if evict is None else evict["runs"]
            cls.r[(fixture, label, "restart")] = None if first is None else first["runs"] + second["runs"]

    def _cases(self, variants=("evict", "restart")):
        for (fixture, label, variant), runs in self.r.items():
            if variant in variants:
                yield fixture, label, variant, runs[0], runs[1]

    def test_cold_run_stores(self):
        for fixture, label, variant, cold, _second in self._cases(("same", "evict", "restart")):
            with self.subTest(fixture=fixture, label=label, variant=variant):
                self.assertTrue(cold["succeeded"], cold["errors"])
                self.assertEqual(_statuses(cold), ["MISS_STORED"])
                self.assertEqual(cold["seams"]["substituted_calls"].get(ENGINE), 1)
                active = self.CASES[(fixture, label)]
                self.assertEqual(cold["seams"]["original_calls"], {} if active == () else {RESOLVE: 1})

    def test_same_process_reuses_comfyui_cache(self):
        for fixture, label, _variant, cold, second in self._cases(("same",)):
            with self.subTest(fixture=fixture, label=label):
                self.assertEqual(_statuses(second), [])
                self.assertEqual(second["seams"]["substituted_calls"], {})
                self.assertEqual(_result(second), _result(cold))

    def test_hit_after_eviction_and_restart_without_compute_or_resolution(self):
        for fixture, label, variant, cold, hit in self._cases():
            with self.subTest(fixture=fixture, label=label, variant=variant):
                self.assertTrue(hit["succeeded"], hit["errors"])
                self.assertEqual(_statuses(hit)[-1], "HIT")
                self.assertNotIn(_compute_class(fixture), hit["function_calls"])
                self.assertEqual(hit["seams"]["substituted_calls"], {})
                self.assertEqual(hit["seams"]["original_calls"], {})
                self.assertEqual(_result(hit), _result(cold))

    def test_canonical_hit_requests_no_model(self):
        # The canonical boundary MODEL output is unread (OD-2): a HIT does not
        # request MODEL (it stays unevaluated) while the MISS forwards it.
        for fixture, label, variant, cold, hit in self._cases():
            with self.subTest(fixture=fixture, label=label, variant=variant):
                self.assertEqual(cold["outputs"]["boundary_model"], {"type": "dict"})
                self.assertEqual(hit["outputs"]["boundary_model"], {"type": "NoneType"})


class SlotToggles(ScenarioCase):
    """FACE n ENABLE is part of the compute fingerprint: a toggle changes the
    boundary key (MISS); the boundary keeps its latest entry per node, so
    returning recomputes once and then HITs."""

    STEPS = (("one", (1,)), ("two", (1, 2)), ("back", (1,)), ("repeat", (1,)))

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        workdir = cls.workdir("toggle")
        for step, active in cls.STEPS:
            result = cls.run_case("run_fixture", workdir=workdir, fixture="facerebuild_adv", fake_postprocess=True,
                                  overrides=_slots("facerebuild_adv", active), inspect=_inspect("facerebuild_adv"))
            cls.r[step] = None if result is None else result["runs"][0]

    def test_toggle_invalidates_and_repeat_hits(self):
        self.assertEqual(_statuses(self.r["one"]), ["MISS_STORED"])
        self.assertEqual(_statuses(self.r["two"]), ["MISS_STORED"])
        self.assertEqual(self.r["two"]["seams"]["substituted_calls"].get(ENGINE), 1)
        self.assertNotEqual(self.r["two"]["outputs"]["image"], self.r["one"]["outputs"]["image"])
        self.assertEqual(_statuses(self.r["back"]), ["MISS_STORED"])
        self.assertEqual(self.r["back"]["outputs"], self.r["one"]["outputs"])
        self.assertEqual(_statuses(self.r["repeat"])[-1], "HIT")
        self.assertEqual(self.r["repeat"]["seams"]["substituted_calls"], {})
        self.assertEqual(_result(self.r["repeat"]), _result(self.r["one"]))


class GlobalOff(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture, name in (("facerebuild_std", "FACEREBUILD ENABLE"), ("facerebuild_adv", "facerebuild_global_enable")):
            result = cls.run_case("run_fixture", fixture=fixture, widget_overrides={f"8417.{name}": False},
                                  fake_postprocess=True, upstream_visual=True, inspect=_inspect(fixture))
            cls.r[fixture] = None if result is None else result["runs"][0]

    def test_bypass_without_module_work(self):
        for fixture, run in self.r.items():
            with self.subTest(fixture=fixture):
                self.assertTrue(run["succeeded"], run["errors"])
                self.assertEqual(_statuses(run), [])
                self.assertEqual(run["seams"]["substituted_calls"], {})
                self.assertEqual(run["seams"]["original_calls"], {})
                self.assertEqual(run["outputs"]["image"], run["outputs"]["loader"])
                self.assertEqual([p["module_type"] for p in run["visual"]["8418"]], ["CMKVisualProvider"])
                self.assertIn("CMKVisualizer", run["function_calls"])


class LegacyModelReader(ScenarioCase):
    """Pre-OD-2 definitions (gate MODEL ACTIVE <- boundary.MODEL): in-process
    HIT returns the session MODEL without compute; after a restart the HIT
    requests MODEL through the old input path, which runs the compute node,
    and still returns the stored result.  Never a None MODEL."""

    FIXTURES = ("facerebuild_std_legacy_pre_od2", "facerebuild_adv_legacy_pre_od2")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in cls.FIXTURES:
            common = dict(fixture=fixture, fake_postprocess=True, inspect=_inspect(fixture))
            evict = cls.run_case("run_fixture", runs=2, mode="ram_pressure", evict_between_runs=True, **common)
            workdir = cls.workdir(f"legacy_{fixture}")
            first = cls.run_case("run_fixture", workdir=workdir, **common)
            second = cls.run_case("run_fixture", workdir=workdir, **common)
            cls.r[(fixture, "evict")] = None if evict is None else evict["runs"]
            cls.r[(fixture, "restart")] = None if first is None else first["runs"] + second["runs"]

    def test_model_is_read_from_the_boundary(self):
        from harness.scenarios import load_api_prompt

        for fixture in self.FIXTURES:
            prompt = load_api_prompt(fixture)
            boundary = inner_ids(fixture, "CMKFaceRebuildBoundaryCache")[0]
            gate = inner_ids(fixture, "CMKModuleBypassGate")[0]
            self.assertEqual(prompt[gate]["inputs"]["MODEL ACTIVE"], [boundary, 0], fixture)

    def test_in_process_hit_uses_the_session_model(self):
        for fixture in self.FIXTURES:
            with self.subTest(fixture=fixture):
                cold, hit = self.r[(fixture, "evict")]
                self.assertEqual(_statuses(hit)[-1], "HIT")
                self.assertEqual(hit["seams"]["substituted_calls"], {})
                self.assertEqual(hit["seams"]["original_calls"], {})
                self.assertEqual(hit["outputs"]["model"], {"type": "dict"})
                self.assertEqual(_result(hit), _result(cold))

    def test_restart_hit_requests_model_through_the_input(self):
        for fixture in self.FIXTURES:
            with self.subTest(fixture=fixture):
                cold, hit = self.r[(fixture, "restart")]
                self.assertEqual(_statuses(hit)[-1], "HIT")
                self.assertEqual(hit["seams"]["substituted_calls"].get(ENGINE), 1)
                self.assertEqual(hit["outputs"]["model"], {"type": "dict"})
                self.assertEqual(_result(hit), _result(cold))


class EntryWithoutDiagnostic(ScenarioCase):
    """F3: an entry without the diagnostic sidecar (stored before this change)
    is recomputed once, then HITs."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.runs = None
        if cls.paths is None:
            return
        workdir = cls.workdir("migration")
        common = dict(fixture="facerebuild_std", fake_postprocess=True, inspect=_inspect("facerebuild_std"))
        cold = cls.run_case("run_fixture", workdir=workdir, **common)["runs"][0]
        removed = []
        for path in (Path(workdir) / "temp" / "cmk" / SCOPE).glob("*.diagnostic.*"):
            path.unlink()
            removed.append(path.name)
        again = cls.run_case("run_fixture", workdir=workdir, **common)["runs"][0]
        third = cls.run_case("run_fixture", workdir=workdir, **common)["runs"][0]
        cls.runs = (cold, removed, again, third)

    def test_recomputed_once_then_hit(self):
        cold, removed, again, third = self.runs
        self.assertEqual(len(removed), 2)
        self.assertEqual(_statuses(again), ["MISS_STORED"])
        self.assertEqual(again["seams"]["substituted_calls"].get(ENGINE), 1)
        self.assertEqual(_statuses(third)[-1], "HIT")
        self.assertEqual(third["seams"]["substituted_calls"], {})
        self.assertEqual(_result(third), _result(cold))


if __name__ == "__main__":
    unittest.main()
