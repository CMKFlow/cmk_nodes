"""K-5 / OD-2 — Detailer passthrough without Prepare; MODEL without module work.

K-5: with the Detailer enabled globally but every local branch OFF, the
boundary used to request IMAGE/LOG from the compute node and the gate pulled
the compute node's diagnostic directly; both dragged in Detailer Prepare and
``resolve_postprocess_model``.  The boundary now answers from its internal
passthrough inputs (IMAGE IN / LOG IN) and owns the module diagnostic
(schema v3 -> v4, a one-time MISS).

OD-2: the public MODEL output of Detailer, FaceProcess and FaceRebuild left
through the result gate, which evaluates the whole module.  It is now CMK
Model Forward from SG_IN.MODEL (PostProcess modules never change MODEL).

Active Detailer runs use the model-free seams of E9 (``fake_postprocess``):
SAM, CLIP layer, LoRA, model patches, CLIP encode and the Smart Detailer run
are deterministic fakes; ``resolve_postprocess_model`` stays real.

Legacy: the pre-K-5 exports (``*_legacy_pre_k5``) keep their embedded wiring
and therefore their previous behaviour.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from harness.scenarios import inner_ids
from harness.support import ScenarioCase, gate_of, work_calls

CMK_TEST_TIER = "T1"
CONTRACTS = Path(__file__).resolve().parent / "contracts"
PINNED = json.loads((CONTRACTS / "boundary_keys_pre_k5.json").read_text(encoding="utf-8"))
DETAILER = ("detailer_std", "detailer_adv")
LEGACY = ("detailer_std_legacy_pre_k5", "detailer_adv_legacy_pre_k5")
OD2_FIXTURES = ("detailer_std", "detailer_adv", "faceprocess_std", "faceprocess_adv",
                "facerebuild_std", "facerebuild_adv")
BOUNDARY = {
    "faceswap_std": "CMKFaceSwapBoundaryCache", "faceswap_adv": "CMKFaceSwapBoundaryCache",
    "detailer_std": "CMKDetailerBoundaryCache", "detailer_adv": "CMKDetailerBoundaryCache",
    "faceprocess_std": "CMKFaceBoundaryCache", "faceprocess_adv": "CMKFaceBoundaryCache",
    "facerebuild_std": "CMKFaceRebuildBoundaryCache", "facerebuild_adv": "CMKFaceRebuildBoundaryCache",
    "upscale_save": "CMKUpscaleSaveBoundaryCache",
}
RESOLVE = "prepare.resolve_postprocess_model"
FAKE_PREPARE = ("prepare.sam_load", "prepare.clip_layer", "prepare.lora", "prepare.model_patch", "prepare.clip_encode")


def _inspect(fixture):
    gate = gate_of(fixture)
    return {"image": f"{gate}:1", "log": f"{gate}:2", "diagnostic": f"{gate}:3", "loader": "1:2", "loader_log": "1:3"}


def _local_off(fixture):
    return {f"{node}.enable": False for node in inner_ids(fixture, "CMK_SmartDetailerPipe")}


def _statuses(run, scope="detailer_boundary"):
    return [status for s, status, _key in run["statuses"] if s == scope]


class DetailerPassthrough(ScenarioCase):
    """Global ON, all local branches OFF (fake seams installed: any Prepare
    work would succeed silently and be counted)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in DETAILER:
            result = cls.run_case("run_fixture", fixture=fixture, overrides=_local_off(fixture),
                                  fake_postprocess=True, upstream_visual=True, inspect=_inspect(fixture))
            cls.r[fixture] = None if result is None else result["runs"][0]

    def test_no_prepare_compute_or_materialisation(self):
        for fixture, run in self.r.items():
            with self.subTest(fixture=fixture):
                self.assertTrue(run["succeeded"], run["errors"])
                self.assertNotIn("CMKDetailerPreparePipe", run["function_calls"])
                self.assertNotIn("CMK_SmartDetailerPipe", run["function_calls"])
                self.assertEqual(run["seams"]["original_calls"], {})
                self.assertEqual(run["seams"]["substituted_calls"], {})
                self.assertEqual(run["seams"]["stopped_calls"], {})

    def test_image_unchanged(self):
        for fixture, run in self.r.items():
            self.assertEqual(run["outputs"]["image"], run["outputs"]["loader"], fixture)

    def test_log_keeps_input_and_adds_boundary_status_block(self):
        for fixture, run in self.r.items():
            with self.subTest(fixture=fixture):
                blocks = run["outputs"]["log"]["blocks"]
                self.assertEqual(blocks[:-1], run["outputs"]["loader_log"]["blocks"])
                title, lines = blocks[-1]
                self.assertEqual(title, "Detailer Boundary")
                self.assertIn("STATUS          : DISABLED PASSTHROUGH", lines)

    def test_lightweight_diagnostic_from_the_boundary(self):
        for fixture, run in self.r.items():
            self.assertEqual(run["outputs"]["diagnostic"],
                             {"type": "diagnostic", "title": "Detailer Boundary", "mode": "disabled / passthrough"},
                             fixture)

    def test_visual_pipe_unchanged(self):
        for fixture, run in self.r.items():
            self.assertEqual([p["module_type"] for p in run["visual"]["8418"]], ["CMKVisualProvider"], fixture)

    def test_status_and_no_cache_access(self):
        for fixture, run in self.r.items():
            statuses = _statuses(run)
            self.assertEqual(statuses[-1], "DISABLED_PASSTHROUGH", fixture)
            self.assertEqual(set(statuses), {"DISABLED_PASSTHROUGH_READY", "DISABLED_PASSTHROUGH"}, fixture)


class DetailerActive(ScenarioCase):
    """Active processing keeps its result and cache contracts: Prepare and
    compute run once on a MISS and the boundary stores; the warm run reuses
    ComfyUI's cache; after eviction and after a restart (cache files kept) the
    boundary HITs with MODEL from the current input and IMAGE, LOG,
    diagnostic and VISUAL from the stored entry -- no Prepare, no compute, no
    model resolution (K5-B fixed: manifest keys from the branch class, no
    session copy of MODEL)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in DETAILER:
            common = dict(fixture=fixture, fake_postprocess=True, upstream_visual=True, inspect=_inspect(fixture))
            warm = cls.run_case("run_fixture", runs=2, **common)
            evict = cls.run_case("run_fixture", runs=2, mode="ram_pressure", evict_between_runs=True, **common)
            workdir = cls.workdir(f"restart_{fixture}")
            first = cls.run_case("run_fixture", workdir=workdir, **common)
            second = cls.run_case("run_fixture", workdir=workdir, **common)
            cls.r[(fixture, "warm")] = None if warm is None else warm["runs"]
            cls.r[(fixture, "evict")] = None if evict is None else evict["runs"]
            cls.r[(fixture, "restart")] = None if first is None else first["runs"] + second["runs"]

    def test_miss_runs_prepare_and_compute_once(self):
        for fixture in DETAILER:
            for variant in ("warm", "evict", "restart"):
                with self.subTest(fixture=fixture, variant=variant):
                    cold = self.r[(fixture, variant)][0]
                    self.assertTrue(cold["succeeded"], cold["errors"])
                    self.assertEqual(cold["function_calls"].get("CMKDetailerPreparePipe"), 1)
                    # One call per branch node; only enabled branches reach the detailer run.
                    self.assertEqual(cold["function_calls"].get("CMK_SmartDetailerPipe"),
                                     len(inner_ids(fixture, "CMK_SmartDetailerPipe")))
                    self.assertEqual(cold["seams"]["original_calls"], {RESOLVE: 1})
                    self.assertEqual(cold["seams"]["substituted_calls"].get("detailer.run"), 1)
                    self.assertNotEqual(cold["outputs"]["image"], cold["outputs"]["loader"])

    def test_module_diagnostic_reaches_the_gate_through_the_boundary(self):
        for fixture in DETAILER:
            cold = self.r[(fixture, "warm")][0]
            self.assertEqual(cold["outputs"]["diagnostic"]["type"], "diagnostic", fixture)
            self.assertNotEqual(cold["outputs"]["diagnostic"]["title"], "Module Result", fixture)

    def test_active_publishes_upstream_and_module_provider(self):
        for fixture in DETAILER:
            cold = self.r[(fixture, "warm")][0]
            modules = [p["module_type"] for p in cold["visual"]["8418"]]
            self.assertEqual(modules[0], "CMKVisualProvider", fixture)
            self.assertEqual(len(modules), 2, fixture)

    def test_warm_run_reuses_comfyui_cache(self):
        for fixture in DETAILER:
            cold, warm = self.r[(fixture, "warm")]
            self.assertEqual(work_calls(warm), {}, fixture)
            self.assertEqual(warm["seams"]["original_calls"], {}, fixture)
            self.assertEqual(warm["outputs"], cold["outputs"], fixture)

    def test_miss_stores_the_boundary(self):
        # detailer_adv (branches 1 and 3 OFF by default) never stores: K5-C in test_s1_k5b.
        for variant in ("evict", "restart"):
            self.assertEqual(_statuses(self.r[("detailer_std", variant)][0]), ["MISS_STORED"], variant)

    def test_boundary_hit_after_eviction_and_restart_runs_no_prepare_or_compute(self):
        for fixture in ("detailer_std",):
            for variant in ("evict", "restart"):
                with self.subTest(fixture=fixture, variant=variant):
                    cold, hit = self.r[(fixture, variant)]
                    self.assertTrue(hit["succeeded"], hit["errors"])
                    self.assertEqual(_statuses(hit)[-1], "HIT")
                    self.assertEqual(set(_statuses(hit)), {"HIT_READY", "HIT"})
                    self.assertNotIn("CMKDetailerPreparePipe", hit["function_calls"])
                    self.assertNotIn("CMK_SmartDetailerPipe", hit["function_calls"])
                    self.assertEqual(hit["seams"]["original_calls"], {})
                    self.assertEqual(hit["seams"]["substituted_calls"], {})
                    self.assertEqual(hit["outputs"], cold["outputs"])
                    self.assertEqual(hit["visual"], cold["visual"])


class PassthroughSourcesMixin(unittest.TestCase):
    """The boundary-owned diagnostic in isolation: MISS requests and stores
    it, HIT restores it without requesting the compute node, legacy
    definitions are untouched and the disabled passthrough never calls the
    core.  (The executor path is covered by DetailerActive.)"""

    @classmethod
    def setUpClass(cls):
        import shutil
        import tempfile

        from harness.comfy_runtime import boot

        cls.root = Path(tempfile.mkdtemp(prefix="cmk-s1-k5-"))
        cls.addClassCleanup(shutil.rmtree, cls.root, True)
        runtime = boot(cls.root)
        cls.mod = runtime.cmk_module("pipe.cmk_module_boundary_cache")
        cls.ExecutionBlocker = cls.mod.ExecutionBlocker
        mod = cls.mod

        class Core:
            _SCOPE = "detailer_boundary"
            cached = False
            disabled = False
            calls = []

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"MODEL": ("CMK_MODEL_PIPE", {"lazy": True})}}

            def _cache_key(self, prompt, unique_id):
                return "k5" * 32, ""

            def _module_disabled(self, prompt, unique_id):
                return type(self).disabled

            def _result_cached(self, prompt, unique_id):
                return type(self).cached

            def check_lazy_status(self, MODEL=None, IMAGE=None, LOG=None, prompt=None, unique_id=None):
                type(self).calls.append("check")
                return [] if type(self).cached else ["IMAGE", "LOG"]

            def boundary(self, MODEL=None, IMAGE=None, LOG=None, prompt=None, unique_id=None):
                type(self).calls.append("boundary")
                return MODEL, "image", {"blocks": []}

        class Boundary(mod._PassthroughSources, mod._VisualPublication, Core):
            _PASSTHROUGH_LABEL = "Detailer"

        cls.Core, cls.Boundary = Core, Boundary
        cls.key = "k5" * 32

    def setUp(self):
        self.Core.cached = self.Core.disabled = False
        self.Core.calls.clear()
        for path in (*self.mod._entry_paths("detailer_boundary", self.key),
                     *self.mod._diagnostic_entry_paths("detailer_boundary", self.key)):
            path.unlink(missing_ok=True)

    def _prompt(self, routed=True):
        inputs = {"MODEL": ["1", 0], "IMAGE": ["9", 2], "LOG": ["9", 3]}
        if routed:
            inputs.update({"IMAGE IN": ["1", 2], "LOG IN": ["1", 3], "diagnostic": ["9", 5]})
        return {"7": {"class_type": "CMKDetailerBoundaryCache", "inputs": inputs}}

    def _store_entry(self):
        for path in self.mod._entry_paths("detailer_boundary", self.key):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")

    def _diagnostic(self):
        import torch

        return {"type": "CMK_DIAGNOSTIC", "version": 1, "title": "Smart Detailer", "previews": [torch.ones((1, 4, 4, 3))]}

    def test_miss_requests_and_stores_the_diagnostic(self):
        boundary = self.Boundary()
        needed = boundary.check_lazy_status(prompt=self._prompt(), unique_id="7")
        self.assertEqual(needed, ["IMAGE", "LOG", "diagnostic"])
        self._store_entry()
        outputs = boundary.boundary(prompt=self._prompt(), unique_id="7", MODEL=None, diagnostic=self._diagnostic())
        self.assertEqual(len(outputs), 5)
        self.assertEqual(outputs[4]["title"], "Smart Detailer")
        self.assertTrue(all(p.is_file() for p in self.mod._diagnostic_entry_paths("detailer_boundary", self.key)))

    def test_hit_restores_the_diagnostic_without_the_compute_node(self):
        boundary = self.Boundary()
        self._store_entry()
        self.mod._save_diagnostic("detailer_boundary", self.key, self._diagnostic())
        self.Core.cached = True
        self.assertEqual(boundary.check_lazy_status(prompt=self._prompt(), unique_id="7"), [])
        outputs = boundary.boundary(prompt=self._prompt(), unique_id="7", MODEL=None)
        self.assertEqual(outputs[4]["title"], "Smart Detailer")
        self.assertEqual(tuple(outputs[4]["previews"][0].shape), (1, 4, 4, 3))

    def test_entry_without_diagnostic_is_recomputed_once(self):
        self._store_entry()
        self.Boundary().check_lazy_status(prompt=self._prompt(), unique_id="7")
        self.assertFalse(any(p.is_file() for p in self.mod._entry_paths("detailer_boundary", self.key)))

    def test_legacy_definition_keeps_the_core_behaviour(self):
        boundary = self.Boundary()
        self.assertEqual(boundary.check_lazy_status(prompt=self._prompt(routed=False), unique_id="7"), ["IMAGE", "LOG"])
        outputs = boundary.boundary(prompt=self._prompt(routed=False), unique_id="7", MODEL=None)
        self.assertEqual(outputs[:3], (None, "image", {"blocks": []}))
        self.assertIsInstance(outputs[4], self.ExecutionBlocker)

    def test_disabled_passthrough_never_calls_the_core(self):
        import torch

        self.Core.disabled = True
        boundary = self.Boundary()
        self.assertEqual(boundary.check_lazy_status(prompt=self._prompt(), unique_id="7"), ["IMAGE IN", "LOG IN", "MODEL"])
        image = torch.zeros((1, 4, 4, 3))
        outputs = boundary.boundary(prompt=self._prompt(), unique_id="7", MODEL=None,
                                    **{"IMAGE IN": image, "LOG IN": {"blocks": []}})
        self.assertEqual(self.Core.calls, [])
        self.assertIs(outputs[1], image)
        self.assertEqual(outputs[2]["blocks"][-1]["title"], "Detailer Boundary")
        self.assertEqual(outputs[4]["mode"], "disabled / passthrough")


class ModelOnly(ScenarioCase):
    """OD-2: requesting only the public MODEL output (the expanded SG_OUT.MODEL
    source) runs no Prepare, compute, boundary or gate -- with and without a
    checkpoint (MODEL None)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in OD2_FIXTURES:
            source = f"{inner_ids(fixture, 'CMKModelForwardPipe')[0]}:0"
            for drop in (False, True):
                cls.r[(fixture, drop)] = cls.run_case("forward_only", fixture=fixture, source=source, drop_model=drop)

    def test_model_only_runs_no_module_work(self):
        for (fixture, drop), result in self.r.items():
            with self.subTest(fixture=fixture, model_none=drop):
                self.assertTrue(result["succeeded"], result["errors"])
                self.assertEqual(work_calls(result), {})
                self.assertNotIn("CMKModuleBypassGate", result["function_calls"])
                self.assertEqual(result["function_calls"].get("CMKModelForwardPipe"), 1)
                self.assertEqual(result["seams"]["stopped_calls"], {})
                self.assertEqual(result["seams"]["original_calls"], {})

    def test_public_model_output_is_the_forward(self):
        from harness.scenarios import load_api_prompt

        for fixture in OD2_FIXTURES:
            prompt = load_api_prompt(fixture)
            forward = inner_ids(fixture, "CMKModelForwardPipe")[0]
            self.assertEqual(prompt["8418"]["inputs"]["MODEL (opt)"], [forward, 0], fixture)
            self.assertEqual(prompt[forward]["inputs"]["MODEL"], ["1", 0], fixture)
            gate = prompt[gate_of(fixture)]["inputs"]
            self.assertNotIn("MODEL ACTIVE", gate, fixture)
            self.assertNotIn("MODEL BYPASS", gate, fixture)


class LegacyPreK5(ScenarioCase):
    """Embedded pre-K-5 definitions keep their behaviour (no migration)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in LEGACY:
            passthrough = cls.run_case("run_fixture", fixture=fixture, overrides=_local_off(fixture))
            active = cls.run_case("run_fixture", fixture=fixture, fake_postprocess=True, inspect=_inspect(fixture))
            model = cls.run_case("forward_only", fixture=fixture, source=f"{gate_of(fixture)}:0")
            cls.r[fixture] = (None if passthrough is None else passthrough["runs"][0],
                              None if active is None else active["runs"][0], model)

    def test_legacy_passthrough_still_runs_prepare(self):
        for fixture, (passthrough, _active, _model) in self.r.items():
            with self.subTest(fixture=fixture):
                self.assertFalse(passthrough["succeeded"])
                self.assertEqual(passthrough["seams"]["stopped_calls"], {RESOLVE: 1})
                self.assertEqual([e["node_type"] for e in passthrough["errors"]], ["CMKDetailerPreparePipe"])

    def test_legacy_active_processing_unchanged(self):
        for fixture, (_passthrough, active, _model) in self.r.items():
            with self.subTest(fixture=fixture):
                self.assertTrue(active["succeeded"], active["errors"])
                self.assertEqual(active["seams"]["original_calls"], {RESOLVE: 1})
                self.assertNotEqual(active["outputs"]["image"], active["outputs"]["loader"])
                self.assertNotEqual(active["outputs"]["diagnostic"]["title"], "Module Result")

    def test_legacy_model_request_still_runs_prepare(self):
        for fixture, (_passthrough, _active, model) in self.r.items():
            self.assertFalse(model["succeeded"], fixture)
            self.assertEqual(model["seams"]["stopped_calls"], {RESOLVE: 1}, fixture)


class BoundaryKeys(ScenarioCase):
    """The rewiring changes no cache identity; only the Detailer schema moves
    from v3 to v4 (one deliberate MISS).  Recorded before the change."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.old = {}
        cls.new = {}
        for fixture, boundary_class in BOUNDARY.items():
            schema = PINNED["schemas"]["detailer"] if fixture in DETAILER else None
            cls.old[fixture] = cls.run_case("boundary_key", fixture=fixture, boundary_class=boundary_class, schema=schema)
            cls.new[fixture] = cls.run_case("boundary_key", fixture=fixture, boundary_class=boundary_class)

    def test_keys_and_manifests_unchanged_with_the_old_schema(self):
        for fixture, result in self.old.items():
            with self.subTest(fixture=fixture):
                self.assertEqual(result["key"], PINNED["fixtures"][fixture]["key"])
                self.assertEqual(result["dependencies"], PINNED["fixtures"][fixture]["deps"])

    def test_detailer_schema_bump_changes_only_the_detailer_keys(self):
        for fixture, result in self.new.items():
            with self.subTest(fixture=fixture):
                if fixture in DETAILER:
                    self.assertNotEqual(result["key"], PINNED["fixtures"][fixture]["key"])
                else:
                    self.assertEqual(result["key"], PINNED["fixtures"][fixture]["key"])
                self.assertEqual(result["dependencies"], PINNED["fixtures"][fixture]["deps"])


class ModelForwardNode(unittest.TestCase):
    """E4: CMK Model Forward takes an optional MODEL and passes None through."""

    @classmethod
    def setUpClass(cls):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "pipe" / "cmk_model_forward.py"
        spec = importlib.util.spec_from_file_location("cmk_s1_model_forward", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.node = module.CMKModelForwardPipe

    def test_model_input_is_optional(self):
        self.assertEqual(self.node.INPUT_TYPES(), {"optional": {"MODEL": ("CMK_MODEL_PIPE",)}})

    def test_none_and_model_pass_through_unchanged(self):
        model = {"model_role": "postprocess"}
        self.assertEqual(self.node.forward(), (None,))
        self.assertEqual(self.node.forward(None), (None,))
        self.assertIs(self.node.forward(model)[0], model)

    def test_non_pipe_is_rejected(self):
        with self.assertRaises(TypeError):
            self.node.forward("model")


if __name__ == "__main__":
    unittest.main()
