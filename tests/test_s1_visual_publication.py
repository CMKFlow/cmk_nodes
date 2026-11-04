"""H-3 / F-03 — VISUAL publication at the module boundary.

H-3: ``CMKModuleBypassGate`` requests one lazy input at a time.  A legitimately
``None`` MODEL (no checkpoint) is re-requested, ComfyUI drops the request
because the input is already evaluated, and the gate runs before VISUAL was
ever requested — the whole VISUAL pipe, upstream providers included, is lost
in the active path and in bypass.

F-03: VISUAL ACTIVE comes from the compute node, in front of the boundary, so
fixing H-3 alone would pull the compute node again on every boundary HIT.

Both are fixed: ``CMKModuleBypassGate`` requests the missing result inputs
(MODEL, IMAGE, LOG) together and the auxiliary inputs (DIAGNOSTIC, VISUAL,
PROCESS) together once IMAGE and LOG exist (G1), and the module boundary
publishes the compute node's providers and restores them on a HIT.  The tests
below are the regression guard; the boundary keys are pinned to the record
made before the change.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from harness.comfy_runtime import boot
from harness.support import ScenarioCase, gate_of

CMK_TEST_TIER = "T1"
CONTRACTS = Path(__file__).resolve().parent / "contracts"
PINNED = json.loads((CONTRACTS / "boundary_keys_pre_h3.json").read_text(encoding="utf-8"))["fixtures"]
BOUNDARY = {
    "faceswap_std": "CMKFaceSwapBoundaryCache", "faceswap_adv": "CMKFaceSwapBoundaryCache",
    "detailer_std": "CMKDetailerBoundaryCache", "detailer_adv": "CMKDetailerBoundaryCache",
    "faceprocess_std": "CMKFaceBoundaryCache", "faceprocess_adv": "CMKFaceBoundaryCache",
    "facerebuild_std": "CMKFaceRebuildBoundaryCache", "facerebuild_adv": "CMKFaceRebuildBoundaryCache",
}
FACESWAP = {"faceswap_std": "CMKFaceSwapImagePipe", "faceswap_adv": "CMKSEGSConcateFaceSwap"}
FACESWAP_COMPUTE = ("CMKFaceSwapImagePipe", "CMK_SEGSConcate")
CHECKPOINT_MODULES = ("detailer_std", "detailer_adv", "faceprocess_std", "faceprocess_adv",
                      "facerebuild_std", "facerebuild_adv")
GLOBAL_ENABLE = {
    "detailer_std": "detailer_global_enable", "detailer_adv": "detailer_global_enable",
    "faceprocess_std": "face_global_enable", "faceprocess_adv": "face_global_enable",
    "facerebuild_std": "FACEREBUILD ENABLE", "facerebuild_adv": "facerebuild_global_enable",
    "faceswap_std": "FACESWAP ENABLE", "faceswap_adv": "FACESWAP ENABLE",
}


def _modules(run, node="8418"):
    return [provider["module_type"] for provider in run["visual"].get(node, [])]


class BoundaryKeysUnchanged(ScenarioCase):
    """VISUAL publication must not change any boundary cache identity.

    The record predates H-2: checkpoint fixtures then took MODEL from
    CMKCheckpointVAELoaderPipe, now from CMKPostProcessModelPipe.  The model
    source is part of the upstream fingerprint, so those keys are compared
    with the old source class restored -- any other difference fails.
    """

    def test_keys_and_dependency_manifests_match_the_pre_change_record(self):
        for fixture, boundary_class in BOUNDARY.items():
            with self.subTest(fixture=fixture):
                source = None if fixture.startswith("faceswap") else "CMKCheckpointVAELoaderPipe"
                result = self.run_case("boundary_key", fixture=fixture, boundary_class=boundary_class,
                                       model_source_class=source)
                self.assertEqual(result["key"], PINNED[fixture]["key"])
                self.assertEqual(result["dependencies"], PINNED[fixture]["deps"])

    def test_h2_model_source_changes_the_checkpoint_module_identity_once(self):
        # Migrated standalone workflows recompute their boundaries once: the
        # key changes, or (FaceProcess, MODEL excluded from its key) the
        # branch dependency manifest does.
        for fixture, boundary_class in BOUNDARY.items():
            if fixture.startswith("faceswap"):
                continue
            with self.subTest(fixture=fixture):
                result = self.run_case("boundary_key", fixture=fixture, boundary_class=boundary_class)
                self.assertNotEqual((result["key"], result["dependencies"]),
                                    (PINNED[fixture]["key"], PINNED[fixture]["deps"]))


class FaceSwapVisualWithoutCheckpoint(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in FACESWAP:
            common = dict(fixture=fixture, upstream_visual=True)
            workdir = cls.workdir(f"restart_{fixture}")
            first = cls.run_case("run_fixture", workdir=workdir, **common)
            second = cls.run_case("run_fixture", workdir=workdir, **common)
            cls.r[(fixture, "restart")] = None if first is None else first["runs"] + second["runs"]
            evict = cls.run_case("run_fixture", runs=2, mode="ram_pressure", evict_between_runs=True, **common)
            cls.r[(fixture, "evict")] = None if evict is None else evict["runs"]
            bypass = cls.run_case("run_fixture", widget_overrides={f"8417.{GLOBAL_ENABLE[fixture]}": False}, **common)
            cls.r[(fixture, "bypass")] = None if bypass is None else bypass["runs"][0]
        local = cls.run_case("run_fixture", fixture="faceswap_adv", upstream_visual=True,
                             overrides={node: False for node in cls._local_switches("faceswap_adv")})
        cls.local_off = None if local is None else local["runs"][0]

    @staticmethod
    def _local_switches(fixture):
        from harness.scenarios import inner_ids
        return [f"{node}.ENABLE" for node in inner_ids(fixture, "CMKFaceSwapImagePipe")]

    def test_hit_runs_no_compute_or_engine(self):
        for fixture in FACESWAP:
            for variant in ("restart", "evict"):
                with self.subTest(fixture=fixture, variant=variant):
                    cold, hit = self.r[(fixture, variant)]
                    self.assertEqual(cold["seams"]["substituted_calls"].get("faceswap.swap"), 1)
                    self.assertEqual([s[1] for s in hit["statuses"]][-1], "HIT")
                    for name in FACESWAP_COMPUTE:
                        self.assertNotIn(name, hit["function_calls"])
                    self.assertEqual(hit["seams"]["substituted_calls"], {})

    def test_miss_publishes_upstream_and_module_provider(self):
        for fixture, module in FACESWAP.items():
            cold = self.r[(fixture, "restart")][0]
            self.assertEqual(_modules(cold), ["CMKVisualProvider", module], fixture)

    def test_hit_restores_the_identical_provider_without_compute(self):
        for fixture in FACESWAP:
            for variant in ("restart", "evict"):
                cold, hit = self.r[(fixture, variant)]
                self.assertTrue(cold["visual"]["8418"], fixture)
                self.assertEqual(hit["visual"], cold["visual"], (fixture, variant))
                self.assertEqual(hit["visual_meta"], cold["visual_meta"], (fixture, variant))

    def test_bypass_keeps_upstream_provider(self):
        for fixture in FACESWAP:
            self.assertEqual(_modules(self.r[(fixture, "bypass")]), ["CMKVisualProvider"], fixture)

    def test_locally_disabled_module_adds_no_provider(self):
        run = self.local_off
        self.assertTrue(run["succeeded"], run["errors"])
        self.assertEqual(_modules(run), ["CMKVisualProvider"])


class CheckpointModulesBypass(ScenarioCase):
    """Detailer, FaceProcess and FaceRebuild cannot run model-free in the
    active path (Prepare materialises the model); bypass is testable."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = {}
        for fixture in CHECKPOINT_MODULES:
            common = dict(fixture=fixture, upstream_visual=True,
                          widget_overrides={f"8417.{GLOBAL_ENABLE[fixture]}": False})
            for label, drop in (("model", False), ("no_model", True)):
                result = cls.run_case("run_fixture", drop_model=drop, **common)
                cls.r[(fixture, label)] = None if result is None else result["runs"][0]

    def test_bypass_with_model_keeps_upstream_provider(self):
        for fixture in CHECKPOINT_MODULES:
            with self.subTest(fixture=fixture):
                run = self.r[(fixture, "model")]
                self.assertTrue(run["succeeded"], run["errors"])
                self.assertEqual(_modules(run), ["CMKVisualProvider"])

    def test_bypass_without_model_keeps_upstream_provider(self):
        for fixture in CHECKPOINT_MODULES:
            self.assertEqual(_modules(self.r[(fixture, "no_model")]), ["CMKVisualProvider"], fixture)


def _simulate_gate(gate, connected: dict, enable: bool) -> tuple[list, set]:
    """ComfyUI's lazy loop (execution.py): requested names that are already
    evaluated are dropped; an empty remainder executes the node."""
    evaluated = {}
    requests = []
    prompt = {"g": {"class_type": "CMKModuleBypassGate", "inputs": {name: ["src", 0] for name in connected}}}
    while True:
        inputs = {name: evaluated.get(name) for name in connected}
        requested = gate.check_lazy_status(ENABLE=enable, prompt=prompt, unique_id="g", **inputs)
        requests.append(list(requested))
        pending = [name for name in requested if name not in evaluated]
        if not pending:
            return requests, set(evaluated)
        for name in pending:
            evaluated[name] = connected[name]


class GateLazyProtocol(unittest.TestCase):
    """G1: before the gate executes, every connected input of the selected
    path is evaluated — also when one of them is legitimately None."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="cmk-s1-gate-"))
        family = boot(cls.root).cmk_module("pipe.cmk_family_result")
        cls.gate = family.CMKModuleBypassGate()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _connected(self, model):
        values = {}
        for prefix in ("ACTIVE", "BYPASS"):
            values.update({f"MODEL {prefix}": model, f"IMAGE {prefix}": "image", f"LOG {prefix}": {"log": 1},
                           f"VISUAL {prefix}": {"providers": []}})
        values["DIAGNOSTIC ACTIVE"] = {"type": "CMK_DIAGNOSTIC"}
        return values

    def _check(self, enable, model):
        connected = self._connected(model)
        requests, evaluated = _simulate_gate(self.gate, connected, enable)
        prefix, other = ("ACTIVE", "BYPASS") if enable else ("BYPASS", "ACTIVE")
        expected = {name for name in connected if name.endswith(prefix)}
        self.assertEqual(evaluated, expected, requests)
        self.assertFalse(any(name.endswith(other) for request in requests for name in request))

    def test_all_inputs_evaluated_when_model_present(self):
        for enable in (True, False):
            with self.subTest(enable=enable):
                self._check(enable, {"model": 1})

    def test_all_inputs_evaluated_when_model_is_none(self):
        for enable in (True, False):
            with self.subTest(enable=enable):
                self._check(enable, None)

    def test_result_inputs_are_requested_before_auxiliary_inputs(self):
        # A blocked result (ExecutionBlocker on IMAGE/LOG) must stop the gate
        # before diagnostic or VISUAL producers run (H-5 path stays silent).
        requests, _ = _simulate_gate(self.gate, self._connected(None), True)
        self.assertEqual(requests, [
            ["MODEL ACTIVE", "IMAGE ACTIVE", "LOG ACTIVE"],
            ["MODEL ACTIVE", "DIAGNOSTIC ACTIVE", "VISUAL ACTIVE"],
            ["MODEL ACTIVE"],
        ])

    def test_upscale_save_boundary_interface_is_unchanged(self):
        """H-4 is closed: the Upscale & Save boundary keeps its recorded ports."""
        boundary = boot(self.root).cmk_module("pipe.cmk_module_boundary_cache").CMKUpscaleSaveBoundaryCache
        self.assertEqual(json.loads(json.dumps(boundary.INPUT_TYPES())), {
            "optional": {"MODEL (opt)": ["CMK_MODEL_PIPE", {"lazy": True}], "IMAGE": ["IMAGE", {"lazy": True}],
                         "LOG": ["CMK_LOG_PIPE", {"lazy": True}], "diagnostic": ["CMK_DIAGNOSTIC", {"lazy": True}]},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        })
        self.assertEqual(list(boundary.RETURN_TYPES), ["CMK_MODEL_PIPE", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC"])
        self.assertEqual(list(boundary.RETURN_NAMES), ["MODEL", "IMAGE", "LOG", "diagnostic"])
        self.assertEqual((boundary.FUNCTION, boundary._SCOPE, boundary._SCHEMA),
                         ("boundary", "upscale_save_boundary", "cmk_upscale_save_boundary_v1"))


if __name__ == "__main__":
    unittest.main()
