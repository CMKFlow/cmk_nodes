"""Inactive CMK gates are decoupled before ComfyUI forms cache keys.

ComfyUI's cache signature contains every linked ancestor, also the ACTIVE path
of a lazy gate that selects BYPASS.  CMK's on_prompt handler
(cmk_prompt_gate_pruning) removes the ``* ACTIVE`` links of gates whose
decision the prompt already holds (cmk_distribution.gate_prompt_state):

* literal ENABLE False, or STOP AFTER HANDOFF for module gates;
* the InstantID gate when module 01 explicitly disables InstantID, selects a
  non-SDXL family, or STOP AFTER HANDOFF is effective.

Any other state is "unknown" and nothing is removed.  The removed links stay
in ``_meta`` so that a prompt replayed from /history gets them back.

Fixtures: ``full_flow`` is the shipped CMK 2.6 Full Flow as the real frontend
expands it; every identity consumer is fed a literal reference, as the Multi
Image Loader delivers it.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from harness.support import ScenarioCase, work_calls

CMK_TEST_TIER = "T1"
TESTS_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = TESTS_ROOT.parent
FIXTURES = TESTS_ROOT / "fixtures"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import cmk_distribution as distribution  # noqa: E402
import cmk_prompt_gate_pruning as pruning  # noqa: E402

START = "8332"
INSTANTID_GATE = "8465:8405"
FACEREBUILD_GATE, FACEREBUILD_ADV_GATE = "8435:7827", "8440:8387"
FACESWAP_GATE, FACESWAP_ADV_GATE = "8436:7856", "8441:7507"
MODULE_GATES_OFF_IN_FIXTURE = {"8440:8387", "8441:7507", "8442:7847", "8443:8399", "8444:8373",
                               "8445:7709", "8446:7887"}
OTHER = "CMK Package · face_identity_reference.png"


def full_flow(**start):
    prompt = json.loads((FIXTURES / "api_prompts" / "full_flow.json").read_text(encoding="utf-8"))
    prompt[START]["inputs"].update(start)
    return prompt


def pruned_gates(prompt):
    return {node_id for node_id, node in prompt.items() if pruning.PRUNED_META_KEY in (node.get("_meta") or {})}


class GateDecisions(unittest.TestCase):
    """cmk_distribution.gate_prompt_state on the expanded Full Flow."""

    def state(self, prompt, gate):
        return distribution.gate_prompt_state(prompt, gate)

    def test_module_gate_follows_literal_enable(self):
        prompt = full_flow()
        self.assertEqual(self.state(prompt, FACESWAP_GATE), distribution.GATE_ACTIVE)
        self.assertEqual(self.state(prompt, FACESWAP_ADV_GATE), distribution.GATE_INACTIVE)

    def test_module_gate_is_inactive_while_stop_after_handoff_is_effective(self):
        prompt = full_flow(instantid_enabled=True, stop_after_handoff=True)
        self.assertEqual(self.state(prompt, FACESWAP_GATE), distribution.GATE_INACTIVE)

    def test_instantid_gate_follows_module_01(self):
        self.assertEqual(self.state(full_flow(instantid_enabled=True), INSTANTID_GATE), distribution.GATE_ACTIVE)
        self.assertEqual(self.state(full_flow(instantid_enabled=False), INSTANTID_GATE), distribution.GATE_INACTIVE)
        self.assertEqual(self.state(full_flow(instantid_enabled=True, stop_after_handoff=True), INSTANTID_GATE),
                         distribution.GATE_INACTIVE)
        for family in ("Z-Image Turbo", "Hybrid"):
            self.assertEqual(self.state(full_flow(instantid_enabled=True, model_family=family), INSTANTID_GATE),
                             distribution.GATE_INACTIVE, family)

    def test_instantid_gate_matches_the_runtime_flag(self):
        """The decision equals CMKProcessEnableFlag.read on the PROCESS that
        CMKPipeCreateImage builds from the same widgets."""
        for enabled in (False, True):
            for stop in (False, True):
                for family in ("SDXL", "Z-Image Turbo", "Hybrid"):
                    inputs = full_flow(instantid_enabled=enabled, stop_after_handoff=stop,
                                       model_family=family)[START]["inputs"]
                    family_key = distribution._start_model_family(family)
                    process = {
                        "instantid_enabled": bool(enabled and family_key == "sdxl"),
                        "stop_after_handoff": stop,
                        "instantid_mode": distribution.start_instantid_mode(inputs["INPAINT_MODE"],
                                                                            inputs["instantid_mode"]),
                        "model_family": family_key,
                        "hybrid_mode": family_key == "hybrid",
                    }
                    runtime_enable = process["instantid_enabled"] and not distribution.stop_after_handoff_active(process)
                    expected = distribution.GATE_ACTIVE if runtime_enable else distribution.GATE_INACTIVE
                    prompt = full_flow(instantid_enabled=enabled, stop_after_handoff=stop, model_family=family)
                    self.assertEqual(self.state(prompt, INSTANTID_GATE), expected, (enabled, stop, family))

    def test_undecidable_states_are_unknown(self):
        cases = {
            "enable from another node": lambda p: p[FACESWAP_GATE]["inputs"].__setitem__("ENABLE", ["8439", 0]),
            "flag reads another key": lambda p: p["8465:8404"]["inputs"].__setitem__("flag", "controlnet_enabled"),
            "instantid widget linked": lambda p: p[START]["inputs"].__setitem__("instantid_enabled", ["8439", 0]),
            "model family missing": lambda p: p[START]["inputs"].pop("model_family"),
            "stop widget linked": lambda p: p[START]["inputs"].__setitem__("stop_after_handoff", ["8439", 0]),
            "second module 01": lambda p: p.__setitem__("9999", copy.deepcopy(p[START])) or
            p["8465:8404"]["inputs"].__setitem__("PROCESS", ["9999", 0]),
        }
        for label, change in cases.items():
            prompt = full_flow(instantid_enabled=True)
            change(prompt)
            gate = FACESWAP_GATE if label == "enable from another node" else INSTANTID_GATE
            self.assertEqual(self.state(prompt, gate), distribution.GATE_UNKNOWN, label)
            self.assertNotIn(gate, pruned_gates(pruning.prune_inactive_gate_links(prompt)[0]), label)

    def test_other_gate_classes_are_never_decided(self):
        prompt = full_flow()
        for node_id, node in prompt.items():
            if node["class_type"] not in distribution.PRUNABLE_GATE_CLASSES:
                self.assertEqual(self.state(prompt, node_id), distribution.GATE_UNKNOWN, node["class_type"])


class Pruning(unittest.TestCase):
    """prune_inactive_gate_links and the on_prompt handler, without execution."""

    def test_only_inactive_gates_lose_their_active_links(self):
        raw = full_flow(instantid_enabled=False)
        prompt, removed = pruning.prune_inactive_gate_links(raw)
        self.assertEqual(set(removed), MODULE_GATES_OFF_IN_FIXTURE | {INSTANTID_GATE})
        for gate, names in removed.items():
            self.assertTrue(names and all(name.endswith(" ACTIVE") for name in names))
            self.assertFalse(any(name.endswith(" ACTIVE") for name in prompt[gate]["inputs"]))
            for name in ("IMAGE BYPASS", "SAMPLED BYPASS"):
                if name in raw[gate]["inputs"]:
                    self.assertEqual(prompt[gate]["inputs"][name], raw[gate]["inputs"][name])
        for node_id in raw:
            if node_id not in removed:
                self.assertEqual(prompt[node_id], raw[node_id], node_id)

    def test_active_gates_keep_every_link(self):
        raw = full_flow(instantid_enabled=True)
        prompt, removed = pruning.prune_inactive_gate_links(raw)
        self.assertNotIn(INSTANTID_GATE, removed)
        self.assertNotIn(FACESWAP_GATE, removed)
        self.assertEqual(prompt[INSTANTID_GATE], raw[INSTANTID_GATE])
        self.assertEqual(prompt[FACESWAP_GATE], raw[FACESWAP_GATE])

    def test_input_prompt_is_not_modified(self):
        raw = full_flow(instantid_enabled=False)
        before = copy.deepcopy(raw)
        pruning.prune_inactive_gate_links(raw)
        pruning.on_prompt({"prompt": raw})
        self.assertEqual(raw, before)

    def test_idempotent(self):
        once = pruning.on_prompt({"prompt": full_flow(instantid_enabled=False)})["prompt"]
        self.assertEqual(pruning.on_prompt({"prompt": once})["prompt"], once)

    def test_restoring_the_stored_links_returns_the_original_prompt(self):
        raw = full_flow(instantid_enabled=False)
        once = pruning.on_prompt({"prompt": raw})["prompt"]
        self.assertEqual(pruning.restore_pruned_links(copy.deepcopy(once)), raw)

    def test_reactivated_replay_gets_its_links_back(self):
        """A pruned prompt from /history with the gate switched on again."""
        replay = pruning.on_prompt({"prompt": full_flow(instantid_enabled=False)})["prompt"]
        replay[START]["inputs"]["instantid_enabled"] = True
        replay[FACESWAP_ADV_GATE]["inputs"]["ENABLE"] = True
        submitted = pruning.on_prompt({"prompt": replay})["prompt"]
        reference = full_flow(instantid_enabled=True)
        reference[FACESWAP_ADV_GATE]["inputs"]["ENABLE"] = True
        for gate in (INSTANTID_GATE, FACESWAP_ADV_GATE):
            self.assertEqual(submitted[gate]["inputs"], reference[gate]["inputs"], gate)
            self.assertNotIn(pruning.PRUNED_META_KEY, submitted[gate].get("_meta") or {})

    def test_switch_off_disables_pruning_but_still_restores(self):
        raw = full_flow(instantid_enabled=False)
        once = pruning.on_prompt({"prompt": raw})["prompt"]
        with mock.patch.dict(os.environ, {pruning.ENVIRONMENT_SWITCH: "0"}):
            self.assertIs(pruning.on_prompt({"prompt": raw})["prompt"], raw)
            self.assertEqual(pruning.on_prompt({"prompt": once})["prompt"], raw)

    def test_pruning_is_on_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(pruning.ENVIRONMENT_SWITCH, None)
            self.assertTrue(pruning.pruning_enabled())

    def test_a_failure_leaves_the_request_unchanged(self):
        request = {"prompt": full_flow(instantid_enabled=False), "client_id": "x"}
        with mock.patch.object(pruning, "gate_prompt_state", side_effect=RuntimeError("boom")):
            self.assertIs(pruning.on_prompt(request), request)

    def test_non_cmk_prompts_are_returned_unchanged(self):
        request = {"prompt": {"1": {"class_type": "KSampler", "inputs": {"seed": 1, "model": ["2", 0]}},
                              "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x"}}}}
        self.assertIs(pruning.on_prompt(request), request)
        for value in (None, {}, {"prompt": []}, []):
            self.assertIs(pruning.on_prompt(value), value)

    def test_pruning_that_would_change_a_decision_is_skipped(self):
        """STOP read through an ACTIVE path: after pruning, module 01 would no
        longer be upstream and the gate would turn active."""
        start = full_flow(instantid_enabled=True, stop_after_handoff=True)[START]
        prompt = {
            "1": start,
            "2": {"class_type": "CMKProcessForwardPipe", "inputs": {"PROCESS": ["1", 3]}},
            "3": {"class_type": "CMKImageFileLoader", "inputs": {"image": "x.png"}},
            "4": {"class_type": "CMKModuleBypassGate", "inputs": {
                "ENABLE": True, "IMAGE ACTIVE": ["2", 0], "IMAGE BYPASS": ["3", 0], "LOG BYPASS": ["3", 0]}},
        }
        self.assertEqual(distribution.gate_prompt_state(prompt, "4"), distribution.GATE_INACTIVE)
        result, removed = pruning.prune_inactive_gate_links(prompt)
        self.assertEqual(removed, {})
        self.assertEqual(result, prompt)

    def test_multi_image_loader_references_pass_unchanged(self):
        for path in sorted((FIXTURES / "multi_image_loader").glob("*.json")):
            for enabled in (True, False):
                raw = json.loads(path.read_text(encoding="utf-8"))
                for node in raw.values():
                    if node["class_type"] == "CMKPipeCreateImage":
                        node["inputs"]["instantid_enabled"] = enabled
                submitted = pruning.on_prompt({"prompt": raw})["prompt"]
                for node_id, node in raw.items():
                    if node["class_type"] in {"CMKLoadImage", "CMKImageFileLoader"}:
                        self.assertEqual(submitted[node_id], node, (path.name, node_id))

    def test_registration_is_single_and_tolerates_servers_without_handlers(self):
        class Server:
            def __init__(self):
                self.handlers = []

            def add_on_prompt_handler(self, handler):
                self.handlers.append(handler)

        server = Server()
        self.assertTrue(pruning.register(server))
        self.assertFalse(pruning.register(server))
        self.assertEqual(server.handlers, [pruning.on_prompt])
        self.assertFalse(pruning.register(object()))


CASES = {
    "iid_off": {"instantid": False},
    "iid_off_ref": {"instantid": False, "references": {"instantid": OTHER}},
    "iid_on": {"instantid": True},
    "iid_on_ref": {"instantid": True, "references": {"instantid": OTHER}},
    "iid_off_again": {"instantid": False},
    "stop": {"instantid": True, "stop": True},
    "stop_ref": {"instantid": True, "stop": True, "references": {"instantid": OTHER}},
    "fr_ref": {"references": {"facerebuild": OTHER}},
    "fr_widget": {"overrides": {"8435:7823.IDENTITY STRENGTH": 0.5}},
    "fr_adv_ref": {"references": {"facerebuild_adv": OTHER}},
    "fs_adv_ref": {"references": {"faceswap_adv": OTHER}},
    "fs_ref": {"references": {"faceswap": OTHER}},
    "fr_on": {"facerebuild": True},
    "fr_on_ref": {"facerebuild": True, "references": {"facerebuild": OTHER}},
}


class CacheKeys(ScenarioCase):
    """ComfyUI signatures and CMK fingerprints (refiner boundary, module
    boundaries, generation completion), raw and after the handler."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.keys = cls.run_case("gate_pruning_keys", cases=CASES)

    def changed(self, base, variant, kind="pruned"):
        a, b = self.keys[base], self.keys[variant]
        skip = set(a["unstable"]) | set(b["unstable"])
        comfy = {n for n in a[kind]["comfy"] if n not in skip and a[kind]["comfy"][n] != b[kind]["comfy"][n]}
        cmk = {n for n in a[kind]["cmk"] if a[kind]["cmk"][n] != b[kind]["cmk"][n]}
        return comfy, cmk

    def outside(self, nodes, *modules):
        return {n for n in nodes if n.split(":")[0] not in modules}

    def test_instantid_off_reference_change_keeps_refiner_and_successors(self):
        comfy, cmk = self.changed("iid_off", "iid_off_ref")
        self.assertEqual(self.outside(comfy, "8465"), set())
        self.assertEqual(self.outside(cmk, "8465"), set())
        # Without the handler the same change reached the refiner and every module.
        raw_comfy, raw_cmk = self.changed("iid_off", "iid_off_ref", "raw")
        self.assertIn("8433:7809", raw_comfy)
        self.assertIn("refiner", raw_cmk)

    def test_instantid_on_reference_change_invalidates(self):
        comfy, cmk = self.changed("iid_on", "iid_on_ref")
        self.assertTrue({"8465:8402", "8465:8405", "8433:7809", "8433:7810", "8436:7849"} <= comfy)
        self.assertTrue({"refiner", "8433:7810", "8436:7851"} <= cmk)
        self.assertEqual((comfy, cmk), self.changed("iid_on", "iid_on_ref", "raw"))

    def test_active_instantid_keeps_its_raw_keys(self):
        on = self.keys["iid_on"]
        self.assertNotIn(INSTANTID_GATE, on["pruned_gates"])
        for node in ("8465:8405", "8433:7809", "8433:7810"):
            self.assertEqual(on["pruned"]["comfy"][node], on["raw"]["comfy"][node], node)
        self.assertEqual(on["pruned"]["cmk"]["refiner"], on["raw"]["cmk"]["refiner"])

    def test_generation_completion_key_survives_an_unused_reference(self):
        # The SDXL completion applies while InstantID is off (pure SDXL).
        off, changed = self.keys["iid_off"]["pruned"]["cmk"], self.keys["iid_off_ref"]["pruned"]["cmk"]
        self.assertFalse(off["completion"].startswith("none"), off["completion"])
        self.assertEqual(off["completion"], changed["completion"])
        raw_off, raw_changed = self.keys["iid_off"]["raw"]["cmk"], self.keys["iid_off_ref"]["raw"]["cmk"]
        self.assertNotEqual(raw_off["completion"], raw_changed["completion"])

    def test_off_on_off_returns_to_the_same_keys(self):
        off, on, again = (self.keys[label]["pruned"] for label in ("iid_off", "iid_on", "iid_off_again"))
        self.assertEqual(off["cmk"], again["cmk"])
        self.assertNotEqual(off["cmk"]["refiner"], on["cmk"]["refiner"])
        self.assertNotEqual(off["comfy"]["8433:7809"], on["comfy"]["8433:7809"])

    def test_stop_after_handoff_decouples_only_inactive_branches(self):
        stop = self.keys["stop"]
        self.assertIn(INSTANTID_GATE, stop["pruned_gates"])
        self.assertIn(FACESWAP_GATE, stop["pruned_gates"])
        comfy, cmk = self.changed("stop", "stop_ref")
        self.assertEqual(self.outside(comfy, "8465"), set())
        self.assertEqual(self.outside(cmk, "8465"), set())

    def test_disabled_modules_do_not_invalidate_active_successors(self):
        for variant, module in (("fr_ref", "8435"), ("fr_widget", "8435"), ("fr_adv_ref", "8440"),
                                ("fs_adv_ref", "8441")):
            comfy, cmk = self.changed("iid_off", variant)
            self.assertEqual(self.outside(comfy, module), set(), variant)
            self.assertEqual(self.outside(cmk, module), set(), variant)
            raw_comfy, _ = self.changed("iid_off", variant, "raw")
            if module != "8441":  # FaceSwap Advanced is the last module of the flow
                self.assertIn("8436:7849", raw_comfy, variant)

    def test_active_modules_invalidate_themselves_and_successors(self):
        comfy, cmk = self.changed("iid_off", "fs_ref")
        self.assertTrue({"8436:7849", "8436:7851", "8441:7507"} <= comfy)
        self.assertIn("8436:7851", cmk)
        comfy, cmk = self.changed("fr_on", "fr_on_ref")
        self.assertTrue({"8435:7823", "8435:7824", "8436:7849"} <= comfy)
        self.assertTrue({"8435:7824", "8436:7851"} <= cmk)

    def test_several_inactive_gates_in_series(self):
        self.assertTrue(MODULE_GATES_OFF_IN_FIXTURE <= set(self.keys["iid_off"]["pruned_gates"]))

    def test_replay_and_active_gates(self):
        for label, record in self.keys.items():
            self.assertTrue(record["idempotent"], label)
            self.assertTrue(record["active_inputs_kept"], label)


class Execution(ScenarioCase):
    """Model-free runs with and without the handler give the same result."""

    COMPARED = ("succeeded", "errors", "visual", "new_file_hashes", "outputs")

    def assert_same(self, raw, pruned):
        self.assertEqual(len(raw["runs"]), len(pruned["runs"]))
        for index, (a, b) in enumerate(zip(raw["runs"], pruned["runs"])):
            for key in self.COMPARED:
                self.assertEqual(a[key], b[key], (index, key))
            self.assertEqual(work_calls(a), work_calls(b), index)

    def test_postprocess_modules_off_on_off(self):
        cases = (
            ("faceswap_std", "8417:7856", "CMKFaceSwapImagePipe", {}),
            ("facerebuild_std", "8417:7827", "CMKInstantIDFaceRebuildSDXL",
             {"fake_postprocess": True, "upstream_visual": True}),
            ("chain_postprocess", "8427:7847", "CMKFaceProcessPipe", {}),
        )
        for fixture, gate, compute, extra in cases:
            overrides = [{f"{gate}.ENABLE": False}, {f"{gate}.ENABLE": True}, {f"{gate}.ENABLE": False}]
            runs = {flag: self.run_case("run_fixture", fixture=fixture, runs=3, run_overrides=overrides,
                                        gate_pruning=flag, inspect={"image": f"{gate}:1"}, **extra)
                    for flag in (False, True)}
            self.assert_same(runs[False], runs[True])
            pruned = runs[True]["runs"]
            self.assertEqual([run["pruned_gates"] for run in pruned], [[gate], [], [gate]], fixture)
            if fixture != "chain_postprocess":  # the chain stops at the Detailer's model seam
                self.assertTrue(all(run["succeeded"] for run in pruned), (fixture, [r["errors"] for r in pruned]))
                self.assertEqual([run["function_calls"].get(compute, 0) for run in pruned], [0, 1, 0], fixture)

    def test_faceswap_reactivation_computes_and_final_off_does_not(self):
        overrides = [{"8417:7856.ENABLE": False}, {"8417:7856.ENABLE": True}, {"8417:7856.ENABLE": False}]
        runs = self.run_case("run_fixture", fixture="faceswap_std", runs=3, run_overrides=overrides,
                             gate_pruning=True)["runs"]
        self.assertTrue(all(run["succeeded"] for run in runs), [run["errors"] for run in runs])
        work = [run["function_calls"].get("CMKFaceSwapImagePipe", 0) for run in runs]
        self.assertEqual(work, [0, 1, 0])


INSTANTID_FIXTURE = "instantid_inpaint"


class InstantIDExecution(ScenarioCase):
    """InstantID 15S with the generation seams: OFF, ON, STOP, OFF, raw and pruned."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        kwargs = dict(fixture=INSTANTID_FIXTURE, stops=[False, False, True, False, True],
                      start_inputs=[{"instantid_enabled": False}, {"instantid_enabled": True},
                                    {"instantid_enabled": True}, {"instantid_enabled": False},
                                    {"instantid_enabled": False}],
                      inspect={"gate_image": "8433:7812:1"})
        cls.raw = cls.run_case("hybrid_stop_sequence", gate_pruning=False, **kwargs)
        cls.pruned = cls.run_case("hybrid_stop_sequence", gate_pruning=True, **kwargs)

    KEYS = ("succeeded", "errors", "sampling", "public_image_png", "instantid_latents", "final_previews",
            "outputs", "sdxl_gate")

    def test_every_run_succeeds(self):
        for record in (self.raw, self.pruned):
            for run in record["runs"]:
                self.assertTrue(run["succeeded"], run["errors"])

    def test_results_are_identical(self):
        for index, (a, b) in enumerate(zip(self.raw["runs"], self.pruned["runs"])):
            for key in self.KEYS:
                self.assertEqual(a[key], b[key], (index, key))

    def test_instantid_runs_only_while_active(self):
        self.assertEqual([run["sampling"]["instantid"] for run in self.pruned["runs"]], [0, 1, 0, 0, 0])

    def test_inactive_runs_are_pruned(self):
        gates = [run["pruned_gates"] for run in self.pruned["runs"]]
        self.assertEqual(["8438:7918" in item for item in gates], [True, False, True, True, True])


class IncompleteReplay(ScenarioCase):
    """A replayed prompt whose ACTIVE links are gone (no stored links) and whose
    gate is switched on fails loudly instead of producing a bypass image."""

    def test_module_gate(self):
        run = self.run_case("run_fixture", fixture="faceswap_std", gate_pruning=True,
                            remove_inputs=[f"8417:7856.{name} ACTIVE" for name in
                                           ("MODEL", "IMAGE", "LOG", "DIAGNOSTIC", "VISUAL")])["runs"][0]
        self.assertFalse(run["succeeded"])
        self.assertIn("CMK Module Bypass Gate is missing the ACTIVE result", json.dumps(run["errors"]))
        self.assertEqual(run["new_file_hashes"], {})

    def test_instantid_gate(self):
        run = self.run_case("hybrid_stop_sequence", fixture=INSTANTID_FIXTURE, stops=[False], gate_pruning=True,
                            start_inputs=[{"instantid_enabled": True}],
                            remove_inputs=[f"8438:7918.{name} ACTIVE" for name in
                                           ("MODEL", "SAMPLED", "LOG", "DIAGNOSTIC", "VISUAL")])["runs"][0]
        self.assertFalse(run["succeeded"])
        self.assertEqual(run["sampling"]["refiner"], 0)


if __name__ == "__main__":
    unittest.main()
