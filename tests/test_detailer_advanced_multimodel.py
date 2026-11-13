"""2.8.0 Advanced Multi-Model Detailer: contracts of the deferred Prepare, the
family-ordered Merge, the Smart Detailer hook and the boundary/cache rules.

Executor behaviour (ZIT before SDXL, one materialisation per family, release
before SDXL, branch-cache independence per slot) is proven with the
PromptExecutor harness; this file pins the code contracts behind it.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_detailer_advanced_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import torch  # noqa: E402

advanced = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_detailer_advanced_multimodel")
mm_mod = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_detailer_prepare_multimodel")
smart = importlib.import_module(f"{TEST_PACKAGE}.nodes.image.smart_detailer")
boundary = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_module_boundary_cache")
mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")

CMK_TEST_TIER = "T0"
PREPARE = advanced.CMKDetailerPrepareAdvancedMultiModelPipe
MERGE = advanced.CMKDetailerMergeMultiModel
SDXL_SPEC = {"model_role": "postprocess", "model_family": "sdxl", "ckpt_name": "ck.safetensors", "materialized": False}
ZIT_SPEC = {"model_role": "postprocess", "model_family": "z_image_turbo", "diffusion_model_name": "z.safetensors",
            "text_encoder_name": "q.safetensors", "vae_name": "ae.safetensors", "weight_dtype": "default",
            "clip_device": "default", "materialized": False}
PROCESS = {"result_contract": "family_neutral", "seed": 5, "prompt_pos": "first pass"}


def prepare_inputs(families, **overrides):
    values = {
        "sam_model_name": "sam_vit_b_01ec64.pth", "sam_device_mode": "CPU", "use_prompt_lora_from_sampler": False,
        "stop_at_clip_layer": -2, "pag_scale": 2.5, "sampling": "v_prediction", "zsnr": True, "freeu_enabled": True,
        "freeu_b1": 1.3, "freeu_b2": 1.4, "freeu_s1": 0.9, "freeu_s2": 0.2,
        "MODEL SDXL": SDXL_SPEC, "MODEL ZIT": ZIT_SPEC,
    }
    for slot, family in zip((1, 2, 3), families):
        values.update({
            f"model_family_{slot}": family, f"steps_{slot}": 30 + slot, f"cfg_{slot}": 7.0, f"sampling_entry_{slot}": 95.0,
            f"sampler_{slot}": "euler", f"scheduler_{slot}": "karras", f"zit_steps_{slot}": 8 + slot, f"zit_cfg_{slot}": 1.0,
            f"zit_sampling_entry_{slot}": 85.0, f"zit_sampler_{slot}": "res_multistep", f"zit_scheduler_{slot}": "simple",
            f"zit_model_shift_{slot}": 3.0,
        })
    values.update(overrides)
    return values


def contains_model(value, seen=None):
    seen = seen if seen is not None else set()
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, dict):
        if value.get("materialized") is True or any(key in value for key in ("model", "detailer_model", "detailer_vae")):
            return value.get("model") is not None or value.get("detailer_model") is not None or value.get("detailer_vae") is not None
        return any(contains_model(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_model(v, seen) for v in value)
    return False


class AdvancedPrepare(unittest.TestCase):
    def run_prepare(self, families, **overrides):
        return PREPARE().prepare(PROCESS=dict(PROCESS), IMAGE=torch.zeros(1, 16, 16, 3), LOG=None,
                                 **prepare_inputs(families, **overrides))

    def test_registration_and_outputs(self):
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKDetailerPrepareAdvancedMultiModelPipe"], PREPARE)
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKDetailerMergeMultiModel"], MERGE)
        self.assertEqual(PREPARE.RETURN_NAMES, ("DETAILER 1", "DETAILER 2", "DETAILER 3", "ENABLE", "LOG", "diagnostic"))
        required = PREPARE.INPUT_TYPES()["required"]
        for slot in (1, 2, 3):
            self.assertEqual(required[f"model_family_{slot}"][0], ["None", "SDXL", "Z-Image Turbo"])
            for name in ("steps", "cfg", "sampler", "scheduler", "sampling_entry", "zit_steps", "zit_model_shift"):
                self.assertIn(f"{name}_{slot}", required)
        self.assertNotIn("prompt_pos", required, "promptless")
        self.assertNotIn("lora_name", required, "no individual LoRA in v1")

    def test_pipes_are_deferred_and_hold_no_model(self):
        d1, d2, d3, enable, _log, _diag = self.run_prepare(("Z-Image Turbo", "SDXL", "None"))
        self.assertTrue(enable)
        self.assertFalse(contains_model([d1, d2, d3]))
        for pipe, family in ((d1, "z_image_turbo"), (d2, "sdxl")):
            self.assertTrue(pipe["detailer_model_deferred"])
            self.assertEqual(pipe["detailer_model_family"], family)
            self.assertNotIn("detailer_conditioning_pos", pipe)
            self.assertNotIn("detailer_sam_model", pipe)
        arguments = d1["detailer_prepare_arguments"]
        self.assertEqual((arguments["model_family"], arguments["prompt_pos"], arguments["lora_name"]),
                         ("Z-Image Turbo", "", "None"))
        self.assertEqual((arguments["zit_steps"], d1["detailer_steps"]), (9, 9))
        self.assertEqual((d2["detailer_prepare_arguments"]["steps"], d2["detailer_steps"]), (32, 32))
        self.assertFalse(d3["detailer_global_enable"])
        self.assertNotIn("detailer_model_deferred", d3)

    def test_all_none_is_disabled(self):
        *_pipes, enable, _log, _diag = self.run_prepare(("None", "None", "None"))
        self.assertFalse(enable)

    def test_lazy_requests_only_needed_model_sources(self):
        node = PREPARE()
        zit_only = prepare_inputs(("Z-Image Turbo", "None", "None"), **{"MODEL SDXL": None, "MODEL ZIT": None})
        self.assertEqual(node.check_lazy_status(**zit_only), ["MODEL ZIT"])
        sdxl_only = prepare_inputs(("SDXL", "None", "None"), **{"MODEL SDXL": None, "MODEL ZIT": None})
        self.assertEqual(node.check_lazy_status(**sdxl_only), ["MODEL SDXL"])
        none = prepare_inputs(("None", "None", "None"), **{"MODEL SDXL": None, "MODEL ZIT": None})
        self.assertEqual(node.check_lazy_status(**none), [])

    def test_global_disable_bypasses_without_losing_settings(self):
        families = ("Z-Image Turbo", "SDXL", "SDXL")
        inputs = prepare_inputs(families, detailer_global_enable="DISABLE")
        kept = dict(inputs)
        d1, d2, d3, enable, log, diag = PREPARE().prepare(PROCESS=dict(PROCESS), IMAGE=torch.zeros(1, 16, 16, 3), LOG=None, **inputs)
        self.assertFalse(enable, "gate and preview get the module switch")
        for pipe in (d1, d2, d3):
            self.assertTrue(pipe["detailer_prepare_bypassed"])
            self.assertNotIn("detailer_prepare_arguments", pipe)
            self.assertFalse(pipe["detailer_global_enable"])
        self.assertEqual(inputs, kept, "Detailer settings are untouched")
        self.assertIn("GLOBAL          : DISABLE · MODULE BYPASS", log["blocks"][-1]["lines"])
        self.assertIn("DETAILER 2      : SDXL · GLOBAL DISABLE", log["blocks"][-1]["lines"])
        # No model source is requested while the module is off.
        lazy = prepare_inputs(families, detailer_global_enable="DISABLE", **{"MODEL SDXL": None, "MODEL ZIT": None})
        self.assertEqual(PREPARE().check_lazy_status(**lazy), [])
        # ENABLE again: the same settings produce the same pipes as before.
        on = self.run_prepare(families, detailer_global_enable="ENABLE")
        default = self.run_prepare(families)
        self.assertTrue(on[3])
        for a, b in zip(on[:3], default[:3]):
            self.assertEqual({k: v for k, v in a.items() if k not in ("detailer_image", "source_pipe", "_source_pipe", "detailer_prepare_arguments")},
                             {k: v for k, v in b.items() if k not in ("detailer_image", "source_pipe", "_source_pipe", "detailer_prepare_arguments")})
        self.assertEqual(on[0]["detailer_prepare_arguments"]["zit_steps"], 9)

    def test_global_switch_is_the_last_widget(self):
        required = list(PREPARE.INPUT_TYPES()["required"])
        self.assertEqual(required[-1], "detailer_global_enable")
        self.assertEqual(PREPARE.INPUT_TYPES()["required"]["detailer_global_enable"], (["ENABLE", "DISABLE"], {"default": "ENABLE"}))

    def test_slot_exclusions(self):
        inputs = prepare_inputs(("Z-Image Turbo", "SDXL", "None"))
        excluded = set(advanced.slot_excluded_inputs(inputs, 1))
        self.assertTrue({"model_family_2", "zit_steps_2", "steps_3", "steps_1", "MODEL SDXL", "LOG"} <= excluded)
        self.assertFalse({"zit_steps_1", "model_family_1", "MODEL ZIT", "sam_model_name"} & excluded)
        excluded = set(advanced.slot_excluded_inputs(inputs, 2))
        self.assertIn("zit_steps_2", excluded)
        self.assertNotIn("steps_2", excluded)
        self.assertNotIn("pag_scale", excluded)


def merge_prompt(families):
    prompt = {"p": {"class_type": "CMKDetailerPrepareAdvancedMultiModelPipe",
                    "inputs": {f"model_family_{slot}": family for slot, family in zip((1, 2, 3), families)}},
              "m": {"class_type": "CMKDetailerMergeMultiModel", "inputs": {}}}
    for slot, node_id in ((1, "d1"), (2, "d2"), (3, "d3")):
        prompt[node_id] = {"class_type": "CMK_SmartDetailerMultiModelPipe", "inputs": {"DETAILER": ["p", slot - 1]}}
        prompt["m"]["inputs"].update({f"SEGS {slot}": [node_id, 1], f"LOG {slot}": [node_id, 3], f"diagnostic {slot}": [node_id, 5]})
    return prompt


def missing_kwargs():
    return {f"{name} {slot}": None for slot in (1, 2, 3) for name in ("SEGS", "LOG", "diagnostic")}


class Merge(unittest.TestCase):
    def test_families_and_request_order(self):
        prompt = merge_prompt(("Z-Image Turbo", "SDXL", "Z-Image Turbo"))
        self.assertEqual(MERGE.slot_families(prompt, "m"), {1: "z_image_turbo", 2: "sdxl", 3: "z_image_turbo"})
        node = MERGE()
        # Plan B: a slot diagnostic joins the request of its own family group,
        # so the single Detailer previews behind the merge keep ZIT before SDXL.
        kwargs = missing_kwargs()
        self.assertEqual(node.check_lazy_status(prompt=prompt, unique_id="m", **kwargs),
                         ["SEGS 1", "LOG 1", "diagnostic 1", "SEGS 3", "LOG 3", "diagnostic 3"])
        for slot in (1, 3):
            for name in ("SEGS", "LOG", "diagnostic"):
                kwargs[f"{name} {slot}"] = object()
        self.assertEqual(node.check_lazy_status(prompt=prompt, unique_id="m", **kwargs),
                         ["SEGS 2", "LOG 2", "diagnostic 2"])
        kwargs.update({"SEGS 2": object(), "LOG 2": object(), "diagnostic 2": object()})
        self.assertEqual(node.check_lazy_status(prompt=prompt, unique_id="m", **kwargs), [])

    def test_none_slots_request_only_their_diagnostic_last(self):
        prompt = merge_prompt(("SDXL", "None", "None"))
        kwargs = missing_kwargs()
        self.assertEqual(MERGE().check_lazy_status(prompt=prompt, unique_id="m", **kwargs),
                         ["SEGS 1", "LOG 1", "diagnostic 1"])
        kwargs.update({"SEGS 1": object(), "LOG 1": object(), "diagnostic 1": object()})
        self.assertEqual(MERGE().check_lazy_status(prompt=prompt, unique_id="m", **kwargs),
                         ["diagnostic 2", "diagnostic 3"], "never SEGS or LOG of a None slot")
        prompt = merge_prompt(("None", "None", "None"))
        self.assertEqual(MERGE().check_lazy_status(prompt=prompt, unique_id="m", **missing_kwargs()),
                         ["diagnostic 1", "diagnostic 2", "diagnostic 3"])

    def test_slot_diagnostics_are_passed_on_unchanged(self):
        self.assertEqual(MERGE.RETURN_NAMES,
                         ("IMAGE", "LOG", "VISUAL", "diagnostic", "diagnostic 1", "diagnostic 2", "diagnostic 3"))
        self.assertEqual(MERGE.RETURN_TYPES[4:], ("CMK_DIAGNOSTIC",) * 3)
        prompt = merge_prompt(("Z-Image Turbo", "None", "SDXL"))
        image = torch.zeros(1, 16, 16, 3)
        empty = ((16, 16), [])
        d1, d2 = {"title": "D1"}, {"title": "D2 bypass"}
        result = MERGE().merge(IMAGE=image, prompt=prompt, unique_id="m",
                               **{"SEGS 1": empty, "diagnostic 1": d1, "diagnostic 2": d2, "SEGS 3": empty})
        self.assertEqual(len(result), 7)
        self.assertIs(result[4], d1)
        self.assertIs(result[5], d2)
        self.assertIsInstance(result[6], advanced.ExecutionBlocker, "a missing slot diagnostic blocks its preview")
        self.assertEqual(result[3]["title"], "CMK Flow · Detailer Advanced · Merge", "merge diagnostic unchanged")

    def test_slot_assignment_is_independent_of_arrival(self):
        prompt = merge_prompt(("Z-Image Turbo", "None", "SDXL"))
        image = torch.zeros(1, 16, 16, 3)
        base = {"blocks": [{"title": "Detailer Prepare Advanced", "order": 70, "lines": ["x"], "enabled": True}]}
        slot_log = lambda title: {"blocks": base["blocks"] + [{"title": title, "order": 70, "lines": ["y"], "enabled": True}]}
        empty = ((16, 16), [])
        _image, log, _visual, _diag, *_slots = MERGE().merge(
            IMAGE=image, LOG=base, prompt=prompt, unique_id="m",
            **{"SEGS 3": empty, "LOG 3": slot_log("Smart Detailer B"), "diagnostic 3": None,
               "SEGS 1": empty, "LOG 1": slot_log("Smart Detailer A"), "diagnostic 1": None})
        titles = [block["title"] for block in log["blocks"]]
        self.assertEqual(titles, ["Detailer Prepare Advanced", "Detailer 1 · Smart Detailer A", "Detailer 2",
                                  "Detailer 3 · Smart Detailer B"])
        self.assertIn("STATUS          : NONE · BYPASS", log["blocks"][2]["lines"])


class SmartDetailerHook(unittest.TestCase):
    def test_sdxl_only_detailer_returns_the_pipe_unchanged(self):
        pipe = {"detailer_model": "M"}
        self.assertIs(smart.CMK_SmartDetailerPipe._materialize_detailer(pipe), pipe)
        self.assertIs(smart.CMK_SmartDetailerMultiModelPipe()._materialize_detailer(pipe), pipe)

    def test_deferred_pipe_runs_the_multimodel_prepare(self):
        calls = []
        original = mm_mod.CMKDetailerPrepareMultiModelPipe.prepare
        self.addCleanup(setattr, mm_mod.CMKDetailerPrepareMultiModelPipe, "prepare", original)

        def fake_prepare(self, **kwargs):
            calls.append(kwargs)
            return ({"detailer_model_family": "z_image_turbo", "detailer_model": "M", "detailer_clip": None}, None, None)

        mm_mod.CMKDetailerPrepareMultiModelPipe.prepare = fake_prepare
        pipe = {"detailer_model_deferred": True, "detailer_prepare_arguments": {"model_family": "Z-Image Turbo"}}
        result = smart.CMK_SmartDetailerMultiModelPipe()._materialize_detailer(pipe, prompt={})
        self.assertEqual(result["detailer_model"], "M")
        self.assertEqual(result["detailer_clip"], smart.CMK_SmartDetailerMultiModelPipe._RELEASED_TEXT_ENCODER)
        self.assertIn(mm_mod.CONDITIONING_MEMO_ARGUMENT, calls[0])
        self.assertIsInstance(calls[0][mm_mod.CONDITIONING_MEMO_ARGUMENT], dict)

    def test_conditioning_memo_belongs_to_one_prompt(self):
        first = mm_mod.prompt_conditioning_memo({"a": 1})
        first["key"] = "value"
        prompt = {"b": 2}
        second = mm_mod.prompt_conditioning_memo(prompt)
        self.assertIs(mm_mod.prompt_conditioning_memo(prompt), second)


class CacheAndBoundary(unittest.TestCase):
    def full_prompt(self, families, **overrides):
        prompt = merge_prompt(families)
        prompt["p"]["inputs"] = prepare_inputs(families, **overrides)
        prompt["p"]["inputs"].pop("MODEL SDXL"); prompt["p"]["inputs"].pop("MODEL ZIT")
        prompt["b"] = {"class_type": "CMKDetailerBoundaryCache", "inputs": {"IMAGE": ["m", 0], "LOG": ["m", 1]}}
        for node_id in ("d1", "d2", "d3"):
            prompt[node_id]["inputs"].update({"enable": True, "model_name": "bbox/face.pt"})
        return prompt

    def test_branch_key_depends_only_on_its_slot(self):
        cls = smart.CMK_SmartDetailerMultiModelPipe
        base = self.full_prompt(("Z-Image Turbo", "SDXL", "Z-Image Turbo"))
        changed = self.full_prompt(("Z-Image Turbo", "SDXL", "Z-Image Turbo"), zit_steps_1=20, steps_2=40)
        keep = [base, changed]  # noqa: F841 (fingerprint memo is keyed by id(prompt))
        self.assertEqual(cls.dependency_cache_key(base, "d3")[0], cls.dependency_cache_key(changed, "d3")[0])
        self.assertNotEqual(cls.dependency_cache_key(base, "d1")[0], cls.dependency_cache_key(changed, "d1")[0])
        self.assertNotEqual(cls.dependency_cache_key(base, "d2")[0], cls.dependency_cache_key(changed, "d2")[0])

    def test_global_switch_does_not_change_branch_keys(self):
        cls = smart.CMK_SmartDetailerMultiModelPipe
        before = self.full_prompt(("Z-Image Turbo", "SDXL", "SDXL"))
        before["p"]["inputs"].pop("detailer_global_enable", None)
        after = self.full_prompt(("Z-Image Turbo", "SDXL", "SDXL"), detailer_global_enable="ENABLE")
        keep = [before, after]  # noqa: F841
        for node_id in ("d1", "d2", "d3"):
            self.assertEqual(cls.dependency_cache_key(before, node_id)[0], cls.dependency_cache_key(after, node_id)[0])

    def test_none_slot_reports_disabled_revision(self):
        prompt = self.full_prompt(("SDXL", "None", "Z-Image Turbo"))
        self.assertEqual(smart.CMK_SmartDetailerMultiModelPipe.dependency_revision(prompt, "d2", "k"),
                         smart.DISABLED_REVISION)

    def test_boundary_bypass_only_when_all_slots_are_none(self):
        for families, disabled in ((("None", "None", "None"), True), (("None", "SDXL", "None"), False)):
            prompt = self.full_prompt(families)
            with self.subTest(families=families):
                self.assertEqual(boundary._detailer_disabled_state(prompt, "b")[0], disabled)
        prompt = self.full_prompt(("SDXL", "Z-Image Turbo", "SDXL"), detailer_global_enable="DISABLE")
        self.assertTrue(boundary._detailer_disabled_state(prompt, "b")[0], "global DISABLE bypasses the module")
        prompt = self.full_prompt(("SDXL", "Z-Image Turbo", "SDXL"), detailer_global_enable="ENABLE")
        self.assertFalse(boundary._detailer_disabled_state(prompt, "b")[0])
        self.assertIsNone(boundary._detailer_retention_family(self.full_prompt(("SDXL", "SDXL", "SDXL")), "b"),
                          "the single-entry retention rule applies to the Advanced boundary")


if __name__ == "__main__":
    unittest.main()
