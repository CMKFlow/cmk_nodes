"""FaceProcess 2.8 F1: CMK FaceProcess Mode Enable and the extended prompt
inline handler.

The Detailer contract is pinned against a frozen copy of the handler as
committed in 9f4f950: every API fixture, every Detailer family combination
and the Detailer part of every mixed Detailer/FaceProcess prompt must stay
identical.
"""

from __future__ import annotations

import copy
import importlib
import itertools
import json
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_faceprocess_mode_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

mode = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_faceprocess_mode")
inline = importlib.import_module(f"{TEST_PACKAGE}.cmk_prompt_family_inline")
mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")
detailer_mm = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_detailer_prepare_multimodel")

CMK_TEST_TIER = "T0"
CONTROL = mode.CMKFaceProcessModeEnable
FP_VALUES = ("None", "Restore", "SDXL", "Z-Image Turbo")
DETAILER_VALUES = ("None", "SDXL", "Z-Image Turbo")
FIXTURES = PACKAGE_ROOT / "tests" / "fixtures" / "api_prompts"


def reference_detailer_inline(prompt):
    """inline_family_links as committed in 9f4f950 (Detailer control only)."""
    selections = {
        str(node_id): (node.get("inputs") or {}).get("model_family")
        for node_id, node in prompt.items()
        if isinstance(node, dict) and str(node.get("class_type", "")) == "CMKDetailerFamilyEnable"
    }
    selections = {key: value for key, value in selections.items() if value in DETAILER_VALUES}
    if not selections:
        return prompt, 0
    result = copy.deepcopy(prompt)
    count = 0
    for node in result.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        for name, value in list(inputs.items()):
            if isinstance(value, (list, tuple)) and len(value) == 2 and str(value[0]) in selections and value[1] == 1:
                inputs[name] = selections[str(value[0])]
                count += 1
    return (result, count) if count else (prompt, 0)


def advanced_prompt(detailer_families, faceprocess_modes=None):
    prompt = {"100": {"class_type": "CMKDetailerPrepareAdvancedMultiModelPipe", "inputs": {"sam_model_name": "sam.pth"}}}
    for slot, family in enumerate(detailer_families, 1):
        prompt[str(slot)] = {"class_type": "CMKDetailerFamilyEnable", "inputs": {"model_family": family}}
        prompt["100"]["inputs"][f"model_family_{slot}"] = [str(slot), 1]
        prompt[str(50 + slot)] = {"class_type": "CMKModuleBypassGate", "inputs": {"ENABLE": [str(slot), 0]}}
    if faceprocess_modes:
        prompt["200"] = {"class_type": "FaceProcessConsumer", "inputs": {}}
        for slot, value in enumerate(faceprocess_modes, 1):
            prompt[str(210 + slot)] = {"class_type": "CMKFaceProcessModeEnable", "inputs": {"process_mode": value}}
            prompt["200"]["inputs"][f"process_mode_{slot}"] = [str(210 + slot), 1]
            prompt[str(220 + slot)] = {"class_type": "CMKModuleBypassGate", "inputs": {"ENABLE": [str(210 + slot), 0]}}
    return prompt


def detailer_part(prompt):
    return {key: value for key, value in prompt.items() if not key.startswith("2")}


class ModeControl(unittest.TestCase):
    def test_contract(self):
        self.assertEqual(CONTROL.INPUT_TYPES(), {"required": {"process_mode": (list(FP_VALUES), {"default": "None"})}})
        self.assertEqual(CONTROL.RETURN_NAMES, ("ENABLE", "PROCESS MODE"))
        self.assertEqual(CONTROL.RETURN_TYPES[0], "BOOLEAN")
        self.assertEqual(list(CONTROL.RETURN_TYPES[1]), list(FP_VALUES), "COMBO output carries the mode values")
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKFaceProcessModeEnable"], CONTROL)
        self.assertEqual(mappings.NODE_DISPLAY_NAME_MAPPINGS["CMKFaceProcessModeEnable"], "CMK FaceProcess Mode Enable")

    def test_read(self):
        self.assertEqual([CONTROL.read(value) for value in FP_VALUES],
                         [(False, "None"), (True, "Restore"), (True, "SDXL"), (True, "Z-Image Turbo")])

    def test_mode_ids_and_family(self):
        self.assertEqual([mode.faceprocess_mode(value) for value in (*FP_VALUES, "", None, "restore", "x")],
                         ["none", "restore", "sdxl", "z_image_turbo", "none", "none", "restore", "none"])
        self.assertEqual([mode.faceprocess_model_family(value) for value in FP_VALUES],
                         [None, None, "sdxl", "z_image_turbo"], "RESTORE needs no diffusion family")

    def test_detailer_control_is_unchanged(self):
        self.assertEqual(detailer_mm.DETAILER_MODEL_FAMILIES, ["None", "SDXL", "Z-Image Turbo"])
        control = detailer_mm.CMKDetailerFamilyEnable
        self.assertEqual(control.INPUT_TYPES(), {"required": {"model_family": (["None", "SDXL", "Z-Image Turbo"], {"default": "None"})}})
        self.assertEqual(control.RETURN_NAMES, ("ENABLE", "MODEL FAMILY"))
        self.assertEqual((inline.FAMILY_CONTROL_CLASS, inline.FAMILY_OUTPUT_SLOT, inline.FAMILY_VALUES),
                         ("CMKDetailerFamilyEnable", 1, ("None", "SDXL", "Z-Image Turbo")))


class DetailerPromptsUnchanged(unittest.TestCase):
    def test_api_fixtures(self):
        files = sorted(FIXTURES.glob("*.json"))
        self.assertGreaterEqual(len(files), 23)
        for path in files:
            with self.subTest(fixture=path.name):
                payload = json.loads(path.read_text(encoding="utf-8"))
                prompt = payload.get("prompt", payload)
                self.assertEqual(inline.inline_family_links(prompt), reference_detailer_inline(prompt))

    def test_all_detailer_family_combinations(self):
        count = 0
        for families in itertools.product(DETAILER_VALUES, repeat=3):
            prompt = advanced_prompt(families)
            self.assertEqual(inline.inline_family_links(prompt), reference_detailer_inline(prompt))
            count += 1
        self.assertEqual(count, 27)

    def test_invalid_detailer_values_stay_links(self):
        prompt = advanced_prompt(("Restore", "SDXL", "x"))
        inlined, count = inline.inline_family_links(prompt)
        self.assertEqual((inlined, count), reference_detailer_inline(prompt))
        self.assertEqual(inlined["100"]["inputs"]["model_family_1"], ["1", 1], "Restore is no Detailer family")


class MixedPrompts(unittest.TestCase):
    def test_detailer_part_identical_and_faceprocess_inlined(self):
        checked = 0
        for families in itertools.product(DETAILER_VALUES, repeat=3):
            for modes in itertools.product(FP_VALUES, repeat=3):
                prompt = advanced_prompt(families, modes)
                original = copy.deepcopy(prompt)
                inlined, count = inline.inline_family_links(prompt)
                reference, reference_count = reference_detailer_inline(prompt)
                self.assertEqual(detailer_part(inlined), detailer_part(reference))
                self.assertEqual([inlined["200"]["inputs"][f"process_mode_{slot}"] for slot in (1, 2, 3)], list(modes))
                self.assertEqual(count, reference_count + 3)
                for slot in (1, 2, 3):  # ENABLE links (output 0) stay links
                    self.assertEqual(inlined[str(220 + slot)]["inputs"]["ENABLE"], [str(210 + slot), 0])
                self.assertEqual(prompt, original, "the input prompt is not modified")
                checked += 1
        self.assertEqual(checked, 1728)

    def test_faceprocess_only_and_invalid_values(self):
        prompt = {
            "1": {"class_type": "CMKFaceProcessModeEnable", "inputs": {"process_mode": "Restore"}},
            "2": {"class_type": "CMKFaceProcessModeEnable", "inputs": {"process_mode": "Bogus"}},
            "3": {"class_type": "CMKFaceProcessModeEnable", "inputs": {"process_mode": ["9", 0]}},
            "4": {"class_type": "Consumer", "inputs": {"a": ["1", 1], "b": ["2", 1], "c": ["3", 1], "d": ["1", 0]}},
        }
        inlined, count = inline.inline_family_links(prompt)
        self.assertEqual(count, 1)
        self.assertEqual(inlined["4"]["inputs"], {"a": "Restore", "b": ["2", 1], "c": ["3", 1], "d": ["1", 0]})
        self.assertEqual(reference_detailer_inline(prompt), (prompt, 0), "the old handler ignored FaceProcess")

    def test_on_prompt(self):
        prompt = advanced_prompt(("SDXL", "None", "Z-Image Turbo"), ("Restore", "None", "SDXL"))
        result = inline.on_prompt({"prompt": prompt, "client_id": "x"})
        self.assertEqual(result["prompt"]["200"]["inputs"]["process_mode_1"], "Restore")
        self.assertEqual(result["prompt"]["100"]["inputs"]["model_family_3"], "Z-Image Turbo")
        self.assertEqual(result["client_id"], "x")


if __name__ == "__main__":
    unittest.main()
