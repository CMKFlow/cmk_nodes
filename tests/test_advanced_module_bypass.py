"""CMK 2.8 Advanced Detailer / FaceProcess: the bypass of the Standard variants.

MODE is a light CMK Advanced Mode node (the FaceRebuild v2 MODE control) that
drives the Bypass Gate, the new Module Diagnostic Gate and Preview & Compare;
at DISABLE nothing requests the Prepare, the branches, the Merge or a
diagnostic render.  The Detailer Merge honours the global switch of a Prepare
that still carries it (as the FaceProcess Merge already did).  The user
subgraphs carry that wiring (checked against the real user files).
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_advanced_module_bypass_tests"
CMK_TEST_TIER = "T0"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")
family_result = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_family_result")
mode_module = importlib.import_module(f"{TEST_PACKAGE}.nodes.image.advanced_module_mode")
detailer = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_detailer_advanced_multimodel")
faceprocess = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_faceprocess_advanced_multimodel")
rebuild = importlib.import_module(f"{TEST_PACKAGE}.nodes.instantid_face_rebuild_v2")

GATE = family_result.CMKModuleDiagnosticGate
DIAGNOSTICS = {f"diagnostic {slot}": {"type": "CMK_DIAGNOSTIC", "title": f"D{slot}"} for slot in (1, 2, 3)}
USER_SUBGRAPHS = COMFY_ROOT / "user" / "default" / "subgraphs"


def gate_prompt(*slots):
    return {"g": {"class_type": "CMKModuleDiagnosticGate",
                  "inputs": {"ENABLE": ["m", 0], **{f"diagnostic {s}": ["merge", 3 + s] for s in slots}}}}


class ModeNodes(unittest.TestCase):
    def test_mode_nodes_are_the_faceRebuild_v2_control(self):
        for name, widget in (("CMKDetailerAdvancedModeEnable", "detailer_global_enable"),
                             ("CMKFaceProcessAdvancedModeEnable", "faceprocess_global_enable")):
            with self.subTest(node=name):
                node = mappings.NODE_CLASS_MAPPINGS[name]
                self.assertTrue(issubclass(node, rebuild.CMKFaceRebuildModeEnable))
                # The widget keeps the former Prepare switch name: the outer MODE stays.
                self.assertEqual(node.INPUT_TYPES()["required"], {widget: (["ENABLE", "DISABLE"], {"default": "ENABLE"})})
                self.assertEqual((node.RETURN_TYPES, node.RETURN_NAMES), (("BOOLEAN",), ("ENABLE",)))
                self.assertEqual((node.read("ENABLE"), node.read("DISABLE")), ((True,), (False,)))
                self.assertIn(name, mappings.NODE_DISPLAY_NAME_MAPPINGS)


class DiagnosticGate(unittest.TestCase):
    def test_contract(self):
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKModuleDiagnosticGate"], GATE)
        types_ = GATE.INPUT_TYPES()
        self.assertEqual(types_["required"], {"ENABLE": ("BOOLEAN", {"default": False})})
        self.assertEqual(types_["optional"], {f"diagnostic {s}": ("CMK_DIAGNOSTIC", {"lazy": True}) for s in (1, 2, 3)})
        self.assertEqual(GATE.RETURN_NAMES, ("diagnostic 1", "diagnostic 2", "diagnostic 3"))

    def test_disable_requests_nothing_and_blocks(self):
        prompt = gate_prompt(1, 2, 3)
        self.assertEqual(GATE().check_lazy_status(ENABLE=False, prompt=prompt, unique_id="g"), [])
        result = GATE.gate(ENABLE=False, prompt=prompt, unique_id="g", **DIAGNOSTICS)
        self.assertTrue(all(isinstance(value, family_result.ExecutionBlocker) for value in result))

    def test_enable_requests_connected_slots_and_passes_them(self):
        prompt = gate_prompt(1, 3)
        self.assertEqual(GATE().check_lazy_status(ENABLE=True, prompt=prompt, unique_id="g"), ["diagnostic 1", "diagnostic 3"])
        self.assertEqual(GATE().check_lazy_status(ENABLE=True, prompt=prompt, unique_id="g", **DIAGNOSTICS), [])
        result = GATE.gate(ENABLE=True, prompt=prompt, unique_id="g",
                           **{"diagnostic 1": DIAGNOSTICS["diagnostic 1"], "diagnostic 3": DIAGNOSTICS["diagnostic 3"]})
        self.assertIs(result[0], DIAGNOSTICS["diagnostic 1"])
        self.assertIsInstance(result[1], family_result.ExecutionBlocker, "unconnected slot")
        self.assertIs(result[2], DIAGNOSTICS["diagnostic 3"])


class DetailerMergeGlobalSwitch(unittest.TestCase):
    def prompt(self, switch):
        prompt = {"p": {"class_type": "CMKDetailerPrepareAdvancedMultiModelPipe",
                        "inputs": {"model_family_1": "SDXL", "model_family_2": "Z-Image Turbo", "model_family_3": "SDXL",
                                   "detailer_global_enable": switch}},
                  "m": {"class_type": "CMKDetailerMergeMultiModel", "inputs": {}}}
        for slot in (1, 2, 3):
            prompt[f"d{slot}"] = {"class_type": "CMK_SmartDetailerMultiModelPipe", "inputs": {"DETAILER": ["p", slot - 1]}}
            prompt["m"]["inputs"].update({f"SEGS {slot}": [f"d{slot}", 1], f"LOG {slot}": [f"d{slot}", 3],
                                          f"diagnostic {slot}": [f"d{slot}", 5]})
        return prompt

    def test_disabled_prepare_has_no_active_slot(self):
        merge = detailer.CMKDetailerMergeMultiModel
        self.assertEqual(merge.slot_families(self.prompt("DISABLE"), "m"), {1: "none", 2: "none", 3: "none"})
        self.assertEqual(merge.slot_families(self.prompt("ENABLE"), "m"), {1: "sdxl", 2: "z_image_turbo", 3: "sdxl"})
        kwargs = {f"{name} {slot}": None for slot in (1, 2, 3) for name in ("SEGS", "LOG", "diagnostic")}
        requested = merge().check_lazy_status(prompt=self.prompt("DISABLE"), unique_id="m", **kwargs)
        self.assertEqual(requested, ["diagnostic 1", "diagnostic 2", "diagnostic 3"], "no SEGS / LOG of a disabled module")

    def test_faceprocess_merge_already_honoured_it(self):
        multimodel = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_faceprocess_multimodel")
        self.assertEqual(multimodel.advanced_slot_mode({"faceprocess_global_enable": "DISABLE", "process_mode_1": "SDXL"}, 1), "none")


class UserSubgraphWiring(unittest.TestCase):
    """The authoritative user subgraphs carry the Standard bypass boundary."""

    SPECS = {
        "CMK Flow · Detailer · Advanced.json": ("CMKDetailerAdvancedModeEnable", "detailer_global_enable",
                                                 "CMKDetailerPrepareAdvancedMultiModelPipe", "CMKDetailerMergeMultiModel"),
        "CMK Flow · FaceProcess · Advanced.json": ("CMKFaceProcessAdvancedModeEnable", "faceprocess_global_enable",
                                                    "CMKFaceProcessPrepareAdvancedMultiModelPipe", "CMKFaceProcessMergeMultiModel"),
    }

    def test_gate_decides_without_prepare_and_previews_sit_behind_gates(self):
        for name, (mode_type, widget, prepare_type, merge_type) in self.SPECS.items():
            path = USER_SUBGRAPHS / name
            if not path.is_file():
                self.skipTest(f"user subgraph {name} not present")
            with self.subTest(subgraph=name):
                definition = json.loads(path.read_text(encoding="utf-8"))["definitions"]["subgraphs"][0]
                nodes = {n["id"]: n for n in definition["nodes"]}
                links = {l["id"]: l for l in definition["links"]}

                def source(node, input_name):
                    link = links[next(i for i in node["inputs"] if i["name"] == input_name)["link"]]
                    origin = nodes.get(link["origin_id"])
                    return (origin["type"], origin["outputs"][link["origin_slot"]]["name"]) if origin else (link["origin_id"],)

                of = lambda t: [n for n in definition["nodes"] if n["type"] == t]
                mode, = of(mode_type)
                self.assertEqual(source(mode, widget), (-10,), "outer MODE feeds the Mode node")
                for consumer, input_name in ((of("CMKModuleBypassGate")[0], "ENABLE"), (of("CMKVisualCompare")[0], "enable"),
                                             (of("CMKModuleDiagnosticGate")[0], "ENABLE")):
                    self.assertEqual(source(consumer, input_name), (mode_type, "ENABLE"))
                prepare, = of(prepare_type)
                self.assertFalse(prepare["outputs"][3]["links"], "nothing reads the Prepare ENABLE")
                self.assertIsNone(next(i for i in prepare["inputs"] if i["name"] == widget)["link"])
                sources = sorted(source(p, "diagnostic") for p in of("CMKPreviewRender"))
                self.assertEqual(sources, [("CMKModuleBypassGate", "diagnostic"), ("CMKModuleDiagnosticGate", "diagnostic 1"),
                                           ("CMKModuleDiagnosticGate", "diagnostic 2"), ("CMKModuleDiagnosticGate", "diagnostic 3")])
                diag, = of("CMKModuleDiagnosticGate")
                for slot in (1, 2, 3):
                    self.assertEqual(source(diag, f"diagnostic {slot}"), (merge_type, f"diagnostic {slot}"))


if __name__ == "__main__":
    unittest.main()
