"""2.8.0 Plan B: no independent output root on a Smart Detailer of the shipped
Advanced Multi-Model Detailer.

An output node (e.g. CMK Preview Render) or a subgraph output fed by a Smart
Detailer past the merge is an output root of its own: ComfyUI runs that
Detailer in the hash-random order of the output roots, outside the merge's
family-ordered requests (Z-Image Turbo before SDXL).  Every downstream path of
a Smart Detailer must therefore pass CMK Detailer Merge Multi-Model first.
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
TEST_PACKAGE = "_cmk_detailer_output_roots_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")

CMK_TEST_TIER = "T0"
DETAILER = "CMK_SmartDetailerMultiModelPipe"
MERGE = "CMKDetailerMergeMultiModel"
CORE_OUTPUT_NODES = {"PreviewImage", "SaveImage", "PreviewAny", "SaveAnimatedWEBP", "SaveAnimatedPNG"}
SUBGRAPH_OUTPUT = "-20"
SHIPPED = [PACKAGE_ROOT / "subgraphs" / "CMK Flow · Detailer · Advanced.json",
           PACKAGE_ROOT / "workflows" / "showcase" / "CMK 2.8 · Detailer Advanced · Modul.json"]


def is_output_node(node_type):
    cls = mappings.NODE_CLASS_MAPPINGS.get(node_type)
    return node_type in CORE_OUTPUT_NODES or bool(getattr(cls, "OUTPUT_NODE", False))


def bypassing_roots(graph):
    """(detailer id, reached root) for every path that skips the merge."""
    nodes = {str(node["id"]): node for node in graph["nodes"]}
    downstream = {}
    for link in graph["links"]:
        downstream.setdefault(str(link["origin_id"]), []).append(str(link["target_id"]))
    found = []
    for start in (key for key, node in nodes.items() if node["type"] == DETAILER):
        seen, stack = set(), list(downstream.get(start, []))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current == SUBGRAPH_OUTPUT:
                found.append((start, "subgraph output"))
                continue
            node = nodes.get(current)
            if node is None or node["type"] == MERGE:
                continue
            if is_output_node(node["type"]):
                found.append((start, f"{node['type']} {current}"))
            stack.extend(downstream.get(current, []))
    return found


def advanced_definitions(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [graph for graph in payload.get("definitions", {}).get("subgraphs", [])
            if any(node["type"] == MERGE for node in graph["nodes"])]


class ShippedAdvancedGraphs(unittest.TestCase):
    def test_no_output_root_on_a_smart_detailer_past_the_merge(self):
        for path in SHIPPED:
            graphs = advanced_definitions(path)
            self.assertTrue(graphs, f"{path.name}: no Advanced Detailer definition")
            for graph in graphs:
                with self.subTest(file=path.name):
                    self.assertEqual(bypassing_roots(graph), [])

    def test_single_previews_hang_on_the_merge_slot_diagnostics(self):
        for path in SHIPPED:
            for graph in advanced_definitions(path):
                with self.subTest(file=path.name):
                    nodes = {str(node["id"]): node for node in graph["nodes"]}
                    links = {link["id"]: link for link in graph["links"]}
                    merge = next(node for node in nodes.values() if node["type"] == MERGE)
                    names = [item["name"] for item in merge["outputs"]]
                    self.assertEqual(names, ["IMAGE", "LOG", "VISUAL", "diagnostic",
                                             "diagnostic 1", "diagnostic 2", "diagnostic 3"])
                    # 2.8 blueprint (2026-11-16): the slot previews reach the merge
                    # through the CMK Module Diagnostic Gate (diagnostic n -> n),
                    # the summary preview through the module bypass gate and the
                    # boundary cache (diagnostic -> diagnostic).
                    def merge_source(link):
                        origin = nodes.get(str(link["origin_id"]))
                        if origin is not None and origin["type"] in ("CMKModuleDiagnosticGate", "CMKModuleBypassGate", "CMKDetailerBoundaryCache"):
                            name = origin["outputs"][link["origin_slot"]]["name"]
                            if origin["type"] == "CMKModuleBypassGate":
                                name = {"diagnostic": "DIAGNOSTIC ACTIVE"}[name]
                            gate_input = next(item for item in origin["inputs"] if item["name"] == name)
                            return merge_source(links[gate_input["link"]])
                        return link
                    previews = [node for node in nodes.values() if node["type"] == "CMKPreviewRender"]
                    resolved = [merge_source(links[node["inputs"][0]["link"]]) for node in previews]
                    sources = sorted(names[link["origin_slot"]] for link in resolved
                                     if str(link["origin_id"]) == str(merge["id"]))
                    self.assertEqual(sources, ["diagnostic", "diagnostic 1", "diagnostic 2", "diagnostic 3"])
                    # Each slot output carries the diagnostic of its own Detailer.
                    for slot in (1, 2, 3):
                        source = links[next(i for i in merge["inputs"] if i["name"] == f"SEGS {slot}")["link"]]["origin_id"]
                        diag_in = links[next(i for i in merge["inputs"] if i["name"] == f"diagnostic {slot}")["link"]]
                        self.assertEqual(diag_in["origin_id"], source)

    def test_detector_sees_a_direct_detailer_preview(self):
        graph = {
            "nodes": [{"id": 1, "type": DETAILER}, {"id": 2, "type": MERGE}, {"id": 3, "type": "CMKPreviewRender"},
                      {"id": 4, "type": "CMKPreviewRender"}],
            "links": [{"origin_id": 1, "target_id": 2}, {"origin_id": 2, "target_id": 3},
                      {"origin_id": 1, "target_id": 4}, {"origin_id": 1, "target_id": -20}],
        }
        self.assertEqual(sorted(bypassing_roots(graph)), [("1", "CMKPreviewRender 4"), ("1", "subgraph output")])


if __name__ == "__main__":
    unittest.main()
