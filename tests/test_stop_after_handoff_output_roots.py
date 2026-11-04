from __future__ import annotations

import json
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FACE_REBUILD_ADVANCED = (
    PACKAGE_ROOT / "subgraphs" / "CMK Flow · FaceRebuild SDXL · Advanced.json"
)

CANONICAL_MODULES = {
    "CMK Flow · Detailer SDXL · Advanced.json",
    "CMK Flow · Detailer SDXL.json",
    "CMK Flow · FaceRebuild SDXL · Advanced.json",
    "CMK Flow · FaceRebuild SDXL.json",
    "CMK Flow · FaceSwap · Advanced.json",
    "CMK Flow · FaceSwap.json",
    "CMK Flow · Upscale & Save.json",
}


def _graphs(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("definitions", {}).get("subgraphs", [])


def _nodes_and_links(graph):
    nodes = {str(node["id"]): node for node in graph.get("nodes", [])}
    links = {
        str(link["id"]): link
        for link in graph.get("links", [])
        if isinstance(link, dict)
    }
    return nodes, links


def _input_link(node, name, links):
    item = next(entry for entry in node.get("inputs", []) if entry.get("name") == name)
    return links[str(item["link"])]


def _reachable_nodes(graph, *, stop: bool):
    nodes, links = _nodes_and_links(graph)
    reached = set()

    def visit_link(link_id):
        link = links.get(str(link_id))
        if link is None:
            return
        visit_node(str(link.get("origin_id")))

    def visit_node(node_id):
        node = nodes.get(node_id)
        if node is None or node_id in reached:
            return
        reached.add(node_id)
        inputs = node.get("inputs", [])
        if node.get("type") == "CMKModuleBypassGate":
            selected = " BYPASS" if stop else " ACTIVE"
            inputs = [item for item in inputs if item.get("name", "").endswith(selected)]
        for item in inputs:
            if item.get("link") is not None:
                visit_link(item["link"])

    for output in graph.get("outputs", []):
        for link_id in output.get("linkIds", []):
            visit_link(link_id)
    for node in nodes.values():
        if node.get("type") == "CMKVisualCompare":
            visit_node(str(node["id"]))
    return {nodes[node_id].get("type") for node_id in reached}


class StopAfterHandoffOutputRootTests(unittest.TestCase):
    def test_canonical_visual_compare_roots_use_existing_bypass_gate(self):
        for name in sorted(CANONICAL_MODULES):
            path = PACKAGE_ROOT / "subgraphs" / name
            with self.subTest(path=path):
                graph = _graphs(path)[0]
                nodes, links = _nodes_and_links(graph)
                gate = next(
                    node for node in nodes.values()
                    if node.get("type") == "CMKModuleBypassGate"
                )
                gate_visual_slot = next(
                    index for index, output in enumerate(gate.get("outputs", []))
                    if output.get("name") == "VISUAL"
                )
                compare = next(
                    node for node in nodes.values()
                    if node.get("type") == "CMKVisualCompare"
                )
                link = _input_link(compare, "VISUAL", links)
                self.assertEqual(str(link["origin_id"]), str(gate["id"]))
                self.assertEqual(link["origin_slot"], gate_visual_slot)
                self.assertIn(link["id"], gate["outputs"][gate_visual_slot]["links"])

    def test_all_modern_reference_compare_roots_respect_visual_gate(self):
        roots = (PACKAGE_ROOT / "subgraphs", PACKAGE_ROOT / "workflows" / "showcase")
        checked = 0
        for root in roots:
            for path in root.rglob("*.json"):
                for graph in _graphs(path):
                    nodes, links = _nodes_and_links(graph)
                    gates = [
                        node for node in nodes.values()
                        if node.get("type") == "CMKModuleBypassGate"
                    ]
                    compares = [
                        node for node in nodes.values()
                        if node.get("type") == "CMKVisualCompare"
                    ]
                    if len(gates) != 1 or not compares:
                        continue
                    gate = gates[0]
                    visual_slots = [
                        index for index, output in enumerate(gate.get("outputs", []))
                        if output.get("name") == "VISUAL"
                    ]
                    if not visual_slots:
                        # These six legacy 2.6 matrix definitions predate the VISUAL
                        # gate port. The task explicitly forbids adding a new port.
                        self.assertEqual(path.name, "Identity Processing Matrix.json")
                        continue
                    checked += len(compares)
                    for compare in compares:
                        link = _input_link(compare, "VISUAL", links)
                        self.assertEqual(
                            str(link["origin_id"]),
                            str(gate["id"]),
                            f"{path}: {graph.get('name')}",
                        )
                        self.assertEqual(link["origin_slot"], visual_slots[0])
        self.assertGreaterEqual(checked, 30)

    def test_expanded_face_rebuild_roots_stop_before_active_module(self):
        graph = _graphs(FACE_REBUILD_ADVANCED)[0]
        nodes, links = _nodes_and_links(graph)
        gate = next(
            node for node in nodes.values()
            if node.get("type") == "CMKModuleBypassGate"
        )
        compare = next(
            node for node in nodes.values()
            if node.get("type") == "CMKVisualCompare"
        )
        compare_link = _input_link(compare, "VISUAL", links)
        public_visual = next(
            output for output in graph["outputs"] if output.get("name") == "VISUAL"
        )
        public_link = links[str(public_visual["linkIds"][0])]

        self.assertEqual(str(compare_link["origin_id"]), str(gate["id"]))
        self.assertEqual(str(public_link["origin_id"]), str(gate["id"]))

        active_type = "CMKInstantIDFaceRebuildAdvancedSDXL"
        self.assertNotIn(active_type, _reachable_nodes(graph, stop=True))
        self.assertIn(active_type, _reachable_nodes(graph, stop=False))
        self.assertIn("CMKVisualCompare", _reachable_nodes(graph, stop=True))


if __name__ == "__main__":
    unittest.main()
