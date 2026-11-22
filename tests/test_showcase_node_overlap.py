"""Shipped showcase workflows: no node may hide another one.

CMK 2.8 · Text2Image · Identity 2 shipped an unconnected copy of the 15A
subgraph exactly on top of the wired one, covering its execution outline and
progress bar.  Unconnected nodes are allowed; only near-congruent rectangles on
the same graph level (root or one subgraph definition) are reported.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = PACKAGE_ROOT / "workflows" / "showcase"
TITLE_HEIGHT = 30
CONGRUENT = 0.8  # share of the smaller node covered by the other one


def _xy(value):
    return (value[0], value[1]) if isinstance(value, list) else (value["0"], value["1"])


def _rect(node):
    x, y = _xy(node["pos"])
    width, height = _xy(node["size"])
    return x, y - TITLE_HEIGHT, x + width, y + height


def _covered(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return width * height / smaller if smaller > 0 else 0.0


def congruent_nodes(workflow):
    """(level, id_a, id_b, coverage) for nodes that hide each other."""
    levels = [("root", workflow.get("nodes", []))]
    levels += [(s.get("name", s["id"]), s.get("nodes", []))
               for s in (workflow.get("definitions") or {}).get("subgraphs", [])]
    found = []
    for level, nodes in levels:
        visible = [n for n in nodes if n.get("pos") is not None and n.get("size") is not None
                   and not (n.get("flags") or {}).get("collapsed")]
        for index, a in enumerate(visible):
            for b in visible[index + 1:]:
                coverage = _covered(_rect(a), _rect(b))
                if coverage >= CONGRUENT:
                    found.append((level, a["id"], b["id"], round(coverage, 2)))
    return found


class ShowcaseNodeOverlap(unittest.TestCase):
    def test_no_shipped_workflow_hides_a_node(self):
        files = sorted(SHOWCASE.glob("*.json"))
        self.assertTrue(files)
        for path in files:
            with self.subTest(workflow=path.name):
                self.assertEqual(congruent_nodes(json.loads(path.read_text(encoding="utf-8"))), [])

    def test_detects_a_congruent_copy(self):
        node = {"id": 1, "type": "x", "pos": [100, 100], "size": [300, 70]}
        copy = dict(node, id=2)
        beside = dict(node, id=3, pos=[500, 100])
        self.assertEqual(congruent_nodes({"nodes": [node, copy, beside]}), [("root", 1, 2, 1.0)])
        self.assertEqual(congruent_nodes({"nodes": [node, beside]}), [])

    def test_unconnected_nodes_alone_are_allowed(self):
        lonely = {"id": 1, "type": "x", "pos": [0, 0], "size": [300, 70], "inputs": [], "outputs": []}
        self.assertEqual(congruent_nodes({"nodes": [lonely]}), [])

    def test_subgraph_definitions_are_checked(self):
        inner = [{"id": 5, "type": "x", "pos": [0, 0], "size": [200, 200]},
                 {"id": 6, "type": "y", "pos": [10, 10], "size": [200, 200]}]
        workflow = {"nodes": [], "definitions": {"subgraphs": [{"id": "s", "name": "Inner", "nodes": inner}]}}
        self.assertEqual([hit[:3] for hit in congruent_nodes(workflow)], [("Inner", 5, 6)])


if __name__ == "__main__":
    unittest.main()
