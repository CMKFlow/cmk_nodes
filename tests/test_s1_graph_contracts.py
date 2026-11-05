"""S1 §4 — static validation of canonical and embedded subgraph definitions.

The validator is read-only.  Differences to the canonical subgraphs are
classified as allowed legacy (contracts/legacy_registry.json), known defects
(contracts/known_defects.json) or undocumented drift awaiting an owner
decision (contracts/drift_review.json).  Known violations use expectedFailure
with their finding ID plus a baseline test that pins the exact current state.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_ROOT))

from validator import checks  # noqa: E402
from validator.graph_model import load_definitions, load_workflow  # noqa: E402
from validator.report import (  # noqa: E402
    CONTRACTS, DRIFT, IDENTICAL, KNOWN_DEFECT, LEGACY_KNOWN, NO_CANONICAL, PENDING_REVIEW,
    PACKAGE_ROOT, canonical_definitions, classify_embedded_definitions, shipped_workflows,
)

CMK_TEST_TIER = "T0"


def _registry(name):
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


class EmbeddedDefinitionClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = classify_embedded_definitions()

    def _keys(self, cls):
        return sorted({(row["workflow"], row["definition"]) for row in self.rows if row["class"] == cls})

    def test_no_unregistered_drift(self):
        unexplained = [row for row in self.rows if row["class"] in {DRIFT, NO_CANONICAL}]
        self.assertEqual(unexplained, [], "every difference must be legacy, a known defect or under review")

    # 2.7 showcase migration: every embedded definition was brought to the
    # canonical state, so there is no legacy, no known definition defect and
    # nothing left under review.
    def test_shipped_workflows_have_no_unreviewed_drift(self):
        self.assertEqual(self._keys(PENDING_REVIEW), [])

    def test_shipped_workflows_have_no_known_definition_defects(self):
        self.assertEqual(self._keys(KNOWN_DEFECT), [])

    def test_shipped_workflows_have_no_legacy_definitions(self):
        self.assertEqual(self._keys(LEGACY_KNOWN), [])

    def test_shipped_workflows_use_only_canonical_definitions(self):
        self.assertEqual(sorted({row["class"] for row in self.rows}), [IDENTICAL])

    def test_registry_entries_still_present(self):
        present = {(row["workflow"], row["definition"], row["signature"]) for row in self.rows}
        stale = []
        for name, key in (("legacy_registry.json", "entries"), ("known_defects.json", "definition_drift"), ("drift_review.json", "entries")):
            for entry in _registry(name)[key]:
                for signature in entry["signatures"]:
                    if (entry["workflow"], entry["definition"], signature) not in present:
                        stale.append((name, entry["workflow"], entry["definition"]))
        self.assertEqual(stale, [], "registry entries must not outlive the definitions they describe")


class GatePaths(unittest.TestCase):
    def test_canonical_gate_paths_complete(self):
        findings = {
            name: checks.gate_path_completeness(graph)
            for name, graph in canonical_definitions().items()
        }
        self.assertEqual({name: f for name, f in findings.items() if f}, {})

    def _embedded_gate_findings(self):
        result = []
        for workflow in shipped_workflows():
            for graph in load_definitions(workflow):
                for finding in checks.gate_path_completeness(graph):
                    result.append((workflow.name, graph.name, finding["role"], tuple(finding["missing"])))
        return sorted(set(result))

    def test_embedded_gate_paths_complete(self):
        self.assertEqual(self._embedded_gate_findings(), [])

    def test_visual_endpoints_connected(self):
        findings = []
        graphs = list(canonical_definitions().values())
        for workflow in shipped_workflows():
            graphs.extend(load_definitions(workflow))
        for graph in graphs:
            findings.extend((graph.origin, graph.name, f["class"]) for f in checks.unconnected_visual_endpoints(graph))
        self.assertEqual(findings, [])


class PublicSurface(unittest.TestCase):
    def test_canonical_output_order_follows_architecture(self):
        findings = {
            name: checks.documented_output_order(graph)
            for name, graph in canonical_definitions().items()
        }
        self.assertEqual({name: f for name, f in findings.items() if f}, {})


class ProviderMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stale = []
        files = sorted((PACKAGE_ROOT / "subgraphs").glob("*.json")) + shipped_workflows()
        for path in files:
            top, definitions = load_workflow(path)
            by_id = {definition.id: definition for definition in definitions}
            for node in top.nodes.values():
                for finding in checks.provider_metadata(node, by_id.get(str(node.get("type")))):
                    cls.stale.append({"file": path.name, "node": str(node["id"]), "key": finding["key"]})

    def test_baseline_m8_stale_provider_metadata(self):
        expected = _registry("known_defects.json")["stale_provider_metadata"]["instances"]
        self.assertEqual(sorted(self.stale, key=lambda x: (x["file"], x["node"], x["key"] or "")),
                         sorted(expected, key=lambda x: (x["file"], x["node"], x["key"] or "")))

    @unittest.expectedFailure  # M8
    def test_provider_metadata_references_existing_nodes(self):
        self.assertEqual(self.stale, [])


if __name__ == "__main__":
    unittest.main()
