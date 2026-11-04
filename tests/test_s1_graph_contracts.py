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
    CONTRACTS, DRIFT, KNOWN_DEFECT, LEGACY_KNOWN, NO_CANONICAL, PENDING_REVIEW,
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

    def test_baseline_pending_review_set(self):
        self.assertEqual(self._keys(PENDING_REVIEW), [
            ("CMK 2.6 · Flow · Inpaint Z-Image Turbo.json", "LoRA Stack · ZIT"),
            ("CMK 2.6 · Full Flow .json", "CMK Flow · 15A InstantID Regional SDXL"),
            ("CMK 2.6 · Full Flow .json", "CMK Flow · FaceRebuild SDXL · Advanced"),
            ("CMK 2.6 · Full Flow .json", "CMK Flow · FaceSwap · Advanced"),
            ("CMK 2.6 · Text2Image Identity 3.json", "CMK Flow · 15A InstantID Regional SDXL"),
            ("CMK 2.6 · Text2Image Identity 3.json", "CMK Flow · FaceRebuild SDXL · Advanced"),
            ("CMK 2.6 · Text2Image Identity 3.json", "CMK Flow · FaceSwap · Advanced"),
        ])

    @unittest.expectedFailure  # S1-REVIEW: owner decision pending
    def test_shipped_workflows_have_no_unreviewed_drift(self):
        self.assertEqual(self._keys(PENDING_REVIEW), [])

    def test_baseline_known_definition_defects(self):
        self.assertEqual(self._keys(KNOWN_DEFECT), [
            ("CMK 2.6 · FaceProcess · Modul.json", "CMK Flow · FaceProcess SDXL"),
            ("CMK 2.6 · FaceRebuild · Modul.json", "CMK Flow · FaceRebuild SDXL"),
            ("CMK 2.6 · FaceSwap · Modul.json", "CMK Flow · FaceSwap"),
            ("CMK 2.6 · SmartDetailer · Modul.json", "CMK Flow · Detailer SDXL"),
        ])

    @unittest.expectedFailure  # F-04 / F-11
    def test_shipped_workflows_have_no_known_definition_defects(self):
        self.assertEqual(self._keys(KNOWN_DEFECT), [])

    def test_legacy_entries_are_identity_matrix_or_frozen_pre_fix_modules(self):
        # Identity Matrix (2.6), pre-H-4 Upscale & Save (K-3 Option A) and the
        # pre-H-3/F-03 module definitions (E1 = a); nothing else is legacy.
        modules = {
            "CMK Flow · Upscale & Save", "CMK Flow · FaceSwap", "CMK Flow · FaceRebuild SDXL",
            "CMK Flow · Detailer SDXL", "CMK Flow · Detailer SDXL · Advanced",
            "CMK Flow · FaceProcess SDXL", "CMK Flow · FaceProcess SDXL · Advanced",
        }
        others = {key for key in self._keys(LEGACY_KNOWN) if key[0] != "Identity Processing Matrix.json"}
        self.assertEqual({definition for _, definition in others} - modules, set())
        self.assertIn(("CMK 2.6 · Full Flow .json", "CMK Flow · Upscale & Save"), others)

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

    def test_baseline_embedded_gate_paths(self):
        missing = ("VISUAL ACTIVE", "VISUAL BYPASS")
        self.assertEqual(self._embedded_gate_findings(), [
            ("CMK 2.6 · FaceProcess · Modul.json", "CMK Flow · FaceProcess SDXL", "VISUAL", missing),
            ("CMK 2.6 · FaceRebuild · Modul.json", "CMK Flow · FaceRebuild SDXL", "VISUAL", missing),
            ("CMK 2.6 · FaceSwap · Modul.json", "CMK Flow · FaceSwap", "VISUAL", missing),
            ("CMK 2.6 · SmartDetailer · Modul.json", "CMK Flow · Detailer SDXL", "VISUAL", missing),
        ])

    @unittest.expectedFailure  # F-04
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
