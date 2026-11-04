"""Collect and classify validator findings.  Read-only; never rewrites files."""

from __future__ import annotations

import glob
import json
from pathlib import Path

from . import checks
from .graph_model import Graph, load_definitions

TESTS_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = TESTS_ROOT.parent
CONTRACTS = TESTS_ROOT / "contracts"

IDENTICAL = "IDENTICAL"
LEGACY_KNOWN = "LEGACY_KNOWN"
KNOWN_DEFECT = "KNOWN_DEFECT"
PENDING_REVIEW = "PENDING_REVIEW"
DRIFT = "DRIFT"
NO_CANONICAL = "NO_CANONICAL"


def _load(name):
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


def canonical_definitions() -> dict[str, Graph]:
    result = {}
    for path in sorted(glob.glob(str(PACKAGE_ROOT / "subgraphs" / "*.json"))):
        for graph in load_definitions(Path(path)):
            result[graph.name] = graph
    return result


def shipped_workflows() -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(PACKAGE_ROOT / "workflows" / "**" / "*.json"), recursive=True))


def _registry_match(entries, workflow: str, definition: str, signature: str):
    for entry in entries:
        if (
            entry["workflow"] == workflow
            and entry["definition"] == definition
            and signature in entry["signatures"]
        ):
            return entry
    return None


def classify_embedded_definitions() -> list[dict]:
    canon = canonical_definitions()
    legacy = _load("legacy_registry.json")["entries"]
    defects = _load("known_defects.json")["definition_drift"]
    review = _load("drift_review.json")["entries"]
    rows = []
    for workflow in shipped_workflows():
        for graph in load_definitions(workflow):
            signature = graph.signature()
            canonical = canon.get(graph.name)
            row = {"workflow": workflow.name, "definition": graph.name, "signature": signature}
            if canonical is not None and canonical.signature() == signature:
                row["class"] = IDENTICAL
            elif (entry := _registry_match(legacy, workflow.name, graph.name, signature)):
                row.update({"class": LEGACY_KNOWN, "reason": entry["reason"]})
            elif (entry := _registry_match(defects, workflow.name, graph.name, signature)):
                row.update({"class": KNOWN_DEFECT, "finding": entry["finding"]})
            elif (entry := _registry_match(review, workflow.name, graph.name, signature)):
                row.update({"class": PENDING_REVIEW, "question": entry["question"]})
            elif canonical is None:
                row["class"] = NO_CANONICAL
            else:
                row["class"] = DRIFT
            rows.append(row)
    return rows
