"""Generic, read-only checks over canonical and embedded subgraph definitions.

Every check returns a list of finding dicts.  Nothing here writes to a
workflow; classification against the registries happens in ``report.py``.
"""

from __future__ import annotations

import re

from .graph_model import SUBGRAPH_INPUT, SUBGRAPH_OUTPUT, Graph

VISUAL_ENDPOINTS = {"CMKVisualCompare", "CMKVisualizer"}


def gate_path_completeness(graph: Graph) -> list[dict]:
    """V-C2/V-C3: a used gate output needs both its ACTIVE and BYPASS input.

    Applies to every node exposing ``X ACTIVE``/``X BYPASS`` input pairs, so no
    gate class has to be listed.  Inputs without a BYPASS counterpart (such as
    ``DIAGNOSTIC ACTIVE``) are not paired paths and are ignored.
    """
    findings = []
    for node_id, node in graph.nodes.items():
        names = {str(entry.get("name")) for entry in node.get("inputs", [])}
        roles = {name[: -len(" ACTIVE")] for name in names if name.endswith(" ACTIVE")}
        roles &= {name[: -len(" BYPASS")] for name in names if name.endswith(" BYPASS")}
        if not roles:
            continue
        incoming = graph.incoming(node_id)
        used_outputs = graph.outgoing_names(node_id)
        for role in sorted(roles):
            if role not in used_outputs:
                continue
            missing = [f"{role} {side}" for side in ("ACTIVE", "BYPASS") if f"{role} {side}" not in incoming]
            if missing:
                findings.append({
                    "check": "gate_path_incomplete", "node": node_id,
                    "class": node.get("type"), "role": role, "missing": missing,
                })
    return findings


def unconnected_visual_endpoints(graph: Graph) -> list[dict]:
    """V-C3: a VISUAL display endpoint without an incoming VISUAL link."""
    findings = []
    for node_id, node in graph.nodes.items():
        if node.get("type") not in VISUAL_ENDPOINTS:
            continue
        if "VISUAL" not in graph.incoming(node_id):
            findings.append({"check": "visual_endpoint_unconnected", "node": node_id, "class": node.get("type")})
    return findings


def link_type_compatibility(graph: Graph, catalog: dict, validate) -> list[dict]:
    """V-C4: every internal link satisfies ComfyUI's own type rule.

    ``catalog`` maps class_type -> {"inputs": {name: type}, "outputs": [type]}
    taken from the loaded node classes; ``validate`` is
    ``comfy_execution.validation.validate_node_input``.
    """
    findings = []
    for link in graph.links.values():
        origin_type = _origin_type(graph, link, catalog)
        target_type = _target_type(graph, link, catalog)
        if origin_type is None or target_type is None:
            continue
        if not validate(origin_type, target_type):
            findings.append({
                "check": "link_type_mismatch", "link": link.id,
                "origin": f"{graph.node_type(link.origin_id)}.{graph.output_name(link)}",
                "target": f"{graph.node_type(link.target_id)}.{graph.input_name(link)}",
                "received": origin_type, "expected": target_type,
            })
    return findings


def _origin_type(graph, link, catalog):
    if link.origin_id == SUBGRAPH_INPUT:
        return str(graph.inputs[link.origin_slot].get("type"))
    node = graph.nodes.get(link.origin_id)
    if node is None:
        return None
    spec = catalog.get(str(node.get("type")))
    if spec and link.origin_slot < len(spec["outputs"]):
        return spec["outputs"][link.origin_slot]
    return None


def _target_type(graph, link, catalog):
    if link.target_id == SUBGRAPH_OUTPUT:
        return str(graph.outputs[link.target_slot].get("type"))
    node = graph.nodes.get(link.target_id)
    if node is None:
        return None
    spec = catalog.get(str(node.get("type")))
    if spec is None:
        return None
    return spec["inputs"].get(graph.input_name(link))


def documented_output_order(graph: Graph) -> list[dict]:
    """V-C6 (ARCHITECTURE.md §3): VISUAL directly after LOG and directly
    before the optional diagnostic output on every public module surface."""
    names = [name for name, _type in graph.public_interface()["outputs"]]
    if "VISUAL" not in names or "LOG" not in names:
        return []
    visual = names.index("VISUAL")
    problems = []
    if names.index("LOG") != visual - 1:
        problems.append("VISUAL is not directly after LOG")
    if "diagnostic" in names and names.index("diagnostic") != visual + 1:
        problems.append("diagnostic is not directly after VISUAL")
    return [{"check": "public_output_order", "problems": problems, "outputs": names}] if problems else []


_PROVIDER_ID = re.compile(r"^cmk-(.+)-(\d+)$")


def provider_metadata(instance_node: dict, definition: Graph | None) -> list[dict]:
    """V-C7: stored cmkVisualProviders must reference existing inner nodes."""
    providers = (instance_node.get("properties") or {}).get("cmkVisualProviders") or []
    if not providers or definition is None:
        return []
    inner = {str(node_id): str(node.get("type")) for node_id, node in definition.nodes.items()}
    findings = []
    for provider in providers:
        problems = []
        match = _PROVIDER_ID.match(str(provider.get("provider_id", "")))
        if not match or inner.get(match.group(2)) != match.group(1):
            problems.append("provider_id references no matching inner node")
        live = provider.get("live_node_id")
        if live not in (None, "") and str(live) not in inner:
            problems.append("live_node_id references no inner node")
        if problems:
            findings.append({
                "check": "stale_provider_metadata", "key": provider.get("key"),
                "provider_id": provider.get("provider_id"), "live_node_id": live,
                "problems": problems,
            })
    return findings
