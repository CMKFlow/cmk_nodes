"""Typed, read-only view of ComfyUI workflow and subgraph JSON."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SUBGRAPH_INPUT = -10
SUBGRAPH_OUTPUT = -20


@dataclass(frozen=True)
class Link:
    id: int
    origin_id: int
    origin_slot: int
    target_id: int
    target_slot: int
    type: str


def parse_link(raw) -> Link:
    if isinstance(raw, dict):
        return Link(int(raw["id"]), int(raw["origin_id"]), int(raw["origin_slot"]),
                    int(raw["target_id"]), int(raw["target_slot"]), str(raw.get("type", "")))
    return Link(int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]), int(raw[4]), str(raw[5]) if len(raw) > 5 else "")


class Graph:
    """One subgraph definition (or the top level of a workflow)."""

    def __init__(self, raw: dict, name: str, origin: str):
        self.raw = raw
        self.name = name
        self.origin = origin
        self.id = str(raw.get("id", ""))
        self.nodes = {int(node["id"]): node for node in raw.get("nodes", [])}
        self.links = {link.id: link for link in (parse_link(item) for item in raw.get("links", []))}
        self.inputs = list(raw.get("inputs", []))
        self.outputs = list(raw.get("outputs", []))

    # -- endpoint helpers -------------------------------------------------
    def node_type(self, node_id: int) -> str:
        if node_id == SUBGRAPH_INPUT:
            return "SG_IN"
        if node_id == SUBGRAPH_OUTPUT:
            return "SG_OUT"
        node = self.nodes.get(node_id)
        return str(node.get("type")) if node else f"?{node_id}"

    def output_name(self, link: Link) -> str:
        if link.origin_id == SUBGRAPH_INPUT:
            return str(self.inputs[link.origin_slot].get("name"))
        node = self.nodes.get(link.origin_id)
        return str(node["outputs"][link.origin_slot].get("name")) if node else "?"

    def input_name(self, link: Link) -> str:
        if link.target_id == SUBGRAPH_OUTPUT:
            return str(self.outputs[link.target_slot].get("name"))
        node = self.nodes.get(link.target_id)
        return str(node["inputs"][link.target_slot].get("name")) if node else "?"

    def incoming(self, node_id: int) -> dict[str, Link]:
        node = self.nodes[node_id]
        result = {}
        for entry in node.get("inputs", []):
            link_id = entry.get("link")
            if link_id is not None and link_id in self.links:
                result[str(entry.get("name"))] = self.links[link_id]
        return result

    def outgoing_names(self, node_id: int) -> set[str]:
        node = self.nodes[node_id]
        return {str(out.get("name")) for out in node.get("outputs", []) if out.get("links")}

    # -- normalised structure ---------------------------------------------
    def public_interface(self) -> dict:
        return {
            "inputs": [(str(i.get("name")), str(i.get("type"))) for i in self.inputs],
            "outputs": [(str(o.get("name")), str(o.get("type"))) for o in self.outputs],
        }

    def edges(self) -> Counter:
        counter = Counter()
        for link in self.links.values():
            if link.origin_id not in self.nodes and link.origin_id != SUBGRAPH_INPUT:
                continue
            if link.target_id not in self.nodes and link.target_id != SUBGRAPH_OUTPUT:
                continue
            counter[(
                self.node_type(link.origin_id), self.output_name(link),
                self.node_type(link.target_id), self.input_name(link),
            )] += 1
        return counter

    def node_types(self) -> Counter:
        return Counter(str(node.get("type")) for node in self.nodes.values())

    def signature(self) -> str:
        payload = {
            "interface": self.public_interface(),
            "node_types": sorted(self.node_types().items()),
            "edges": sorted([list(key) + [count] for key, count in self.edges().items()]),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_definitions(path: Path) -> list[Graph]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Graph(definition, str(definition.get("name", "")), str(path))
        for definition in (data.get("definitions", {}) or {}).get("subgraphs", [])
    ]


def load_workflow(path: Path) -> tuple[Graph, list[Graph]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    top = Graph(data, Path(path).name, str(path))
    return top, load_definitions(path)
