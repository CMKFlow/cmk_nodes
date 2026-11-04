"""Assemble the S1 fixture workflows (UI format) for frontend expansion.

The workflows are built from the canonical module definitions in
``subgraphs/`` and from the manually exported, validated reference
``workflows/detailer_std.json`` (loader, checkpoint and Visualizer nodes and
their widget values).  This script only writes *staging* workflows; the files
stored as fixtures are the real ComfyUI frontend's serialisation and
``app.graphToPrompt()`` output of these staging workflows (see
FIXTURES_EXPORT.md, "Automatisierter Export").  No subgraph expansion happens
here.

    python -I tests/fixtures/build_workflows.py <staging-dir>
"""

from __future__ import annotations

import copy
import json
import sys
import uuid
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent
PACKAGE_ROOT = FIXTURES.parents[1]
SUBGRAPHS = PACKAGE_ROOT / "subgraphs"
TEMPLATE = FIXTURES / "workflows" / "detailer_std.json"
FULL_FLOW = PACKAGE_ROOT / "workflows" / "showcase" / "CMK 2.6 · Full Flow .json"

LOADER_ID, VISUALIZER_ID, CHECKPOINT_ID = 1, 8418, 8419

# name -> (setup, [module files in chain order], instance widget overrides)
FIXTURES_SPEC = {
    # The template itself; rebuilt from the canonical definition since H-3/F-03.
    "detailer_std": ("B", ["CMK Flow · Detailer SDXL.json"], {}),
    "detailer_adv": ("B", ["CMK Flow · Detailer SDXL · Advanced.json"], {}),
    "faceprocess_std": ("B", ["CMK Flow · FaceProcess SDXL.json"], {}),
    "faceprocess_adv": ("B", ["CMK Flow · FaceProcess SDXL · Advanced.json"], {}),
    "facerebuild_std": ("B", ["CMK Flow · FaceRebuild SDXL.json"], {}),
    "facerebuild_adv": ("B", ["CMK Flow · FaceRebuild SDXL · Advanced.json"], {}),
    "faceswap_std": ("A", ["CMK Flow · FaceSwap.json"], {}),
    "faceswap_adv": ("A", ["CMK Flow · FaceSwap · Advanced.json"], {}),
    "maskdetailer": ("B", ["CMK Flow · MaskDetailer SDXL.json"], {}),
    "upscale_save": ("A", ["CMK Flow · Upscale & Save.json"], {"SAVE ENABLED": True}),
    "chain_postprocess": (
        "B",
        ["CMK Flow · Detailer SDXL.json", "CMK Flow · FaceProcess SDXL.json", "CMK Flow · Upscale & Save.json"],
        {"SAVE ENABLED": True},
    ),
}

ROLE_INPUTS = {
    "MODEL": ("MODEL", "MODEL (opt)"),
    "PROCESS": ("PROCESS",),
    "IMAGE": ("IMAGE", "IMAGE_TARGET"),
    "LOG": ("LOG",),
    "VISUAL": ("VISUAL",),
}
ROLE_OUTPUTS = {
    "MODEL": ("MODEL",),
    "PROCESS": ("PROCESS", "MASK DETAILER PROCESS"),
    "IMAGE": ("IMAGE",),
    "LOG": ("LOG",),
    "VISUAL": ("VISUAL",),
}
VISUALIZER_INPUT = {"MODEL": "MODEL (opt)", "PROCESS": "PROCESS", "IMAGE": "IMAGE", "LOG": "LOG", "VISUAL": "VISUAL"}


class Workflow:
    def __init__(self, name: str, template: dict):
        self.data = {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"cmk-s1-fixture/{name}")),
            "revision": 0,
            "last_node_id": 0,
            "last_link_id": 0,
            "nodes": [],
            "links": [],
            "groups": [],
            "definitions": {"subgraphs": []},
            "config": copy.deepcopy(template.get("config", {})),
            "extra": copy.deepcopy(template.get("extra", {})),
            "version": template.get("version", 0.4),
        }
        self.next_link = 30000

    def add_node(self, node: dict) -> dict:
        node = copy.deepcopy(node)
        for entry in node.get("inputs", []):
            entry["link"] = None
        for entry in node.get("outputs", []):
            entry["links"] = []
        self.data["nodes"].append(node)
        self.data["last_node_id"] = max(self.data["last_node_id"], int(node["id"]))
        return node

    def add_definition(self, definition: dict) -> None:
        if not any(item["id"] == definition["id"] for item in self.data["definitions"]["subgraphs"]):
            self.data["definitions"]["subgraphs"].append(copy.deepcopy(definition))

    @staticmethod
    def _slot(entries, names):
        for index, entry in enumerate(entries):
            if entry.get("name") in names:
                return index, entry
        return None, None

    def connect(self, origin: dict, out_names, target: dict, in_names) -> bool:
        out_names = (out_names,) if isinstance(out_names, str) else out_names
        in_names = (in_names,) if isinstance(in_names, str) else in_names
        origin_slot, output = self._slot(origin["outputs"], out_names)
        target_slot, entry = self._slot(target["inputs"], in_names)
        if output is None or entry is None:
            return False
        link_id = self.next_link
        self.next_link += 1
        self.data["links"].append([link_id, origin["id"], origin_slot, target["id"], target_slot, output["type"]])
        output["links"].append(link_id)
        entry["link"] = link_id
        self.data["last_link_id"] = max(self.data["last_link_id"], link_id)
        return True

    def connect_role(self, origin, target, role, target_names=None) -> bool:
        return self.connect(origin, ROLE_OUTPUTS[role], target, target_names or ROLE_INPUTS[role])


def _canonical(file_name: str):
    data = json.loads((SUBGRAPHS / file_name).read_text(encoding="utf-8"))
    return data["nodes"][0], data["definitions"]["subgraphs"][0]


def _set_widget(node: dict, name: str, value) -> None:
    widget_inputs = [entry["name"] for entry in node.get("inputs", []) if entry.get("widget")]
    if name in widget_inputs:
        node["widgets_values"][widget_inputs.index(name)] = value
        named = node.get("widgets_values_named")
        if isinstance(named, dict):
            named[name] = value


def _postprocess_model_source(node: dict) -> dict:
    """Standalone PostProcess flows use CMK PostProcess Model since H-2 (same
    widgets as the checkpoint loader, no PROCESS input)."""
    node = copy.deepcopy(node)
    node["type"] = "CMKPostProcessModelPipe"
    node["inputs"] = [entry for entry in node.get("inputs", []) if entry.get("name") != "PROCESS"]
    if "Node name for S&R" in (node.get("properties") or {}):
        node["properties"]["Node name for S&R"] = "CMKPostProcessModelPipe"
    return node


def build(name: str, template: dict) -> dict:
    nodes = {node["id"]: node for node in template["nodes"]}
    workflow = Workflow(name, template)
    setup, modules, overrides = FIXTURES_SPEC[name]

    loader = workflow.add_node(nodes[LOADER_ID])
    loader["pos"] = [0, 0]
    upstream = loader
    if setup == "B":
        checkpoint = workflow.add_node(_postprocess_model_source(nodes[CHECKPOINT_ID]))
        checkpoint["pos"] = [-500, 0]
        workflow.connect(checkpoint, "MODEL SDXL", loader, "MODEL SDXL (opt)")

    last_model_process = loader  # last node that still carries MODEL/PROCESS
    for index, module_file in enumerate(modules):
        instance, definition = _canonical(module_file)
        instance["id"] = 8417 + index * 10
        instance["pos"] = [700 + index * 400, 0]
        for key, value in overrides.items():
            _set_widget(instance, key, value)
        workflow.add_definition(definition)
        module = workflow.add_node(instance)
        for role in ("MODEL", "PROCESS"):
            workflow.connect_role(last_model_process, module, role)
        for role in ("IMAGE", "LOG", "VISUAL"):
            workflow.connect_role(upstream, module, role)
        upstream = module
        if any(out["name"] in ROLE_OUTPUTS["PROCESS"] for out in module["outputs"]):
            last_model_process = module

    visualizer = workflow.add_node(nodes[VISUALIZER_ID])
    visualizer["pos"] = [700 + len(modules) * 400, 0]
    for role in ("MODEL", "PROCESS"):
        workflow.connect_role(last_model_process, visualizer, role, VISUALIZER_INPUT[role])
    for role in ("IMAGE", "LOG", "VISUAL"):
        workflow.connect_role(upstream, visualizer, role, VISUALIZER_INPUT[role])
    return workflow.data


def build_legacy(template: dict) -> dict:
    """Keep Full Flow's embedded Upscale & Save definition byte-for-byte."""
    full_flow = json.loads(FULL_FLOW.read_text(encoding="utf-8"))
    definition = next(
        item for item in full_flow["definitions"]["subgraphs"] if item["name"] == "CMK Flow · Upscale & Save"
    )
    instance = next(node for node in full_flow["nodes"] if node["type"] == definition["id"])
    nodes = {node["id"]: node for node in template["nodes"]}
    workflow = Workflow("upscale_save_legacy_fullflow", template)
    loader = workflow.add_node(nodes[LOADER_ID])
    loader["pos"] = [0, 0]
    instance = copy.deepcopy(instance)
    _set_widget(instance, "SAVE ENABLED", True)
    instance["pos"] = [700, 0]
    workflow.add_definition(definition)
    module = workflow.add_node(instance)
    for role in ("PROCESS", "IMAGE", "LOG", "VISUAL"):
        workflow.connect_role(loader, module, role)
    visualizer = workflow.add_node(nodes[VISUALIZER_ID])
    visualizer["pos"] = [1100, 0]
    for role in ("MODEL", "PROCESS"):
        workflow.connect_role(loader, visualizer, role, VISUALIZER_INPUT[role])
    for role in ("IMAGE", "LOG", "VISUAL"):
        workflow.connect_role(module, visualizer, role, VISUALIZER_INPUT[role])
    return workflow.data


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    staging = Path(argv[0])
    staging.mkdir(parents=True, exist_ok=True)
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    outputs = {name: build(name, template) for name in FIXTURES_SPEC}
    outputs["upscale_save_legacy_fullflow"] = build_legacy(template)
    for name, data in outputs.items():
        (staging / f"{name}.workflow.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(sorted(outputs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
