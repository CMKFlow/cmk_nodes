"""Replace outdated embedded CMK subgraph definitions by the canonical ones.

Reproducible, contract-based migration of workflow files (used for the 2.7
showcase migration).  For every embedded definition whose topology signature
differs from the canonical subgraph of the same name (or of a documented
alias), the definition body is replaced by the canonical definition while

* the embedded subgraph id is kept (instances keep referring to it),
* widget values of inner nodes are carried over when the node is matched
  (same id and type, otherwise the only unmatched node of that type) and the
  value count is unchanged,
* instance ports are rebuilt by name in canonical order; existing links keep
  their link ids and get their slot indices updated,
* a new input inherits the external link of the old input that fed exactly
  the same inner targets (e.g. a shared ``opt_image_file`` split into
  ``opt_image_file`` / ``_1`` / ``_2``),
* outputs are matched by name, renamed outputs by slot when type and slot
  agree and the rename is documented.

Everything that cannot be transferred unambiguously is reported as a conflict
and the file is not written unless ``--allow-conflicts`` is given.

    python -I tests/migration/migrate_showcase.py OUT_DIR WORKFLOW.json [...] [--base NAME=PATH] [--report REPORT.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
PACKAGE = TESTS.parent
sys.path.insert(0, str(TESTS))

from validator.graph_model import Graph  # noqa: E402

# Pre-2.7 definition names whose interface and node types equal a canonical subgraph.
ALIASES = {
    "10 1st Pass": "CMK Flow · 10 KSampler SDXL 1st Pass",
    "15 InstantID": "CMK Flow · 15S InstantID Regional SDXL",
    "20 Refiner": "CMK Flow · 20 Refiner SDXL",
}
# Output renames with identical slot and type.
OUTPUT_RENAMES = {("CMK Flow · 20 Refiner SDXL", "IMAGE REFINED"): "IMAGE"}
# Approved positional widget reorders for historic serializations whose widget
# name list is incomplete.  key: node type -> (old value kinds, permutation
# old index per canonical position, decision reference).
# CMKFaceSwapImagePipe (2.6): POST RESTORE was serialized last, after
# POST RESTORE MODEL / RESTORE VISIBILITY; the canonical order has it first.
# Approved 2026-11-05 ("Variante A": POST RESTORE = old value).
WIDGET_REORDERS = {
    "CMKFaceSwapImagePipe": (
        ("bool", "bool", "str", "str", "str", "str", "bool", "str", "number",
         "number", "number", "number", "number", "number", "number", "str", "str", "number", "bool"),
        list(range(16)) + [18, 16, 17],
        "2.7 showcase migration, decision 'Variante A' (POST RESTORE kept from the trailing old value)",
    ),
}


def _reorder(node_type, old_values, new_values):
    rule = WIDGET_REORDERS.get(node_type)
    if rule is None or not isinstance(old_values, list) or not isinstance(new_values, list):
        return None
    kinds, permutation, reference = rule
    if len(old_values) != len(kinds) or len(new_values) != len(permutation):
        return None
    if tuple(_kind(value) for value in old_values) != kinds:
        return None
    reordered = [old_values[index] for index in permutation]
    if any(_kind(a) != _kind(b) for a, b in zip(reordered, new_values)):
        return None
    return reordered, reference


def load_canonical():
    result = {}
    for path in sorted((PACKAGE / "subgraphs").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        definition = doc["definitions"]["subgraphs"][0]
        instance = next(node for node in doc["nodes"] if node["type"] == definition["id"])
        result[definition["name"]] = (definition, instance)
    return result


def _feeders(definition):
    """inner (node id, input name) -> subgraph input name"""
    names = [item["name"] for item in definition["inputs"]]
    nodes = {node["id"]: node for node in definition["nodes"]}
    result = {}
    for link in definition["links"]:
        if link["origin_id"] == -10:
            node = nodes[link["target_id"]]
            result[(link["target_id"], node["inputs"][link["target_slot"]]["name"])] = names[link["origin_slot"]]
    return result


def _neighbours(definition):
    """node id -> {(direction, own slot name, other node id, other slot name)}"""
    nodes = {node["id"]: node for node in definition["nodes"]}
    result = {node_id: set() for node_id in nodes}
    for link in definition["links"]:
        origin, target = link["origin_id"], link["target_id"]
        if origin in nodes and target in nodes:
            out_name = nodes[origin]["outputs"][link["origin_slot"]]["name"]
            in_name = nodes[target]["inputs"][link["target_slot"]]["name"]
            result[origin].add(("out", out_name, target, in_name))
            result[target].add(("in", in_name, origin, out_name))
    return result


def match_nodes(old, new):
    """canonical node id -> old node id: same id and type, else the only
    unmatched node of that type, else the unique candidate whose links reach
    the already matched neighbours in the same way."""
    old_nodes = {node["id"]: node for node in old["nodes"]}
    mapping = {}
    for node in new["nodes"]:
        match = old_nodes.get(node["id"])
        if match is not None and match["type"] == node["type"]:
            mapping[node["id"]] = node["id"]
    old_links, new_links = _neighbours(old), _neighbours(new)
    progress = True
    while progress:
        progress = False
        used = set(mapping.values())
        for node in new["nodes"]:
            if node["id"] in mapping:
                continue
            candidates = [o for o in old["nodes"] if o["type"] == node["type"] and o["id"] not in used]
            same_type_new = [n for n in new["nodes"] if n["type"] == node["type"] and n["id"] not in mapping]
            if len(candidates) == 1 and len(same_type_new) == 1:
                choice = candidates[0]
            else:
                wanted = {(d, own, mapping[other], name) for d, own, other, name in new_links[node["id"]] if other in mapping}
                fitting = [o for o in candidates if wanted and wanted <= old_links[o["id"]]]
                if len(fitting) != 1:
                    continue
                choice = fitting[0]
            mapping[node["id"]] = choice["id"]
            used.add(choice["id"])
            progress = True
    return mapping


def _widget_names(node, values):
    """Widget names in serialization order when the node lists all of them."""
    names = [item["name"] for item in node.get("inputs", []) if "widget" in item]
    return names if isinstance(values, list) and names and len(names) == len(values) else None


def _kind(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _drift(old_values, new_values):
    """Positions whose value type differs from the canonical serialization
    (the canonical definitions are exported by the current frontend), i.e.
    a positional widget order that changed since the old definition."""
    if not (isinstance(old_values, list) and isinstance(new_values, list)) or len(old_values) != len(new_values):
        return []
    return [f"[{index}] {old!r} vs canonical {new!r}" for index, (old, new) in enumerate(zip(old_values, new_values))
            if None not in (_kind(old), _kind(new)) and _kind(old) != _kind(new)]


def migrate_definition(old, canonical, report):
    new = copy.deepcopy(canonical)
    new["id"] = old["id"]
    mapping = match_nodes(old, new)
    old_nodes = {node["id"]: node for node in old["nodes"]}
    for node in new["nodes"]:
        source = old_nodes.get(mapping.get(node["id"]))
        if source is None:
            report["added_nodes"].append(f'{node["id"]}:{node["type"]}')
            continue
        old_values, new_values = source.get("widgets_values"), node.get("widgets_values")
        old_names, new_names = _widget_names(source, old_values), _widget_names(node, new_values)
        if old_names and new_names:
            # Both nodes list every serialized widget by name: transfer by name.
            by_name = dict(zip(old_names, old_values))
            merged = [by_name.get(name, value) for name, value in zip(new_names, new_values)]
            if merged != new_values:
                node["widgets_values"] = merged
                report["widgets_carried"].append({"node": f'{node["id"]}:{node["type"]}', "by": "name",
                                                  "canonical": new_values, "kept": merged})
            dropped = [name for name in old_names if name not in new_names]
            if dropped:
                report["widgets_normalized"].append({"node": f'{node["id"]}:{node["type"]}', "dropped_widgets": dropped})
        elif (
            isinstance(old_values, list) and isinstance(new_values, list) and new_values
            and len(old_values) > len(new_values)
            and all(value == new_values[-1] for value in old_values[len(new_values) - 1:])
        ):
            # Older frontends serialized the image-upload pseudo widget twice.
            report["widgets_normalized"].append({"node": f'{node["id"]}:{node["type"]}', "old": old_values})
            old_values = old_values[:len(new_values)]
        reordered = None if (old_names and new_names) else _reorder(node["type"], old_values, new_values)
        if reordered is not None:
            report["widgets_reordered"].append({"node": f'{node["id"]}:{node["type"]}', "old": old_values,
                                                "migrated": reordered[0], "decision": reordered[1],
                                                "old_widget_names": [item["name"] for item in source.get("inputs", []) if "widget" in item]})
            old_values = reordered[0]
        drift =[] if old_names and new_names else _drift(old_values, new_values)
        if old_names and new_names:
            pass
        elif drift:
            report["conflicts"].append({"kind": "widget_schema_drift", "node": f'{node["id"]}:{node["type"]}',
                                        "positions": drift, "old": old_values, "canonical": new_values})
        elif old_values != new_values:
            if isinstance(old_values, list) and isinstance(new_values, list) and len(old_values) == len(new_values):
                node["widgets_values"] = copy.deepcopy(old_values)
                report["widgets_carried"].append({"node": f'{node["id"]}:{node["type"]}', "canonical": new_values, "kept": old_values})
            elif old_values is None and new_values is not None:
                pass
            else:
                report["conflicts"].append({"kind": "widget_shape", "node": f'{node["id"]}:{node["type"]}',
                                            "old": old_values, "canonical": new_values})
        if source.get("mode", 0) != node.get("mode", 0):
            report["mode_carried"].append({"node": f'{node["id"]}:{node["type"]}', "mode": source.get("mode", 0)})
            node["mode"] = source.get("mode", 0)
    matched_old = set(mapping.values())
    new_types = {node["type"] for node in new["nodes"]}
    for node in old["nodes"]:
        if node["id"] not in matched_old:
            report["dropped_nodes"].append({"node": f'{node["id"]}:{node["type"]}', "widgets": node.get("widgets_values")})
            if node["type"] in new_types:
                report["conflicts"].append({"kind": "unmatched_same_type", "node": f'{node["id"]}:{node["type"]}'})
    return new, mapping


def migrate_instances(doc, old, new, mapping, template, report):
    old_inputs = [item["name"] for item in old["inputs"]]
    new_inputs = [item["name"] for item in new["inputs"]]
    old_outputs = [item["name"] for item in old["outputs"]]
    new_outputs = [item["name"] for item in new["outputs"]]
    renamed = {}
    for slot, name in enumerate(old_outputs):
        if name not in new_outputs and slot < len(new_outputs) and OUTPUT_RENAMES.get((new["name"], name)) == new_outputs[slot] \
                and old["outputs"][slot]["type"] == new["outputs"][slot]["type"]:
            renamed[name] = new_outputs[slot]
            report["output_renames"].append(f"{name} -> {new_outputs[slot]}")
    # inheritance of new inputs: canonical targets -> old feeding input
    old_feed = _feeders(old)
    new_feed = _feeders(new)
    reverse = {}
    for (node_id, input_name), sg_name in new_feed.items():
        old_id = mapping.get(node_id)
        reverse.setdefault(sg_name, set()).add(old_feed.get((old_id, input_name)) if old_id is not None else None)
    inherit = {name: next(iter(sources)) for name, sources in reverse.items()
               if name not in old_inputs and len(sources) == 1 and next(iter(sources)) in old_inputs}
    links = {link[0]: link for link in doc["links"]}
    nodes = {node["id"]: node for node in doc["nodes"]}
    template_inputs = {item["name"]: item for item in template["inputs"]}
    template_outputs = {item["name"]: item for item in template["outputs"]}
    for instance in [node for node in doc["nodes"] if node["type"] == old["id"]]:
        by_name = {item["name"]: item for item in instance.get("inputs", [])}
        rebuilt = []
        for name in new_inputs:
            entry = by_name.pop(name, None)
            if entry is None:
                entry = copy.deepcopy(template_inputs.get(name) or {"name": name, "type": new["inputs"][new_inputs.index(name)]["type"]})
                entry["link"] = None
            rebuilt.append(entry)
        for name, entry in by_name.items():
            if entry.get("link") is not None:
                report["conflicts"].append({"kind": "removed_input_linked", "instance": instance["id"], "input": name})
            elif "widget" in entry:
                report["conflicts"].append({"kind": "removed_widget_input", "instance": instance["id"], "input": name})
        instance["inputs"] = rebuilt
        for slot, entry in enumerate(rebuilt):
            if entry.get("link") is not None:
                links[entry["link"]][4] = slot
        for name, source in inherit.items():
            source_entry = next(e for e in rebuilt if e["name"] == source)
            target_entry = next(e for e in rebuilt if e["name"] == name)
            if source_entry.get("link") is None:
                continue
            origin = links[source_entry["link"]]
            doc["last_link_id"] = int(doc.get("last_link_id", max(links))) + 1
            new_id = doc["last_link_id"]
            slot = rebuilt.index(target_entry)
            link = [new_id, origin[1], origin[2], instance["id"], slot, origin[5]]
            doc["links"].append(link)
            links[new_id] = link
            target_entry["link"] = new_id
            nodes[origin[1]]["outputs"][origin[2]].setdefault("links", []).append(new_id)
            report["inherited_links"].append({"instance": instance["id"], "input": name, "from_input": source, "link": new_id})
        out_by_name = {renamed.get(item["name"], item["name"]): item for item in instance.get("outputs", [])}
        rebuilt_out = []
        for name in new_outputs:
            entry = out_by_name.pop(name, None)
            if entry is None:
                entry = copy.deepcopy(template_outputs.get(name) or {"name": name, "type": new["outputs"][new_outputs.index(name)]["type"]})
                entry["links"] = []
            else:
                entry["name"] = name
                if "localized_name" in entry:
                    entry["localized_name"] = name
            rebuilt_out.append(entry)
        for name, entry in out_by_name.items():
            if entry.get("links"):
                report["conflicts"].append({"kind": "removed_output_linked", "instance": instance["id"], "output": name})
        instance["outputs"] = rebuilt_out
        for slot, entry in enumerate(rebuilt_out):
            for link_id in entry.get("links") or []:
                links[link_id][2] = slot


def migrate_workflow(doc, canonical, report):
    definitions = (doc.get("definitions") or {}).get("subgraphs", [])
    for index, old in enumerate(definitions):
        target = old["name"] if old["name"] in canonical else ALIASES.get(old["name"])
        entry = {"definition": old["name"], "id": old["id"], "canonical": target, "added_nodes": [], "dropped_nodes": [],
                 "widgets_carried": [], "widgets_normalized": [], "widgets_reordered": [], "mode_carried": [], "inherited_links": [],
                 "output_renames": [], "conflicts": []}
        if target is None:
            entry["conflicts"].append({"kind": "no_canonical"})
            report.append(entry)
            continue
        canonical_def, template = canonical[target]
        if Graph(old, old["name"], "").signature() == Graph(canonical_def, target, "").signature() and old["name"] == target:
            continue
        if old["name"] != target:
            same_types = sorted(n["type"] for n in old["nodes"]) == sorted(n["type"] for n in canonical_def["nodes"])
            same_inputs = [i["name"] for i in old["inputs"]] == [i["name"] for i in canonical_def["inputs"]]
            if not (same_types and same_inputs):
                entry["conflicts"].append({"kind": "alias_mismatch"})
                report.append(entry)
                continue
        new, mapping = migrate_definition(old, canonical_def, entry)
        migrate_instances(doc, old, new, mapping, template, entry)
        definitions[index] = new
        report.append(entry)
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("workflows", nargs="+")
    parser.add_argument("--base", action="append", default=[], help="NAME=PATH: migrate PATH and write it as NAME")
    parser.add_argument("--report")
    parser.add_argument("--allow-conflicts", action="store_true")
    args = parser.parse_args(argv)
    canonical = load_canonical()
    sources = {Path(p).name: Path(p) for p in args.workflows}
    for item in args.base:
        name, path = item.split("=", 1)
        sources[name] = Path(path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_report = {}
    blocked = []
    for name, path in sorted(sources.items()):
        doc = json.loads(path.read_text(encoding="utf-8"))
        report = []
        migrate_workflow(doc, canonical, report)
        full_report[name] = {"source": str(path), "definitions": report}
        conflicts = [c for entry in report for c in entry["conflicts"]]
        if conflicts and not args.allow_conflicts:
            blocked.append(name)
            continue
        (out_dir / name).write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    if args.report:
        Path(args.report).write_text(json.dumps(full_report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"written": sorted(set(sources) - set(blocked)), "blocked": blocked}, ensure_ascii=False))
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
