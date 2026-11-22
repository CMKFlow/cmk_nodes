"""CMK Mask Detailer · Intake: port compatibility with CMK Result Unpack and
the published snapshot state.

* The IMAGE / LOG inputs of the Intake accept exactly the IMAGE / LOG outputs
  of CMK Result Unpack -Pipe- (backend declaration), and every saved link into
  them in the package carries the same type.
* No published workflow or package subgraph ships a locked mask snapshot: the
  image widget is "None" or a pending ``incoming-*`` candidate, so a fresh
  install starts in NO SNAPSHOT / LIVE (2.8: CMK 2.8 · Text2Image showed
  LOCKED from a stored ``clipspace-painted-masked-…`` reference).
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_mask_detailer_intake_contract_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")

CMK_TEST_TIER = "T0"
INTAKE = "CMKMaskDetailerIntake"
UNPACK = "CMKResultUnpackPipe"
PENDING = re.compile(r"(^|/)incoming-[^/]+\.png$", re.IGNORECASE)


def shipped_files():
    return sorted((PACKAGE_ROOT / "workflows").rglob("*.json")) + sorted((PACKAGE_ROOT / "subgraphs").glob("*.json"))


def graphs(document):
    yield document.get("nodes", []), document.get("links", [])
    for definition in document.get("definitions", {}).get("subgraphs", []):
        yield definition.get("nodes", []), definition.get("links", [])


def link_rows(links):
    for link in links:
        if isinstance(link, dict):
            yield link["id"], link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"], link.get("type")
        else:
            yield link[0], link[1], link[2], link[3], link[4], link[5]


class IntakePortContract(unittest.TestCase):
    def test_backend_types_match_result_unpack(self):
        intake = mappings.NODE_CLASS_MAPPINGS[INTAKE]
        unpack = mappings.NODE_CLASS_MAPPINGS[UNPACK]
        required = intake.INPUT_TYPES()["required"]
        outputs = dict(zip(unpack.RETURN_NAMES, unpack.RETURN_TYPES))
        for name in ("IMAGE", "LOG"):
            self.assertEqual(required[name][0], outputs[name], name)
        self.assertEqual((required["IMAGE"][0], required["LOG"][0]), ("IMAGE", "CMK_LOG_PIPE"))
        # The terminal PROCESS input is the only wildcard on the Intake.
        wildcards = [name for name, spec in required.items() if isinstance(spec[0], str) and spec[0] == "*"]
        self.assertEqual(wildcards, ["PROCESS"])

    def test_saved_links_into_image_and_log_are_type_identical(self):
        checked = 0
        for path in shipped_files():
            document = json.loads(path.read_text(encoding="utf-8"))
            for nodes, links in graphs(document):
                by_id = {str(node["id"]): node for node in nodes}
                for _, origin, origin_slot, target, target_slot, _ in link_rows(links):
                    node = by_id.get(str(target))
                    if node is None or node.get("type") != INTAKE:
                        continue
                    port = node["inputs"][target_slot]
                    if port["name"] not in ("IMAGE", "LOG"):
                        continue
                    source = by_id.get(str(origin))
                    if source is None:  # subgraph input node
                        continue
                    with self.subTest(file=path.name, node=node["id"], port=port["name"]):
                        self.assertEqual(source["outputs"][origin_slot]["type"], port["type"])
                        self.assertEqual(source["outputs"][origin_slot]["name"], port["name"])
                        checked += 1
        self.assertGreater(checked, 0)


class PublishedSnapshotState(unittest.TestCase):
    def test_no_locked_snapshot_is_shipped(self):
        seen = 0
        for path in shipped_files():
            document = json.loads(path.read_text(encoding="utf-8"))
            for nodes, _ in graphs(document):
                for node in nodes:
                    if node.get("type") != INTAKE:
                        continue
                    seen += 1
                    # The snapshot is stored three times; the frontend reads
                    # widgets_values_named first (cmk_mask_detailer_intake.js).
                    values = [node["widgets_values"][1], (node.get("properties") or {}).get("image", "None"),
                              (node.get("widgets_values_named") or {}).get("image", "None")]
                    for value in values:
                        with self.subTest(file=path.name, node=node["id"]):
                            self.assertTrue(value == "None" or PENDING.search(str(value)), value)
        self.assertGreater(seen, 0)

    def test_text2image_intake_starts_without_snapshot(self):
        document = json.loads((PACKAGE_ROOT / "workflows" / "showcase" / "CMK 2.8 · Text2Image.json").read_text(encoding="utf-8"))
        intake = [node for nodes, _ in graphs(document) for node in nodes if node.get("type") == INTAKE]
        self.assertEqual([(node["widgets_values"][1], node["properties"]["image"], node["widgets_values_named"]["image"])
                          for node in intake], [("None", "None", "None")])


if __name__ == "__main__":
    unittest.main()
