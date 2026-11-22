"""2.8.0: OUTPUT IMAGE PROCEED is an output switch of the Multi-Model
execution nodes only.

With the switch off, IMAGE PROCEED is None by contract.  A module boundary fed
from IMAGE PROCEED therefore had nothing to store on a cache MISS ("cache miss
requires IMAGE").  The boundary of a Multi-Model branch reads the output that
always carries the processed image instead: DETAILER IMAGE of the Smart
Detailer Multi-Model, FACE IMAGE of the FaceProcess Multi-Model execution.
The legacy nodes and the boundary ports are unchanged.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CMK_TEST_TIER = "T0"
BOUNDARIES = {"CMKFaceBoundaryCache", "CMKDetailerBoundaryCache"}
IMAGE_OUTPUTS = {"CMK_SmartDetailerMultiModelPipe": "DETAILER IMAGE", "CMKFaceProcessMultiModelPipe": "FACE IMAGE"}


def boundary_image_sources(graph):
    """(branch class, output name) feeding the IMAGE input of each boundary."""
    nodes = {node["id"]: node for node in graph.get("nodes", [])}
    found = []
    for link in graph.get("links", []):
        if isinstance(link, list):
            _lid, origin, origin_slot, target, target_slot = link[:5]
        else:
            origin, origin_slot = link["origin_id"], link["origin_slot"]
            target, target_slot = link["target_id"], link["target_slot"]
        source, boundary = nodes.get(origin), nodes.get(target)
        if not source or not boundary or boundary["type"] not in BOUNDARIES:
            continue
        if boundary["inputs"][target_slot]["name"] == "IMAGE" and source["type"] in IMAGE_OUTPUTS:
            found.append((source["type"], source["outputs"][origin_slot]["name"]))
    return found


def graphs(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return [data, *(data.get("definitions") or {}).get("subgraphs", [])]


class BoundaryImageSource(unittest.TestCase):
    # 2.8 package and showcase sync: every package subgraph and every shipped
    # workflow feeds the boundary from the switch-independent image output.
    def test_package_subgraphs_store_the_switch_independent_image(self):
        checked = 0
        paths = [*(PACKAGE_ROOT / "subgraphs").glob("*.json"), *(PACKAGE_ROOT / "workflows").glob("**/*.json")]
        for path in sorted(paths):
            for graph in graphs(path):
                for source_type, output in boundary_image_sources(graph):
                    checked += 1
                    with self.subTest(file=path.name, source=source_type):
                        self.assertEqual(output, IMAGE_OUTPUTS[source_type])
        self.assertGreaterEqual(checked, 1, "the Multi-Model Detailer subgraph is checked")

    def test_detector_sees_image_proceed(self):
        graph = {"nodes": [
            {"id": 1, "type": "CMKFaceProcessMultiModelPipe", "outputs": [{"name": "IMAGE PROCEED"}]},
            {"id": 2, "type": "CMKFaceBoundaryCache", "inputs": [{"name": "MODEL"}, {"name": "IMAGE"}]},
        ], "links": [[5, 1, 0, 2, 1, "IMAGE"]]}
        self.assertEqual(boundary_image_sources(graph), [("CMKFaceProcessMultiModelPipe", "IMAGE PROCEED")])

    def test_execution_outputs(self):
        from importlib import import_module
        import sys
        import types

        sys.path.insert(0, str(PACKAGE_ROOT.parents[1]))
        package = types.ModuleType("_cmk_boundary_image_tests")
        package.__path__ = [str(PACKAGE_ROOT)]
        sys.modules.setdefault("_cmk_boundary_image_tests", package)
        mappings = import_module("_cmk_boundary_image_tests.cmk_mappings").NODE_CLASS_MAPPINGS
        for class_type, output in IMAGE_OUTPUTS.items():
            with self.subTest(node=class_type):
                names = mappings[class_type].RETURN_NAMES
                self.assertEqual(names[-1], output)
                self.assertEqual(mappings[class_type].RETURN_TYPES[-1], "IMAGE")
                self.assertIn("IMAGE PROCEED", names)


def _cache_module():
    from importlib import import_module
    import sys
    import types

    sys.path.insert(0, str(PACKAGE_ROOT.parents[1]))
    package = types.ModuleType("_cmk_boundary_image_tests")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules.setdefault("_cmk_boundary_image_tests", package)
    return import_module("_cmk_boundary_image_tests.pipe.cmk_persistent_cache")


class SwitchIdentity(unittest.TestCase):
    """The switch is no identity of the node itself; only a link that reads
    IMAGE PROCEED carries it."""

    def key(self, class_type, slot, switch):
        cache = _cache_module()
        prompt = {
            "1": {"class_type": "CMKImageLoadAndResizePipe", "inputs": {"image": "a.png"}},
            "2": {"class_type": class_type, "inputs": {"output_image_proceed": switch, "select_face": "Largest",
                                                      "FACE": ["1", 0]}},
            "3": {"class_type": "CMKDetailerBoundaryCache", "inputs": {"IMAGE": ["2", slot], "LOG": ["2", 4]}},
        }
        return cache.build_node_fingerprint(prompt, "3", ("CMKDetailerBoundaryCache",), "test_v1")[0]

    def test_boundary_on_the_image_output_ignores_the_switch(self):
        for class_type, slot in (("CMK_SmartDetailerMultiModelPipe", 6), ("CMKFaceProcessMultiModelPipe", 7)):
            with self.subTest(node=class_type):
                self.assertEqual(self.key(class_type, slot, True), self.key(class_type, slot, False))

    def test_a_consumer_of_image_proceed_sees_the_switch(self):
        for class_type, slot in (("CMK_SmartDetailerMultiModelPipe", 2), ("CMKFaceProcessMultiModelPipe", 0)):
            with self.subTest(node=class_type):
                self.assertNotEqual(self.key(class_type, slot, True), self.key(class_type, slot, False))

    def test_legacy_smart_detailer_keeps_its_identity(self):
        self.assertNotEqual(self.key("CMK_SmartDetailerPipe", 6, True), self.key("CMK_SmartDetailerPipe", 6, False))


if __name__ == "__main__":
    unittest.main()
