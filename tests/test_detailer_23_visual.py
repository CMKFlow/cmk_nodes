import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Detailer23VisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · Detailer SDXL.json").read_text(
                encoding="utf-8"
            )
        )
        cls.outer = cls.document["nodes"][0]
        cls.definition = cls.document["definitions"]["subgraphs"][0]

    def test_outer_contract_is_compact_and_canonical(self):
        expected_inputs = [
            "MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL",
            "detailer_global_enable",
        ]
        expected_outputs = ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"]
        self.assertEqual([300, 190], self.outer["size"])
        self.assertEqual(expected_inputs, [item["name"] for item in self.outer["inputs"]])
        self.assertEqual(expected_outputs, [item["name"] for item in self.outer["outputs"]])
        self.assertEqual(expected_inputs, [item["name"] for item in self.definition["inputs"]])
        self.assertEqual(expected_outputs, [item["name"] for item in self.definition["outputs"]])
        self.assertNotIn("proxyWidgets", self.outer["properties"])
        self.assertEqual(
            self.outer["widgets_values"],
            [True],
        )
        self.assertNotIn("model_name", self.outer["widgets_values_named"])

    def test_smart_detailer_owns_log_and_diagnostic_transport(self):
        self.assertNotIn("CMKLogConcat", {node["type"] for node in self.definition["nodes"]})
        detailer = next(node for node in self.definition["nodes"] if node["type"] == "CMK_SmartDetailerPipe")
        self.assertEqual(detailer["widgets_values"][2], "bbox/face/face_yolov8s.pt")
        self.assertEqual(
            detailer["widgets_values_named"]["model_name"],
            "bbox/face/face_yolov8s.pt",
        )
        model_input = next(item for item in detailer["inputs"] if item["name"] == "model_name")
        self.assertIsNone(model_input["link"])
        input_names = [item["name"] for item in detailer["inputs"]]
        self.assertIn("opt_log", input_names)
        self.assertIn("VISUAL", input_names)
        self.assertIn("opt_diagnostic", input_names)
        self.assertEqual(
            ["LOG", "VISUAL", "diagnostic", "DETAILER IMAGE"],
            [item["name"] for item in detailer["outputs"][-4:]],
        )

    def test_detailer_process_owns_visual_and_internal_compare(self):
        links = {link["id"]: link for link in self.definition["links"]}
        detailer = next(node for node in self.definition["nodes"] if node["type"] == "CMK_SmartDetailerPipe")
        visual_input = next(item for item in detailer["inputs"] if item["name"] == "VISUAL")
        visual_output_index = next(index for index, item in enumerate(detailer["outputs"]) if item["name"] == "VISUAL")
        self.assertEqual(-10, links[visual_input["link"]]["origin_id"])
        compare = next(node for node in self.definition["nodes"] if node["type"] == "CMKVisualCompare")
        compare_link = links[compare["inputs"][0]["link"]]
        self.assertEqual((detailer["id"], visual_output_index), (compare_link["origin_id"], compare_link["origin_slot"]))
        self.assertNotIn("CMKVisualProvider", {node["type"] for node in self.definition["nodes"]})
        self.assertNotIn("CMKImageCompareEnableGate", {node["type"] for node in self.definition["nodes"]})
        self.assertNotIn("ImageCompare", {node["type"] for node in self.definition["nodes"]})
        declaration = self.outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("detailer_global_enable", declaration["enable_widget"])
        self.assertEqual("sdxl.detailer.standard", declaration["stage_key"])
        self.assertTrue(declaration["capabilities"]["compare"])
        self.assertTrue(declaration["capabilities"]["live"])

    def test_lazy_status_materializes_detailer_before_cache_lookup(self):
        source = (ROOT / "nodes" / "image" / "smart_detailer.py").read_text(
            encoding="utf-8"
        )
        method = source.split("    def check_lazy_status(", 1)[1].split(
            "    def _load_cached_result(", 1
        )[0]
        self.assertLess(
            method.index("if DETAILER is None:"),
            method.index("pickle_available("),
        )
        self.assertIn('missing_transport.append("DETAILER")', method)

    def test_native_preview_is_cleared_at_every_workflow_start(self):
        source = (ROOT / "web" / "js" / "cmk_smart_detailer_labels.js").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'api.addEventListener("execution_start", clearPreviousPreviewsAtWorkflowStart)',
            source,
        )
        self.assertIn("node.imgs = [];", source)
        self.assertIn("node.imageIndex = null;", source)

    def test_smart_detailer_does_not_publish_a_second_static_preview(self):
        source = (ROOT / "nodes" / "image" / "smart_detailer.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("PreviewImage().save_images", source)
        self.assertNotIn('return {"ui": ui, "result": result}', source)


class Detailer23AdvancedVisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · Detailer SDXL · Advanced.json").read_text(
                encoding="utf-8"
            )
        )
        cls.outer = cls.document["nodes"][0]
        cls.definition = cls.document["definitions"]["subgraphs"][0]

    def test_outer_contract_matches_standard_detailer(self):
        expected_inputs = ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "detailer_global_enable"]
        expected_outputs = ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"]
        self.assertEqual([300, 190], self.outer["size"])
        self.assertEqual(expected_inputs, [item["name"] for item in self.definition["inputs"]])
        self.assertEqual(expected_outputs, [item["name"] for item in self.definition["outputs"]])

    def test_three_detailers_chain_log_and_diagnostic_without_concat_nodes(self):
        nodes = self.definition["nodes"]
        detailers = [node for node in nodes if node["type"] == "CMK_SmartDetailerPipe"]
        self.assertEqual(3, len(detailers))
        self.assertNotIn("CMKLogConcat", {node["type"] for node in nodes})
        self.assertNotIn("CMKDiagnosticConcat", {node["type"] for node in nodes})
        links = {link["id"]: link for link in self.definition["links"]}
        for previous, current in zip(detailers, detailers[1:]):
            inputs = {item["name"]: links[item["link"]] for item in current["inputs"] if item.get("link") is not None}
            self.assertEqual((previous["id"], 3), (inputs["opt_log"]["origin_id"], inputs["opt_log"]["origin_slot"]))
            self.assertEqual((previous["id"], 5), (inputs["opt_diagnostic"]["origin_id"], inputs["opt_diagnostic"]["origin_slot"]))

    def test_serialized_link_endpoints_are_consistent(self):
        links = {link["id"]: link for link in self.definition["links"]}
        for node in self.definition["nodes"]:
            for slot, item in enumerate(node.get("inputs", [])):
                if item.get("link") is None:
                    continue
                link = links[item["link"]]
                self.assertEqual((node["id"], slot), (link["target_id"], link["target_slot"]))
            for slot, item in enumerate(node.get("outputs", [])):
                for link_id in item.get("links") or []:
                    link = links[link_id]
                    self.assertEqual((node["id"], slot), (link["origin_id"], link["origin_slot"]))

    def test_concat_owns_advanced_visual(self):
        detailers = [node for node in self.definition["nodes"] if node["type"] == "CMK_SmartDetailerPipe"]
        concat = next(node for node in self.definition["nodes"] if node["type"] == "CMK_SEGSConcate")
        self.assertIn("VISUAL", [item["name"] for item in concat["inputs"]])
        self.assertIn("VISUAL", [item["name"] for item in concat["outputs"]])
        self.assertTrue(all("VISUAL" in [item["name"] for item in node["outputs"]] for node in detailers))
        self.assertNotIn("CMKVisualProvider", {node["type"] for node in self.definition["nodes"]})
        declaration = self.outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("sdxl.detailer.advanced", declaration["stage_key"])


if __name__ == "__main__":
    unittest.main()
