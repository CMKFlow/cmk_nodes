import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Detailer23VisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · 23 Detailer SDXL.json").read_text(
                encoding="utf-8"
            )
        )
        cls.outer = cls.document["nodes"][0]
        cls.definition = cls.document["definitions"]["subgraphs"][0]

    def test_outer_contract_is_compact_and_canonical(self):
        expected_inputs = [
            "MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL",
            "detailer_global_enable", "model_name",
        ]
        expected_outputs = ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"]
        self.assertEqual([450, 230], self.outer["size"])
        self.assertEqual(expected_inputs, [item["name"] for item in self.outer["inputs"]])
        self.assertEqual(expected_outputs, [item["name"] for item in self.outer["outputs"]])
        self.assertEqual(expected_inputs, [item["name"] for item in self.definition["inputs"]])
        self.assertEqual(expected_outputs, [item["name"] for item in self.definition["outputs"]])
        self.assertNotIn("proxyWidgets", self.outer["properties"])
        self.assertEqual(self.outer["inputs"][-1]["type"], "COMBO")
        self.assertEqual(
            self.outer["widgets_values"],
            [True, "bbox/face/face_yolov8s.pt"],
        )
        self.assertEqual(
            self.outer["widgets_values_named"]["model_name"],
            "bbox/face/face_yolov8s.pt",
        )

    def test_smart_detailer_owns_log_and_diagnostic_transport(self):
        self.assertNotIn("CMKLogConcat", {node["type"] for node in self.definition["nodes"]})
        detailer = next(node for node in self.definition["nodes"] if node["type"] == "CMK_SmartDetailerPipe")
        self.assertEqual(detailer["widgets_values"][2], "bbox/face/face_yolov8s.pt")
        self.assertEqual(
            detailer["widgets_values_named"]["model_name"],
            "bbox/face/face_yolov8s.pt",
        )
        model_input = next(item for item in detailer["inputs"] if item["name"] == "model_name")
        self.assertEqual(model_input["link"], 20285)
        model_link = next(item for item in self.definition["links"] if item["id"] == 20285)
        self.assertEqual(
            (model_link["origin_id"], model_link["origin_slot"], model_link["target_id"], model_link["target_slot"]),
            (-10, 6, detailer["id"], 5),
        )
        self.assertEqual(["opt_log", "opt_diagnostic"], [item["name"] for item in detailer["inputs"][-2:]])
        self.assertEqual(["LOG", "diagnostic", "DETAILER IMAGE"], [item["name"] for item in detailer["outputs"][-3:]])

    def test_detailer_provider_uses_before_after_and_authoritative_result(self):
        provider = next(
            node for node in self.definition["nodes"]
            if node["type"] == "CMKVisualProvider"
        )
        self.assertEqual(
            [
                "Detailer",
                23,
                "CMK_SmartDetailerPipe",
                "sdxl",
                "sdxl.detailer.standard",
            ],
            provider["widgets_values"],
        )
        links = {link["id"]: link for link in self.definition["links"]}
        by_name = {item["name"]: links[item["link"]] for item in provider["inputs"] if item.get("link") is not None}
        self.assertEqual((-10, 2), (by_name["BEFORE"]["origin_id"], by_name["BEFORE"]["origin_slot"]))
        detailer = next(node for node in self.definition["nodes"] if node["type"] == "CMK_SmartDetailerPipe")
        self.assertEqual((detailer["id"], 5), (by_name["IMAGE"]["origin_id"], by_name["IMAGE"]["origin_slot"]))
        self.assertEqual((detailer["id"], 5), (by_name["AFTER"]["origin_id"], by_name["AFTER"]["origin_slot"]))
        self.assertEqual((-10, 5), (by_name["enable"]["origin_id"], by_name["enable"]["origin_slot"]))
        declaration = self.outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("detailer_global_enable", declaration["enable_widget"])
        self.assertEqual(f"cmk-CMKVisualProvider-{provider['id']}", declaration["provider_id"])
        self.assertEqual("sdxl.detailer.standard", declaration["stage_key"])
        self.assertTrue(declaration["capabilities"]["compare"])
        self.assertTrue(declaration["capabilities"]["live"])
        self.assertEqual(str(detailer["id"]), declaration["live_node_id"])

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
        self.assertIn('return ["DETAILER"]', method)


class Detailer23AdvancedVisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · 23 Detailer SDXL · Advanced.json").read_text(
                encoding="utf-8"
            )
        )
        cls.outer = cls.document["nodes"][0]
        cls.definition = cls.document["definitions"]["subgraphs"][0]

    def test_outer_contract_matches_standard_detailer(self):
        expected_inputs = ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "detailer_global_enable"]
        expected_outputs = ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"]
        self.assertEqual([450, 230], self.outer["size"])
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
            self.assertEqual((previous["id"], 4), (inputs["opt_diagnostic"]["origin_id"], inputs["opt_diagnostic"]["origin_slot"]))

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

    def test_provider_declares_all_advanced_live_sources(self):
        provider = next(node for node in self.definition["nodes"] if node["type"] == "CMKVisualProvider")
        detailers = [node for node in self.definition["nodes"] if node["type"] == "CMK_SmartDetailerPipe"]
        declaration = self.outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("sdxl.detailer.advanced", provider["widgets_values"][4])
        self.assertEqual("sdxl.detailer.advanced", declaration["stage_key"])
        self.assertEqual(f"cmk-CMKVisualProvider-{provider['id']}", declaration["provider_id"])
        self.assertEqual([str(node["id"]) for node in detailers], declaration["live_node_ids"])


if __name__ == "__main__":
    unittest.main()
