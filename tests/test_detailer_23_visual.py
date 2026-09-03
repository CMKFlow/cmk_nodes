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
        expected_inputs = ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "detailer_global_enable"]
        expected_outputs = ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"]
        self.assertEqual([450, 210], self.outer["size"])
        self.assertEqual(expected_inputs, [item["name"] for item in self.outer["inputs"]])
        self.assertEqual(expected_outputs, [item["name"] for item in self.outer["outputs"]])
        self.assertEqual(expected_inputs, [item["name"] for item in self.definition["inputs"]])
        self.assertEqual(expected_outputs, [item["name"] for item in self.definition["outputs"]])
        self.assertNotIn("proxyWidgets", self.outer["properties"])

    def test_smart_detailer_owns_log_and_diagnostic_transport(self):
        self.assertNotIn("CMKLogConcat", {node["type"] for node in self.definition["nodes"]})
        detailer = next(node for node in self.definition["nodes"] if node["type"] == "CMK_SmartDetailerPipe")
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
                True,
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


if __name__ == "__main__":
    unittest.main()
