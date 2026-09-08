import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBGRAPHS = ROOT / "subgraphs"


class ControlNet05SubgraphTests(unittest.TestCase):
    def load(self, name):
        return json.loads((SUBGRAPHS / name).read_text())

    def assert_reference_layout(self, payload):
        outer = payload["nodes"][0]
        graph = payload["definitions"]["subgraphs"][0]
        self.assertEqual([450, 230], outer["size"])
        self.assertEqual([450, 230], outer["properties"]["cmkOuterSize"])
        self.assertEqual([450, 230], outer["properties"]["cmkManualSize"])
        self.assertEqual(
            [["1", "ENABLE"], ["1501", "image"]],
            outer["properties"]["proxyWidgets"],
        )
        self.assertEqual("#334155", outer["color"])
        self.assertEqual("#1f2937", outer["bgcolor"])
        self.assertEqual([8360, -360, 1230, 1310], graph["groups"][0]["bounding"])
        self.assertNotIn("CMK Boundary Cache", {node["type"] for node in graph["nodes"]})

    def test_zit_contract_and_visual_provider(self):
        payload = self.load("CMK Flow · 05 ControlNet ZIT.json")
        self.assert_reference_layout(payload)
        outer = payload["nodes"][0]
        self.assertEqual(
            ["PROCESS", "IMAGE", "LOG", "VISUAL", "ENABLE", "image", "upload"],
            [item["name"] for item in outer["inputs"]],
        )
        self.assertEqual(
            ["PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"],
            [item["name"] for item in outer["outputs"]],
        )
        provider = outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("zit", provider["branch"])
        self.assertEqual("zit.controlnet", provider["stage_key"])
        graph = payload["definitions"]["subgraphs"][0]
        visual_provider = next(
            node for node in graph["nodes"] if node["type"] == "CMKVisualProvider"
        )
        self.assertEqual(
            "CMKZITControlNetPreparePipe",
            visual_provider["widgets_values_named"]["live_node_type"],
        )

    def test_combined_has_only_split_process_ports(self):
        payload = self.load("CMK Flow · 05 ControlNet Combined.json")
        self.assert_reference_layout(payload)
        outer = payload["nodes"][0]
        self.assertEqual(
            ["PROCESS SDXL", "PROCESS ZIT", "IMAGE", "LOG", "VISUAL", "ENABLE", "image", "upload"],
            [item["name"] for item in outer["inputs"]],
        )
        self.assertEqual(
            ["PROCESS SDXL", "PROCESS ZIT", "IMAGE", "LOG", "VISUAL", "diagnostic"],
            [item["name"] for item in outer["outputs"]],
        )
        provider = outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("", provider["branch"])
        self.assertEqual("controlnet", provider["stage_key"])
        metadata = payload["extra"]["CMKFlow"]
        self.assertEqual("CMK Flow · 05 ControlNet SDXL", metadata["variantOf"])
        self.assertEqual("COMBINED", metadata["variantLabel"])


if __name__ == "__main__":
    unittest.main()
