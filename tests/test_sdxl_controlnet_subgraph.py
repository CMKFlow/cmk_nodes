import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SDXLControlNetSubgraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "subgraphs" / "CMK Flow · 05 ControlNet SDXL.json"
        cls.workflow = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.outer = cls.workflow["nodes"][0]
        cls.definition = cls.workflow["definitions"]["subgraphs"][0]

    def test_outer_node_is_compact_and_exposes_only_requested_widgets(self):
        self.assertEqual([450, 210], self.outer["size"])
        self.assertEqual([450, 210], self.outer["properties"]["cmkOuterSize"])
        self.assertEqual(
            [["1", "ENABLE"], ["1501", "image"]],
            self.outer["properties"]["proxyWidgets"],
        )
        self.assertEqual(
            ["ENABLE", "image", "upload"],
            [item["name"] for item in self.outer["inputs"] if item.get("widget")],
        )

    def test_contract_is_sdxl_only_and_forwards_visual(self):
        expected_inputs = ["PROCESS", "IMAGE", "LOG", "VISUAL", "ENABLE", "image"]
        expected_outputs = ["PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"]
        self.assertEqual(expected_inputs, [item["name"] for item in self.definition["inputs"]])
        self.assertEqual(expected_outputs, [item["name"] for item in self.definition["outputs"]])
        types = {item["type"] for item in self.definition["inputs"] + self.definition["outputs"]}
        self.assertIn("CMK_PROCESS_SDXL", types)
        self.assertNotIn("CMK_PROCESS_Z_IMAGE", types)
        self.assertEqual(
            {"CMKControlNetPreparePipe", "CMKLoadImage", "CMKVisualProvider"},
            {node["type"] for node in self.definition["nodes"]},
        )

    def test_outer_picker_feeds_reference_without_replacing_flow_image(self):
        loader = next(node for node in self.definition["nodes"] if node["type"] == "CMKLoadImage")
        control = next(
            node for node in self.definition["nodes"]
            if node["type"] == "CMKControlNetPreparePipe"
        )
        links = self.definition["links"]
        flow_image = next(link for link in links if link["target_id"] == control["id"] and link["target_slot"] == 1)
        self.assertEqual(-10, flow_image["origin_id"])
        reference_image = next(link for link in links if link["id"] == 11)
        reference_name = next(link for link in links if link["id"] == 12)
        self.assertEqual(loader["id"], reference_image["origin_id"])
        self.assertEqual(loader["id"], reference_name["origin_id"])
        self.assertEqual("REFERENCE IMAGE INPUT", control["inputs"][reference_image["target_slot"]]["name"])
        self.assertEqual("REFERENCE IMAGE NAME", control["inputs"][reference_name["target_slot"]]["name"])
        reference_widget_index = [
            item["name"] for item in control["inputs"] if item.get("widget")
        ].index("REFERENCE IMAGE")
        self.assertEqual(
            "CMK Package · controlnet_reference.png",
            control["widgets_values"][reference_widget_index],
        )

    def test_processed_controlnet_image_is_published_to_visualizer(self):
        prepare_source = (
            ROOT / "pipe" / "controlnet" / "cmk_controlnet_prepare.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'RETURN_NAMES = ("PROCESS", "IMAGE", "LOG", "diagnostic", "CONTROLNET IMAGE")',
            prepare_source,
        )
        provider = next(
            node for node in self.definition["nodes"]
            if node["type"] == "CMKVisualProvider"
        )
        self.assertEqual(
            ["ControlNet", 5, "", "sdxl", "sdxl.controlnet"],
            provider["widgets_values"],
        )
        image_link = next(
            link for link in self.definition["links"]
            if link["target_id"] == provider["id"] and link["target_slot"] == 0
        )
        self.assertEqual(1, image_link["origin_id"])
        self.assertEqual(4, image_link["origin_slot"])
        declaration = self.outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("cmk-CMKVisualProvider-7005", declaration["provider_id"])
        self.assertEqual("sdxl.controlnet", declaration["stage_key"])
        self.assertEqual("sdxl", declaration["branch"])

    def test_metadata_keeps_single_branch_independent_of_module_35(self):
        metadata = self.workflow["extra"]["CMKFlow"]
        self.assertEqual(["SDXL"], metadata["compatibility"])
        self.assertEqual("STABLE", metadata["status"])
        self.assertIn("Modul 35", metadata["placementNote"])
        self.assertIn("nicht erforderlich", metadata["placementNote"])


if __name__ == "__main__":
    unittest.main()
