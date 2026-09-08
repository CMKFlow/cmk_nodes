import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "subgraphs" / "CMK Flow · 25 FaceRebuild SDXL.json"


class FaceRebuild25SubgraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = json.loads(PATH.read_text(encoding="utf-8"))
        cls.outer = cls.workflow["nodes"][0]
        cls.definition = cls.workflow["definitions"]["subgraphs"][0]

    def test_outer_ui_is_compact_and_has_no_image_views(self):
        self.assertEqual(self.outer["size"], [450, 230])
        self.assertEqual(self.outer["properties"]["cmkOuterSize"], [450, 230])
        self.assertEqual(self.outer["properties"]["cmkManualSize"], [450, 230])
        self.assertEqual(
            self.outer["properties"]["proxyWidgets"],
            [["6200", "FACEREBUILD ENABLE"]],
        )
        self.assertNotIn("previewExposures", self.outer["properties"])

    def test_source_selection_is_a_direct_outer_combo(self):
        self.assertEqual(
            [item["name"] for item in self.outer["inputs"]],
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "FACEREBUILD ENABLE", "image"],
        )
        self.assertEqual(self.outer["inputs"][-1]["type"], "COMBO")
        self.assertEqual(self.outer["inputs"][-1]["label"], "source face")
        self.assertEqual(
            self.outer["widgets_values_named"],
            {"image": "CMK Package · face_reference.png"},
        )

        exposed = next(item for item in self.definition["inputs"] if item["name"] == "image")
        self.assertEqual(exposed["type"], "COMBO")
        self.assertEqual(exposed["label"], "source face")
        self.assertEqual(exposed["linkIds"], [15042])
        source = next(node for node in self.definition["nodes"] if node["type"] == "CMKLoadImage")
        source_image = next(item for item in source["inputs"] if item["name"] == "image")
        self.assertEqual(source_image["link"], 15042)
        link = next(item for item in self.definition["links"] if item["id"] == 15042)
        self.assertEqual(
            (link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"]),
            (-10, 6, source["id"], 1),
        )

    def test_visualizer_receives_live_and_boundary_result_without_outer_view(self):
        self.assertEqual(
            [item["name"] for item in self.outer["outputs"]],
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"],
        )
        provider = next(
            node for node in self.definition["nodes"] if node["type"] == "CMKVisualProvider"
        )
        self.assertEqual(
            provider["widgets_values"],
            [
                "FaceRebuild",
                25,
                "CMKInstantIDFaceRebuildSDXL",
                "sdxl",
                "sdxl.facerebuild.standard",
            ],
        )
        self.assertEqual(provider["inputs"][0]["link"], 15044)
        self.assertEqual(provider["inputs"][3]["link"], 15045)
        self.assertEqual(provider["inputs"][4]["link"], 15046)
        self.assertEqual(provider["inputs"][5]["link"], 15047)
        self.assertEqual(provider["outputs"][0]["links"], [15048])
        metadata = self.outer["properties"]["cmkVisualProviders"]
        self.assertEqual(len(metadata), 1)
        self.assertEqual(metadata[0]["live_node_id"], "6200")
        self.assertEqual(metadata[0]["stage_key"], "sdxl.facerebuild.standard")

    def test_pasteback_neck_is_enabled_by_default(self):
        rebuild = next(
            node
            for node in self.definition["nodes"]
            if node["type"] == "CMKInstantIDFaceRebuildSDXL"
        )
        pasteback_index = next(
            index
            for index, item in enumerate(rebuild["inputs"])
            if item["name"] == "PASTEBACK NECK"
        )
        widget_index = sum(
            1
            for item in rebuild["inputs"][:pasteback_index]
            if item.get("widget") is not None
        )
        self.assertIs(rebuild["widgets_values"][widget_index], True)


class FaceRebuild25AdvancedSubgraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "subgraphs" / "CMK Flow · 25 FaceRebuild SDXL · Advanced.json"
        cls.workflow = json.loads(path.read_text(encoding="utf-8"))
        cls.outer = cls.workflow["nodes"][0]
        cls.definition = cls.workflow["definitions"]["subgraphs"][0]

    def test_outer_ui_matches_advanced_contract(self):
        expected_inputs = [
            "MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "facerebuild_global_enable",
        ]
        expected_outputs = ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"]
        self.assertEqual([450, 230], self.outer["size"])
        self.assertEqual(expected_inputs, [item["name"] for item in self.outer["inputs"]])
        self.assertEqual(expected_outputs, [item["name"] for item in self.outer["outputs"]])
        self.assertEqual(expected_inputs, [item["name"] for item in self.definition["inputs"]])
        self.assertEqual(expected_outputs, [item["name"] for item in self.definition["outputs"]])
        self.assertEqual("FACEREBUILD ENABLE", self.outer["inputs"][-1]["label"])
        self.assertNotIn("proxyWidgets", self.outer["properties"])
        self.assertEqual([True], self.outer["widgets_values"])

    def test_source_and_individual_controls_remain_internal(self):
        outer_names = {item["name"] for item in self.outer["inputs"]}
        self.assertFalse(any(name.startswith("SOURCE FACE") for name in outer_names))
        self.assertFalse(any(name.startswith("FACE ") for name in outer_names))
        loaders = [node for node in self.definition["nodes"] if node["type"] == "CMKLoadImage"]
        self.assertEqual(3, len(loaders))
        rebuild = next(
            node for node in self.definition["nodes"]
            if node["type"] == "CMKInstantIDFaceRebuildAdvancedSDXL"
        )
        self.assertEqual(15042, next(item for item in rebuild["inputs"] if item["name"] == "FACEREBUILD ENABLE")["link"])

    def test_visual_provider_uses_boundary_and_global_enable(self):
        provider = next(
            node for node in self.definition["nodes"] if node["type"] == "CMKVisualProvider"
        )
        self.assertEqual(
            ["FaceRebuild", 25, "CMKInstantIDFaceRebuildAdvancedSDXL", "sdxl", "sdxl.facerebuild.advanced"],
            provider["widgets_values"],
        )
        self.assertEqual(15044, provider["inputs"][0]["link"])
        self.assertEqual(15043, provider["inputs"][1]["link"])
        self.assertEqual(15045, provider["inputs"][3]["link"])
        self.assertEqual(15046, provider["inputs"][4]["link"])
        self.assertEqual(15047, provider["inputs"][5]["link"])
        self.assertEqual([15048], provider["outputs"][0]["links"])


if __name__ == "__main__":
    unittest.main()
