import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "subgraphs" / "CMK Flow · FaceRebuild SDXL.json"


class FaceRebuild25SubgraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = json.loads(PATH.read_text(encoding="utf-8"))
        cls.outer = cls.workflow["nodes"][0]
        cls.definition = cls.workflow["definitions"]["subgraphs"][0]

    def test_outer_ui_is_compact_and_has_no_image_views(self):
        self.assertEqual(self.outer["size"], [300, 190])
        self.assertEqual(self.outer["properties"]["cmkOuterSize"], [300, 100])
        self.assertEqual(self.outer["properties"]["cmkManualSize"], [300, 100])
        self.assertNotIn("proxyWidgets", self.outer["properties"])
        self.assertEqual([], self.outer["properties"]["previewExposures"])

    def test_source_selection_is_a_direct_outer_combo(self):
        self.assertEqual(
            [item["name"] for item in self.outer["inputs"]],
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "FACEREBUILD ENABLE", "opt_image_file"],
        )
        self.assertEqual(self.outer["inputs"][-1]["type"], "STRING")
        self.assertEqual(
            self.outer["widgets_values_named"],
            {"FACEREBUILD ENABLE": True},
        )
        self.assertNotIn("image", [item["name"] for item in self.definition["inputs"]])
        source = next(node for node in self.definition["nodes"] if node["type"] == "CMKLoadImage")
        self.assertIn("CMK Package · face_identity_reference.png", source["widgets_values"])

    def test_process_owns_visual_and_internal_preview_compare(self):
        self.assertEqual(
            [item["name"] for item in self.outer["outputs"]],
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"],
        )
        process = next(node for node in self.definition["nodes"] if node["type"] == "CMKInstantIDFaceRebuildSDXL")
        compare = next(node for node in self.definition["nodes"] if node["type"] == "CMKVisualCompare")
        self.assertIn("VISUAL", [item["name"] for item in process["inputs"]])
        self.assertEqual(["VISUAL", "diagnostic"], [item["name"] for item in process["outputs"][-2:]])
        self.assertEqual(["VISUAL", "enable"], [item["name"] for item in compare["inputs"]])
        self.assertFalse({"CMKVisualProvider", "CMKImageCompareEnableGate", "ImageCompare"} & {
            node["type"] for node in self.definition["nodes"]
        })
        metadata = self.outer["properties"]["cmkVisualProviders"]
        self.assertEqual(len(metadata), 1)
        self.assertEqual(metadata[0]["live_node_id"], "6200")
        self.assertEqual(metadata[0]["enable_widget"], "FACEREBUILD ENABLE")
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

    def test_both_backend_processes_publish_their_own_visual(self):
        source = (ROOT / "nodes" / "instantid_face_rebuild_advanced.py").read_text(encoding="utf-8")
        self.assertIn('"optional": {"VISUAL": ("CMK_VISUAL_PIPE",)}', source)
        self.assertIn('module_type="CMKInstantIDFaceRebuildSDXL"', source)
        self.assertIn('stage_key="sdxl.facerebuild.standard"', source)
        self.assertIn('module_type="CMKInstantIDFaceRebuildAdvancedSDXL"', source)
        self.assertIn('stage_key="sdxl.facerebuild.advanced"', source)


class FaceRebuild25AdvancedSubgraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "subgraphs" / "CMK Flow · FaceRebuild SDXL · Advanced.json"
        cls.workflow = json.loads(path.read_text(encoding="utf-8"))
        cls.outer = cls.workflow["nodes"][0]
        cls.definition = cls.workflow["definitions"]["subgraphs"][0]

    def test_outer_ui_matches_advanced_contract(self):
        expected_inputs = [
            "MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "facerebuild_global_enable",
            "opt_image_file",
        ]
        expected_outputs = ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"]
        self.assertEqual([300, 190], self.outer["size"])
        self.assertEqual(expected_inputs, [item["name"] for item in self.outer["inputs"]])
        self.assertEqual(expected_outputs, [item["name"] for item in self.outer["outputs"]])
        self.assertEqual(expected_inputs, [item["name"] for item in self.definition["inputs"]])
        self.assertEqual(expected_outputs, [item["name"] for item in self.definition["outputs"]])
        self.assertEqual("FACEREBUILD ENABLE", self.outer["inputs"][-2]["label"])
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

    def test_process_owns_visual_and_global_compare_enable(self):
        process = next(node for node in self.definition["nodes"] if node["type"] == "CMKInstantIDFaceRebuildAdvancedSDXL")
        compare = next(node for node in self.definition["nodes"] if node["type"] == "CMKVisualCompare")
        self.assertIn("VISUAL", [item["name"] for item in process["inputs"]])
        self.assertEqual(["VISUAL", "diagnostic"], [item["name"] for item in process["outputs"][-2:]])
        self.assertEqual(["VISUAL", "enable"], [item["name"] for item in compare["inputs"]])
        self.assertFalse({"CMKVisualProvider", "CMKImageCompareEnableGate", "ImageCompare"} & {
            node["type"] for node in self.definition["nodes"]
        })
        metadata = self.outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("6213", metadata["live_node_id"])
        self.assertEqual("facerebuild_global_enable", metadata["enable_widget"])
        self.assertEqual("sdxl.facerebuild.advanced", metadata["stage_key"])


if __name__ == "__main__":
    unittest.main()
