import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NodeDimensionsUITests(unittest.TestCase):
    def test_visualizer_respects_manual_node_dimensions(self):
        source = (ROOT / "web" / "js" / "cmk_visualizer.js").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("this.setSize(", source)
        self.assertIn("min-width:0;min-height:0;overflow:hidden", source)
        self.assertNotIn("min-height:160px", source)

    def test_compact_flow_widgets_share_a_bottom_aligned_grid(self):
        source = (ROOT / "web" / "js" / "cmk_compact_subgraph_layout_v1.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("const COMPACT_FLOW_SIZE = [450, 230]", source)
        self.assertIn("function applyLayout(node)", source)
        self.assertIn("node.widgets_start_y = startY", source)
        self.assertIn("widget.y = y", source)
        self.assertIn("this._arrangeWidgetInputSlots?.", source)
        self.assertIn("loadedGraphNode(node)", source)
        self.assertIn("retryInstall(node)", source)
        self.assertIn('const ADVANCED_SPACER_NAME = "cmk_advanced_combo_spacer"', source)
        self.assertIn("serialize: false", source)
        self.assertIn('name: "cmk.compact_subgraph_layout.v1"', source)
        dimensions = (ROOT / "web" / "js" / "cmk_node_dimensions.js").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("alignCompactFlowWidgets", dimensions)

    @classmethod
    def setUpClass(cls):
        cls.source = (
            ROOT / "web" / "js" / "cmk_node_dimensions.js"
        ).read_text(encoding="utf-8")

    def test_context_menu_is_limited_to_cmk_nodes(self):
        self.assertIn("function isCmkNode(node)", self.source)
        self.assertIn('const MENU_LABEL = "CMK · Node-Dimensionen …"', self.source)
        self.assertIn("getExtraMenuOptions", self.source)

    def test_dialog_supports_complete_size_workflow(self):
        for label in (
            "BREITE (px)",
            "HÖHE (px)",
            "Seitenverhältnis beibehalten",
            "Auf Inhalt anpassen",
            "Als Typ-Standard speichern",
            "Typ-Standard löschen",
        ):
            self.assertIn(label, self.source)

    def test_workflow_and_type_sizes_are_persisted_without_overwriting_loads(self):
        self.assertIn("node.properties.cmkOuterSize", self.source)
        self.assertIn("node.properties.cmkManualSize", self.source)
        self.assertIn('const STORAGE_KEY = "cmk-node-size-defaults-v1"', self.source)
        self.assertIn("node.__cmkLoadedFromWorkflow", self.source)

    def test_sdxl_lora_stack_has_no_forced_type_size(self):
        self.assertNotIn('"CMK Flow · 02 SDXL LoRA Stack":', self.source)

    def test_curated_toolbox_default_dimensions_are_built_in(self):
        expected = {
            "CMKCheckpointVAELoader": (600, 240),
            "CMKControlNetPrepare": (540, 420),
            "CMKCombinedControlNetPreparePipe": (600, 1225),
            "CMKPipeCreateImage": (600, 960),
            "CMKControlNetPreparePipe": (600, 1225),
            "CMKZITControlNetPreparePipe": (600, 1225),
            "CMKCheckpointVAELoaderPipe": (600, 165),
            "CMKImageLoadAndResizePipe": (600, 860),
            "CMKLoadImage": (600, 1225),
            "CMKSwapImageLoaderPipe": (1200, 800),
            "CMKFamilyResultMergePipe": (300, 250),
            "CMKDiagnosticConcat": (320, 135),
            "CMK_FaceProcess": (540, 1015),
            "CMKFaceSwapImage": (400, 395),
            "CMKFaceSwapVideo": (540, 435),
            "CMKFaceSwapVideoLoader": (1180, 810),
            "CMKLoRATextLoader": (280, 145),
            "CMKMergeAndSaveVideo": (600, 780),
            "CMK_SmartDetailer": (540, 775),
            "CMK_SourcePathInfo": (240, 100),
            "CMKSplitVideoIntoSegments": (600, 910),
            "CMKVideoCompare": (1180, 600),
        }
        for node_type, (width, height) in expected.items():
            self.assertIn(f"{node_type}: [{width}, {height}]", self.source)

    def test_core_sdxl_flow_modules_share_the_compact_outer_size(self):
        expected_manual_sizes = {
            "CMK Flow · 05 ControlNet SDXL.json": [450, 230],
            "CMK Flow · 10 KSampler SDXL 1st Pass.json": [450, 222],
            "CMK Flow · 15 InstantID-Sampler SDXL.json": [450, 222],
            "CMK Flow · 20 Refiner SDXL.json": [450, 222],
            "CMK Flow · 23 Detailer SDXL.json": [450, 222],
            "CMK Flow · 23 Detailer SDXL · Advanced.json": [450, 222],
            "CMK Flow · 25 FaceRebuild SDXL.json": [450, 222],
            "CMK Flow · 25 FaceRebuild SDXL · Advanced.json": [450, 230],
            "CMK Flow · 30 FaceProcess SDXL.json": [450, 230],
        }
        for name, manual_size in expected_manual_sizes.items():
            with self.subTest(name=name):
                document = json.loads((ROOT / "subgraphs" / name).read_text(encoding="utf-8"))
                outer = document["nodes"][0]
                self.assertEqual([450, 230], outer["size"])
                self.assertEqual(manual_size, outer["properties"]["cmkOuterSize"])
                self.assertEqual(manual_size, outer["properties"]["cmkManualSize"])


if __name__ == "__main__":
    unittest.main()
