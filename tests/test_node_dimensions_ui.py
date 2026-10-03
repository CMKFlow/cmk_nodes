import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NodeDimensionsUITests(unittest.TestCase):
    def test_create_image_has_six_explicit_800px_standard_layouts(self):
        source = (
            ROOT / "web" / "js" / "cmk_flow_start_guidance_v51.js"
        ).read_text(encoding="utf-8")
        self.assertIn("const BASE_NODE_HEIGHT = 800", source)
        self.assertIn("const MODE_PROMPT_HEIGHTS = Object.freeze", source)

        expected_prompt_heights = {
            "text2image": {"GLOBAL PROMPT POS": 264, "PROMPT NEG": 91},
            "inpaint": {"GLOBAL PROMPT POS": 108, "PROMPT NEG": 72},
            "z-image": {"GLOBAL PROMPT POS": 362},
            "z-image-inpaint": {"GLOBAL PROMPT POS": 316},
            "hybrid": {"GLOBAL PROMPT POS": 286, "PROMPT NEG": 91},
            "hybrid-inpaint": {"GLOBAL PROMPT POS": 134, "PROMPT NEG": 72},
        }
        fixed_chrome_heights = {
            "text2image": 445,
            "inpaint": 620,
            "z-image": 438,
            "z-image-inpaint": 484,
            "hybrid": 423,
            "hybrid-inpaint": 594,
        }
        for mode, heights in expected_prompt_heights.items():
            with self.subTest(mode=mode):
                self.assertIn(f'"{mode}": Object.freeze({{', source)
                for widget, height in heights.items():
                    self.assertIn(f'"{widget}": {height}', source)
                self.assertEqual(
                    800,
                    fixed_chrome_heights[mode] + sum(heights.values()),
                )

        self.assertIn("const modePromptHeights = MODE_PROMPT_HEIGHTS[mode]", source)
        self.assertIn("panel.computeLayoutSize = undefined", source)
        self.assertNotIn("SINGLE_PROMPT_AREA_HEIGHT", source)
        self.assertNotIn('mode === "inpaint"\n', source)

    def test_create_image_prompt_fields_have_minimum_but_no_maximum_height(self):
        source = (
            ROOT / "web" / "js" / "cmk_flow_start_guidance_v51.js"
        ).read_text(encoding="utf-8")
        self.assertIn("widget.options.getMinHeight = () => height", source)
        self.assertIn("delete widget.options.getMaxHeight", source)
        self.assertIn('typeof widget.computeLayoutSize === "function"', source)
        self.assertIn("? undefined", source)
        self.assertIn('element.style.height = "100%"', source)
        self.assertIn('element.style.removeProperty("--comfy-widget-max-height")', source)

    def test_source_target_slider_centers_only_its_own_widget_row(self):
        source = (
            ROOT / "web" / "js" / "cmk_source_target_slider.js"
        ).read_text(encoding="utf-8")
        for label in ("SAMPLING START", "HYBRID BALANCE"):
            scoped_selector = (
                '[data-testid="node-widget"]:has('
                f'[data-slot="slider"][aria-label^="{label}"])'
            )
            self.assertIn(scoped_selector, source)
            self.assertNotIn(
                f'div.grid:has([data-slot="slider"][aria-label^="{label}"]) {{\n'
                "            align-items: center;",
                source,
            )

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

    def test_parameter_free_advanced_modules_keep_natural_widget_layout(self):
        stages = (
            "sdxl.detailer.advanced",
            "sdxl.facerebuild.advanced",
            "sdxl.faceprocess.advanced",
            "result.faceswap.advanced",
        )
        for version in range(1, 8):
            source = (
                ROOT / "web" / "js" / f"cmk_compact_subgraph_layout_v{version}.js"
            ).read_text(encoding="utf-8")
            self.assertIn("NATURAL_WIDGET_LAYOUT_STAGES", source)
            for stage in stages:
                self.assertIn(f'"{stage}"', source)

    @classmethod
    def setUpClass(cls):
        cls.source = (
            ROOT / "web" / "js" / "cmk_node_dimensions.js"
        ).read_text(encoding="utf-8")

    def test_context_menu_is_limited_to_cmk_nodes(self):
        self.assertIn("function isCmkNode(node)", self.source)
        self.assertIn('const MENU_LABEL = "CMK · Node Dimensions …"', self.source)
        self.assertIn("getExtraMenuOptions", self.source)

    def test_dialog_supports_complete_size_workflow(self):
        for label in (
            "WIDTH (px)",
            "HEIGHT (px)",
            "Keep aspect ratio",
            "Fit to Content",
            "Save as Type Default",
            "Clear Type Default",
            "Cancel",
            "Apply",
        ):
            self.assertIn(label, self.source)

    def test_workflow_and_type_sizes_are_persisted_without_overwriting_loads(self):
        self.assertIn("node.properties.cmkOuterSize", self.source)
        self.assertIn("node.properties.cmkManualSize", self.source)
        self.assertIn('const STORAGE_KEY = "cmk-node-size-defaults-v1"', self.source)
        self.assertIn("node.__cmkLoadedFromWorkflow", self.source)

    def test_pointer_resize_uses_vue_dom_size_for_dialog_and_clean_view(self):
        self.assertIn("function installManualResizeTracking()", self.source)
        self.assertIn("node.__cmkManualResizeActive = true", self.source)
        self.assertIn("const size = renderedSize(root)", self.source)
        self.assertIn('getPropertyValue("--node-width")', self.source)
        self.assertIn('getPropertyValue("--node-height")', self.source)
        self.assertIn("rememberDraggedSize(node, size)", self.source)
        self.assertIn("node.properties.cmkManualSize = [...size]", self.source)
        self.assertIn("node.properties.cmkCleanViewSize = [...size]", self.source)
        self.assertIn("node.properties.cmkCleanViewExpandedSize = [...size]", self.source)
        self.assertIn("const initial = normalizeSize(node, node.size?.[0], node.size?.[1])", self.source)
        clean_view = (ROOT / "web" / "js" / "cmk_clean_view_prototype.js").read_text(encoding="utf-8")
        compare = (ROOT / "web" / "js" / "cmk_image_compare_hold.js").read_text(encoding="utf-8")
        self.assertNotIn("function installManualResizeTracking()", clean_view)
        self.assertIn("if (node.__cmkManualResizeActive) return false", clean_view)
        self.assertIn("if (node.__cmkManualResizeActive) return", compare)

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
            "CMKFamilyResultMergePipe": (600, 500),
            "CMKPostProcessBoundarySDXLPipe": (600, 300),
            "CMKPostProcessBoundaryZITPipe": (600, 300),
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

    def test_requested_pipe_minimum_dimensions_are_built_in(self):
        expected = {
            "CMKSamplerPrepareSDXLPipe": (450, 360),
            "CMKRefinerPrepareSDXLPipe": (450, 360),
            "CMKInstantIDSamplerSDXLPipe": (450, 800),
            "CMKInstantIDFaceRebuildSDXL": (450, 590),
            "CMKFaceSwapImagePipe": (380, 400),
        }
        self.assertIn("const BUILTIN_MINIMUMS", self.source)
        self.assertIn("installTypeMinimum(nodeType", self.source)
        self.assertIn("function enforceBuiltInMinimum(node)", self.source)
        self.assertIn("loadedGraphNode(node)", self.source)
        for node_type, (width, height) in expected.items():
            self.assertIn(f"{node_type}: [{width}, {height}]", self.source)

    def test_standard_facerebuild_uses_a_minimum_not_a_fixed_size(self):
        source = (
            ROOT / "web" / "js" / "cmk_instantid_face_rebuild_advanced_ui.js"
        ).read_text(encoding="utf-8")
        self.assertIn("const STANDARD_MIN_SIZE = [450, 590]", source)
        self.assertIn("function enforceStandardMinimum(node)", source)
        self.assertNotIn("STANDARD_NODE_SIZE", source)

    def test_core_sdxl_flow_modules_share_the_compact_outer_size(self):
        expected_manual_sizes = {
            "CMK Flow · 05 ControlNet SDXL.json": [300, 100],
            "CMK Flow · 10 KSampler SDXL 1st Pass.json": [300, 190],
            "CMK Flow · 15 InstantID-Sampler SDXL.json": [300, 190],
            "CMK Flow · 20 Refiner SDXL.json": [300, 190],
            "CMK Flow · Detailer SDXL.json": [300, 100],
            "CMK Flow · Detailer SDXL · Advanced.json": [300, 100],
            "CMK Flow · FaceRebuild SDXL.json": [300, 100],
            "CMK Flow · FaceRebuild SDXL · Advanced.json": [300, 100],
            "CMK Flow · FaceProcess SDXL.json": [300, 100],
        }
        for name, manual_size in expected_manual_sizes.items():
            with self.subTest(name=name):
                document = json.loads((ROOT / "subgraphs" / name).read_text(encoding="utf-8"))
                outer = document["nodes"][0]
                self.assertEqual([300, 190], outer["size"])
                self.assertEqual(manual_size, outer["properties"]["cmkOuterSize"])
                self.assertEqual(manual_size, outer["properties"]["cmkManualSize"])


if __name__ == "__main__":
    unittest.main()
