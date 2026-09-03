import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NodeDimensionsUITests(unittest.TestCase):
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

    def test_curated_toolbox_default_dimensions_are_built_in(self):
        expected = {
            "CMKCheckpointVAELoader": (600, 240),
            "CMKControlNetPrepare": (540, 420),
            "CMKCombinedControlNetPreparePipe": (600, 1225),
            "CMKPipeCreateImage": (600, 1060),
            '"CMK Flow · 02 SDXL LoRA Stack"': (600, 125),
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


if __name__ == "__main__":
    unittest.main()
