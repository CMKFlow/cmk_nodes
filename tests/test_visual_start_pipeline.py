import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class VisualStartPipelineContractTests(unittest.TestCase):
    def test_create_image_places_visual_before_final_diagnostic(self):
        source = (ROOT / "pipe" / "cmk_pipe_image.py").read_text()
        tree = ast.parse(source)
        node = next(
            item for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == "CMKPipeCreateImage"
        )
        assignments = {
            item.targets[0].id: ast.literal_eval(item.value)
            for item in node.body
            if isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
            and item.targets[0].id in {"RETURN_TYPES", "RETURN_NAMES"}
        }
        self.assertEqual(assignments["RETURN_TYPES"][-2:], (
            "CMK_VISUAL_PIPE", "CMK_DIAGNOSTIC",
        ))
        self.assertEqual(assignments["RETURN_NAMES"][-2:], ("VISUAL", "diagnostic"))

    def test_regional_conditioning_places_visual_before_final_diagnostic(self):
        source = (ROOT / "pipe" / "cmk_regional_conditioning.py").read_text()
        self.assertIn('"optional": {"VISUAL": ("CMK_VISUAL_PIPE",)}', source)
        self.assertIn('"CMK_VISUAL_PIPE",', source)
        self.assertIn('normalize_visual(kwargs.get("VISUAL"))', source)
        self.assertIn('"VISUAL", "diagnostic")', source)


if __name__ == "__main__":
    unittest.main()
