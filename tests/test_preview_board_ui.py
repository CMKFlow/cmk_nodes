import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PreviewBoardUIContractTests(unittest.TestCase):
    def test_dynamic_normalization_preserves_manual_width_and_height(self):
        source = (
            ROOT / "web" / "js" / "cmk_preview_board_dynamic.js"
        ).read_text(encoding="utf-8")
        self.assertIn("Math.max(node.size[0], size[0])", source)
        self.assertIn("Math.max(node.size[1], size[1])", source)
        self.assertNotIn(
            "node.setSize([Math.max(node.size[0], size[0]), size[1]])",
            source,
        )


if __name__ == "__main__":
    unittest.main()
