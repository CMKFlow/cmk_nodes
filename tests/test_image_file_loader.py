import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOADER_PATH = ROOT / "pipe" / "loaders" / "cmk_load_image.py"
MAPPINGS_PATH = ROOT / "cmk_mappings.py"


class ImageFileLoaderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = LOADER_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.mapping_source = MAPPINGS_PATH.read_text(encoding="utf-8")
        cls.node = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CMKImageFileLoader"
        )

    def test_contract_has_one_named_string_output(self):
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in self.node.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"RETURN_TYPES", "RETURN_NAMES", "FUNCTION"}
        }
        self.assertEqual(assignments["RETURN_TYPES"], ("STRING",))
        self.assertEqual(assignments["RETURN_NAMES"], ("image_file",))
        self.assertEqual(assignments["FUNCTION"], "select_image_file")

    def test_uses_native_image_upload_and_preview(self):
        segment = ast.get_source_segment(self.source, self.node)
        self.assertIn('{"image_upload": True}', segment)
        self.assertIn('"ui": {"images":', segment)
        self.assertNotIn("Image.open", segment)

    def test_node_is_registered(self):
        self.assertIn('"CMKImageFileLoader": CMKImageFileLoader', self.mapping_source)
        self.assertIn('"CMKImageFileLoader": "CMK Image File Loader"', self.mapping_source)


if __name__ == "__main__":
    unittest.main()
