import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_SUBGRAPHS = (
    "CMK Flow · 15 InstantID-Sampler SDXL.json",
    "CMK Flow · FaceRebuild SDXL.json",
    "CMK Flow · FaceRebuild SDXL · Advanced.json",
    "CMK Flow · FaceSwap.json",
    "CMK Flow · FaceSwap · Advanced.json",
)


class IdentityImageFileContractTests(unittest.TestCase):
    def test_image_input_exposes_selected_file_reference(self):
        source = (ROOT / "pipe/loaders/cmk_image_load_resize.py").read_text(encoding="utf-8")
        self.assertIn('"MASK",\n        "STRING",', source)
        self.assertIn('"diagnostic", "MASK", "image_file")', source)
        self.assertIn("resized_mask, image_name", source)

    def test_internal_loader_accepts_optional_file_override(self):
        source = (ROOT / "pipe/loaders/cmk_load_image.py").read_text(encoding="utf-8")
        self.assertIn('"opt_image_file": ("STRING", {"forceInput": True})', source)
        self.assertIn("if opt_image_file is not None and str(opt_image_file).strip():", source)

    def test_identity_subgraphs_expose_and_route_optional_file(self):
        for filename in IDENTITY_SUBGRAPHS:
            with self.subTest(filename=filename):
                document = json.loads((ROOT / "subgraphs" / filename).read_text(encoding="utf-8"))
                definition = document["definitions"]["subgraphs"][0]
                external = next(item for item in definition["inputs"] if item["name"] == "opt_image_file")
                self.assertEqual("STRING", external["type"])
                external_slot = definition["inputs"].index(external)
                loaders = [node for node in definition["nodes"] if node["type"] == "CMKLoadImage"]
                self.assertTrue(loaders)
                self.assertEqual(len(loaders), len(external["linkIds"]))
                external_links = {
                    link["id"]: link
                    for link in definition["links"]
                    if link["id"] in external["linkIds"]
                }
                for loader in loaders:
                    socket = next(item for item in loader["inputs"] if item["name"] == "opt_image_file")
                    self.assertIn(socket["link"], external["linkIds"])
                    self.assertEqual(-10, external_links[socket["link"]]["origin_id"])
                    self.assertEqual(external_slot, external_links[socket["link"]]["origin_slot"])

    def test_full_flow_embeds_the_same_identity_contract(self):
        document = json.loads((ROOT / "workflows/showcase/CMK 2.5 · Full Flow .json").read_text(encoding="utf-8"))
        expected = {name.removesuffix(".json") for name in IDENTITY_SUBGRAPHS}
        definitions = {
            definition["name"]: definition
            for definition in document["definitions"]["subgraphs"]
            if definition["name"] in expected
        }
        self.assertEqual(expected, set(definitions))
        for definition in definitions.values():
            external = next(item for item in definition["inputs"] if item["name"] == "opt_image_file")
            self.assertEqual("STRING", external["type"])


if __name__ == "__main__":
    unittest.main()
