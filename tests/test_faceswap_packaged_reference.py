import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "CMK Package · faceswap_reference.png"


class FaceSwapPackagedReferenceTests(unittest.TestCase):
    def test_reference_asset_and_loader_stay_inside_package(self):
        backend = (ROOT / "pipe" / "loaders" / "cmk_load_image.py").read_text(
            encoding="utf-8"
        )
        frontend = (
            ROOT / "web" / "js" / "cmk_packaged_faceswap_reference.js"
        ).read_text(encoding="utf-8")
        routes = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn(REFERENCE, backend)
        self.assertIn('assets" / "references', backend)
        self.assertIn('"faceswap_reference.png"', frontend)
        self.assertIn("`/cmk/reference-assets/${filename}`", frontend)
        self.assertIn('request.path.rstrip("/").endswith("/view")', routes)
        self.assertIn("_PACKAGED_REFERENCES", routes)
        self.assertTrue(
            (ROOT / "assets" / "references" / "faceswap_reference.png").is_file()
        )

    def test_40_subgraphs_compare_only_after_boundary_cache(self):
        for filename in (
            "CMK Flow · 40 FaceSwap.json",
            "CMK Flow · 40 FaceSwap · Advanced.json",
        ):
            with self.subTest(filename=filename):
                definition = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["definitions"]["subgraphs"][0]
                nodes = {node["id"]: node for node in definition["nodes"]}
                compare = next(
                    node for node in nodes.values() if node["type"] == "ImageCompare"
                )
                cache = next(
                    node
                    for node in nodes.values()
                    if node["type"] == "CMKFaceSwapBoundaryCache"
                )
                pack = next(
                    node for node in nodes.values() if node["type"] == "CMKResultPackPipe"
                )
                source_link = next(
                    link for link in definition["links"]
                    if link["target_id"] == compare["id"] and link["target_slot"] == 0
                )
                result_link = next(
                    link for link in definition["links"]
                    if link["origin_id"] == cache["id"]
                    and link["origin_slot"] == source_link["origin_slot"]
                    and link["target_id"] == pack["id"]
                )
                self.assertEqual(source_link["origin_id"], cache["id"])
                self.assertEqual(result_link["target_id"], pack["id"])
                self.assertEqual(compare["outputs"], [])
                self.assertIn(REFERENCE, {
                    value
                    for node in nodes.values()
                    if node["type"] == "CMKLoadImage"
                    for value in (node.get("widgets_values") or [])
                })

    def test_40_outer_dimensions_are_600_by_1225(self):
        for filename in (
            "CMK Flow · 40 FaceSwap.json",
            "CMK Flow · 40 FaceSwap · Advanced.json",
        ):
            with self.subTest(filename=filename):
                document = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )
                outer = document["nodes"][0]
                self.assertEqual(outer["size"], [600, 1225])
                self.assertEqual(
                    outer.get("properties", {}).get("cmkOuterSize", [600, 1225]),
                    [600, 1225],
                )

    def test_faceswap_showcases_embed_portable_reference_and_compare(self):
        for filename in (
            "CMK FaceSwap Image.json",
            "CMK Text2Image SDXL + FaceSwap.json",
            "CMK Text2Image ZIT + FaceSwap.json",
        ):
            with self.subTest(filename=filename):
                document = json.loads(
                    (ROOT / "workflows" / "showcase" / filename).read_text(
                        encoding="utf-8"
                    )
                )
                definition = next(
                    item for item in document["definitions"]["subgraphs"]
                    if "40 FaceSwap" in item["name"]
                )
                self.assertIn(
                    REFERENCE,
                    {
                        value
                        for node in definition["nodes"]
                        if node["type"] == "CMKLoadImage"
                        for value in (node.get("widgets_values") or [])
                    },
                )
                self.assertIn(
                    "ImageCompare", {node["type"] for node in definition["nodes"]}
                )


if __name__ == "__main__":
    unittest.main()
