import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "references"


class PackagedReferenceManifestTests(unittest.TestCase):
    def test_manifest_hashes_match_packaged_files(self):
        manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "cmk.reference-assets.v1")
        for item in manifest["assets"]:
            with self.subTest(package_file=item["package_file"]):
                data = (ASSETS / item["package_file"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])

    def test_distinct_face_roles_use_distinct_files(self):
        manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
        by_name = {item["package_file"]: item["sha256"] for item in manifest["assets"]}
        names = (
            "face_reference.png",
            "face_identity_reference.png",
            "faceswap_reference.png",
        )
        self.assertEqual(len({by_name[name] for name in names}), len(names))

    def test_showcases_use_the_documented_reference_roles(self):
        expected = {
            "Flow #03 - CMK Inpaint ZIT · Experimentell.json": {
                5188: "CMK Package · inpaint_reference3.png",
            },
            "Flow #09 - CMK Inpaint SDXL.json": {
                5348: "CMK Package · inpaint_reference.png",
            },
            "Flow #10 - CMK Inpaint InstantID SDXL.json": {
                5348: "CMK Package · inpaint_reference2.png",
                1501: "CMK Package · face_reference.png",
            },
            "Ref #01 - CMK Detailer.json": {
                5205: "CMK Package · detailer_reference.png",
            },
            "Ref #02 - CMK FaceRebuild.json": {
                5205: "CMK Package · faceswap_reference.png",
                6202: "CMK Package · face_identity_reference.png",
            },
            "Ref #04 - CMK FaceSwap.json": {
                5205: "CMK Package · faceswap_reference.png",
                5063: "CMK Package · face_identity_reference.png",
            },
            "CMK: FaceSwap vs FaceRebuild.json": {
                6244: "CMK Package · face_identity_reference.png",
                6250: "CMK Package · faceswap_reference.png",
            },
            "CMK - Full Flow.json": {
                6202: "CMK Package · face_identity_reference.png",
            },
        }

        def collect(value, result, require_image_sync=False):
            if isinstance(value, dict):
                if value.get("id") in result and value.get("type") in {
                    "CMKLoadImage",
                    "CMKImageLoadAndResizePipe",
                }:
                    selected = value["widgets_values"][0]
                    if require_image_sync:
                        self.assertEqual(value.get("properties", {}).get("image"), selected)
                    result[value["id"]] = selected
                for child in value.values():
                    collect(child, result, require_image_sync)
            elif isinstance(value, list):
                for child in value:
                    collect(child, result, require_image_sync)

        showcase = ROOT / "workflows" / "showcase"
        for filename, node_images in expected.items():
            with self.subTest(filename=filename):
                found = {node_id: None for node_id in node_images}
                collect(
                    json.loads((showcase / filename).read_text(encoding="utf-8")),
                    found,
                    filename in {
                        "Flow #03 - CMK Inpaint ZIT · Experimentell.json",
                        "Flow #09 - CMK Inpaint SDXL.json",
                    },
                )
                self.assertEqual(found, node_images)


if __name__ == "__main__":
    unittest.main()
