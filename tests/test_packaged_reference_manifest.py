import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "references"


class PackagedReferenceManifestTests(unittest.TestCase):
    def test_loader_preview_is_driven_by_the_executed_input(self):
        source = (ROOT / "pipe" / "loaders" / "cmk_load_image.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"ui": {"images": [self._preview_descriptor(filename_string)]}', source)
        self.assertIn('"result": (pipe, loaded_image, loaded_mask, filename_string, log_pipe)', source)
        self.assertIn("folder_paths.annotated_filepath(value)", source)

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

    def test_showcase_cmk_loaders_use_available_reference_assets(self):
        selected_images = []

        def collect(value):
            if isinstance(value, dict):
                if value.get("type") in {
                    "CMKLoadImage",
                    "CMKImageLoadAndResizePipe",
                }:
                    selected = value["widgets_values"][0]
                    selected_images.append(selected)
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        showcase = ROOT / "workflows" / "showcase"
        for workflow in showcase.glob("*.json"):
            collect(json.loads(workflow.read_text(encoding="utf-8")))

        for selected in selected_images:
            if selected.endswith(" [input]"):
                continue
            filename = selected.removeprefix("CMK Package · ")
            self.assertTrue((ASSETS / filename).is_file(), selected)

        self.assertEqual(selected_images.count("controlnet_reference4.png"), 5)
        self.assertEqual(
            hashlib.sha256((ASSETS / "controlnet_reference4.png").read_bytes()).hexdigest(),
            "12d429aad2fd5572f91a9cd34b25da3b1953f45e05e93544a6fa024a4508bc54",
        )
        self.assertFalse(any(ASSETS.rglob("incoming-7783-7572.png")))


if __name__ == "__main__":
    unittest.main()
