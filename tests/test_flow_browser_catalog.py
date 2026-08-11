import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FlowBrowserCatalogTests(unittest.TestCase):
    EXPECTED_FLOWS = {
        "02 SDXL LoRA Stack",
        "10 KSampler SDXL 1st Pass",
        "10 KSampler Z-Image Turbo",
        "20 Refiner SDXL",
        "25 Detailer SDXL",
        "25 Detailer SDXL · Advanced",
        "30 FaceProcess SDXL",
        "30 FaceProcess SDXL · Advanced",
        "40 FaceSwap",
        "40 FaceSwap · Advanced",
        "90 Upscale & Save",
    }
    EXPECTED_REFERENCE_WORKFLOWS = {
        "CMK Detailer.json",
        "CMK FaceProcess.json",
        "CMK FaceSwap Image.json",
        "CMK FaceSwap Video.json",
        "CMK Full Flow.json",
        "CMK Inpaint SDXL.json",
        "CMK Inpaint ZIT - experimentell.json",
        "CMK Text2Image SDXL + Detailer.json",
        "CMK Text2Image SDXL + FaceProcess.json",
        "CMK Text2Image SDXL + FaceSwap.json",
        "CMK Text2Image SDXL ControlNet.json",
        "CMK Text2Image SDXL.json",
        "CMK Text2Image ZIT + FaceSwap.json",
        "CMK Text2Image ZIT ControlNet.json",
        "CMK Text2Image ZIT.json",
    }

    def test_reference_catalog_contains_only_confirmed_workflows(self):
        showcase = ROOT / "workflows" / "showcase"
        metadata_root = showcase / "metadata"
        workflows = {path.name for path in showcase.glob("*.json")}
        metadata_files = {path.name for path in metadata_root.glob("*.json")}
        self.assertEqual(workflows, self.EXPECTED_REFERENCE_WORKFLOWS)
        self.assertEqual(metadata_files, self.EXPECTED_REFERENCE_WORKFLOWS)

        for filename in sorted(workflows):
            with self.subTest(filename=filename):
                workflow = json.loads((showcase / filename).read_text(encoding="utf-8"))
                self.assertIsInstance(workflow.get("nodes"), list)
                metadata = json.loads(
                    (metadata_root / filename).read_text(encoding="utf-8")
                )
                self.assertTrue(metadata.get("published"))
                for field in (
                    "displayName", "category", "category_en", "description",
                    "description_en", "cmkHighlight", "cmkHighlight_en",
                ):
                    self.assertTrue(metadata.get(field), f"{filename}: {field}")
                self.assertTrue(metadata.get("previews"), filename)
                for preview in metadata["previews"]:
                    asset = ROOT / "web" / preview["src"]
                    self.assertTrue(asset.is_file(), f"{filename}: {preview['src']}")
                    self.assertGreater(asset.stat().st_size, 0)

    def test_technical_reference_directory_keeps_only_video_reference(self):
        reference_root = ROOT / "workflows" / "reference"
        self.assertEqual(
            {path.name for path in reference_root.glob("*.json")},
            {"CMK_FaceSwap_Video_Reference_v2.2.json"},
        )

    def test_all_curated_flow_subgraphs_are_published_with_real_previews(self):
        published = set()
        for path in sorted((ROOT / "subgraphs").glob("CMK Flow · *.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            metadata = document.get("extra", {}).get("CMKFlow", {})
            if not metadata.get("published"):
                continue
            published.add(metadata["displayName"])
            self.assertTrue(metadata.get("previews"), path.name)
            for preview in metadata["previews"]:
                asset = ROOT / "web" / preview["src"]
                self.assertTrue(asset.is_file(), f"{path.name}: {preview['src']}")
                self.assertGreater(asset.stat().st_size, 0)
        self.assertEqual(published, self.EXPECTED_FLOWS)

    def test_faceswap_flows_expose_both_families_and_native_engine(self):
        expected = {"SDXL", "Z-Image Turbo", "CMK Native FaceSwap"}
        for filename in (
            "CMK Flow · 40 FaceSwap.json",
            "CMK Flow · 40 FaceSwap · Advanced.json",
        ):
            with self.subTest(filename=filename):
                metadata = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["extra"]["CMKFlow"]
                self.assertEqual(set(metadata["compatibility"]), expected)
                self.assertNotIn("ReActor", metadata["compatibility"])

    def test_custom_flow_nodes_reference_existing_previews(self):
        metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]
        self.assertFalse(
            any(
                "LaMa" in str(feature)
                for feature in metadata["CMKPipeCreateImage"]["features"]
            )
        )
        required_catalog_fields = {
            "category", "compatibility", "version", "author", "status"
        }
        for node_name, entry in metadata.items():
            self.assertTrue(
                required_catalog_fields.issubset(entry),
                f"{node_name}: incomplete catalog column metadata",
            )
            self.assertTrue(entry["compatibility"], node_name)
            self.assertNotEqual(entry["version"], "—", node_name)
            self.assertTrue(entry["author"], node_name)
        for node_name, entry in metadata.items():
            for preview in entry.get("previews", []):
                asset = ROOT / "web" / preview["src"]
                self.assertTrue(asset.is_file(), f"{node_name}: {preview['src']}")
                self.assertGreater(asset.stat().st_size, 0)

    def test_english_browser_localizes_preview_tab_labels(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('Modul: "Module"', source)
        self.assertIn('Aufbau: "Structure"', source)
        self.assertIn('`View ${index + 1}`', source)

        english = json.loads(
            (ROOT / "web" / "browser_content_en.json").read_text(encoding="utf-8")
        )
        create_features = english["flows"]["CMKPipeCreateImage"]["features"]
        self.assertFalse(any("LaMa" in str(feature) for feature in create_features))

    def test_flow_browser_uses_custom_node_catalog_metadata(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("metadata.category || nodeDef.category", source)
        self.assertIn('metadata.version || "1.0.0"', source)
        self.assertIn("metadata.author ||", source)
        self.assertIn("metadata.compatibility", source)
        self.assertIn(".cmk-flow-meta-value", source)
        self.assertIn("overflow-wrap: anywhere", source)
        self.assertNotIn(
            ".cmk-flow-meta-value { display: block; margin-top: 3px; overflow: hidden",
            source,
        )


if __name__ == "__main__":
    unittest.main()
