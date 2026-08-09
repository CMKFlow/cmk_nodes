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

    def test_custom_flow_nodes_reference_existing_previews(self):
        metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]
        for node_name, entry in metadata.items():
            for preview in entry.get("previews", []):
                asset = ROOT / "web" / preview["src"]
                self.assertTrue(asset.is_file(), f"{node_name}: {preview['src']}")
                self.assertGreater(asset.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
