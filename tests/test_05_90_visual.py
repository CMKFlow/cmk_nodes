import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "CMK Flow · 05 ControlNet SDXL": ("CMKControlNetPreparePipe", "ENABLE", 3, False),
    "CMK Flow · 05 ControlNet ZIT": ("CMKZITControlNetPreparePipe", "ENABLE", 3, False),
    "CMK Flow · 05 ControlNet Combined": ("CMKCombinedControlNetPreparePipe", "ENABLE", 4, False),
    "CMK Flow · Upscale & Save": ("CMK_SmartUpscalerPipe", "SAVE ENABLED", 2, True),
}

class VisualMigration0590Tests(unittest.TestCase):
    def test_process_owns_visual_and_compare_is_pure(self):
        for name, (process_type, enable_name, visual_slot, has_compare) in SPECS.items():
            with self.subTest(name=name):
                document = json.loads((ROOT / "subgraphs" / f"{name}.json").read_text(encoding="utf-8"))
                outer = document["nodes"][0]
                definition = document["definitions"]["subgraphs"][0]
                process = next(node for node in definition["nodes"] if node["type"] == process_type)
                types = {node["type"] for node in definition["nodes"]}
                self.assertFalse({"CMKVisualProvider", "CMKImageCompareEnableGate", "ImageCompare"} & types)
                self.assertIn("VISUAL", [item["name"] for item in process["inputs"]])
                self.assertEqual("VISUAL", process["outputs"][visual_slot]["name"])
                self.assertEqual(has_compare, "CMKVisualCompare" in types)
                if has_compare:
                    compare = next(node for node in definition["nodes"] if node["type"] == "CMKVisualCompare")
                    self.assertEqual(["VISUAL", "enable"], [item["name"] for item in compare["inputs"]])
                declaration = outer["properties"]["cmkVisualProviders"][0]
                self.assertEqual(enable_name, declaration["enable_widget"])

    def test_save_enable_also_controls_upscaler(self):
        document = json.loads(
            (ROOT / "subgraphs/CMK Flow · Upscale & Save.json").read_text(encoding="utf-8")
        )
        definition = document["definitions"]["subgraphs"][0]
        process = next(
            node for node in definition["nodes"] if node["type"] == "CMK_SmartUpscalerPipe"
        )
        save_slot = next(
            index for index, item in enumerate(definition["inputs"])
            if item["name"] == "SAVE ENABLED"
        )
        local_enable_slot = next(
            index for index, item in enumerate(process["inputs"])
            if item["name"] == "enable"
        )
        save_enable_slot = next(
            index for index, item in enumerate(process["inputs"])
            if item["name"] == "save_enabled"
        )
        self.assertIsNone(process["inputs"][local_enable_slot]["link"])
        link_id = process["inputs"][save_enable_slot]["link"]
        self.assertIsNotNone(link_id)
        link = next(link for link in definition["links"] if link["id"] == link_id)
        self.assertEqual(-10, link["origin_id"])
        self.assertEqual(save_slot, link["origin_slot"])
        self.assertEqual(process["id"], link["target_id"])
        self.assertEqual(save_enable_slot, link["target_slot"])

        save = next(
            node for node in definition["nodes"] if node["type"] == "CMK_SaveProjectImage"
        )
        self.assertNotIn("PROJECT FOLDER", [item["name"] for item in save["inputs"]])

    def test_upscale_save_accepts_family_neutral_faceswap_outputs(self):
        document = json.loads(
            (ROOT / "subgraphs/CMK Flow · Upscale & Save.json").read_text(
                encoding="utf-8"
            )
        )
        outer = document["nodes"][0]
        definition = document["definitions"]["subgraphs"][0]
        for contract in (outer, definition):
            types = {item["name"]: item["type"] for item in contract["inputs"]}
            self.assertEqual("*", types["PROCESS"])
            self.assertEqual("*", types["IMAGE"])
            self.assertEqual("*", types["LOG"])

        for node_type in ("CMK_SmartUpscalerPipe", "CMK_SaveProjectImage"):
            node = next(node for node in definition["nodes"] if node["type"] == node_type)
            types = {item["name"]: item["type"] for item in node["inputs"]}
            self.assertEqual("*", types["IMAGE"])
            self.assertEqual("*", types["LOG"])

        upscaler_source = (ROOT / "nodes" / "image" / "smart_upscale.py").read_text(
            encoding="utf-8"
        )
        save_source = (ROOT / "nodes" / "io" / "save_project_image.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"IMAGE": (CMK_RESULT_MEDIA_INPUT,)', upscaler_source)
        self.assertIn('"LOG": (CMK_RESULT_MEDIA_INPUT,)', upscaler_source)
        self.assertIn('"IMAGE": (CMK_PROCESS_METADATA_INPUT,)', save_source)
        self.assertIn('"LOG": (CMK_PROCESS_METADATA_INPUT,)', save_source)

if __name__ == "__main__":
    unittest.main()
