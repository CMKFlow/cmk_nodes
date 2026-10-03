import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "CMK Flow · Detailer SDXL": "detailer_global_enable",
    "CMK Flow · Detailer SDXL · Advanced": "detailer_global_enable",
    "CMK Flow · FaceProcess SDXL": "face_global_enable",
    "CMK Flow · FaceProcess SDXL · Advanced": "face_global_enable",
    "CMK Flow · FaceSwap": "FACESWAP ENABLE",
    "CMK Flow · FaceSwap · Advanced": "FACESWAP ENABLE",
}


class VisualCompareEnableTests(unittest.TestCase):
    def test_compare_enable_is_wired_to_the_module_global_switch(self):
        for name, enable_name in NAMES.items():
            with self.subTest(name=name):
                document = json.loads((ROOT / "subgraphs" / f"{name}.json").read_text(encoding="utf-8"))
                definition = document["definitions"]["subgraphs"][0]
                compare = next(node for node in definition["nodes"] if node["type"] == "CMKVisualCompare")
                enable_input = next(item for item in compare["inputs"] if item["name"] == "enable")
                link = next(item for item in definition["links"] if item["id"] == enable_input["link"])
                enable_slot = next(index for index, item in enumerate(definition["inputs"]) if item["name"] == enable_name)
                self.assertEqual((-10, enable_slot), (link["origin_id"], link["origin_slot"]))
                self.assertEqual((compare["id"], 1), (link["target_id"], link["target_slot"]))

    def test_frontend_honors_unconnected_true_and_connected_false(self):
        source = (ROOT / "web/js/cmk_visualizer.js").read_text(encoding="utf-8")
        self.assertIn("function displayNodeEnabled(node)", source)
        self.assertIn("if (!input || input.link == null) return true;", source)
        self.assertIn("stateByNode.has(node) && displayNodeEnabled(node)", source)

    def test_backend_does_not_request_visual_while_disabled(self):
        source = (ROOT / "pipe/cmk_visual.py").read_text(encoding="utf-8")
        section = source[source.index("class CMKVisualCompare:"):]
        self.assertIn('"VISUAL": (VISUAL_TYPE, {"lazy": True})', section)
        self.assertIn("if not bool(enable):\n            return []", section)
        self.assertIn('return ["VISUAL"] if VISUAL is None else []', section)


if __name__ == "__main__":
    unittest.main()
