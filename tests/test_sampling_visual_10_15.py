import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "CMK Flow · 10 KSampler SDXL 1st Pass": (
        "CMKKSamplerPipe", "1st-Pass SDXL", "sdxl.first_pass",
        ["SAMPLED", "VISUAL"],
    ),
    "CMK Flow · 10 KSampler Z-Image Turbo": (
        "CMKKSamplerPipe", "Sampling ZIT", "z_image_turbo.sampling",
        ["SAMPLED", "VISUAL"],
    ),
    "CMK Flow · 15 InstantID-Sampler SDXL": (
        "CMKInstantIDSamplerSDXLPipe", "Identity", "sdxl.identity",
        ["MODEL", "PROCESS", "SAMPLED", "LOG", "IDENTITY IMAGE", "VISUAL", "diagnostic"],
    ),
}


class SamplingVisualMigrationTests(unittest.TestCase):
    def test_process_nodes_own_visual_without_inner_display(self):
        for name, (process_type, label, stage, outputs) in SPECS.items():
            with self.subTest(name=name):
                document = json.loads((ROOT / "subgraphs" / f"{name}.json").read_text(encoding="utf-8"))
                outer = document["nodes"][0]
                definition = document["definitions"]["subgraphs"][0]
                process = next(node for node in definition["nodes"] if node["type"] == process_type)
                self.assertIn("VISUAL", [item["name"] for item in process["inputs"]])
                self.assertEqual(outputs, [item["name"] for item in process["outputs"]])
                self.assertFalse({
                    "CMKVisualProvider", "CMKVisualCompare",
                    "CMKImageCompareEnableGate", "ImageCompare",
                } & {node["type"] for node in definition["nodes"]})
                provider = outer["properties"]["cmkVisualProviders"][0]
                self.assertTrue(provider["provider_id"].startswith(f"cmk-{process_type}-"))
                self.assertTrue(str(provider["live_node_id"]).isdigit())
                self.assertEqual(label, provider["label"])
                self.assertEqual(stage, provider["stage_key"])

    def test_refiner_has_no_inner_display(self):
        document = json.loads(
            (ROOT / "subgraphs/CMK Flow · 20 Refiner SDXL.json").read_text(encoding="utf-8")
        )
        definition = document["definitions"]["subgraphs"][0]
        self.assertFalse({"CMKVisualProvider", "CMKVisualCompare"} & {
            node["type"] for node in definition["nodes"]
        })
        process = next(node for node in definition["nodes"] if node["type"] == "CMKRefinerPipe")
        self.assertEqual([12716], process["outputs"][-1]["links"])

    def test_hybrid_zit_and_refiner_publish_real_compare_channels(self):
        sampler = (ROOT / "pipe/cmk_pipe_sampler.py").read_text(encoding="utf-8")
        zit = (ROOT / "pipe/cmk_z_image_turbo.py").read_text(encoding="utf-8")
        refiner = (ROOT / "pipe/cmk_refiner.py").read_text(encoding="utf-8")
        self.assertIn('"hybrid_source_image": IMAGE', zit)
        self.assertIn('label = "1st-Pass SDXL"', sampler)
        self.assertIn('label = "2nd-Pass ZIT"', sampler)
        self.assertIn('label = "Sampling ZIT"', sampler)
        self.assertIn('{"before": before, "after": after}', sampler)
        self.assertIn('stage_key = "hybrid.second_pass"', sampler)
        self.assertIn('module_label="2nd-Pass SDXL"', refiner)
        self.assertIn('channels={"before": source_image, "after": refined_image}', refiner)


if __name__ == "__main__":
    unittest.main()
