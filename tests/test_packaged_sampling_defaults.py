import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBGRAPHS = ROOT / "subgraphs"


def inner_node(filename, node_type):
    document = json.loads((SUBGRAPHS / filename).read_text(encoding="utf-8"))
    return next(
        node
        for definition in document["definitions"]["subgraphs"]
        for node in definition["nodes"]
        if node["type"] == node_type
    )


class PackagedSamplingDefaultsTests(unittest.TestCase):
    def test_sdxl_first_pass_uses_euler_ancestral(self):
        node = inner_node(
            "CMK Flow · 10 KSampler SDXL 1st Pass.json",
            "CMKSamplerPrepareSDXLPipe",
        )
        self.assertEqual("euler_ancestral", node["widgets_values_named"]["sampler"])
        self.assertEqual("euler_ancestral", node["widgets_values"][13])

    def test_full_flow_embeds_the_same_first_pass_sampler_default(self):
        document = json.loads(
            (ROOT / "workflows" / "showcase" / "CMK 2.5 · Full Flow .json").read_text(
                encoding="utf-8"
            )
        )
        definition = next(
            item for item in document["definitions"]["subgraphs"]
            if item["name"] == "CMK Flow · 10 KSampler SDXL 1st Pass"
        )
        node = next(
            item for item in definition["nodes"]
            if item["type"] == "CMKSamplerPrepareSDXLPipe"
        )
        self.assertEqual("euler_ancestral", node["widgets_values_named"]["sampler"])
        self.assertEqual("euler_ancestral", node["widgets_values"][13])

    def test_face_rebuild_variants_use_euler_ancestral(self):
        for filename in (
            "CMK Flow · FaceRebuild SDXL.json",
            "CMK Flow · FaceRebuild SDXL · Advanced.json",
        ):
            with self.subTest(filename=filename):
                document = json.loads((SUBGRAPHS / filename).read_text(encoding="utf-8"))
                node = next(
                    item
                    for definition in document["definitions"]["subgraphs"]
                    for item in definition["nodes"]
                    if item["type"].startswith("CMKInstantIDFaceRebuild")
                )
                self.assertEqual("euler_ancestral", node["widgets_values_named"]["sampler"])
                widget_inputs = [item for item in node["inputs"] if item.get("widget")]
                sampler_index = next(
                    index for index, item in enumerate(widget_inputs)
                    if item["name"] == "sampler"
                )
                self.assertEqual("euler_ancestral", node["widgets_values"][sampler_index])

    def test_refiner_inherits_first_pass_sampling(self):
        node = inner_node(
            "CMK Flow · 20 Refiner SDXL.json",
            "CMKRefinerPrepareSDXLPipe",
        )
        self.assertEqual("1st pass", node["widgets_values_named"]["sampling_source"])
        self.assertEqual("1st pass", node["widgets_values"][5])


if __name__ == "__main__":
    unittest.main()
