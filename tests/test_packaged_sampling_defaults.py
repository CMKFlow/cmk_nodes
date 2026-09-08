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

    def test_refiner_inherits_first_pass_sampling(self):
        node = inner_node(
            "CMK Flow · 20 Refiner SDXL.json",
            "CMKRefinerPrepareSDXLPipe",
        )
        self.assertEqual("1st pass", node["widgets_values_named"]["sampling_source"])
        self.assertEqual("1st pass", node["widgets_values"][5])


if __name__ == "__main__":
    unittest.main()
