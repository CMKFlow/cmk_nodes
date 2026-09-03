import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RefinerBypassContractTests(unittest.TestCase):
    def test_refiner_default_starts_at_ninety_percent(self):
        source = (ROOT / "pipe/cmk_refiner_prepare.py").read_text(encoding="utf-8")
        self.assertIn('"start_with_instantid": ("FLOAT", {', source)
        self.assertIn('"default": 90.0', source)
        self.assertIn('"start_without_instantid": ("FLOAT", {', source)

    def test_prepare_no_longer_exposes_global_enable(self):
        source = (ROOT / "pipe/cmk_refiner_prepare.py").read_text(encoding="utf-8")
        self.assertNotIn('"refiner_global_enable":', source)
        self.assertNotIn('if not bool(refiner_global_enable)', source)

    def test_execute_decodes_base_latent_without_sampling(self):
        source = (ROOT / "pipe/cmk_refiner.py").read_text(encoding="utf-8")
        bypass_at = source.index('if REFINER.get("refiner_prepare_bypassed")')
        sampler_at = source.index('KSamplerAdvanced().sample')
        self.assertLess(bypass_at, sampler_at)
        self.assertIn("20 REFINER BYPASS VAE DECODE", source)
        self.assertIn("return (image, image)", source[bypass_at:sampler_at])

    def test_subgraph_no_longer_exposes_enable_proxy(self):
        workflow = json.loads(
            (ROOT / "subgraphs/CMK Flow · 20 Refiner SDXL.json").read_text(encoding="utf-8")
        )
        self.assertEqual(workflow["nodes"][0]["properties"]["proxyWidgets"], [])
        prepare = next(
            node for node in workflow["definitions"]["subgraphs"][0]["nodes"]
            if node["type"] == "CMKRefinerPrepareSDXLPipe"
        )
        self.assertNotIn("refiner_global_enable", [item["name"] for item in prepare["inputs"]])
        self.assertIn(prepare["widgets_values"][5], ("1st pass", "Local settings"))

    def test_refiner_inherits_final_sampled_steps_with_sampling_settings(self):
        source = (ROOT / "pipe/cmk_refiner_prepare.py").read_text(encoding="utf-8")
        self.assertIn('SAMPLED.get("steps_1st_pass", SAMPLED.get("steps"))', source)
        self.assertIn('"refiner_steps_source": "sampled" if inherit_sampling else "local"', source)


if __name__ == "__main__":
    unittest.main()
