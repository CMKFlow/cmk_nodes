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
        self.assertIn("return (image, image, visual)", source[bypass_at:sampler_at])

    def test_process_owns_visual_provider(self):
        source = (ROOT / "pipe/cmk_refiner.py").read_text(encoding="utf-8")
        self.assertIn('"optional": {"VISUAL": ("CMK_VISUAL_PIPE",)}', source)
        self.assertIn('RETURN_NAMES = ("IMAGE 1ST PASS", "IMAGE REFINED", "VISUAL")', source)
        self.assertIn('module_type="CMKRefinerPipe"', source)
        self.assertIn('channels={"before": source_image, "after": refined_image}', source)

    def test_diffusion_remove_reaches_the_second_pass(self):
        source = (ROOT / "pipe/cmk_refiner.py").read_text(encoding="utf-8")
        sampler = (ROOT / "pipe/cmk_pipe_sampler.py").read_text(encoding="utf-8")
        self.assertNotIn("remove_result_image", source)
        self.assertNotIn("remove_result_image", sampler)
        self.assertNotIn("LaMa", source)
        self.assertNotIn("LaMa", sampler)
        self.assertIn('KSamplerAdvanced().sample', source)

    def test_refiner_inherits_effective_remove_prompts(self):
        source = (ROOT / "pipe/cmk_refiner_prepare.py").read_text(encoding="utf-8")
        self.assertIn('"effective_prompt_pos"', source)
        self.assertIn('"effective_prompt_neg"', source)
        self.assertLess(
            source.index('"effective_prompt_pos"'),
            source.index('SAMPLED.get("prompt_pos"'),
        )
        self.assertLess(
            source.index('"effective_prompt_neg"'),
            source.index('SAMPLED.get("prompt_neg"'),
        )

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

    def test_refiner_evicts_its_generation_clip_after_conditioning(self):
        source = (ROOT / "pipe/cmk_refiner_prepare.py").read_text(encoding="utf-8")
        self.assertIn('label="SDXL Refiner"', source)
        self.assertIn('PROCESS.get("unload_models_after_use", True)', source)
        self.assertIn('refiner_pipe["refiner_text_encoder_status"]', source)

        loader_source = (
            ROOT / "pipe/loaders/checkpoint_vae_loader.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"refiner_clip"', loader_source)

    def test_refiner_restores_an_evicted_cached_text_encoder(self):
        source = (ROOT / "pipe/cmk_refiner_prepare.py").read_text(encoding="utf-8")
        self.assertIn("ensure_sdxl_text_encoder", source)
        self.assertIn(
            "if clip is None:\n            clip, _ = ensure_sdxl_text_encoder(MODEL)",
            source,
        )
        self.assertLess(
            source.index("ensure_sdxl_text_encoder(MODEL)"),
            source.index("CMKLoRATextLoader().load_loras"),
        )


if __name__ == "__main__":
    unittest.main()
