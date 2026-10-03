import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_finish_step_helper():
    source = (ROOT / "pipe" / "cmk_z_image_turbo.py").read_text(encoding="utf-8")
    # Loading the complete node requires ComfyUI. Extract the deliberately pure
    # helper instead so this contract test also runs outside a ComfyUI process.
    import ast

    module = ast.parse(source)
    helper = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_hybrid_finish_steps"
    )
    namespace = {"math": __import__("math")}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "cmk_z_image_turbo.py", "exec"), namespace)
    return namespace["_hybrid_finish_steps"]


def _load_masked_composite_helper():
    source = (ROOT / "pipe" / "cmk_z_image_turbo.py").read_text(encoding="utf-8")
    import ast

    module = ast.parse(source)
    helper = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_hybrid_masked_composite"
    )
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "cmk_z_image_turbo.py", "exec"), namespace)
    return namespace["_hybrid_masked_composite"]


class HybridSDXLZITContractTests(unittest.TestCase):
    def test_sdxl_sampler_hands_off_clean_x0_at_real_progress(self):
        source = (ROOT / "pipe" / "cmk_pipe_sampler.py").read_text(encoding="utf-8")
        self.assertIn('"10 SDXL HYBRID HANDOFF"', source)
        self.assertIn("last_step=handoff_step", source)
        self.assertIn("force_full_denoise=False", source)
        self.assertIn('x0_output["x0"]', source)
        self.assertIn('new_pipe["hybrid_x0_latent"] = samples', source)

    def test_zit_finisher_reencodes_and_uses_low_denoise(self):
        source = (ROOT / "pipe" / "cmk_z_image_turbo.py").read_text(encoding="utf-8")
        self.assertIn('elif hybrid_mode:', source)
        self.assertIn('_call_node(("VAEEncode",), vae, IMAGE)', source)
        self.assertIn('PROCESS.get("hybrid_zit_denoise", 0.20)', source)
        self.assertIn("HYBRID FINISH", source)

    def test_hybrid_inpaint_sets_noise_mask_and_preserves_handoff(self):
        source = (ROOT / "pipe" / "cmk_z_image_turbo.py").read_text(encoding="utf-8")
        self.assertIn('_call_node(("SetLatentNoiseMask",), latent, mask)', source)
        self.assertIn('"hybrid_source_image": IMAGE', source)
        self.assertIn('"hybrid_finish_mask": mask', source)
        self.assertIn("_hybrid_masked_composite(source_image, image, finish_mask)", source)

    def test_hybrid_inpaint_keeps_inpaint_output_provenance(self):
        source = (ROOT / "pipe" / "cmk_z_image_turbo.py").read_text(encoding="utf-8")
        self.assertIn('if hybrid_inpaint_mode:\n            generation_mode = "inpaint"', source)
        self.assertIn('"Hybrid Inpaint Finish" if hybrid_inpaint_mode', source)
        self.assertNotIn('"hybrid" if bool(PROCESS.get("hybrid_mode", False))', source)

    def test_hybrid_composite_changes_only_masked_pixels(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        composite = _load_masked_composite_helper()
        before = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        after = torch.ones((1, 4, 4, 3), dtype=torch.float32)
        mask = torch.zeros((1, 4, 4), dtype=torch.float32)
        mask[:, 1:3, 1:3] = 1.0
        merged = composite(before, after, mask)
        self.assertTrue(torch.equal(merged[:, 0], before[:, 0]))
        self.assertTrue(torch.equal(merged[:, :, 0], before[:, :, 0]))
        self.assertTrue(torch.equal(merged[:, 1:3, 1:3], after[:, 1:3, 1:3]))

    def test_zit_finisher_scales_actual_steps_to_denoise_tail(self):
        finish_steps = _load_finish_step_helper()
        self.assertEqual(1, finish_steps(8, 0.10))
        self.assertEqual(2, finish_steps(8, 0.15))
        self.assertEqual(2, finish_steps(8, 0.20))
        self.assertEqual(2, finish_steps(8, 0.25))
        self.assertEqual(3, finish_steps(8, 0.30))

    def test_global_model_lifecycle_is_declared_at_flow_start(self):
        start_source = (ROOT / "pipe" / "cmk_pipe_image.py").read_text(encoding="utf-8")
        finish_source = (ROOT / "pipe" / "cmk_z_image_turbo.py").read_text(encoding="utf-8")
        ui_source = (ROOT / "web" / "js" / "cmk_flow_start_guidance_v51.js").read_text(encoding="utf-8")
        self.assertIn('"unload_models_after_use"', start_source)
        self.assertIn('"default": True', start_source)
        self.assertIn('PROCESS.get(\n                "unload_models_after_use"', finish_source)
        self.assertIn('unload_all_phase_models("10 ZIT complete")', finish_source)
        self.assertIn('name !== "unload_zit_after"', ui_source)

    def test_bridge_selects_sdxl_result_only_for_hybrid(self):
        source = (ROOT / "pipe" / "cmk_hybrid_bridge.py").read_text(encoding="utf-8")
        self.assertIn('process.get("hybrid_mode", False)', source)
        self.assertIn('sampled.get("image_1st_pass")', source)
        self.assertIn("SDXL x0 decode -> ZIT VAE encode", source)
        self.assertIn('process.get("unload_models_after_use", True)', source)
        self.assertIn('unload_all_phase_models("Hybrid SDXL -> ZIT handoff")', source)
        self.assertIn("clear_sdxl_sampled_boundary_cache", source)

    def test_global_lifecycle_reaches_all_terminal_generation_consumers(self):
        sampler = (ROOT / "pipe" / "cmk_pipe_sampler.py").read_text(encoding="utf-8")
        refiner = (ROOT / "pipe" / "cmk_refiner.py").read_text(encoding="utf-8")
        instantid = (ROOT / "pipe" / "instantid" / "cmk_instantid_sampler.py").read_text(encoding="utf-8")
        boundary = (ROOT / "pipe" / "cmk_family_result.py").read_text(encoding="utf-8")
        self.assertIn('unload_all_phase_models(\n                "10 SDXL complete"', sampler)
        self.assertIn('not bool(pipe.get("instantid_enabled", False))', sampler)
        self.assertIn('unload_all_phase_models("15 InstantID complete")', instantid)
        self.assertIn('unload_all_phase_models("20 Refiner complete")', refiner)
        self.assertIn('process.get("unload_models_after_use", True)', boundary)

    def test_hybrid_inpaint_is_sdxl_only(self):
        start_source = (ROOT / "pipe" / "cmk_pipe_image.py").read_text(encoding="utf-8")
        ui_source = (ROOT / "web" / "js" / "cmk_flow_start_guidance_v51.js").read_text(encoding="utf-8")
        self.assertIn('"hybrid_inpaint_mode": bool(hybrid_mode and INPAINT_MODE)', start_source)
        self.assertIn('"boolean_inpaint_mode": False', start_source)
        self.assertIn('"inpaint_process_mode": "hybrid_finish"', start_source)
        self.assertIn('"hybrid-inpaint"', ui_source)

    def test_hybrid_inpaint_releases_terminal_sdxl_patch_tensors(self):
        source = (ROOT / "pipe" / "cmk_hybrid_bridge.py").read_text(encoding="utf-8")
        self.assertIn("def _release_completed_sdxl_patch", source)
        self.assertIn('getattr(patcher, "patches", {})', source)
        self.assertIn("patcher.model_options = {\"transformer_options\": {}}", source)
        self.assertIn('process.get("hybrid_inpaint_mode", False)', source)
        self.assertIn("_release_completed_sdxl_patch(sdxl_model)", source)

    def test_start_ui_has_three_family_tabs_and_hybrid_controls(self):
        source = (ROOT / "web" / "js" / "cmk_flow_start_guidance_v51.js").read_text(encoding="utf-8")
        self.assertIn('["SDXL", "Hybrid", "Z-Image Turbo"]', source)
        self.assertIn('"hybrid_sdxl_handoff"', source)
        self.assertIn('"hybrid_zit_denoise"', source)
        self.assertIn('"HYBRID BALANCE"', source)
        self.assertIn('"SDXL", "ZIT"', source)

    def test_hybrid_balance_has_five_fixed_transition_profiles(self):
        source = (ROOT / "pipe" / "cmk_pipe_image.py").read_text(encoding="utf-8")
        self.assertIn("HYBRID_TRANSITION_PROFILES", source)
        for profile in ("(90, 0.10)", "(90, 0.15)", "(85, 0.20)", "(85, 0.25)", "(80, 0.30)"):
            self.assertIn(profile, source)
        self.assertIn('inputs.get("HYBRID BALANCE", 2)', source)


if __name__ == "__main__":
    unittest.main()
