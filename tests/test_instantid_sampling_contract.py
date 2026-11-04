from __future__ import annotations

import ast
from pathlib import Path
import sys
import types
import unittest

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
PREPARE_PATH = PACKAGE_ROOT / "pipe" / "cmk_sampler_prepare.py"
ADVANCED_MODEL_PATH = COMFY_ROOT / "comfy_extras" / "nodes_model_advanced.py"
K_DIFFUSION_PATH = COMFY_ROOT / "comfy" / "k_diffusion" / "sampling.py"
INSTANTID_PATHS = (
    PACKAGE_ROOT / "pipe" / "instantid" / "cmk_instantid_sampler.py",
    PACKAGE_ROOT / "pipe" / "instantid" / "cmk_instantid_regional_advanced.py",
)

sys.path.insert(0, str(COMFY_ROOT))
import comfy.model_sampling as model_sampling


def _top_level_nodes(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]


def _load_alignment_predicate():
    namespace = {}
    nodes = _top_level_nodes(PREPARE_PATH, {"_align_identity_sampling"})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PREPARE_PATH), "exec"), namespace)
    return namespace["_align_identity_sampling"]


def _load_sampling_contracts():
    comfy = types.SimpleNamespace(model_sampling=model_sampling)
    namespace = {"comfy": comfy, "torch": torch}
    nodes = _top_level_nodes(
        ADVANCED_MODEL_PATH,
        {"LCM", "ModelSamplingDiscreteDistilled"},
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ADVANCED_MODEL_PATH), "exec"), namespace)
    return namespace["LCM"], namespace["ModelSamplingDiscreteDistilled"]


def _load_karras_schedule():
    namespace = {"torch": torch}
    nodes = _top_level_nodes(K_DIFFUSION_PATH, {"append_zero", "get_sigmas_karras"})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(K_DIFFUSION_PATH), "exec"), namespace)
    return namespace["get_sigmas_karras"]


def _start_step_expression(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    sample_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {
            "CMKInstantIDSamplerSDXLPipe",
            "CMKInstantIDRegionalAdvancedSDXLPipe",
        }
    )
    sample = next(
        node
        for node in sample_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sample"
    )
    assignment = next(
        node
        for node in ast.walk(sample)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "start_at_step"
            for target in node.targets
        )
    )
    return compile(ast.Expression(assignment.value), str(path), "eval")


ALIGN_IDENTITY_SAMPLING = _load_alignment_predicate()
LCM, MODEL_SAMPLING_DISTILLED = _load_sampling_contracts()
GET_SIGMAS_KARRAS = _load_karras_schedule()


def _sampling_contract(mode):
    sampling_base = model_sampling.ModelSamplingDiscrete
    sampling_type = model_sampling.EPS
    if mode == "lcm":
        sampling_base = MODEL_SAMPLING_DISTILLED
        sampling_type = LCM

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    return ModelSamplingAdvanced(None, zsnr=False)


def _karras_sigmas(contract):
    return GET_SIGMAS_KARRAS(
        n=25,
        sigma_min=float(contract.sigma_min),
        sigma_max=float(contract.sigma_max),
    )


class InstantIDSamplingContractTests(unittest.TestCase):
    def test_only_hybrid_continuation_aligns_identity_sampling(self):
        self.assertTrue(ALIGN_IDENTITY_SAMPLING(True, "HYBRID"))
        self.assertFalse(ALIGN_IDENTITY_SAMPLING(False, "HYBRID"))
        self.assertFalse(ALIGN_IDENTITY_SAMPLING(True, "ZERO-PASS"))

    def test_lcm_schedule_and_denoising_contract_are_identical(self):
        layout = _sampling_contract("lcm")
        identity = _sampling_contract("lcm")

        torch.testing.assert_close(_karras_sigmas(layout), _karras_sigmas(identity))
        self.assertIs(type(layout).calculate_denoised, LCM.calculate_denoised)
        self.assertIs(type(identity).calculate_denoised, LCM.calculate_denoised)
        self.assertIsNot(type(identity).calculate_denoised, model_sampling.EPS.calculate_denoised)

    def test_lcm_step_4_and_12_are_the_same_sigma_positions(self):
        layout_sigmas = _karras_sigmas(_sampling_contract("lcm"))
        identity_sigmas = _karras_sigmas(_sampling_contract("lcm"))
        expected = {4: 8.068455, 12: 2.060930}

        for step in (4, 12):
            with self.subTest(step=step):
                self.assertEqual(layout_sigmas[step].item(), identity_sigmas[step].item())
                self.assertAlmostEqual(layout_sigmas[step].item(), expected[step], places=5)

    def test_eps_configuration_remains_consistent(self):
        layout = _sampling_contract("eps")
        identity = _sampling_contract("eps")

        torch.testing.assert_close(_karras_sigmas(layout), _karras_sigmas(identity))
        self.assertIs(type(layout).calculate_denoised, model_sampling.EPS.calculate_denoised)
        self.assertIs(type(identity).calculate_denoised, model_sampling.EPS.calculate_denoised)

    def test_prepare_reuses_sampling_settings_and_keeps_patch_isolation(self):
        source = PREPARE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        prepare_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CMKSamplerPrepareSDXLPipe"
        )
        prepare = next(
            node
            for node in prepare_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "prepare"
        )

        calls = [node for node in ast.walk(prepare) if isinstance(node, ast.Call)]
        sampling_calls = [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "_apply_sampling"
        ]
        sampling_targets = [ast.unparse(node.args[0]) for node in sampling_calls]
        self.assertEqual(sampling_targets.count("identity_model"), 2)
        self.assertEqual(sampling_targets.count("prepared_model"), 2)
        for node in sampling_calls:
            self.assertEqual(ast.unparse(node.args[1]), "str(sampling)")
            self.assertEqual(ast.unparse(node.args[2]), "bool(zsnr)")

        pag_targets = [
            ast.unparse(node.args[0])
            for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "_apply_pag"
        ]
        local_lora_targets = [
            ast.unparse(node.args[0])
            for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "_apply_single_lora"
        ]
        self.assertEqual(pag_targets, ["prepared_model", "prepared_model"])
        self.assertEqual(local_lora_targets, ["prepared_model", "prepared_model"])
        self.assertEqual(source.count("identity_source_model = prepared_model"), 2)
        self.assertEqual(
            source.count(
                "identity_model = identity_source_model if identity_source_model is not None else prepared_model"
            ),
            2,
        )

    def test_hybrid_latent_and_disable_noise_contract_remain_unchanged(self):
        sampler_source = (
            PACKAGE_ROOT / "pipe" / "cmk_pipe_sampler.py"
        ).read_text(encoding="utf-8")
        self.assertIn('samples["samples"] = sampled_tensor', sampler_source)

        for path in INSTANTID_PATHS:
            source = path.read_text(encoding="utf-8")
            self.assertIn(
                'layout_latent = _first(sampled_in.get("samples"), sampled_in.get("latent_1st_pass"), sampled_in.get("latent_image"))',
                source,
            )
            self.assertIn(
                'add_noise = "enable" if (zero_pass_mode or inpaint_bridge_mode) else "disable"',
                source,
            )

    def test_15s_and_15a_share_the_centrally_prepared_identity_model(self):
        for path in INSTANTID_PATHS:
            source = path.read_text(encoding="utf-8")
            identity = source.index('sampled_in.get("model_identity")')
            compatibility = source.index('sampled_in.get("model_patched")', identity)
            self.assertLess(identity, compatibility)
            self.assertEqual(source.count('_DIAGNOSTIC_USE_LAYOUT_PATCHED_MODEL = False'), 1)
            self.assertIn('"aligned Hybrid ModelSampling"', source)
            self.assertIn('else "native ModelSampling"', source)

    def test_reference_start_is_deserialization_only(self):
        for path in INSTANTID_PATHS:
            source = path.read_text(encoding="utf-8")
            self.assertIn('"reference_start_at_step": ("INT", {', source)
            self.assertIn('"default": 8, "min": 0, "max": 200, "step": 1', source)
            self.assertIn("Legacy workflow compatibility value", source)
            self.assertNotIn("int(reference_start_at_step)", source)
            self.assertNotIn("reference_latent_mode", source)
            self.assertIn('"start_at_step": start_at_step', source)

    def test_legacy_reference_values_cannot_change_active_start_step(self):
        for path in INSTANTID_PATHS:
            expression = _start_step_expression(path)

            def resolve(_legacy_value, *, zero=False, handoff=18):
                return eval(expression, {}, {
                    "zero_pass_mode": zero,
                    "sampled_in": {"instantid_end_at_step": handoff},
                })

            with self.subTest(path=path.name, mode="hybrid"):
                self.assertEqual(resolve(0), 18)
                self.assertEqual(resolve(8), 18)
                self.assertEqual(resolve(200), 18)
            with self.subTest(path=path.name, mode="zero-pass"):
                self.assertEqual(resolve(0, zero=True), 0)
                self.assertEqual(resolve(8, zero=True), 0)
                self.assertEqual(resolve(200, zero=True), 0)

if __name__ == "__main__":
    unittest.main()
