"""Adaptive SDXL -> InstantID Advanced handshake.  Model-free.

* Window rule (target step, four forward steps, at most steps - 1) and the
  search order target -> forward -> backward down to step 1.
* The gate that keeps every other configuration on the fixed handoff.
* Pairing of x0, latent and handoff step, proven on ComfyUI's real samplers
  with a deterministic stand-in denoiser (no UNet): every candidate, forward
  and backward, equals the fixed handoff at the same step, bit for bit.
* Prefix stability of every sampler in PREFIX_STABLE_SAMPLERS.
* Candidate choice: first success in search order wins, no confirmation over
  several steps, unchanged TARGET selection and A/B collision check (real
  _regional_hints with synthetic face candidates).
* Cache (real persistence in a temporary directory), revision and wiring
  contracts of module 10 and Advanced.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_adaptive_handoff"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

adaptive = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_adaptive_handoff")
distribution = importlib.import_module(f"{TEST_PACKAGE}.cmk_distribution")
advanced = importlib.import_module(f"{TEST_PACKAGE}.pipe.instantid.cmk_instantid_regional_advanced")
persistent = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_persistent_cache")
prepare = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_sampler_prepare")

import comfy.k_diffusion.sampling as k_sampling  # noqa: E402
import comfy.model_sampling as model_sampling  # noqa: E402
import comfy.samplers as samplers  # noqa: E402

SAMPLER_SOURCE = (PACKAGE_ROOT / "pipe" / "cmk_pipe_sampler.py").read_text()
ADVANCED_SOURCE = (PACKAGE_ROOT / "pipe" / "instantid" / "cmk_instantid_regional_advanced.py").read_text()
STANDARD_SOURCE = (PACKAGE_ROOT / "pipe" / "instantid" / "cmk_instantid_sampler.py").read_text()

STEPS, K_TARGET, K_FORWARD = 25, 8, 12
ORDER = [8, 9, 10, 11, 12, 7, 6, 5, 4, 3, 2, 1]
SCALE = 0.13025  # SDXL latent scale factor; process_latent_out divides by it


# --------------------------------------------------------------------------
# Deterministic sampler harness (real ComfyUI sampler functions, no UNet)

class _Sampling(model_sampling.ModelSamplingDiscrete, model_sampling.EPS):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self._register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                                linear_start=0.00085, linear_end=0.012)
        self.sigma_data = 1.0


SAMPLING = _Sampling()


class _Inner:
    model_sampling = SAMPLING

    def __init__(self):
        self.inner_model = self
        self.model_patcher = self
        self.model_options = {}

    def get_model_object(self, name):
        return SAMPLING


class _Denoiser:
    def __init__(self):
        self.inner_model = _Inner()
        self.model_sampling = SAMPLING

    def __call__(self, x, sigma, **kwargs):
        s = sigma.reshape(-1, *([1] * (x.ndim - 1)))
        return torch.tanh(x / (1 + s)) * 0.8 + 0.1 * torch.sin(3 * x)


FULL_SIGMAS = k_sampling.get_sigmas_karras(STEPS, float(SAMPLING.sigma_min), float(SAMPLING.sigma_max))


def process_out(latent):
    return latent / SCALE


def sample(name, last_step, callback, seed=1565304366):
    """comfy.samplers.KSampler.sample: sigmas[:last_step + 1], force_full_denoise=False."""
    noise = torch.randn(1, 4, 13, 19, generator=torch.Generator().manual_seed(seed))
    sigmas = FULL_SIGMAS[: last_step + 1].clone()
    ksampler = samplers.ksampler(name)
    k_callback = lambda data: callback(data["i"], data["denoised"], data["x"], len(sigmas) - 1)
    out = ksampler.sampler_function(_Denoiser(), noise * sigmas[0], sigmas, extra_args={"seed": seed},
                                    callback=k_callback, disable=True, **ksampler.extra_options)
    return process_out(out.to(torch.float32))


def fixed_handoff(name, step):
    """Today's fixed handoff: last callback x0 (CPU, then out) + returned latent."""
    last = {}

    def callback(i, x0, x, total):
        last["i"], last["x0"] = i, x0

    latent = sample(name, step, callback)
    return last["i"], process_out(last["x0"].cpu()), latent


def window_candidates(name, target=K_TARGET, forward=K_FORWARD):
    recorder = adaptive.CandidateRecorder(target, forward)
    final = sample(name, forward, recorder)
    return recorder.candidates(final, process_out, torch.device("cpu"))


# --------------------------------------------------------------------------

def slider_step(steps, percent):
    return prepare._resolve_instantid_handoff(
        active=True, instantid_mode="HYBRID", effective_steps=steps, handoff_percent=percent,
    )[0]


class WindowRule(unittest.TestCase):
    def test_example_from_the_order_25_steps_target_8(self):
        self.assertEqual(adaptive.handoff_window(25, 8), (8, 12))
        self.assertEqual(adaptive.search_order(8, 12), ORDER)
        self.assertEqual(adaptive.search_label(ORDER), "8-12 forward · 7-1 backward")

    def test_slider_limits_20_and_80_percent(self):
        self.assertEqual(slider_step(25, 20), 5)
        self.assertEqual(adaptive.handoff_window(25, 5), (5, 9))
        self.assertEqual(slider_step(25, 80), 20)
        self.assertEqual(adaptive.handoff_window(25, 20), (20, 24))  # no fixed 60 % cap any more
        self.assertEqual(adaptive.search_order(20, 24), [20, 21, 22, 23, 24, *range(19, 0, -1)])

    def test_last_handoff_leaves_one_instantid_step(self):
        for steps in (2, 3, 4, 5, 6):
            for percent in (20, 30, 40, 50, 60, 70, 80):
                target, forward = adaptive.handoff_window(steps, slider_step(steps, percent))
                self.assertLessEqual(forward, steps - 1, (steps, percent))
        self.assertEqual(adaptive.handoff_window(2, 1), (1, 1))
        self.assertEqual(adaptive.search_order(1, 1), [1])  # target 1: no backward search
        self.assertEqual(adaptive.search_label([1]), "1 forward")
        self.assertEqual(adaptive.handoff_window(4, slider_step(4, 80)), (3, 3))
        self.assertEqual(adaptive.search_order(3, 3), [3, 2, 1])
        self.assertEqual(adaptive.search_label([3, 2, 1]), "3 forward · 2-1 backward")

    def test_every_slider_setting_from_small_to_large_step_counts(self):
        for steps in range(2, 151):
            for percent in (20, 30, 40, 50, 60, 70, 80):
                requested = slider_step(steps, percent)
                target, forward = adaptive.handoff_window(steps, requested)
                order = adaptive.search_order(target, forward)
                self.assertEqual(target, requested)
                self.assertEqual(forward, min(target + adaptive.ADAPTIVE_FORWARD_STEPS, steps - 1))
                self.assertEqual(order[0], target)
                self.assertEqual(sorted(order), list(range(1, forward + 1)))  # every state 1 .. forward once
                self.assertEqual(order[: forward - target + 1], list(range(target, forward + 1)))
                self.assertEqual(order[forward - target + 1:], list(range(target - 1, 0, -1)))


def _prompt(*samplers_downstream, ksampler="10"):
    prompt = {
        "1": {"class_type": "CMKPipeCreateImage", "inputs": {}},
        ksampler: {"class_type": "CMKKSamplerPipe", "inputs": {"SAMPLER": ["1", 0]}},
        "12": {"class_type": "CMKSamplerBypassGate", "inputs": {"SAMPLED": [ksampler, 0]}},
    }
    for index, class_type in enumerate(samplers_downstream, 20):
        prompt[str(index)] = {"class_type": class_type, "inputs": {"SAMPLED": ["12", 0], "PROCESS": ["1", 0]}}
    return prompt


def _pipe(**overrides):
    pipe = {
        "instantid_enabled": True, "instantid_mode": "HYBRID", "model_family": "sdxl",
        "sampler": "euler_ancestral", "steps_1st_pass": 25, "instantid_end_at_step": 8,
    }
    pipe.update(overrides)
    return pipe


class AdaptiveGate(unittest.TestCase):
    def plan(self, pipe=None, prompt=None):
        return adaptive.adaptive_handoff_plan(
            pipe or _pipe(), prompt or _prompt(adaptive.ADVANCED_CLASS), "10",
        )

    def test_advanced_hybrid_text2image_gets_the_search(self):
        self.assertEqual(self.plan(), {"version": adaptive.ADAPTIVE_HANDOFF_VERSION, "target": 8, "forward": 12})
        self.assertEqual(adaptive.ADAPTIVE_HANDOFF_VERSION, "cmk-adaptive-handoff-v2")

    def test_high_slider_settings_stay_adaptive(self):
        self.assertEqual(self.plan(_pipe(instantid_end_at_step=20)), {
            "version": adaptive.ADAPTIVE_HANDOFF_VERSION, "target": 20, "forward": 24,
        })

    def test_instantid_standard_keeps_the_fixed_handoff(self):
        self.assertIsNone(self.plan(prompt=_prompt(adaptive.STANDARD_CLASS)))

    def test_ambiguous_or_missing_downstream_sampler_keeps_fixed(self):
        self.assertIsNone(self.plan(prompt=_prompt(adaptive.ADVANCED_CLASS, adaptive.STANDARD_CLASS)))
        self.assertIsNone(self.plan(prompt=_prompt(adaptive.ADVANCED_CLASS, adaptive.ADVANCED_CLASS)))
        self.assertIsNone(self.plan(prompt=_prompt()))
        unrelated = _prompt()
        unrelated["30"] = {"class_type": adaptive.ADVANCED_CLASS, "inputs": {"SAMPLED": ["99", 0]}}
        self.assertIsNone(self.plan(prompt=unrelated))

    def test_stop_after_handoff_samples_the_same_window(self):
        self.assertEqual(self.plan(_pipe(stop_after_handoff=True)), self.plan())

    def test_inpaint_and_other_modes_keep_fixed(self):
        self.assertIsNone(self.plan(_pipe(boolean_inpaint_mode=True)))
        self.assertIsNone(self.plan(_pipe(instantid_mode="HYBRID INPAINT")))
        self.assertIsNone(self.plan(_pipe(instantid_mode="ZERO-PASS", instantid_end_at_step=0)))
        self.assertIsNone(self.plan(_pipe(hybrid_mode=True)))
        self.assertIsNone(self.plan(_pipe(instantid_enabled=False)))
        self.assertIsNone(self.plan(_pipe(model_family="z_image_turbo")))

    def test_unproven_samplers_keep_fixed(self):
        for name in ("dpmpp_sde", "dpmpp_2m_sde", "dpmpp_3m_sde_gpu", "deis", "ipndm", "heunpp2",
                     "dpm_fast", "dpm_adaptive", "euler_cfg_pp", "uni_pc", "ddim"):
            self.assertIsNone(self.plan(_pipe(sampler=name)), name)
        for name in ("euler", "euler_ancestral", "dpmpp_2m"):
            self.assertIsNotNone(self.plan(_pipe(sampler=name)), name)

    def test_plan_token_separates_fixed_adaptive_and_the_old_window(self):
        self.assertEqual(adaptive.plan_token(None), "fixed")
        self.assertEqual(adaptive.plan_token(self.plan()), "cmk-adaptive-handoff-v2:8-12")
        self.assertNotEqual(adaptive.plan_token(self.plan()), "cmk-adaptive-handoff-v1:8-14")


class PairingAndPrefixStability(unittest.TestCase):
    """The proof that candidate k is today's fixed handoff at step k, in both directions."""

    def test_fixed_handoff_uses_the_x0_of_callback_step_k_minus_1(self):
        last_i, _x0, _latent = fixed_handoff("euler", K_TARGET)
        self.assertEqual(last_i, K_TARGET - 1)

    def test_every_candidate_forward_and_backward_equals_the_fixed_handoff_at_its_step(self):
        for name in ("euler", "euler_ancestral", "dpmpp_2m"):
            candidates = window_candidates(name)
            self.assertEqual([c["step"] for c in candidates], ORDER)
            for candidate in candidates:
                _i, x0, latent = fixed_handoff(name, candidate["step"])
                self.assertTrue(torch.equal(candidate["x0"], x0), (name, candidate["step"]))
                self.assertTrue(torch.equal(candidate["latent"], latent), (name, candidate["step"]))

    def test_target_one_and_step_one_as_backward_candidate(self):
        candidates = window_candidates("euler", target=1, forward=5)
        self.assertEqual([c["step"] for c in candidates], [1, 2, 3, 4, 5])
        step_one = window_candidates("euler")[-1]
        self.assertEqual(step_one["step"], 1)
        _i, x0, latent = fixed_handoff("euler", 1)
        self.assertTrue(torch.equal(step_one["x0"], x0))
        self.assertTrue(torch.equal(step_one["latent"], latent))

    def test_every_listed_sampler_is_prefix_stable(self):
        for name in sorted(adaptive.PREFIX_STABLE_SAMPLERS):
            self.assertIn(name, samplers.KSampler.SAMPLERS)
            candidates = {c["step"]: c for c in window_candidates(name)}
            for step in (K_TARGET, 3):  # target and a backward candidate
                _i, x0, latent = fixed_handoff(name, step)
                self.assertTrue(torch.equal(candidates[step]["latent"], latent), (name, step))
                self.assertTrue(torch.equal(candidates[step]["x0"], x0), (name, step))

    def test_known_schedule_dependent_samplers_are_not_prefix_stable(self):
        for name in ("dpmpp_sde", "dpmpp_2m_sde", "deis", "ipndm", "heunpp2"):
            self.assertNotIn(name, adaptive.PREFIX_STABLE_SAMPLERS)
            candidate = window_candidates(name)[0]
            _i, _x0, latent = fixed_handoff(name, K_TARGET)
            self.assertFalse(torch.equal(candidate["latent"], latent), name)

    def test_missing_sampler_state_is_an_error_not_a_silent_shift(self):
        recorder = adaptive.CandidateRecorder(K_TARGET, K_FORWARD)
        for step in range(K_TARGET - 1, K_FORWARD):
            recorder(step, torch.zeros(1), torch.zeros(1), STEPS)
        with self.assertRaisesRegex(RuntimeError, "step 7"):  # first backward candidate
            recorder.candidates(torch.zeros(1), process_out, torch.device("cpu"))

    def test_recorder_keeps_only_the_needed_states_and_forwards_every_step(self):
        seen = []
        recorder = adaptive.CandidateRecorder(K_TARGET, K_FORWARD, lambda i, *_: seen.append(i))
        sample("euler", K_FORWARD, recorder)
        self.assertEqual(seen, list(range(K_FORWARD)))
        self.assertEqual(sorted(recorder.x0), list(range(0, K_FORWARD)))
        self.assertEqual(sorted(recorder.latents), list(range(1, K_FORWARD)))


class FirstSuccessWins(unittest.TestCase):
    @staticmethod
    def candidates(*steps):
        return [{"step": step} for step in steps]

    def evaluate(self, accepted, evaluated=None):
        def run(candidate):
            if evaluated is not None:
                evaluated.append(candidate["step"])
            if candidate["step"] not in accepted:
                raise ValueError(f"rejected {candidate['step']}")
            return f"hints {candidate['step']}"
        return run

    def choose(self, accepted):
        evaluated = []
        candidate, result, attempts = adaptive.first_resolved_candidate(
            self.candidates(*ORDER), self.evaluate(accepted, evaluated),
        )
        return candidate["step"], evaluated

    def test_target_step_succeeds_at_once(self):
        self.assertEqual(self.choose({8, 9, 3}), (8, [8]))

    def test_first_additional_sdxl_step(self):
        self.assertEqual(self.choose({9, 10}), (9, [8, 9]))

    def test_last_forward_candidate(self):
        self.assertEqual(self.choose({12}), (12, [8, 9, 10, 11, 12]))

    def test_no_forward_hit_first_backward_candidate(self):
        self.assertEqual(self.choose({7, 6}), (7, [8, 9, 10, 11, 12, 7]))

    def test_no_forward_hit_much_earlier_step(self):
        self.assertEqual(self.choose({2}), (2, [8, 9, 10, 11, 12, 7, 6, 5, 4, 3, 2]))

    def test_step_one_as_last_backward_candidate(self):
        self.assertEqual(self.choose({1}), (1, ORDER))

    def test_forward_hit_beats_backward_hits(self):
        self.assertEqual(self.choose({7, 6, 11}), (11, [8, 9, 10, 11]))

    def test_several_forward_hits_the_first_in_order_wins(self):
        self.assertEqual(self.choose({10, 11, 12}), (10, [8, 9, 10]))

    def test_isolated_success_between_failures_needs_no_confirmation(self):
        self.assertEqual(self.choose({5}), (5, [8, 9, 10, 11, 12, 7, 6, 5]))

    def test_no_success_rejects_with_every_attempt_in_search_order(self):
        with self.assertRaises(adaptive.CandidateRejected) as caught:
            adaptive.first_resolved_candidate(self.candidates(*ORDER), self.evaluate(set()))
        self.assertEqual(caught.exception.args[0], [(step, f"rejected {step}") for step in ORDER])


# --------------------------------------------------------------------------
# Real target search with synthetic faces

W, H = 1216, 832


def face(cx, cy, size=80, score=0.8):
    kps = np.array([[cx - size * .2, cy - size * .1], [cx + size * .2, cy - size * .1], [cx, cy + .05 * size],
                    [cx - size * .15, cy + .25 * size], [cx + size * .15, cy + .25 * size]])
    return {"bbox": np.array([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2]), "kps": kps,
            "det_score": score}


LEFT, RIGHT = face(274, 490, score=0.62), face(898, 388, score=0.71)
ONE_LEFT = "1 face [(234, 450, 314, 530) 0.62]"


class FakeAnalyser:
    """Returns the faces planted for the candidate whose image is analysed."""

    def __init__(self):
        self.det_model = types.SimpleNamespace(input_size=(640, 640))
        self.faces = {}

    def get(self, image):
        return list(self.faces.get(int(round(float(image[0, 0, 0]))), []))


def run_search(faces_by_step, *, target=K_TARGET, forward=K_FORWARD, source_b=True,
               target_a="Leftmost", target_b="Rightmost"):
    """Candidates in search order; steps without planted faces show no face."""
    analyser = FakeAnalyser()
    analyser.faces = faces_by_step
    candidates = [{"step": step, "x0": step} for step in adaptive.search_order(target, forward)]

    def decode(step):  # the image carries its step so the analyser can recognise it
        image = torch.zeros(1, H, W, 3)
        image[0, 0, 0, :] = step / 255.0
        return image

    return advanced._adaptive_target_regions(
        candidates=candidates, decode=decode, faceanalysis=analyser,
        target_face_a=target_a, target_face_b=target_b, source_b_enabled=source_b,
    )


class TargetSearch(unittest.TestCase):
    def test_target_step_misses_the_right_face_a_later_step_resolves_both(self):
        candidate, _image, hints, rejected = run_search({8: [LEFT], 9: [LEFT], 10: [LEFT, RIGHT], 11: [LEFT, RIGHT]})
        self.assertEqual(candidate["step"], 10)
        hint_a, mask_a, hint_b, mask_b, overlap = hints
        self.assertLess(overlap, 0.9)
        self.assertLess(float(mask_a[0, 490, 898]), 0.5)  # A's mask is on the left person
        self.assertGreater(float(mask_b[0, 388, 898]), 0.5)
        self.assertEqual(rejected, [
            f"step 8: TARGET A and TARGET B resolve to the same person · {ONE_LEFT}",
            f"step 9: TARGET A and TARGET B resolve to the same person · {ONE_LEFT}",
        ])

    def test_backward_candidate_after_a_fruitless_forward_search(self):
        candidate, _image, _hints, rejected = run_search({6: [LEFT, RIGHT], 3: [LEFT, RIGHT]})
        self.assertEqual(candidate["step"], 6)
        self.assertEqual([item.split(":")[0] for item in rejected],
                         ["step 8", "step 9", "step 10", "step 11", "step 12", "step 7"])

    def test_backward_candidate_with_ab_collision_is_rejected(self):
        candidate, _image, _hints, rejected = run_search({7: [LEFT], 6: [face(600, 250), face(600, 560)],
                                                          5: [LEFT, RIGHT]})
        self.assertEqual(candidate["step"], 5)
        self.assertIn(f"step 7: TARGET A and TARGET B resolve to the same person · {ONE_LEFT}", rejected)
        self.assertTrue(any(item.startswith("step 6: TARGET A and TARGET B resolve to the same person · 2 faces")
                            for item in rejected))

    def test_forward_hit_has_priority_over_an_earlier_backward_hit(self):
        candidate, *_ = run_search({7: [LEFT, RIGHT], 12: [LEFT, RIGHT]})
        self.assertEqual(candidate["step"], 12)

    def test_no_candidate_resolves_aborts_with_the_whole_search(self):
        faces = {step: [LEFT] for step in ORDER}
        faces[1] = []
        with self.assertRaises(ValueError) as caught:
            run_search(faces)
        message = str(caught.exception)
        self.assertTrue(message.startswith(
            "CMK InstantID Regional Advanced: no handoff candidate (8-12 forward · 7-1 backward) "
            "resolves TARGET A and TARGET B · "
            f"step 8: TARGET A and TARGET B resolve to the same person · {ONE_LEFT} | "
        ), message)
        self.assertTrue(message.endswith("step 1: target face not detected · 0 faces"), message)
        self.assertEqual(re.findall(r"step (\d+):", message), [str(step) for step in ORDER])

    def test_vertically_stacked_faces_keep_the_collision_check(self):
        with self.assertRaisesRegex(ValueError, "step 8: TARGET A and TARGET B resolve to the same person · 2 faces"):
            run_search({8: [face(600, 250), face(600, 560)]}, target=8, forward=8)

    def test_two_different_faces_on_the_target_step(self):
        candidate, _image, _hints, rejected = run_search({8: [LEFT, RIGHT], 9: [LEFT, RIGHT]})
        self.assertEqual((candidate["step"], rejected), (8, []))

    def test_target_selection_is_the_unchanged_one(self):
        candidate, _image, hints, _r = run_search({8: [LEFT, RIGHT]}, target_a="Rightmost", target_b="Leftmost")
        self.assertGreater(float(hints[1][0, 388, 898]), 0.5)  # A on the right person now

    def test_single_identity_needs_only_target_a(self):
        candidate, _image, hints, rejected = run_search({9: [LEFT], 10: [LEFT, RIGHT]}, source_b=False)
        self.assertEqual(candidate["step"], 9)
        self.assertIsNone(hints[2])
        self.assertEqual(rejected, ["step 8: target face not detected · 0 faces"])

    def test_single_identity_backward(self):
        candidate, *_ = run_search({4: [LEFT]}, source_b=False)
        self.assertEqual(candidate["step"], 4)

    def test_single_identity_without_any_face_still_aborts(self):
        with self.assertRaisesRegex(ValueError, r"resolves TARGET A · step 8: target face not detected · 0 faces"):
            run_search({}, source_b=False)


# --------------------------------------------------------------------------

class ModuleTenContract(unittest.TestCase):
    def test_cache_schema_fields_and_key(self):
        self.assertIn('_SDXL_HANDOFF_CACHE_SCHEMA = "cmk_sdxl_handoff_boundary_v2"', SAMPLER_SOURCE)
        result_fields = re.search(r"_SDXL_HANDOFF_RESULT_FIELDS = \((.*?)\)", SAMPLER_SOURCE, re.S).group(1)
        contract_fields = re.search(r"_SDXL_HANDOFF_CONTRACT_FIELDS = \((.*?)\)", SAMPLER_SOURCE, re.S).group(1)
        self.assertIn('"instantid_handoff_candidates"', result_fields)
        self.assertIn('"instantid_handoff_window"', contract_fields)
        self.assertIn('"handoff": str(handoff)', SAMPLER_SOURCE)
        self.assertEqual(SAMPLER_SOURCE.count('plan_token(pipe.get("instantid_handoff_window"))'), 2)

    def test_window_is_decided_before_the_cache_lookup_and_only_for_the_handoff(self):
        decide = SAMPLER_SOURCE.index("plan = adaptive_handoff_plan(")
        lookup = SAMPLER_SOURCE.index("cached = _load_sdxl_handoff(")
        self.assertLess(decide, lookup)
        self.assertIn("plan = adaptive_handoff_plan(pipe, prompt, unique_id)", SAMPLER_SOURCE)
        self.assertNotIn("stop_after_handoff", SAMPLER_SOURCE)  # STOP never changes what module 10 samples

    def test_module_10_never_loads_instantid_or_detects_faces(self):
        for name in ("CMKInternalInstantIDModelLoader", "ControlNetLoader", "insightface", "faceanalysis",
                     "from .instantid", "cmk_instantid_regional_advanced"):
            self.assertNotIn(name, SAMPLER_SOURCE)

    def test_sampling_reaches_the_forward_limit_and_keeps_the_target_as_the_pipe_handoff(self):
        self.assertIn('last_step = int(handoff_window["forward"]) if handoff_window else end_at_step', SAMPLER_SOURCE)
        self.assertIn("last_step=last_step,", SAMPLER_SOURCE)
        self.assertIn("callback=recorder if recorder is not None else preview_callback,", SAMPLER_SOURCE)
        self.assertIn('samples["samples"] = handoff_candidates[0]["latent"]', SAMPLER_SOURCE)
        self.assertIn('keypoints_latent["samples"] = handoff_candidates[0]["x0"]', SAMPLER_SOURCE)
        # Module 10 neither detects faces nor knows the targets.
        self.assertNotIn("faceanalysis", SAMPLER_SOURCE)
        self.assertNotIn("target_face", SAMPLER_SOURCE)


class AdvancedContract(unittest.TestCase):
    def test_candidates_set_latent_and_start_step(self):
        self.assertIn('layout_latent["samples"] = candidate["latent"]', ADVANCED_SOURCE)
        self.assertIn("if handoff_step is not None:\n            # Adaptive handoff: the step of the chosen candidate.\n"
                      "            start_at_step = handoff_step", ADVANCED_SOURCE)
        self.assertIn('"latent_image": layout_latent,', ADVANCED_SOURCE)

    def test_zero_pass_and_inpaint_bridge_ignore_candidates(self):
        self.assertIn("if zero_pass_mode or inpaint_bridge_mode", ADVANCED_SOURCE)

    def test_candidates_do_not_travel_downstream(self):
        self.assertIn('sampled_out.pop("instantid_handoff_candidates", None)', ADVANCED_SOURCE)

    def test_content_guard_checks_every_chosen_candidate_module_10_did_not_inspect(self):
        # Module 10 guards the target-step candidate (first in search order);
        # Advanced guards any other choice, forward or backward.
        self.assertIn("if content_guard_enabled and candidate is not handoff_candidates[0]:", ADVANCED_SOURCE)
        self.assertIn('guard.inspect_content(tensor_to_uint8_rgb(image), "target_intermediate")', ADVANCED_SOURCE)

    def test_target_logic_is_called_unchanged(self):
        self.assertIn("return _regional_hints(faceanalysis, layout_image, target_face_a, target_face_b)", ADVANCED_SOURCE)
        self.assertIn('"CMK InstantID Regional Advanced: TARGET A and TARGET B resolve to the same person"', ADVANCED_SOURCE)
        self.assertIn("if overlap >= 0.9:", ADVANCED_SOURCE)

    def test_log_reports_target_actual_search_and_result(self):
        for text in (
            'f"Handoff target    : step {handoff_candidates[0][\'step\']}/{final_steps}"',
            'f"Handoff actual    : step {start_at_step}/{final_steps}"',
            'f"Handoff search    : {handoff_search}"',
            'f"Handoff result    : {handoff_direction} · step {start_at_step} · success · "',
            "Handoff rejected  : {item}",
            '"TARGET"', '"FORWARD"', '"BACKWARD"',
        ):
            self.assertIn(text, ADVANCED_SOURCE)

    def test_instantid_standard_is_untouched(self):
        self.assertNotIn("handoff_candidates", STANDARD_SOURCE)
        self.assertNotIn("adaptive", STANDARD_SOURCE)

    def test_advanced_results_are_recomputed_once(self):
        revisions = persistent.NODE_BEHAVIOUR_REVISIONS
        self.assertIn("cmk-adaptive-handoff-v2", revisions["CMKInstantIDRegionalAdvancedSDXLPipe"])
        self.assertNotIn("adaptive", revisions["CMKInstantIDSamplerSDXLPipe"])


class ModuleTenRun(unittest.TestCase):
    """CMKKSamplerPipe._run with ComfyUI's sampler replaced by the stand-in above."""

    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_pipe_sampler")

    def setUp(self):
        import comfy.model_management
        import comfy.sample
        import latent_preview
        import nodes
        from unittest.mock import patch

        self.calls = []
        self.persisted = []

        def fake_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                        last_step=None, callback=None, **kwargs):
            self.calls.append(last_step)
            return sample(sampler_name, last_step, callback)

        def fake_prepare_callback(model, steps, x0_output_dict=None):
            def callback(step, x0, x, total):
                x0_output_dict["x0"] = x0
            return callback

        class FakeDecode:
            def decode(self, vae, samples):
                return (torch.zeros(1, 8, 8, 3),)

        module = self.module
        patches = [
            patch.object(comfy.sample, "sample", fake_sample),
            patch.object(comfy.sample, "fix_empty_latent_channels", lambda model, latent, *a, **k: latent),
            patch.object(comfy.sample, "prepare_noise", lambda latent, seed, inds=None: torch.zeros_like(latent)),
            patch.object(comfy.model_management, "intermediate_device", lambda: torch.device("cpu")),
            patch.object(latent_preview, "prepare_callback", fake_prepare_callback),
            patch.object(nodes, "VAEDecode", FakeDecode),
            patch.object(module, "_load_sdxl_handoff", lambda *a, **k: None),
            patch.object(module, "_persist_sdxl_handoff", lambda pipe, *a, **k: self.persisted.append(pipe)),
            patch.object(module, "effective_content_guard_enabled", lambda pipe: False),
            patch.object(module, "send_final_preview", lambda *a, **k: None),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    @staticmethod
    def pipe(sampler_name="euler_ancestral", **overrides):
        model = types.SimpleNamespace(model=types.SimpleNamespace(process_latent_out=process_out))
        conditioning = [[torch.zeros(1, 77, 8), {}]]
        pipe = _pipe(
            sampler=sampler_name, steps=25, scheduler="karras", seed=1565304366, cfg=5.0, denoise=1.0,
            model_patched=model, conditioning_pos=conditioning, conditioning_neg=conditioning,
            latent_image={"samples": torch.zeros(1, 4, 13, 19)}, vae=object(),
            content_guard_enabled=False, instantid_handoff_percent=30,
        )
        pipe.update(overrides)
        return pipe

    def run_module(self, pipe, prompt):
        new_pipe, _visual = self.module.CMKKSamplerPipe()._run(pipe, None, prompt, "10")
        return new_pipe

    def test_advanced_samples_to_the_forward_limit_and_hands_over_the_target_step(self):
        new_pipe = self.run_module(self.pipe(), _prompt(adaptive.ADVANCED_CLASS))
        self.assertEqual(self.calls, [K_FORWARD])
        candidates = new_pipe["instantid_handoff_candidates"]
        self.assertEqual([c["step"] for c in candidates], ORDER)
        _i, x0, latent = fixed_handoff("euler_ancestral", K_TARGET)
        self.assertTrue(torch.equal(new_pipe["samples"]["samples"], latent))
        self.assertTrue(torch.equal(new_pipe["instantid_keypoints_latent"]["samples"], x0))
        self.assertEqual(new_pipe["instantid_end_at_step"], K_TARGET)
        self.assertEqual(new_pipe["instantid_handoff_window"]["forward"], K_FORWARD)
        self.assertIn(
            "step 8 · target, adaptive search 8-12 forward · 7-1 backward · actual step chosen by InstantID Advanced",
            new_pipe["ksampler_log"],
        )
        self.assertIs(self.persisted[0], new_pipe)
        for candidate in candidates:  # backward candidates too, inside module 10
            _i, x0, latent = fixed_handoff("euler_ancestral", candidate["step"])
            self.assertTrue(torch.equal(candidate["x0"], x0), candidate["step"])
            self.assertTrue(torch.equal(candidate["latent"], latent), candidate["step"])

    def test_standard_and_sde_keep_the_fixed_handoff_unchanged_with_and_without_stop(self):
        _i, x0, latent = fixed_handoff("euler_ancestral", K_TARGET)
        for pipe, prompt in (
            (self.pipe(), _prompt(adaptive.STANDARD_CLASS)),
            (self.pipe(stop_after_handoff=True), _prompt(adaptive.STANDARD_CLASS)),
        ):
            self.calls.clear()
            new_pipe = self.run_module(pipe, prompt)
            self.assertEqual(self.calls, [K_TARGET])
            self.assertNotIn("instantid_handoff_candidates", new_pipe)
            self.assertNotIn("instantid_handoff_window", new_pipe)
            self.assertTrue(torch.equal(new_pipe["samples"]["samples"], latent))
            self.assertTrue(torch.equal(new_pipe["instantid_keypoints_latent"]["samples"], x0))
        self.calls.clear()
        for stop in (False, True):
            self.calls.clear()
            new_pipe = self.run_module(self.pipe("dpmpp_2m_sde", stop_after_handoff=stop), _prompt(adaptive.ADVANCED_CLASS))
            self.assertEqual(self.calls, [K_TARGET])
            self.assertNotIn("instantid_handoff_candidates", new_pipe)


class HandoffCache(unittest.TestCase):
    """Real persistence of module 10's boundary in a temporary directory."""

    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_pipe_sampler")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)

        def directory(scope):
            path = root / scope
            path.mkdir(parents=True, exist_ok=True)
            return path

        cache_patch = patch.object(persistent, "cache_directory", directory)
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        self.addCleanup(persistent.reset_persistent_fingerprint_memo)
        self.prompt = _prompt(adaptive.ADVANCED_CLASS)
        self.plan = adaptive.adaptive_handoff_plan(_pipe(), self.prompt, "10")

    def pipe(self, plan, steps, approved=True):
        # A handoff the ContentGuard approved, as module 10 hands it over in
        # BASELINE and RELEASE alike.
        pipe = _pipe(content_guard_enabled=True, samples={"samples": torch.ones(1)})
        if approved:
            pipe["content_guard_handoff"] = "PASS · SDXL HANDOFF"
        if plan is not None:
            pipe["instantid_handoff_window"] = plan
            pipe["instantid_handoff_candidates"] = [
                {"step": step, "x0": torch.full((1,), float(step)), "latent": torch.full((1,), -float(step))}
                for step in steps
            ]
        return pipe

    def persist(self, pipe):
        persistent.reset_persistent_fingerprint_memo()
        return self.module._persist_sdxl_handoff(pipe, {"providers": []}, self.prompt, "10")

    def load(self, pipe):
        persistent.reset_persistent_fingerprint_memo()
        return self.module._load_sdxl_handoff(pipe, {"providers": []}, self.prompt, "10")

    def test_adaptive_entry_is_reused_with_every_candidate_in_search_order(self):
        self.assertIsNotNone(self.persist(self.pipe(self.plan, ORDER)))
        restored, _visual = self.load(self.pipe(self.plan, []))
        self.assertTrue(restored["sdxl_handoff_cache_hit"])
        self.assertEqual([c["step"] for c in restored["instantid_handoff_candidates"]], ORDER)
        backward = restored["instantid_handoff_candidates"][ORDER.index(3)]
        self.assertEqual((float(backward["x0"][0]), float(backward["latent"][0])), (3.0, -3.0))

    def test_old_adaptive_window_entry_is_not_found(self):
        old_plan = {"version": "cmk-adaptive-handoff-v1", "target": 8, "forward": 14}
        self.persist(self.pipe(old_plan, list(range(8, 15))))
        self.assertIsNone(self.load(self.pipe(self.plan, [])))

    def test_entry_without_the_backward_candidates_is_rejected_and_removed(self):
        key = self.persist(self.pipe(self.plan, [8, 9, 10, 11, 12]))
        self.assertIsNone(self.load(self.pipe(self.plan, [])))
        self.assertFalse(persistent.pickle_available("sdxl_handoff_boundary", key))

    def test_fixed_and_adaptive_entries_live_side_by_side(self):
        fixed_key = self.persist(self.pipe(None, []))
        adaptive_key = self.persist(self.pipe(self.plan, ORDER))
        self.assertNotEqual(fixed_key, adaptive_key)
        restored, _visual = self.load(self.pipe(None, []))
        self.assertNotIn("instantid_handoff_candidates", restored)
        self.assertIsNotNone(self.load(self.pipe(self.plan, [])))

    def test_release_never_persists_an_unapproved_handoff(self):
        pipe = self.pipe(self.plan, ORDER, approved=False)
        pipe["content_guard_enabled"] = False  # a saved OFF value
        with patch.object(distribution, "DISTRIBUTION_MODE", distribution.RELEASE):
            self.assertIsNone(self.persist(pipe))


class StopAndResume(unittest.TestCase):
    """STOP AFTER HANDOFF -> resume through CMKKSamplerPipe._run with real persistence.

    Module 01 carries the STOP widget; the node fingerprint ignores it, so STOP
    and resume share a key exactly when module 10 samples the same thing.
    """

    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_pipe_sampler")

    def setUp(self):
        import comfy.model_management
        import comfy.sample
        import latent_preview
        import nodes

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)

        def directory(scope):
            path = root / scope
            path.mkdir(parents=True, exist_ok=True)
            return path

        self.calls = []
        self.decoded = []

        def fake_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                        last_step=None, callback=None, **kwargs):
            self.calls.append(last_step)
            return sample(sampler_name, last_step, callback)

        def fake_prepare_callback(model, steps, x0_output_dict=None):
            def callback(step, x0, x, total):
                x0_output_dict["x0"] = x0
            return callback

        decoded = self.decoded

        class RecordingDecode:
            def decode(self, vae, samples):
                decoded.append(samples["samples"].clone())
                return (torch.zeros(1, 8, 8, 3),)

        module = self.module
        for item in (
            patch.object(comfy.sample, "sample", fake_sample),
            patch.object(comfy.sample, "fix_empty_latent_channels", lambda model, latent, *a, **k: latent),
            patch.object(comfy.sample, "prepare_noise", lambda latent, seed, inds=None: torch.zeros_like(latent)),
            patch.object(comfy.model_management, "intermediate_device", lambda: torch.device("cpu")),
            patch.object(latent_preview, "prepare_callback", fake_prepare_callback),
            patch.object(nodes, "VAEDecode", RecordingDecode),
            patch.object(persistent, "cache_directory", directory),
            patch.object(module, "effective_content_guard_enabled", lambda pipe: False),
            patch.object(module, "send_final_preview", lambda *a, **k: None),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(persistent.reset_persistent_fingerprint_memo)

    @staticmethod
    def prompt(downstream=adaptive.ADVANCED_CLASS, stop=False, seed=1565304366, sampler="euler_ancestral",
               percent=30, positive="two women in a park"):
        prompt = _prompt(downstream)
        prompt["1"]["inputs"] = {
            "stop_after_handoff": stop, "seed": seed, "sampler_name": sampler,
            "instantid_handoff_percent": percent, "prompt_pos": positive,
        }
        return prompt

    def run_module(self, stop=False, sampler="euler_ancestral", target=K_TARGET, downstream=adaptive.ADVANCED_CLASS,
                   **prompt_inputs):
        persistent.reset_persistent_fingerprint_memo()
        self.calls.clear()
        self.decoded.clear()
        pipe = ModuleTenRun.pipe(sampler, stop_after_handoff=stop, instantid_end_at_step=target)
        prompt = self.prompt(downstream, stop, sampler=sampler, **prompt_inputs)
        new_pipe, _visual = self.module.CMKKSamplerPipe()._run(pipe, None, prompt, "10")
        key = self.module._sdxl_handoff_cache_key(
            prompt, "10", adaptive.plan_token(new_pipe.get("instantid_handoff_window")),
        )
        return new_pipe, key

    @staticmethod
    def steps(new_pipe):
        return [candidate["step"] for candidate in new_pipe.get("instantid_handoff_candidates") or []]

    def assert_same_candidates(self, left, right):
        self.assertEqual(self.steps(left), self.steps(right))
        for a, b in zip(left["instantid_handoff_candidates"], right["instantid_handoff_candidates"]):
            self.assertTrue(torch.equal(a["x0"], b["x0"]), a["step"])
            self.assertTrue(torch.equal(a["latent"], b["latent"]), a["step"])

    def test_stop_resume_resume_stop_samples_once_then_hits(self):
        stop, stop_key = self.run_module(stop=True)
        self.assertEqual(self.calls, [K_FORWARD])
        self.assertFalse(stop.get("sdxl_handoff_cache_hit", False))
        self.assertEqual(self.steps(stop), ORDER)
        self.assertEqual(adaptive.plan_token(stop["instantid_handoff_window"]), "cmk-adaptive-handoff-v2:8-12")
        sequence = [("STOP", stop_key, False)]
        for label, stop_flag in (("resume", False), ("resume again", False), ("STOP again", True)):
            new_pipe, key = self.run_module(stop=stop_flag)
            sequence.append((label, key, new_pipe.get("sdxl_handoff_cache_hit", False)))
            self.assertEqual(self.calls, [], label)  # no SDXL sampling at all
            self.assert_same_candidates(new_pipe, stop)
            self.assertTrue(torch.equal(new_pipe["samples"]["samples"], stop["samples"]["samples"]))
            self.assertTrue(torch.equal(new_pipe["instantid_keypoints_latent"]["samples"],
                                        stop["instantid_keypoints_latent"]["samples"]))
        self.assertEqual({key for _label, key, _hit in sequence}, {stop_key})
        self.assertEqual([hit for _label, _key, hit in sequence], [False, True, True, True])

    def test_stop_candidates_equal_the_direct_adaptive_run_bit_for_bit(self):
        stop, stop_key = self.run_module(stop=True)
        with tempfile.TemporaryDirectory() as other:
            def directory(scope):
                path = Path(other) / scope
                path.mkdir(parents=True, exist_ok=True)
                return path

            with patch.object(persistent, "cache_directory", directory):
                direct, direct_key = self.run_module(stop=False)
        self.assertEqual(self.calls, [K_FORWARD])  # the direct run really sampled
        self.assertEqual(stop_key, direct_key)
        self.assert_same_candidates(stop, direct)
        for candidate in stop["instantid_handoff_candidates"]:  # and both equal the fixed handoff per step
            _i, x0, latent = fixed_handoff("euler_ancestral", candidate["step"])
            self.assertTrue(torch.equal(candidate["x0"], x0), candidate["step"])
            self.assertTrue(torch.equal(candidate["latent"], latent), candidate["step"])

    def test_stop_preview_shows_the_target_step_x0(self):
        stop, _key = self.run_module(stop=True)
        _i, x0, latent = fixed_handoff("euler_ancestral", K_TARGET)
        self.assertEqual(len(self.decoded), 1)
        self.assertTrue(torch.equal(self.decoded[0], x0))  # image_1st_pass = x0 of the target step
        self.assertTrue(torch.equal(stop["samples"]["samples"], latent))
        self.assertEqual(stop["instantid_end_at_step"], K_TARGET)

    def test_standard_and_unsupported_samplers_keep_the_fixed_handoff_across_stop(self):
        for downstream, sampler in ((adaptive.STANDARD_CLASS, "euler_ancestral"), (adaptive.ADVANCED_CLASS, "dpmpp_2m_sde")):
            stop, stop_key = self.run_module(stop=True, sampler=sampler, downstream=downstream)
            self.assertEqual(self.calls, [K_TARGET], (downstream, sampler))
            self.assertNotIn("instantid_handoff_window", stop)
            resumed, key = self.run_module(stop=False, sampler=sampler, downstream=downstream)
            self.assertEqual((self.calls, key, resumed["sdxl_handoff_cache_hit"]), ([], stop_key, True))
            self.assertNotIn("instantid_handoff_candidates", resumed)

    def test_changes_invalidate(self):
        _stop, stop_key = self.run_module(stop=True)
        for label, kwargs, forward in (
            ("seed", {"seed": 1}, K_FORWARD),
            ("sampler", {"sampler": "euler"}, K_FORWARD),
            ("target step", {"percent": 40, "target": 10}, 14),
            ("conditioning", {"positive": "two men in a park"}, K_FORWARD),
        ):
            new_pipe, key = self.run_module(stop=False, **kwargs)
            self.assertNotEqual(key, stop_key, label)
            self.assertFalse(new_pipe.get("sdxl_handoff_cache_hit", False), label)
            self.assertEqual(self.calls, [forward], label)

    def test_old_fixed_stop_entry_is_not_reused_as_adaptive(self):
        # A STOP entry written before this change: fixed handoff, no candidates.
        persistent.reset_persistent_fingerprint_memo()
        old = _pipe(stop_after_handoff=True, content_guard_enabled=False, samples={"samples": torch.ones(1)})
        self.module._persist_sdxl_handoff(old, {"providers": []}, self.prompt(stop=True), "10")
        resumed, _key = self.run_module(stop=False)
        self.assertEqual(self.calls, [K_FORWARD])
        self.assertFalse(resumed.get("sdxl_handoff_cache_hit", False))
        self.assertEqual(self.steps(resumed), ORDER)

    def test_contract_mismatch_under_the_same_key_is_rejected_and_removed(self):
        _stop, key = self.run_module(stop=True)
        persistent.reset_persistent_fingerprint_memo()
        pipe = ModuleTenRun.pipe("euler_ancestral", denoise=0.9)  # not part of this fake prompt -> same key
        pipe["instantid_handoff_window"] = adaptive.adaptive_handoff_plan(pipe, self.prompt(), "10")
        self.assertIsNone(self.module._load_sdxl_handoff(pipe, {"providers": []}, self.prompt(), "10"))
        self.assertFalse(persistent.pickle_available("sdxl_handoff_boundary", key))

    def test_contract_covers_every_sampling_input(self):
        contract_fields = re.search(r"_SDXL_HANDOFF_CONTRACT_FIELDS = \((.*?)\)", SAMPLER_SOURCE, re.S).group(1)
        for name in ("steps_1st_pass", "instantid_end_at_step", "instantid_handoff_percent", "seed", "cfg",
                     "sampler", "scheduler", "denoise", "sdxl_width", "sdxl_height", "boolean_inpaint_mode",
                     "controlnet_applied", "instantid_handoff_window"):
            self.assertIn(f'"{name}"', contract_fields)


if __name__ == "__main__":
    unittest.main()
