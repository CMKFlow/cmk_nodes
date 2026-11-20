"""Adaptive SDXL -> InstantID Advanced handshake.

The fixed handoff (module 10 samples to step k, InstantID continues from k)
detects the target faces on a single x0 estimate.  Faces still form at that
point, so one of two persons may not be detectable yet.  For InstantID
Advanced the handoff slider's step becomes the *target* handoff: module 10
samples up to ``k_target + ADAPTIVE_FORWARD_STEPS`` and keeps one candidate
per step from 1 on.  Advanced checks them in the search order

    k_target, k_target + 1 .. k_forward   (more SDXL work first)
    k_target - 1 .. 1                     (earlier states as last resort)

and continues from the first candidate whose target faces it can resolve with
its unchanged TARGET selection.

Pairing (identical to the fixed handoff, see tests/test_instantid_adaptive_handoff.py):
ComfyUI's sampler callback reports ``(i, denoised, x)`` with ``x`` the latent
*before* step i.  The fixed handoff at k uses ``denoised`` of i = k - 1 as the
keypoint x0 and continues from the latent returned by ``last_step = k``, which
equals the callback ``x`` of i = k.  Candidate k therefore pairs x0(i = k - 1)
with x(i = k); for k = k_forward the latent is the returned tensor.

Only samplers whose trajectory up to k does not depend on where sampling stops
(prefix-stable) take part; every other configuration keeps the fixed handoff.
Module 10 never detects faces and never sees the TARGET settings.
"""

from __future__ import annotations

ADAPTIVE_HANDOFF_VERSION = "cmk-adaptive-handoff-v2"
ADAPTIVE_FORWARD_STEPS = 4

ADVANCED_CLASS = "CMKInstantIDRegionalAdvancedSDXLPipe"
STANDARD_CLASS = "CMKInstantIDSamplerSDXLPipe"
INSTANTID_SAMPLER_CLASSES = (ADVANCED_CLASS, STANDARD_CLASS)

# ComfyUI samplers proven prefix-stable by the model-free reference test
# (tests/test_instantid_adaptive_handoff.py runs every name against a stopped
# reference run).  Brownian-tree SDE samplers, ipndm/deis, heunpp2 and
# dpm_fast/dpm_adaptive depend on the full sigma schedule and are excluded;
# untested CFG++ variants are excluded as well.
PREFIX_STABLE_SAMPLERS = frozenset({
    "euler",
    "euler_ancestral",
    "heun",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpmpp_2s_ancestral",
    "dpmpp_2m",
    "ddpm",
    "lcm",
    "res_multistep",
    "res_multistep_ancestral",
    "gradient_estimation",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
})


def handoff_window(steps: int, target: int) -> tuple[int, int]:
    """``(k_target, k_forward)``: the slider's handoff step and the last forward one.

    A handoff at k continues with InstantID from step k, so k <= steps - 1
    leaves at least one InstantID step.  ``k_forward`` = k_target +
    ADAPTIVE_FORWARD_STEPS within that limit.
    """
    steps = int(steps)
    last = max(1, steps - 1)
    k_target = max(1, min(last, int(target)))
    return k_target, min(k_target + ADAPTIVE_FORWARD_STEPS, last)


def search_order(target: int, forward: int) -> list[int]:
    """Candidate steps in the order Advanced checks them."""
    return list(range(int(target), int(forward) + 1)) + list(range(int(target) - 1, 0, -1))


def _prompt_node(prompt, node_id):
    node = prompt.get(str(node_id)) if isinstance(prompt, dict) else None
    return node if isinstance(node, dict) else None


def _is_link(prompt, value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and str(value[0]) in prompt
    )


def downstream_instantid_samplers(prompt, ksampler_id) -> list[tuple[str, str]]:
    """``(node id, class)`` of every InstantID sampler fed by this module 10."""
    if not isinstance(prompt, dict) or ksampler_id is None:
        return []
    target = str(ksampler_id)
    found = []
    for node_id, node in prompt.items():
        if not isinstance(node, dict) or node.get("class_type") not in INSTANTID_SAMPLER_CLASSES:
            continue
        pending = [value[0] for value in (node.get("inputs") or {}).values() if _is_link(prompt, value)]
        visited = set()
        while pending:
            current = str(pending.pop())
            if current in visited:
                continue
            visited.add(current)
            if current == target:
                found.append((str(node_id), str(node["class_type"])))
                break
            upstream = _prompt_node(prompt, current)
            if upstream is None:
                continue
            pending.extend(
                value[0] for value in (upstream.get("inputs") or {}).values() if _is_link(prompt, value)
            )
    return found


def adaptive_handoff_plan(pipe, prompt, ksampler_id):
    """The adaptive window for this run, or ``None`` for the fixed handoff.

    STOP AFTER HANDOFF samples the same window: STOP and the later resume then
    produce the identical boundary and share one handoff cache entry.
    """
    if not isinstance(pipe, dict):
        return None
    if not pipe.get("instantid_enabled", False) or pipe.get("hybrid_mode", False):
        return None
    if str(pipe.get("instantid_mode", "HYBRID") or "HYBRID").strip().upper() != "HYBRID":
        return None
    if pipe.get("boolean_inpaint_mode", False):
        return None
    if str(pipe.get("model_family", "sdxl")).strip().lower() != "sdxl":
        return None
    if str(pipe.get("sampler", "")) not in PREFIX_STABLE_SAMPLERS:
        return None
    steps = int(pipe.get("steps_1st_pass", pipe.get("steps", 20)))
    target = int(pipe.get("instantid_end_at_step", 0) or 0)
    if target < 1 or steps < 2:
        return None
    samplers = downstream_instantid_samplers(prompt, ksampler_id)
    if len(samplers) != 1 or samplers[0][1] != ADVANCED_CLASS:
        return None
    k_target, k_forward = handoff_window(steps, target)
    return {"version": ADAPTIVE_HANDOFF_VERSION, "target": k_target, "forward": k_forward}


def plan_token(plan) -> str:
    """Cache identity of the handoff mode; differs between fixed and adaptive."""
    if not plan:
        return "fixed"
    return f"{plan['version']}:{plan['target']}-{plan['forward']}"


def search_label(steps) -> str:
    """``8-12 forward · 7-1 backward`` for candidate steps in search order."""
    target = steps[0]
    forward = [step for step in steps if step >= target]
    backward = [step for step in steps if step < target]
    label = f"{forward[0]}-{forward[-1]} forward" if len(forward) > 1 else f"{forward[0]} forward"
    if backward:
        label += f" · {backward[0]}-{backward[-1]} backward" if len(backward) > 1 else f" · {backward[0]} backward"
    return label


class CandidateRecorder:
    """Wraps the preview callback and keeps the sampler states of steps 0 .. k_forward - 1."""

    def __init__(self, target: int, forward: int, preview_callback=None):
        self.order = search_order(target, forward)
        self.forward = int(forward)
        self.preview_callback = preview_callback
        self.x0 = {}
        self.latents = {}

    def __call__(self, step, x0, x, total_steps):
        step = int(step)
        if step <= self.forward - 1:
            self.x0[step] = x0.detach().clone()
            if step >= 1:
                self.latents[step] = x.detach().clone()
        if self.preview_callback is not None:
            self.preview_callback(step, x0, x, total_steps)

    def candidates(self, final_samples, process_latent_out, intermediate_device):
        """Candidate ``k`` = ``{"step", "x0", "latent"}`` for every step, in search order.

        x0 is converted like the fixed handoff's keypoint latent (CPU, then
        process_latent_out); the latent like ComfyUI's own sampler result
        (float32 process_latent_out, then the intermediate device).
        """
        import torch

        result = []
        for step in self.order:
            x0 = self.x0.get(step - 1)
            if step == self.forward:
                latent = final_samples
            else:
                raw = self.latents.get(step)
                latent = None if raw is None else process_latent_out(raw.to(torch.float32)).to(intermediate_device)
            if x0 is None or latent is None:
                raise RuntimeError(
                    f"CMK KSampler -Pipe-: adaptive InstantID handoff is missing the sampler state of step {step}"
                )
            result.append({
                "step": step,
                "x0": process_latent_out(x0.cpu()),
                "latent": latent,
            })
        return result


class CandidateRejected(ValueError):
    """A candidate whose target faces cannot be resolved."""


def first_resolved_candidate(candidates, evaluate):
    """First candidate ``evaluate`` accepts; ``evaluate`` raises ValueError to reject.

    Returns ``(candidate, result, attempts)``.  ``attempts`` lists
    ``(step, None | reason)`` for every evaluated candidate.  Raises
    CandidateRejected with every attempt when none is accepted.
    """
    attempts = []
    for candidate in candidates:
        try:
            result = evaluate(candidate)
        except ValueError as exc:
            attempts.append((int(candidate["step"]), str(exc)))
            continue
        attempts.append((int(candidate["step"]), None))
        return candidate, result, attempts
    raise CandidateRejected(attempts)
