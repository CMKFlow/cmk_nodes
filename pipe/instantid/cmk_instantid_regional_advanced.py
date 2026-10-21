from __future__ import annotations

import inspect
import os

import numpy as np
import torch
import torch.nn.functional as F
from comfy.controlnet import ControlBase

from ..cmk_log_pipe import cmk_add_block
from ..cmk_module_cache_contract import stamp_artifact
from ..cmk_visual import empty_visual, register_provider
from ..cmk_visual import empty_visual, register_provider
from ...utils.cmk_translation import translate_prompt
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...utils.cmk_timing import cmk_timed
from ...utils.cmk_sampling_warnings import ignore_torchsde_boundary_rounding
from ...cmk_common import SAMPLERS, SCHEDULERS
from ...cmk_distribution import (
    CONTENT_GUARD_EXECUTION_KEY,
    content_guard_state_token,
    effective_content_guard_enabled,
)
from ...engine.content_guard import (
    ContentGuardBlocked,
    GUARD_VERSION,
    cleanup_content_guard_abort,
    get_content_guard,
)
from ...engine.instantid_face_rebuild import select_target_face
from ...utils.tensor_utils import tensor_to_uint8_rgb
from .upstream.InstantID import InstantIDAttentionPatch, extractFeatures


INSTANTID_PROVIDERS = ["CoreML", "CPU", "CUDA", "ROCM"]
INSTANTID_GUARD_VERSION = f"{GUARD_VERSION}:instantid-midpoint-final-target-v2"
REGIONAL_ADVANCED_VERSION = "instantid-regional-advanced-v3"

# TEMPORARY DIAGNOSTIC: compare module 10's complete layout-patched model with
# the dedicated identity branch. Revert to False after the controlled test.
_DIAGNOSTIC_USE_LAYOUT_PATCHED_MODEL = False
_INSTANTID_STAGE_KEY = "sdxl.identity"

def _filename_list(category):
    try:
        import folder_paths

        return list(folder_paths.get_filename_list(category))
    except Exception:
        return []


def _instantid_models():
    models = _filename_list("instantid")
    if models:
        return models
    try:
        import folder_paths

        directory = os.path.join(folder_paths.models_dir, "instantid")
        extensions = tuple(str(ext).lower() for ext in folder_paths.supported_pt_extensions)
        return sorted(
            name for name in os.listdir(directory) if name.lower().endswith(extensions)
        ) if os.path.isdir(directory) else []
    except Exception:
        return []


def _node_class(*names):
    try:
        import nodes
    except Exception as exc:
        raise RuntimeError(f"ComfyUI node registry unavailable: {exc}") from exc
    mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for name in names:
        if name in mappings:
            return mappings[name]
        value = getattr(nodes, name, None)
        if value is not None:
            return value
    if any(name.startswith("CMKInternalInstantID") or name == "CMKInternalApplyInstantIDAdvanced" for name in names):
        try:
            from .upstream import CMK_INSTANTID_CLASSES

            for name in names:
                if name in CMK_INSTANTID_CLASSES:
                    return CMK_INSTANTID_CLASSES[name]
        except Exception as exc:
            raise RuntimeError(
                "CMK InstantID runtime unavailable. Install the CMK requirements "
                f"(insightface/onnxruntime). Details: {exc}"
            ) from exc
    raise RuntimeError(f"Required node is missing: {' / '.join(names)}")


def _call_node(names, **kwargs):
    cls = _node_class(*names)
    instance = cls()
    function = getattr(instance, getattr(cls, "FUNCTION"))
    signature = inspect.signature(function)
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return function(**accepted)


def _outputs(value):
    if isinstance(value, dict) and "result" in value:
        value = value["result"]
    return value if isinstance(value, (tuple, list)) else (value,)


def _require_dict(pipe, name):
    if not isinstance(pipe, dict):
        raise TypeError(f"CMK InstantID Sampler SDXL -Pipe-: {name} must be a CMK pipe")
    return pipe


def _first(*values):
    return next((value for value in values if value is not None), None)


def _guard_faces(faceanalysis, image_rgb):
    """Detect policy faces without creating a ComfyUI-visible image output."""
    bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    detector = getattr(faceanalysis, "det_model", None)
    for size in range(640, 128, -64):
        if detector is not None:
            detector.input_size = (size, size)
        faces = list(faceanalysis.get(bgr) or [])
        if faces:
            return faces
    return []


def _guard_source_images(faceanalysis, source_images):
    guard = get_content_guard()
    for image in source_images:
        rgb = tensor_to_uint8_rgb(image)
        faces = _guard_faces(faceanalysis, rgb)
        if not faces:
            raise ContentGuardBlocked("CG_FACE_UNAVAILABLE", "source")
        # InstantID averages all detected donor embeddings. Therefore every
        # participating source face must satisfy the source policy.
        for face in faces:
            guard.inspect_image(rgb, face, "source")


def _guard_generated_latent(result, vae, faceanalysis, target_selection):
    """Internally decode and inspect; never return, preview, cache, or persist it."""
    decoded = None
    try:
        decoded = _outputs(_call_node(("VAEDecode",), samples=result, vae=vae))[0]
        if decoded is None or int(decoded.shape[0]) < 1:
            raise ContentGuardBlocked("CG_INPUT_INVALID", "target")
        guard = get_content_guard()
        for image in decoded:
            rgb = tensor_to_uint8_rgb(image)
            faces = _guard_faces(faceanalysis, rgb)
            if not faces:
                raise ContentGuardBlocked("CG_FACE_UNAVAILABLE", "target")
            face = select_target_face(
                faces, str(target_selection), rgb.shape[1], rgb.shape[0],
            )
            guard.inspect_image(rgb, face, "target")
    finally:
        # Policy-local only: no node/subgraph preview, diagnostic, pipe value,
        # boundary-cache entry, or temporary file may retain this tensor.
        del decoded


def _guard_generated_latent_multi(result, vae, faceanalysis, target_selections):
    """Decode once and inspect every distinct generated identity target."""
    decoded = None
    try:
        decoded = _outputs(_call_node(("VAEDecode",), samples=result, vae=vae))[0]
        if decoded is None or int(decoded.shape[0]) < 1:
            raise ContentGuardBlocked("CG_INPUT_INVALID", "target")
        guard = get_content_guard()
        for image in decoded:
            rgb = tensor_to_uint8_rgb(image)
            faces = _guard_faces(faceanalysis, rgb)
            if not faces:
                raise ContentGuardBlocked("CG_FACE_UNAVAILABLE", "target")
            inspected = set()
            for selection in target_selections:
                face = select_target_face(faces, str(selection), rgb.shape[1], rgb.shape[0])
                bbox = tuple(round(float(value), 1) for value in face["bbox"])
                if bbox not in inspected:
                    guard.inspect_image(rgb, face, "target")
                    inspected.add(bbox)
    finally:
        del decoded


def _regional_hint(faceanalysis, layout_image, target):
    result = extractFeatures(
        faceanalysis, layout_image, extract_kps=True,
        selection=str(target), return_mask=True,
    )
    if not isinstance(result, tuple):
        raise ValueError(
            "CMK InstantID Regional Advanced: target face could not be detected "
            "in the preliminary prompt layout"
        )
    return result


def _regional_hints(faceanalysis, layout_image, target_a, target_b):
    hint_a, mask_a = _regional_hint(faceanalysis, layout_image, target_a)
    hint_b, mask_b = _regional_hint(faceanalysis, layout_image, target_b)
    binary_a = mask_a.detach().float().cpu() >= 0.5
    binary_b = mask_b.detach().float().cpu() >= 0.5
    union = (binary_a | binary_b).sum().item()
    overlap = ((binary_a & binary_b).sum().item() / union) if union else 1.0
    if overlap >= 0.9:
        raise ValueError(
            "CMK InstantID Regional Advanced: TARGET A and TARGET B resolve to the same person"
        )
    return hint_a, mask_a, hint_b, mask_b, overlap


def _merge_control_outputs(current, addition):
    if current is None:
        return addition
    if addition is None:
        return current
    merged = {}
    for key in set(current) | set(addition):
        left = current.get(key, [])
        right = addition.get(key, [])
        values = []
        for index in range(max(len(left), len(right))):
            a = left[index] if index < len(left) else None
            b = right[index] if index < len(right) else None
            values.append(b if a is None else (a if b is None else a + b))
        merged[key] = values
    return merged


class _RegionalInstantIDControl(ControlBase):
    """Run each InstantID ControlNet with its own embedding and spatial mask."""

    def __init__(self, branches):
        super().__init__()
        self.branches = branches

    @staticmethod
    def _context(branch, x_noisy, transformer_options):
        cond_or_uncond = transformer_options.get("cond_or_uncond", [0])
        batch_prompt = max(1, x_noisy.shape[0] // max(1, len(cond_or_uncond)))
        cond = branch[1].repeat(batch_prompt, 1, 1)
        uncond = branch[2].repeat(batch_prompt, 1, 1)
        return torch.cat([(cond, uncond)[index] for index in cond_or_uncond], dim=0)

    @staticmethod
    def _masked(control, mask):
        if control is None:
            return None
        masked = {}
        for key, values in control.items():
            masked[key] = []
            for value in values:
                if value is None or value.ndim < 4:
                    masked[key].append(value)
                    continue
                spatial = F.interpolate(
                    mask.to(device=value.device, dtype=value.dtype).unsqueeze(1),
                    size=value.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                if spatial.shape[0] != value.shape[0]:
                    spatial = spatial.repeat(value.shape[0], 1, 1, 1)
                masked[key].append(value * spatial)
        return masked

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        result = None
        if self.previous_controlnet is not None:
            result = self.previous_controlnet.get_control(
                x_noisy, t, cond, batched_number, transformer_options,
            )
        for branch in self.branches:
            local_cond = dict(cond)
            local_cond["crossattn_controlnet"] = self._context(
                branch, x_noisy, transformer_options,
            )
            regional = branch[0].get_control(
                x_noisy, t, local_cond, batched_number, transformer_options,
            )
            result = _merge_control_outputs(result, self._masked(regional, branch[3]))
        return result

    def pre_run(self, model, percent_to_timestep_function):
        if self.previous_controlnet is not None:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)
        for control, *_ in self.branches:
            control.pre_run(model, percent_to_timestep_function)

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        for control, *_ in self.branches:
            control.cleanup()

    def get_models(self):
        models = self.previous_controlnet.get_models() if self.previous_controlnet is not None else []
        for control, *_ in self.branches:
            models += control.get_models()
        return models

    def get_extra_hooks(self):
        hooks = self.previous_controlnet.get_extra_hooks() if self.previous_controlnet is not None else []
        for control, *_ in self.branches:
            hooks += control.get_extra_hooks()
        return hooks

    def inference_memory_requirements(self, dtype):
        memory = self.previous_controlnet.inference_memory_requirements(dtype) if self.previous_controlnet is not None else 0
        return memory + sum(control.inference_memory_requirements(dtype) for control, *_ in self.branches)

    def copy(self):
        copied = _RegionalInstantIDControl([
            (control.copy(), cond, uncond, mask)
            for control, cond, uncond, mask in self.branches
        ])
        if self.previous_controlnet is not None:
            copied.set_previous_controlnet(self.previous_controlnet.copy())
        return copied


def _apply_regional_controlnet(control_net, branches, positive, negative, start_at, end_at):
    cached = {}
    outputs = []
    for conditioning in (positive, negative):
        branch_conditioning = []
        for tensor, metadata in conditioning:
            data = metadata.copy()
            previous = data.get("control")
            cache_key = id(previous)
            regional = cached.get(cache_key)
            if regional is None:
                regional_branches = []
                for hint, mask, embeds, strength in branches:
                    control = control_net.copy().set_cond_hint(
                        hint.movedim(-1, 1), float(strength),
                        (float(start_at), float(end_at)),
                    )
                    regional_branches.append(
                        (control, embeds["cond"], embeds["uncond"], mask)
                    )
                regional = _RegionalInstantIDControl(regional_branches)
                regional.set_previous_controlnet(previous)
                cached[cache_key] = regional
            data["control"] = regional
            data["control_apply_to_uncond"] = False
            data.pop("cross_attn_controlnet", None)
            branch_conditioning.append([tensor, data])
        outputs.append(branch_conditioning)
    return outputs[0], outputs[1]


def _guard_midpoint_x0(x0_latent, vae):
    """Early explicit-content gate for a clean midpoint x0 estimate."""
    import comfy.model_management

    decoded = None
    # VAEDecode may need enough MPS memory to make ComfyUI offload models that
    # belong to the sampler currently paused inside this callback.  Sampling
    # resumes in the same prepared executor, so those models (in particular the
    # InstantID ControlNet) must be restored before returning to the next step.
    loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
    try:
        decoded = _outputs(_call_node(("VAEDecode",), samples=x0_latent, vae=vae))[0]
        if decoded is None or int(decoded.shape[0]) < 1:
            raise ContentGuardBlocked("CG_INPUT_INVALID", "target_intermediate")
        guard = get_content_guard()
        for image in decoded:
            guard.inspect_content(
                tensor_to_uint8_rgb(image), "target_intermediate",
            )
    finally:
        # The midpoint decode is policy-local and never enters a public output.
        del decoded
        comfy.model_management.load_models_gpu(loaded_models)


def _sample_with_midpoint_guard(
    *, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
    positive, negative, latent_image, start_at_step, end_at_step, vae,
):
    """Run the normal preview callback and synchronously gate its midpoint x0."""
    import torch
    import comfy.sample
    import comfy.utils
    import latent_preview

    latent_tensor = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image["samples"],
        latent_image.get("downscale_ratio_spacial"),
        latent_image.get("downscale_ratio_temporal"),
    )
    if add_noise == "disable":
        noise = torch.zeros(
            latent_tensor.size(), dtype=latent_tensor.dtype, layout=latent_tensor.layout,
            device="cpu",
        )
    else:
        noise = comfy.sample.prepare_noise(
            latent_tensor, noise_seed, latent_image.get("batch_index"),
        )

    preview_callback = latent_preview.prepare_callback(model, steps)
    midpoint_checked = False

    def guarded_preview_callback(step, x0, x, total_steps):
        nonlocal midpoint_checked
        # Preserve the established sampler preview exactly; policy evaluation
        # then pauses the same callback before the next denoising step begins.
        preview_callback(step, x0, x, total_steps)
        midpoint = max(1, (int(total_steps) + 1) // 2)
        if not midpoint_checked and int(total_steps) >= 2 and int(step) + 1 >= midpoint:
            midpoint_checked = True
            x0_latent = latent_image.copy()
            x0_latent.pop("downscale_ratio_spacial", None)
            x0_latent.pop("downscale_ratio_temporal", None)
            x0_latent["samples"] = model.model.process_latent_out(x0.detach().cpu())
            try:
                _guard_midpoint_x0(x0_latent, vae)
            finally:
                del x0_latent

    sampled_tensor = comfy.sample.sample(
        model,
        noise,
        int(steps),
        float(cfg),
        str(sampler_name),
        str(scheduler),
        positive,
        negative,
        latent_tensor,
        denoise=1.0,
        disable_noise=add_noise == "disable",
        start_step=int(start_at_step),
        last_step=int(end_at_step),
        force_full_denoise=True,
        noise_mask=latent_image.get("noise_mask"),
        callback=guarded_preview_callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=int(noise_seed),
    )
    result = latent_image.copy()
    result.pop("downscale_ratio_spacial", None)
    result.pop("downscale_ratio_temporal", None)
    result["samples"] = sampled_tensor
    return result


def _sample_without_content_guard(
    *, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
    positive, negative, latent_image, start_at_step, end_at_step,
):
    """Use ComfyUI's unchanged sampler path when the baseline guard is disabled."""
    return _outputs(_call_node(
        ("KSamplerAdvanced",),
        model=model,
        add_noise=add_noise,
        noise_seed=int(noise_seed),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name=str(sampler_name),
        scheduler=str(scheduler),
        positive=positive,
        negative=negative,
        latent_image=latent_image,
        start_at_step=int(start_at_step),
        end_at_step=int(end_at_step),
        return_with_leftover_noise="disable",
    ))[0]


class CMKInstantIDRegionalAdvancedSDXLPipe:
    """Apply one or two regional InstantID identities during one SDXL sampling pass."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE",),
                "PROCESS": ("CMK_PROCESS_SDXL",),
                "SAMPLED": ("CMK_SAMPLED_PIPE",),
                "LOG": ("CMK_LOG_PIPE",),
                "source_face_a": ("IMAGE",),
                "enable_source_b": ("BOOLEAN", {
                    "default": True,
                    "label": "SOURCE B",
                    "tooltip": "Disable to skip all Source B loading, analysis, attention, ControlNet, and guard work.",
                }),
                "source_face_b": ("IMAGE", {"lazy": True}),
                "identity_strength_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "pose_strength_a": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "target_face_a": (("Largest", "Leftmost", "Rightmost", "Topmost", "Bottommost", "Center"), {
                    "default": "Leftmost",
                    "tooltip": "Person in the preliminary prompt layout that receives identity A.",
                }),
                "identity_strength_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "pose_strength_b": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "target_face_b": (("Largest", "Leftmost", "Rightmost", "Topmost", "Bottommost", "Center"), {
                    "default": "Rightmost",
                    "tooltip": "Person in the preliminary prompt layout that receives identity B.",
                }),
                "instantid_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Start of both InstantID identity attention and keypoint ControlNet guidance.",
                }),
                "instantid_end": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "End of both InstantID identity attention and keypoint ControlNet guidance.",
                }),
                "instantid_model": ((_instantid_models() or [""]), {"advanced": True}),
                "controlnet_model": ((_filename_list("controlnet") or [""]), {"advanced": True}),
                "provider": (INSTANTID_PROVIDERS, {"default": "CoreML", "advanced": True}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.1, "advanced": True}),
                "noise": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                    "advanced": True,
                    "tooltip": (
                        "Noise applied to the unconditional InstantID face embedding. Higher "
                        "values can strengthen identity transfer, especially without module 05. "
                        "This does not control latent sampling noise."
                    ),
                }),
                "conditioning_mode": (
                    ["CMK prepared", "InstantID reference"],
                    {
                        "default": "InstantID reference",
                        "advanced": True,
                        "tooltip": (
                            "InstantID reference: encodes the prompts with the native checkpoint CLIP "
                            "for the identity pass (current CMK default). CMK prepared: reuses the "
                            "conditioning already prepared by module 10, including its CLIP settings."
                        ),
                    },
                ),
                "reference_conditioning_weight": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                    "advanced": True,
                    "tooltip": (
                        "Blend used by InstantID reference: 1.0 is pure native reference "
                        "conditioning; 0.0 is pure CMK prepared conditioning. Ignored in "
                        "CMK prepared mode."
                    ),
                }),
                "reference_start_at_step": ("INT", {
                    "default": 8, "min": 0, "max": 200, "step": 1,
                    "advanced": True,
                    "tooltip": (
                        "Absolute sampling entry for the VAE-encoded reference latent. Higher "
                        "steps preserve more reference composition; lower steps allow more "
                        "redesign. Clamped below Total Steps and used only with InstantID + module 05."
                    ),
                }),
            },
            "optional": {
                "VISUAL": ("CMK_VISUAL_PIPE",),
                "LOG_B": ("CMK_LOG_PIPE", {"lazy": True}),
                CONTENT_GUARD_EXECUTION_KEY: ("BOOLEAN", {"default": True, "advanced": True}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "CMK_PROCESS_SDXL",
        "CMK_SAMPLED_PIPE",
        "CMK_LOG_PIPE",
        "IMAGE",
        "CMK_VISUAL_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("MODEL", "PROCESS", "SAMPLED", "LOG", "IDENTITY IMAGE", "VISUAL", "diagnostic")
    FUNCTION = "sample"
    CATEGORY = "CMK/Toolbox/Face"

    @classmethod
    def IS_CHANGED(cls, PROCESS=None, **kwargs):
        return (
            f"{INSTANTID_GUARD_VERSION}:{REGIONAL_ADVANCED_VERSION}:"
            f"{content_guard_state_token(PROCESS, kwargs.get(CONTENT_GUARD_EXECUTION_KEY))}"
        )

    def check_lazy_status(
        self,
        PROCESS=None,
        enable_source_b=True,
        source_face_b=None,
        **kwargs,
    ):
        if not isinstance(PROCESS, dict) or not PROCESS.get("instantid_enabled", False):
            return []
        if not bool(enable_source_b):
            return []
        missing = []
        if source_face_b is None:
            missing.append("source_face_b")
        if "LOG_B" in kwargs and kwargs["LOG_B"] is None:
            missing.append("LOG_B")
        return missing

    def sample(self, *args, **kwargs):
        try:
            return self._sample(*args, **kwargs)
        except ContentGuardBlocked as exc:
            model_pipe = kwargs.get("MODEL", args[0] if args else None)
            process = kwargs.get("PROCESS", args[1] if len(args) > 1 else None)
            cleanup_content_guard_abort(
                exc,
                label="15R InstantID Regional Advanced",
                model_pipe=model_pipe,
                unload_models=bool(
                    isinstance(process, dict)
                    and process.get("unload_models_after_use", True)
                ),
            )
            args = ()
            kwargs.clear()
            model_pipe = None
            process = None
            raise exc.with_traceback(None) from None

    def _sample(
        self,
        MODEL,
        PROCESS,
        SAMPLED,
        LOG,
        source_face_a,
        enable_source_b,
        identity_strength_a,
        pose_strength_a,
        target_face_a,
        identity_strength_b,
        pose_strength_b,
        target_face_b,
        instantid_start,
        instantid_end,
        instantid_model,
        controlnet_model,
        provider,
        cfg,
        noise,
        conditioning_mode,
        reference_conditioning_weight,
        reference_start_at_step,
        source_face_b=None,
        VISUAL=None,
        LOG_B=None,
        _cmk_content_guard_enabled=None,
        dynprompt=None,
        unique_id=None,
    ):
        _reset_instantid_boundary_cache(dynprompt)
        model_pipe = _require_dict(MODEL, "MODEL")
        process = _require_dict(PROCESS, "PROCESS")
        content_guard_enabled = effective_content_guard_enabled(
            process, _cmk_content_guard_enabled,
        )
        sampled_in = _require_dict(SAMPLED, "SAMPLED")
        visual = empty_visual() if VISUAL is None else VISUAL
        if not bool(process.get("instantid_enabled", False)):
            lines = ["InstantID          : disabled in module 01", "Sampling           : bypassed"]
            log_out = cmk_add_block(LOG, "InstantID Regional Advanced SDXL", 45, lines, True)
            diagnostic = make_diagnostic_payload(
                title="InstantID Regional Advanced SDXL",
                node="CMK InstantID Regional Advanced SDXL -Pipe-",
                previews=[],
                summary="InstantID disabled; pipeline passed through",
                details="\n".join(lines),
                mode="Bypass",
                metadata={"identity_applied": False},
            )
            return model_pipe, process, sampled_in, log_out, None, visual, diagnostic
        source_b_enabled = bool(enable_source_b)
        if source_face_a is None:
            raise ValueError("CMK InstantID Regional Advanced: SOURCE A is required")
        if source_b_enabled and source_face_b is None:
            raise ValueError("CMK InstantID Regional Advanced: SOURCE B is required when SOURCE B is enabled")
        if not instantid_model or not controlnet_model:
            raise ValueError("CMK InstantID Sampler SDXL -Pipe-: InstantID and ControlNet models are required")

        reference_latent_mode = bool(process.get("instantid_reference_latent_mode", False))
        zero_pass_mode = bool(sampled_in.get("instantid_zero_pass", False))
        inpaint_bridge_mode = bool(sampled_in.get("instantid_inpaint_bridge", False))
        reference_image = process.get("instantid_reference_image") if reference_latent_mode else None
        layout_latent = _first(sampled_in.get("samples"), sampled_in.get("latent_1st_pass"), sampled_in.get("latent_image"))
        keypoints_latent = None if (reference_latent_mode or zero_pass_mode) else sampled_in.get("instantid_keypoints_latent")
        vae = _first(model_pipe.get("vae"), sampled_in.get("vae"))
        if _DIAGNOSTIC_USE_LAYOUT_PATCHED_MODEL:
            base_model = _first(
                sampled_in.get("model_patched"),
                sampled_in.get("model_identity"),
                sampled_in.get("model"),
                model_pipe.get("model"),
            )
            model_source = "DIAGNOSTIC: SAMPLED.model_patched (complete module 10 layout patches)"
        else:
            # Normal path: source LoRAs + FreeU, without the local layout
            # Hyper-LoRA, PAG, or layout-only ModelSamplingDiscrete patch.
            base_model = _first(
                sampled_in.get("model_identity"),
                sampled_in.get("model_patched"),
                sampled_in.get("model"),
                model_pipe.get("model"),
            )
            model_source = (
                "SAMPLED.model_identity (source LoRAs + FreeU; no local Hyper-LoRA / PAG / layout sampling patch)"
                if sampled_in.get("model_identity") is not None
                else "SAMPLED.model_patched (compatibility fallback)"
            )
        positive = (
            sampled_in.get("conditioning_identity_pos")
            if inpaint_bridge_mode
            else sampled_in.get("conditioning_pos")
        )
        negative = (
            sampled_in.get("conditioning_identity_neg")
            if inpaint_bridge_mode
            else sampled_in.get("conditioning_neg")
        )
        missing = [name for name, value in (
            ("layout latent", layout_latent),
            (
                ("reference image", reference_image)
                if reference_latent_mode
                else (("zero-pass mode", True) if zero_pass_mode else ("keypoints latent", keypoints_latent))
            ),
            ("VAE", vae), ("MODEL", base_model),
            ("positive conditioning", positive), ("negative conditioning", negative),
        ) if value is None]
        if missing:
            raise ValueError("CMK InstantID Sampler SDXL -Pipe-: missing " + ", ".join(missing))

        conditioning_source = "CMK prepared"
        if str(conditioning_mode) == "InstantID reference":
            prepared_positive = positive
            prepared_negative = negative
            clean_clip = model_pipe.get("clip")
            if clean_clip is None:
                from ..loaders.checkpoint_vae_loader import ensure_sdxl_text_encoder

                clean_clip, _ = ensure_sdxl_text_encoder(model_pipe)
            prompt_pos = str(process.get("prompt_pos", sampled_in.get("prompt_pos", "")) or "")
            prompt_neg = str(process.get("prompt_neg", sampled_in.get("prompt_neg", "")) or "")
            translation_pos = translate_prompt(prompt_pos)
            translation_neg = translate_prompt(prompt_neg)
            with cmk_timed("15 INSTANTID REFERENCE CONDITIONING", "native CLIP Text Encode"):
                reference_positive = _outputs(_call_node(("CLIPTextEncode",), clip=clean_clip, text=translation_pos.text))[0]
                reference_negative = _outputs(_call_node(("CLIPTextEncode",), clip=clean_clip, text=translation_neg.text))[0]
            if bool(process.get("unload_models_after_use", True)):
                from ..loaders.checkpoint_vae_loader import evict_sdxl_text_encoder

                evict_sdxl_text_encoder(model_pipe, sampled_in)
                clean_clip = None
                import gc

                gc.collect()
                try:
                    import comfy.model_management

                    comfy.model_management.cleanup_models_gc()
                    comfy.model_management.soft_empty_cache(force=True)
                except Exception:
                    pass
            reference_weight = max(0.0, min(1.0, float(reference_conditioning_weight)))
            with cmk_timed(
                "15 INSTANTID HYBRID CONDITIONING",
                f"reference={reference_weight:.2f} | CMK prepared={1.0 - reference_weight:.2f}",
            ):
                positive = _outputs(_call_node(
                    ("ConditioningAverage",),
                    conditioning_to=prepared_positive,
                    conditioning_from=reference_positive,
                    conditioning_to_strength=1.0 - reference_weight,
                ))[0]
                negative = _outputs(_call_node(
                    ("ConditioningAverage",),
                    conditioning_to=prepared_negative,
                    conditioning_from=reference_negative,
                    conditioning_to_strength=1.0 - reference_weight,
                ))[0]
            conditioning_source = (
                f"hybrid: {reference_weight:.0%} native reference | "
                f"{1.0 - reference_weight:.0%} CMK prepared"
            )

        if reference_latent_mode:
            width = int(sampled_in.get("sdxl_width", process.get("sdxl_width", 1024)))
            height = int(sampled_in.get("sdxl_height", process.get("sdxl_height", 1024)))
            with cmk_timed("15 INSTANTID REFERENCE RESIZE", f"{width}x{height}"):
                layout_image = _outputs(_call_node(
                    ("ImageScale",),
                    image=reference_image,
                    upscale_method="lanczos",
                    width=width,
                    height=height,
                    crop="disabled",
                ))[0]
            with cmk_timed("15 INSTANTID REFERENCE VAE ENCODE"):
                layout_latent = _outputs(_call_node(("VAEEncode",), pixels=layout_image, vae=vae))[0]
        elif zero_pass_mode:
            width = int(sampled_in.get("sdxl_width", process.get("sdxl_width", 1024)))
            height = int(sampled_in.get("sdxl_height", process.get("sdxl_height", 1024)))
            with cmk_timed("15 INSTANTID ZERO-PASS NEUTRAL KPS", f"{width}x{height}"):
                layout_image = _outputs(_call_node(
                    ("EmptyImage",),
                    width=width,
                    height=height,
                    batch_size=1,
                    color=0,
                ))[0]
        else:
            # Decode the sampler's clean x0 estimate for face/keypoint detection.
            # The separate noisy layout latent remains untouched for continuation.
            with cmk_timed("15 INSTANTID KEYPOINTS X0 VAE DECODE"):
                layout_image = _outputs(_call_node(("VAEDecode",), samples=keypoints_latent, vae=vae))[0]
        with cmk_timed("15 INSTANTID FACEANALYSIS LOAD", str(provider)):
            faceanalysis = _outputs(_call_node(("CMKInternalInstantIDFaceAnalysis",), provider=provider))[0]
        if content_guard_enabled:
            with cmk_timed(
                "15R ADVANCED CONTENTGUARD SOURCE",
                "both donor identities" if source_b_enabled else "donor identity A",
            ):
                _guard_source_images(faceanalysis, source_face_a)
                if source_b_enabled:
                    _guard_source_images(faceanalysis, source_face_b)
        with cmk_timed("15 INSTANTID MODEL LOAD", str(instantid_model)):
            instantid = _outputs(_call_node(("CMKInternalInstantIDModelLoader",), instantid_file=instantid_model))[0]
        with cmk_timed("15 INSTANTID CONTROLNET LOAD", str(controlnet_model)):
            control_net = _outputs(_call_node(("ControlNetLoader",), control_net_name=controlnet_model))[0]
        target_detail = f"A={target_face_a} | B={target_face_b}" if source_b_enabled else f"A={target_face_a} | B=disabled"
        with cmk_timed("15R ADVANCED TARGET REGIONS", target_detail):
            if source_b_enabled:
                hint_a, mask_a, hint_b, mask_b, region_overlap = _regional_hints(
                    faceanalysis, layout_image, target_face_a, target_face_b,
                )
            else:
                hint_a, mask_a = _regional_hint(faceanalysis, layout_image, target_face_a)
                hint_b = mask_b = None
                region_overlap = 0.0
        with cmk_timed(
            "15R ADVANCED APPLY A",
            f"target={target_face_a} | identity={float(identity_strength_a):.2f} | pose={float(pose_strength_a):.2f}",
        ):
            patched_a, embeds_a = InstantIDAttentionPatch().patch_attention(
                instantid=instantid,
                insightface=faceanalysis,
                image=source_face_a,
                mask=mask_a,
                model=base_model,
                weight=float(identity_strength_a),
                start_at=float(instantid_start),
                end_at=float(instantid_end),
                noise=float(noise),
            )
        branches = [(hint_a, mask_a, embeds_a, pose_strength_a)]
        patched_model = patched_a
        if source_b_enabled:
            with cmk_timed(
                "15R ADVANCED APPLY B",
                f"target={target_face_b} | identity={float(identity_strength_b):.2f} | pose={float(pose_strength_b):.2f}",
            ):
                patched_model, embeds_b = InstantIDAttentionPatch().patch_attention(
                    instantid=instantid,
                    insightface=faceanalysis,
                    image=source_face_b,
                    mask=mask_b,
                    model=patched_a,
                    weight=float(identity_strength_b),
                    start_at=float(instantid_start),
                    end_at=float(instantid_end),
                    noise=float(noise),
                )
            branches.append((hint_b, mask_b, embeds_b, pose_strength_b))
        control_detail = "isolated A/B embedding contexts" if source_b_enabled else "Source A embedding context"
        with cmk_timed("15R ADVANCED CONTROLNET", control_detail):
            positive_out, negative_out = _apply_regional_controlnet(
                control_net,
                branches, positive, negative, instantid_start, instantid_end,
            )

        final_steps = int(sampled_in.get("steps_1st_pass", sampled_in.get("steps", 20)))
        start_at_step = (
            max(0, min(
                max(0, final_steps - 1),
                int(reference_start_at_step),
            ))
            if reference_latent_mode
            else (0 if zero_pass_mode else int(sampled_in.get("instantid_end_at_step", 2)))
        )
        add_noise = "enable" if (reference_latent_mode or zero_pass_mode or inpaint_bridge_mode) else "disable"
        workflow_mode = (
            "ZERO-PASS · CONTROLNET REFERENCE"
            if reference_latent_mode
            else (
                "ZERO-PASS"
                if zero_pass_mode
                else (
                    f"HYBRID · INPAINT · HANDOFF {start_at_step}"
                    if inpaint_bridge_mode
                    else f"HYBRID · TEXT2IMAGE · HANDOFF {start_at_step}"
                )
            )
        )
        shared_denoise = float(sampled_in.get("denoise", 1.0))
        sampler_name = str(sampled_in.get("sampler", "euler_ancestral"))
        scheduler = str(sampled_in.get("scheduler", "karras"))
        with cmk_timed(
            "15 INSTANTID KSAMPLER",
            f"steps {start_at_step}-{final_steps}/{final_steps} | {sampler_name} | {scheduler}",
        ):
            with ignore_torchsde_boundary_rounding():
                sampling_args = {
                    "model": patched_model,
                    "add_noise": add_noise,
                    "noise_seed": int(sampled_in.get("seed", process.get("seed", 0))),
                    "steps": final_steps,
                    "cfg": float(cfg),
                    "sampler_name": str(sampler_name),
                    "scheduler": str(scheduler),
                    "positive": positive_out,
                    "negative": negative_out,
                    "latent_image": layout_latent,
                    "start_at_step": start_at_step,
                    "end_at_step": final_steps,
                }
                if content_guard_enabled:
                    result = _sample_with_midpoint_guard(
                        **sampling_args,
                        vae=vae,
                    )
                else:
                    result = _sample_without_content_guard(**sampling_args)
        if content_guard_enabled:
            target_selections = (target_face_a, target_face_b) if source_b_enabled else (target_face_a,)
            with cmk_timed(
                "15R ADVANCED CONTENTGUARD TARGET",
                "both generated identity targets" if source_b_enabled else "generated identity target A",
            ):
                _guard_generated_latent_multi(
                    result, vae, faceanalysis, target_selections,
                )
        with cmk_timed(
            "15 INSTANTID VISUAL VAE DECODE",
            "guard-approved result" if content_guard_enabled else "baseline guard bypass",
        ):
            identity_decoded = _outputs(_call_node(("VAEDecode",), samples=result, vae=vae))
            identity_image = identity_decoded[0]
        sampled_out = dict(sampled_in)
        sampled_out.update({
            "model_patched": patched_model,
            "conditioning_pos": positive_out,
            "conditioning_neg": negative_out,
            "samples": result,
            "latent": result,
            "latent_image": result,
            "latent_1st_pass": result,
            "image_1st_pass": identity_image,
            "steps": final_steps,
            "steps_1st_pass": final_steps,
            "denoise": shared_denoise,
            "identity_layout_latent": layout_latent,
            "identity_applied": True,
            "instantid_reference_latent_mode": reference_latent_mode,
            "instantid_zero_pass": zero_pass_mode,
            "instantid_inpaint_bridge": inpaint_bridge_mode,
            "instantid_workflow": workflow_mode,
            "identity_count": 2 if source_b_enabled else 1,
            "identity_target_a": str(target_face_a),
            "identity_target_b": str(target_face_b) if source_b_enabled else None,
            "identity_region_overlap": float(region_overlap),
            "content_guard_version": INSTANTID_GUARD_VERSION,
            "content_guard_enabled": content_guard_enabled,
        })
        lines = [
            f"Workflow          : {workflow_mode}",
            f"Model source      : {model_source}",
            f"Conditioning      : {conditioning_source}",
            f"Identity A        : {target_face_a} | strength {float(identity_strength_a):.2f} | pose {float(pose_strength_a):.2f}",
            (
                f"Identity B        : {target_face_b} | strength {float(identity_strength_b):.2f} | pose {float(pose_strength_b):.2f}"
                if source_b_enabled
                else "Identity B        : disabled"
            ),
            f"Region overlap    : {region_overlap:.1%}" if source_b_enabled else "Region overlap    : n/a",
            f"InstantID range   : {float(instantid_start):.2f} - {float(instantid_end):.2f}",
            f"Sampling schedule : {start_at_step}-{final_steps}/{final_steps} | add_noise={add_noise}",
            f"Sampling mode     : {('reference latent + noise' if reference_latent_mode else ('zero-pass + noise' if zero_pass_mode else ('clean Inpaint bridge + noise' if inpaint_bridge_mode else 'split continuation')))}",
            f"KPS source        : {('reference image' if reference_latent_mode else ('neutral black' if zero_pass_mode else ('clean Inpaint bridge' if inpaint_bridge_mode else '1st-pass x0')))}",
            (
                f"Reference start   : step {start_at_step}/{final_steps}"
                if reference_latent_mode
                else (
                    f"Inpaint bridge   : x0 decode/encode | step {start_at_step}/{final_steps}"
                    if inpaint_bridge_mode
                    else "Reference start   : n/a"
                )
            ),
            f"InstantID CFG     : {float(cfg):.2f}",
            f"Identity emb noise: {float(noise):.2f}",
            "Target mapping     : preliminary x0 face regions",
            *(
                [
                    "ContentGuard      : PASS · SOURCE + GENERATED TARGET",
                    "Early Guard       : PASS · 50% x0 explicit-content gate",
                ]
                if content_guard_enabled
                else ["ContentGuard      : BYPASS · BASELINE"]
            ),
            f"Method            : {sampler_name} | {scheduler}",
            f"InstantID model   : {instantid_model}",
            f"ControlNet model  : {controlnet_model}",
        ]
        if str(conditioning_mode) == "InstantID reference":
            for label, result in (("POS", translation_pos), ("NEG", translation_neg)):
                if result.status not in {"empty", "disabled", "not_configured"}:
                    lines.append(result.log_line(label))
        log_in = LOG_B if source_b_enabled and LOG_B is not None else LOG
        log_out = cmk_add_block(log_in, "InstantID Regional Advanced SDXL", 45, lines, True)
        diagnostic = make_diagnostic_payload(
            title="InstantID Regional Advanced SDXL",
            node="CMK InstantID Regional Advanced SDXL -Pipe-",
            previews=[],
            summary=workflow_mode,
            details="\n".join(lines),
            mode="InstantID",
            metadata={
                "identity_applied": True,
                "identity_count": 2 if source_b_enabled else 1,
                "steps": final_steps,
                "identity_embedding_noise": float(noise),
                "target_a": str(target_face_a),
                "target_b": str(target_face_b) if source_b_enabled else None,
                "region_overlap": float(region_overlap),
                "workflow": workflow_mode,
            },
        )
        visual = register_provider(
            visual,
            module_instance_id=unique_id or "instantid-sampler",
            module_type="CMKInstantIDRegionalAdvancedSDXLPipe",
            module_label="Identity Regional Advanced",
            sequence=15,
            channels={"result": identity_image},
            status="completed",
            live_node_id=unique_id,
            branch="sdxl",
            stage_key="sdxl.identity.regional.advanced",
        )
        if bool(process.get("unload_models_after_use", True)):
            from ..loaders.checkpoint_vae_loader import unload_all_phase_models

            unload_all_phase_models("15R InstantID Regional Advanced complete")
        return model_pipe, process, sampled_out, log_out, identity_image, visual, diagnostic


_INSTANTID_BOUNDARY_CACHE = {}
_INSTANTID_BOUNDARY_EXECUTION_TOKEN = None


def clear_instantid_boundary_cache():
    count = len(_INSTANTID_BOUNDARY_CACHE)
    _INSTANTID_BOUNDARY_CACHE.clear()
    return count


def _reset_instantid_boundary_cache(dynprompt):
    global _INSTANTID_BOUNDARY_EXECUTION_TOKEN
    token = id(dynprompt) if dynprompt is not None else None
    if token != _INSTANTID_BOUNDARY_EXECUTION_TOKEN:
        _INSTANTID_BOUNDARY_CACHE.clear()
        _INSTANTID_BOUNDARY_EXECUTION_TOKEN = token


