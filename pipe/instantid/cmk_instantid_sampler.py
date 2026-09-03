from __future__ import annotations

import inspect
import os

import numpy as np

from ..cmk_log_pipe import cmk_add_block
from ..cmk_module_cache_contract import stamp_artifact
from ...utils.cmk_translation import translate_prompt
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...utils.cmk_timing import cmk_timed
from ...utils.cmk_sampling_warnings import ignore_torchsde_boundary_rounding
from ...cmk_common import SAMPLERS, SCHEDULERS
from ...engine.content_guard import ContentGuardBlocked, GUARD_VERSION, get_content_guard
from ...engine.instantid_face_rebuild import select_target_face
from ...utils.tensor_utils import tensor_to_uint8_rgb


INSTANTID_PROVIDERS = ["CoreML", "CPU", "CUDA", "ROCM"]
INSTANTID_GUARD_VERSION = f"{GUARD_VERSION}:instantid-midpoint-final-target-v2"

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


class CMKInstantIDSamplerSDXLPipe:
    """Apply InstantID to an existing layout latent and return a sampled CMK pipe."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE",),
                "PROCESS": ("CMK_PROCESS_SDXL",),
                "SAMPLED": ("CMK_SAMPLED_PIPE",),
                "LOG": ("CMK_LOG_PIPE",),
                "source_face": ("IMAGE",),
                "identity_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "pose_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "target_face": (("Largest", "Leftmost", "Rightmost", "Topmost", "Bottommost", "Center"), {
                    "default": "Largest",
                    "tooltip": "Selects which detected person receives the InstantID identity.",
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
            "hidden": {"dynprompt": "DYNPROMPT"},
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "CMK_PROCESS_SDXL",
        "CMK_SAMPLED_PIPE",
        "CMK_LOG_PIPE",
        "IMAGE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("MODEL", "PROCESS", "SAMPLED", "LOG", "IDENTITY IMAGE", "diagnostic")
    FUNCTION = "sample"
    CATEGORY = "CMK/Toolbox/Face"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return INSTANTID_GUARD_VERSION

    def sample(
        self,
        MODEL,
        PROCESS,
        SAMPLED,
        LOG,
        source_face,
        identity_strength,
        pose_strength,
        target_face,
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
        dynprompt=None,
    ):
        _reset_instantid_boundary_cache(dynprompt)
        model_pipe = _require_dict(MODEL, "MODEL")
        process = _require_dict(PROCESS, "PROCESS")
        sampled_in = _require_dict(SAMPLED, "SAMPLED")
        if not bool(process.get("instantid_enabled", False)):
            lines = ["InstantID          : disabled in module 01", "Sampling           : bypassed"]
            log_out = cmk_add_block(LOG, "InstantID Sampler SDXL", 45, lines, True)
            diagnostic = make_diagnostic_payload(
                title="InstantID Sampler SDXL",
                node="CMK InstantID Sampler SDXL -Pipe-",
                previews=[],
                summary="InstantID disabled; pipeline passed through",
                details="\n".join(lines),
                mode="Bypass",
                metadata={"identity_applied": False},
            )
            return model_pipe, process, sampled_in, log_out, None, diagnostic
        if source_face is None:
            raise ValueError("CMK InstantID Sampler SDXL -Pipe-: source_face is missing")
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
                raise ValueError(
                    "CMK InstantID Sampler SDXL -Pipe-: reference conditioning requires MODEL['clip']"
                )
            prompt_pos = str(process.get("prompt_pos", sampled_in.get("prompt_pos", "")) or "")
            prompt_neg = str(process.get("prompt_neg", sampled_in.get("prompt_neg", "")) or "")
            translation_pos = translate_prompt(prompt_pos)
            translation_neg = translate_prompt(prompt_neg)
            with cmk_timed("15 INSTANTID REFERENCE CONDITIONING", "native CLIP Text Encode"):
                reference_positive = _outputs(_call_node(("CLIPTextEncode",), clip=clean_clip, text=translation_pos.text))[0]
                reference_negative = _outputs(_call_node(("CLIPTextEncode",), clip=clean_clip, text=translation_neg.text))[0]
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
        with cmk_timed("15 INSTANTID CONTENTGUARD SOURCE", "all detected donor faces"):
            _guard_source_images(faceanalysis, source_face)
        with cmk_timed("15 INSTANTID MODEL LOAD", str(instantid_model)):
            instantid = _outputs(_call_node(("CMKInternalInstantIDModelLoader",), instantid_file=instantid_model))[0]
        with cmk_timed("15 INSTANTID CONTROLNET LOAD", str(controlnet_model)):
            control_net = _outputs(_call_node(("ControlNetLoader",), control_net_name=controlnet_model))[0]
        with cmk_timed(
            "15 INSTANTID APPLY",
            f"identity={float(identity_strength):.2f} | pose={float(pose_strength):.2f}",
        ):
            applied = _outputs(_call_node(
                ("CMKInternalApplyInstantIDAdvanced",),
                instantid=instantid,
                insightface=faceanalysis,
                control_net=control_net,
                image=source_face,
                image_kps=layout_image,
                model=base_model,
                positive=positive,
                negative=negative,
                ip_weight=float(identity_strength),
                cn_strength=float(pose_strength),
                weight=float(identity_strength),
                start_at=float(instantid_start),
                end_at=float(instantid_end),
                noise=float(noise),
                combine_embeds="average",
                target_selection=str(target_face),
                image_kps_is_hint=zero_pass_mode,
            ))
        if len(applied) < 3:
            raise TypeError("CMK InstantID Sampler SDXL -Pipe-: Apply InstantID returned invalid outputs")
        patched_model, positive_out, negative_out = applied[:3]

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
                result = _sample_with_midpoint_guard(
                    model=patched_model,
                    add_noise=add_noise,
                    noise_seed=int(sampled_in.get("seed", process.get("seed", 0))),
                    steps=final_steps,
                    cfg=float(cfg),
                    sampler_name=str(sampler_name),
                    scheduler=str(scheduler),
                    positive=positive_out,
                    negative=negative_out,
                    latent_image=layout_latent,
                    start_at_step=start_at_step,
                    end_at_step=final_steps,
                    vae=vae,
                )
        with cmk_timed("15 INSTANTID CONTENTGUARD TARGET", "internal final decode"):
            _guard_generated_latent(result, vae, faceanalysis, target_face)
        with cmk_timed("15 INSTANTID VISUAL VAE DECODE", "guard-approved result"):
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
            "content_guard_version": INSTANTID_GUARD_VERSION,
        })
        lines = [
            f"Workflow          : {workflow_mode}",
            f"Model source      : {model_source}",
            f"Conditioning      : {conditioning_source}",
            f"Identity strength : {float(identity_strength):.2f}",
            f"Pose strength     : {float(pose_strength):.2f}",
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
            f"Target face       : {target_face}",
            "ContentGuard      : PASS · SOURCE + GENERATED TARGET",
            "Early Guard       : PASS · 50% x0 explicit-content gate",
            f"Method            : {sampler_name} | {scheduler}",
            f"InstantID model   : {instantid_model}",
            f"ControlNet model  : {controlnet_model}",
        ]
        if str(conditioning_mode) == "InstantID reference":
            for label, result in (("POS", translation_pos), ("NEG", translation_neg)):
                if result.status not in {"empty", "disabled", "not_configured"}:
                    lines.append(result.log_line(label))
        log_out = cmk_add_block(LOG, "InstantID Sampler SDXL", 45, lines, True)
        diagnostic = make_diagnostic_payload(
            title="InstantID Sampler SDXL",
            node="CMK InstantID Sampler SDXL -Pipe-",
            previews=[],
            summary=workflow_mode,
            details="\n".join(lines),
            mode="InstantID",
            metadata={
                "identity_applied": True,
                "steps": final_steps,
                "identity_embedding_noise": float(noise),
                "workflow": workflow_mode,
            },
        )
        return model_pipe, process, sampled_out, log_out, identity_image, diagnostic


_INSTANTID_BOUNDARY_CACHE = {}
_INSTANTID_BOUNDARY_EXECUTION_TOKEN = None


def _reset_instantid_boundary_cache(dynprompt):
    global _INSTANTID_BOUNDARY_EXECUTION_TOKEN
    token = id(dynprompt) if dynprompt is not None else None
    if token != _INSTANTID_BOUNDARY_EXECUTION_TOKEN:
        _INSTANTID_BOUNDARY_CACHE.clear()
        _INSTANTID_BOUNDARY_EXECUTION_TOKEN = token


class CMKInstantIDBoundary:
    """Materialize every InstantID result before the subgraph fans out.

    Global-subgraph outputs are resolved independently by ComfyUI.  Without a
    common non-lazy consumer, downstream MODEL/PROCESS/SAMPLED/LOG requests can
    reopen the expensive InstantID sampler after its preview has completed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS": ("CMK_PROCESS_SDXL", {"lazy": True}),
                "SAMPLED": ("CMK_SAMPLED_PIPE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "diagnostic": ("CMK_DIAGNOSTIC", {"lazy": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "dynprompt": "DYNPROMPT",
            }
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "CMK_PROCESS_SDXL",
        "CMK_SAMPLED_PIPE",
        "CMK_LOG_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("MODEL", "PROCESS", "SAMPLED", "LOG", "diagnostic")
    FUNCTION = "materialize"
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    def check_lazy_status(self, prompt=None, unique_id=None, dynprompt=None, **inputs):
        _reset_instantid_boundary_cache(dynprompt)
        key = self._cache_key(prompt, unique_id)
        if key in _INSTANTID_BOUNDARY_CACHE:
            return []
        return [
            name for name in ("MODEL", "PROCESS", "SAMPLED", "LOG", "diagnostic")
            if inputs.get(name) is None
        ]

    @staticmethod
    def _cache_key(prompt, unique_id):
        try:
            from ..cmk_family_result import _sampled_boundary_key
            return _sampled_boundary_key(prompt, unique_id)
        except Exception:
            return None

    def materialize(
        self,
        MODEL=None,
        PROCESS=None,
        SAMPLED=None,
        LOG=None,
        diagnostic=None,
        prompt=None,
        unique_id=None,
        dynprompt=None,
    ):
        _reset_instantid_boundary_cache(dynprompt)
        key = self._cache_key(prompt, unique_id)
        if key in _INSTANTID_BOUNDARY_CACHE and any(
            value is None for value in (MODEL, PROCESS, SAMPLED, LOG, diagnostic)
        ):
            print(f"[CMK InstantID Boundary] HIT {key[:12]}")
            return _INSTANTID_BOUNDARY_CACHE[key]
        values = (MODEL, PROCESS, SAMPLED, LOG, diagnostic)
        if any(value is None for value in values):
            raise ValueError("CMK InstantID Boundary is missing a materialized input")
        result_process = (
            stamp_artifact(PROCESS, _INSTANTID_STAGE_KEY, key)
            if key and isinstance(PROCESS, dict)
            else PROCESS
        )
        values = (MODEL, result_process, SAMPLED, LOG, diagnostic)
        if key:
            _INSTANTID_BOUNDARY_CACHE[key] = values
            print(f"[CMK InstantID Boundary] STORE {key[:12]}")
        return values
