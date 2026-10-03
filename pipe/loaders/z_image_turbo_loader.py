from __future__ import annotations

import gc

import folder_paths
from comfy_execution.graph_utils import ExecutionBlocker

from ..cmk_log_pipe import cmk_add_block
from ..cmk_sampler_prepare import _call_node_kwargs, _unwrap_node_output


DEFAULT_UNET = "z_image_turbo_bf16.safetensors"
DEFAULT_TEXT_ENCODER = "qwen_3_4b.safetensors"
DEFAULT_VAE = "ae.safetensors"


# The wrapper must not reconstruct ~20 GB of ZIT resources merely because its
# family PROCESS (prompt, seed, ControlNet settings) changed. Native ComfyUI
# loader nodes normally avoid this because their cache key contains filenames
# only; this wrapper has PROCESS as a contract input, so it needs the same
# resource-level cache explicitly.
_ZIT_RESOURCE_CACHE = {}


def _resource_key_from_pipe(model_pipe):
    return (
        str(model_pipe.get("diffusion_model_name", DEFAULT_UNET)),
        str(model_pipe.get("text_encoder_name", DEFAULT_TEXT_ENCODER)),
        str(model_pipe.get("vae_name", DEFAULT_VAE)),
        str(model_pipe.get("weight_dtype", "default")),
        str(model_pipe.get("clip_device", "default")),
    )


def _release_stale_resource_cache(target_key):
    """Release the previous ZIT resource set before loading another one."""
    stale_keys = [key for key in _ZIT_RESOURCE_CACHE if key != target_key]
    if not stale_keys:
        return 0
    for key in stale_keys:
        _ZIT_RESOURCE_CACHE.pop(key, None)
    try:
        import comfy.model_management

        comfy.model_management.unload_all_models()
    except Exception:
        pass
    gc.collect()
    try:
        import comfy.model_management

        comfy.model_management.cleanup_models_gc()
        comfy.model_management.soft_empty_cache(force=True)
    except Exception:
        pass
    print(
        f"[CMK Model Lifecycle] ZIT resource switch: DROPPED "
        f"{len(stale_keys)} STALE RESOURCE CACHE ENTRIES",
        flush=True,
    )
    return len(stale_keys)


def invalidate_zit_resource_cache(reason="ZIT sampling interrupted"):
    """Discard a ZIT resource set after an incomplete sampler lifecycle."""
    entry_count = len(_ZIT_RESOURCE_CACHE)
    _ZIT_RESOURCE_CACHE.clear()
    unload_status = "UNLOADED"
    try:
        import comfy.model_management

        comfy.model_management.unload_all_models()
    except Exception as exc:
        unload_status = f"UNLOAD FAILED: {exc}"
    gc.collect()
    try:
        import comfy.model_management

        comfy.model_management.cleanup_models_gc()
        comfy.model_management.soft_empty_cache(force=True)
    except Exception:
        pass
    print(
        f"[CMK Model Lifecycle] {reason}: INVALIDATED "
        f"{entry_count} ZIT RESOURCE CACHE ENTRIES · {unload_status}",
        flush=True,
    )
    return entry_count


def _load_text_encoder(text_encoder, clip_device):
    return _unwrap_node_output(
        _call_node_kwargs(
            ("CLIPLoader",),
            clip_name=text_encoder,
            type="lumina2",
            device=clip_device,
        )
    )


def ensure_zit_text_encoder(model_pipe):
    """Materialise a previously evicted ZIT text encoder on demand."""
    if not isinstance(model_pipe, dict):
        raise TypeError("CMK Z-Image Turbo Loader requires a CMK model pipe")
    clip = model_pipe.get("clip")
    if clip is not None:
        return clip, "AVAILABLE"

    resource_key = _resource_key_from_pipe(model_pipe)
    resources = _ZIT_RESOURCE_CACHE.get(resource_key)
    if resources is not None and resources[1] is not None:
        clip = resources[1]
        status = "CACHE HIT · REUSED"
    else:
        clip = _load_text_encoder(resource_key[1], resource_key[4])
        model = model_pipe.get("model")
        loaded_vae = model_pipe.get("vae")
        if resources is not None:
            model = resources[0]
            loaded_vae = resources[2]
        _ZIT_RESOURCE_CACHE.clear()
        _ZIT_RESOURCE_CACHE[resource_key] = (model, clip, loaded_vae)
        status = "REMATERIALIZED"
    model_pipe["clip"] = clip
    model_pipe["text_encoder_status"] = status
    print(f"[CMK Model Lifecycle] ZIT text encoder: {status}", flush=True)
    return clip, status


def evict_zit_text_encoder(model_pipe):
    """Drop every CMK strong reference after prompt conditioning is complete."""
    if not isinstance(model_pipe, dict):
        return "NOT AVAILABLE"
    resource_key = _resource_key_from_pipe(model_pipe)
    resources = _ZIT_RESOURCE_CACHE.get(resource_key)
    cached_clip = None
    if resources is not None:
        model, cached_clip, loaded_vae = resources
        _ZIT_RESOURCE_CACHE[resource_key] = (model, None, loaded_vae)
    active_clip = model_pipe.get("clip")
    model_pipe["clip"] = None
    model_pipe["text_encoder_status"] = "EVICTED AFTER ENCODE"

    # ComfyUI's MPS-wide unload path does not include CPU-only text encoders.
    # Once the CMK references above are gone, weak model-manager entries can be
    # collected and the shared Apple-Silicon memory can actually be reclaimed.
    active_clip = None
    cached_clip = None
    resources = None
    gc.collect()
    try:
        import comfy.model_management

        comfy.model_management.cleanup_models_gc()
        comfy.model_management.soft_empty_cache(force=True)
    except Exception:
        # Unit-test stubs and older supported ComfyUI builds need no special
        # cleanup; dropping the strong references remains the essential step.
        pass
    print(
        "[CMK Model Lifecycle] ZIT text encoder: EVICTED AFTER ENCODE",
        flush=True,
    )
    return "EVICTED AFTER ENCODE"


def _choices(folder_name: str, preferred: str) -> list[str]:
    values = list(folder_paths.get_filename_list(folder_name))
    if preferred in values:
        values.remove(preferred)
        values.insert(0, preferred)
    return values or [preferred]


class CMKZImageTurboLoaderPipe:
    """Load the complete Z-Image Turbo model family through ComfyUI Core."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PROCESS": ("CMK_PROCESS_Z_IMAGE",),
                "diffusion_model": (
                    _choices("diffusion_models", DEFAULT_UNET),
                    {"default": DEFAULT_UNET, "label": "DIFFUSION MODEL"},
                ),
                "text_encoder": (
                    _choices("text_encoders", DEFAULT_TEXT_ENCODER),
                    {"default": DEFAULT_TEXT_ENCODER, "label": "TEXT ENCODER"},
                ),
                "vae": (
                    _choices("vae", DEFAULT_VAE),
                    {"default": DEFAULT_VAE, "label": "VAE"},
                ),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    {"default": "default", "advanced": True},
                ),
                "clip_device": (
                    ["default", "cpu"],
                    {"default": "default", "advanced": True},
                ),
            },
            "optional": {
                "LOG": ("CMK_LOG_PIPE",),
            },
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_LOG_PIPE")
    RETURN_NAMES = ("MODEL", "LOG")
    FUNCTION = "load"
    CATEGORY = "CMK/Developer/Pipe/Load"

    def load(
        self,
        PROCESS,
        diffusion_model,
        text_encoder,
        vae,
        weight_dtype="default",
        clip_device="default",
        LOG=None,
    ):
        if PROCESS is None or (
            isinstance(PROCESS, dict) and not PROCESS.get("family_active", True)
        ):
            blocked = ExecutionBlocker(None)
            return (blocked, blocked)
        if not isinstance(PROCESS, dict):
            raise TypeError(
                "CMK Z-Image Turbo Loader: PROCESS must be a CMK process pipe"
            )
        family = str(PROCESS.get("model_family", "z_image_turbo")).strip().lower()
        if family != "z_image_turbo":
            raise ValueError(
                "CMK Z-Image Turbo Loader requires PROCESS ZIT"
            )
        resource_key = (
            str(diffusion_model),
            str(text_encoder),
            str(vae),
            str(weight_dtype),
            str(clip_device),
        )
        if PROCESS.get("unload_models_after_use") is True:
            _release_stale_resource_cache(resource_key)
        resources = _ZIT_RESOURCE_CACHE.get(resource_key)
        if resources is None:
            model = _unwrap_node_output(
                _call_node_kwargs(
                    ("UNETLoader",),
                    unet_name=diffusion_model,
                    weight_dtype=weight_dtype,
                )
            )
            clip = _load_text_encoder(text_encoder, clip_device)
            loaded_vae = _unwrap_node_output(
                _call_node_kwargs(("VAELoader",), vae_name=vae)
            )
            resources = (model, clip, loaded_vae)
            _ZIT_RESOURCE_CACHE.clear()
            _ZIT_RESOURCE_CACHE[resource_key] = resources
            resource_cache_status = "MISS · LOADED"
        else:
            model, clip, loaded_vae = resources
            if clip is None:
                clip = _load_text_encoder(text_encoder, clip_device)
                resources = (model, clip, loaded_vae)
                _ZIT_RESOURCE_CACHE[resource_key] = resources
                resource_cache_status = "HIT · TEXT ENCODER REMATERIALIZED"
            else:
                resource_cache_status = "HIT · REUSED"
        if model is None or clip is None or loaded_vae is None:
            raise RuntimeError(
                "CMK Z-Image Turbo Loader: ComfyUI Core did not return model, "
                "text encoder and VAE."
            )

        model_pipe = {
            "model": model,
            "clip": clip,
            "vae": loaded_vae,
            "model_family": "z_image_turbo",
            "diffusion_model_name": str(diffusion_model),
            "text_encoder_name": str(text_encoder),
            "vae_name": str(vae),
            "weight_dtype": str(weight_dtype),
            "clip_device": str(clip_device),
            "model_pipe_source": "CMK Z-Image Turbo Loader -Pipe-",
            "resource_cache_status": resource_cache_status,
        }
        log = cmk_add_block(
            LOG,
            "Z-Image Turbo Loader",
            15,
            [
                "MODEL FAMILY    : Z-IMAGE TURBO",
                f"DIFFUSION MODEL: {diffusion_model}",
                f"TEXT ENCODER    : {text_encoder}",
                f"VAE             : {vae}",
                f"WEIGHT DTYPE    : {weight_dtype}",
                f"CLIP DEVICE     : {clip_device}",
                f"RESOURCE CACHE  : {resource_cache_status}",
            ],
            True,
        )
        return (model_pipe, log)
