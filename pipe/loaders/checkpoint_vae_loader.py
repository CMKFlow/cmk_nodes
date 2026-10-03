from __future__ import annotations

import gc

import folder_paths
from comfy_execution.graph_utils import ExecutionBlocker

from ...loader.checkpoint_vae_loader import load_checkpoint_vae_resources


DEFAULT_CHECKPOINT = "juggernautXL_ragnarok.safetensors"
POSTPROCESS_MODEL_NONE = "None"
_PROCESS_NOT_CONNECTED = object()


# PROCESS is an optional lazy-family gate input. It must not become the cache
# key for several gigabytes of immutable checkpoint resources. Keep the base
# and refiner configurations side by side; a full SDXL flow needs both.
_SDXL_RESOURCE_CACHE = {}
_SDXL_RESOURCE_CACHE_LIMIT = 4


def _resource_key(ckpt_name, vae_name, checkpoint_vae):
    return (str(ckpt_name), str(vae_name), bool(checkpoint_vae))


def _resource_key_from_pipe(model_pipe):
    return _resource_key(
        model_pipe.get("ckpt_name", DEFAULT_CHECKPOINT),
        model_pipe.get("vae_selection_name", model_pipe.get("vae_name", "")),
        model_pipe.get("checkpoint_vae", True),
    )


def _release_stale_resource_cache(target_key, label):
    """Drop old checkpoint objects before materialising a different model.

    ComfyUI can unload a patcher from the active device while Python still
    retains the complete checkpoint through this module cache. On unified
    memory that strong reference matters. The global lifecycle switch opts
    into releasing those stale cache entries before the replacement model is
    loaded, avoiding an old+new peak during checkpoint changes.
    """
    stale_keys = [key for key in _SDXL_RESOURCE_CACHE if key != target_key]
    if not stale_keys:
        return 0
    for key in stale_keys:
        _SDXL_RESOURCE_CACHE.pop(key, None)
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
        f"[CMK Model Lifecycle] {label}: DROPPED {len(stale_keys)} "
        "STALE RESOURCE CACHE ENTRIES",
        flush=True,
    )
    return len(stale_keys)


def _load_checkpoint_clip(ckpt_name):
    import comfy.sd

    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
    result = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=False,
        output_clip=True,
        output_clipvision=False,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        output_model=False,
    )
    clip = result[1]
    if clip is None:
        raise RuntimeError(f"CMK SDXL text encoder could not be loaded from {ckpt_name}")
    return clip


def make_postprocess_model_spec(ckpt_name, vae_name, checkpoint_vae):
    """Return a cheap MODEL contract without materialising checkpoint tensors."""
    disabled = str(ckpt_name or "").strip().lower() in {"", "none"}
    return {
        "model_role": "postprocess",
        "model_family": "sdxl",
        "model_architecture": "sdxl",
        "ckpt_name": POSTPROCESS_MODEL_NONE if disabled else str(ckpt_name),
        "vae_name": str(vae_name),
        "vae_selection_name": str(vae_name),
        "checkpoint_vae": bool(checkpoint_vae),
        "postprocess_model_disabled": disabled,
        "model_pipe_source": "CMK Flow · PostProcess Boundary",
        "materialized": False,
    }


def is_postprocess_model_spec(value):
    return (
        isinstance(value, dict)
        and value.get("model_role") == "postprocess"
        and not all(value.get(name) is not None for name in ("model", "clip", "vae"))
    )


def is_postprocess_model_disabled(value):
    return (
        isinstance(value, dict)
        and value.get("model_role") == "postprocess"
        and bool(value.get("postprocess_model_disabled", False))
    )


def _load_cached_resources(ckpt_name, vae_name, checkpoint_vae):
    resource_key = _resource_key(ckpt_name, vae_name, checkpoint_vae)
    resources = _SDXL_RESOURCE_CACHE.get(resource_key)
    if resources is None:
        model, clip, vae, metadata = load_checkpoint_vae_resources(
            ckpt_name, vae_name, checkpoint_vae
        )
        resources = (model, clip, vae, dict(metadata))
        _SDXL_RESOURCE_CACHE[resource_key] = resources
        while len(_SDXL_RESOURCE_CACHE) > _SDXL_RESOURCE_CACHE_LIMIT:
            oldest_key = next(iter(_SDXL_RESOURCE_CACHE))
            if oldest_key == resource_key:
                break
            _SDXL_RESOURCE_CACHE.pop(oldest_key, None)
        status = "MISS · LOADED"
    else:
        model, clip, vae, metadata = resources
        if clip is None:
            clip = _load_checkpoint_clip(ckpt_name)
            resources = (model, clip, vae, metadata)
            _SDXL_RESOURCE_CACHE[resource_key] = resources
            status = "HIT · TEXT ENCODER REMATERIALIZED"
        else:
            status = "HIT · REUSED"
        metadata = dict(metadata)
    return resources, status


def ensure_sdxl_text_encoder(model_pipe):
    """Materialise an evicted SDXL CLIP only when a later module needs it."""
    if not isinstance(model_pipe, dict):
        raise TypeError("CMK SDXL text encoder requires a CMK model pipe")
    clip = model_pipe.get("clip")
    if clip is not None:
        return clip, "AVAILABLE"

    resource_key = _resource_key_from_pipe(model_pipe)
    resources = _SDXL_RESOURCE_CACHE.get(resource_key)
    if resources is not None and resources[1] is not None:
        clip = resources[1]
        status = "CACHE HIT · REUSED"
    else:
        clip = _load_checkpoint_clip(resource_key[0])
        model = model_pipe.get("model")
        vae = model_pipe.get("vae")
        metadata = {
            key: model_pipe.get(key)
            for key in (
                "ckpt_name", "vae_name", "checkpoint_vae", "vae_source",
                "model_pipe_source",
            )
            if key in model_pipe
        }
        if resources is not None:
            model, _, vae, metadata = resources
        _SDXL_RESOURCE_CACHE[resource_key] = (model, clip, vae, metadata)
        status = "REMATERIALIZED"
    model_pipe["clip"] = clip
    model_pipe["text_encoder_status"] = status
    print(f"[CMK Model Lifecycle] SDXL text encoder: {status}", flush=True)
    return clip, status


def evict_sdxl_text_encoder(model_pipe, *working_pipes, label="SDXL"):
    """Remove SDXL CLIP references after all required conditioning exists."""
    if not isinstance(model_pipe, dict):
        return "NOT AVAILABLE"
    resource_key = _resource_key_from_pipe(model_pipe)
    resources = _SDXL_RESOURCE_CACHE.get(resource_key)
    cached_clip = None
    if resources is not None:
        model, cached_clip, vae, metadata = resources
        _SDXL_RESOURCE_CACHE[resource_key] = (model, None, vae, metadata)

    active_clips = []
    for pipe in (model_pipe, *working_pipes):
        if not isinstance(pipe, dict):
            continue
        for key in ("clip", "clip_base", "clip_patched", "refiner_clip"):
            value = pipe.get(key)
            if value is not None:
                active_clips.append(value)
            if key in pipe:
                pipe[key] = None
        pipe["text_encoder_status"] = "EVICTED AFTER ENCODE"

    active_clips.clear()
    cached_clip = None
    resources = None
    gc.collect()
    try:
        import comfy.model_management

        comfy.model_management.cleanup_models_gc()
        comfy.model_management.soft_empty_cache(force=True)
    except Exception:
        pass
    print(
        f"[CMK Model Lifecycle] {label} text encoder: EVICTED AFTER ENCODE",
        flush=True,
    )
    return "EVICTED AFTER ENCODE"


def resolve_postprocess_model(value):
    """Materialise a module-35 MODEL specification at its first real consumer."""
    if not isinstance(value, dict):
        raise TypeError("CMK PostProcess Model requires a CMK model pipe")
    if is_postprocess_model_disabled(value):
        raise ValueError(
            "CMK PostProcess Model is set to None in the PostProcess Boundary. "
            "Select a checkpoint before enabling a model-dependent PostProcess module."
        )
    if not is_postprocess_model_spec(value):
        return value
    resources, status = _load_cached_resources(
        value.get("ckpt_name", DEFAULT_CHECKPOINT),
        value.get("vae_name", ""),
        value.get("checkpoint_vae", True),
    )
    model, clip, vae, metadata = resources
    result = dict(value)
    result.update(metadata)
    result.update({
        "model": model,
        "clip": clip,
        "vae": vae,
        "model_role": "postprocess",
        "model_family": "sdxl",
        "model_architecture": "sdxl",
        "materialized": True,
        "resource_cache_status": status,
    })
    print(f"[CMK PostProcess Model] {status} | {result.get('ckpt_name', '')}")
    return result


def unload_model_pipe(value):
    """Unload a concrete CMK model patcher; specifications remain a no-op."""
    if not isinstance(value, dict):
        return "NOT AVAILABLE"
    if is_postprocess_model_disabled(value):
        return "NOT CONFIGURED"
    model = value.get("model_patched")
    if model is None:
        model = value.get("model")
    if model is None and value.get("model_role") == "postprocess":
        resources = _SDXL_RESOURCE_CACHE.get(_resource_key(
            value.get("ckpt_name", DEFAULT_CHECKPOINT),
            value.get("vae_name", ""),
            value.get("checkpoint_vae", True),
        ))
        if resources is not None:
            model = resources[0]
    if model is None:
        return "NOT MATERIALIZED"
    try:
        import comfy.model_management

        comfy.model_management.unload_model_and_clones(model)
        return "UNLOADED"
    except Exception as exc:
        return f"UNLOAD FAILED: {exc}"


def unload_all_phase_models(label="CMK phase"):
    """Release every currently loaded model at an explicit phase boundary."""
    import gc
    import comfy.model_management

    try:
        comfy.model_management.unload_all_models()
        gc.collect()
        comfy.model_management.soft_empty_cache(force=True)
        status = "UNLOADED"
    except Exception as exc:
        status = f"UNLOAD FAILED: {exc}"
    print(f"[CMK Model Lifecycle] {label}: {status}", flush=True)
    return status


def _checkpoint_choices():
    choices = list(folder_paths.get_filename_list("checkpoints"))
    if DEFAULT_CHECKPOINT in choices:
        choices.remove(DEFAULT_CHECKPOINT)
        choices.insert(0, DEFAULT_CHECKPOINT)
    return choices


class CMKCheckpointVAELoaderPipe:
    """Create the shared read-only CMK MODEL resource pipe.

    Public contract:
        no PROCESS input
        MODEL output only

    Downstream modules may read model/clip/vae, but must never mutate MODEL
    or write module working state back into it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (_checkpoint_choices(),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "checkpoint_vae": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Family-bound sampler subgraphs connect this directly. It
                # remains optional for the loader's other internal uses.
                "PROCESS": ("CMK_PROCESS_SDXL",),
            },
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE",)
    RETURN_NAMES = ("MODEL SDXL",)
    FUNCTION = "load_checkpoint_vae_pipe"
    CATEGORY = "CMK/Flow/Input"

    def load_checkpoint_vae_pipe(
        self,
        ckpt_name,
        vae_name,
        checkpoint_vae,
        PROCESS=_PROCESS_NOT_CONNECTED,
    ):
        if PROCESS is None or (
            isinstance(PROCESS, dict) and not PROCESS.get("family_active", True)
        ):
            return (ExecutionBlocker(None),)
        if PROCESS is not _PROCESS_NOT_CONNECTED:
            if not isinstance(PROCESS, dict):
                raise TypeError("CMK Checkpoint VAE Loader: PROCESS must be a CMK process pipe")
            family = str(PROCESS.get("model_family", "sdxl")).strip().lower()
            if family != "sdxl":
                raise ValueError("CMK Checkpoint VAE Loader requires PROCESS SDXL")
            if PROCESS.get("unload_models_after_use") is True:
                _release_stale_resource_cache(
                    _resource_key(ckpt_name, vae_name, checkpoint_vae),
                    "SDXL checkpoint switch",
                )
        resources, resource_cache_status = _load_cached_resources(
            ckpt_name, vae_name, checkpoint_vae
        )
        model, clip, vae, metadata = resources
        metadata = dict(metadata)
        metadata["vae_selection_name"] = str(vae_name)
        metadata["resource_cache_status"] = resource_cache_status
        print(
            f"[CMK SDXL Resources] {resource_cache_status} | "
            f"{ckpt_name}"
        )
        return ({
            "model": model,
            "clip": clip,
            "vae": vae,
            **metadata,
        },)
