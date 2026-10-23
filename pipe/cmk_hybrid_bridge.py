from __future__ import annotations

import gc
import hashlib
import json

from comfy_execution.graph_utils import ExecutionBlocker

from .cmk_log_pipe import cmk_add_block
from .cmk_prompt_result_state import (
    IncompleteResultError,
    MISSING,
    PromptStateUnavailableError,
    UnsafeSnapshotTypeError,
    get_result,
    make_key,
    publish_result,
)
from .cmk_refiner_boundary_cache import _canonical_node, _is_link, _resolve_current_node
from .cmk_visual import empty_visual


_HYBRID_HANDOFF_PHASE = "hybrid.handoff"
_HYBRID_HANDOFF_SCHEMA = "cmk_hybrid_handoff_result_v1"
_HYBRID_HANDOFF_INPUTS = (
    "PROCESS ZIT",
    "SAMPLED SDXL",
    "LOG SDXL",
    "VISUAL SDXL",
)


def _hybrid_handoff_state_key(prompt, unique_id):
    if not isinstance(prompt, dict) or unique_id is None:
        return None
    _, current = _resolve_current_node(prompt, unique_id)
    if not isinstance(current, dict):
        return None
    current_inputs = current.get("inputs", {}) or {}
    memo = {}
    roots = {}
    for name in _HYBRID_HANDOFF_INPUTS:
        link = current_inputs.get(name)
        if not _is_link(prompt, link):
            return None
        roots[name] = {
            "node": _canonical_node(prompt, link[0], memo, set()),
            "output": int(link[1]),
        }
    payload = {
        "schema": _HYBRID_HANDOFF_SCHEMA,
        "inputs": roots,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    try:
        return make_key(
            node_id=unique_id,
            model_family="hybrid",
            phase=_HYBRID_HANDOFF_PHASE,
            fingerprint=fingerprint,
            schema_version=_HYBRID_HANDOFF_SCHEMA,
        )
    except PromptStateUnavailableError:
        return None


def _hybrid_handoff_hit(state_key):
    if state_key is None:
        return None
    result = get_result(state_key)
    return None if result is MISSING else result


def _publish_hybrid_handoff(state_key, image, log, visual, process):
    if state_key is None:
        return
    result = {
        "image": image,
        "log": log,
        "visual": visual,
        "metadata": {
            "handoff_percent": int(process.get("hybrid_sdxl_handoff", 90)),
            "zit_denoise": float(process.get("hybrid_zit_denoise", 0.20)),
            "hybrid_inpaint_mode": bool(process.get("hybrid_inpaint_mode", False)),
        },
    }
    try:
        publish_result(state_key, result)
    except (IncompleteResultError, UnsafeSnapshotTypeError) as exc:
        print(f"[CMK Hybrid Handoff Result-State] NOT PUBLISHED: {exc}", flush=True)


def _release_completed_sdxl_patch(patcher):
    """Drop patch-only tensors from a completed Hybrid-Inpaint clone."""
    released_entries = sum(
        len(entries) for entries in getattr(patcher, "patches", {}).values()
    )
    try:
        patcher.cleanup()
    except Exception:
        pass
    for name in (
        "patches",
        "object_patches",
        "weight_wrapper_patches",
        "attachments",
        "additional_models",
    ):
        value = getattr(patcher, name, None)
        if hasattr(value, "clear"):
            value.clear()
    # Fooocus' spatial block patch retains its computed feature through the
    # patcher's transformer options. The clone is terminal at this point.
    patcher.model_options = {"transformer_options": {}}
    gc.collect()
    return released_entries


class CMKHybridZITInputPipe:
    """Select the ordinary ZIT source or the completed SDXL hybrid handoff."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PROCESS ZIT": ("CMK_PROCESS_Z_IMAGE",),
            },
            "optional": {
                "IMAGE SOURCE": ("IMAGE", {"lazy": True}),
                "LOG SOURCE": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL SOURCE": ("CMK_VISUAL_PIPE", {"lazy": True}),
                "SAMPLED SDXL": ("CMK_SAMPLED_PIPE", {"lazy": True}),
                "LOG SDXL": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL SDXL": ("CMK_VISUAL_PIPE", {"lazy": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "CMK_LOG_PIPE", "CMK_VISUAL_PIPE")
    RETURN_NAMES = ("IMAGE", "LOG", "VISUAL")
    FUNCTION = "select"
    CATEGORY = "CMK/Flow/Bridge"
    DEV_ONLY = True

    def check_lazy_status(self, **inputs):
        process = inputs.get("PROCESS ZIT")
        if process is None:
            return ["PROCESS ZIT"]
        if isinstance(process, dict) and not process.get("family_active", True):
            return []
        if bool(process.get("hybrid_mode", False)):
            state_key = _hybrid_handoff_state_key(
                inputs.get("prompt"), inputs.get("unique_id")
            )
            if _hybrid_handoff_hit(state_key) is not None:
                return []
            for name in ("SAMPLED SDXL", "LOG SDXL", "VISUAL SDXL"):
                if inputs.get(name) is None:
                    return [name]
            return []
        required = ["LOG SOURCE", "VISUAL SOURCE"]
        if bool(process.get("boolean_inpaint_mode", False)):
            required.insert(0, "IMAGE SOURCE")
        for name in required:
            if inputs.get(name) is None:
                return [name]
        return []

    @staticmethod
    def select(**inputs):
        process = inputs.get("PROCESS ZIT")
        if process is None or (
            isinstance(process, dict) and not process.get("family_active", True)
        ):
            blocked = ExecutionBlocker(None)
            return (blocked, blocked, blocked)
        if not isinstance(process, dict):
            raise TypeError("CMK Hybrid ZIT Input requires PROCESS ZIT")

        if not bool(process.get("hybrid_mode", False)):
            return (
                inputs.get("IMAGE SOURCE"),
                inputs.get("LOG SOURCE"),
                inputs.get("VISUAL SOURCE") or empty_visual(),
            )

        state_key = _hybrid_handoff_state_key(
            inputs.get("prompt"), inputs.get("unique_id")
        )
        hit = _hybrid_handoff_hit(state_key)
        if hit is not None:
            return (hit["image"], hit["log"], hit["visual"])

        sampled = inputs.get("SAMPLED SDXL")
        if not isinstance(sampled, dict):
            raise TypeError("CMK Hybrid ZIT Input requires SAMPLED SDXL")
        image = sampled.get("image_1st_pass")
        if image is None:
            raise ValueError(
                "CMK Hybrid ZIT Input: SDXL handoff contains no clean x0 image"
            )
        unload_requested = bool(process.get("unload_models_after_use", True))
        sdxl_model = sampled.get("model_patched") or sampled.get("model")
        sdxl_unload_status = "NOT AVAILABLE"
        patch_release_status = "NOT REQUIRED"
        cache_release_status = "KEPT"
        if unload_requested and sdxl_model is not None:
            if bool(process.get("hybrid_inpaint_mode", False)):
                released_patch_entries = _release_completed_sdxl_patch(sdxl_model)
                patch_release_status = f"RELEASED ({released_patch_entries} entries)"
            from .cmk_family_result import clear_sdxl_sampled_boundary_cache
            from .loaders.checkpoint_vae_loader import unload_all_phase_models

            cleared = clear_sdxl_sampled_boundary_cache()
            cache_release_status = f"CLEARED ({cleared})"
            sdxl_unload_status = unload_all_phase_models("Hybrid SDXL -> ZIT handoff")
        elif not unload_requested:
            sdxl_unload_status = "KEPT LOADED"
        handoff = int(process.get("hybrid_sdxl_handoff", 90))
        denoise = float(process.get("hybrid_zit_denoise", 0.20))
        log = cmk_add_block(
            inputs.get("LOG SDXL"),
            "Hybrid Handoff",
            35,
            [
                "MODE            : SDXL -> ZIT",
                f"SDXL HANDOFF   : {handoff}%",
                "BRIDGE          : SDXL x0 decode -> ZIT VAE encode",
                f"SDXL MODEL     : {sdxl_unload_status}",
                f"SDXL PATCH DATA: {patch_release_status}",
                f"SDXL CACHE      : {cache_release_status}",
                f"ZIT DENOISE     : {denoise:.2f}",
            ],
            True,
        )
        visual = inputs.get("VISUAL SDXL") or empty_visual()
        _publish_hybrid_handoff(state_key, image, log, visual, process)
        return (image, log, visual)
