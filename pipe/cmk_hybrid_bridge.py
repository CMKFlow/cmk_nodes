from __future__ import annotations

import gc
import hashlib
import json
import pickle
import threading

from comfy_execution.graph_utils import ExecutionBlocker

from ..cmk_distribution import (
    content_guard_prompt_token,
    effective_content_guard_enabled,
)
from ..engine.content_guard import GUARD_VERSION, get_content_guard
from ..utils.tensor_utils import tensor_to_uint8_rgb
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
_HYBRID_BOUNDARY_SCOPE = "sdxl_handoff_boundary"
_HYBRID_BOUNDARY_SCHEMA = "cmk_hybrid_sdxl_image_boundary_v1"
_HYBRID_BOUNDARY_GUARD_VERSION = f"{GUARD_VERSION}:hybrid-sdxl-handoff-v1"
_HYBRID_BOUNDARY_MAX_ENTRIES = 8
_HYBRID_FINGERPRINT_IGNORED_SOURCE_INPUTS = {
    "LoRA ZIT",
    "instantid_enabled",
    "instantid_handoff_percent",
    "instantid_mode",
    "live_preview_zit",
    "stop_after_handoff",
    "unload_models_after_use",
    "unload_zit_after",
}
_HYBRID_BOUNDARY_PROMPT_LOCK = threading.RLock()
_HYBRID_BOUNDARY_PROMPT_SOURCE = None
_HYBRID_BOUNDARY_PROMPT_PREPARED = None


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


def _hybrid_boundary_prompt(prompt):
    """Remove source controls that cannot affect the SDXL pre-pass pixels."""
    global _HYBRID_BOUNDARY_PROMPT_SOURCE, _HYBRID_BOUNDARY_PROMPT_PREPARED
    if not isinstance(prompt, dict):
        return prompt
    with _HYBRID_BOUNDARY_PROMPT_LOCK:
        if prompt is _HYBRID_BOUNDARY_PROMPT_SOURCE:
            return _HYBRID_BOUNDARY_PROMPT_PREPARED
        prepared = dict(prompt)
        for node_id, node in prompt.items():
            if not isinstance(node, dict) or node.get("class_type") != "CMKPipeCreateImage":
                continue
            copied = dict(node)
            copied["inputs"] = {
                name: value
                for name, value in (node.get("inputs", {}) or {}).items()
                if str(name) not in _HYBRID_FINGERPRINT_IGNORED_SOURCE_INPUTS
            }
            prepared[node_id] = copied
        # Keep the prepared prompt alive while cmk_persistent_cache memoizes by
        # object identity. This prevents Python id reuse across queue prompts.
        _HYBRID_BOUNDARY_PROMPT_SOURCE = prompt
        _HYBRID_BOUNDARY_PROMPT_PREPARED = prepared
        return prepared


def _upstream_sampled_gate_id(prompt, unique_id):
    if not isinstance(prompt, dict):
        return None
    _, current = _resolve_current_node(prompt, unique_id)
    if not isinstance(current, dict):
        return None
    if current.get("class_type") == "CMKFamilyBranchGateSDXLSampled":
        return unique_id
    pending = []
    sampled_link = (current.get("inputs", {}) or {}).get("SAMPLED SDXL")
    if _is_link(prompt, sampled_link):
        pending.append(sampled_link[0])
    visited = set()
    matches = []
    while pending:
        node_id = pending.pop()
        key = str(node_id)
        if key in visited:
            continue
        visited.add(key)
        node = prompt.get(node_id) or prompt.get(key)
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == "CMKFamilyBranchGateSDXLSampled":
            matches.append(node_id)
            continue
        for value in (node.get("inputs", {}) or {}).values():
            if _is_link(prompt, value):
                pending.append(value[0])
    return matches[0] if len(matches) == 1 else None


def _hybrid_boundary_cache_key(prompt, unique_id):
    gate_id = _upstream_sampled_gate_id(prompt, unique_id)
    if gate_id is None:
        return None
    try:
        from .cmk_persistent_cache import build_node_fingerprint

        key, _detail = build_node_fingerprint(
            _hybrid_boundary_prompt(prompt),
            gate_id,
            ("CMKFamilyBranchGateSDXLSampled",),
            _HYBRID_BOUNDARY_SCHEMA,
            exclude_inputs=("MODEL", "PROCESS", "LOG", "VISUAL", "diagnostic"),
            include_node_identity=True,
            extra_identity={
                "content_guard": (
                    f"{_HYBRID_BOUNDARY_GUARD_VERSION}:"
                    f"{content_guard_prompt_token(prompt)}"
                ),
            },
        )
        return key
    except Exception:
        return None


def _hybrid_boundary_contract(process, cache_key, prompt):
    return {
        "schema": _HYBRID_BOUNDARY_SCHEMA,
        "cache_key": cache_key,
        "content_guard": (
            f"{_HYBRID_BOUNDARY_GUARD_VERSION}:"
            f"{content_guard_prompt_token(prompt)}"
        ),
        "sdxl": {
            "handoff_percent": int(process.get("hybrid_sdxl_handoff", 90)),
            "hybrid_inpaint_mode": bool(process.get("hybrid_inpaint_mode", False)),
            "width": int(process.get("sdxl_width", process.get("width", 0)) or 0),
            "height": int(process.get("sdxl_height", process.get("height", 0)) or 0),
        },
    }


def _cpu_snapshot(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu").clone()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return value


def _invalidate_hybrid_boundary(cache_key):
    if not cache_key:
        return
    try:
        from .cmk_persistent_cache import cache_path

        for suffix in (".pkl", ".rev"):
            cache_path(_HYBRID_BOUNDARY_SCOPE, cache_key, suffix).unlink(
                missing_ok=True
            )
    except Exception:
        pass


def _guard_hybrid_handoff_image(image, process):
    if not effective_content_guard_enabled(process):
        return "BYPASS · BASELINE"
    guard = get_content_guard()
    for item in image:
        guard.inspect_content(
            tensor_to_uint8_rgb(item), "target_intermediate",
        )
    return "PASS · SDXL/ZIT HANDOFF"


def persist_hybrid_sdxl_boundary(process, image, log, visual, prompt, unique_id):
    cache_key = _hybrid_boundary_cache_key(prompt, unique_id)
    if not cache_key:
        return None
    guard_status = _guard_hybrid_handoff_image(image, process)
    image_snapshot = _cpu_snapshot(image)
    payload = {
        "contract": _hybrid_boundary_contract(process, cache_key, prompt),
        "image": image_snapshot,
        "log": _cpu_snapshot(log),
        "visual": _cpu_snapshot(visual or empty_visual()),
        "metadata": {
            "content_guard_handoff": guard_status,
            "image_shape": tuple(getattr(image_snapshot, "shape", ())),
            "image_dtype": str(getattr(image_snapshot, "dtype", "unknown")),
            "image_device": str(getattr(image_snapshot, "device", "cpu")),
        },
    }
    payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {
        "schema": _HYBRID_BOUNDARY_SCHEMA,
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "payload": payload_bytes,
    }
    try:
        from .cmk_persistent_cache import save_pickle

        save_pickle(
            _HYBRID_BOUNDARY_SCOPE,
            cache_key,
            envelope,
            max_entries=_HYBRID_BOUNDARY_MAX_ENTRIES,
        )
        print(f"SDXL/ZIT HANDOFF · PERSISTED · {cache_key[:12]}", flush=True)
        return cache_key
    except Exception as exc:
        _invalidate_hybrid_boundary(cache_key)
        print(f"[CMK SDXL/ZIT Boundary] PERSIST FAILED · {exc}", flush=True)
        return None


def load_hybrid_sdxl_boundary(process, prompt, unique_id):
    cache_key = _hybrid_boundary_cache_key(prompt, unique_id)
    if not cache_key:
        return None
    try:
        from .cmk_persistent_cache import load_pickle, pickle_available

        if not pickle_available(_HYBRID_BOUNDARY_SCOPE, cache_key):
            return None
        envelope = load_pickle(_HYBRID_BOUNDARY_SCOPE, cache_key)
        if not isinstance(envelope, dict) or envelope.get("schema") != _HYBRID_BOUNDARY_SCHEMA:
            raise ValueError("invalid boundary envelope")
        payload_bytes = envelope.get("payload")
        if not isinstance(payload_bytes, bytes):
            raise ValueError("boundary payload is missing")
        if hashlib.sha256(payload_bytes).hexdigest() != envelope.get("sha256"):
            raise ValueError("boundary payload checksum mismatch")
        payload = pickle.loads(payload_bytes)
        expected = _hybrid_boundary_contract(process, cache_key, prompt)
        if not isinstance(payload, dict) or payload.get("contract") != expected:
            raise ValueError("boundary contract mismatch")
        image = payload.get("image")
        log = payload.get("log")
        visual = payload.get("visual")
        metadata = payload.get("metadata")
        if image is None or not isinstance(log, dict) or not isinstance(visual, dict):
            raise ValueError("boundary payload is incomplete")
        if not isinstance(metadata, dict):
            raise ValueError("boundary metadata is missing")
        if (
            effective_content_guard_enabled(process)
            and metadata.get("content_guard_handoff") != "PASS · SDXL/ZIT HANDOFF"
        ):
            raise ValueError("boundary ContentGuard approval is missing")
        expected_shape = tuple(metadata.get("image_shape", ()))
        if expected_shape and tuple(getattr(image, "shape", ())) != expected_shape:
            raise ValueError("boundary image shape mismatch")
        if str(getattr(image, "dtype", "unknown")) != metadata.get("image_dtype"):
            raise ValueError("boundary image dtype mismatch")
        if str(getattr(image, "device", "cpu")) != metadata.get("image_device"):
            raise ValueError("boundary image device mismatch")
        log = cmk_add_block(
            log,
            "Hybrid Handoff",
            35,
            [
                "MODE            : SDXL -> ZIT",
                "SDXL BOUNDARY   : PERSISTENT CACHE HIT",
                "SDXL SAMPLING   : NOT REQUESTED",
                f"ZIT DENOISE     : {float(process.get('hybrid_zit_denoise', 0.20)):.2f}",
            ],
            True,
        )
        print(f"SDXL/ZIT HANDOFF · CACHE HIT · {cache_key[:12]}", flush=True)
        return image, log, visual
    except Exception as exc:
        _invalidate_hybrid_boundary(cache_key)
        print(
            "[CMK SDXL/ZIT Boundary] MISS · invalid entry removed · "
            f"{exc}",
            flush=True,
        )
        return None


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
            persistent_hit = load_hybrid_sdxl_boundary(
                process, inputs.get("prompt"), inputs.get("unique_id")
            )
            if persistent_hit is not None:
                return []
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

        persistent_hit = load_hybrid_sdxl_boundary(
            process, inputs.get("prompt"), inputs.get("unique_id")
        )
        if persistent_hit is not None:
            state_key = _hybrid_handoff_state_key(
                inputs.get("prompt"), inputs.get("unique_id")
            )
            _publish_hybrid_handoff(
                state_key,
                persistent_hit[0],
                persistent_hit[1],
                persistent_hit[2],
                process,
            )
            return persistent_hit

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
