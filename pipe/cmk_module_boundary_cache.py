from __future__ import annotations

import json
import os

from comfy_execution.graph_utils import ExecutionBlocker

from .cmk_persistent_cache import (
    build_node_fingerprint,
    build_upstream_cache_manifest,
    cache_directory,
    prune,
    write_status,
)
from .cmk_prompt_result_state import active_prompt_id
from . import cmk_visual
from ..cmk_distribution import (
    content_guard_prompt_token,
    stop_after_handoff_prompt_state,
)
from ..utils.cmk_timing import cmk_timed_call
from ..utils.preview_payload import _as_rgb_uint8


_DETAILER_SESSION: dict[str, dict] = {}
_Z_IMAGE_SESSION: dict[str, tuple[dict, object]] = {}
_FACE_REBUILD_SESSION: dict[str, tuple[dict, object]] = {}


def _stop_after_handoff_blocks_module(prompt, unique_id, scope, label):
    """Block a complete postprocess module before any lazy upstream request."""
    if not stop_after_handoff_prompt_state(prompt, unique_id):
        return False
    write_status(
        scope,
        "STOP_AFTER_HANDOFF_PREVIEW_BLOCKED",
        detail="preview-only contract blocks downstream postprocess",
        unique_id=unique_id,
    )
    print(
        f"[CMK Boundary Cache / {label}] "
        "STOP AFTER HANDOFF -> UPSTREAM POSTPROCESS BLOCKED"
    )
    return True


def _blocked_outputs(count):
    blocked = ExecutionBlocker(None)
    return tuple(blocked for _ in range(count))

# Dependency manifests take scope and key from the branch class itself (K5-B).
# Imported lazily: the branch modules pull in the detector/sampler engines.
def _detailer_branch():
    from ..nodes.image.smart_detailer import CMK_SmartDetailerPipe

    return CMK_SmartDetailerPipe


def _faceprocess_branch():
    from .cmk_faceprocess import CMKFaceProcessPipe

    return CMKFaceProcessPipe


_DETAILER_BRANCH_SPECS = {"CMK_SmartDetailerPipe": _detailer_branch}
_FACEPROCESS_BRANCH_SPECS = {"CMKFaceProcessPipe": _faceprocess_branch}


def reset_module_boundary_cache_state() -> dict[str, int]:
    """Release only in-memory state paired with persistent boundary entries."""
    states = {
        "detailer_session": _DETAILER_SESSION,
        "z_image_session": _Z_IMAGE_SESSION,
        "face_rebuild_session": _FACE_REBUILD_SESSION,
    }
    counts = {name: len(state) for name, state in states.items()}
    for state in states.values():
        state.clear()
    return counts


def _detailer_diag_context(unique_id=None):
    """Return prompt/node context for temporary Detailer diagnostics."""
    try:
        prompt_id = active_prompt_id()
    except Exception:
        prompt_id = None
    return (
        f"prompt_id={prompt_id if prompt_id is not None else 'UNAVAILABLE'} "
        f"node_id={unique_id if unique_id is not None else 'UNAVAILABLE'}"
    )



def _prompt_node(prompt, node_id):
    if not isinstance(prompt, dict):
        return None
    if node_id in prompt:
        return prompt[node_id]
    return prompt.get(str(node_id))


def _prompt_link(prompt, value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
        and isinstance(_prompt_node(prompt, value[0]), dict)
    )


def _resolve_boundary_node(prompt, unique_id, class_type):
    if not isinstance(prompt, dict):
        return None, None

    for candidate in (unique_id, str(unique_id)):
        node = _prompt_node(prompt, candidate)
        if isinstance(node, dict):
            return candidate, node

    matches = [
        (node_id, node)
        for node_id, node in prompt.items()
        if isinstance(node, dict)
        and str(node.get("class_type", "")) == str(class_type)
    ]
    if len(matches) == 1:
        return matches[0]
    return None, None


def _collect_upstream_nodes(prompt, node_id, wanted, found, visited):
    key = str(node_id)
    # Expanded subgraphs have an instance prefix. Never inspect another
    # module's switches; unscoped graphs stop at the preceding boundary.
    if visited:
        scopes = {item.rsplit(":", 1)[0] for item in visited if ":" in item}
        if scopes and (":" not in key or key.rsplit(":", 1)[0] not in scopes):
            return
    if key in visited:
        return
    visited.add(key)

    node = _prompt_node(prompt, node_id)
    if not isinstance(node, dict):
        return

    class_type = str(node.get("class_type", ""))
    if class_type.endswith("BoundaryCache"):
        return
    if class_type in wanted:
        found.append((node_id, node))
        if "Prepare" in class_type:
            return

    for value in (node.get("inputs", {}) or {}).values():
        if _prompt_link(prompt, value):
            _collect_upstream_nodes(
                prompt,
                value[0],
                wanted,
                found,
                visited,
            )


def _faceprocess_disabled_state(prompt, unique_id):
    """Return whether the complete FaceProcess module is statically disabled.

    This inspection is deliberately shallow/linear. It reads only Boolean widget
    values from the FaceProcess prepare and execute nodes and never fingerprints
    the expanded upstream graph.
    """
    resolved_id, boundary = _resolve_boundary_node(
        prompt,
        unique_id,
        "CMKFaceBoundaryCache",
    )
    if not isinstance(boundary, dict):
        return False, "boundary unavailable"

    found = []
    visited = set()
    wanted = {"CMKFaceProcessPreparePipe", "CMKFaceProcessPipe"}

    for input_name in ("IMAGE", "LOG"):
        value = (boundary.get("inputs", {}) or {}).get(input_name)
        if _prompt_link(prompt, value):
            _collect_upstream_nodes(
                prompt,
                value[0],
                wanted,
                found,
                visited,
            )

    prepares = [
        node for _node_id, node in found
        if str(node.get("class_type", "")) == "CMKFaceProcessPreparePipe"
    ]
    branches = [
        node for _node_id, node in found
        if str(node.get("class_type", "")) == "CMKFaceProcessPipe"
    ]

    global_values = [
        bool((node.get("inputs", {}) or {}).get("face_global_enable", True))
        for node in prepares
    ]
    local_values = [
        bool((node.get("inputs", {}) or {}).get("enable", True))
        for node in branches
    ]

    if global_values and not any(global_values):
        return True, (
            f"boundary={resolved_id}; global OFF; "
            f"branches={len(local_values)}"
        )

    if local_values and not any(local_values):
        return True, (
            f"boundary={resolved_id}; all local branches OFF; "
            f"branches={len(local_values)}"
        )

    return False, (
        f"boundary={resolved_id}; "
        f"global={global_values or ['unknown']}; "
        f"local={local_values or ['unknown']}"
    )


def _detailer_disabled_state(prompt, unique_id):
    """Return whether the complete Detailer module is statically disabled."""
    resolved_id, boundary = _resolve_boundary_node(
        prompt,
        unique_id,
        "CMKDetailerBoundaryCache",
    )
    if not isinstance(boundary, dict):
        return False, "boundary unavailable"

    found = []
    visited = set()
    wanted = {"CMKDetailerPreparePipe", "CMK_SmartDetailerPipe"}
    for input_name in ("IMAGE", "LOG"):
        value = (boundary.get("inputs", {}) or {}).get(input_name)
        if _prompt_link(prompt, value):
            _collect_upstream_nodes(prompt, value[0], wanted, found, visited)

    prepares = [
        node for _node_id, node in found
        if str(node.get("class_type", "")) == "CMKDetailerPreparePipe"
    ]
    branches = [
        node for _node_id, node in found
        if str(node.get("class_type", "")) == "CMK_SmartDetailerPipe"
    ]
    global_values = [
        bool((node.get("inputs", {}) or {}).get("detailer_global_enable", True))
        for node in prepares
    ]
    local_values = [
        bool((node.get("inputs", {}) or {}).get("enable", True))
        for node in branches
    ]
    if global_values and not any(global_values):
        return True, f"boundary={resolved_id}; global OFF; branches={len(local_values)}"
    if local_values and not any(local_values):
        return True, f"boundary={resolved_id}; all local branches OFF; branches={len(local_values)}"
    return False, (
        f"boundary={resolved_id}; global={global_values or ['unknown']}; "
        f"local={local_values or ['unknown']}"
    )


def _face_rebuild_disabled_state(prompt, unique_id):
    """Return whether the complete FaceRebuild module is statically disabled."""
    resolved_id, boundary = _resolve_boundary_node(
        prompt,
        unique_id,
        "CMKFaceRebuildBoundaryCache",
    )
    if not isinstance(boundary, dict):
        return False, "boundary unavailable"

    found = []
    visited = set()
    wanted = {
        "CMKInstantIDFaceRebuildPreparePipe",
        "CMKInstantIDFaceRebuildSDXL",
        "CMKInstantIDFaceRebuildAdvancedSDXL",
    }
    for input_name in ("IMAGE", "LOG", "diagnostic"):
        value = (boundary.get("inputs", {}) or {}).get(input_name)
        if _prompt_link(prompt, value):
            _collect_upstream_nodes(prompt, value[0], wanted, found, visited)

    prepares = [
        node for _node_id, node in found
        if str(node.get("class_type", "")) in wanted
    ]
    global_values = [
        bool((node.get("inputs", {}) or {}).get(
            "FACEREBUILD ENABLE",
            (node.get("inputs", {}) or {}).get("GLOBAL ENABLE", True),
        ))
        for node in prepares
    ]
    local_values = [
        bool((node.get("inputs", {}) or {}).get("ENABLE", True))
        for node in prepares
    ]
    if global_values and not any(global_values):
        return True, f"boundary={resolved_id}; global OFF"
    if local_values and not any(local_values):
        return True, f"boundary={resolved_id}; local OFF"
    return False, (
        f"boundary={resolved_id}; global={global_values or ['unknown']}; "
        f"local={local_values or ['unknown']}"
    )


def _faceswap_disabled_state(prompt, unique_id):
    resolved_id, boundary = _resolve_boundary_node(
        prompt, unique_id, "CMKFaceSwapBoundaryCache",
    )
    if not isinstance(boundary, dict):
        return False, "boundary unavailable"
    found, visited = [], set()
    for name in ("IMAGE", "LOG"):
        value = (boundary.get("inputs") or {}).get(name)
        if _prompt_link(prompt, value):
            _collect_upstream_nodes(
                prompt, value[0], {"CMKFaceSwapImagePipe"}, found, visited,
            )
    # Unknown or dynamically linked switches must not be interpreted as OFF.
    disabled = bool(found) and all(
        (node.get("inputs") or {}).get("GLOBAL ENABLE", True) is False
        or (node.get("inputs") or {}).get("ENABLE", True) is False
        for _, node in found
    )
    return disabled, f"boundary={resolved_id}; own branches={len(found)}"


def _entry_paths(scope: str, cache_key: str):
    base = cache_directory(scope) / cache_key
    return (
        base.with_suffix(".safetensors"),
        base.with_suffix(".json"),
        base.with_suffix(".deps"),
    )


def _disk_available(scope: str, cache_key: str) -> bool:
    return all(path.is_file() for path in _entry_paths(scope, cache_key))


def _remove_entry(scope: str, cache_key: str) -> None:
    for path in (
        *_entry_paths(scope, cache_key),
        *_diagnostic_entry_paths(scope, cache_key),
        *_visual_entry_paths(scope, cache_key),
    ):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _retain_latest_boundary_entry(
    scope: str,
    cache_key: str,
    unique_id,
    session: dict | None = None,
) -> None:
    """Keep only the latest cache revision for one concrete boundary node."""
    directory = cache_directory(scope)
    index_path = directory / "latest_by_node.json"
    index_tmp = directory / "latest_by_node.json.tmp"
    slot = str(unique_id)

    try:
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        if not isinstance(index, dict):
            index = {}
    except Exception:
        index = {}

    previous_key = index.get(slot)
    index[slot] = cache_key
    try:
        with index_tmp.open("w", encoding="utf-8") as handle:
            json.dump(index, handle, ensure_ascii=False, sort_keys=True)
        os.replace(index_tmp, index_path)
    finally:
        try:
            index_tmp.unlink(missing_ok=True)
        except Exception:
            pass

    if isinstance(previous_key, str) and previous_key != cache_key:
        _remove_entry(scope, previous_key)
        if session is not None:
            session.pop(previous_key, None)


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _diagnostic_entry_paths(scope: str, cache_key: str):
    base = cache_directory(scope) / cache_key
    return (
        base.with_suffix(".diagnostic.safetensors"),
        base.with_suffix(".diagnostic.json"),
    )


def _overlaps_stored_tensor(tensor, tensors: dict) -> bool:
    """Same rule safetensors uses to reject shared memory: one storage, overlapping bytes."""
    if not tensor.numel():
        return False
    storage = tensor.untyped_storage().data_ptr()
    start = tensor.data_ptr()
    stop = start + tensor.numel() * tensor.element_size()
    for other in tensors.values():
        if not other.numel() or other.untyped_storage().data_ptr() != storage:
            continue
        other_start = other.data_ptr()
        if start < other_start + other.numel() * other.element_size() and other_start < stop:
            return True
    return False


def _encode_diagnostic_value(value, tensors: dict, counter: list[int], views: dict):
    """Move tensor/ndarray payloads into safetensors and retain structure in JSON.

    Diagnostics legitimately reference one image from several places (preview,
    images, stages, nested diagnostic groups). An identical view reuses its
    stored key; a different view that overlaps stored memory is copied, because
    safetensors refuses to write shared memory.
    """
    try:
        import numpy as np
        import torch

        if isinstance(value, (torch.Tensor, np.ndarray)):
            source = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
            view = (
                str(source.device),
                source.untyped_storage().data_ptr(),
                source.storage_offset(),
                tuple(source.shape),
                tuple(source.stride()),
                source.dtype,
            )
            if view in views:
                return {"__cmk_diagnostic_tensor__": views[view]}
            key = f"diagnostic_{counter[0]:04d}"
            counter[0] += 1
            tensor = source.detach().to("cpu").contiguous()
            if _overlaps_stored_tensor(tensor, tensors):
                tensor = tensor.clone()
            tensors[key] = tensor
            views[view] = key
            return {"__cmk_diagnostic_tensor__": key}
    except Exception:
        pass

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _encode_diagnostic_value(item, tensors, counter, views)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _encode_diagnostic_value(item, tensors, counter, views)
            for item in value
        ]
    return str(value)


def _decode_diagnostic_value(value, tensors: dict):
    if isinstance(value, dict):
        key = value.get("__cmk_diagnostic_tensor__")
        if isinstance(key, str) and len(value) == 1:
            if key not in tensors:
                raise KeyError(f"cached diagnostic tensor {key!r} is missing")
            return tensors[key].detach().cpu().numpy()
        return {
            key: _decode_diagnostic_value(item, tensors)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_decode_diagnostic_value(item, tensors) for item in value]
    return value


def _save_diagnostic(scope: str, cache_key: str, diagnostic: dict) -> None:
    _write_encoded(*_diagnostic_entry_paths(scope, cache_key), diagnostic, "CMK_DIAGNOSTIC_BOUNDARY", scope)
    prune(scope, ".diagnostic.safetensors", 16)
    prune(scope, ".diagnostic.json", 16)


def _load_diagnostic(scope: str, cache_key: str):
    diagnostic = _read_encoded(*_diagnostic_entry_paths(scope, cache_key))
    if not isinstance(diagnostic, dict) or diagnostic.get("type") != "CMK_DIAGNOSTIC":
        raise TypeError("cached boundary diagnostic is invalid")
    return diagnostic


# -- VISUAL publication at the boundary (H-3 / F-03) ----------------------
# Module definitions route VISUAL through their boundary: "VISUAL IN" is the
# incoming pipe, "VISUAL" the pipe the compute node returned with its own
# providers.  The boundary stores exactly those providers with its result and
# re-attaches them on a HIT, so the compute node is never needed for VISUAL.
# Older embedded definitions leave both inputs unconnected and are unaffected.
VISUAL_PUBLICATION_INPUTS = {
    "VISUAL IN": ("CMK_VISUAL_PIPE", {"lazy": True}),
    "VISUAL": ("CMK_VISUAL_PIPE", {"lazy": True}),
}


def _visual_entry_paths(scope: str, cache_key: str):
    # Suffixes outside the "*.safetensors" / "*.json" prune globs of the result entry.
    base = cache_directory(scope) / cache_key
    return base.with_suffix(".visual-tensors"), base.with_suffix(".visual")


def _connected_inputs(prompt, unique_id) -> dict:
    node = prompt.get(str(unique_id)) if isinstance(prompt, dict) else None
    return (node.get("inputs") or {}) if isinstance(node, dict) else {}


def _visual_routed(prompt, unique_id) -> bool:
    # "VISUAL" is an internal link from the compute node; "VISUAL IN" disappears
    # from the expanded prompt when the module's VISUAL input is unconnected.
    return "VISUAL" in _connected_inputs(prompt, unique_id)


def _module_providers(visual_in, visual) -> list[dict]:
    """Providers the compute node added, channels reduced to the RGB the Visualizer renders."""
    incoming = {item.get("provider_id") for item in cmk_visual.normalize_visual(visual_in)["providers"]}
    return [
        {**provider, "channels": {name: _as_rgb_uint8(image) for name, image in provider["channels"].items()}}
        for provider in cmk_visual.normalize_visual(visual)["providers"]
        if provider.get("provider_id") not in incoming
    ]


def _save_visual(scope: str, cache_key: str, providers: list[dict]) -> None:
    # Channels as ordered pairs: the JSON index is written with sorted keys.
    stored = [{**provider, "channels": [[name, image] for name, image in provider["channels"].items()]}
              for provider in providers]
    _write_encoded(*_visual_entry_paths(scope, cache_key), {"providers": stored}, "CMK_VISUAL_BOUNDARY", scope)
    prune(scope, ".visual-tensors", 16)
    prune(scope, ".visual", 16)


def _load_visual(scope: str, cache_key: str) -> list[dict]:
    providers = _read_encoded(*_visual_entry_paths(scope, cache_key)).get("providers")
    if not isinstance(providers, list):
        raise TypeError("cached boundary VISUAL providers are invalid")
    return [{**provider, "channels": dict(provider["channels"])} for provider in providers]


VISUAL_INPUT_NAMES = tuple(VISUAL_PUBLICATION_INPUTS)


class _VisualPublication:
    """Boundary mixin: publish the module's VISUAL providers with its result.

    MISS: the compute node's providers (the difference between "VISUAL" and
    "VISUAL IN") are stored next to the result and published.  HIT: the stored
    providers are re-attached to "VISUAL IN" without requesting the compute
    node.  Disabled passthrough forwards "VISUAL IN"; blocked outputs block
    VISUAL as well.  The cores provide ``_module_disabled`` and
    ``_result_cached`` and exclude both VISUAL inputs from their cache key.
    """

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types.setdefault("optional", {}).update(VISUAL_PUBLICATION_INPUTS)
        return types

    def check_lazy_status(self, prompt=None, unique_id=None, **inputs):
        routed = _visual_routed(prompt, unique_id) and not stop_after_handoff_prompt_state(prompt, unique_id)
        if routed:
            key, _detail = self._cache_key(prompt, unique_id)
            if key and _disk_available(self._SCOPE, key) and not all(
                path.is_file() for path in _visual_entry_paths(self._SCOPE, key)
            ):
                # Stored before the boundary published VISUAL: recompute once.
                _remove_entry(self._SCOPE, key)
        core = {name: value for name, value in inputs.items() if name not in VISUAL_PUBLICATION_INPUTS}
        needed = list(super().check_lazy_status(prompt=prompt, unique_id=unique_id, **core))
        if not routed:
            return needed
        names = ["VISUAL IN"]
        if not self._module_disabled(prompt, unique_id) and not self._result_cached(prompt, unique_id):
            names.append("VISUAL")
        connected = _connected_inputs(prompt, unique_id)
        return needed + [
            name for name in names
            if name in connected and inputs.get(name) is None and name not in needed
        ]

    def boundary(self, prompt=None, unique_id=None, **inputs):
        core = {name: value for name, value in inputs.items() if name not in VISUAL_PUBLICATION_INPUTS}
        outputs = tuple(super().boundary(prompt=prompt, unique_id=unique_id, **core))
        if isinstance(outputs[1], ExecutionBlocker):
            return (*outputs, outputs[1])
        if not _visual_routed(prompt, unique_id):
            return (*outputs, None)
        visual_in = inputs.get("VISUAL IN")
        if self._module_disabled(prompt, unique_id):
            return (*outputs, cmk_visual.normalize_visual(visual_in))
        key, _detail = self._cache_key(prompt, unique_id)
        if inputs.get("VISUAL") is None:
            providers = _load_visual(self._SCOPE, key)
        else:
            providers = _module_providers(visual_in, inputs["VISUAL"])
            if key:
                _save_visual(self._SCOPE, key, providers)
        return (*outputs, cmk_visual.publish_providers(visual_in, providers))


# -- Disabled passthrough without the compute node (K-5) -------------------
# "IMAGE IN" / "LOG IN" carry the module's own inputs; "diagnostic" is the
# compute node's diagnostic, which the module definition now routes through
# the boundary instead of straight to the gate.  With every local branch OFF
# the boundary answers from these inputs alone: no Prepare, no compute, no
# model materialisation.  Older embedded definitions leave the inputs
# unconnected and keep the previous behaviour.
PASSTHROUGH_INPUTS = {
    "IMAGE IN": ("IMAGE", {"lazy": True}),
    "LOG IN": ("CMK_LOG_PIPE", {"lazy": True}),
    "diagnostic": ("CMK_DIAGNOSTIC", {"lazy": True}),
}
PASSTHROUGH_INPUT_NAMES = tuple(PASSTHROUGH_INPUTS)


class _PassthroughSources:
    """Boundary mixin: disabled passthrough from the module inputs and the
    boundary-owned diagnostic.  Appends ``diagnostic`` after VISUAL so the
    existing output slots keep their indices.  The class provides
    ``_PASSTHROUGH_LABEL`` and the core hooks used by ``_VisualPublication``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types.setdefault("optional", {}).update(PASSTHROUGH_INPUTS)
        return types

    def _routed(self, prompt, unique_id) -> bool:
        connected = _connected_inputs(prompt, unique_id)
        return "IMAGE IN" in connected and "LOG IN" in connected

    def _passthrough(self, prompt, unique_id) -> bool:
        return (
            self._routed(prompt, unique_id)
            and not stop_after_handoff_prompt_state(prompt, unique_id)
            and self._module_disabled(prompt, unique_id)
        )

    def check_lazy_status(self, prompt=None, unique_id=None, **inputs):
        own = {name: inputs.pop(name, None) for name in PASSTHROUGH_INPUT_NAMES}
        connected = _connected_inputs(prompt, unique_id)
        if self._passthrough(prompt, unique_id):
            write_status(self._SCOPE, "DISABLED_PASSTHROUGH_READY",
                         detail="module inputs; compute skipped", unique_id=unique_id)
            names = ["IMAGE IN", "LOG IN", "MODEL"]
            if _visual_routed(prompt, unique_id):
                names.append("VISUAL IN")
            values = {**inputs, **own}
            return [name for name in names if name in connected and values.get(name) is None]
        diagnostic_routed = "diagnostic" in connected and not stop_after_handoff_prompt_state(prompt, unique_id)
        if diagnostic_routed:
            key, _detail = self._cache_key(prompt, unique_id)
            if key and _disk_available(self._SCOPE, key) and not all(
                path.is_file() for path in _diagnostic_entry_paths(self._SCOPE, key)
            ):
                _remove_entry(self._SCOPE, key)
        needed = list(super().check_lazy_status(prompt=prompt, unique_id=unique_id, **inputs))
        if (
            diagnostic_routed
            and own["diagnostic"] is None
            and not self._module_disabled(prompt, unique_id)
            and not self._result_cached(prompt, unique_id)
        ):
            needed.append("diagnostic")
        return needed

    def boundary(self, prompt=None, unique_id=None, **inputs):
        own = {name: inputs.pop(name, None) for name in PASSTHROUGH_INPUT_NAMES}
        if self._passthrough(prompt, unique_id):
            return self._disabled_passthrough(prompt, unique_id, own, inputs)
        outputs = tuple(super().boundary(prompt=prompt, unique_id=unique_id, **inputs))
        if isinstance(outputs[1], ExecutionBlocker):
            return (*outputs, outputs[1])
        if "diagnostic" not in _connected_inputs(prompt, unique_id):
            return (*outputs, ExecutionBlocker(None))
        if self._module_disabled(prompt, unique_id):
            # Legacy passthrough through the compute node (IMAGE IN unrouted).
            return (*outputs, own["diagnostic"] if own["diagnostic"] is not None else ExecutionBlocker(None))
        key, _detail = self._cache_key(prompt, unique_id)
        if own["diagnostic"] is None:
            return (*outputs, _load_diagnostic(self._SCOPE, key))
        if key and _disk_available(self._SCOPE, key):
            _save_diagnostic(self._SCOPE, key, own["diagnostic"])
        return (*outputs, own["diagnostic"])

    def _disabled_passthrough(self, prompt, unique_id, own, inputs):
        from .cmk_log_pipe import cmk_add_block
        from ..utils.cmk_diagnostic import make_diagnostic_payload

        image, log = own["IMAGE IN"], own["LOG IN"]
        if image is None:
            raise RuntimeError(f"CMK Boundary Cache / {self._PASSTHROUGH_LABEL}: disabled passthrough requires IMAGE IN")
        if not isinstance(log, dict):
            raise TypeError(f"CMK Boundary Cache / {self._PASSTHROUGH_LABEL}: LOG IN is not a CMK log pipe")
        lines = [
            "STATUS          : DISABLED PASSTHROUGH",
            "REASON          : module enabled, all local branches OFF",
            "PREPARE/COMPUTE : SKIPPED",
            "CACHE           : SKIPPED",
        ]
        log_out = cmk_add_block(log, f"{self._PASSTHROUGH_LABEL} Boundary", 71, lines, True)
        diagnostic = make_diagnostic_payload(
            title=f"{self._PASSTHROUGH_LABEL} Boundary",
            node=f"CMK Boundary Cache / {self._PASSTHROUGH_LABEL}",
            previews=[image],
            summary="disabled passthrough | prepare and compute skipped",
            details="\n".join(lines),
            mode="disabled / passthrough",
            metadata={"global_enabled": True, "local_enabled": False},
        )
        visual = None
        if _visual_routed(prompt, unique_id):
            visual = cmk_visual.normalize_visual(inputs.get("VISUAL IN"))
        write_status(self._SCOPE, "DISABLED_PASSTHROUGH",
                     detail="module inputs; compute skipped", unique_id=unique_id)
        print(f"[CMK Boundary Cache / {self._PASSTHROUGH_LABEL}] DISABLED PASSTHROUGH -> PREPARE/COMPUTE SKIPPED")
        return (inputs.get("MODEL"), image, log_out, visual, diagnostic)


def _write_encoded(tensor_path, json_path, value, file_format: str, scope: str) -> None:
    from safetensors.torch import save_file  # type: ignore

    tensor_tmp = tensor_path.with_name(tensor_path.name + ".tmp")
    json_tmp = json_path.with_name(json_path.name + ".tmp")
    tensors: dict = {}
    encoded = _encode_diagnostic_value(value, tensors, [0], {})
    if not tensors:
        # safetensors requires at least one tensor. Diagnostics emitted by CMK
        # normally contain previews, but retain a harmless marker for text-only
        # payloads as well.
        import torch
        tensors["diagnostic_marker"] = torch.zeros((0,), dtype=torch.uint8)

    try:
        save_file(
            tensors,
            str(tensor_tmp),
            metadata={"format": file_format, "scope": scope},
        )
        with json_tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                encoded,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        os.replace(tensor_tmp, tensor_path)
        os.replace(json_tmp, json_path)
    finally:
        for path in (tensor_tmp, json_tmp):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def _read_encoded(tensor_path, json_path):
    from safetensors.torch import load_file  # type: ignore

    tensors = load_file(str(tensor_path), device="cpu")
    with json_path.open("r", encoding="utf-8") as handle:
        encoded = json.load(handle)
    value = _decode_diagnostic_value(encoded, tensors)
    os.utime(tensor_path, None)
    os.utime(json_path, None)
    return value


def _save_image_log(
    scope: str,
    cache_key: str,
    image,
    log_pipe: dict,
    dependencies: dict,
) -> None:
    from safetensors.torch import save_file  # type: ignore

    image_path, log_path, dependencies_path = _entry_paths(scope, cache_key)
    image_tmp = image_path.with_name(image_path.name + ".tmp")
    log_tmp = log_path.with_name(log_path.name + ".tmp")
    dependencies_tmp = dependencies_path.with_name(
        dependencies_path.name + ".tmp"
    )

    try:
        save_file(
            {
                "image": (
                    image.detach()
                    .to("cpu")
                    .contiguous()
                )
            },
            str(image_tmp),
            metadata={
                "format": "CMK_MODULE_BOUNDARY",
                "scope": scope,
            },
        )
        with log_tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(log_pipe),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        with dependencies_tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(dependencies),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        os.replace(image_tmp, image_path)
        os.replace(log_tmp, log_path)
        os.replace(dependencies_tmp, dependencies_path)
    finally:
        for path in (image_tmp, log_tmp, dependencies_tmp):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    prune(scope, ".safetensors", 16)
    prune(scope, ".json", 16)
    prune(scope, ".deps", 16)


def _load_image_log(scope: str, cache_key: str):
    from safetensors.torch import load_file  # type: ignore

    image_path, log_path, _dependencies_path = _entry_paths(scope, cache_key)
    try:
        tensors = load_file(str(image_path), device="cpu")
        image = tensors["image"]

        with log_path.open("r", encoding="utf-8") as handle:
            log_pipe = json.load(handle)

        if getattr(image, "ndim", None) != 4:
            raise ValueError("cached boundary IMAGE is invalid")
        if not isinstance(log_pipe, dict):
            raise TypeError("cached boundary LOG is invalid")

        os.utime(image_path, None)
        os.utime(log_path, None)
        return image, log_pipe
    except Exception:
        _remove_entry(scope, cache_key)
        raise


def _load_dependencies(scope: str, cache_key: str) -> dict:
    _image_path, _log_path, dependencies_path = _entry_paths(
        scope,
        cache_key,
    )
    try:
        with dependencies_path.open("r", encoding="utf-8") as handle:
            dependencies = json.load(handle)
        if not isinstance(dependencies, dict):
            raise TypeError("cached boundary dependency manifest is invalid")
        return dependencies
    except Exception:
        _remove_entry(scope, cache_key)
        raise


def _dependencies_match(
    scope: str,
    cache_key: str,
    current_dependencies: dict,
) -> bool:
    if not _disk_available(scope, cache_key):
        return False
    if not isinstance(current_dependencies, dict):
        return False
    if not bool(current_dependencies.get("complete", False)):
        return False

    try:
        stored_dependencies = _load_dependencies(scope, cache_key)
    except Exception:
        return False

    return stored_dependencies == current_dependencies


class _DetailerBoundaryCore:
    _SCOPE = "detailer_boundary"
    # v4 (K-5): entries carry the boundary diagnostic next to IMAGE/LOG/VISUAL.
    _SCHEMA = "cmk_detailer_boundary_v4"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "IMAGE": ("IMAGE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "IMAGE",
        "CMK_LOG_PIPE",
    )
    RETURN_NAMES = (
        "MODEL",
        "IMAGE",
        "LOG",
    )
    FUNCTION = "boundary"
    CATEGORY = "CMK/Internal/Boundary & Cache"
    DEV_ONLY = True

    def _cache_key(self, prompt, unique_id):
        return build_node_fingerprint(
            prompt,
            unique_id,
            ("CMKDetailerBoundaryCache",),
            self._SCHEMA,
            exclude_inputs=VISUAL_INPUT_NAMES + PASSTHROUGH_INPUT_NAMES,
            include_node_identity=True,
        )

    def _module_disabled(self, prompt, unique_id) -> bool:
        return _detailer_disabled_state(prompt, unique_id)[0]

    def _result_cached(self, prompt, unique_id) -> bool:
        cache_key, _detail = self._cache_key(prompt, unique_id)
        return self._cache_ready(cache_key, self._dependencies(prompt, unique_id)[0])

    def _dependencies(self, prompt, unique_id):
        return build_upstream_cache_manifest(
            prompt,
            unique_id,
            ("CMKDetailerBoundaryCache",),
            _DETAILER_BRANCH_SPECS,
            input_names=("IMAGE", "LOG"),
        )

    def _cache_ready(self, cache_key, dependencies) -> bool:
        # MODEL comes from the current input on a HIT (K5-B), so a valid disk
        # entry is reusable after eviction and restart without a session copy.
        return bool(cache_key) and _dependencies_match(self._SCOPE, cache_key, dependencies)

    @cmk_timed_call("LAZY 25 DETAILER BOUNDARY")
    def check_lazy_status(
        self,
        MODEL=None,
        IMAGE=None,
        LOG=None,
        prompt=None,
        unique_id=None,
    ):
        if _stop_after_handoff_blocks_module(
            prompt, unique_id, self._SCOPE, "Detailer"
        ):
            return []
        module_disabled, disabled_detail = _detailer_disabled_state(
            prompt,
            unique_id,
        )
        if module_disabled:
            write_status(
                self._SCOPE,
                "DISABLED_PASSTHROUGH_READY",
                detail=disabled_detail,
                unique_id=unique_id,
            )
            needed = []
            if MODEL is None:
                needed.append("MODEL")
            if IMAGE is None:
                needed.append("IMAGE")
            if LOG is None:
                needed.append("LOG")
            return needed

        cache_key, detail = self._cache_key(prompt, unique_id)
        dependencies, dependency_detail = self._dependencies(
            prompt,
            unique_id,
        )

        if self._cache_ready(cache_key, dependencies):
            write_status(
                self._SCOPE,
                "HIT_READY",
                cache_key=cache_key,
                detail=f"{detail}; {dependency_detail}",
                unique_id=unique_id,
            )
            return ["MODEL"] if MODEL is None else []

        if cache_key and _disk_available(self._SCOPE, cache_key):
            write_status(
                self._SCOPE,
                "STALE_DEPENDENCIES",
                cache_key=cache_key,
                detail=dependency_detail,
                unique_id=unique_id,
            )

        # Materialize the computed branch before requesting the public MODEL.
        # Requesting MODEL and IMAGE together lets ComfyUI load the upstream
        # SDXL/Refiner model path while sampling is still active, which can
        # exceed unified memory on constrained systems.
        branch_needed = []
        if IMAGE is None:
            branch_needed.append("IMAGE")
        if LOG is None:
            branch_needed.append("LOG")
        if branch_needed:
            return branch_needed
        if MODEL is None:
            return ["MODEL"]
        return []

    def boundary(
        self,
        MODEL=None,
        IMAGE=None,
        LOG=None,
        prompt=None,
        unique_id=None,
    ):
        if _stop_after_handoff_blocks_module(
            prompt, unique_id, self._SCOPE, "Detailer"
        ):
            return _blocked_outputs(3)
        module_disabled, disabled_detail = _detailer_disabled_state(
            prompt,
            unique_id,
        )
        if module_disabled:
            missing = [
                name for name, value in (
                    ("MODEL", MODEL), ("IMAGE", IMAGE), ("LOG", LOG),
                ) if value is None
            ]
            if missing:
                raise RuntimeError(
                    "CMK Boundary Cache / Detailer: disabled passthrough requires "
                    + ", ".join(missing)
                )
            if not isinstance(LOG, dict):
                raise TypeError("LOG is not a CMK log pipe")
            write_status(
                self._SCOPE,
                "DISABLED_PASSTHROUGH",
                detail=disabled_detail,
                unique_id=unique_id,
            )
            print(
                "[CMK Boundary Cache / Detailer] "
                "DISABLED PASSTHROUGH -> CACHE SKIPPED"
            )
            return MODEL, IMAGE, LOG

        cache_key, detail = self._cache_key(prompt, unique_id)
        dependencies, dependency_detail = self._dependencies(
            prompt,
            unique_id,
        )

        if self._cache_ready(cache_key, dependencies):
            try:
                cached_image, cached_log = _load_image_log(
                    self._SCOPE,
                    cache_key,
                )
                _retain_latest_boundary_entry(self._SCOPE, cache_key, unique_id)

                write_status(
                    self._SCOPE,
                    "HIT",
                    cache_key=cache_key,
                    detail=f"{detail}; {dependency_detail}",
                    unique_id=unique_id,
                )
                print(
                    "[CMK Boundary Cache / Detailer] HIT "
                    f"{cache_key[:12]}"
                )
                return (
                    MODEL,
                    cached_image,
                    cached_log,
                )
            except Exception as exc:
                write_status(
                    self._SCOPE,
                    "HIT_FAILED",
                    cache_key=cache_key,
                    detail=str(exc),
                    unique_id=unique_id,
                )

        missing = [
            name
            for name, value in (
                ("MODEL", MODEL),
                ("IMAGE", IMAGE),
                ("LOG", LOG),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "CMK Boundary Cache / Detailer: cache miss requires "
                + ", ".join(missing)
            )

        if not isinstance(MODEL, dict):
            raise TypeError("MODEL is not a CMK model pipe")
        if not isinstance(LOG, dict):
            raise TypeError("LOG is not a CMK log pipe")

        # Re-read the branch revisions after the lazy inputs have materialized.
        # This is the authoritative dependency state stored with the merged image.
        dependencies, dependency_detail = self._dependencies(
            prompt,
            unique_id,
        )

        if cache_key and bool(dependencies.get("complete", False)):
            try:
                _save_image_log(
                    self._SCOPE,
                    cache_key,
                    IMAGE,
                    LOG,
                    dependencies,
                )
                _retain_latest_boundary_entry(self._SCOPE, cache_key, unique_id)
                write_status(
                    self._SCOPE,
                    "MISS_STORED",
                    cache_key=cache_key,
                    detail=f"{detail}; {dependency_detail}",
                    unique_id=unique_id,
                )
                print(
                    "[CMK Boundary Cache / Detailer] MISS "
                    f"{cache_key[:12]} -> STORED"
                )
            except Exception as exc:
                write_status(
                    self._SCOPE,
                    "STORE_FAILED",
                    cache_key=cache_key,
                    detail=str(exc),
                    unique_id=unique_id,
                )
                print(
                    "[CMK Boundary Cache / Detailer] STORE FAILED: "
                    f"{exc}"
                )

        return MODEL, IMAGE, LOG


class CMKDetailerBoundaryCache(_PassthroughSources, _VisualPublication, _DetailerBoundaryCore):
    """Detailer boundary that also publishes the module's VISUAL providers and
    owns its diagnostic and disabled passthrough (K-5)."""

    _PASSTHROUGH_LABEL = "Detailer"

    RETURN_TYPES = (*_DetailerBoundaryCore.RETURN_TYPES, "CMK_VISUAL_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = (*_DetailerBoundaryCore.RETURN_NAMES, "VISUAL", "diagnostic")


class _FaceRebuildBoundaryCore:
    """Lazy persistent boundary for the complete InstantID FaceRebuild module."""

    _SCOPE = "face_rebuild_boundary"
    _SCHEMA = "cmk_face_rebuild_boundary_v1"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "IMAGE": ("IMAGE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "diagnostic": ("CMK_DIAGNOSTIC", {"lazy": True}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("MODEL", "IMAGE", "LOG", "diagnostic")
    FUNCTION = "boundary"
    CATEGORY = "CMK/Internal/Boundary & Cache"
    DEV_ONLY = True

    def _cache_key(self, prompt, unique_id):
        return build_node_fingerprint(
            prompt,
            unique_id,
            ("CMKFaceRebuildBoundaryCache",),
            self._SCHEMA,
            exclude_inputs=VISUAL_INPUT_NAMES,
            include_node_identity=True,
            extra_identity={"content_guard": content_guard_prompt_token(prompt)},
        )

    def _ready(self, key):
        return bool(key) and key in _FACE_REBUILD_SESSION and _disk_available(self._SCOPE, key)

    def _module_disabled(self, prompt, unique_id) -> bool:
        return _face_rebuild_disabled_state(prompt, unique_id)[0]

    def _result_cached(self, prompt, unique_id) -> bool:
        return self._ready(self._cache_key(prompt, unique_id)[0])

    @cmk_timed_call("LAZY 42 FACE REBUILD BOUNDARY")
    def check_lazy_status(
        self, MODEL=None, IMAGE=None, LOG=None, diagnostic=None,
        prompt=None, unique_id=None,
    ):
        if _stop_after_handoff_blocks_module(
            prompt, unique_id, self._SCOPE, "FaceRebuild"
        ):
            return []
        disabled, detail = _face_rebuild_disabled_state(prompt, unique_id)
        if disabled:
            write_status(
                self._SCOPE,
                "DISABLED_ACTIVE_BRANCH_BLOCKED",
                detail=detail,
                unique_id=unique_id,
            )
            return []

        key, detail = self._cache_key(prompt, unique_id)
        if self._ready(key):
            write_status(self._SCOPE, "HIT_READY", cache_key=key, detail=detail, unique_id=unique_id)
            return []

        branch = [
            name for name, value in (("IMAGE", IMAGE), ("LOG", LOG), ("diagnostic", diagnostic))
            if value is None
        ]
        if branch:
            return branch
        if MODEL is None:
            return ["MODEL"]
        return []

    def boundary(
        self, MODEL=None, IMAGE=None, LOG=None, diagnostic=None,
        prompt=None, unique_id=None,
    ):
        if _stop_after_handoff_blocks_module(
            prompt, unique_id, self._SCOPE, "FaceRebuild"
        ):
            return _blocked_outputs(4)
        disabled, disabled_detail = _face_rebuild_disabled_state(prompt, unique_id)
        if disabled:
            write_status(
                self._SCOPE,
                "DISABLED_ACTIVE_BRANCH_BLOCKED",
                detail=disabled_detail,
                unique_id=unique_id,
            )
            print("[CMK Boundary Cache / FaceRebuild] DISABLED -> ACTIVE BRANCH BLOCKED")
            blocked = ExecutionBlocker(None)
            return blocked, blocked, blocked, blocked

        key, detail = self._cache_key(prompt, unique_id)
        if self._ready(key):
            try:
                cached_image, cached_log = _load_image_log(self._SCOPE, key)
                cached_model, cached_diagnostic = _FACE_REBUILD_SESSION[key]
                _retain_latest_boundary_entry(
                    self._SCOPE, key, unique_id, _FACE_REBUILD_SESSION,
                )
                write_status(self._SCOPE, "HIT", cache_key=key, detail=detail, unique_id=unique_id)
                print(f"[CMK Boundary Cache / FaceRebuild] HIT {key[:12]}")
                return cached_model, cached_image, cached_log, cached_diagnostic
            except Exception as exc:
                _FACE_REBUILD_SESSION.pop(key, None)
                write_status(self._SCOPE, "HIT_FAILED", cache_key=key, detail=str(exc), unique_id=unique_id)

        missing = [
            name for name, value in (
                ("MODEL", MODEL), ("IMAGE", IMAGE), ("LOG", LOG), ("diagnostic", diagnostic),
            ) if value is None
        ]
        if missing:
            raise RuntimeError(
                "CMK Boundary Cache / FaceRebuild: cache miss requires " + ", ".join(missing)
            )
        if not isinstance(MODEL, dict) or not isinstance(LOG, dict):
            raise TypeError("CMK Boundary Cache / FaceRebuild received an invalid CMK contract")

        if key:
            try:
                _save_image_log(
                    self._SCOPE, key, IMAGE, LOG,
                    {"complete": True, "schema": self._SCHEMA},
                )
                _FACE_REBUILD_SESSION[key] = (MODEL, diagnostic)
                _retain_latest_boundary_entry(
                    self._SCOPE, key, unique_id, _FACE_REBUILD_SESSION,
                )
                write_status(self._SCOPE, "MISS_STORED", cache_key=key, detail=detail, unique_id=unique_id)
                print(f"[CMK Boundary Cache / FaceRebuild] MISS {key[:12]} -> STORED")
            except Exception as exc:
                write_status(self._SCOPE, "STORE_FAILED", cache_key=key, detail=str(exc), unique_id=unique_id)

        return MODEL, IMAGE, LOG, diagnostic


class CMKFaceRebuildBoundaryCache(_VisualPublication, _FaceRebuildBoundaryCore):
    """FaceRebuild boundary that also publishes the module's VISUAL providers."""

    RETURN_TYPES = (*_FaceRebuildBoundaryCore.RETURN_TYPES, "CMK_VISUAL_PIPE")
    RETURN_NAMES = (*_FaceRebuildBoundaryCore.RETURN_NAMES, "VISUAL")


class CMKZImageBoundaryCache:
    """Persistent decoded ZIT boundary for inexpensive downstream iteration."""

    _SCOPE = "z_image_boundary"
    _SCHEMA = "cmk_z_image_boundary_v2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "IMAGE": ("IMAGE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "diagnostic": ("CMK_DIAGNOSTIC", {"lazy": True}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("MODEL", "IMAGE", "LOG", "diagnostic")
    FUNCTION = "boundary"
    CATEGORY = "CMK/Internal/Boundary & Cache"
    DEV_ONLY = True

    def _cache_key(self, prompt, unique_id):
        return build_node_fingerprint(
            prompt, unique_id, ("CMKZImageBoundaryCache",), self._SCHEMA,
            include_node_identity=True,
        )

    def _ready(self, key):
        return bool(key) and key in _Z_IMAGE_SESSION and _disk_available(self._SCOPE, key)

    def check_lazy_status(self, MODEL=None, IMAGE=None, LOG=None, diagnostic=None, prompt=None, unique_id=None):
        key, detail = self._cache_key(prompt, unique_id)
        if self._ready(key):
            write_status(self._SCOPE, "HIT_READY", cache_key=key, detail=detail, unique_id=unique_id)
            return []
        return [name for name, value in (("MODEL", MODEL), ("IMAGE", IMAGE), ("LOG", LOG), ("diagnostic", diagnostic)) if value is None]

    def boundary(self, MODEL=None, IMAGE=None, LOG=None, diagnostic=None, prompt=None, unique_id=None):
        key, detail = self._cache_key(prompt, unique_id)
        if self._ready(key):
            try:
                image, log = _load_image_log(self._SCOPE, key)
                model, cached_diagnostic = _Z_IMAGE_SESSION[key]
                _retain_latest_boundary_entry(
                    self._SCOPE, key, unique_id, _Z_IMAGE_SESSION,
                )
                print(f"[CMK Boundary Cache / ZIT] HIT {key[:12]}")
                write_status(self._SCOPE, "HIT", cache_key=key, detail=detail, unique_id=unique_id)
                return model, image, log, cached_diagnostic
            except Exception:
                _Z_IMAGE_SESSION.pop(key, None)

        missing = [name for name, value in (("MODEL", MODEL), ("IMAGE", IMAGE), ("LOG", LOG), ("diagnostic", diagnostic)) if value is None]
        if missing:
            raise RuntimeError("CMK Boundary Cache / ZIT: cache miss requires " + ", ".join(missing))
        if not isinstance(MODEL, dict) or not isinstance(LOG, dict):
            raise TypeError("CMK Boundary Cache / ZIT received an invalid CMK contract")
        if key:
            _save_image_log(self._SCOPE, key, IMAGE, LOG, {"complete": True, "schema": self._SCHEMA})
            _Z_IMAGE_SESSION[key] = (MODEL, diagnostic)
            _retain_latest_boundary_entry(
                self._SCOPE, key, unique_id, _Z_IMAGE_SESSION,
            )
            print(f"[CMK Boundary Cache / ZIT] MISS {key[:12]} -> STORED")
        return MODEL, IMAGE, LOG, diagnostic


class _FaceBoundaryCore:
    _SCOPE = "faceprocess_boundary"
    _SCHEMA = "cmk_faceprocess_boundary_v5"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "IMAGE": ("IMAGE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "IMAGE",
        "CMK_LOG_PIPE",
    )
    RETURN_NAMES = ("MODEL", "IMAGE", "LOG")
    FUNCTION = "boundary"
    CATEGORY = "CMK/Internal/Boundary & Cache"
    DEV_ONLY = True

    def __init__(self):
        self._analysis_prompt_id = None
        self._analysis_unique_id = None
        self._analysis_result = None

    def _analysis(self, prompt, unique_id):
        prompt_id = id(prompt) if isinstance(prompt, dict) else None
        uid = str(unique_id)
        if (
            self._analysis_result is not None
            and self._analysis_prompt_id == prompt_id
            and self._analysis_unique_id == uid
        ):
            return self._analysis_result

        cache_key, detail = self._cache_key(prompt, unique_id)
        dependencies, dependency_detail = self._dependencies(
            prompt,
            unique_id,
        )
        self._analysis_prompt_id = prompt_id
        self._analysis_unique_id = uid
        self._analysis_result = (
            cache_key,
            detail,
            dependencies,
            dependency_detail,
        )
        return self._analysis_result

    def _cache_key(self, prompt, unique_id):
        return build_node_fingerprint(
            prompt,
            unique_id,
            ("CMKFaceBoundaryCache",),
            self._SCHEMA,
            exclude_inputs=("MODEL", "IMAGE", "LOG", *VISUAL_INPUT_NAMES, *PASSTHROUGH_INPUT_NAMES),
            include_node_identity=True,
        )

    def _module_disabled(self, prompt, unique_id) -> bool:
        return _faceprocess_disabled_state(prompt, unique_id)[0]

    def _result_cached(self, prompt, unique_id) -> bool:
        cache_key, _detail, dependencies, _dependency_detail = self._analysis(prompt, unique_id)
        return self._cache_ready(cache_key, dependencies)

    def _dependencies(self, prompt, unique_id):
        return build_upstream_cache_manifest(
            prompt,
            unique_id,
            ("CMKFaceBoundaryCache",),
            _FACEPROCESS_BRANCH_SPECS,
            input_names=("IMAGE", "LOG"),
        )

    def _cache_ready(self, cache_key, dependencies) -> bool:
        return (
            bool(cache_key)
            and _dependencies_match(
                self._SCOPE,
                cache_key,
                dependencies,
            )
        )

    @cmk_timed_call("LAZY 30 FACEPROCESS BOUNDARY")
    def check_lazy_status(
        self,
        MODEL=None,
        IMAGE=None,
        LOG=None,
        prompt=None,
        unique_id=None,
    ):
        context = _detailer_diag_context(unique_id)
        if _stop_after_handoff_blocks_module(
            prompt, unique_id, self._SCOPE, "FaceProcess"
        ):
            return []
        print(
            "[CMK DIAG DETAILER] FACE_BOUNDARY LAZY_ENTER "
            f"model={'PRESENT' if MODEL is not None else 'MISSING'} "
            f"image={'PRESENT' if IMAGE is not None else 'MISSING'} "
            f"log={'PRESENT' if LOG is not None else 'MISSING'} {context}"
        )
        module_disabled, disabled_detail = _faceprocess_disabled_state(
            prompt,
            unique_id,
        )
        if module_disabled:
            print(f"[CMK DIAG DETAILER] FACE_BOUNDARY DISABLED no_request {context}")
            write_status(
                self._SCOPE,
                "DISABLED_ACTIVE_BRANCH_BLOCKED",
                detail=disabled_detail,
                unique_id=unique_id,
            )
            return []

        (
            cache_key,
            detail,
            dependencies,
            dependency_detail,
        ) = self._analysis(prompt, unique_id)

        if self._cache_ready(cache_key, dependencies):
            write_status(
                self._SCOPE,
                "HIT_READY",
                cache_key=cache_key,
                detail=f"{detail}; {dependency_detail}",
                unique_id=unique_id,
            )
            needed = []
            if MODEL is None:
                needed.append("MODEL")
            print(
                "[CMK DIAG DETAILER] FACE_BOUNDARY HIT "
                f"request={','.join(needed) if needed else 'NONE'} "
                f"model_after_cached_result={'YES' if MODEL is None else 'NO'} "
                f"cache_key={cache_key[:12] if cache_key else 'UNAVAILABLE'} {context}"
            )
            return needed

        if cache_key and _disk_available(self._SCOPE, cache_key):
            write_status(
                self._SCOPE,
                "STALE_DEPENDENCIES",
                cache_key=cache_key,
                detail=dependency_detail,
                unique_id=unique_id,
            )

        needed = []
        if MODEL is None:
            needed.append("MODEL")
        if IMAGE is None:
            needed.append("IMAGE")
        if LOG is None:
            needed.append("LOG")
        print(
            "[CMK DIAG DETAILER] FACE_BOUNDARY MISS "
            f"request={','.join(needed) if needed else 'NONE'} "
            f"model_after_image_log={'YES' if MODEL is None and IMAGE is not None and LOG is not None else 'NO'} "
            f"cache_key={cache_key[:12] if cache_key else 'UNAVAILABLE'} {context}"
        )
        return needed

    def boundary(
        self,
        MODEL=None,
        IMAGE=None,
        LOG=None,
        prompt=None,
        unique_id=None,
    ):
        context = _detailer_diag_context(unique_id)
        if _stop_after_handoff_blocks_module(
            prompt, unique_id, self._SCOPE, "FaceProcess"
        ):
            return _blocked_outputs(3)
        print(
            "[CMK DIAG DETAILER] FACE_BOUNDARY EXECUTE "
            f"model={'PRESENT' if MODEL is not None else 'MISSING'} "
            f"image={'PRESENT' if IMAGE is not None else 'MISSING'} "
            f"log={'PRESENT' if LOG is not None else 'MISSING'} {context}"
        )
        module_disabled, disabled_detail = _faceprocess_disabled_state(
            prompt,
            unique_id,
        )
        if module_disabled:
            print(f"[CMK DIAG DETAILER] FACE_BOUNDARY EXECUTE_DISABLED {context}")
            write_status(
                self._SCOPE,
                "DISABLED_ACTIVE_BRANCH_BLOCKED",
                detail=disabled_detail,
                unique_id=unique_id,
            )
            print(
                "[CMK Boundary Cache / FaceProcess] "
                "DISABLED -> ACTIVE BRANCH BLOCKED"
            )
            blocked = ExecutionBlocker(None)
            return blocked, blocked, blocked

        cache_key, detail = self._cache_key(prompt, unique_id)
        dependencies, dependency_detail = self._dependencies(
            prompt,
            unique_id,
        )

        if self._cache_ready(cache_key, dependencies):
            try:
                cached_image, cached_log = _load_image_log(
                    self._SCOPE,
                    cache_key,
                )
                _retain_latest_boundary_entry(
                    self._SCOPE, cache_key, unique_id,
                )
                write_status(
                    self._SCOPE,
                    "HIT",
                    cache_key=cache_key,
                    detail=f"{detail}; {dependency_detail}",
                    unique_id=unique_id,
                )
                print(
                    "[CMK Boundary Cache / FaceProcess] HIT "
                    f"{cache_key[:12]}"
                )
                print(
                    "[CMK DIAG DETAILER] FACE_BOUNDARY EXECUTE_HIT "
                    f"cache_key={cache_key[:12]} returns_model_reference={'YES' if MODEL is not None else 'NO'} {context}"
                )
                return MODEL, cached_image, cached_log
            except Exception as exc:
                write_status(
                    self._SCOPE,
                    "HIT_FAILED",
                    cache_key=cache_key,
                    detail=str(exc),
                    unique_id=unique_id,
                )

        missing = [
            name
            for name, value in (
                ("MODEL", MODEL),
                ("IMAGE", IMAGE),
                ("LOG", LOG),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "CMK Boundary Cache / FaceProcess: cache miss requires "
                + ", ".join(missing)
            )

        if not isinstance(LOG, dict):
            raise TypeError("LOG is not a CMK log pipe")

        if cache_key and bool(dependencies.get("complete", False)):
            try:
                _save_image_log(
                    self._SCOPE,
                    cache_key,
                    IMAGE,
                    LOG,
                    dependencies,
                )
                _retain_latest_boundary_entry(
                    self._SCOPE, cache_key, unique_id,
                )
                write_status(
                    self._SCOPE,
                    "MISS_STORED",
                    cache_key=cache_key,
                    detail=f"{detail}; {dependency_detail}",
                    unique_id=unique_id,
                )
                print(
                    "[CMK Boundary Cache / FaceProcess] MISS "
                    f"{cache_key[:12]} -> STORED"
                )
                print(
                    "[CMK DIAG DETAILER] FACE_BOUNDARY EXECUTE_MISS "
                    f"stored=YES cache_key={cache_key[:12]} {context}"
                )
            except Exception as exc:
                write_status(
                    self._SCOPE,
                    "STORE_FAILED",
                    cache_key=cache_key,
                    detail=str(exc),
                    unique_id=unique_id,
                )
                print(
                    "[CMK Boundary Cache / FaceProcess] STORE FAILED: "
                    f"{exc}"
                )
                print(
                    "[CMK DIAG DETAILER] FACE_BOUNDARY EXECUTE_MISS "
                    f"stored=NO reason=STORE_FAILED {context}"
                )

        print(f"[CMK DIAG DETAILER] FACE_BOUNDARY RETURN live_result=YES {context}")
        return MODEL, IMAGE, LOG


class CMKFaceBoundaryCache(_PassthroughSources, _VisualPublication, _FaceBoundaryCore):
    """FaceProcess boundary that also publishes the module's VISUAL providers and
    owns its diagnostic and disabled passthrough (H-5, shared with the Detailer)."""

    _PASSTHROUGH_LABEL = "FaceProcess"

    RETURN_TYPES = (*_FaceBoundaryCore.RETURN_TYPES, "CMK_VISUAL_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = (*_FaceBoundaryCore.RETURN_NAMES, "VISUAL", "diagnostic")


def _faceswap_entry_paths(scope: str, cache_key: str):
    base = cache_directory(scope) / cache_key
    return (
        base.with_suffix(".safetensors"),
        base.with_suffix(".json"),
        *_diagnostic_entry_paths(scope, cache_key),
    )


def _faceswap_disk_available(scope: str, cache_key: str) -> bool:
    return all(path.is_file() for path in _faceswap_entry_paths(scope, cache_key))


class _FaceSwapBoundaryCore:
    """Persistent module boundary for the closed FaceSwap subgraph.

    MODEL is read-only and never serialized. IMAGE, LOG and the complete
    diagnostic payload are materialized together so every public module output
    and internal preview consumes the same authoritative boundary result.
    """

    _SCOPE = "faceswap_boundary"
    _SCHEMA = "cmk_faceswap_boundary_v4"
    _NODE_TYPE = "CMKFaceSwapBoundaryCache"
    _LABEL = "FaceSwap"

    def _disabled_state(self, prompt, unique_id):
        return _faceswap_disabled_state(prompt, unique_id)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "MODEL (opt)": ("CMK_MODEL_PIPE", {"lazy": True}),
                "IMAGE": ("IMAGE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "diagnostic": ("CMK_DIAGNOSTIC", {"lazy": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "IMAGE",
        "CMK_LOG_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = (
        "MODEL",
        "IMAGE",
        "LOG",
        "diagnostic",
    )
    FUNCTION = "boundary"
    CATEGORY = "CMK/Internal/Boundary & Cache"
    DEV_ONLY = True

    def _cache_key(self, prompt, unique_id):
        return build_node_fingerprint(
            prompt,
            unique_id,
            (self._NODE_TYPE,),
            self._SCHEMA,
            exclude_inputs=VISUAL_INPUT_NAMES,
            include_node_identity=True,
            extra_identity={"content_guard": content_guard_prompt_token(prompt)},
        )

    def _module_disabled(self, prompt, unique_id) -> bool:
        return self._disabled_state(prompt, unique_id)[0]

    def _result_cached(self, prompt, unique_id) -> bool:
        cache_key, _detail = self._cache_key(prompt, unique_id)
        return bool(cache_key) and _faceswap_disk_available(self._SCOPE, cache_key)

    @cmk_timed_call("LAZY 40 FACESWAP BOUNDARY")
    def check_lazy_status(
        self,
        IMAGE=None,
        LOG=None,
        diagnostic=None,
        prompt=None,
        unique_id=None,
        **kwargs,
    ):
        if _stop_after_handoff_blocks_module(
            prompt, unique_id, self._SCOPE, self._LABEL
        ):
            return []
        _, node = _resolve_boundary_node(
            prompt, unique_id, self._NODE_TYPE,
        )
        model_connected = "MODEL (opt)" in ((node or {}).get("inputs") or {})
        model_missing = model_connected and kwargs.get("MODEL (opt)") is None
        disabled, detail = self._disabled_state(prompt, unique_id)
        if disabled:
            needed = [
                name for name, value in (("IMAGE", IMAGE), ("LOG", LOG), ("diagnostic", diagnostic))
                if value is None
            ]
            if model_missing:
                needed.append("MODEL (opt)")
            return needed
        cache_key, detail = self._cache_key(prompt, unique_id)
        if cache_key and _faceswap_disk_available(self._SCOPE, cache_key):
            write_status(
                self._SCOPE,
                "HIT_READY",
                cache_key=cache_key,
                detail=detail,
                unique_id=unique_id,
            )
            return ["MODEL (opt)"] if model_missing else []

        needed = []
        for name, value in (
            ("IMAGE", IMAGE),
            ("LOG", LOG),
            ("diagnostic", diagnostic),
        ):
            if value is None:
                needed.append(name)
        if needed:
            return needed
        return ["MODEL (opt)"] if model_missing else []

    def boundary(
        self,
        IMAGE=None,
        LOG=None,
        diagnostic=None,
        prompt=None,
        unique_id=None,
        **kwargs,
    ):
        if _stop_after_handoff_blocks_module(
            prompt, unique_id, self._SCOPE, self._LABEL
        ):
            return _blocked_outputs(4)
        MODEL = kwargs.get("MODEL (opt)")
        disabled, detail = self._disabled_state(prompt, unique_id)
        if disabled:
            missing = [
                name for name, value in (("IMAGE", IMAGE), ("LOG", LOG), ("diagnostic", diagnostic))
                if value is None
            ]
            if missing:
                raise RuntimeError(f"CMK {self._LABEL} bypass requires " + ", ".join(missing))
            write_status(self._SCOPE, "DISABLED_PASSTHROUGH", detail=detail, unique_id=unique_id)
            print(f"[CMK Boundary Cache / {self._LABEL}] DISABLED PASSTHROUGH -> CACHE SKIPPED")
            return MODEL, IMAGE, LOG, diagnostic
        cache_key, detail = self._cache_key(prompt, unique_id)

        if (
            cache_key
            and _faceswap_disk_available(self._SCOPE, cache_key)
        ):
            try:
                cached_image, cached_log = _load_image_log(
                    self._SCOPE,
                    cache_key,
                )
                cached_diagnostic = _load_diagnostic(self._SCOPE, cache_key)
                _retain_latest_boundary_entry(
                    self._SCOPE, cache_key, unique_id,
                )
                write_status(
                    self._SCOPE,
                    "HIT",
                    cache_key=cache_key,
                    detail=detail,
                    unique_id=unique_id,
                )
                print(
                    f"[CMK Boundary Cache / {self._LABEL}] HIT "
                    f"{cache_key[:12]}"
                )
                return MODEL, cached_image, cached_log, cached_diagnostic
            except Exception as exc:
                write_status(
                    self._SCOPE,
                    "HIT_FAILED",
                    cache_key=cache_key,
                    detail=str(exc),
                    unique_id=unique_id,
                )

        missing = [
            name
            for name, value in (
                ("IMAGE", IMAGE),
                ("LOG", LOG),
                ("diagnostic", diagnostic),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                f"CMK Boundary Cache / {self._LABEL}: cache miss requires "
                + ", ".join(missing)
            )

        if MODEL is not None and not isinstance(MODEL, dict):
            raise TypeError("MODEL is not a CMK model pipe")
        if not isinstance(LOG, dict):
            raise TypeError("LOG is not a CMK log pipe")
        if not isinstance(diagnostic, dict) or diagnostic.get("type") != "CMK_DIAGNOSTIC":
            raise TypeError("diagnostic is not a CMK diagnostic payload")

        if cache_key:
            try:
                _save_image_log(
                    self._SCOPE,
                    cache_key,
                    IMAGE,
                    LOG,
                    {"complete": True, "schema": self._SCHEMA},
                )
                _save_diagnostic(self._SCOPE, cache_key, diagnostic)
                _retain_latest_boundary_entry(
                    self._SCOPE, cache_key, unique_id,
                )
                write_status(
                    self._SCOPE,
                    "MISS_STORED",
                    cache_key=cache_key,
                    detail=detail,
                    unique_id=unique_id,
                )
                print(
                    f"[CMK Boundary Cache / {self._LABEL}] MISS "
                    f"{cache_key[:12]} -> STORED"
                )
            except Exception as exc:
                write_status(
                    self._SCOPE,
                    "STORE_FAILED",
                    cache_key=cache_key,
                    detail=str(exc),
                    unique_id=unique_id,
                )
                print(
                    f"[CMK Boundary Cache / {self._LABEL}] STORE FAILED: "
                    f"{exc}"
                )

        return MODEL, IMAGE, LOG, diagnostic


class CMKFaceSwapBoundaryCache(_VisualPublication, _FaceSwapBoundaryCore):
    """FaceSwap boundary that also publishes the module's VISUAL providers."""

    RETURN_TYPES = (*_FaceSwapBoundaryCore.RETURN_TYPES, "CMK_VISUAL_PIPE")
    RETURN_NAMES = (*_FaceSwapBoundaryCore.RETURN_NAMES, "VISUAL")


class CMKUpscaleSaveBoundaryCache(_FaceSwapBoundaryCore):
    """Persistent side-branch boundary for Upscale & Save output."""

    _SCOPE = "upscale_save_boundary"
    _SCHEMA = "cmk_upscale_save_boundary_v1"
    _NODE_TYPE = "CMKUpscaleSaveBoundaryCache"
    _LABEL = "Upscale & Save"

    @staticmethod
    def _disabled_state(prompt, unique_id):
        return False, "side branch active"
