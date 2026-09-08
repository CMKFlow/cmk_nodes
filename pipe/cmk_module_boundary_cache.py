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
from ..utils.cmk_timing import cmk_timed_call


_DETAILER_SESSION: dict[str, dict] = {}
_Z_IMAGE_SESSION: dict[str, tuple[dict, object]] = {}
_FACE_REBUILD_SESSION: dict[str, tuple[dict, object]] = {}

_DETAILER_BRANCH_SPECS = {
    "CMK_SmartDetailerPipe": (
        "detailer_branch",
        "cmk_detailer_branch_v6",
    ),
}

_FACEPROCESS_BRANCH_SPECS = {
    "CMKFaceProcessPipe": (
        "faceprocess_branch",
        "cmk_faceprocess_branch_v4",
    ),
}



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
    for path in (*_entry_paths(scope, cache_key), *_diagnostic_entry_paths(scope, cache_key)):
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


def _encode_diagnostic_value(value, tensors: dict, counter: list[int]):
    """Move tensor/ndarray payloads into safetensors and retain structure in JSON."""
    try:
        import numpy as np
        import torch

        if isinstance(value, (torch.Tensor, np.ndarray)):
            key = f"diagnostic_{counter[0]:04d}"
            counter[0] += 1
            tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
            tensors[key] = tensor.detach().to("cpu").contiguous()
            return {"__cmk_diagnostic_tensor__": key}
    except Exception:
        pass

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _encode_diagnostic_value(item, tensors, counter)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _encode_diagnostic_value(item, tensors, counter)
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
    from safetensors.torch import save_file  # type: ignore

    tensor_path, json_path = _diagnostic_entry_paths(scope, cache_key)
    tensor_tmp = tensor_path.with_name(tensor_path.name + ".tmp")
    json_tmp = json_path.with_name(json_path.name + ".tmp")
    tensors: dict = {}
    encoded = _encode_diagnostic_value(diagnostic, tensors, [0])
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
            metadata={"format": "CMK_DIAGNOSTIC_BOUNDARY", "scope": scope},
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

    prune(scope, ".diagnostic.safetensors", 16)
    prune(scope, ".diagnostic.json", 16)


def _load_diagnostic(scope: str, cache_key: str):
    from safetensors.torch import load_file  # type: ignore

    tensor_path, json_path = _diagnostic_entry_paths(scope, cache_key)
    tensors = load_file(str(tensor_path), device="cpu")
    with json_path.open("r", encoding="utf-8") as handle:
        encoded = json.load(handle)
    diagnostic = _decode_diagnostic_value(encoded, tensors)
    if not isinstance(diagnostic, dict) or diagnostic.get("type") != "CMK_DIAGNOSTIC":
        raise TypeError("cached boundary diagnostic is invalid")
    os.utime(tensor_path, None)
    os.utime(json_path, None)
    return diagnostic


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


class CMKDetailerBoundaryCache:
    _SCOPE = "detailer_boundary"
    _SCHEMA = "cmk_detailer_boundary_v3"

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
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    def _cache_key(self, prompt, unique_id):
        return build_node_fingerprint(
            prompt,
            unique_id,
            ("CMKDetailerBoundaryCache",),
            self._SCHEMA,
            include_node_identity=True,
        )

    def _dependencies(self, prompt, unique_id):
        return build_upstream_cache_manifest(
            prompt,
            unique_id,
            ("CMKDetailerBoundaryCache",),
            _DETAILER_BRANCH_SPECS,
            input_names=("IMAGE", "LOG"),
        )

    def _cache_ready(self, cache_key, dependencies) -> bool:
        return (
            bool(cache_key)
            and cache_key in _DETAILER_SESSION
            and _dependencies_match(
                self._SCOPE,
                cache_key,
                dependencies,
            )
        )

    @cmk_timed_call("LAZY 25 DETAILER BOUNDARY")
    def check_lazy_status(
        self,
        MODEL=None,
        IMAGE=None,
        LOG=None,
        prompt=None,
        unique_id=None,
    ):
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
            return []

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
                cached_model = _DETAILER_SESSION[cache_key]
                _retain_latest_boundary_entry(
                    self._SCOPE, cache_key, unique_id, _DETAILER_SESSION,
                )

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
                    cached_model,
                    cached_image,
                    cached_log,
                )
            except Exception as exc:
                _DETAILER_SESSION.pop(cache_key, None)
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
                _DETAILER_SESSION[cache_key] = MODEL
                _retain_latest_boundary_entry(
                    self._SCOPE,
                    cache_key,
                    unique_id,
                    _DETAILER_SESSION,
                )
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


class CMKFaceRebuildBoundaryCache:
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
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    def _cache_key(self, prompt, unique_id):
        return build_node_fingerprint(
            prompt,
            unique_id,
            ("CMKFaceRebuildBoundaryCache",),
            self._SCHEMA,
            include_node_identity=True,
        )

    def _ready(self, key):
        return bool(key) and key in _FACE_REBUILD_SESSION and _disk_available(self._SCOPE, key)

    @cmk_timed_call("LAZY 42 FACE REBUILD BOUNDARY")
    def check_lazy_status(
        self, MODEL=None, IMAGE=None, LOG=None, diagnostic=None,
        prompt=None, unique_id=None,
    ):
        disabled, detail = _face_rebuild_disabled_state(prompt, unique_id)
        if disabled:
            write_status(self._SCOPE, "DISABLED_PASSTHROUGH_READY", detail=detail, unique_id=unique_id)
            return [
                name for name, value in (
                    ("MODEL", MODEL), ("IMAGE", IMAGE), ("LOG", LOG), ("diagnostic", diagnostic),
                ) if value is None
            ]

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
        disabled, disabled_detail = _face_rebuild_disabled_state(prompt, unique_id)
        if disabled:
            missing = [
                name for name, value in (
                    ("MODEL", MODEL), ("IMAGE", IMAGE), ("LOG", LOG), ("diagnostic", diagnostic),
                ) if value is None
            ]
            if missing:
                raise RuntimeError(
                    "CMK Boundary Cache / FaceRebuild: disabled passthrough requires "
                    + ", ".join(missing)
                )
            write_status(
                self._SCOPE, "DISABLED_PASSTHROUGH", detail=disabled_detail, unique_id=unique_id,
            )
            print("[CMK Boundary Cache / FaceRebuild] DISABLED PASSTHROUGH -> CACHE SKIPPED")
            return MODEL, IMAGE, LOG, diagnostic

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
    CATEGORY = "CMK/Developer/Boundary & Cache"
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


class CMKFaceBoundaryCache:
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
    CATEGORY = "CMK/Developer/Boundary & Cache"
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
            exclude_inputs=("MODEL", "IMAGE", "LOG"),
            include_node_identity=True,
        )

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
        module_disabled, disabled_detail = _faceprocess_disabled_state(
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
        return needed

    def boundary(
        self,
        MODEL=None,
        IMAGE=None,
        LOG=None,
        prompt=None,
        unique_id=None,
    ):
        module_disabled, disabled_detail = _faceprocess_disabled_state(
            prompt,
            unique_id,
        )
        if module_disabled:
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
                    "CMK Boundary Cache / FaceProcess: disabled passthrough "
                    "requires " + ", ".join(missing)
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
                "[CMK Boundary Cache / FaceProcess] "
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

        return MODEL, IMAGE, LOG


def _faceswap_entry_paths(scope: str, cache_key: str):
    base = cache_directory(scope) / cache_key
    return (
        base.with_suffix(".safetensors"),
        base.with_suffix(".json"),
        *_diagnostic_entry_paths(scope, cache_key),
    )


def _faceswap_disk_available(scope: str, cache_key: str) -> bool:
    return all(path.is_file() for path in _faceswap_entry_paths(scope, cache_key))


class CMKFaceSwapBoundaryCache:
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
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    def _cache_key(self, prompt, unique_id):
        return build_node_fingerprint(
            prompt,
            unique_id,
            (self._NODE_TYPE,),
            self._SCHEMA,
            include_node_identity=True,
        )

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


class CMKUpscaleSaveBoundaryCache(CMKFaceSwapBoundaryCache):
    """Persistent side-branch boundary for module 90 upscale and save output."""

    _SCOPE = "upscale_save_boundary"
    _SCHEMA = "cmk_upscale_save_boundary_v1"
    _NODE_TYPE = "CMKUpscaleSaveBoundaryCache"
    _LABEL = "Upscale & Save"

    @staticmethod
    def _disabled_state(prompt, unique_id):
        return False, "side branch active"
