from __future__ import annotations

import json
import os

from .cmk_persistent_cache import (
    build_node_fingerprint,
    build_upstream_cache_manifest,
    cache_directory,
    prune,
    write_status,
)


_DETAILER_SESSION: dict[str, tuple[dict, dict]] = {}

_DETAILER_BRANCH_SPECS = {
    "CMK_SmartDetailerPipe": (
        "detailer_branch",
        "cmk_detailer_branch_v4",
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
    if key in visited:
        return
    visited.add(key)

    node = _prompt_node(prompt, node_id)
    if not isinstance(node, dict):
        return

    class_type = str(node.get("class_type", ""))
    if class_type in wanted:
        found.append((node_id, node))

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
    for path in _entry_paths(scope, cache_key):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


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
    _SCHEMA = "cmk_detailer_boundary_v2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS": ("CMK_PIPE", {"lazy": True}),
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
        "CMK_PIPE",
        "IMAGE",
        "CMK_LOG_PIPE",
    )
    RETURN_NAMES = (
        "MODEL",
        "PROCESS",
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

    def check_lazy_status(
        self,
        MODEL=None,
        PROCESS=None,
        IMAGE=None,
        LOG=None,
        prompt=None,
        unique_id=None,
    ):
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

        needed = []
        for name, value in (
            ("MODEL", MODEL),
            ("PROCESS", PROCESS),
            ("IMAGE", IMAGE),
            ("LOG", LOG),
        ):
            if value is None:
                needed.append(name)
        return needed

    def boundary(
        self,
        MODEL=None,
        PROCESS=None,
        IMAGE=None,
        LOG=None,
        prompt=None,
        unique_id=None,
    ):
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
                cached_model, cached_process = _DETAILER_SESSION[cache_key]

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
                    dict(cached_process),
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
                ("PROCESS", PROCESS),
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
        if not isinstance(PROCESS, dict):
            raise TypeError("PROCESS is not a CMK process pipe")
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
                _DETAILER_SESSION[cache_key] = (
                    MODEL,
                    dict(PROCESS),
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

        return MODEL, PROCESS, IMAGE, LOG


class CMKFaceBoundaryCache:
    _SCOPE = "faceprocess_boundary"
    _SCHEMA = "cmk_faceprocess_boundary_v4"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE": ("IMAGE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "CMK_LOG_PIPE")
    RETURN_NAMES = ("IMAGE", "LOG")
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
            exclude_inputs=("IMAGE", "LOG"),
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

    def check_lazy_status(
        self,
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
            return []

        if cache_key and _disk_available(self._SCOPE, cache_key):
            write_status(
                self._SCOPE,
                "STALE_DEPENDENCIES",
                cache_key=cache_key,
                detail=dependency_detail,
                unique_id=unique_id,
            )

        needed = []
        if IMAGE is None:
            needed.append("IMAGE")
        if LOG is None:
            needed.append("LOG")
        return needed

    def boundary(
        self,
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
                for name, value in (("IMAGE", IMAGE), ("LOG", LOG))
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
            return IMAGE, LOG

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
                return cached_image, cached_log
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

        return IMAGE, LOG


def _faceswap_entry_paths(scope: str, cache_key: str):
    base = cache_directory(scope) / cache_key
    return base.with_suffix(".safetensors"), base.with_suffix(".json")


def _faceswap_disk_available(scope: str, cache_key: str) -> bool:
    return all(path.is_file() for path in _faceswap_entry_paths(scope, cache_key))


def _remove_faceswap_entry(scope: str, cache_key: str) -> None:
    for path in _faceswap_entry_paths(scope, cache_key):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _save_faceswap_boundary(
    scope: str,
    cache_key: str,
    process_pipe: dict,
    image,
    log_pipe: dict,
) -> None:
    from safetensors.torch import save_file  # type: ignore

    image_path, state_path = _faceswap_entry_paths(scope, cache_key)
    image_tmp = image_path.with_name(image_path.name + ".tmp")
    state_tmp = state_path.with_name(state_path.name + ".tmp")

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
                "format": "CMK_FACESWAP_MODULE_BOUNDARY",
                "scope": scope,
            },
        )
        with state_tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "process": _json_safe(process_pipe),
                    "log": _json_safe(log_pipe),
                },
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        os.replace(image_tmp, image_path)
        os.replace(state_tmp, state_path)
    finally:
        for path in (image_tmp, state_tmp):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    prune(scope, ".safetensors", 16)
    prune(scope, ".json", 16)


def _load_faceswap_boundary(scope: str, cache_key: str):
    from safetensors.torch import load_file  # type: ignore

    image_path, state_path = _faceswap_entry_paths(scope, cache_key)
    try:
        tensors = load_file(str(image_path), device="cpu")
        image = tensors["image"]
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)

        process_pipe = state.get("process") if isinstance(state, dict) else None
        log_pipe = state.get("log") if isinstance(state, dict) else None
        if getattr(image, "ndim", None) != 4:
            raise ValueError("cached FaceSwap boundary IMAGE is invalid")
        if not isinstance(process_pipe, dict):
            raise TypeError("cached FaceSwap boundary PROCESS is invalid")
        if not isinstance(log_pipe, dict):
            raise TypeError("cached FaceSwap boundary LOG is invalid")

        os.utime(image_path, None)
        os.utime(state_path, None)
        return process_pipe, image, log_pipe
    except Exception:
        _remove_faceswap_entry(scope, cache_key)
        raise


class CMKFaceSwapBoundaryCache:
    """Persistent module boundary for the closed FaceSwap subgraph.

    MODEL is read-only and never serialized. PROCESS, IMAGE and LOG are
    materialized together, so every public module output and every internal
    preview can consume the same authoritative boundary result.
    """

    _SCOPE = "faceswap_boundary"
    _SCHEMA = "cmk_faceswap_boundary_v3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS": ("CMK_PIPE", {"lazy": True}),
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
        "CMK_PIPE",
        "IMAGE",
        "CMK_LOG_PIPE",
    )
    RETURN_NAMES = (
        "MODEL",
        "PROCESS",
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
            ("CMKFaceSwapBoundaryCache",),
            self._SCHEMA,
            include_node_identity=True,
        )

    def check_lazy_status(
        self,
        MODEL=None,
        PROCESS=None,
        IMAGE=None,
        LOG=None,
        prompt=None,
        unique_id=None,
    ):
        cache_key, detail = self._cache_key(prompt, unique_id)
        if cache_key and _faceswap_disk_available(self._SCOPE, cache_key):
            write_status(
                self._SCOPE,
                "HIT_READY",
                cache_key=cache_key,
                detail=detail,
                unique_id=unique_id,
            )
            return ["MODEL"] if MODEL is None else []

        needed = []
        for name, value in (
            ("MODEL", MODEL),
            ("PROCESS", PROCESS),
            ("IMAGE", IMAGE),
            ("LOG", LOG),
        ):
            if value is None:
                needed.append(name)
        return needed

    def boundary(
        self,
        MODEL=None,
        PROCESS=None,
        IMAGE=None,
        LOG=None,
        prompt=None,
        unique_id=None,
    ):
        cache_key, detail = self._cache_key(prompt, unique_id)

        if (
            cache_key
            and MODEL is not None
            and _faceswap_disk_available(self._SCOPE, cache_key)
        ):
            try:
                cached_process, cached_image, cached_log = _load_faceswap_boundary(
                    self._SCOPE,
                    cache_key,
                )
                write_status(
                    self._SCOPE,
                    "HIT",
                    cache_key=cache_key,
                    detail=detail,
                    unique_id=unique_id,
                )
                print(
                    "[CMK Boundary Cache / FaceSwap] HIT "
                    f"{cache_key[:12]}"
                )
                return MODEL, cached_process, cached_image, cached_log
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
                ("PROCESS", PROCESS),
                ("IMAGE", IMAGE),
                ("LOG", LOG),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "CMK Boundary Cache / FaceSwap: cache miss requires "
                + ", ".join(missing)
            )

        if not isinstance(MODEL, dict):
            raise TypeError("MODEL is not a CMK model pipe")
        if not isinstance(PROCESS, dict):
            raise TypeError("PROCESS is not a CMK process pipe")
        if not isinstance(LOG, dict):
            raise TypeError("LOG is not a CMK log pipe")

        if cache_key:
            try:
                _save_faceswap_boundary(
                    self._SCOPE,
                    cache_key,
                    PROCESS,
                    IMAGE,
                    LOG,
                )
                write_status(
                    self._SCOPE,
                    "MISS_STORED",
                    cache_key=cache_key,
                    detail=detail,
                    unique_id=unique_id,
                )
                print(
                    "[CMK Boundary Cache / FaceSwap] MISS "
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
                    "[CMK Boundary Cache / FaceSwap] STORE FAILED: "
                    f"{exc}"
                )

        return MODEL, PROCESS, IMAGE, LOG
