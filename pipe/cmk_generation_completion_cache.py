from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .cmk_persistent_cache import build_node_fingerprint
from .cmk_prompt_result_state import UnsafeSnapshotTypeError, snapshot_value


CACHE_SCHEMA_VERSION = "cmk_generation_completion_v1"
STORAGE_FORMAT_VERSION = 1
CACHE_SCOPE = "generation_completion"
DEFAULT_MAX_ENTRIES = 16

_SUPPORTED_FAMILIES = {"sdxl", "z_image_turbo", "hybrid"}
_PROCESS_RESOURCE_FIELDS = {
    "clip",
    "clip_base",
    "clip_patched",
    "conditioning_identity_neg",
    "conditioning_identity_pos",
    "conditioning_neg",
    "conditioning_pos",
    "control_net",
    "controlnet",
    "latent",
    "latent_image",
    "lora_stack",
    "model",
    "model_base",
    "model_identity",
    "model_patched",
    "model_patcher",
    "resource_state",
    "sampled",
    "sampler",
    "samples",
    "vae",
}
_LOCK = threading.RLock()
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def build_completion_key(
    prompt: Any,
    generation_roots: Mapping[str, Any],
    *,
    model_family: str,
    generation_mode: str,
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> tuple[str | None, str]:
    """Build a consumer-independent key from expanded generation roots."""
    if not isinstance(prompt, dict):
        return None, "PROMPT is unavailable"
    if not isinstance(generation_roots, Mapping) or not generation_roots:
        return None, "generation roots are unavailable"

    family = str(model_family or "").strip().lower()
    mode = str(generation_mode or "").strip().lower()
    schema = str(schema_version or "").strip()
    if family not in _SUPPORTED_FAMILIES:
        return None, f"unsupported generation family {family!r}"
    if not mode:
        return None, "generation mode is unavailable"
    if not schema:
        return None, "cache schema is unavailable"

    roots = []
    for role, link in sorted(generation_roots.items(), key=lambda item: str(item[0])):
        role_name = str(role or "").strip()
        if not role_name:
            return None, "generation root role is empty"
        if not _is_prompt_link(prompt, link):
            return None, f"generation root {role_name!r} is not a connected prompt link"

        node_id, output_index = link
        node = _prompt_node(prompt, node_id)
        class_type = str(node.get("class_type", "")).strip()
        if not class_type:
            return None, f"generation root {role_name!r} has no class type"
        fingerprint, detail = build_node_fingerprint(
            prompt,
            node_id,
            (class_type,),
            f"{schema}:root:{role_name}",
            include_node_identity=True,
            extra_identity={
                "output_index": int(output_index),
                "output_role": role_name,
            },
        )
        if not fingerprint:
            return None, f"generation root {role_name!r}: {detail}"
        roots.append(
            {
                "role": role_name,
                "node_id": str(node_id),
                "class_type": class_type,
                "output_index": int(output_index),
                "fingerprint": fingerprint,
            }
        )

    identity = {
        "schema_version": schema,
        "model_family": family,
        "generation_mode": mode,
        "roots": roots,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()
    detail = ", ".join(
        f"{root['role']}={root['node_id']}:{root['output_index']}"
        for root in roots
    )
    return cache_key, detail


def lookup_completion(cache_key: str) -> dict[str, Any] | None:
    """Return one complete resource-free entry, or ``None`` for any invalid entry."""
    key = _validated_cache_key(cache_key)
    with _LOCK:
        tensor_path, manifest_path = _entry_paths(key)
        if not tensor_path.is_file() or not manifest_path.is_file():
            return None
        try:
            from safetensors.torch import load_file  # type: ignore

            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            _validate_manifest(manifest, key)
            tensors = load_file(str(tensor_path), device="cpu")
            expected_tensors = set(manifest["tensor_keys"])
            if set(tensors) != expected_tensors:
                raise ValueError("completion tensor manifest does not match its payload")
            payload = _decode_value(manifest["payload"], tensors, {})
            result = _validate_completion_payload(
                payload,
                key,
                expected_schema=manifest["cache_schema_version"],
            )
            os.utime(tensor_path, None)
            os.utime(manifest_path, None)
            return snapshot_value(result)
        except Exception:
            _remove_entry(key)
            return None


def store_completion(
    cache_key: str,
    *,
    image: Any,
    process: Mapping[str, Any],
    log: Mapping[str, Any],
    visual: Mapping[str, Any],
    model_family: str,
    generation_mode: str,
    fingerprint: str,
    metadata: Mapping[str, Any] | None = None,
    model_descriptor: Mapping[str, Any] | None = None,
    schema_version: str = CACHE_SCHEMA_VERSION,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> Path:
    """Atomically publish a complete generation result after strict validation."""
    key = _validated_cache_key(cache_key)
    if str(fingerprint or "").strip() != key:
        raise ValueError("completion fingerprint must equal the cache key")
    family = str(model_family or "").strip().lower()
    mode = str(generation_mode or "").strip().lower()
    schema = str(schema_version or "").strip()
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported generation family {family!r}")
    if not mode or not schema:
        raise ValueError("generation mode and cache schema must not be empty")
    if not isinstance(image, (torch.Tensor, np.ndarray)):
        raise UnsafeSnapshotTypeError("completion IMAGE must be a tensor or NumPy array")
    if not all(isinstance(value, Mapping) for value in (process, log, visual)):
        raise UnsafeSnapshotTypeError("completion PROCESS, LOG and VISUAL must be dictionaries")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise UnsafeSnapshotTypeError("completion metadata must be a dictionary")

    resource_free_process = _resource_free_process(process)
    descriptor = _validated_model_descriptor(model_descriptor)
    payload = snapshot_value(
        {
            "schema_version": schema,
            "fingerprint": key,
            "model_family": family,
            "generation_mode": mode,
            "image": image,
            "process": resource_free_process,
            "log": dict(log),
            "visual": dict(visual),
            "metadata": dict(metadata or {}),
            "model_descriptor": descriptor,
        }
    )
    _validate_completion_payload(payload, key, expected_schema=schema)
    tensors: dict[str, torch.Tensor] = {}
    encoded_payload = _encode_value(payload, tensors, {})
    manifest = {
        "storage_format_version": STORAGE_FORMAT_VERSION,
        "cache_schema_version": schema,
        "cache_key": key,
        "complete": True,
        "tensor_keys": sorted(tensors),
        "payload": encoded_payload,
    }

    with _LOCK:
        from safetensors.torch import save_file  # type: ignore

        tensor_path, manifest_path = _entry_paths(key)
        tensor_tmp = tensor_path.with_name(tensor_path.name + ".tmp")
        manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
        try:
            save_file(tensors, str(tensor_tmp))
            with manifest_tmp.open("w", encoding="utf-8") as handle:
                json.dump(
                    manifest,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.flush()
                os.fsync(handle.fileno())

            manifest_path.unlink(missing_ok=True)
            os.replace(tensor_tmp, tensor_path)
            os.replace(manifest_tmp, manifest_path)
        finally:
            tensor_tmp.unlink(missing_ok=True)
            manifest_tmp.unlink(missing_ok=True)
        cleanup_completion_cache(max_entries=max_entries)
        return manifest_path


def invalidate_completion(cache_key: str | None = None) -> int:
    """Remove one completion entry, or all entries when no key is supplied."""
    with _LOCK:
        if cache_key is not None:
            key = _validated_cache_key(cache_key)
            return int(_remove_entry(key))

        directory = _cache_directory()
        keys = {
            path.name.split(".", 1)[0]
            for path in directory.iterdir()
            if (path.is_symlink() or path.is_file()) and path.name != "last_status.json"
        }
        removed = 0
        for key in keys:
            if _is_cache_key(key):
                removed += int(_remove_entry(key))
        for path in directory.glob("*.tmp"):
            path.unlink(missing_ok=True)
        return removed


def completion_cache_inventory() -> dict[str, int]:
    """Return a lock-consistent inventory without opening cache payloads."""
    with _LOCK:
        directory = _cache_directory()
        files = [
            path
            for path in directory.iterdir()
            if (path.is_symlink() or path.is_file()) and path.name != "last_status.json"
        ]
        keys = {
            path.name.split(".", 1)[0]
            for path in files
            if _is_cache_key(path.name.split(".", 1)[0])
        }
        return {
            "entries": len(keys),
            "files": len(files),
            "bytes": sum(path.lstat().st_size for path in files),
        }


def completion_cache_statistics_inventory(
    package_root: Path | str | None = None,
) -> dict[str, int]:
    """Count complete entries and files without creating or opening payloads."""
    root = Path(package_root) if package_root is not None else _PACKAGE_ROOT
    directory = root / "cache" / CACHE_SCOPE
    if directory.is_symlink():
        raise OSError(f"refusing symbolic-link completion cache: {directory}")
    if not directory.exists():
        return {"entries": 0, "files": 0, "bytes": 0}
    if not directory.is_dir():
        raise OSError(f"completion cache is not a directory: {directory}")
    try:
        paths = list(directory.iterdir())
    except FileNotFoundError:
        return {"entries": 0, "files": 0, "bytes": 0}
    files = []
    suffixes: dict[str, set[str]] = {}
    for path in paths:
        if path.name == "last_status.json":
            continue
        try:
            if path.is_symlink():
                files.append(path.lstat().st_size)
                continue
            if not path.is_file():
                continue
            size = path.stat(follow_symlinks=False).st_size
        except FileNotFoundError:
            continue
        files.append(size)
        key = path.name.split(".", 1)[0]
        if _is_cache_key(key):
            suffixes.setdefault(key, set()).add(path.name[len(key):])
    complete = {
        key for key, present in suffixes.items()
        if {".safetensors", ".json"}.issubset(present)
    }
    return {
        "entries": len(complete),
        "files": len(files),
        "bytes": sum(files),
    }


def cleanup_completion_cache(max_entries: int = DEFAULT_MAX_ENTRIES) -> int:
    """Remove incomplete entries and retain only the newest bounded set."""
    limit = max(1, int(max_entries))
    with _LOCK:
        directory = _cache_directory()
        removed = 0
        valid_entries = []
        keys = {
            path.name.split(".", 1)[0]
            for path in directory.iterdir()
            if path.is_file() and path.name != "last_status.json"
        }
        for key in keys:
            if not _is_cache_key(key):
                continue
            tensor_path, manifest_path = _entry_paths(key)
            if not tensor_path.is_file() or not manifest_path.is_file():
                removed += int(_remove_entry(key))
                continue
            valid_entries.append(
                (max(tensor_path.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns), key)
            )

        valid_entries.sort(reverse=True)
        for _, key in valid_entries[limit:]:
            removed += int(_remove_entry(key))
        for path in directory.glob("*.tmp"):
            path.unlink(missing_ok=True)
        return removed


def _prompt_node(prompt: dict, node_id: Any) -> dict | None:
    node = prompt.get(node_id)
    if not isinstance(node, dict):
        node = prompt.get(str(node_id))
    return node if isinstance(node, dict) else None


def _is_prompt_link(prompt: dict, value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
        and _prompt_node(prompt, value[0]) is not None
    )


def _validated_cache_key(cache_key: str) -> str:
    key = str(cache_key or "").strip().lower()
    if not _is_cache_key(key):
        raise ValueError("completion cache key must be a SHA-256 hexadecimal digest")
    return key


def _is_cache_key(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _entry_paths(cache_key: str) -> tuple[Path, Path]:
    base = _cache_directory() / cache_key
    return base.with_suffix(".safetensors"), base.with_suffix(".json")


def _cache_directory() -> Path:
    directory = _PACKAGE_ROOT / "cache" / CACHE_SCOPE
    if directory.is_symlink():
        raise RuntimeError(
            f"CMK Generation Completion Cache refuses symbolic-link path: {directory}"
        )
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"CMK Generation Completion Cache cannot create {directory}: {exc}"
        ) from exc
    if not directory.is_dir():
        raise RuntimeError(
            f"CMK Generation Completion Cache path is not a directory: {directory}"
        )
    return directory


def _remove_entry(cache_key: str) -> bool:
    removed = False
    for path in _entry_paths(cache_key):
        if path.is_symlink() or path.exists():
            path.unlink(missing_ok=True)
            removed = True
    return removed


def _resource_free_process(process: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in process.items()
        if not _is_resource_field(key)
    }
    _reject_resource_fields(result, "PROCESS")
    return result


def _reject_resource_fields(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _is_resource_field(key):
                raise UnsafeSnapshotTypeError(
                    f"{path} contains forbidden resource field {key!r}"
                )
            _reject_resource_fields(item, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_resource_fields(item, f"{path}[{index}]")


def _is_resource_field(value: Any) -> bool:
    name = str(value).strip().lower()
    return (
        name in _PROCESS_RESOURCE_FIELDS
        or "conditioning" in name
        or "latent" in name
        or name.startswith("sampler_")
        or name.endswith("_sampler")
    )


def _validated_model_descriptor(value: Mapping[str, Any] | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise UnsafeSnapshotTypeError("MODEL descriptor must be a dictionary")
    from .cmk_family_result import _released_generation_model_descriptor

    descriptor = _released_generation_model_descriptor(dict(value))
    if descriptor is None:
        raise UnsafeSnapshotTypeError(
            "MODEL descriptor does not prove completed resource release"
        )
    return snapshot_value(descriptor)


def _validate_manifest(manifest: Any, cache_key: str) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("completion manifest is not a dictionary")
    if manifest.get("storage_format_version") != STORAGE_FORMAT_VERSION:
        raise ValueError("unsupported completion storage format")
    if manifest.get("cache_key") != cache_key or manifest.get("complete") is not True:
        raise ValueError("completion manifest is incomplete or belongs to another key")
    if not isinstance(manifest.get("cache_schema_version"), str):
        raise ValueError("completion manifest has no cache schema")
    tensor_keys = manifest.get("tensor_keys")
    if not isinstance(tensor_keys, list) or any(not isinstance(key, str) for key in tensor_keys):
        raise ValueError("completion manifest has invalid tensor keys")
    if "payload" not in manifest:
        raise ValueError("completion manifest has no payload")


def _validate_completion_payload(
    payload: Any,
    cache_key: str,
    *,
    expected_schema: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("completion payload is not a dictionary")
    expected = {
        "fingerprint",
        "generation_mode",
        "image",
        "log",
        "metadata",
        "model_descriptor",
        "model_family",
        "process",
        "schema_version",
        "visual",
    }
    if set(payload) != expected:
        raise ValueError("completion payload fields do not match the storage contract")
    if payload["fingerprint"] != cache_key:
        raise ValueError("completion payload fingerprint does not match its cache key")
    if payload["model_family"] not in _SUPPORTED_FAMILIES:
        raise ValueError("completion payload has an unsupported model family")
    if not isinstance(payload["generation_mode"], str) or not payload["generation_mode"]:
        raise ValueError("completion payload has no generation mode")
    if payload["schema_version"] != expected_schema:
        raise ValueError("completion payload and manifest schemas do not match")
    if not isinstance(payload["image"], (torch.Tensor, np.ndarray)):
        raise ValueError("completion payload has no valid IMAGE")
    if not all(
        isinstance(payload[name], dict)
        for name in ("process", "log", "visual", "metadata")
    ):
        raise ValueError("completion payload contains invalid structured results")
    _reject_resource_fields(payload["process"], "PROCESS")
    descriptor = payload["model_descriptor"]
    if descriptor is not None:
        _validated_model_descriptor(descriptor)
    return payload


def _encode_value(value: Any, tensors: dict[str, torch.Tensor], memo: dict[int, str]) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnsafeSnapshotTypeError("completion contains a non-finite float")
        return value
    if isinstance(value, bytes):
        return {"__cmk_type__": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (torch.Tensor, np.ndarray)):
        object_id = id(value)
        tensor_key = memo.get(object_id)
        if tensor_key is None:
            tensor_key = f"tensor_{len(tensors):06d}"
            if isinstance(value, np.ndarray):
                try:
                    tensor = torch.from_numpy(np.ascontiguousarray(value))
                except (TypeError, ValueError) as exc:
                    raise UnsafeSnapshotTypeError(
                        f"completion contains unsupported NumPy dtype {value.dtype}"
                    ) from exc
                kind = "numpy"
            else:
                if value.layout != torch.strided or value.is_quantized:
                    raise UnsafeSnapshotTypeError(
                        "completion contains a non-strided or quantized tensor"
                    )
                tensor = value
                kind = "torch"
            tensors[tensor_key] = tensor.detach().to(device="cpu").contiguous()
            memo[object_id] = tensor_key
        else:
            kind = "numpy" if isinstance(value, np.ndarray) else "torch"
        return {"__cmk_type__": kind, "key": tensor_key}
    if isinstance(value, dict):
        return {
            "__cmk_type__": "dict",
            "items": [
                [_encode_value(key, tensors, memo), _encode_value(item, tensors, memo)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, list):
        return {
            "__cmk_type__": "list",
            "items": [_encode_value(item, tensors, memo) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "__cmk_type__": "tuple",
            "items": [_encode_value(item, tensors, memo) for item in value],
        }
    raise UnsafeSnapshotTypeError(
        f"completion contains unsupported type {type(value).__name__}"
    )


def _decode_value(value: Any, tensors: Mapping[str, torch.Tensor], memo: dict[tuple[str, str], Any]) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if not isinstance(value, dict):
        raise ValueError("completion manifest contains an invalid encoded value")
    value_type = value.get("__cmk_type__")
    if value_type == "bytes" and set(value) == {"__cmk_type__", "base64"}:
        return base64.b64decode(value["base64"], validate=True)
    if value_type in {"torch", "numpy"} and set(value) == {"__cmk_type__", "key"}:
        tensor_key = value["key"]
        if not isinstance(tensor_key, str) or tensor_key not in tensors:
            raise ValueError("completion manifest references a missing tensor")
        memo_key = (value_type, tensor_key)
        if memo_key not in memo:
            tensor = tensors[tensor_key]
            memo[memo_key] = tensor if value_type == "torch" else tensor.numpy()
        return memo[memo_key]
    if value_type in {"list", "tuple"} and set(value) == {"__cmk_type__", "items"}:
        items = value["items"]
        if not isinstance(items, list):
            raise ValueError("completion sequence encoding is invalid")
        decoded = [_decode_value(item, tensors, memo) for item in items]
        return decoded if value_type == "list" else tuple(decoded)
    if value_type == "dict" and set(value) == {"__cmk_type__", "items"}:
        items = value["items"]
        if not isinstance(items, list):
            raise ValueError("completion dictionary encoding is invalid")
        result = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("completion dictionary item is invalid")
            key = _decode_value(pair[0], tensors, memo)
            item = _decode_value(pair[1], tensors, memo)
            if not isinstance(key, (bool, int, float, str, bytes, type(None))):
                raise ValueError("completion dictionary key type is invalid")
            result[key] = item
        return result
    raise ValueError("completion manifest contains an unknown encoded value")


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_MAX_ENTRIES",
    "build_completion_key",
    "completion_cache_inventory",
    "completion_cache_statistics_inventory",
    "cleanup_completion_cache",
    "invalidate_completion",
    "lookup_completion",
    "store_completion",
]
