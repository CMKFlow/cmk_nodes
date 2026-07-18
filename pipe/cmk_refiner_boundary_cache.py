from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


_CACHE_SCHEMA = "cmk_refiner_boundary_cache_v3"
_MAX_DISK_ENTRIES = 16

# MODEL cannot be serialized safely. MODEL and PROCESS remain available for
# the current ComfyUI process; both IMAGE tensors and LOG are materialized.
_SESSION_STATE: dict[str, tuple[Any, dict]] = {}


def _lookup_node(prompt: dict, node_id: Any):
    if not isinstance(prompt, dict):
        return None
    if node_id in prompt:
        return prompt[node_id]
    return prompt.get(str(node_id))


def _resolve_current_node(prompt: dict, unique_id: Any):
    if not isinstance(prompt, dict):
        return None, None

    for candidate in (unique_id, str(unique_id)):
        if candidate in prompt and isinstance(prompt[candidate], dict):
            return candidate, prompt[candidate]

    unique_text = str(unique_id) if unique_id is not None else ""
    suffix_matches = []
    class_matches = []

    for key, value in prompt.items():
        if not isinstance(value, dict):
            continue

        key_text = str(key)
        if unique_text and (
            key_text.endswith(f":{unique_text}")
            or key_text.endswith(f"/{unique_text}")
            or key_text.endswith(f".{unique_text}")
        ):
            suffix_matches.append((key, value))

        if value.get("class_type") == "CMKRefinerBoundaryCache":
            class_matches.append((key, value))

    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(class_matches) == 1:
        return class_matches[0]
    return None, None


def _is_link(prompt: dict, value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
        and _lookup_node(prompt, value[0]) is not None
    )


def _canonical_scalar(value: Any):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _canonical_scalar(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_scalar(item) for item in value]
    return repr(value)


def _canonical_node(prompt: dict, node_id: Any, memo: dict, stack: set):
    key = str(node_id)
    if key in memo:
        return memo[key]
    if key in stack:
        return {"cycle": key}

    node = _lookup_node(prompt, node_id)
    if not isinstance(node, dict):
        return {"missing": key}

    stack.add(key)
    canonical_inputs = {}
    for input_name, value in sorted(
        (node.get("inputs", {}) or {}).items(),
        key=lambda pair: str(pair[0]),
    ):
        if _is_link(prompt, value):
            canonical_inputs[str(input_name)] = {
                "link": {
                    "node": _canonical_node(
                        prompt,
                        value[0],
                        memo,
                        stack,
                    ),
                    "output": int(value[1]),
                }
            }
        else:
            canonical_inputs[str(input_name)] = _canonical_scalar(value)

    result = {
        "class_type": str(node.get("class_type", "")),
        "inputs": canonical_inputs,
    }
    stack.remove(key)
    memo[key] = result
    return result


def build_refiner_fingerprint(prompt: Any, unique_id: Any) -> tuple[str | None, str]:
    if not isinstance(prompt, dict):
        return None, "PROMPT is unavailable"

    resolved_id, current = _resolve_current_node(prompt, unique_id)
    if not isinstance(current, dict):
        return None, (
            "boundary node not found in expanded prompt "
            f"(unique_id={unique_id!r})"
        )

    inputs = current.get("inputs", {}) or {}
    first_link = inputs.get("IMAGE_1ST_PASS")
    refined_link = inputs.get("IMAGE_REFINED")

    if not _is_link(prompt, first_link):
        return None, (
            "IMAGE_1ST_PASS is not an expanded graph link "
            f"(resolved_id={resolved_id!r})"
        )
    if not _is_link(prompt, refined_link):
        return None, (
            "IMAGE_REFINED is not an expanded graph link "
            f"(resolved_id={resolved_id!r})"
        )

    memo = {}
    payload = {
        "schema": _CACHE_SCHEMA,
        "image_1st_pass_root": {
            "node": _canonical_node(
                prompt,
                first_link[0],
                memo,
                set(),
            ),
            "output": int(first_link[1]),
        },
        "image_refined_root": {
            "node": _canonical_node(
                prompt,
                refined_link[0],
                memo,
                set(),
            ),
            "output": int(refined_link[1]),
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), f"resolved_id={resolved_id!r}"


def _cache_directory() -> Path:
    import folder_paths  # type: ignore

    directory = (
        Path(folder_paths.get_temp_directory())
        / "cmk"
        / "refiner_boundary"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _entry_paths(cache_key: str) -> tuple[Path, Path]:
    base = _cache_directory() / f"refiner_{cache_key}"
    return base.with_suffix(".safetensors"), base.with_suffix(".json")


def _status_path() -> Path:
    return _cache_directory() / "last_status.json"


def _write_status(status: str, **fields) -> None:
    payload = {
        "schema": _CACHE_SCHEMA,
        "status": status,
        **fields,
    }
    try:
        with _status_path().open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
    except Exception:
        pass


def _disk_available(cache_key: str) -> bool:
    image_path, log_path = _entry_paths(cache_key)
    return image_path.is_file() and log_path.is_file()


def _remove_disk_entry(cache_key: str) -> None:
    image_path, log_path = _entry_paths(cache_key)
    for path in (image_path, log_path):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _json_safe(value: Any):
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


def _save_disk(
    cache_key: str,
    image_1st_pass,
    image_refined,
    log_pipe: dict,
) -> None:
    from safetensors.torch import save_file  # type: ignore

    image_path, log_path = _entry_paths(cache_key)
    image_tmp = image_path.with_name(image_path.name + ".tmp")
    log_tmp = log_path.with_name(log_path.name + ".tmp")

    try:
        save_file(
            {
                "image_1st_pass": (
                    image_1st_pass.detach()
                    .to("cpu")
                    .contiguous()
                ),
                "image_refined": (
                    image_refined.detach()
                    .to("cpu")
                    .contiguous()
                ),
            },
            str(image_tmp),
            metadata={
                "format": "CMK_REFINER_BOUNDARY_CACHE",
                "schema": _CACHE_SCHEMA,
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

        os.replace(image_tmp, image_path)
        os.replace(log_tmp, log_path)
    finally:
        for path in (image_tmp, log_tmp):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    _prune_disk()


def _load_disk(cache_key: str):
    from safetensors.torch import load_file  # type: ignore

    image_path, log_path = _entry_paths(cache_key)
    try:
        tensors = load_file(str(image_path), device="cpu")
        image_1st_pass = tensors["image_1st_pass"]
        image_refined = tensors["image_refined"]

        with log_path.open("r", encoding="utf-8") as handle:
            log_pipe = json.load(handle)

        for label, image in (
            ("IMAGE 1ST PASS", image_1st_pass),
            ("IMAGE REFINED", image_refined),
        ):
            if getattr(image, "ndim", None) != 4:
                raise ValueError(f"cached {label} tensor is invalid")
        if not isinstance(log_pipe, dict):
            raise TypeError("cached LOG is not a dictionary")

        os.utime(image_path, None)
        os.utime(log_path, None)
        return image_1st_pass, image_refined, log_pipe
    except Exception:
        _remove_disk_entry(cache_key)
        raise


def _prune_disk() -> None:
    directory = _cache_directory()
    entries = sorted(
        directory.glob("refiner_*.safetensors"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for image_path in entries[_MAX_DISK_ENTRIES:]:
        cache_key = image_path.stem.removeprefix("refiner_")
        _remove_disk_entry(cache_key)
        _SESSION_STATE.pop(cache_key, None)


class CMKRefinerBoundaryCache:
    """Materialized boundary for all Refiner image consumers.

    The public Refiner module still exports only IMAGE REFINED. Internally the
    first-pass/refined comparer also consumes this boundary, so no preview or
    output branch can request CMKRefinerPipe directly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS": ("CMK_PIPE", {"lazy": True}),
                "IMAGE_1ST_PASS": ("IMAGE", {"lazy": True}),
                "IMAGE_REFINED": ("IMAGE", {"lazy": True}),
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
        "IMAGE",
        "CMK_LOG_PIPE",
    )
    RETURN_NAMES = (
        "MODEL",
        "PROCESS",
        "IMAGE 1ST PASS",
        "IMAGE REFINED",
        "LOG",
    )
    FUNCTION = "boundary"
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    def check_lazy_status(
        self,
        MODEL=None,
        PROCESS=None,
        IMAGE_1ST_PASS=None,
        IMAGE_REFINED=None,
        LOG=None,
        prompt=None,
        unique_id=None,
    ):
        cache_key, detail = build_refiner_fingerprint(
            prompt,
            unique_id,
        )

        if cache_key is None:
            _write_status(
                "NO_FINGERPRINT",
                detail=detail,
                unique_id=unique_id,
            )
            print(
                "[CMK Refiner Boundary Cache] NO FINGERPRINT: "
                f"{detail}"
            )
        elif (
            cache_key in _SESSION_STATE
            and _disk_available(cache_key)
        ):
            _write_status(
                "HIT_READY",
                cache_key=cache_key,
                detail=detail,
            )
            return []

        needed = []
        for name, value in (
            ("MODEL", MODEL),
            ("PROCESS", PROCESS),
            ("IMAGE_1ST_PASS", IMAGE_1ST_PASS),
            ("IMAGE_REFINED", IMAGE_REFINED),
            ("LOG", LOG),
        ):
            if value is None:
                needed.append(name)
        return needed

    def boundary(
        self,
        MODEL=None,
        PROCESS=None,
        IMAGE_1ST_PASS=None,
        IMAGE_REFINED=None,
        LOG=None,
        prompt=None,
        unique_id=None,
    ):
        cache_key, detail = build_refiner_fingerprint(
            prompt,
            unique_id,
        )

        if (
            cache_key
            and cache_key in _SESSION_STATE
            and _disk_available(cache_key)
            and all(
                value is None
                for value in (
                    MODEL,
                    PROCESS,
                    IMAGE_1ST_PASS,
                    IMAGE_REFINED,
                    LOG,
                )
            )
        ):
            cached_model, cached_process = _SESSION_STATE[cache_key]
            try:
                (
                    cached_first,
                    cached_refined,
                    cached_log,
                ) = _load_disk(cache_key)
                _write_status(
                    "HIT",
                    cache_key=cache_key,
                    detail=detail,
                )
                print(
                    "[CMK Refiner Boundary Cache] HIT "
                    f"{cache_key[:12]} -> BOTH IMAGES/LOG"
                )
                return (
                    cached_model,
                    dict(cached_process),
                    cached_first,
                    cached_refined,
                    cached_log,
                )
            except Exception as exc:
                _SESSION_STATE.pop(cache_key, None)
                _write_status(
                    "HIT_FAILED",
                    cache_key=cache_key,
                    detail=str(exc),
                )
                print(
                    "[CMK Refiner Boundary Cache] HIT FAILED; "
                    f"rebuilding ({exc})"
                )

        missing = [
            name
            for name, value in (
                ("MODEL", MODEL),
                ("PROCESS", PROCESS),
                ("IMAGE_1ST_PASS", IMAGE_1ST_PASS),
                ("IMAGE_REFINED", IMAGE_REFINED),
                ("LOG", LOG),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "CMK Refiner Boundary Cache: cache miss requires "
                + ", ".join(missing)
            )

        if not isinstance(MODEL, dict):
            raise TypeError("MODEL is not a CMK model pipe")
        if not isinstance(PROCESS, dict):
            raise TypeError("PROCESS is not a CMK process pipe")
        if not isinstance(LOG, dict):
            raise TypeError("LOG is not a CMK log pipe")

        if cache_key is None:
            _write_status(
                "BYPASS_NO_FINGERPRINT",
                detail=detail,
                unique_id=unique_id,
            )
            print(
                "[CMK Refiner Boundary Cache] BYPASS: "
                f"{detail}"
            )
            return (
                MODEL,
                PROCESS,
                IMAGE_1ST_PASS,
                IMAGE_REFINED,
                LOG,
            )

        try:
            _save_disk(
                cache_key,
                IMAGE_1ST_PASS,
                IMAGE_REFINED,
                LOG,
            )
            _SESSION_STATE[cache_key] = (
                MODEL,
                dict(PROCESS),
            )
            _write_status(
                "MISS_STORED",
                cache_key=cache_key,
                detail=detail,
            )
            print(
                "[CMK Refiner Boundary Cache] MISS "
                f"{cache_key[:12]} -> BOTH IMAGES STORED"
            )
        except Exception as exc:
            _write_status(
                "STORE_FAILED",
                cache_key=cache_key,
                detail=str(exc),
            )
            print(
                "[CMK Refiner Boundary Cache] STORE FAILED: "
                f"{exc}"
            )

        return (
            MODEL,
            PROCESS,
            IMAGE_1ST_PASS,
            IMAGE_REFINED,
            LOG,
        )
