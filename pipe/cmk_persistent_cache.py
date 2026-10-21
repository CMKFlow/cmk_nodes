from __future__ import annotations

import hashlib
import json
import os
import pickle
import secrets
import time
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..cmk_distribution import CONTENT_GUARD_WIDGET


PERSISTENT_DISK_SCOPES = (
    "detailer_branch",
    "faceprocess_branch",
    "detailer_boundary",
    "face_rebuild_boundary",
    "z_image_boundary",
    "faceprocess_boundary",
    "faceswap_boundary",
    "upscale_save_boundary",
    "refiner_boundary",
)


def _lookup_node(prompt: dict, node_id: Any):
    if not isinstance(prompt, dict):
        return None
    if node_id in prompt:
        return prompt[node_id]
    return prompt.get(str(node_id))


def _resolve_current_node(
    prompt: dict,
    unique_id: Any,
    class_types: Iterable[str],
):
    if not isinstance(prompt, dict):
        return None, None

    class_types = set(str(value) for value in class_types)

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

        if str(value.get("class_type", "")) in class_types:
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


def _canonical_node(
    prompt: dict,
    node_id: Any,
    memo: dict,
    stack: set,
):
    key = str(node_id)
    if key in memo:
        return memo[key]
    if key in stack:
        return {"cycle": key}

    node = _lookup_node(prompt, node_id)
    if not isinstance(node, dict):
        return {"missing": key}

    stack.add(key)
    inputs = {}
    class_type = str(node.get("class_type", ""))
    for input_name, value in sorted(
        (node.get("inputs", {}) or {}).items(),
        key=lambda pair: str(pair[0]),
    ):
        # ContentGuard is a downstream policy setting.  It must not invalidate
        # generation/refiner materializations that do not consume it; the
        # affected module boundaries add its effective state explicitly.
        if class_type == "CMKPipeCreateImage" and str(input_name) == CONTENT_GUARD_WIDGET:
            continue
        if _is_link(prompt, value):
            inputs[str(input_name)] = {
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
            inputs[str(input_name)] = _canonical_scalar(value)

    payload = {
        "class_type": class_type,
        "inputs": inputs,
    }
    stack.remove(key)
    # Return a content address instead of embedding the complete upstream
    # object. Expanded CMK graphs are DAGs with many converging pipe links.
    # Reusing nested Python objects still makes json.dumps expand every path,
    # which turned a 46-node FaceSwap prompt into minutes of serialization.
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    result = {"sha256": hashlib.sha256(encoded).hexdigest()}
    memo[key] = result
    return result


# Fingerprints are requested repeatedly for the same immutable expanded prompt:
# first by lazy checks, then by execution nodes, then again by module boundaries.
# Canonicalizing a large upstream graph each time is prohibitively expensive.
# Keep only the current prompt object's results; a new prompt object clears the
# memo automatically, so no cache decision survives across workflow submissions.
_FINGERPRINT_MEMO_LOCK = threading.RLock()
_FINGERPRINT_MEMO_PROMPT_ID = None
_FINGERPRINT_MEMO = {}


def _fingerprint_memo_key(
    unique_id,
    class_types,
    schema,
    exclude_inputs,
    include_node_identity,
    extra_identity,
):
    return (
        str(unique_id),
        tuple(sorted(str(value) for value in class_types)),
        str(schema),
        tuple(sorted(str(value) for value in exclude_inputs)),
        bool(include_node_identity),
        json.dumps(_canonical_scalar(extra_identity), sort_keys=True, default=str),
    )


def build_node_fingerprint(
    prompt: Any,
    unique_id: Any,
    class_types: Iterable[str],
    schema: str,
    exclude_inputs: Iterable[str] = (),
    include_node_identity: bool = False,
    extra_identity: Mapping[str, Any] | None = None,
) -> tuple[str | None, str]:
    if not isinstance(prompt, dict):
        return None, "PROMPT is unavailable"

    class_types_tuple = tuple(str(value) for value in class_types)
    excluded_tuple = tuple(str(value) for value in exclude_inputs)
    memo_key = _fingerprint_memo_key(
        unique_id,
        class_types_tuple,
        schema,
        excluded_tuple,
        include_node_identity,
        extra_identity,
    )

    global _FINGERPRINT_MEMO_PROMPT_ID, _FINGERPRINT_MEMO
    prompt_id = id(prompt)
    with _FINGERPRINT_MEMO_LOCK:
        if _FINGERPRINT_MEMO_PROMPT_ID != prompt_id:
            _FINGERPRINT_MEMO_PROMPT_ID = prompt_id
            _FINGERPRINT_MEMO = {}
        cached = _FINGERPRINT_MEMO.get(memo_key)
        if cached is not None:
            return cached

    resolved_id, current = _resolve_current_node(
        prompt,
        unique_id,
        class_types_tuple,
    )
    if not isinstance(current, dict):
        result = (
            None,
            "cache node not found in expanded prompt "
            f"(unique_id={unique_id!r})",
        )
        with _FINGERPRINT_MEMO_LOCK:
            _FINGERPRINT_MEMO[memo_key] = result
        return result

    excluded = set(excluded_tuple)
    canonical_memo = {}
    canonical_inputs = {}

    for input_name, value in sorted(
        (current.get("inputs", {}) or {}).items(),
        key=lambda pair: str(pair[0]),
    ):
        if str(input_name) in excluded:
            continue

        if _is_link(prompt, value):
            canonical_inputs[str(input_name)] = {
                "link": {
                    "node": _canonical_node(
                        prompt,
                        value[0],
                        canonical_memo,
                        set(),
                    ),
                    "output": int(value[1]),
                }
            }
        else:
            canonical_inputs[str(input_name)] = _canonical_scalar(value)

    payload = {
        "schema": str(schema),
        "class_type": str(current.get("class_type", "")),
        "inputs": canonical_inputs,
    }
    if extra_identity:
        payload["extra_identity"] = _canonical_scalar(extra_identity)
    if include_node_identity:
        payload["node_id"] = str(resolved_id)

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    result = (
        hashlib.sha256(encoded).hexdigest(),
        f"resolved_id={resolved_id!r}",
    )
    with _FINGERPRINT_MEMO_LOCK:
        _FINGERPRINT_MEMO[memo_key] = result
    return result


def cache_directory(scope: str) -> Path:
    import folder_paths  # type: ignore

    directory = (
        Path(folder_paths.get_temp_directory())
        / "cmk"
        / str(scope)
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _persistent_cache_root(temp_root: Path | str | None = None) -> Path:
    if temp_root is None:
        import folder_paths  # type: ignore

        temp_root = folder_paths.get_temp_directory()
    return Path(temp_root) / "cmk"


def validate_persistent_disk_scopes(temp_root: Path | str | None = None) -> Path:
    """Validate every fixed CMK disk-cache scope before any deletion starts."""
    root = _persistent_cache_root(temp_root)
    if root.is_symlink():
        raise OSError(f"refusing symbolic-link CMK cache root: {root}")
    if root.exists() and not root.is_dir():
        raise OSError(f"CMK cache root is not a directory: {root}")
    for scope in PERSISTENT_DISK_SCOPES:
        path = root / scope
        if path.is_symlink():
            raise OSError(f"refusing symbolic-link CMK cache scope: {path}")
        if path.exists() and not path.is_dir():
            raise OSError(f"CMK cache scope is not a directory: {path}")
    return root


def _entry_identity(path: Path) -> str | None:
    if path.name in {"last_status.json", "latest_by_node.json"} or path.name.endswith(".tmp"):
        return None
    for suffix in (
        ".diagnostic.safetensors",
        ".diagnostic.json",
        ".safetensors",
        ".json",
        ".deps",
        ".pkl",
        ".rev",
    ):
        if path.name.endswith(suffix):
            return path.name[:-len(suffix)]
    return path.name


def _scope_inventory(path: Path) -> dict[str, int]:
    files = []
    entries = set()
    def visit(directory: Path) -> None:
        for entry in os.scandir(directory):
            candidate = Path(entry.path)
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                files.append(candidate)
                identity = _entry_identity(candidate)
                if identity:
                    entries.add(identity)
            else:
                visit(candidate)

    if path.exists():
        visit(path)
    return {
        "entries": len(entries),
        "files": len(files),
        "bytes": sum(candidate.lstat().st_size for candidate in files),
    }


def persistent_disk_scope_inventory(
    temp_root: Path | str | None = None,
) -> dict[str, dict[str, int]]:
    """Inventory the clearable CMK disk scopes without creating or changing them."""
    root = validate_persistent_disk_scopes(temp_root)
    return {
        scope: _scope_inventory(root / scope)
        for scope in PERSISTENT_DISK_SCOPES
    }


def _remove_scope_contents(path: Path) -> None:
    """Remove contents without following symbolic links."""
    if not path.exists():
        return
    for entry in os.scandir(path):
        candidate = Path(entry.path)
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            candidate.unlink()
        else:
            _remove_scope_contents(candidate)
            candidate.rmdir()


def clear_persistent_disk_scopes(temp_root: Path | str | None = None) -> dict[str, dict[str, int]]:
    """Clear only the explicitly registered CMK temp-cache scopes."""
    root = validate_persistent_disk_scopes(temp_root)
    root.mkdir(parents=True, exist_ok=True)
    inventories = {
        scope: _scope_inventory(root / scope)
        for scope in PERSISTENT_DISK_SCOPES
    }
    for scope in PERSISTENT_DISK_SCOPES:
        path = root / scope
        path.mkdir(parents=True, exist_ok=True)
        _remove_scope_contents(path)
    return inventories


def reset_persistent_fingerprint_memo() -> int:
    """Drop prompt-scoped fingerprint results coupled to persistent caches."""
    global _FINGERPRINT_MEMO_PROMPT_ID, _FINGERPRINT_MEMO
    with _FINGERPRINT_MEMO_LOCK:
        count = len(_FINGERPRINT_MEMO)
        _FINGERPRINT_MEMO_PROMPT_ID = None
        _FINGERPRINT_MEMO = {}
        return count


def write_status(scope: str, status: str, **fields) -> None:
    payload = {
        "status": str(status),
        **fields,
    }
    try:
        path = cache_directory(scope) / "last_status.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
    except Exception:
        pass


def cache_path(scope: str, cache_key: str, suffix: str) -> Path:
    return cache_directory(scope) / f"{cache_key}{suffix}"


def revision_path(scope: str, cache_key: str) -> Path:
    return cache_path(scope, cache_key, ".rev")


def read_pickle_revision(scope: str, cache_key: str) -> str | None:
    path = revision_path(scope, cache_key)
    try:
        token = path.read_text(encoding="utf-8").strip()
        return token or None
    except Exception:
        return None


def pickle_available(scope: str, cache_key: str) -> bool:
    return (
        cache_path(scope, cache_key, ".pkl").is_file()
        and revision_path(scope, cache_key).is_file()
    )


def _write_revision(scope: str, cache_key: str) -> str:
    token = f"{time.time_ns()}-{secrets.token_hex(8)}"
    path = revision_path(scope, cache_key)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(token, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
    return token


def write_pickle_revision(scope: str, cache_key: str, revision: str) -> str:
    """Publish a deterministic branch revision for a dependent boundary."""
    token = str(revision or "").strip()
    if not token:
        raise ValueError("cache revision must not be empty")
    path = revision_path(scope, cache_key)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(token, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
    return token


def _remove_pickle_entry(scope: str, cache_key: str) -> None:
    for suffix in (".pkl", ".rev"):
        try:
            cache_path(scope, cache_key, suffix).unlink(missing_ok=True)
        except Exception:
            pass


def _prune_orphan_revisions(scope: str) -> None:
    directory = cache_directory(scope)
    for path in directory.glob("*.rev"):
        cache_key = path.name[:-4]
        if not cache_path(scope, cache_key, ".pkl").is_file():
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def save_pickle(
    scope: str,
    cache_key: str,
    payload: Any,
    max_entries: int = 32,
) -> Path:
    path = cache_path(scope, cache_key, ".pkl")
    temporary = path.with_name(path.name + ".tmp")

    try:
        with temporary.open("wb") as handle:
            pickle.dump(
                payload,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # The old revision must disappear before the payload is replaced. If a
        # crash occurs between payload and revision replacement, the entry is
        # deliberately incomplete and will be recomputed instead of appearing
        # valid with a stale revision token.
        try:
            revision_path(scope, cache_key).unlink(missing_ok=True)
        except Exception:
            pass

        os.replace(temporary, path)
        _write_revision(scope, cache_key)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass

    prune(scope, ".pkl", max_entries)
    _prune_orphan_revisions(scope)
    return path


def load_pickle(scope: str, cache_key: str):
    path = cache_path(scope, cache_key, ".pkl")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        os.utime(path, None)
        return payload
    except Exception:
        _remove_pickle_entry(scope, cache_key)
        raise


def _collect_upstream_targets(
    prompt: dict,
    node_id: Any,
    target_class_types: set[str],
    found: dict[str, tuple[Any, str]],
    visited: set[str],
) -> None:
    key = str(node_id)
    if key in visited:
        return
    visited.add(key)

    node = _lookup_node(prompt, node_id)
    if not isinstance(node, dict):
        return

    class_type = str(node.get("class_type", ""))
    if class_type in target_class_types:
        found[key] = (node_id, class_type)

    for value in (node.get("inputs", {}) or {}).values():
        if _is_link(prompt, value):
            _collect_upstream_targets(
                prompt,
                value[0],
                target_class_types,
                found,
                visited,
            )


def build_upstream_cache_manifest(
    prompt: Any,
    unique_id: Any,
    current_class_types: Iterable[str],
    branch_specs: Mapping[str, tuple[str, str]],
    input_names: Iterable[str] = (),
) -> tuple[dict, str]:
    """Describe the exact persistent branch materializations feeding a boundary.

    The manifest is intentionally based on tiny revision tokens written whenever a
    branch pickle is replaced. A static prompt fingerprint alone cannot detect that
    a branch with the same settings was recomputed after the boundary was stored.
    """
    empty = {
        "version": 1,
        "complete": False,
        "branches": [],
    }
    if not isinstance(prompt, dict):
        return empty, "PROMPT is unavailable"

    resolved_id, current = _resolve_current_node(
        prompt,
        unique_id,
        current_class_types,
    )
    if not isinstance(current, dict):
        return empty, (
            "boundary node not found in expanded prompt "
            f"(unique_id={unique_id!r})"
        )

    selected_inputs = set(str(name) for name in input_names)
    targets = set(str(name) for name in branch_specs)
    found: dict[str, tuple[Any, str]] = {}
    visited: set[str] = set()

    for input_name, value in (current.get("inputs", {}) or {}).items():
        if selected_inputs and str(input_name) not in selected_inputs:
            continue
        if _is_link(prompt, value):
            _collect_upstream_targets(
                prompt,
                value[0],
                targets,
                found,
                visited,
            )

    entries = []
    complete = True
    details = []

    for node_key, (node_id, class_type) in sorted(found.items()):
        scope, schema = branch_specs[class_type]
        cache_key, fingerprint_detail = build_node_fingerprint(
            prompt,
            node_id,
            (class_type,),
            schema,
            include_node_identity=True,
        )
        revision = (
            read_pickle_revision(scope, cache_key)
            if cache_key
            else None
        )
        if cache_key is None or revision is None:
            complete = False

        entries.append(
            {
                "node_id": str(node_id),
                "class_type": class_type,
                "scope": scope,
                "cache_key": cache_key,
                "revision": revision,
            }
        )
        details.append(
            f"{node_key}:{cache_key[:12] if cache_key else 'NO_KEY'}:"
            f"{'READY' if revision else 'MISSING'}"
        )
        if cache_key is None:
            details.append(fingerprint_detail)

    manifest = {
        "version": 1,
        "boundary_node_id": str(resolved_id),
        "complete": complete,
        "branches": entries,
    }
    detail = ", ".join(details) if details else "no branch dependencies"
    return manifest, detail


def prune(
    scope: str,
    suffix: str,
    max_entries: int,
) -> None:
    directory = cache_directory(scope)
    entries = sorted(
        (
            path
            for path in directory.glob(f"*{suffix}")
            if path.name != "last_status.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in entries[max(1, int(max_entries)):]:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
