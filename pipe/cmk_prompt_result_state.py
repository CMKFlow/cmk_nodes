from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
from typing import Any

import numpy as np
import torch

from comfy_execution.cache_provider import (
    CacheContext,
    CacheProvider,
    CacheValue,
    register_cache_provider,
)


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"


MISSING = _Missing()


class UnsafeSnapshotTypeError(TypeError):
    pass


class IncompleteResultError(ValueError):
    pass


class PromptStateUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResultStateKey:
    prompt_id: str
    node_id: str
    model_family: str
    phase: str
    fingerprint: str
    schema_version: str


@dataclass(slots=True)
class _ResultEntry:
    value: Any
    complete: bool


@dataclass(slots=True)
class _PromptState:
    results: dict[ResultStateKey, _ResultEntry] = field(default_factory=dict)
    resources: dict[ResultStateKey, Any] = field(default_factory=dict)


_ACTIVE_PROMPT_ID: ContextVar[str | None] = ContextVar(
    "cmk_prompt_result_state_prompt_id",
    default=None,
)
_STATE_LOCK = threading.RLock()
_PROMPT_STATES: dict[str, _PromptState] = {}


def active_prompt_id() -> str | None:
    return _ACTIVE_PROMPT_ID.get()


def begin_prompt(prompt_id: str) -> None:
    normalized = _required_identity(prompt_id, "prompt_id")
    with _STATE_LOCK:
        _PROMPT_STATES[normalized] = _PromptState()
    _ACTIVE_PROMPT_ID.set(normalized)


def end_prompt(prompt_id: str) -> None:
    normalized = _required_identity(prompt_id, "prompt_id")
    with _STATE_LOCK:
        _PROMPT_STATES.pop(normalized, None)
    if _ACTIVE_PROMPT_ID.get() == normalized:
        _ACTIVE_PROMPT_ID.set(None)


def make_key(
    *,
    node_id: Any,
    model_family: str,
    phase: str,
    fingerprint: str,
    schema_version: str | int,
    prompt_id: str | None = None,
) -> ResultStateKey:
    resolved_prompt_id = prompt_id if prompt_id is not None else active_prompt_id()
    if resolved_prompt_id is None:
        raise PromptStateUnavailableError("no CMK prompt result-state is active")
    return ResultStateKey(
        prompt_id=_required_identity(resolved_prompt_id, "prompt_id"),
        node_id=_required_identity(node_id, "node_id"),
        model_family=_required_identity(model_family, "model_family").lower(),
        phase=_required_identity(phase, "phase").lower(),
        fingerprint=_required_identity(fingerprint, "fingerprint"),
        schema_version=_required_identity(schema_version, "schema_version"),
    )


def snapshot_value(value: Any) -> Any:
    """Copy supported result data without retaining arbitrary Python objects."""
    return _snapshot_value(value, {}, set(), "result")


def stage_result(key: ResultStateKey, value: Any) -> None:
    snapshot = snapshot_value(value)
    with _STATE_LOCK:
        state = _state_for_key(key)
        state.results[key] = _ResultEntry(snapshot, False)


def publish_result(key: ResultStateKey, value: Any) -> None:
    snapshot = snapshot_value(value)
    if _contains_missing(snapshot, set()):
        raise IncompleteResultError("a complete result cannot contain MISSING")
    with _STATE_LOCK:
        state = _state_for_key(key)
        state.results[key] = _ResultEntry(snapshot, True)


def publish_staged_result(key: ResultStateKey) -> None:
    with _STATE_LOCK:
        state = _state_for_key(key)
        entry = state.results.get(key)
        if entry is None:
            raise IncompleteResultError("no staged result exists")
        if _contains_missing(entry.value, set()):
            raise IncompleteResultError("a complete result cannot contain MISSING")
        state.results[key] = _ResultEntry(entry.value, True)


def get_result(key: ResultStateKey, default: Any = MISSING) -> Any:
    with _STATE_LOCK:
        state = _PROMPT_STATES.get(key.prompt_id)
        entry = state.results.get(key) if state is not None else None
        if entry is None or not entry.complete:
            return default
        return entry.value


def has_result(key: ResultStateKey) -> bool:
    return get_result(key) is not MISSING


def store_resource(key: ResultStateKey, value: Any) -> None:
    with _STATE_LOCK:
        state = _state_for_key(key)
        state.resources[key] = value


def get_resource(key: ResultStateKey, default: Any = MISSING) -> Any:
    with _STATE_LOCK:
        state = _PROMPT_STATES.get(key.prompt_id)
        if state is None:
            return default
        return state.resources.get(key, default)


def release_resources(
    *,
    prompt_id: str | None = None,
    model_family: str | None = None,
    phase: str | None = None,
) -> int:
    resolved_prompt_id = _resolve_prompt_id(prompt_id)
    family = model_family.strip().lower() if model_family is not None else None
    normalized_phase = phase.strip().lower() if phase is not None else None
    with _STATE_LOCK:
        state = _PROMPT_STATES.get(resolved_prompt_id)
        if state is None:
            return 0
        keys = [
            key
            for key in state.resources
            if (family is None or key.model_family == family)
            and (normalized_phase is None or key.phase == normalized_phase)
        ]
        for key in keys:
            del state.resources[key]
        return len(keys)


def release_results(
    *,
    prompt_id: str | None = None,
    model_family: str | None = None,
    phase: str | None = None,
) -> int:
    resolved_prompt_id = _resolve_prompt_id(prompt_id)
    family = model_family.strip().lower() if model_family is not None else None
    normalized_phase = phase.strip().lower() if phase is not None else None
    with _STATE_LOCK:
        state = _PROMPT_STATES.get(resolved_prompt_id)
        if state is None:
            return 0
        keys = [
            key
            for key in state.results
            if (family is None or key.model_family == family)
            and (normalized_phase is None or key.phase == normalized_phase)
        ]
        for key in keys:
            del state.results[key]
        return len(keys)


def _required_identity(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _resolve_prompt_id(prompt_id: str | None) -> str:
    resolved = prompt_id if prompt_id is not None else active_prompt_id()
    if resolved is None:
        raise PromptStateUnavailableError("no CMK prompt result-state is active")
    return _required_identity(resolved, "prompt_id")


def _state_for_key(key: ResultStateKey) -> _PromptState:
    state = _PROMPT_STATES.get(key.prompt_id)
    if state is None:
        raise PromptStateUnavailableError(
            f"prompt result-state {key.prompt_id!r} is not active"
        )
    return state


def _snapshot_value(
    value: Any,
    tensor_memo: dict[int, Any],
    active_containers: set[int],
    path: str,
) -> Any:
    if value is MISSING:
        return MISSING
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value

    object_id = id(value)
    if isinstance(value, torch.Tensor):
        cached = tensor_memo.get(object_id)
        if cached is not None:
            return cached
        copied = value.detach().to(device="cpu").clone()
        tensor_memo[object_id] = copied
        return copied
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise UnsafeSnapshotTypeError(
                f"{path} contains a NumPy object array"
            )
        cached = tensor_memo.get(object_id)
        if cached is not None:
            return cached
        copied = np.array(value, copy=True, order="C")
        tensor_memo[object_id] = copied
        return copied
    if isinstance(value, np.generic):
        return _snapshot_value(
            value.item(), tensor_memo, active_containers, path
        )

    if isinstance(value, dict):
        _enter_container(value, active_containers, path)
        try:
            result = {}
            for key, item in value.items():
                if not isinstance(key, (bool, int, float, str, bytes, type(None))):
                    raise UnsafeSnapshotTypeError(
                        f"{path} contains an unsupported dictionary key "
                        f"of type {type(key).__name__}"
                    )
                result[key] = _snapshot_value(
                    item,
                    tensor_memo,
                    active_containers,
                    f"{path}[{key!r}]",
                )
            return result
        finally:
            active_containers.remove(object_id)

    if isinstance(value, list):
        _enter_container(value, active_containers, path)
        try:
            return [
                _snapshot_value(
                    item,
                    tensor_memo,
                    active_containers,
                    f"{path}[{index}]",
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(object_id)

    if isinstance(value, tuple):
        _enter_container(value, active_containers, path)
        try:
            return tuple(
                _snapshot_value(
                    item,
                    tensor_memo,
                    active_containers,
                    f"{path}[{index}]",
                )
                for index, item in enumerate(value)
            )
        finally:
            active_containers.remove(object_id)

    raise UnsafeSnapshotTypeError(
        f"{path} contains unsupported type {type(value).__name__}"
    )


def _enter_container(value: Any, active_containers: set[int], path: str) -> None:
    object_id = id(value)
    if object_id in active_containers:
        raise UnsafeSnapshotTypeError(f"{path} contains a recursive container")
    active_containers.add(object_id)


def _contains_missing(value: Any, visited: set[int]) -> bool:
    if value is MISSING:
        return True
    if isinstance(value, dict):
        object_id = id(value)
        if object_id in visited:
            return False
        visited.add(object_id)
        return any(_contains_missing(item, visited) for item in value.values())
    if isinstance(value, (list, tuple)):
        object_id = id(value)
        if object_id in visited:
            return False
        visited.add(object_id)
        return any(_contains_missing(item, visited) for item in value)
    return False


class CMKPromptResultLifecycleProvider(CacheProvider):
    def should_cache(
        self,
        context: CacheContext,
        value: CacheValue | None = None,
    ) -> bool:
        return False

    async def on_lookup(self, context: CacheContext) -> CacheValue | None:
        return None

    async def on_store(self, context: CacheContext, value: CacheValue) -> None:
        return None

    def on_prompt_start(self, prompt_id: str) -> None:
        begin_prompt(prompt_id)

    def on_prompt_end(self, prompt_id: str) -> None:
        end_prompt(prompt_id)


_LIFECYCLE_PROVIDER = CMKPromptResultLifecycleProvider()
_PROVIDER_REGISTRATION_LOCK = threading.Lock()
_PROVIDER_REGISTERED = False


def register_prompt_result_provider() -> CMKPromptResultLifecycleProvider:
    global _PROVIDER_REGISTERED
    with _PROVIDER_REGISTRATION_LOCK:
        if not _PROVIDER_REGISTERED:
            register_cache_provider(_LIFECYCLE_PROVIDER)
            _PROVIDER_REGISTERED = True
    return _LIFECYCLE_PROVIDER


__all__ = [
    "MISSING",
    "CMKPromptResultLifecycleProvider",
    "IncompleteResultError",
    "PromptStateUnavailableError",
    "ResultStateKey",
    "UnsafeSnapshotTypeError",
    "active_prompt_id",
    "begin_prompt",
    "end_prompt",
    "get_resource",
    "get_result",
    "has_result",
    "make_key",
    "publish_result",
    "publish_staged_result",
    "register_prompt_result_provider",
    "release_resources",
    "release_results",
    "snapshot_value",
    "stage_result",
    "store_resource",
]
