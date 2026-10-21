from __future__ import annotations

from typing import Any

from ..pipe.cmk_generation_completion_cache import (
    completion_cache_inventory,
    invalidate_completion,
)
from ..pipe.cmk_module_boundary_cache import reset_module_boundary_cache_state
from ..pipe.cmk_persistent_cache import (
    clear_persistent_disk_scopes,
    reset_persistent_fingerprint_memo,
    validate_persistent_disk_scopes,
)
from ..pipe.cmk_refiner_boundary_cache import reset_refiner_boundary_cache_state


class PersistentCacheBusyError(RuntimeError):
    pass


class PersistentCacheClearError(RuntimeError):
    def __init__(self, message: str, *, phase: str):
        super().__init__(message)
        self.phase = phase


def _totals(areas: dict[str, dict[str, int]]) -> dict[str, int]:
    return {
        field: sum(int(area.get(field, 0)) for area in areas.values())
        for field in ("entries", "files", "bytes")
    }


def clear_persistent_cmk_cache(prompt_queue: Any, *, temp_root=None) -> dict[str, Any]:
    """Clear fixed CMK disk caches atomically against PromptQueue execution."""
    phase = "queue_check"
    with prompt_queue.mutex:
        if prompt_queue.currently_running:
            raise PersistentCacheBusyError(
                "Persistent cache cannot be cleared while a generation is running."
            )

        try:
            phase = "path_validation"
            validate_persistent_disk_scopes(temp_root)

            phase = "generation_completion"
            completion_before = completion_cache_inventory()
            removed_completion_entries = invalidate_completion(None)
            completion_after = completion_cache_inventory()
            completion = {
                "entries": removed_completion_entries,
                "files": max(0, completion_before["files"] - completion_after["files"]),
                "bytes": max(0, completion_before["bytes"] - completion_after["bytes"]),
            }

            phase = "temporary_disk_caches"
            disk_scopes = clear_persistent_disk_scopes(temp_root)

            phase = "runtime_state"
            runtime_state = {
                "persistent_fingerprint_memo": reset_persistent_fingerprint_memo(),
                **reset_module_boundary_cache_state(),
                **reset_refiner_boundary_cache_state(),
            }
        except Exception as error:
            raise PersistentCacheClearError(str(error), phase=phase) from error

    areas = {"generation_completion": completion, **disk_scopes}
    return {
        "ok": True,
        "cleared": True,
        "areas": areas,
        "totals": _totals(areas),
        "runtime_state": runtime_state,
    }


__all__ = [
    "PersistentCacheBusyError",
    "PersistentCacheClearError",
    "clear_persistent_cmk_cache",
]
