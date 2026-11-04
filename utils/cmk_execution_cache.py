"""Persistent CMK selection for ComfyUI's process-wide execution cache."""

from __future__ import annotations

import json
import os
from pathlib import Path


_MODES = {"classic", "ram_pressure"}
_ACTIVE_AT_START = "unknown"


def _config_path() -> Path:
    import folder_paths  # type: ignore

    root = Path(folder_paths.get_user_directory()) / "cmk"
    root.mkdir(parents=True, exist_ok=True)
    return root / "execution_cache.json"


def _read_mode() -> str | None:
    try:
        payload = json.loads(_config_path().read_text(encoding="utf-8"))
        mode = str(payload.get("mode", "")).strip().lower()
        return mode if mode in _MODES else None
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def _effective_mode(args) -> str:
    if bool(getattr(args, "cache_classic", False)):
        return "classic"
    return "ram_pressure"


def apply_config() -> dict:
    """Apply the saved mode before ComfyUI creates its PromptExecutor."""
    from comfy.cli_args import args  # type: ignore

    configured = _read_mode()
    if configured == "classic":
        args.cache_classic = True
        args.cache_none = False
        args.cache_lru = 0
    elif configured == "ram_pressure":
        args.cache_classic = False
        args.cache_none = False
        args.cache_lru = 0

    global _ACTIVE_AT_START
    _ACTIVE_AT_START = _effective_mode(args)
    return public_status()


def public_status() -> dict:
    configured = _read_mode()
    selected = configured or _ACTIVE_AT_START
    return {
        "active_mode": _ACTIVE_AT_START,
        "configured_mode": selected,
        "restart_required": (
            _ACTIVE_AT_START != "unknown" and selected != _ACTIVE_AT_START
        ),
        "modes": ["classic", "ram_pressure"],
    }


def save_mode(mode: str) -> dict:
    normalized = str(mode or "").strip().lower()
    if normalized not in _MODES:
        raise ValueError("mode must be classic or ram_pressure")

    path = _config_path()
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps({"mode": normalized}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return public_status()
