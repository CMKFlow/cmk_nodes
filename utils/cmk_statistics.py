from __future__ import annotations

import json
import platform
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..pipe.cmk_generation_completion_cache import completion_cache_statistics_inventory
from ..pipe.cmk_persistent_cache import persistent_disk_scope_inventory


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CATEGORY_ORDER = (
    "REAL-WORLD WORKFLOWS",
    "TASK WORKFLOWS",
    "MODULE WORKFLOWS",
    "SYSTEM WORKFLOW",
    "COMPARISONS",
    "LEGACY",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _package_version(package_root: Path) -> str | None:
    try:
        source = (package_root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        import tomllib  # Python 3.11+

        data = tomllib.loads(source)
        value = data.get("project", {}).get("version")
        return str(value) if value else None
    except ImportError:
        # CMK supports Python 3.10 too. Keep the fallback scoped to the static
        # PEP 621 field instead of introducing a second version source.
        project = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", source)
        match = re.search(
            r'^version\s*=\s*["\']([^"\']+)["\']\s*$',
            project.group(1) if project else "",
            re.MULTILINE,
        )
        return match.group(1) if match else None
    except (ValueError, TypeError):
        return None


def _regular_json_files(directory: Path) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        return []
    result = []
    for path in directory.glob("*.json"):
        if path.is_symlink():
            continue
        try:
            if path.is_file():
                result.append(path)
        except OSError:
            continue
    return sorted(result, key=lambda path: path.name.casefold())


def _read_metadata(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _showcase_records(package_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    workflows = package_root / "workflows" / "showcase"
    metadata = workflows / "metadata"
    return [
        (path, _read_metadata(metadata / path.name))
        for path in _regular_json_files(workflows)
    ]


def cmk_flow_statistics(
    node_mappings: Mapping[str, type], package_root: Path | str = PACKAGE_ROOT,
) -> dict[str, Any]:
    root = Path(package_root)
    unique_classes = {node_class for node_class in node_mappings.values() if isinstance(node_class, type)}
    return {
        "version": _package_version(root),
        "registered_nodes": len(unique_classes),
        "subgraphs": len(_regular_json_files(root / "subgraphs")),
    }


def flow_browser_statistics(package_root: Path | str = PACKAGE_ROOT) -> dict[str, Any]:
    records = _showcase_records(Path(package_root))
    categories = {category: 0 for category in CATEGORY_ORDER}
    visible = 0
    for _, metadata in records:
        visible += int(metadata.get("published") is True)
        category = str(metadata.get("category") or "UNASSIGNED")
        categories[category] = categories.get(category, 0) + 1
    return {
        "total": len(records),
        "visible": visible,
        "hidden": len(records) - visible,
        "categories": categories,
    }


def _safe_image_file(web_root: Path, source: Any) -> tuple[Path, int] | None:
    if not isinstance(source, str) or not source.strip():
        return None
    relative = PurePosixPath(source)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    candidate = web_root.joinpath(*relative.parts)
    current = web_root
    try:
        if web_root.is_symlink() or not web_root.is_dir():
            return None
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return None
        stat = candidate.stat(follow_symlinks=False)
        if not candidate.is_file():
            return None
        return candidate, stat.st_size
    except OSError:
        return None


def showcase_statistics(package_root: Path | str = PACKAGE_ROOT) -> dict[str, int]:
    root = Path(package_root)
    web_root = root / "web"
    images: dict[Path, int] = {}
    galleries = 0
    for _, metadata in _showcase_records(root):
        gallery_images = set()
        previews = metadata.get("previews", [])
        if not isinstance(previews, list):
            continue
        for preview in previews:
            source = preview.get("src") if isinstance(preview, dict) else None
            result = _safe_image_file(web_root, source)
            if result is None:
                continue
            path, size = result
            images[path] = size
            gallery_images.add(path)
        galleries += int(bool(gallery_images))
    return {
        "galleries": galleries,
        "screenshots": len(images),
        "bytes": sum(images.values()),
    }


def _sum_inventory(items: list[dict[str, int]]) -> dict[str, int]:
    return {
        key: sum(int(item.get(key, 0)) for item in items)
        for key in ("entries", "files", "bytes")
    }


def persistent_cache_statistics(
    package_root: Path | str = PACKAGE_ROOT,
    temp_root: Path | str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        result["generation_completion"] = completion_cache_statistics_inventory(package_root)
    except OSError as error:
        result["generation_completion"] = {"available": False, "error": str(error)}
    try:
        scopes = persistent_disk_scope_inventory(temp_root)
        result["branch"] = _sum_inventory([
            inventory for scope, inventory in scopes.items() if scope.endswith("_branch")
        ])
        result["boundary"] = _sum_inventory([
            inventory for scope, inventory in scopes.items() if not scope.endswith("_branch")
        ])
    except OSError as error:
        result["branch"] = {"available": False, "error": str(error)}
        result["boundary"] = {"available": False, "error": str(error)}
    available = [
        area for area in (
            result["generation_completion"], result["branch"], result["boundary"],
        ) if area.get("available", True)
    ]
    result["total"] = _sum_inventory(available)
    result["complete"] = len(available) == 3
    return result


def runtime_statistics() -> dict[str, str | None]:
    comfyui = None
    pytorch = None
    compute = None
    try:
        import comfyui_version  # type: ignore

        comfyui = str(comfyui_version.__version__)
    except (ImportError, AttributeError):
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        value = getattr(torch, "__version__", None)
        pytorch = str(value) if value is not None else None
    try:
        model_management = sys.modules.get("comfy.model_management")
        if model_management is None:
            raise RuntimeError("ComfyUI device API is not loaded")
        device = model_management.get_torch_device()
        device_type = str(getattr(device, "type", device)).upper()
        try:
            device_name = str(model_management.get_torch_device_name(device))
        except (AttributeError, RuntimeError):
            device_name = str(device)
        compute = device_type if device_name.casefold() == device_type.casefold() else f"{device_type} · {device_name}"
    except Exception:
        pass
    return {
        "comfyui": comfyui,
        "python": platform.python_version(),
        "pytorch": pytorch,
        "compute": compute,
    }


def collect_statistics(
    node_mappings: Mapping[str, type],
    package_root: Path | str = PACKAGE_ROOT,
    temp_root: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(package_root)
    return {
        "cmk_flow": cmk_flow_statistics(node_mappings, root),
        "flow_browser": flow_browser_statistics(root),
        "showcase": showcase_statistics(root),
        "persistent_cache": persistent_cache_statistics(root, temp_root),
        "runtime": runtime_statistics(),
    }


__all__ = [
    "CATEGORY_ORDER",
    "collect_statistics",
    "cmk_flow_statistics",
    "flow_browser_statistics",
    "persistent_cache_statistics",
    "runtime_statistics",
    "showcase_statistics",
]
