from __future__ import annotations

import os
from pathlib import Path
from typing import List

try:
    import folder_paths
except Exception:  # pragma: no cover - only outside ComfyUI
    folder_paths = None


def _models_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.models_dir)
    return Path.cwd() / "models"


def _registered_roots(category: str, fallback: Path) -> list[Path]:
    roots: list[Path] = []
    if folder_paths is not None:
        registration = folder_paths.folder_names_and_paths.get(category)
        if registration:
            roots.extend(Path(path) for path in registration[0])
    roots.append(fallback)
    return list(dict.fromkeys(roots))


def _register_cmk_model_folders() -> None:
    """Register model categories that used to arrive through other node packs."""
    if folder_paths is None:
        return
    extensions = set(getattr(folder_paths, "supported_pt_extensions", {".pt", ".pth"}))
    extensions.update({".onnx"})
    registrations = {
        "facerestore_models": _models_root() / "facerestore_models",
    }
    for name, path in registrations.items():
        if name not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths[name] = ([str(path)], extensions)


_register_cmk_model_folders()


def insightface_root() -> Path:
    return _registered_roots("insightface", _models_root() / "insightface")[0]


def swap_model_dir() -> Path:
    return insightface_root()


def list_swap_models() -> List[str]:
    root = swap_model_dir()
    if not root.exists():
        return ["inswapper_128.onnx"]
    names = sorted(p.name for p in root.glob("*.onnx"))
    return names or ["inswapper_128.onnx"]


def resolve_swap_model(name: str) -> str:
    path = swap_model_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            "FaceSwap model not found: "
            f"{path}\n"
            "Expected location: ComfyUI/models/insightface/inswapper_128.onnx"
        )
    return str(path)


def list_detector_models() -> List[str]:
    root = insightface_root() / "models"
    if not root.exists():
        return ["buffalo_l"]
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    return names or ["buffalo_l"]


def face_restore_model_dir() -> Path:
    return _registered_roots("facerestore_models", _models_root() / "facerestore_models")[0]


def _candidate_model_paths(root: Path, name: str) -> list[Path]:
    stem = Path(name).stem
    return [
        root / name,
        root / stem / name,
        root / stem,
        root / f"{stem}.onnx",
    ]


def resolve_face_restore_model(name: str) -> str:
    suffix = Path(name).suffix.lower()
    glob_pattern = f"*{suffix}" if suffix else "*"
    roots = _registered_roots("facerestore_models", _models_root() / "facerestore_models")
    for root in roots:
        for path in _candidate_model_paths(root, name):
            if path.is_file():
                return str(path)
            if path.is_dir():
                matches = sorted(p for p in path.glob(glob_pattern) if p.is_file())
                if matches:
                    return str(matches[0])
        if str(name).lower() == "codeformer-v0.1.0.pth":
            compatible = root / "codeformer.pth"
            if compatible.is_file():
                return str(compatible)
        matches = sorted(root.rglob(name)) if root.exists() else []
        if matches:
            return str(matches[0])
    root = roots[0]
    raise FileNotFoundError(
        "Face restore model not found: "
        f"{root / name}\n"
        "Expected locations include ComfyUI/models/facerestore_models/<model-file> "
        "or ComfyUI/models/facerestore_models/<model-folder>/<model-file>."
    )
