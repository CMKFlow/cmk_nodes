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


def insightface_root() -> Path:
    return _models_root() / "insightface"


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
    return _models_root() / "facerestore_models"


def _candidate_model_paths(root: Path, name: str) -> list[Path]:
    stem = Path(name).stem
    return [
        root / name,
        root / stem / name,
        root / stem,
        root / f"{stem}.onnx",
    ]


def resolve_face_restore_model(name: str) -> str:
    root = face_restore_model_dir()
    suffix = Path(name).suffix.lower()
    glob_pattern = f"*{suffix}" if suffix else "*"
    for path in _candidate_model_paths(root, name):
        if path.is_file():
            return str(path)
        if path.is_dir():
            matches = sorted(p for p in path.glob(glob_pattern) if p.is_file())
            if matches:
                return str(matches[0])
    matches = sorted(root.rglob(name)) if root.exists() else []
    if matches:
        return str(matches[0])
    raise FileNotFoundError(
        "Face restore model not found: "
        f"{root / name}\n"
        "Expected locations include ComfyUI/models/facerestore_models/<model-file> "
        "or ComfyUI/models/facerestore_models/<model-folder>/<model-file>."
    )
