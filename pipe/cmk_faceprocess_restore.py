"""FaceProcess 2.8 RESTORE (F3): YOLO-led face selection and resolved backends.

Selection: YOLO detects and selects (one selection rule for every mode).
InsightFace only assigns its complete face object -- bbox, five keypoints and
the 106 landmarks of the restore pasteback -- to each selected YOLO face
(``utils.face_matching``).  It never selects; an unassigned face is skipped.

Restore models (E5): the selected file decides the backend, nothing falls back
silently.  Supported:

- ``*gfpgan*.pth``      GFPGAN through Spandrel on the CPU, exactly that file;
- CodeFormer checkpoint the fixed CMK CodeFormer reference behaviour (fidelity
                        w = 0.72, then 72 % result / 28 % input); only the file
                        that backend actually loads is accepted;
- ``GPEN-BFR-512.onnx``  GPEN (ONNX, CPU) with its fixed edge protection.

Everything else, ``none`` and missing files are configuration errors.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..engine import enhance_backends as backends
from ..models.model_manager import resolve_face_restore_model

GPEN_FILE = "GPEN-BFR-512.onnx"
CODEFORMER_REFERENCE = "codeformer-v0.1.0.pth"


class RestoreConfigurationError(ValueError):
    """A RESTORE branch cannot run with its configuration."""


def _resolve(name):
    try:
        return resolve_face_restore_model(name)
    except FileNotFoundError as exc:
        raise RestoreConfigurationError(f"Restore model not found: {name}") from exc


def _same_file(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


def resolve_restore_backend(model_name):
    """Selected restore model -> (backend, info).  No model is loaded here."""
    name = str(model_name or "").strip()
    if not name or name.lower() == "none":
        raise RestoreConfigurationError("RESTORE requires a restore model (restore_model is none)")
    lower = Path(name).name.lower()
    path = _resolve(name)
    if "gfpgan" in lower and lower.endswith(".pth"):
        return backends.GFPGANEnhancerBackend(name), {
            "backend": "GFPGAN", "file": path, "runtime": "Spandrel · CPU",
            "parameters": {"model_size": backends.GFPGANEnhancerBackend.model_size},
        }
    if "codeformer" in lower:
        available, reason = backends.codeformer_availability()
        if not available:
            raise RestoreConfigurationError(f"CodeFormer is unavailable in this environment: {reason}")
        reference = _resolve(CODEFORMER_REFERENCE)
        if not _same_file(path, reference):
            raise RestoreConfigurationError(
                f"CodeFormer loads only {Path(reference).name}; the selected file {name} is not supported")
        backend = backends.CodeFormerEnhancerBackend()
        return backend, {
            "backend": "CodeFormer", "file": reference, "runtime": "PyTorch · CPU",
            "parameters": {"fidelity_w": backend.fidelity, "blend_result": 0.72, "blend_input": 0.28},
        }
    if lower == GPEN_FILE.lower():
        return backends.GPENEnhancerBackend(), {
            "backend": "GPEN", "file": path, "runtime": "ONNX · CPU",
            "parameters": {"weight_smooth": 0.42, "weight_edges": 0.12, "detail_limit": "1.35x + 5"},
        }
    raise RestoreConfigurationError(
        f"Unsupported restore model: {name} (supported: GFPGAN *.pth, the CMK CodeFormer checkpoint, {GPEN_FILE})")


def backend_identity(model_name):
    """Cache identity of a restore choice without loading anything: the
    resolved backend and file, or the configuration error."""
    try:
        _backend, info = resolve_restore_backend(model_name)
    except RestoreConfigurationError as exc:
        return {"error": str(exc)}
    path = Path(info["file"])
    try:
        stat = path.stat()
        content = f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        content = "unavailable"
    return {"backend": info["backend"], "file": path.name, "content": content}
