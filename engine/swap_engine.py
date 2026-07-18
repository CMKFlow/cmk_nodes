from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

import numpy as np

from ..models.model_manager import insightface_root, resolve_swap_model


@dataclass(frozen=True)
class SwapSettings:
    swap_model: str
    detector_model: str
    detector_size: int
    source_face_index: int
    target_face_index: int


def _providers() -> List[str]:
    # Safe default for macOS/Apple Silicon and normal CPU installs.
    # CUDA users can extend this later without changing the node UI.
    return ["CPUExecutionProvider"]


@lru_cache(maxsize=4)
def _get_face_analyser(detector_model: str, detector_size: int):
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        raise ImportError(
            "Missing dependency: insightface. Install it in the ComfyUI Python environment."
        ) from exc

    analyser = FaceAnalysis(
        name=detector_model,
        root=str(insightface_root()),
        providers=_providers(),
    )
    analyser.prepare(ctx_id=-1, det_size=(detector_size, detector_size))
    return analyser


@lru_cache(maxsize=4)
def _get_swapper(swap_model: str):
    try:
        from insightface import model_zoo
    except Exception as exc:
        raise ImportError(
            "Missing dependency: insightface. Install it in the ComfyUI Python environment."
        ) from exc

    model_path = resolve_swap_model(swap_model)
    return model_zoo.get_model(model_path, providers=_providers())


def _select_face(faces, index: int, role: str):
    if not faces:
        raise RuntimeError(f"No face detected in {role} image.")
    faces = sorted(faces, key=lambda f: (float(f.bbox[0]), float(f.bbox[1])))
    if index < 0 or index >= len(faces):
        raise RuntimeError(
            f"{role} face index {index} is out of range. Detected faces: {len(faces)}."
        )
    return faces[index]


class CMKNativeSwapEngine:
    """Native CMK face-swap engine. No dependency on external ComfyUI nodes."""

    def swap_image(self, target_rgb: np.ndarray, source_rgb: np.ndarray, settings: SwapSettings) -> np.ndarray:
        analyser = _get_face_analyser(settings.detector_model, settings.detector_size)
        swapper = _get_swapper(settings.swap_model)

        source_faces = analyser.get(source_rgb)
        target_faces = analyser.get(target_rgb)

        source_face = _select_face(source_faces, settings.source_face_index, "source")
        target_face = _select_face(target_faces, settings.target_face_index, "target")

        result = swapper.get(target_rgb.copy(), target_face, source_face, paste_back=True)
        return np.clip(result, 0, 255).astype(np.uint8)
