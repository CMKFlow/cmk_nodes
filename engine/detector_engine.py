from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np

from ..models.model_manager import insightface_root


@dataclass(frozen=True)
class DetectorSettings:
    detector_model: str
    detector_size: int


def _providers() -> List[str]:
    # CPU is the safest cross-platform default, especially on macOS/Apple Silicon.
    return ["CPUExecutionProvider"]


@lru_cache(maxsize=4)
def get_face_analyser(detector_model: str, detector_size: int):
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


def _face_to_dict(face: Any, index: int) -> Dict[str, Any]:
    bbox = getattr(face, "bbox", None)
    kps = getattr(face, "kps", None)
    det_score = getattr(face, "det_score", None)

    return {
        "index": index,
        "bbox": None if bbox is None else np.asarray(bbox, dtype=np.float32),
        "kps": None if kps is None else np.asarray(kps, dtype=np.float32),
        "det_score": None if det_score is None else float(det_score),
        "raw": face,
    }


class CMKDetectorEngine:
    """Native CMK face-detection engine. No dependency on external ComfyUI nodes."""

    def detect_image(self, image_rgb: np.ndarray, settings: DetectorSettings) -> List[Dict[str, Any]]:
        analyser = get_face_analyser(settings.detector_model, settings.detector_size)
        faces = analyser.get(image_rgb)
        faces = sorted(faces, key=lambda f: (float(f.bbox[0]), float(f.bbox[1])))
        return [_face_to_dict(face, index) for index, face in enumerate(faces)]
