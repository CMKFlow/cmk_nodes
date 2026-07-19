from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..utils.face_set_utils import normalize_selected_face
from .swap_backends import BackendSwapSettings, get_default_swap_backend
from .content_guard import get_content_guard


@dataclass(frozen=True)
class SelectedSwapSettings:
    swap_model: str
    enhancer_mode: str = "GPEN"
    bbox_dilation: int = 0
    crop_factor: float = 1.0
    feather: int = 0
    identity_strength: float = 1.0


def _raw_face(selected_face: Dict[str, Any], role: str):
    sf = normalize_selected_face(selected_face)
    face = sf.get("face", {})
    raw = face.get("raw")
    if raw is None:
        raise RuntimeError(
            f"{role} selected_face does not contain a raw InsightFace face object. "
            "Create it with CMK Detect Faces -> CMK Face Select in the same workflow run."
        )
    return raw


class CMKSelectedSwapEngine:
    """Swap using selected faces. Public selected-face behavior remains unchanged."""

    def swap_faces(self, target_rgb: np.ndarray, source_rgb: np.ndarray, source_face, target_face, settings: SelectedSwapSettings) -> np.ndarray:
        get_content_guard().assert_swap_allowed(
            target_rgb=target_rgb,
            source_rgb=source_rgb,
            target_face=target_face,
            source_face=source_face,
        )
        backend = get_default_swap_backend()
        return backend.swap(
            target_rgb=target_rgb, source_face=source_face, target_face=target_face,
            settings=BackendSwapSettings(
                swap_model=settings.swap_model, enhancer_mode=settings.enhancer_mode,
                bbox_dilation=int(settings.bbox_dilation), crop_factor=float(settings.crop_factor),
                feather=int(settings.feather),
                identity_strength=float(settings.identity_strength),
            ),
        )

    def swap_faces_with_mask(
        self,
        target_rgb: np.ndarray,
        source_rgb: np.ndarray,
        source_face,
        target_face,
        settings: SelectedSwapSettings,
    ):
        get_content_guard().assert_swap_allowed(
            target_rgb=target_rgb,
            source_rgb=source_rgb,
            target_face=target_face,
            source_face=source_face,
        )
        backend = get_default_swap_backend()
        method = getattr(backend, "swap_with_mask", None)
        if method is None:
            raise RuntimeError("CMK FaceSwap backend does not expose a pasteback mask")
        return method(
            target_rgb=target_rgb,
            source_face=source_face,
            target_face=target_face,
            settings=BackendSwapSettings(
                swap_model=settings.swap_model,
                enhancer_mode=settings.enhancer_mode,
                bbox_dilation=int(settings.bbox_dilation),
                crop_factor=float(settings.crop_factor),
                feather=int(settings.feather),
                identity_strength=float(settings.identity_strength),
            ),
        )

    def swap_selected_with_mask(
        self,
        target_rgb: np.ndarray,
        source_rgb: np.ndarray,
        source_selected_face: Dict[str, Any],
        target_selected_face: Dict[str, Any],
        settings: SelectedSwapSettings,
    ):
        source_face = _raw_face(source_selected_face, "source")
        target_face = _raw_face(target_selected_face, "target")
        return self.swap_faces_with_mask(
            target_rgb,
            source_rgb,
            source_face,
            target_face,
            settings,
        )

    def swap_selected(
        self,
        target_rgb: np.ndarray,
        source_rgb: np.ndarray,
        source_selected_face: Dict[str, Any],
        target_selected_face: Dict[str, Any],
        settings: SelectedSwapSettings,
    ) -> np.ndarray:
        source_face = _raw_face(source_selected_face, "source")
        target_face = _raw_face(target_selected_face, "target")
        return self.swap_faces(target_rgb, source_rgb, source_face, target_face, settings)
