from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Protocol

import numpy as np

from ..models.model_manager import resolve_swap_model
from .pasteback import pasteback_native
from .enhance_backends import get_enhancer_backend, validate_enhancer_mode

MIN_CROP_FACTOR = 1.0
MAX_CROP_FACTOR = 3.0


@dataclass(frozen=True)
class BackendSwapSettings:
    swap_model: str
    enhancer_mode: str = "GPEN"
    bbox_dilation: int = 0
    crop_factor: float = 1.0
    feather: int = 0
    identity_strength: float = 1.0


class FaceSwapBackend(Protocol):
    def swap(
        self,
        *,
        target_rgb: np.ndarray,
        source_face,
        target_face,
        settings: BackendSwapSettings,
    ) -> np.ndarray:
        ...


def _providers() -> List[str]:
    return ["CPUExecutionProvider"]


@lru_cache(maxsize=4)
def _get_inswapper(swap_model: str):
    try:
        from insightface import model_zoo
    except Exception as exc:
        raise ImportError(
            "Missing dependency: insightface. Install it in the ComfyUI Python environment."
        ) from exc

    model_path = resolve_swap_model(swap_model)
    return model_zoo.get_model(model_path, providers=_providers())


def _extract_aligned_result(value):
    """Return aligned patch and the exact affine matrix produced by INSwapper."""
    if isinstance(value, (tuple, list)):
        patch = value[0] if len(value) >= 1 else None
        matrix = value[1] if len(value) >= 2 else None
        return patch, matrix
    return value, None


def _identity_latent(swapper, source_face, strength: float) -> np.ndarray:
    """Build INSwapper's normalized identity latent and apply its strength."""
    strength = min(1.5, max(0.5, float(strength)))
    embedding = np.asarray(source_face.normed_embedding, dtype=np.float32).reshape((1, -1))
    latent = np.dot(embedding, swapper.emap)
    norm = float(np.linalg.norm(latent))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise RuntimeError("INSwapper produced an invalid identity latent")
    return ((latent / norm) * strength).astype(np.float32)


def _get_aligned_swap(swapper, target_rgb: np.ndarray, target_face, source_face, strength: float):
    """Run aligned INSwapper inference with a weighted identity latent."""
    strength = min(1.5, max(0.5, float(strength)))
    if abs(strength - 1.0) < 1e-6:
        return swapper.get(target_rgb.copy(), target_face, source_face, paste_back=False)

    try:
        import cv2
        from insightface.utils import face_align
    except Exception as exc:
        raise RuntimeError(f"Identity-strength preprocessing is unavailable: {exc}") from exc

    aligned, matrix = face_align.norm_crop2(
        target_rgb.copy(),
        target_face.kps,
        swapper.input_size[0],
    )
    blob = cv2.dnn.blobFromImage(
        aligned,
        1.0 / swapper.input_std,
        swapper.input_size,
        (swapper.input_mean, swapper.input_mean, swapper.input_mean),
        swapRB=True,
    )
    latent = _identity_latent(swapper, source_face, strength)
    prediction = swapper.session.run(
        swapper.output_names,
        {swapper.input_names[0]: blob, swapper.input_names[1]: latent},
    )[0]
    image = prediction.transpose((0, 2, 3, 1))[0]
    return np.clip(255 * image, 0, 255).astype(np.uint8)[:, :, ::-1], matrix


class INSwapperBackend:
    """Shared INSwapper backend for Image, Image -Pipe-, and Video."""

    name = "inswapper"

    def _swap_impl(
        self,
        *,
        target_rgb: np.ndarray,
        source_face,
        target_face,
        settings: BackendSwapSettings,
        return_mask: bool = False,
    ):
        canonical_mode = validate_enhancer_mode(settings.enhancer_mode)
        crop_factor = min(MAX_CROP_FACTOR, max(MIN_CROP_FACTOR, float(settings.crop_factor)))
        swapper = _get_inswapper(settings.swap_model)

        try:
            raw_aligned = _get_aligned_swap(
                swapper,
                target_rgb,
                target_face,
                source_face,
                settings.identity_strength,
            )
        except Exception as exc:
            raise RuntimeError(f"INSwapper aligned swap failed: {exc}") from exc

        aligned_result, affine_matrix = _extract_aligned_result(raw_aligned)
        if aligned_result is None:
            raise RuntimeError("INSwapper aligned swap returned no face patch")
        aligned_result = np.clip(aligned_result, 0, 255).astype(np.uint8)

        try:
            enhancer = get_enhancer_backend(canonical_mode)
            enhanced_result = enhancer.enhance_aligned(aligned_result)
        except Exception as exc:
            raise RuntimeError(
                f"{canonical_mode} enhancer failed before paste-back: {exc}"
            ) from exc

        enhanced_result = np.clip(enhanced_result, 0, 255).astype(np.uint8)
        if enhanced_result.shape != aligned_result.shape:
            try:
                import cv2

                enhanced_result = cv2.resize(
                    enhanced_result,
                    (int(aligned_result.shape[1]), int(aligned_result.shape[0])),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{canonical_mode} enhancer returned incompatible dimensions "
                    f"{enhanced_result.shape}; expected {aligned_result.shape}: {exc}"
                ) from exc

        if enhanced_result.shape != aligned_result.shape:
            raise RuntimeError(
                f"{canonical_mode} enhancer returned incompatible dimensions "
                f"{enhanced_result.shape}; expected {aligned_result.shape}"
            )

        return pasteback_native(
            target_rgb,
            enhanced_result,
            target_face,
            affine_matrix=affine_matrix,
            bbox_dilation=settings.bbox_dilation,
            crop_factor=crop_factor,
            feather=settings.feather,
            return_mask=return_mask,
        )

    def swap(
        self,
        *,
        target_rgb: np.ndarray,
        source_face,
        target_face,
        settings: BackendSwapSettings,
    ) -> np.ndarray:
        return self._swap_impl(
            target_rgb=target_rgb,
            source_face=source_face,
            target_face=target_face,
            settings=settings,
            return_mask=False,
        )

    def swap_with_mask(
        self,
        *,
        target_rgb: np.ndarray,
        source_face,
        target_face,
        settings: BackendSwapSettings,
    ):
        return self._swap_impl(
            target_rgb=target_rgb,
            source_face=source_face,
            target_face=target_face,
            settings=settings,
            return_mask=True,
        )


@lru_cache(maxsize=1)
def get_default_swap_backend() -> FaceSwapBackend:
    return INSwapperBackend()
