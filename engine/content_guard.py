from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from typing import Any

import numpy as np


GUARD_VERSION = "cmk-content-guard-v1"
MIN_CONFIRMED_ADULT_AGE = 25
EXPLICIT_SCORE_THRESHOLD = 0.35
EXPLICIT_CLASSES = frozenset(
    {
        "ANUS_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
    }
)


class ContentGuardBlocked(RuntimeError):
    """Fail-closed refusal raised before any CMK face swap is executed."""

    def __init__(self, code: str, role: str):
        self.code = str(code)
        self.role = str(role)
        super().__init__(
            "ContentGuard activated — FaceSwap aborted. "
            f"({self.code}, {self.role})"
        )


@dataclass(frozen=True)
class GuardResult:
    role: str
    estimated_age: int
    explicit_detections: tuple[tuple[str, float], ...]


def _rgb_uint8(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ContentGuardBlocked("CG_INPUT_INVALID", "unknown")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _raw_face(face: Any):
    if isinstance(face, dict):
        raw = face.get("raw")
        return raw if raw is not None else face
    return face


def _estimated_age(face: Any, role: str) -> int:
    raw = _raw_face(face)
    value = getattr(raw, "age", None)
    try:
        age = int(round(float(value)))
    except Exception as exc:
        raise ContentGuardBlocked("CG_AGE_UNAVAILABLE", role) from exc
    if age < 0 or age > 120:
        raise ContentGuardBlocked("CG_AGE_INVALID", role)
    if age < 18:
        raise ContentGuardBlocked("CG_AGE_MINOR", role)
    if age < MIN_CONFIRMED_ADULT_AGE:
        raise ContentGuardBlocked("CG_AGE_UNCERTAIN", role)
    return age


@lru_cache(maxsize=1)
def _nudity_detector():
    try:
        from nudenet import NudeDetector

        return NudeDetector()
    except Exception as exc:
        raise ContentGuardBlocked("CG_EXPLICIT_GUARD_UNAVAILABLE", "unknown") from exc


@lru_cache(maxsize=8)
def _explicit_detections(image_hash: str, image_bytes: bytes, shape: tuple[int, int, int]):
    del image_hash
    try:
        rgb = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)
        # NudeNet accepts OpenCV arrays and therefore expects BGR channel order.
        detections = _nudity_detector().detect(np.ascontiguousarray(rgb[:, :, ::-1]))
    except ContentGuardBlocked:
        raise
    except Exception as exc:
        raise ContentGuardBlocked("CG_EXPLICIT_INFERENCE_FAILED", "unknown") from exc

    blocked = []
    for detection in detections or []:
        label = str(detection.get("class", "")).upper()
        score = float(detection.get("score", 0.0) or 0.0)
        if label in EXPLICIT_CLASSES and score >= EXPLICIT_SCORE_THRESHOLD:
            blocked.append((label, score))
    return tuple(sorted(blocked))


def _inspect_explicit(image_rgb: np.ndarray, role: str) -> tuple[tuple[str, float], ...]:
    rgb = _rgb_uint8(image_rgb)
    payload = rgb.tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        detections = _explicit_detections(digest, payload, tuple(rgb.shape))
    except ContentGuardBlocked as exc:
        if exc.role == "unknown":
            raise ContentGuardBlocked(exc.code, role) from exc
        raise
    if detections:
        raise ContentGuardBlocked("CG_EXPLICIT_CONTENT", role)
    return detections


class CMKContentGuard:
    """Mandatory local guard shared by all public CMK FaceSwap paths."""

    def inspect_image(self, image_rgb: np.ndarray, face: Any, role: str) -> GuardResult:
        explicit = _inspect_explicit(image_rgb, role)
        age = _estimated_age(face, role)
        return GuardResult(role=str(role), estimated_age=age, explicit_detections=explicit)

    def inspect_content(self, image_rgb: np.ndarray, role: str) -> None:
        _inspect_explicit(image_rgb, role)

    def assert_swap_allowed(
        self,
        *,
        target_rgb: np.ndarray,
        source_rgb: np.ndarray | None,
        target_face: Any,
        source_face: Any,
    ) -> tuple[GuardResult, GuardResult]:
        if source_rgb is None:
            raise ContentGuardBlocked("CG_SOURCE_IMAGE_REQUIRED", "source")
        source = self.inspect_image(source_rgb, source_face, "source")
        target = self.inspect_image(target_rgb, target_face, "target")
        return source, target


@lru_cache(maxsize=1)
def get_content_guard() -> CMKContentGuard:
    return CMKContentGuard()
