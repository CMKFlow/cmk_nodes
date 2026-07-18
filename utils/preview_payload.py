from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    """Convert numpy / torch / list image-like objects to numpy without assuming type."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _as_rgb_uint8(image: Any) -> np.ndarray:
    arr = _to_numpy(image)

    # Accept Comfy-style [B,H,W,C] arrays/tensors after numpy conversion.
    if arr.ndim == 4:
        arr = arr[0]

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)

    # Defensive CHW -> HWC conversion.
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim != 3 or arr.shape[2] < 1:
        raise RuntimeError(f"Preview image must be image-like, got shape {arr.shape}")

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        if max_value <= 1.5:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    return arr.copy()


def _normalize_images(images: Iterable[Any], *, fallback: bool = True) -> List[np.ndarray]:
    result: List[np.ndarray] = []
    for img in images or []:
        if img is None:
            continue
        try:
            result.append(_as_rgb_uint8(img))
        except Exception:
            # Keep diagnostics robust: one broken preview image must not kill the board.
            continue
    if fallback and not result:
        result = [np.zeros((64, 64, 3), dtype=np.uint8)]
    return result


def _normalize_stages(raw_stages: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_stages, list):
        return []

    stages: List[Dict[str, Any]] = []
    for idx, stage in enumerate(raw_stages):
        title = f"Stage {idx + 1}"
        subtitle = ""
        image = None

        if isinstance(stage, dict):
            title = str(stage.get("title") or stage.get("label") or title)
            subtitle = str(stage.get("subtitle") or stage.get("note") or "")
            image = stage.get("image")
            if image is None:
                image = stage.get("preview")
        elif isinstance(stage, (tuple, list)) and len(stage) >= 2:
            title = str(stage[0] or title)
            image = stage[1]
            subtitle = str(stage[2] if len(stage) >= 3 else "")
        else:
            continue

        try:
            normalized_image = _as_rgb_uint8(image)
        except Exception:
            continue

        stages.append({
            "title": title,
            "subtitle": subtitle,
            "image": normalized_image,
        })

    return stages


def _legacy_stages_from_metadata(payload: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("stages", "preview_steps"):
        raw = payload.get(key)
        if raw is None:
            raw = metadata.get(key)
        stages = _normalize_stages(raw)
        if stages:
            return stages
    return []


def make_preview_payload(
    *,
    title: str,
    node: str,
    images: Iterable[Any] = (),
    stages: Optional[Iterable[Any]] = None,
    summary: str = "",
    mode: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    details: str = "",
) -> Dict[str, Any]:
    """Compatibility wrapper for older nodes.

    New code should prefer CMK_DIAGNOSTIC v2 with ``stages``. This function keeps
    older imports alive while returning the unified renderable payload.
    """
    normalized_stages = _normalize_stages(list(stages or []))
    imgs = _normalize_images(images, fallback=not bool(normalized_stages))
    if not imgs and normalized_stages:
        imgs = [stage["image"] for stage in normalized_stages]
    return {
        "type": "CMK_DIAGNOSTIC",
        "version": 2 if normalized_stages else 1,
        "title": str(title),
        "node": str(node),
        "mode": str(mode or ""),
        "summary": str(summary or ""),
        "details": str(details or summary or ""),
        "metadata": dict(metadata or {}),
        "metrics": {},
        "warnings": [],
        "stages": normalized_stages,
        "preview": imgs,
        "images": imgs,
    }


def normalize_preview_payload(payload: Any) -> Dict[str, Any]:
    """Normalize CMK_PREVIEW and CMK_DIAGNOSTIC into one renderable shape.

    Diagnostic v2 canonicalizes visual information as ``stages``. Legacy
    ``preview`` / ``images`` remain as fallback so older nodes keep rendering.
    """
    if not isinstance(payload, dict):
        raise RuntimeError("Expected preview payload")

    payload_type = payload.get("type")

    if payload_type == "CMK_DIAGNOSTIC":
        metadata = dict(payload.get("metadata", {}) or {})
        stages = _normalize_stages(payload.get("stages"))
        if not stages:
            stages = _legacy_stages_from_metadata(payload, metadata)

        images = payload.get("preview", payload.get("images", []))
        normalized_images = _normalize_images(images, fallback=not bool(stages))
        if stages and (not normalized_images or (len(normalized_images) == 1 and normalized_images[0].shape[:2] == (64, 64))):
            normalized_images = [stage["image"] for stage in stages]

        return {
            "type": "CMK_DIAGNOSTIC",
            "version": int(payload.get("version", 2 if stages else 1)),
            "title": str(payload.get("title", "CMK Diagnostic")),
            "node": str(payload.get("node", "")),
            "mode": str(payload.get("mode", "")),
            "summary": str(payload.get("summary", "")),
            "details": str(payload.get("details", payload.get("summary", ""))),
            "metadata": metadata,
            "metrics": dict(payload.get("metrics", {}) or {}),
            "warnings": [str(w) for w in (payload.get("warnings", []) or []) if str(w).strip()],
            "stages": stages,
            "preview": normalized_images,
            "images": normalized_images,
        }

    if payload_type == "CMK_PREVIEW":
        metadata = dict(payload.get("metadata", {}) or {})
        stages = _legacy_stages_from_metadata(payload, metadata)
        images = payload.get("images", [])
        normalized_images = _normalize_images(images, fallback=not bool(stages))
        if stages and not normalized_images:
            normalized_images = [stage["image"] for stage in stages]
        return {
            "type": "CMK_DIAGNOSTIC",
            "version": 2 if stages else int(payload.get("version", 1)),
            "title": str(payload.get("title", "CMK Preview")),
            "node": str(payload.get("node", "")),
            "mode": str(payload.get("mode", "")),
            "summary": str(payload.get("summary", "")),
            "details": str(payload.get("details", payload.get("summary", ""))),
            "metadata": metadata,
            "metrics": {},
            "warnings": [],
            "stages": stages,
            "preview": normalized_images,
            "images": normalized_images,
        }

    raise RuntimeError("Expected preview payload")


def normalize_diagnostic_payload(payload: Any) -> Dict[str, Any]:
    """Explicit alias for new code."""
    return normalize_preview_payload(payload)


def select_preview_image(payload: Any, image_index: int = 0) -> np.ndarray:
    data = normalize_preview_payload(payload)
    stages = data.get("stages", []) or []
    if stages:
        return stages[0]["image"]
    images: List[np.ndarray] = data.get("images", [])
    if not images:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    idx = max(0, min(int(image_index), len(images) - 1))
    return images[idx]


def preview_summary(payload: Any) -> str:
    data = normalize_preview_payload(payload)
    lines = [
        f"title: {data.get('title', '')}",
        f"node: {data.get('node', '')}",
    ]
    if data.get("mode"):
        lines.append(f"mode: {data.get('mode')}")
    if data.get("stages"):
        lines.append(f"stages: {len(data.get('stages', []))}")
    lines.append(f"images: {len(data.get('images', []))}")
    summary = data.get("summary", "")
    if summary:
        lines.extend(["", str(summary)])
    warnings = data.get("warnings", []) or []
    if warnings:
        lines.extend(["", "warnings:"])
        lines.extend(f"- {w}" for w in warnings)
    return "\n".join(lines)
