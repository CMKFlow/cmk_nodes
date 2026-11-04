from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np


def _tensor_first_rgb_uint8(image) -> np.ndarray:
    """Convert first ComfyUI IMAGE tensor entry to RGB uint8."""
    try:
        import torch
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                image = image[0]
            if image.ndim != 3:
                raise ValueError(f"expected IMAGE tensor [B,H,W,C] or [H,W,C], got {tuple(image.shape)}")
            arr = image.detach().cpu().numpy()
        else:
            arr = np.asarray(image)
            if arr.ndim == 4:
                arr = arr[0]
    except Exception:
        arr = np.asarray(image)
        if arr.ndim == 4:
            arr = arr[0]

    if arr.ndim != 3 or arr.shape[-1] < 3:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        if max_value <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    return arr.copy()


def _as_rgb_uint8(image: Any) -> np.ndarray:
    if image is None:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    return _tensor_first_rgb_uint8(image)


def _normalize_stage(stage: Any, index: int) -> Optional[Dict[str, Any]]:
    """Normalize one Diagnostic v2 stage.

    A stage is a semantic processing step, not a pre-rendered card.
    """
    if stage is None:
        return None

    if isinstance(stage, dict):
        title = stage.get("title") or stage.get("label") or f"Stage {index + 1}"
        subtitle = stage.get("subtitle") or stage.get("note") or ""
        image = stage.get("image")
        if image is None:
            image = stage.get("preview")
    elif isinstance(stage, (tuple, list)) and len(stage) >= 2:
        title = stage[0]
        image = stage[1]
        subtitle = stage[2] if len(stage) >= 3 else ""
    else:
        return None

    if image is None:
        return None

    try:
        image_rgb = _as_rgb_uint8(image)
    except Exception:
        return None

    return {
        "title": str(title),
        "subtitle": str(subtitle),
        "image": image_rgb,
    }


def _normalize_stages(stages: Optional[Iterable[Any]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for idx, stage in enumerate(stages or []):
        normalized = _normalize_stage(stage, idx)
        if normalized is not None:
            out.append(normalized)
    return out


def make_diagnostic_payload(
    *,
    title: str,
    node: str,
    previews: Iterable[Any] = (),
    stages: Optional[Iterable[Any]] = None,
    summary: str = "",
    details: str = "",
    mode: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    warnings: Optional[Iterable[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create one local CMK diagnostic payload.

    Diagnostic v2 adds ``stages`` as the canonical visual model. A stage is a
    semantic processing step such as Source, Detect, Restore, Refine, Final.
    ``preview`` remains for legacy consumers and for nodes not yet migrated.
    """
    normalized_stages = _normalize_stages(stages)
    preview_images = [_as_rgb_uint8(img) for img in previews if img is not None]

    # Legacy fallback: if a node already sends stages but no preview images,
    # expose stage images through preview as well so old boards still show data.
    if not preview_images and normalized_stages:
        preview_images = [stage["image"] for stage in normalized_stages]

    return {
        "type": "CMK_DIAGNOSTIC",
        "version": 2 if normalized_stages else 1,
        "title": str(title),
        "node": str(node),
        "mode": str(mode or ""),
        "summary": str(summary or ""),
        "details": str(details or summary or ""),
        "metadata": dict(metadata or {}),
        "metrics": dict(metrics or {}),
        "warnings": [str(w) for w in (warnings or []) if str(w).strip()],
        "preview": preview_images,
        "images": preview_images,
        "stages": normalized_stages,
    }
