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


# --------------------------------------------------------------------------
# CMK design rule (2.8): Preview / Compare show what came out; a diagnostic
# shows where, how and with which effect it came out.
#
# A diagnostic does not show a plain full image that only exists as a step of
# the processing: the source, a primary or Refine result, the final result or
# the ordinary preview of a node.  Those belong to Preview / Compare.  The
# producers keep them in their payloads (caches and other consumers read
# them); the diagnostic renderers leave them out.  Detection, ROI, masks,
# alpha, pasteback, overlaps, shares and differences stay.
#
# Classified per (node, stage title), because the same title can carry
# technical content elsewhere (the "Source" of CMK Face Crop shows the
# selection with boxes and landmarks).
# --------------------------------------------------------------------------

PLAIN_IMAGE_STAGES = {
    "CMK Smart Detailer": {"Source", "Detailer", "Final"},
    "CMK FaceProcess": {"Source", "Primary Restore", "Primary Detailer", "Primary Off",
                        "Refine Detail", "Refine Sharpen", "Refine Smooth", "Final"},
    "CMKImageResize": {"Source", "Resized"},
    "CMKMaskDetailerPrepare": {"Source"},
    "CMKMaskDetailerProcess": {"Source", "Result"},
    "CMK Smart Outpaint Pad": {"Source"},
    "CMK Smart Upscaler": {"Source", "Final"},
    "CMK InstantID Face Detailer SDXL": {"Rebuilt ROI", "Bypass"},
    "CMK InstantID Face Rebuild Prepare": {"Bypass"},
    "CMK InstantID Face Rebuild Pasteback": {"Final Pasteback", "Bypass"},
    "CMKInstantIDFaceRebuildAdvancedSDXL": {"Bypass"},
    "CMKInstantIDFaceRebuildSDXL": {"Bypass"},
    "CMK Face Select": {"No Detection"},
    "CMK FaceSwap Image": {"Target", "Source", "Swap Result", "Passthrough", "Final"},
    "CMK FaceSwap Image -Pipe-": {"Target", "Source", "Swap Result", "Passthrough", "Final"},
    "CMK Face Crop": {"Cropped Image"},
    "CMK Merge and Save Video": {"MERGED VIDEO"},
    "CMK Split Video into Segments": {"SOURCE VIDEO"},
    "CMK Detailer Finalize -Pipe-": {"Source", "Final"},
    "CMK Pipe Create Image -Pipe-": {"Source"},
    "CMK Z-Image Turbo Finalize -Pipe-": {"Z-Image Turbo Result"},
    "CMK Swap Image Loader -Pipe-": {"SOURCE", "TARGET"},
    "CMK ControlNet Prepare": {"Source"},
    "CMK ControlNet Prepare -Pipe-": {"Source"},
    "CMK Detailer Merge Multi-Model": {"FINAL"},
    "CMK FaceProcess Merge Multi-Model": {"FINAL"},
}

# Nodes whose diagnostic without stages carries only the ordinary preview
# (the input or output image itself).
PLAIN_PREVIEW_NODES = frozenset({
    "CMK Image Metrics",
    "CMK Smart Detailer -Pipe-",
    "CMK Detailer Prepare -Pipe-",
    "CMK Detailer Prepare Multi-Model -Pipe-",
    "CMKMaskDetailerPrepareMultiModel",
    "CMKMaskDetailerProcessMultiModel",
    "CMK FaceRebuild v2 · Prepare",
    "CMK FaceRebuild v2",
    "CMK FaceRebuild v2 · Advanced",
    "CMK FaceSwap v2 · Prepare",
    "CMK FaceSwap v2",
    "CMK FaceSwap v2 · Advanced",
    "CMK Detailer Prepare Advanced Multi-Model -Pipe-",
    "CMK FaceProcess -Pipe-",
    "CMK FaceProcess Prepare -Pipe-",
    "CMK FaceProcess Prepare Multi-Model -Pipe-",
    "CMK FaceProcess Prepare Advanced Multi-Model -Pipe-",
    "CMK FaceProcess Multi-Model -Pipe-",
    "CMK Boundary Cache / Detailer",
    "CMK Boundary Cache / FaceProcess",
    "CMK Sampler Prepare SDXL -Pipe-",
    "CMK Image Load and Resize -Pipe-",
})


def bare_stage_title(title) -> str:
    """Stage title without its running number ("05 Final", "04 · FINAL")."""
    text = str(title or "").strip()
    head, _, rest = text.partition(" ")
    if head.isdigit() and rest:
        text = rest.strip()
        if text.startswith("·"):
            text = text[1:].strip()
    return text


def plain_image_stage(node, title) -> bool:
    """Whether the stage ``title`` of a diagnostic from ``node`` is a plain full image."""
    return bare_stage_title(title) in PLAIN_IMAGE_STAGES.get(str(node or ""), ())


def plain_preview_node(node) -> bool:
    return str(node or "") in PLAIN_PREVIEW_NODES
