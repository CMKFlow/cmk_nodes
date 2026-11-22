"""Spatial SEGS diagnostics (2.8.0): pure, model-independent image functions.

Everything here is derived from data a Detailer already produced: the source
IMAGE, the SEG fields ``bbox``, ``crop_region`` and ``cropped_mask`` and the
cache-stable branch of CMK stable SEGS.  Nothing detects, segments or samples
again, and no function receives or keeps a model object.

The geometry reuses the functions the image path itself uses:

* the sampling ROI and the merge support are ``crop_region``, rounded and
  clipped exactly like ``SEGSPaste`` and ``_spatial_support_mask``;
* the mask projection resizes ``cropped_mask`` with ``_mask_tensor`` and
  feathers it with ``_soft_mask``, the helpers of ``SEGSPaste.doit``;
* the merge diagnostic follows ``merge_segs_collections_masked``: slot order,
  every branch alpha-blended with its effective pasteback weight.

The diagnostic shows the first image of a batch.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..engine.native_detailer import _mask_tensor, _soft_mask
from .cmk_diagnostic import _normalize_stages
from .stable_segs import BRANCH_PASTE_ALPHA, BRANCH_PASTE_FEATHER, stable_branch_components

SPATIAL_FLAG = "spatial_stages"
SLOT_COLORS = {1: (0, 190, 255), 2: (255, 70, 200), 3: (255, 205, 0)}
DETECTION_COLOR = (255, 255, 255)
ROI_COLOR = (255, 140, 0)
MASK_COLOR = (0, 230, 160)
OVERLAP_COLOR = (235, 235, 235)
MASK_OVERLAP_COLOR = (255, 40, 40)
FAMILY_LABELS = {"sdxl": "SDXL", "z_image_turbo": "ZIT", "none": "NONE", "unknown": "?"}


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _valid_segs(value) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], tuple)
        and isinstance(value[1], list)
    )


def _first_image_segs(segs) -> list:
    """SEGs that belong to the first batch image (``batch_index`` None or 0)."""
    if not _valid_segs(segs):
        return []
    out = []
    for seg in segs[1]:
        index = getattr(seg, "batch_index", None)
        if index is None or int(index) == 0:
            out.append(seg)
    return out


def _rect(box, width: int, height: int):
    """Round and clip like SEGSPaste / _spatial_support_mask."""
    if box is None or len(box) < 4:
        return None
    try:
        x1 = max(0, min(width, int(round(float(box[0])))))
        y1 = max(0, min(height, int(round(float(box[1])))))
        x2 = max(0, min(width, int(round(float(box[2])))))
        y2 = max(0, min(height, int(round(float(box[3])))))
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def roi_rect(seg, width: int, height: int):
    """Sampling ROI = the pasteback region ``crop_region``."""
    return _rect(getattr(seg, "crop_region", None), width, height)


def bbox_rect(seg, width: int, height: int):
    return _rect(getattr(seg, "bbox", None), width, height)


def projected_mask(segs, height: int, width: int, *, feather: int = 0, alpha: int = 255) -> torch.Tensor:
    """Full-image weight of the SEG masks for the first batch image.

    With ``feather``/``alpha`` of the pasteback this is the weight with which
    ``SEGSPaste.doit`` takes processed pixels: per SEG ``_soft_mask(mask) *
    alpha`` inside ``crop_region``; overlapping SEGs paste sequentially, so the
    processed share is ``1 - prod(1 - m_i)``.  ``feather=0`` gives the raw
    ``cropped_mask`` projection.
    """
    keep = torch.ones((height, width), dtype=torch.float32)
    alpha_scale = max(0.0, min(1.0, float(alpha) / 255.0))
    for seg in _first_image_segs(segs):
        region = roi_rect(seg, width, height)
        if region is None:
            continue
        x1, y1, x2, y2 = region
        mask = _mask_tensor(
            getattr(seg, "cropped_mask", None),
            batch=1, height=y2 - y1, width=x2 - x1,
            device="cpu", dtype=torch.float32,
        )
        if int(feather) > 0:
            mask = _soft_mask(mask, int(feather))
        mask = mask[0] * alpha_scale
        keep[y1:y2, x1:x2] = keep[y1:y2, x1:x2] * (1.0 - mask)
    return (1.0 - keep).clamp(0.0, 1.0)


def roi_support(segs, height: int, width: int) -> torch.Tensor:
    """Union of the ``crop_region`` rectangles (bool [H, W])."""
    mask = torch.zeros((height, width), dtype=torch.bool)
    for seg in _first_image_segs(segs):
        region = roi_rect(seg, width, height)
        if region is not None:
            x1, y1, x2, y2 = region
            mask[y1:y2, x1:x2] = True
    return mask


def merge_weights(segs_collections: Sequence, height: int, width: int, *, feather: int = 5, alpha: int = 255) -> Dict[int, torch.Tensor]:
    """Per slot the weight with which ``merge_segs_collections_masked`` pastes it.

    A stable branch is pasted with its Detailer's pasteback values, ordinary
    SEGS with the merge's ``feather``/``alpha``.
    """
    out = {}
    for slot, segs in enumerate(segs_collections, start=1):
        if not _valid_segs(segs) or not segs[1]:
            continue
        if stable_branch_components(segs) is not None:
            out[slot] = projected_mask(segs, height, width, feather=BRANCH_PASTE_FEATHER, alpha=BRANCH_PASTE_ALPHA)
        else:
            out[slot] = projected_mask(segs, height, width, feather=feather, alpha=alpha)
    return out


def alpha_shares(weights: Dict[int, torch.Tensor], height: int, width: int):
    """Share of each slot and of the source in every merged pixel.

    Slot order D1 -> D2 -> D3, ``result = previous * (1 - w) + branch * w``:
    a later weight scales down every earlier share.  The shares sum to 1.
    """
    source = torch.ones((height, width), dtype=torch.float32)
    shares = {}
    for slot in sorted(weights):
        keep = 1.0 - weights[slot]
        for earlier in shares:
            shares[earlier] = shares[earlier] * keep
        source = source * keep
        shares[slot] = weights[slot].clone()
    return shares, source


def mask_overlaps(weights: Dict[int, torch.Tensor]):
    """Pairs of slots whose effective pasteback masks overlap (bool [H, W])."""
    slots = sorted(weights)
    return {
        (earlier, later): (weights[earlier] > 0) & (weights[later] > 0)
        for index, earlier in enumerate(slots) for later in slots[index + 1:]
        if bool(((weights[earlier] > 0) & (weights[later] > 0)).any())
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _rgb(image) -> np.ndarray:
    tensor = image[0] if hasattr(image, "ndim") and image.ndim == 4 else image
    arr = tensor.detach().to("cpu", torch.float32).numpy()[..., :3]
    return np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32) / 255.0)[None,]


def _dim(arr: np.ndarray, factor: float = 0.45, gray: bool = False) -> np.ndarray:
    value = arr.astype(np.float32)
    if gray:
        value = np.repeat(value.mean(axis=2, keepdims=True), 3, axis=2)
    return value * factor


def _tint(base: np.ndarray, weight: np.ndarray, color, strength: float) -> np.ndarray:
    w = (np.clip(weight, 0.0, 1.0) * strength)[..., None]
    return base * (1.0 - w) + np.asarray(color, dtype=np.float32) * w


def _thickness(height: int, width: int) -> int:
    return max(2, int(round(max(height, width) / 400)))


def _draw_rect(arr: np.ndarray, rect, color, thickness: int, dashed: bool = False) -> None:
    if rect is None:
        return
    height, width = arr.shape[:2]
    x1, y1, x2, y2 = rect
    x2, y2 = min(width, x2), min(height, y2)
    dash = max(6, thickness * 4)

    def keep(n):
        mask = np.ones(n, dtype=bool)
        if dashed:
            mask = (np.arange(n) // dash) % 2 == 0
        return mask

    t = thickness
    row = keep(x2 - x1)
    col = keep(y2 - y1)
    for y in range(y1, min(y1 + t, y2)):
        arr[y, x1:x2][row] = color
    for y in range(max(y2 - t, y1), y2):
        arr[y, x1:x2][row] = color
    for x in range(x1, min(x1 + t, x2)):
        arr[y1:y2, x][col] = color
    for x in range(max(x2 - t, x1), x2):
        arr[y1:y2, x][col] = color


def _slot_rect(rect, slot: int, thickness: int):
    """Frame of a slot, inset per slot so identical ROIs stay distinguishable.

    Drawing only: the geometry itself is ``crop_region``.
    """
    if rect is None:
        return None
    inset = (slot - 1) * (thickness + 1)
    x1, y1, x2, y2 = rect
    if x2 - x1 <= 2 * inset + 2 or y2 - y1 <= 2 * inset + 2:
        return rect
    return x1 + inset, y1 + inset, x2 - inset, y2 - inset


def _font(height: int):
    return ImageFont.load_default(size=max(12, int(height / 55)))


def _draw_labels(arr: np.ndarray, labels: Iterable[tuple]) -> np.ndarray:
    """Draw ``(x, y, text, color)`` tags; returns a new uint8 array."""
    labels = list(labels)
    out = np.clip(arr, 0, 255).astype(np.uint8)
    if not labels:
        return out
    image = Image.fromarray(out)
    draw = ImageDraw.Draw(image)
    font = _font(out.shape[0])
    for x, y, text, color in labels:
        box = draw.textbbox((x, y), text, font=font)
        pad = 3
        x0 = max(0, min(out.shape[1] - (box[2] - box[0]) - 2 * pad, box[0]))
        y0 = max(0, min(out.shape[0] - (box[3] - box[1]) - 2 * pad, box[1]))
        shift_x, shift_y = x0 - box[0], y0 - box[1]
        draw.rectangle((x0 - pad, y0 - pad, box[2] + shift_x + pad, box[3] + shift_y + pad), fill=(0, 0, 0))
        draw.text((x + shift_x, y + shift_y), text, fill=tuple(color), font=font)
    return np.asarray(image).copy()


def _legend(height: int, items: Sequence[tuple]) -> List[tuple]:
    step = max(16, int(height / 55)) + 10
    return [(8, 8 + index * step, text, color) for index, (text, color) in enumerate(items)]


# --------------------------------------------------------------------------
# Smart Detailer stages
# --------------------------------------------------------------------------

def detailer_spatial_stages(image, segs, *, feather: int = 5, alpha: int = 255, bbox_label: str = "DETECTION BOX",
                            mask_label: str = "cropped_mask", result=None) -> List[Dict[str, Any]]:
    """SAMPLING ROI, MASK and PASTEBACK of one Detailer from its SEGS.

    ``bbox_label`` names the SEG ``bbox`` (Mask Detailer: the painted mask's
    bounds).  With ``result`` the PASTEBACK stage shows the pasted result
    weighted with the pasteback weight instead of the bare weight.
    """
    if image is None or not _first_image_segs(segs):
        return []
    base = _rgb(image)
    height, width = base.shape[:2]
    t = _thickness(height, width)
    items = _first_image_segs(segs)
    support = roi_support(segs, height, width).numpy()

    roi = _dim(base, 0.4)
    roi[support] = base[support]
    roi = roi.astype(np.uint8)
    for seg in items:
        _draw_rect(roi, roi_rect(seg, width, height), ROI_COLOR, t, dashed=True)
        _draw_rect(roi, bbox_rect(seg, width, height), DETECTION_COLOR, t)
    roi = _draw_labels(roi, _legend(height, [(bbox_label, DETECTION_COLOR), ("SAMPLING ROI (crop_region)", ROI_COLOR)]))

    raw = projected_mask(segs, height, width, feather=0, alpha=255).numpy()
    mask = _tint(_dim(base, 0.45), raw, MASK_COLOR, 0.85).astype(np.uint8)
    for seg in items:
        _draw_rect(mask, roi_rect(seg, width, height), ROI_COLOR, t, dashed=True)
    mask = _draw_labels(mask, _legend(height, [(mask_label, MASK_COLOR), ("SAMPLING ROI", ROI_COLOR)]))

    weight = projected_mask(segs, height, width, feather=feather, alpha=alpha).numpy()
    if result is None:
        paste = np.repeat((weight * 255.0)[..., None], 3, axis=2)
        paste = (paste * 0.75 + _dim(base, 0.25)).astype(np.float32)
        paste_legend = "PASTEBACK WEIGHT (white = 1)"
    else:
        pasted = _rgb(result).astype(np.float32)
        paste = pasted * weight[..., None] + _dim(pasted, 0.3, gray=True) * (1.0 - weight[..., None])
        paste_legend = "PASTED RESULT x PASTEBACK WEIGHT"
    paste = paste.clip(0, 255).astype(np.uint8)
    for seg in items:
        _draw_rect(paste, roi_rect(seg, width, height), ROI_COLOR, t, dashed=True)
    paste = _draw_labels(paste, _legend(height, [(paste_legend, (255, 255, 255)), ("SAMPLING ROI", ROI_COLOR)]))

    roi_px = int(support.sum())
    mask_px = int((raw > 0).sum())
    paste_px = int((weight > 0).sum())
    share = (100.0 * float(weight.sum()) / roi_px) if roi_px else 0.0
    return [
        {"title": "Sampling ROI", "image": _to_tensor(roi),
         "subtitle": f"ROI {roi_px} px"},
        {"title": "Mask", "image": _to_tensor(mask),
         "subtitle": f"mask {mask_px} px"},
        {"title": "Pasteback", "image": _to_tensor(paste),
         "subtitle": f"feather {int(feather)} · {share:.1f}% of ROI"},
    ]


def _renumber(title: str, index: int) -> str:
    text = str(title)
    head, _, rest = text.partition(" ")
    if head.isdigit() and rest:
        text = rest
    return f"{index:02d} {text}"


def with_spatial_stages(payload, image, segs, *, feather: int = 5, alpha: int = 255):
    """Insert the spatial stages after the Detection stage of a diagnostic.

    Idempotent (``metadata[SPATIAL_FLAG]``).  Also completes diagnostics read
    from a branch cache written before 2.8.0, from the cached SEGS.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), list):
        return payload
    metadata = dict(payload.get("metadata") or {})
    if metadata.get(SPATIAL_FLAG):
        return payload
    stages = detailer_spatial_stages(image, segs, feather=feather, alpha=alpha)
    if not stages:
        return payload

    existing = list(payload["stages"])
    position = next(
        (index + 1 for index, stage in enumerate(existing) if str(stage.get("title", "")).endswith("Detection")),
        min(1, len(existing)),
    )
    merged = existing[:position] + _normalize_stages(stages) + existing[position:]
    merged = [dict(stage, title=_renumber(stage.get("title", ""), index)) for index, stage in enumerate(merged, start=1)]
    metadata[SPATIAL_FLAG] = True
    result = dict(payload)
    result["stages"] = merged
    result["preview"] = [stage["image"] for stage in merged]
    result["images"] = result["preview"]
    result["metadata"] = metadata
    result["version"] = 2
    return result


# --------------------------------------------------------------------------
# Merge stages
# --------------------------------------------------------------------------

def _slot_label(slot: int, families: Dict[int, str]) -> str:
    return f"D{slot} · {FAMILY_LABELS.get(families.get(slot, 'unknown'), '?')}"


def merge_diagnostic_stages(image, result_image, segs_collections: Sequence, families: Dict[int, str],
                            *, feather: int = 5, alpha: int = 255):
    """``01 · INPUT SEGS`` … ``04 · FINAL`` and metrics of the Advanced Merge."""
    base = _rgb(image)
    height, width = base.shape[:2]
    t = _thickness(height, width)
    weights = merge_weights(segs_collections, height, width, feather=feather, alpha=alpha)
    shares, source_share = alpha_shares(weights, height, width)
    overlaps = mask_overlaps(weights)
    roi_count = torch.zeros((height, width), dtype=torch.int64)
    for slot in weights:
        roi_count += roi_support(segs_collections[slot - 1], height, width).to(torch.int64)
    roi_overlap = roi_count >= 2

    def frames(arr, dashed):
        for slot, segs in enumerate(segs_collections, start=1):
            for seg in _first_image_segs(segs):
                _draw_rect(arr, _slot_rect(roi_rect(seg, width, height), slot, t), SLOT_COLORS[slot], t, dashed=dashed)

    # 01 · INPUT SEGS: per slot its effective pasteback mask and its ROIs.
    inputs = _dim(base, 0.5)
    for slot, weight in weights.items():
        inputs = _tint(inputs, weight.numpy(), SLOT_COLORS[slot], 0.6)
    inputs = inputs.astype(np.uint8)
    frames(inputs, dashed=True)
    step = max(16, int(height / 55)) + 10
    labels = []
    for slot, segs in enumerate(segs_collections, start=1):
        for seg in _first_image_segs(segs):
            rect = _slot_rect(roi_rect(seg, width, height), slot, t)
            if rect is not None:
                labels.append((rect[0] + t + 2, rect[1] + t + 2 + (slot - 1) * step, f"D{slot}", SLOT_COLORS[slot]))
    legend = [(_slot_label(slot, families) + ("" if slot in weights else " · no SEGS"), SLOT_COLORS[slot]) for slot in (1, 2, 3)]
    legend.append(("fill = pasteback mask · dashed = sampling ROI", (220, 220, 220)))
    inputs = _draw_labels(inputs, _legend(height, legend) + labels)

    # 02 · OVERLAP: ROI overlap (reference only) and overlapping masks (blend).
    overlap = _dim(base, 0.35, gray=True)
    overlap = _tint(overlap, roi_overlap.numpy().astype(np.float32), OVERLAP_COLOR, 0.3)
    blend = torch.zeros((height, width), dtype=torch.float32)
    for (earlier, later), area in overlaps.items():
        blend = torch.maximum(blend, torch.minimum(weights[earlier], weights[later]) * area)
    overlap = _tint(overlap, np.where(blend.numpy() > 0, 0.35 + 0.65 * blend.numpy(), 0.0), MASK_OVERLAP_COLOR, 0.95)
    overlap = overlap.astype(np.uint8)
    frames(overlap, dashed=True)
    pairs = [f"D{earlier} x D{later}: {int(area.sum())} px" for (earlier, later), area in sorted(overlaps.items())]
    overlap = _draw_labels(overlap, _legend(height, [
        ("ROI OVERLAP (no effect)", OVERLAP_COLOR), ("MASK OVERLAP (alpha blend, later on top)", MASK_OVERLAP_COLOR),
    ] + [(text, MASK_OVERLAP_COLOR) for text in pairs]))

    # 03 · COMPOSITE: color mix of the exact per-pixel shares.
    gray = _dim(base, 0.4, gray=True)
    composite = gray * source_share.numpy()[..., None]
    for slot, share in shares.items():
        composite = composite + np.asarray(SLOT_COLORS[slot], dtype=np.float32) * share.numpy()[..., None]
    effective = {slot: float(share.sum()) for slot, share in shares.items()}
    composite = _draw_labels(composite.clip(0, 255).astype(np.uint8), _legend(height, [
        ("ORDER D1 > D2 > D3 · ALPHA BLEND, LATER ON TOP", (255, 255, 255)),
    ] + [(f"{_slot_label(slot, families)}: {effective.get(slot, 0.0):.0f} px", SLOT_COLORS[slot]) for slot in (1, 2, 3)]
      + [(f"SOURCE: {float(source_share.sum()):.0f} px", (170, 170, 170))]))

    mask_overlap_px = int(sum(int(area.sum()) for area in overlaps.values()))
    stages = [
        {"title": "01 · INPUT SEGS", "image": _to_tensor(inputs),
         "subtitle": "fill = mask · dashed = ROI"},
        {"title": "02 · OVERLAP", "image": _to_tensor(overlap),
         "subtitle": f"mask overlap {mask_overlap_px} px · ROI {int(roi_overlap.sum())} px"},
        {"title": "03 · COMPOSITE", "image": _to_tensor(composite),
         "subtitle": "alpha shares · D1 → D2 → D3"},
        {"title": "04 · FINAL", "image": result_image, "subtitle": "merged result"},
    ]
    metrics = {
        "share_px": {f"D{slot}": round(effective.get(slot, 0.0), 1) for slot in (1, 2, 3)},
        "source_px": round(float(source_share.sum()), 1),
        "roi_overlap_px": int(roi_overlap.sum()),
        "mask_overlap_px": {f"D{earlier}xD{later}": int(area.sum()) for (earlier, later), area in sorted(overlaps.items())},
        "reduced_by_later_px": {
            f"D{slot}": round(float((weights[slot] - shares[slot]).sum()), 1) for slot in sorted(weights)
            if float((weights[slot] - shares[slot]).sum()) > 0
        },
    }
    return stages, metrics
