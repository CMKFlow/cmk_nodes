"""Spatial diagnostics of FaceProcess 2.8 branches and the Advanced merge (F5).

Pure image functions over data a branch already produced: the source image,
the YOLO detection and selection, the RESTORE assignment records and the F4
processing layers (``cmk_layers``).  Nothing detects, restores, samples or
composes again, and the image result is never touched.

The effective weight of a layer is the share of processed pixels its blend
takes, exactly as ``face_layers._apply`` blends:

``restore``   pasteback alpha ``a`` and visibility ``v``: ``X*(1 - a*v) + W*a*v``
``detailer``  feathered SEGSPaste mask ``A``
``refine``    ``0.75 * ellipse``

Layers of one branch apply one after another, so the branch's processed
share is ``1 - prod(1 - w)``; ``quantize`` (the 2.7 uint8 conversion, at
most 1/255) carries no weight.  The diagnostic shows the first image of a batch.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from .cmk_diagnostic import _normalize_stages
from .segs_diagnostic import (
    DETECTION_COLOR,
    MASK_COLOR,
    MASK_OVERLAP_COLOR,
    OVERLAP_COLOR,
    ROI_COLOR,
    SLOT_COLORS,
    _dim,
    _draw_labels,
    _draw_rect,
    _first_image_segs,
    _legend,
    _rgb,
    _slot_rect,
    _thickness,
    _tint,
    _to_tensor,
    alpha_shares,
    bbox_rect,
    mask_overlaps,
    projected_mask,
    roi_rect,
)

FACE_STAGES_FLAG = "face_spatial_stages"
SELECTED_COLOR = (0, 230, 160)
STATUS_COLORS = {"OK": (0, 230, 160), "UNMATCHED": (255, 140, 0), "AMBIGUOUS": (255, 40, 40), "CONFLICT": (255, 40, 40)}
REFINE_COLOR = (180, 120, 255)
MODE_LABELS = {"restore": "RESTORE", "sdxl": "SDXL", "z_image_turbo": "ZIT", "none": "NONE", "unknown": "?"}


def _first(layers, kinds):
    return [layer for layer in layers if layer["kind"] in kinds and layer.get("batch_index") in (None, 0)]


def layer_weight(layers, height: int, width: int, kinds=("restore", "detailer", "refine"), *, visibility: bool = True) -> np.ndarray:
    """Processed share of ``layers`` of ``kinds`` in every pixel (float32 [H, W])."""
    keep = np.ones((height, width), dtype=np.float32)
    for layer in _first(layers, kinds):
        x1, y1, x2, y2 = layer["roi"]
        alpha = np.asarray(layer["alpha"], dtype=np.float32)
        if layer["kind"] == "restore" and visibility:
            alpha = alpha * float(layer["visibility"])
        keep[y1:y2, x1:x2] *= 1.0 - alpha
    return 1.0 - keep


def _layer_rects(layers, kinds):
    return [tuple(layer["roi"]) for layer in _first(layers, kinds)]


def _weight_view(base, weight, color, rects, legend, height):
    view = _tint(_dim(base, 0.45), weight, color, 0.85).astype(np.uint8)
    t = _thickness(*base.shape[:2])
    for rect in rects:
        _draw_rect(view, rect, ROI_COLOR, t, dashed=True)
    return _draw_labels(view, _legend(height, legend))


def _alpha_view(base, weight, rects, legend, height):
    view = np.repeat((weight * 255.0)[..., None], 3, axis=2) * 0.75 + _dim(base, 0.25)
    view = view.clip(0, 255).astype(np.uint8)
    t = _thickness(*base.shape[:2])
    for rect in rects:
        _draw_rect(view, rect, ROI_COLOR, t, dashed=True)
    return _draw_labels(view, _legend(height, legend))


def _record_for(seg, records):
    box = [float(v) for v in getattr(seg, "bbox", ())[:4]]
    return next((row for row in records or () if row.get("batch_index", 0) == 0
                 and [float(v) for v in row.get("yolo_bbox", ())[:4]] == box), None)


def detection_view(image, detected, selected, records=None):
    """Detected faces (white) and the faces the branch actually processes:
    the selection, coloured by its InsightFace assignment for RESTORE.
    Returns the image and its subtitle."""
    base = _rgb(image)
    height, width = base.shape[:2]
    t = _thickness(height, width)
    view = base.copy()
    for seg in _first_image_segs(detected):
        _draw_rect(view, bbox_rect(seg, width, height), DETECTION_COLOR, max(1, t // 2))
    labels = []
    for index, seg in enumerate(_first_image_segs(selected), 1):
        rect = bbox_rect(seg, width, height)
        row = _record_for(seg, records) if records is not None else None
        color = STATUS_COLORS.get(row["status"], ROI_COLOR) if row else SELECTED_COLOR
        _draw_rect(view, rect, color, t + 1)
        if rect is not None:
            text = f"F{index}" + (f" {row['status']} IoU {row['iou']:.2f}" if row else "")
            labels.append((rect[0] + t + 2, rect[1] + t + 2, text, color))
    legend = [("DETECTED (YOLO)", DETECTION_COLOR)]
    if records is None:
        legend.append(("SELECTED", SELECTED_COLOR))
    else:
        legend += [("ASSIGNED", STATUS_COLORS["OK"]), ("NO ASSIGNMENT", STATUS_COLORS["UNMATCHED"]),
                   ("AMBIGUOUS", STATUS_COLORS["AMBIGUOUS"])]
    detected_count, selected_count = len(_first_image_segs(detected)), len(_first_image_segs(selected))
    subtitle = f"detected {detected_count} · selected {selected_count}"
    if records is not None:
        subtitle += f" · assigned {sum(1 for row in records if row.get('status') == 'OK')}"
    return _to_tensor(_draw_labels(view, _legend(height, legend) + labels)), subtitle


def branch_spatial_stages(mode, image, coverage, layers) -> List[dict]:
    """Processing region, effective masks and pasteback of one branch, and its
    Refine mask if Refine ran.  RESTORE: warp region, pasteback alpha,
    alpha x visibility.  SDXL/ZIT: sampling ROI, SEG mask, pasteback weight."""
    base = _rgb(image)
    height, width = base.shape[:2]
    t = _thickness(height, width)
    stages = []
    if mode == "restore":
        rects = _layer_rects(layers, ("restore",))
        if rects:
            region = np.zeros((height, width), dtype=bool)
            for x1, y1, x2, y2 in rects:
                region[y1:y2, x1:x2] = True
            view = _dim(base, 0.4)
            view[region] = base[region]
            view = view.astype(np.uint8)
            for rect in rects:
                _draw_rect(view, rect, ROI_COLOR, t, dashed=True)
            for seg in _first_image_segs(coverage):
                _draw_rect(view, bbox_rect(seg, width, height), DETECTION_COLOR, t)
            view = _draw_labels(view, _legend(height, [("ASSIGNED FACE (YOLO)", DETECTION_COLOR),
                                                       ("WARP / PASTEBACK REGION", ROI_COLOR)]))
            alpha = layer_weight(layers, height, width, ("restore",), visibility=False)
            weight = layer_weight(layers, height, width, ("restore",))
            visibility = sorted({round(float(layer["visibility"]), 3) for layer in _first(layers, ("restore",))})
            stages += [
                {"title": "Restore Region", "image": _to_tensor(view), "subtitle": f"{len(rects)} face(s) · {int(region.sum())} px"},
                {"title": "Restore Alpha", "image": _to_tensor(_alpha_view(base, alpha, rects, [
                    ("PASTEBACK ALPHA (white = 1)", (255, 255, 255)), ("REGION", ROI_COLOR)], height)),
                 "subtitle": f"alpha > 0: {int((alpha > 0).sum())} px"},
                {"title": "Visibility", "image": _to_tensor(_alpha_view(base, weight, rects, [
                    ("ALPHA x VISIBILITY = RESTORE SHARE", (255, 255, 255))], height)),
                 "subtitle": "visibility " + ", ".join(f"{v:g}" for v in visibility)},
            ]
    else:
        rects = [roi_rect(seg, width, height) for seg in _first_image_segs(coverage)]
        rects = [rect for rect in rects if rect is not None]
        if rects and _first(layers, ("detailer",)):
            region = np.zeros((height, width), dtype=bool)
            for x1, y1, x2, y2 in rects:
                region[y1:y2, x1:x2] = True
            view = _dim(base, 0.4)
            view[region] = base[region]
            view = view.astype(np.uint8)
            for seg in _first_image_segs(coverage):
                _draw_rect(view, roi_rect(seg, width, height), ROI_COLOR, t, dashed=True)
                _draw_rect(view, bbox_rect(seg, width, height), DETECTION_COLOR, t)
            view = _draw_labels(view, _legend(height, [("SELECTED FACE", DETECTION_COLOR), ("SAMPLING ROI (crop_region)", ROI_COLOR)]))
            raw = projected_mask(coverage, height, width, feather=0, alpha=255).numpy()
            weight = layer_weight(layers, height, width, ("detailer",))
            stages += [
                {"title": "Sampling ROI", "image": _to_tensor(view), "subtitle": f"ROI {int(region.sum())} px"},
                {"title": "Mask", "image": _to_tensor(_weight_view(base, raw, MASK_COLOR, rects, [
                    ("cropped_mask", MASK_COLOR), ("SAMPLING ROI", ROI_COLOR)], height)),
                 "subtitle": f"mask {int((raw > 0).sum())} px"},
                {"title": "Pasteback", "image": _to_tensor(_alpha_view(base, weight, rects, [
                    ("PASTEBACK WEIGHT (white = 1)", (255, 255, 255)), ("SAMPLING ROI", ROI_COLOR)], height)),
                 "subtitle": f"{100.0 * float(weight.sum()) / max(1, int(region.sum())):.1f}% of ROI"},
            ]
    return stages


def refine_mask_stage(image, layers):
    base = _rgb(image)
    height, width = base.shape[:2]
    rects = _layer_rects(layers, ("refine",))
    if not rects:
        return None
    weight = layer_weight(layers, height, width, ("refine",))
    return {"title": "Refine Mask", "image": _to_tensor(_weight_view(base, weight, REFINE_COLOR, rects, [
        ("REFINE ALPHA (0.75 x ellipse)", REFINE_COLOR), ("REFINE REGION", ROI_COLOR)], height)),
        "subtitle": f"alpha > 0: {int((weight > 0).sum())} px"}


def _bare(title) -> str:
    head, _, rest = str(title).partition(" ")
    return rest if head.isdigit() and rest else str(title)


def with_face_stages(payload, image, mode, coverage, layers, detection=None):
    """Insert the spatial stages of a branch into its diagnostic.

    After the detection stage (replaced by ``detection`` when given) come the
    processing region, mask/alpha and pasteback stages; the Refine mask
    precedes the Refine result.  The existing Source, primary, difference,
    Refine and Final stages stay as they are.  Idempotent
    (``metadata[FACE_STAGES_FLAG]``), so a branch-cache entry written before
    these stages is completed on its HIT from the cached SEGS and layers.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), list):
        return payload
    metadata = dict(payload.get("metadata") or {})
    if metadata.get(FACE_STAGES_FLAG) or image is None:
        return payload
    spatial = _normalize_stages(branch_spatial_stages(mode, image, coverage, layers))
    refine = refine_mask_stage(image, layers)
    stages = []
    for stage in payload["stages"]:
        title = _bare(stage.get("title", ""))
        if title == "Detected Faces":
            if detection is not None:
                view, subtitle = detection
                stage = dict(stage, title="Detection · Assignment" if mode == "restore" else "Detection", subtitle=subtitle,
                             image=_normalize_stages([{"title": "", "image": view}])[0]["image"])
            stages.append(dict(stage, title=_bare(stage["title"])))
            stages += spatial
            continue
        if title.startswith("Refine ") and refine is not None:
            stages += _normalize_stages([refine])
        stages.append(dict(stage, title=title))
    stages = [dict(stage, title=f"{index:02d} {_bare(stage['title'])}") for index, stage in enumerate(stages, start=1)]
    metadata[FACE_STAGES_FLAG] = True
    result = dict(payload)
    result["stages"] = stages
    result["preview"] = [stage["image"] for stage in stages]
    result["images"] = result["preview"]
    result["metadata"] = metadata
    result["version"] = 2
    return result


# --------------------------------------------------------------------------
# Advanced merge
# --------------------------------------------------------------------------

def merge_layer_stages(image, result_image, slot_layers: Dict[int, list], modes: Dict[int, str]):
    """``01 · INPUT LAYERS`` … ``04 · FINAL`` and metrics of the FaceProcess merge.

    Weights are the effective layer weights of each branch; the shares follow
    the composition D1 -> D2 -> D3, each branch blended over the previous result.
    """
    base = _rgb(image)
    height, width = base.shape[:2]
    t = _thickness(height, width)
    weights = {slot: torch.from_numpy(layer_weight(layers, height, width)) for slot, layers in slot_layers.items()
               if _first(layers, ("restore", "detailer", "refine"))}
    shares, source_share = alpha_shares(weights, height, width)
    overlaps = mask_overlaps(weights)
    region_count = np.zeros((height, width), dtype=np.int64)
    for slot in weights:
        region = np.zeros((height, width), dtype=bool)
        for x1, y1, x2, y2 in _layer_rects(slot_layers[slot], ("restore", "detailer", "refine")):
            region[y1:y2, x1:x2] = True
        region_count += region
    region_overlap = region_count >= 2

    def label(slot):
        kinds = [layer["kind"] for layer in slot_layers.get(slot, []) if layer["kind"] != "quantize"]
        counted = ", ".join(f"{kind} x{kinds.count(kind)}" for kind in dict.fromkeys(kinds)) or "no layers"
        return f"D{slot} · {MODE_LABELS.get(modes.get(slot, 'none'), '?')} · {counted}"

    def frames(arr):
        for slot, layers in slot_layers.items():
            for layer in _first(layers, ("restore", "detailer", "refine")):
                _draw_rect(arr, _slot_rect(tuple(layer["roi"]), slot, t), SLOT_COLORS[slot], t,
                           dashed=layer["kind"] == "refine")

    # 01 · INPUT LAYERS: per branch its effective layer weight and layer regions.
    inputs = _dim(base, 0.5)
    for slot, weight in weights.items():
        inputs = _tint(inputs, weight.numpy(), SLOT_COLORS[slot], 0.6)
    inputs = inputs.astype(np.uint8)
    frames(inputs)
    inputs = _draw_labels(inputs, _legend(height, [(label(slot), SLOT_COLORS[slot]) for slot in (1, 2, 3)]
                                          + [("fill = effective alpha · frame = layer region (dashed = refine)", (220, 220, 220))]))

    # 02 · OVERLAP: layer regions (reference) and overlapping effective weights.
    overlap = _tint(_dim(base, 0.35, gray=True), region_overlap.astype(np.float32), OVERLAP_COLOR, 0.3)
    blend = torch.zeros((height, width), dtype=torch.float32)
    for (earlier, later), area in overlaps.items():
        blend = torch.maximum(blend, torch.minimum(weights[earlier], weights[later]) * area)
    overlap = _tint(overlap, np.where(blend.numpy() > 0, 0.35 + 0.65 * blend.numpy(), 0.0), MASK_OVERLAP_COLOR, 0.95)
    overlap = overlap.astype(np.uint8)
    frames(overlap)
    pairs = [(f"D{earlier} x D{later}: {int(area.sum())} px · D{later} on top", MASK_OVERLAP_COLOR)
             for (earlier, later), area in sorted(overlaps.items())]
    overlap = _draw_labels(overlap, _legend(height, [("REGION OVERLAP (frames only)", OVERLAP_COLOR),
                                                     ("ALPHA OVERLAP (later branch on top)", MASK_OVERLAP_COLOR)] + pairs))

    # 03 · COMPOSITE: colour mix of the exact per-pixel shares.
    composite = _dim(base, 0.4, gray=True) * source_share.numpy()[..., None]
    for slot, share in shares.items():
        composite = composite + np.asarray(SLOT_COLORS[slot], dtype=np.float32) * share.numpy()[..., None]
    effective = {slot: float(share.sum()) for slot, share in shares.items()}
    composite = _draw_labels(composite.clip(0, 255).astype(np.uint8), _legend(height, [
        ("ORDER D1 > D2 > D3 · LAYERS ON THE PREVIOUS RESULT", (255, 255, 255)),
    ] + [(f"{label(slot)}: {effective.get(slot, 0.0):.0f} px", SLOT_COLORS[slot]) for slot in (1, 2, 3)]
      + [(f"SOURCE: {float(source_share.sum()):.0f} px", (170, 170, 170))]))

    alpha_overlap_px = int(sum(int(area.sum()) for area in overlaps.values()))
    stages = [
        {"title": "01 · INPUT LAYERS", "image": _to_tensor(inputs), "subtitle": "fill = effective alpha · frame = region"},
        {"title": "02 · OVERLAP", "image": _to_tensor(overlap),
         "subtitle": f"alpha overlap {alpha_overlap_px} px · regions {int(region_overlap.sum())} px"},
        {"title": "03 · COMPOSITE", "image": _to_tensor(composite), "subtitle": "shares · D1 → D2 → D3"},
        {"title": "04 · FINAL", "image": result_image, "subtitle": "merged result"},
    ]
    metrics = {
        "share_px": {f"D{slot}": round(effective.get(slot, 0.0), 1) for slot in (1, 2, 3)},
        "source_px": round(float(source_share.sum()), 1),
        "region_overlap_px": int(region_overlap.sum()),
        "alpha_overlap_px": {f"D{earlier}xD{later}": int(area.sum()) for (earlier, later), area in sorted(overlaps.items())},
        "reduced_by_later_px": {
            f"D{slot}": round(float((weights[slot] - shares[slot]).sum()), 1) for slot in sorted(weights)
            if float((weights[slot] - shares[slot]).sum()) > 0
        },
        "layers": {f"D{slot}": [layer["kind"] for layer in slot_layers.get(slot, [])] for slot in (1, 2, 3)},
    }
    return stages, metrics
