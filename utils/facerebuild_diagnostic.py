"""Spatial diagnostics of FaceRebuild v2 (2.8): pure image functions.

Everything comes from data the FaceRebuild run already produced: the target
image, the rebuild context of every slot (face bbox, head/neck ROI, feather)
and the final result.  The masks are rebuilt with the engine functions the
processing itself uses (``make_inpaint_mask`` for sampling,
``make_pasteback_mask`` with the slot's PASTEBACK NECK for the composite), in
ROI coordinates and placed at the ROI exactly like ``pasteback_exact``.
Nothing detects, samples or composes again.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from ..engine.instantid_face_rebuild import FaceRebuildGeometry, make_inpaint_mask, make_pasteback_mask
from .face_diagnostic import _weight_view
from .segs_diagnostic import (
    DETECTION_COLOR,
    MASK_COLOR,
    MASK_OVERLAP_COLOR,
    ROI_COLOR,
    SLOT_COLORS,
    _dim,
    _draw_labels,
    _draw_rect,
    _legend,
    _rect,
    _rgb,
    _slot_rect,
    _thickness,
    _tint,
    _to_tensor,
    alpha_shares,
    mask_overlaps,
)


def slot_geometry(context) -> FaceRebuildGeometry:
    return FaceRebuildGeometry(
        face_bbox=tuple(int(value) for value in context["face_bbox"]),
        roi=tuple(int(value) for value in context["roi"]),
        working_size=tuple(int(value) for value in context["working_size"]),
        feather=int(context["feather"]),
    )


def _place(local, roi, height, width) -> np.ndarray:
    full = np.zeros((height, width), dtype=np.float32)
    left, top, right, bottom = roi
    full[top:bottom, left:right] = np.clip(local.astype(np.float32), 0.0, 1.0)
    return full


def slot_masks(slot, height: int, width: int):
    """Full-image inpaint mask and pasteback weight of one slot."""
    geometry = slot_geometry(slot["context"])
    inpaint = _place(make_inpaint_mask(geometry), geometry.roi, height, width)
    pasteback = _place(make_pasteback_mask(geometry, include_neck=bool(slot["pasteback_neck"])), geometry.roi, height, width)
    return geometry, inpaint, pasteback


def _pasted_view(result, weight, rects, colors, legend, height):
    """The pasted result where the pasteback weight takes it, grey elsewhere."""
    pasted = _rgb(result).astype(np.float32)
    view = pasted * weight[..., None] + _dim(pasted, 0.3, gray=True) * (1.0 - weight[..., None])
    view = view.clip(0, 255).astype(np.uint8)
    t = _thickness(*view.shape[:2])
    for rect, color in zip(rects, colors):
        _draw_rect(view, rect, color, t, dashed=True)
    return _draw_labels(view, _legend(height, legend))


def standard_stages(source, result, slot) -> List[Dict]:
    """Rebuild ROI, Inpaint Mask and Pasteback of the one FaceRebuild face."""
    base = _rgb(source)
    height, width = base.shape[:2]
    t = _thickness(height, width)
    geometry, inpaint, pasteback = slot_masks(slot, height, width)
    left, top, right, bottom = geometry.roi
    roi = _dim(base, 0.4)
    roi[top:bottom, left:right] = base[top:bottom, left:right]
    roi = roi.astype(np.uint8)
    _draw_rect(roi, geometry.roi, ROI_COLOR, t, dashed=True)
    _draw_rect(roi, _rect(geometry.face_bbox, width, height), DETECTION_COLOR, t)
    roi = _draw_labels(roi, _legend(height, [(f"TARGET FACE · {slot['target_face'].upper()}", DETECTION_COLOR),
                                             ("HEAD / NECK ROI", ROI_COLOR)]))
    neck = "on" if slot["pasteback_neck"] else "off"
    return [
        {"title": "Rebuild ROI", "image": _to_tensor(roi),
         "subtitle": f"ROI {right - left} × {bottom - top} · working {geometry.working_size[0]} × {geometry.working_size[1]}"},
        {"title": "Inpaint Mask", "image": _to_tensor(_weight_view(base, inpaint, MASK_COLOR, [geometry.roi], [
            ("INPAINT MASK (sampling)", MASK_COLOR), ("HEAD / NECK ROI", ROI_COLOR)], height)),
         "subtitle": f"feather {geometry.feather} px · {int((inpaint > 0).sum())} px"},
        {"title": "Pasteback", "image": _to_tensor(_pasted_view(result, pasteback, [geometry.roi], [ROI_COLOR], [
            ("PASTED RESULT x PASTEBACK WEIGHT", (255, 255, 255)), ("HEAD / NECK ROI", ROI_COLOR)], height)),
         "subtitle": f"neck {neck} · {int((pasteback > 0).sum())} px"},
    ]


def advanced_stages(source, result, slots: Sequence[Dict]) -> List[Dict]:
    """Target faces, inpaint masks, pasteback and (if any) overlap of up to
    three slots.  ``slots`` is in pasteback order (largest face first), the
    order in which the slots were sampled and composed."""
    base = _rgb(source)
    height, width = base.shape[:2]
    t = _thickness(height, width)
    masks = {slot["index"]: slot_masks(slot, height, width) for slot in slots}
    order = {slot["index"]: position for position, slot in enumerate(slots, start=1)}

    def rect(index):
        return _slot_rect(masks[index][0].roi, index, t)

    # Target faces: per slot face bbox and ROI, labelled with selector and order.
    faces = _dim(base, 0.5).astype(np.uint8)
    labels = []
    for slot in slots:
        index = slot["index"]
        geometry = masks[index][0]
        _draw_rect(faces, rect(index), SLOT_COLORS[index], t, dashed=True)
        _draw_rect(faces, _rect(geometry.face_bbox, width, height), SLOT_COLORS[index], t + 1)
        labels.append((geometry.face_bbox[0] + t + 2, geometry.face_bbox[1] + t + 2, f"F{index}", SLOT_COLORS[index]))
    legend = [(f"F{slot['index']} · {slot['target_face'].upper()} · #{order[slot['index']]}", SLOT_COLORS[slot["index"]])
              for slot in slots]
    legend.append(("box = target face · dashed = head / neck ROI · #n = order", (220, 220, 220)))
    faces = _draw_labels(faces, _legend(height, legend) + labels)

    inpaint_view = _dim(base, 0.5)
    for slot in slots:
        inpaint_view = _tint(inpaint_view, masks[slot["index"]][1], SLOT_COLORS[slot["index"]], 0.6)
    inpaint_view = inpaint_view.astype(np.uint8)
    for slot in slots:
        _draw_rect(inpaint_view, rect(slot["index"]), SLOT_COLORS[slot["index"]], t, dashed=True)
    inpaint_view = _draw_labels(inpaint_view, _legend(height, [
        (f"F{slot['index']} INPAINT MASK · feather {masks[slot['index']][0].feather} px", SLOT_COLORS[slot["index"]])
        for slot in slots]))

    # Composite weight in pasteback order: each slot is blended over the previous result.
    weights = {order[index]: torch.from_numpy(masks[index][2]) for index in masks}
    shares, _source_share = alpha_shares(weights, height, width)
    combined = np.clip(sum(share.numpy() for share in shares.values()), 0.0, 1.0)
    paste = _pasted_view(result, combined, [rect(slot["index"]) for slot in slots],
                         [SLOT_COLORS[slot["index"]] for slot in slots],
                         [("PASTED RESULT x PASTEBACK WEIGHT", (255, 255, 255))]
                         + [(f"F{slot['index']} · neck {'on' if slot['pasteback_neck'] else 'off'}", SLOT_COLORS[slot["index"]])
                            for slot in slots], height)
    stages = [
        {"title": "Target Faces", "image": _to_tensor(faces),
         "subtitle": " · ".join(f"F{slot['index']} #{order[slot['index']]}" for slot in slots)},
        {"title": "Inpaint Masks", "image": _to_tensor(inpaint_view),
         "subtitle": f"{len(slots)} slot(s)"},
        {"title": "Pasteback", "image": _to_tensor(paste),
         "subtitle": f"order {' → '.join('F' + str(slot['index']) for slot in slots)}"},
    ]
    overlaps = mask_overlaps(weights)
    if overlaps:
        slot_of = {position: index for index, position in order.items()}
        view = _dim(base, 0.35, gray=True)
        for (earlier, later), area in overlaps.items():
            view = _tint(view, area.numpy().astype(np.float32), MASK_OVERLAP_COLOR, 0.9)
        view = view.astype(np.uint8)
        for slot in slots:
            _draw_rect(view, rect(slot["index"]), SLOT_COLORS[slot["index"]], t, dashed=True)
        pairs = [(f"F{slot_of[earlier]} x F{slot_of[later]}: {int(area.sum())} px · F{slot_of[later]} on top", MASK_OVERLAP_COLOR)
                 for (earlier, later), area in sorted(overlaps.items())]
        stages.append({"title": "Overlap", "image": _to_tensor(_draw_labels(view, _legend(height, pairs))),
                       "subtitle": f"{sum(int(area.sum()) for area in overlaps.values())} px · later slot on top"})
    return stages
