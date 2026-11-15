"""Spatial diagnostics and Prepare cards of FaceSwap v2 (2.8): pure functions.

Everything comes from data the unchanged FaceSwap run already produced:

* the selected target face of the run (selector, detected count, bbox): the
  executor's own diagnostic (metadata, summary, ``TARGET DETAILS`` bbox),
* the paste support: the ``branch_mask`` of the executor's CMK stable SEGS,
  i.e. ``paste mask x BLEND > 0``, the exact region the swapped branch takes
  (SEGS CONCAT merges a branch with ``torch.where`` over this mask),
* the target image and the result (Standard: IMAGE PROCEED; Advanced: the
  SEGS CONCAT image).

Nothing detects, swaps, restores or merges again.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np
import torch

from .face_diagnostic import _weight_view
from .prepare_card import _ROW, card_text
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
from .stable_segs import stable_branch_components

_BBOX = re.compile(r"^bbox:\s*\[([^\]]*)\]", re.MULTILINE)


# --------------------------------------------------------------------------
# Data of one FaceSwap run (from the executor's own diagnostic and SEGS)
# --------------------------------------------------------------------------

def _summary_fields(summary) -> Dict[str, str]:
    fields = {}
    for line in str(summary or "").splitlines():
        match = _ROW.match(line.rstrip())
        if match:
            fields.setdefault(match.group(1).strip(), (match.group(2) or "").strip())
    return fields


def _section(text, start, stop=None) -> str:
    text = str(text or "")
    head = text.find(f"\n{start}\n")
    if head < 0:
        return ""
    body = text[head + len(start) + 2:]
    if stop:
        end = body.find(f"\n{stop}\n")
        body = body if end < 0 else body[:end]
    return body


def _target_bbox(details):
    match = _BBOX.search(_section(details, "TARGET DETAILS", "SOURCE DETAILS"))
    if not match:
        return None
    try:
        values = [float(value) for value in match.group(1).split(",")[:4]]
    except ValueError:
        return None
    return values if len(values) == 4 else None


def _target_lines(summary) -> Dict[str, str]:
    """``Confidence`` / ``Face size`` of the selected target face."""
    fields = {}
    for line in _section(summary, "TARGET", "SOURCE").splitlines():
        label, sep, value = line.partition(":")
        if sep:
            fields[label.strip()] = value.strip()
    return fields


def run_record(diagnostic, segs=None, *, index=1) -> Dict:
    """The FaceSwap run of one executor as plain data (no tensors)."""
    payload = diagnostic if isinstance(diagnostic, dict) else {}
    meta = dict(payload.get("metadata") or {})
    fields = _summary_fields(payload.get("summary"))
    target = _target_lines(payload.get("summary"))
    enabled = bool(meta.get("enabled"))
    return {
        "index": int(index),
        "enabled": enabled,
        "global_enabled": bool(meta.get("global_enabled", True)),
        "local_enabled": bool(meta.get("local_enabled", True)),
        "target_face": str(meta.get("target_selection") or fields.get("TARGET FACE") or ""),
        "source_face": str(meta.get("source_selection") or fields.get("SOURCE FACE") or ""),
        "target_faces": int(meta.get("target_detected_faces", 0) or 0),
        "source_faces": int(meta.get("source_detected_faces", 0) or 0),
        "face_bbox": _target_bbox(payload.get("details")) if enabled else None,
        "face_size": target.get("Face size", "").replace("px", " px").strip(),
        "confidence": target.get("Confidence", ""),
        "swap_model": str(meta.get("swap_model") or fields.get("SWAP MODEL") or ""),
        "detector_model": str(meta.get("detector_model") or fields.get("DETECT MODEL") or ""),
        "enhancer": str(meta.get("face_enhancer") or fields.get("PATCH ENHANCER") or "Off"),
        "enhancer_strength": fields.get("PATCH STRENGTH", ""),
        "enhancer_fallback": fields.get("ENHANCER FALLBACK", ""),
        "post_restore": bool(meta.get("post_restore", False)),
        "restore_model": str(meta.get("restore_model") or ""),
        "restore_visibility": float(meta.get("restore_visibility", 1.0) or 0.0),
        "profile": fields.get("QUALITY PROFILE", ""),
        "blend": float(meta.get("blend", 1.0) or 0.0),
        "crop_factor": float(meta.get("crop_factor", 1.5) or 0.0),
        "feather": int(meta.get("feather", 0) or 0),
        "bbox_dilation": int(meta.get("bbox_dilation", 0) or 0),
        "drop_size": int(meta.get("drop_size", 10) or 0),
        "identity_strength": float(meta.get("identity_strength", 1.0) or 0.0),
        "content_guard": fields.get("CONTENT GUARD", ""),
        "changed": float(meta.get("changed_avg", 0.0) or 0.0),
    }


def paste_support(segs):
    """``branch_mask`` of the first batch image as float [H, W], or None."""
    stable = stable_branch_components(segs)
    if stable is None:
        return None
    mask = stable[1]
    return mask[0, ..., 0].detach().to("cpu", torch.float32).numpy()


def _support_rect(support):
    rows = np.flatnonzero(support.any(axis=1))
    cols = np.flatnonzero(support.any(axis=0))
    if not len(rows) or not len(cols):
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


# --------------------------------------------------------------------------
# Prepare cards
# --------------------------------------------------------------------------

def _faces(count) -> str:
    return f"{count} face" if int(count) == 1 else f"{count} faces"


def run_rows(record) -> List[tuple]:
    """Prepare rows of one FaceSwap run (selection, models, paste settings)."""
    face = [part for part in (record["face_size"], f"score {record['confidence']}" if record["confidence"] else "") if part]
    enhancer = record["enhancer"]
    if enhancer.lower() != "off" and record["enhancer_strength"]:
        enhancer = f"{enhancer} · strength {record['enhancer_strength']}"
    if record["enhancer_fallback"]:
        enhancer = f"{enhancer} · fallback {record['enhancer_fallback']}"
    restore = (f"{record['restore_model']} · visibility {record['restore_visibility']:.2f}"
               if record["post_restore"] else "OFF")
    rows = [
        ("TARGET", f"{record['target_face']} · 1 of {_faces(record['target_faces'])}"),
        ("FACE", " · ".join(face) or "n/a"),
        ("SOURCE", f"{record['source_face']} · {_faces(record['source_faces'])}"),
        ("SWAP", record["swap_model"]),
        ("DETECT", f"{record['detector_model']} · drop < {record['drop_size']} px"),
    ]
    if record["profile"]:
        rows.append(("PROFILE", record["profile"]))
    rows += [
        ("PASTE", f"blend {record['blend']:.2f} · feather {record['feather']}"),
        ("CROP", f"factor {record['crop_factor']:g} · dilation {record['bbox_dilation']}"),
        ("IDENTITY", f"{record['identity_strength']:.2f}"),
        ("ENHANCER", enhancer),
        ("RESTORE", restore),
    ]
    if record["content_guard"]:
        rows.append(("CONTENTGUARD", record["content_guard"]))
    return rows


def bypass_text(record) -> str:
    return card_text([
        ("GLOBAL", "ON" if record["global_enabled"] else "OFF"),
        ("ENABLE", "ON" if record["local_enabled"] else "OFF"),
        ("RESULT", "IMAGE TARGET PASSTHROUGH"),
        ("SOURCE", "not loaded · no face detection"),
    ])


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def _pasted_view(result, weight, rects, colors, legend, height):
    """The result where the paste support takes it, grey elsewhere."""
    pasted = _rgb(result).astype(np.float32)
    view = pasted * weight[..., None] + _dim(pasted, 0.3, gray=True) * (1.0 - weight[..., None])
    view = view.clip(0, 255).astype(np.uint8)
    t = _thickness(*view.shape[:2])
    for rect, color in zip(rects, colors):
        _draw_rect(view, rect, color, t, dashed=True)
    return _draw_labels(view, _legend(height, legend))


def _post(record) -> str:
    return f"restore {record['restore_model']}" if record["post_restore"] else "restore off"


def standard_stages(target, result, record, support) -> List[Dict]:
    """Swap Face, Paste Mask and Pasteback of the one FaceSwap run."""
    base = _rgb(target)
    height, width = base.shape[:2]
    t = _thickness(height, width)
    region = _support_rect(support)
    face = _rect(record["face_bbox"], width, height)
    swap = _dim(base, 0.4)
    if region:
        left, top, right, bottom = region
        swap[top:bottom, left:right] = base[top:bottom, left:right]
    swap = swap.astype(np.uint8)
    _draw_rect(swap, region, ROI_COLOR, t, dashed=True)
    _draw_rect(swap, face, DETECTION_COLOR, t)
    swap = _draw_labels(swap, _legend(height, [
        (f"TARGET FACE · {record['target_face'].upper()} · {_faces(record['target_faces'])}", DETECTION_COLOR),
        ("PASTE REGION", ROI_COLOR)]))
    face_text = f"face {face[2] - face[0]} × {face[3] - face[1]}" if face else "face n/a"
    region_text = f"region {region[2] - region[0]} × {region[3] - region[1]}" if region else "no paste region"
    pixels = int((support > 0).sum())
    return [
        {"title": "Swap Face", "image": _to_tensor(swap), "subtitle": f"{face_text} · {region_text}"},
        {"title": "Paste Mask", "image": _to_tensor(_weight_view(base, support, MASK_COLOR, [region] if region else [], [
            ("PASTE MASK (swap support)", MASK_COLOR), ("PASTE REGION", ROI_COLOR)], height)),
         "subtitle": f"feather {record['feather']} · blend {record['blend']:.2f} · {pixels} px"},
        {"title": "Pasteback", "image": _to_tensor(_pasted_view(result, support, [region] if region else [], [ROI_COLOR], [
            ("SWAPPED RESULT x PASTE MASK", (255, 255, 255)), ("PASTE REGION", ROI_COLOR)], height)),
         "subtitle": f"enhancer {record['enhancer']} · {_post(record)}"},
    ]


def advanced_stages(target, merged, slots: Sequence[Dict], merge_order: Sequence[int]) -> List[Dict]:
    """Target Faces, Paste Masks, Merge and (if any) Overlap of the active
    slots.  ``slots``: records with their ``support``; ``merge_order``: slot
    indices in SEGS CONCAT input order (a later slot replaces an earlier one
    inside its paste mask)."""
    base = _rgb(target)
    height, width = base.shape[:2]
    t = _thickness(height, width)
    by_index = {slot["index"]: slot for slot in slots}
    order = [index for index in merge_order if index in by_index]
    position = {index: number for number, index in enumerate(order, start=1)}
    regions = {index: _support_rect(by_index[index]["support"]) for index in order}

    def rect(index):
        return _slot_rect(regions[index], index, t)

    faces = _dim(base, 0.5).astype(np.uint8)
    labels = []
    for index in order:
        slot = by_index[index]
        face = _rect(slot["face_bbox"], width, height)
        _draw_rect(faces, rect(index), SLOT_COLORS[index], t, dashed=True)
        _draw_rect(faces, face, SLOT_COLORS[index], t + 1)
        if face:
            labels.append((face[0] + t + 2, face[1] + t + 2, f"F{index}", SLOT_COLORS[index]))
    legend = [(f"F{index} · {by_index[index]['target_face'].upper()} · merge #{position[index]}", SLOT_COLORS[index])
              for index in order]
    legend.append(("box = target face · dashed = paste region", (220, 220, 220)))
    faces = _draw_labels(faces, _legend(height, legend) + labels)

    mask_view = _dim(base, 0.5)
    for index in order:
        mask_view = _tint(mask_view, by_index[index]["support"], SLOT_COLORS[index], 0.6)
    mask_view = mask_view.astype(np.uint8)
    for index in order:
        _draw_rect(mask_view, rect(index), SLOT_COLORS[index], t, dashed=True)
    mask_view = _draw_labels(mask_view, _legend(height, [
        (f"F{index} PASTE MASK · feather {by_index[index]['feather']} · blend {by_index[index]['blend']:.2f}",
         SLOT_COLORS[index]) for index in order]))

    weights = {position[index]: torch.from_numpy(by_index[index]["support"]) for index in order}
    shares, _source = alpha_shares(weights, height, width)
    combined = np.clip(sum(share.numpy() for share in shares.values()), 0.0, 1.0) if shares else np.zeros((height, width))
    chain = " → ".join(f"F{index}" for index in order)
    merge = _pasted_view(merged, combined, [rect(index) for index in order], [SLOT_COLORS[index] for index in order],
                         [("MERGED RESULT x PASTE MASKS", (255, 255, 255)),
                          (f"order {chain} · later on top", (220, 220, 220))], height)
    stages = [
        {"title": "Target Faces", "image": _to_tensor(faces),
         "subtitle": " · ".join(f"F{index} {by_index[index]['target_face'].lower()}" for index in order)},
        {"title": "Paste Masks", "image": _to_tensor(mask_view), "subtitle": f"{len(order)} slot(s)"},
        {"title": "Merge", "image": _to_tensor(merge), "subtitle": f"order {chain}"},
    ]
    overlaps = mask_overlaps(weights)
    if overlaps:
        slot_of = {number: index for index, number in position.items()}
        view = _dim(base, 0.35, gray=True)
        for area in overlaps.values():
            view = _tint(view, area.numpy().astype(np.float32), MASK_OVERLAP_COLOR, 0.9)
        view = view.astype(np.uint8)
        for index in order:
            _draw_rect(view, rect(index), SLOT_COLORS[index], t, dashed=True)
        pairs = [(f"F{slot_of[earlier]} x F{slot_of[later]}: {int(area.sum())} px · F{slot_of[later]} on top",
                  MASK_OVERLAP_COLOR) for (earlier, later), area in sorted(overlaps.items())]
        stages.append({"title": "Overlap", "image": _to_tensor(_draw_labels(view, _legend(height, pairs))),
                       "subtitle": f"{sum(int(area.sum()) for area in overlaps.values())} px · later slot on top"})
    return stages
