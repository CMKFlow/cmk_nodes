"""Processing layers of a FaceProcess 2.8 branch (F4).

A branch result is the ordered application of its layers to the branch input.
Every layer carries its region, its content and the alpha the processing
actually used, and replays the exact arithmetic of its 2.7 path:

``quantize``  the whole-image uint8 conversion of a path, applied to the union
              of the branch's layer regions: ``round`` (restore input,
              ``tensor_to_uint8_rgb``) or ``trunc`` (refine input);
``restore``   one restored face: pasteback ``X*(1-a) + W*a`` on uint8 with the
              pasteback alpha, then the visibility blend ``X*(1-v) + R*v``;
``detailer``  one SEGSPaste: ``X*(1-A) + C*A`` in float32 with the feathered
              mask already applied;
``refine``    ``X*(1-A) + F*A`` on uint8, A = 0.75 * refine mask (its full
              ellipse, not clipped to any crop region).

``compose_layers(image, layers)`` reproduces the branch result inside the
union of the layer regions exactly; outside, the branch differs from its input
only by the whole-image quantisation of the 2.7 paths (at most 1/255 when the
input is off the 8-bit grid).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..engine.native_detailer import _mask_tensor, _soft_mask
from .stable_segs import CMKStableSEGS, _STABLE_SEGS_SCHEMA

LAYER_SCHEMA = "faceprocess_layers_v1"


def alpha_roi(alpha):
    """Bounding box (x1, y1, x2, y2) of alpha > 0, or None."""
    ys, xs = np.nonzero(np.asarray(alpha) > 0)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _crop(array, roi):
    x1, y1, x2, y2 = roi
    return np.ascontiguousarray(array[y1:y2, x1:x2])


def restore_layer(batch_index, warped, alpha, visibility):
    roi = alpha_roi(alpha)
    if roi is None:
        return None
    return {"kind": "restore", "batch_index": batch_index, "roi": roi, "content": _crop(warped, roi),
            "alpha": _crop(alpha, roi).astype(np.float32), "visibility": float(visibility)}


def refine_layer(batch_index, refined, alpha):
    roi = alpha_roi(alpha)
    if roi is None:
        return None
    return {"kind": "refine", "batch_index": batch_index, "roi": roi, "content": _crop(refined, roi),
            "alpha": _crop(alpha, roi).astype(np.float32)}


def detailer_layer(batch_index, roi, content, alpha):
    """``content`` [h, w, 3] float32 and ``alpha`` [h, w] float32 of one SEGSPaste."""
    return {"kind": "detailer", "batch_index": batch_index, "roi": tuple(int(v) for v in roi),
            "content": np.ascontiguousarray(content, dtype=np.float32), "alpha": np.ascontiguousarray(alpha, dtype=np.float32)}


def paste_layers(image, segs, feather, alpha=255):
    """Detailer layers of ``SEGSPaste.doit(image, segs, feather, alpha)``: per SEG
    its paste region, the (resized) crop and the feathered mask, as pasted."""
    _batch, height, width, _ = image.shape
    scale = max(0.0, min(1.0, float(alpha) / 255.0))
    layers = []
    for seg in segs[1]:
        crop = getattr(seg, "cropped_image", None)
        region = getattr(seg, "crop_region", None)
        if not isinstance(crop, torch.Tensor) or region is None or len(region) < 4:
            continue
        x1, y1, x2, y2 = (max(0, min(limit, int(round(float(v))))) for v, limit in zip(region[:4], (width, height, width, height)))
        if x2 <= x1 or y2 <= y1:
            continue
        candidate = crop.to(dtype=torch.float32)
        if tuple(candidate.shape[1:3]) != (y2 - y1, x2 - x1):
            candidate = F.interpolate(candidate.movedim(-1, 1), size=(y2 - y1, x2 - x1), mode="bilinear",
                                      align_corners=False).movedim(1, -1)
        mask = _mask_tensor(getattr(seg, "cropped_mask", None), batch=1, height=y2 - y1, width=x2 - x1,
                            device="cpu", dtype=torch.float32)
        mask = _soft_mask(mask, int(feather)) * scale
        layers.append(detailer_layer(getattr(seg, "batch_index", None), (x1, y1, x2, y2),
                                     candidate[0].numpy(), mask[0].numpy()))
    return layers


def quantize_layer(mode, layers):
    """Branch-level uint8 conversion over the union of ``layers``' regions."""
    rois = [layer["roi"] for layer in layers if layer is not None]
    if not rois:
        return None
    roi = (min(r[0] for r in rois), min(r[1] for r in rois), max(r[2] for r in rois), max(r[3] for r in rois))
    return {"kind": "quantize", "batch_index": None, "roi": roi, "mode": mode}


def _to_uint8(region, mode):
    if mode == "round":  # tensor_to_uint8_rgb
        return (np.clip(region, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.clip(region.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)  # refine input


def _apply(item, layer):
    """Apply one layer to one image [H, W, 3] float32 (numpy, in place)."""
    x1, y1, x2, y2 = layer["roi"]
    region = item[y1:y2, x1:x2]
    kind = layer["kind"]
    if kind == "quantize":
        item[y1:y2, x1:x2] = _to_uint8(region, layer["mode"]).astype(np.float32) / 255.0
        return
    if kind == "detailer":
        base = torch.from_numpy(np.ascontiguousarray(region))
        mask = torch.from_numpy(layer["alpha"]).unsqueeze(-1)
        content = torch.from_numpy(layer["content"])
        item[y1:y2, x1:x2] = (base * (1.0 - mask) + content * mask).numpy()
        return
    # restore / refine: uint8 arithmetic of the 2.7 paths on an already quantised region
    current = (region * 255.0).round().astype(np.uint8)
    alpha = layer["alpha"][..., None]
    blended = np.clip(current.astype(np.float32) * (1.0 - alpha) + layer["content"].astype(np.float32) * alpha,
                      0, 255).astype(np.uint8)
    if kind == "restore":
        v = layer["visibility"]
        blended = np.clip(current.astype(np.float32) * (1.0 - v) + blended.astype(np.float32) * v, 0, 255).astype(np.uint8)
    item[y1:y2, x1:x2] = blended.astype(np.float32) / 255.0


def compose_layers(image, layers):
    """Apply ``layers`` in order to an IMAGE [B, H, W, 3]; returns a new tensor."""
    out = image.detach().cpu().numpy().astype(np.float32, copy=True)
    for layer in layers:
        targets = range(out.shape[0]) if layer.get("batch_index") is None else [int(layer["batch_index"])]
        for index in targets:
            _apply(out[index], layer)
    return torch.from_numpy(out)


def layers_roi(layers):
    rois = [layer["roi"] for layer in layers if layer["kind"] != "quantize"]
    if not rois:
        return None
    return min(r[0] for r in rois), min(r[1] for r in rois), max(r[2] for r in rois), max(r[3] for r in rois)


class CMKFaceLayerSEGS(CMKStableSEGS):
    """Stable FaceProcess SEGS that also carry the branch's processing layers.

    A subclass, so the Detailer's CMKStableSEGS and its pickles stay unchanged.
    """

    def __new__(cls, header, items=None, branch_image=None, branch_mask=None, source_signature=None,
                schema: int = _STABLE_SEGS_SCHEMA, layers=None, layer_schema=LAYER_SCHEMA):
        obj = super().__new__(cls, header, items, branch_image, branch_mask, source_signature, schema)
        obj.cmk_layers = list(layers or [])
        obj.cmk_layer_schema = layer_schema
        return obj

    def __reduce__(self):
        return (self.__class__, (self[0], self[1], self.cmk_branch_image, self.cmk_branch_mask,
                                 self.cmk_source_signature, self.cmk_stable_schema, self.cmk_layers, self.cmk_layer_schema))


def with_layers(stable, layers):
    """``make_stable_segs`` result -> CMKFaceLayerSEGS with ``layers``."""
    return CMKFaceLayerSEGS(stable[0], stable[1], branch_image=stable.cmk_branch_image, branch_mask=stable.cmk_branch_mask,
                            source_signature=stable.cmk_source_signature, schema=stable.cmk_stable_schema, layers=layers)
