from __future__ import annotations

import numpy as np
import torch

from ...utils.face_set_utils import face_bbox, normalize_selected_face, summarize_selected_face
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...utils.preview_utils import draw_face_boxes
from ...utils.tensor_utils import tensor_to_uint8_rgb


_MASK_SHAPES = ["Ellipse", "Box"]
_PREVIEW_MODES = ["Off", "Mask", "Overlay", "Outline", "Coverage"]


def _clamp_float(value, minimum: float, maximum: float, fallback: float) -> float:
    try:
        v = float(value)
    except Exception:
        return float(fallback)
    if not np.isfinite(v):
        return float(fallback)
    return max(float(minimum), min(float(maximum), v))


def _clamp_int(value, minimum: int, maximum: int, fallback: int) -> int:
    try:
        v = int(value)
    except Exception:
        return int(fallback)
    return max(int(minimum), min(int(maximum), v))


def _clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int):
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    return left, top, right, bottom


def _box_from_face(face: dict, width: int, height: int, padding: float):
    x1, y1, x2, y2 = face_bbox(face)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * float(padding)
    pad_y = bh * float(padding)
    return _clamp_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, width, height)


def _blur_mask(mask: np.ndarray, feather: int) -> np.ndarray:
    feather = int(feather)
    if feather <= 0:
        return mask.astype(np.float32)
    try:
        import cv2
        k = max(3, feather * 2 + 1)
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    except Exception:
        out = mask.astype(np.float32)
        for _ in range(max(1, feather // 4)):
            padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
            out = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0
        return out


def _make_mask(height: int, width: int, face: dict, shape: str, padding: float, feather: int, invert: bool) -> np.ndarray:
    left, top, right, bottom = _box_from_face(face, width, height, padding)
    mask = np.zeros((height, width), dtype=np.float32)

    shape = str(shape)
    if shape == "Box":
        mask[top:bottom, left:right] = 1.0
    else:
        yy, xx = np.ogrid[:height, :width]
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        rx = max(1.0, (right - left) / 2.0)
        ry = max(1.0, (bottom - top) / 2.0)
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        mask[ellipse] = 1.0

    mask = _blur_mask(mask, int(feather))
    mask = np.clip(mask, 0.0, 1.0)
    if bool(invert):
        mask = 1.0 - mask
    return mask.astype(np.float32)


def _outline_from_mask(mask: np.ndarray, thickness: int) -> np.ndarray:
    t = max(1, int(thickness))
    binary = mask > 0.5
    padded = np.pad(binary, ((t, t), (t, t)), mode="edge")
    eroded = np.ones_like(binary, dtype=bool)
    for dy in range(0, 2 * t + 1):
        for dx in range(0, 2 * t + 1):
            eroded &= padded[dy:dy + binary.shape[0], dx:dx + binary.shape[1]]
    return (binary ^ eroded).astype(np.float32)


def _add_footer(rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        line_h = 14
        pad = 6
        footer_h = pad * 2 + line_h * len(lines)
        out = Image.new("RGB", (img.width, img.height + footer_h), (0, 0, 0))
        out.paste(img, (0, 0))
        d = ImageDraw.Draw(out)
        y = img.height + pad
        for line in lines:
            d.text((pad, y), line, fill=(255, 255, 255))
            y += line_h
        return np.asarray(out, dtype=np.uint8)
    except Exception:
        return rgb


def _preview_from_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    face: dict,
    preview_mode: str,
    opacity: float,
    thickness: int,
    coverage_percent: float,
    coverage_pixels: int,
    shape: str,
    padding: float,
    feather: int,
    invert: bool,
) -> np.ndarray:
    mode = str(preview_mode)
    if mode == "Off":
        return rgb

    mask_rgb = np.repeat((mask * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    if mode == "Mask":
        return mask_rgb

    if mode == "Outline":
        out = rgb.copy()
        outline = _outline_from_mask(mask, int(thickness)) > 0
        out[outline] = 255
        return out

    alpha = max(0.0, min(1.0, float(opacity)))
    overlay = rgb.astype(np.float32) * (1.0 - mask[..., None] * alpha) + 255.0 * (mask[..., None] * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay = draw_face_boxes(overlay, [face], thickness=int(thickness), draw_boxes=True, draw_landmarks=False)

    if mode == "Coverage":
        overlay = _add_footer(overlay, [
            f"Shape: {shape} | Padding: {float(padding):.2f} | Feather: {int(feather)} px | Invert: {bool(invert)}",
            f"Coverage: {coverage_percent:.2f}% ({coverage_pixels} px)",
        ])
    return overlay


class CMKFaceMask:
    """Create a soft MASK from a CMK_SELECTED_FACE payload."""

    CATEGORY = "CMK/Toolbox/Face"
    RETURN_TYPES = ("IMAGE", "CMK_SELECTED_FACE", "MASK", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("image", "selected_face", "mask", "diagnostic")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "selected_face": ("CMK_SELECTED_FACE",),
                "shape": (_MASK_SHAPES, {"default": "Ellipse"}),
                "padding": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
                "feather": ("INT", {"default": 24, "min": 0, "max": 256, "step": 1}),
                "invert": ("BOOLEAN", {"default": False}),
                "preview_mode": (_PREVIEW_MODES, {"default": "Overlay"}),
                "preview_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.05}),
            }
        }

    def run(
        self,
        image: torch.Tensor,
        selected_face,
        shape: str,
        padding: float,
        feather: int,
        invert: bool,
        preview_mode: str,
        preview_opacity: float,
    ):
        payload = normalize_selected_face(selected_face)
        face = payload.get("face", {})

        # Compatibility guard: older workflows may contain stale widget values.
        # The UI default is 0.5, but execution clamps any persisted out-of-range value.
        shape = str(shape) if str(shape) in _MASK_SHAPES else "Ellipse"
        preview_mode = str(preview_mode) if str(preview_mode) in _PREVIEW_MODES else "Overlay"
        padding = _clamp_float(padding, 0.0, 2.0, 0.25)
        feather = _clamp_int(feather, 0, 256, 24)
        preview_opacity = _clamp_float(preview_opacity, 0.0, 1.0, 0.5)
        outline_thickness = 8

        masks = []
        previews = []
        coverage_values = []
        for i in range(int(image.shape[0])):
            rgb = tensor_to_uint8_rgb(image[i])
            h, w = rgb.shape[:2]
            mask = _make_mask(h, w, face, str(shape), float(padding), int(feather), bool(invert))
            coverage_pixels = int(np.count_nonzero(mask > 0.01))
            coverage_percent = (coverage_pixels / float(max(1, h * w))) * 100.0
            coverage_values.append(coverage_percent)
            preview = _preview_from_mask(
                rgb,
                mask,
                face,
                str(preview_mode),
                float(preview_opacity),
                int(outline_thickness),
                coverage_percent,
                coverage_pixels,
                str(shape),
                float(padding),
                int(feather),
                bool(invert),
            )
            masks.append(torch.from_numpy(mask).float())
            previews.append(preview)

        coverage_avg = float(np.mean(coverage_values)) if coverage_values else 0.0
        summary = "\n".join([
            "operation: Face Mask",
            f"shape: {shape}",
            f"padding: {float(padding)}",
            f"feather: {int(feather)}",
            f"invert: {bool(invert)}",
            f"coverage_avg: {coverage_avg:.2f}%",
            "",
            summarize_selected_face(payload),
        ])
        diagnostic = make_diagnostic_payload(
            title="Face Mask",
            node="CMK Face Mask",
            previews=previews,
            summary=summary,
            details=summary,
            mode=str(preview_mode),
            metadata={
                "shape": str(shape),
                "padding": float(padding),
                "feather": int(feather),
                "invert": bool(invert),
                "coverage_avg": coverage_avg,
            },
            metrics={"coverage_avg": coverage_avg},
        )
        return (image, selected_face, torch.stack(masks, dim=0), diagnostic)
