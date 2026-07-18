from __future__ import annotations

import numpy as np
import torch

from ...utils.face_set_utils import face_bbox, normalize_selected_face, summarize_selected_face
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...utils.preview_utils import draw_face_boxes
from ...utils.tensor_utils import tensor_to_uint8_rgb, uint8_rgb_to_tensor


_RESTORE_METHODS = ["Detail", "Sharpen", "Smooth"]
_PREVIEW_MODES = ["Off", "Result", "Difference", "Comparison", "Mask"]


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


def _blur_rgb(rgb: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    try:
        import cv2
        k = max(3, radius * 2 + 1)
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(rgb, (k, k), 0)
    except Exception:
        out = rgb.astype(np.float32)
        for _ in range(max(1, radius)):
            padded = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
            out = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0
        return np.clip(out, 0, 255).astype(np.uint8)


def _soften_mask(mask: np.ndarray, radius: int = 8) -> np.ndarray:
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if radius <= 0:
        return mask
    try:
        import cv2
        k = max(3, int(radius) * 2 + 1)
        if k % 2 == 0:
            k += 1
        return np.clip(cv2.GaussianBlur(mask, (k, k), 0), 0.0, 1.0).astype(np.float32)
    except Exception:
        out = mask.astype(np.float32)
        for _ in range(max(1, int(radius) // 4)):
            padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
            out = (
                padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
            ) / 9.0
        return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fallback_face_mask(height: int, width: int, face: dict, padding: float) -> np.ndarray:
    x1, y1, x2, y2 = face_bbox(face)
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    pad_x = bw * float(padding)
    pad_y = bh * float(padding)
    left = max(0.0, float(x1) - pad_x)
    top = max(0.0, float(y1) - pad_y)
    right = min(float(width), float(x2) + pad_x)
    bottom = min(float(height), float(y2) + pad_y)

    yy, xx = np.ogrid[:height, :width]
    cx = (left + right) * 0.5
    cy = (top + bottom) * 0.5
    rx = max(1.0, (right - left) * 0.5)
    ry = max(1.0, (bottom - top) * 0.5)
    mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(np.float32)
    return _soften_mask(mask, 10)


def _mask_for_batch(mask, index: int, height: int, width: int) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)

    if arr.ndim == 3:
        idx = max(0, min(int(index), arr.shape[0] - 1))
        arr = arr[idx]
    elif arr.ndim == 2:
        pass
    else:
        return None

    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != (height, width):
        try:
            import cv2
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _apply_restore(rgb: np.ndarray, method: str, strength: float) -> np.ndarray:
    strength = _clamp_float(strength, 0.0, 2.0, 0.65)
    method = str(method)
    base = rgb.astype(np.float32)

    if method == "Smooth":
        blurred = _blur_rgb(rgb, radius=max(1, int(2 + 4 * strength))).astype(np.float32)
        # Smooth removes small artifacts but keeps some original texture.
        out = base * (1.0 - 0.65 * strength) + blurred * (0.65 * strength)
        return np.clip(out, 0, 255).astype(np.uint8)

    if method == "Sharpen":
        blurred = _blur_rgb(rgb, radius=max(1, int(1 + 2 * strength))).astype(np.float32)
        detail = base - blurred
        out = base + detail * (1.25 * strength)
        return np.clip(out, 0, 255).astype(np.uint8)

    # Detail: mild local contrast + unsharp mask. This is the default native restore.
    small_blur = _blur_rgb(rgb, radius=max(1, int(1 + strength))).astype(np.float32)
    large_blur = _blur_rgb(rgb, radius=max(2, int(3 + 4 * strength))).astype(np.float32)
    fine = base - small_blur
    local = small_blur - large_blur
    out = base + fine * (0.85 * strength) + local * (0.35 * strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def _blend_with_mask(original: np.ndarray, restored: np.ndarray, mask: np.ndarray, blend: float) -> np.ndarray:
    b = _clamp_float(blend, 0.0, 1.0, 0.75)
    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None] * b
    out = original.astype(np.float32) * (1.0 - alpha) + restored.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _difference_image(original: np.ndarray, result: np.ndarray, mask: np.ndarray) -> np.ndarray:
    diff = np.abs(result.astype(np.int16) - original.astype(np.int16)).astype(np.float32)
    diff *= np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None]
    # Amplify for visibility; native restore may be subtle.
    diff = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
    return diff


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == height:
        return image
    out = np.zeros((height, w, 3), dtype=np.uint8)
    out[:h, :w] = image
    return out


def _side_by_side(*images: np.ndarray) -> np.ndarray:
    height = max(int(img.shape[0]) for img in images)
    parts = []
    for idx, img in enumerate(images):
        if idx:
            parts.append(np.zeros((height, 12, 3), dtype=np.uint8))
        parts.append(_pad_to_height(img, height))
    return np.concatenate(parts, axis=1)


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None], 3, axis=2)


def _make_preview(
    original: np.ndarray,
    result: np.ndarray,
    face: dict,
    mask: np.ndarray,
    preview_mode: str,
    preview_thickness: int,
) -> np.ndarray:
    mode = str(preview_mode)
    if mode == "Off":
        return result
    if mode == "Mask":
        return _mask_rgb(mask)

    marked_result = draw_face_boxes(
        result.copy(),
        [face],
        thickness=int(preview_thickness),
        draw_boxes=True,
        draw_landmarks=False,
    )

    if mode == "Result":
        return marked_result

    diff = _difference_image(original, result, mask)
    if mode == "Difference":
        return diff

    if mode == "Comparison":
        marked_original = draw_face_boxes(
            original.copy(),
            [face],
            thickness=int(preview_thickness),
            draw_boxes=True,
            draw_landmarks=False,
        )
        return _side_by_side(marked_original, marked_result, diff)

    return marked_result


class CMKFaceRestore:
    """Native mask-aware face restore/enhance node."""

    CATEGORY = "CMK/Toolbox/Face"
    RETURN_TYPES = ("IMAGE", "CMK_SELECTED_FACE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("image", "selected_face", "diagnostic")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "selected_face": ("CMK_SELECTED_FACE",),
                "mask": ("MASK",),
                "enabled": ("BOOLEAN", {"default": True}),
                "method": (_RESTORE_METHODS, {"default": "Detail"}),
                "restore_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 2.0, "step": 0.05}),
                "blend": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "fallback_padding": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
                "preview_mode": (_PREVIEW_MODES, {"default": "Comparison"}),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        selected_face,
        mask,
        enabled: bool,
        method: str,
        restore_strength: float,
        blend: float,
        fallback_padding: float,
        preview_mode: str,
    ):
        payload = normalize_selected_face(selected_face)
        face = payload.get("face", {})

        method = str(method) if str(method) in _RESTORE_METHODS else "Detail"
        preview_mode = str(preview_mode) if str(preview_mode) in _PREVIEW_MODES else "Comparison"
        restore_strength = _clamp_float(restore_strength, 0.0, 2.0, 0.65)
        blend = _clamp_float(blend, 0.0, 1.0, 0.75)
        fallback_padding = _clamp_float(fallback_padding, 0.0, 2.0, 0.25)
        preview_thickness = 8

        outputs = []
        previews = []
        changed_values = []
        mask_source = "input_mask" if mask is not None else "fallback_face_ellipse"

        for i in range(int(image.shape[0])):
            original = tensor_to_uint8_rgb(image[i])
            h, w = original.shape[:2]
            work_mask = _mask_for_batch(mask, i, h, w)
            if work_mask is None:
                work_mask = _fallback_face_mask(h, w, face, fallback_padding)
            else:
                work_mask = _soften_mask(work_mask, 4)

            if bool(enabled):
                restored_full = _apply_restore(original, method, restore_strength)
                result = _blend_with_mask(original, restored_full, work_mask, blend)
            else:
                result = original.copy()

            diff_amount = float(np.mean(np.abs(result.astype(np.float32) - original.astype(np.float32))))
            changed_values.append(diff_amount)
            previews.append(_make_preview(original, result, face, work_mask, preview_mode, preview_thickness))
            outputs.append(uint8_rgb_to_tensor(result))

        changed_avg = float(np.mean(changed_values)) if changed_values else 0.0
        summary = "\n".join([
            "operation: Face Restore",
            f"enabled: {bool(enabled)}",
            f"method: {method}",
            f"restore_strength: {restore_strength:.3f}",
            f"blend: {blend:.3f}",
            f"mask_source: {mask_source}",
            f"changed_avg: {changed_avg:.4f}",
            "",
            summarize_selected_face(payload),
        ])

        diagnostic = make_diagnostic_payload(
            title="Face Restore",
            node="CMK Face Restore",
            previews=previews,
            summary=summary,
            details=summary,
            mode=str(preview_mode),
            metadata={
                "enabled": bool(enabled),
                "method": method,
                "restore_strength": restore_strength,
                "blend": blend,
                "mask_source": mask_source,
                "changed_avg": changed_avg,
            },
            metrics={"changed_avg": changed_avg},
        )

        return (torch.stack(outputs, dim=0), selected_face, diagnostic)
