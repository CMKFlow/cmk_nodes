from __future__ import annotations

import numpy as np


def _clip_rgb(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _bbox_values(face) -> tuple[float, float, float, float] | None:
    bbox = getattr(face, "bbox", None)
    if bbox is None and isinstance(face, dict):
        bbox = face.get("bbox")
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _landmarks(face) -> np.ndarray | None:
    for name in ("landmark_2d_106", "kps", "landmark_3d_68"):
        pts = getattr(face, name, None)
        if pts is None and isinstance(face, dict):
            pts = face.get(name)
        if pts is None:
            continue
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 5 and arr.shape[1] >= 2:
            return arr[:, :2]
    return None


def _blur_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    try:
        import cv2
        k = radius * 2 + 1
        return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0), 0.0, 1.0)
    except Exception:
        out = mask.astype(np.float32)
        for _ in range(min(radius, 16)):
            padded = np.pad(out, 1, mode="edge")
            out = (
                padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
                + padded[1:-1, 1:-1]
            ) / 5.0
        return np.clip(out, 0.0, 1.0)


def _face_shape_mask(height: int, width: int, face, *, bbox_dilation: int = 0, crop_factor: float = 1.0) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    bbox = _bbox_values(face)
    pts = _landmarks(face)
    crop_factor = min(3.0, max(1.0, float(crop_factor)))
    bbox_dilation = int(round(float(bbox_dilation)))

    try:
        import cv2
        if pts is not None:
            points = pts.copy()
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                if bbox_dilation != 0 or crop_factor > 1.0:
                    grow_x = max(0.0, (crop_factor - 1.0) * 0.10 * bw) + max(0.0, float(bbox_dilation))
                    grow_y = max(0.0, (crop_factor - 1.0) * 0.13 * bh) + max(0.0, float(bbox_dilation))
                    shrink_x = max(0.0, -float(bbox_dilation))
                    shrink_y = max(0.0, -float(bbox_dilation))
                    x1 = x1 - grow_x + shrink_x
                    x2 = x2 + grow_x - shrink_x
                    y1 = y1 - grow_y + shrink_y
                    y2 = y2 + grow_y - shrink_y
                    bw = max(1.0, x2 - x1)
                    bh = max(1.0, y2 - y1)
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                scale_x = 1.04 + max(0.0, crop_factor - 1.0) * 0.05
                scale_y = 1.03 + max(0.0, crop_factor - 1.0) * 0.07
                points[:, 0] = cx + (points[:, 0] - cx) * scale_x
                points[:, 1] = cy + (points[:, 1] - cy) * scale_y
                points[:, 0] = np.clip(points[:, 0], x1 + bw * 0.01, x2 - bw * 0.01)
                points[:, 1] = np.clip(points[:, 1], y1 + bh * 0.01, y2 - bh * 0.005)
            hull = cv2.convexHull(np.round(points).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 1.0)
            return mask

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            grow_x = max(0.0, (crop_factor - 1.0) * 0.10 * bw) + max(0.0, float(bbox_dilation))
            grow_y = max(0.0, (crop_factor - 1.0) * 0.13 * bh) + max(0.0, float(bbox_dilation))
            center = (int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.52)))
            axes = (int(round((bw * 0.50) + grow_x)), int(round((bh * 0.57) + grow_y)))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
            return mask
    except Exception:
        pass

    if bbox is None:
        mask[:, :] = 1.0
        return mask

    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    grow_x = max(0.0, (crop_factor - 1.0) * 0.10 * bw) + max(0.0, float(bbox_dilation))
    grow_y = max(0.0, (crop_factor - 1.0) * 0.13 * bh) + max(0.0, float(bbox_dilation))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.52
    yy, xx = np.ogrid[:height, :width]
    ellipse = ((xx - cx) / max(1.0, (bw * 0.50) + grow_x)) ** 2 + ((yy - cy) / max(1.0, (bh * 0.57) + grow_y)) ** 2 <= 1.0
    mask[ellipse] = 1.0
    return mask


def _morph_mask(mask: np.ndarray, radius: int, mode: str) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return np.clip(mask.astype(np.float32), 0.0, 1.0)
    try:
        import cv2
        kernel_size = max(3, radius * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        base = np.clip(mask.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
        if mode == "erode":
            morphed = cv2.erode(base, kernel, iterations=1)
        else:
            morphed = cv2.dilate(base, kernel, iterations=1)
        return (morphed > 0).astype(np.float32)
    except Exception:
        out = np.clip(mask.astype(np.float32), 0.0, 1.0)
        for _ in range(min(radius, 16)):
            padded = np.pad(out, 1, mode="edge")
            if mode == "erode":
                out = np.minimum.reduce([
                    padded[:-2, 1:-1],
                    padded[2:, 1:-1],
                    padded[1:-1, :-2],
                    padded[1:-1, 2:],
                    padded[1:-1, 1:-1],
                ])
            else:
                out = np.maximum.reduce([
                    padded[:-2, 1:-1],
                    padded[2:, 1:-1],
                    padded[1:-1, :-2],
                    padded[1:-1, 2:],
                    padded[1:-1, 1:-1],
                ])
        return np.clip(out, 0.0, 1.0)


def _expand_face_mask(mask: np.ndarray, face, *, bbox_dilation: int = 0, crop_factor: float = 1.0) -> np.ndarray:
    out = np.clip(mask.astype(np.float32), 0.0, 1.0)
    crop_factor = min(3.0, max(1.0, float(crop_factor)))
    dilation_px = int(round(float(bbox_dilation)))
    bbox = _bbox_values(face)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        face_extent = max(1.0, max(x2 - x1, y2 - y1))
    else:
        face_extent = max(1.0, float(min(mask.shape[0], mask.shape[1])))

    crop_extra = max(0.0, crop_factor - 1.0)
    crop_radius = int(round(face_extent * min(0.18, crop_extra * 0.09)))
    radius = max(0, dilation_px) + max(0, crop_radius)
    if radius > 0:
        out = _morph_mask(out, radius, "dilate")

    if dilation_px < 0:
        out = _morph_mask(out, abs(dilation_px), "erode")

    return np.clip(out, 0.0, 1.0)


def _changed_mask(target: np.ndarray, result: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(result.astype(np.float32) - target.astype(np.float32)), axis=2)
    return (diff > 2.0).astype(np.float32)


def _refine_mask(mask: np.ndarray, height: int, width: int, feather: int = 0) -> tuple[np.ndarray, np.ndarray]:
    scale = max(1, min(int(height), int(width)))
    try:
        import cv2
        base = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        close_k = max(3, int(round(scale * 0.006)) | 1)
        erode_k = max(3, int(round(scale * 0.007)) | 1)
        dilate_k = max(3, int(round(scale * 0.004)) | 1)
        feather_px = max(1, int(feather)) if int(feather) > 0 else max(3, int(round(scale * 0.007)))

        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8), iterations=1)
        core_u8 = cv2.erode(base, np.ones((erode_k, erode_k), np.uint8), iterations=1)
        outer_u8 = cv2.dilate(base, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)

        outer = (outer_u8 > 0).astype(np.uint8)
        dist_in = cv2.distanceTransform(outer, cv2.DIST_L2, 3).astype(np.float32)
        alpha = np.clip(dist_in / float(feather_px), 0.0, 1.0)

        # Guarantee a solid center while keeping the transition band narrow.
        core = (core_u8 > 0).astype(np.float32)
        soft = np.maximum(alpha * outer.astype(np.float32), core)
        soft = cv2.GaussianBlur(soft, (0, 0), max(0.35, feather_px * 0.18))
        soft = np.clip(soft, 0.0, 1.0)
    except Exception:
        core = np.clip(mask, 0.0, 1.0)
        feather_px = max(1, int(feather)) if int(feather) > 0 else max(3, int(round(scale * 0.007)))
        soft = np.clip(_blur_mask(mask, feather_px), 0.0, 1.0)

    core = np.clip(core, 0.0, 1.0)
    return soft.astype(np.float32), core.astype(np.float32)


def _match_color_subtle(target: np.ndarray, result: np.ndarray, core_mask: np.ndarray) -> np.ndarray:
    weights = np.clip(core_mask.astype(np.float32), 0.0, 1.0)
    if float(weights.mean()) < 0.0001:
        return result
    w = weights[..., None]
    denom = float(np.sum(weights)) + 1e-6
    target_f = target.astype(np.float32)
    result_f = result.astype(np.float32)
    target_mean = np.sum(target_f * w, axis=(0, 1)) / denom
    result_mean = np.sum(result_f * w, axis=(0, 1)) / denom
    target_var = np.sum(((target_f - target_mean) ** 2) * w, axis=(0, 1)) / denom
    result_var = np.sum(((result_f - result_mean) ** 2) * w, axis=(0, 1)) / denom
    target_std = np.sqrt(np.maximum(target_var, 1.0))
    result_std = np.sqrt(np.maximum(result_var, 1.0))

    matched = (result_f - result_mean) * (target_std / result_std) + target_mean
    # Conservative correction: reduce visible seams without overriding the swapper result.
    corrected = result_f * 0.70 + matched * 0.30
    return _clip_rgb(corrected)


def _sharpen_face_subtle(result: np.ndarray, core_mask: np.ndarray) -> np.ndarray:
    weights = np.clip(core_mask.astype(np.float32), 0.0, 1.0)
    if float(weights.mean()) < 0.0001:
        return result
    try:
        import cv2
        blur = cv2.GaussianBlur(result.astype(np.float32), (0, 0), 1.05)
        mask = cv2.GaussianBlur(weights, (0, 0), 1.25)
    except Exception:
        blur = result.astype(np.float32)
        for _ in range(2):
            padded = np.pad(blur, ((1, 1), (1, 1), (0, 0)), mode="edge")
            blur = (
                padded[:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
                + padded[1:-1, 1:-1]
            ) / 5.0
        mask = _blur_mask(weights, 2)

    result_f = result.astype(np.float32)
    detail = result_f - blur.astype(np.float32)
    sharpened = result_f + detail * 0.38
    mask = np.clip(mask * 0.85, 0.0, 0.85)[..., None]
    out = result_f * (1.0 - mask) + sharpened * mask
    return _clip_rgb(out)



def _odd_kernel(value: int, minimum: int = 1) -> int:
    size = max(int(minimum), int(value))
    return size if size % 2 == 1 else size + 1


def pasteback_native(
    target_rgb: np.ndarray,
    aligned_result: np.ndarray,
    target_face,
    *,
    affine_matrix: np.ndarray | None = None,
    bbox_dilation: int = 0,
    crop_factor: float = 1.0,
    feather: int = 0,
    return_mask: bool = False,
):
    """Paste an aligned face with separate identity and blend masks.

    The exact swap affine is retained. Geometric crop padding is excluded first;
    identity transfer is then restricted to a slightly eroded face-shaped core,
    while a separately dilated and Gaussian-blurred mask provides the external
    transition. Hair, ears, jewellery and background therefore remain target
    content instead of becoming part of the swapped identity region.
    """
    target = _clip_rgb(target_rgb)
    aligned = _clip_rgb(aligned_result)
    if aligned.ndim != 3 or aligned.shape[2] != 3:
        raise ValueError("CMK Face Engine: aligned swap result is not an RGB image")

    try:
        import cv2

        matrix = None
        if affine_matrix is not None:
            candidate = np.asarray(affine_matrix, dtype=np.float32)
            if candidate.shape == (2, 3) and bool(np.isfinite(candidate).all()):
                matrix = candidate

        if matrix is None:
            from insightface.utils import face_align
            kps = getattr(target_face, "kps", None)
            if kps is None and isinstance(target_face, dict):
                kps = target_face.get("kps")
            if kps is None:
                raise ValueError("target face has no five-point landmarks")
            image_size = int(aligned.shape[0])
            _, matrix = face_align.norm_crop2(
                target,
                landmark=np.asarray(kps),
                image_size=image_size,
            )

        inverse = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32))
        height, width = target.shape[:2]
        aligned_height, aligned_width = aligned.shape[:2]

        warped = cv2.warpAffine(
            aligned,
            inverse,
            (width, height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Build the paste mask from pixels that genuinely originate inside the
        # target frame. A rotated/aligned face crop may contain black padding
        # where the affine transform sampled outside the source image. Using a
        # full white crop mask would paste that padding back as a visible,
        # rotated rectangle around the face.
        target_support = np.full((height, width), 255, dtype=np.uint8)
        aligned_support = cv2.warpAffine(
            target_support,
            np.asarray(matrix, dtype=np.float32),
            (aligned_width, aligned_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        aligned_support = np.where(aligned_support >= 250, 255, 0).astype(np.uint8)

        # Keep a small safety distance from the affine crop boundary. This is
        # independent of image brightness, so dark hair and clothing remain
        # valid content while only geometric padding is excluded.
        support_margin = max(2, int(round(min(aligned_width, aligned_height) * 0.04)))
        aligned_support = cv2.erode(
            aligned_support,
            np.ones((support_margin * 2 + 1, support_margin * 2 + 1), dtype=np.uint8),
            iterations=1,
        )
        if not np.any(aligned_support):
            raise RuntimeError("aligned valid-content mask is empty")

        support_mask = cv2.warpAffine(
            aligned_support,
            inverse,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        support_mask = np.where(support_mask > 20, 1.0, 0.0).astype(np.float32)

        # Phase 1 compositing:
        #   swap_mask  = the actual identity-transfer area
        #   blend_mask = a slightly larger, softly blurred transition area
        #
        # The previous path derived both responsibilities from the affine crop
        # support. That support is geometrically safe but can extend into hair,
        # ears and jewellery. Restrict identity transfer to the landmark/bbox
        # face shape, while retaining a narrow external blend band.
        crop_factor = min(3.0, max(1.0, float(crop_factor)))
        face_shape = _face_shape_mask(
            height,
            width,
            target_face,
            bbox_dilation=int(bbox_dilation),
            crop_factor=crop_factor,
        )
        base_mask = np.clip(face_shape * support_mask, 0.0, 1.0)

        if float(base_mask.max()) <= 0.0:
            # Defensive fallback for incomplete third-party face objects.
            base_mask = support_mask

        bbox = _bbox_values(target_face)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            face_extent = max(1.0, float(max(x2 - x1, y2 - y1)))
        else:
            ys, xs = np.where(base_mask > 0.5)
            if len(xs) == 0 or len(ys) == 0:
                raise RuntimeError("inverse-warped face mask is empty")
            face_extent = max(
                1.0,
                float(max(xs.max() - xs.min(), ys.max() - ys.min())),
            )

        # Close one-pixel landmark gaps before separating identity and blend.
        close_radius = max(1, int(round(face_extent * 0.004)))
        base_mask = _morph_mask(base_mask, close_radius, "dilate")
        base_mask = _morph_mask(base_mask, close_radius, "erode")

        # Keep hairline, ears and jewellery in the target image. The identity
        # core is pulled inward only minimally (typically 1–2 px).
        swap_erode = max(1, int(round(face_extent * 0.008)))
        swap_mask = _morph_mask(base_mask, swap_erode, "erode")
        if float(swap_mask.max()) <= 0.0:
            swap_mask = base_mask

        # The blend zone extends beyond the identity core without broadening the
        # actual swapped region. Automatic width scales with the detected face.
        blend_dilate = max(2, int(round(face_extent * 0.015)))
        blend_outer = _morph_mask(swap_mask, blend_dilate, "dilate")
        blend_outer = np.clip(blend_outer * support_mask, 0.0, 1.0)

        # Existing FEATHER remains an optional explicit override. With FEATHER=0
        # the transition is adaptive: roughly 1.5% of the detected face extent.
        feather_px = (
            min(100, max(1, int(feather)))
            if int(feather) > 0
            else max(4, int(round(face_extent * 0.015)))
        )
        # Feather continuously from the visible mask boundary into its center.
        # The previous maximum(solid_core, blurred_outer) retained a fixed alpha
        # jump at the core edge, so values such as 20 and 50 looked effectively
        # identical.  A distance ramp has an unambiguous pixel meaning: the
        # requested number is the width needed to reach full opacity.
        blend_distance = cv2.distanceTransform(
            (blend_outer > 0.5).astype(np.uint8),
            cv2.DIST_L2,
            3,
        ).astype(np.float32)
        soft_outer = np.clip(
            blend_distance / float(max(1, feather_px)),
            0.0,
            1.0,
        )

        # The affine crop support is a safety boundary, not a visible paste
        # boundary.  Multiplying the finished alpha by its binary mask used to
        # cut the feathered transition off again and could expose the slanted
        # top edge of the aligned face crop across the forehead.  Fade that
        # support boundary inward instead: pixels outside the valid crop remain
        # strictly excluded, while FEATHER can still soften its edge.
        support_distance = cv2.distanceTransform(
            (support_mask > 0.5).astype(np.uint8),
            cv2.DIST_L2,
            3,
        ).astype(np.float32)
        support_alpha = np.clip(
            support_distance / float(max(1, feather_px)),
            0.0,
            1.0,
        )

        alpha_2d = np.clip(soft_outer * support_alpha, 0.0, 1.0)
        alpha = alpha_2d[..., None]

        result = target.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha
        result_rgb = _clip_rgb(result)
        if return_mask:
            return result_rgb, np.clip(alpha[..., 0], 0.0, 1.0).astype(np.float32)
        return result_rgb
    except Exception as exc:
        raise RuntimeError(f"CMK Face Engine paste-back failed: {exc}") from exc

def pasteback_lanczos(
    target_rgb: np.ndarray,
    aligned_result: np.ndarray,
    target_face,
    *,
    affine_matrix: np.ndarray | None = None,
    bbox_dilation: int = 0,
    crop_factor: float = 1.0,
    feather: int = 0,
) -> np.ndarray:
    """Compatibility wrapper for the seam-safe native-style paste-back."""
    return pasteback_native(
        target_rgb,
        aligned_result,
        target_face,
        affine_matrix=affine_matrix,
        bbox_dilation=bbox_dilation,
        crop_factor=crop_factor,
        feather=feather,
    )


def clean_pasteback(target_rgb: np.ndarray, raw_result: np.ndarray, target_face, *, bbox_dilation: int = 0, crop_factor: float = 1.0, feather: int = 0) -> np.ndarray:
    """Improve InsightFace paste-back boundaries without adding UI controls."""
    target = _clip_rgb(target_rgb)
    result = _clip_rgb(raw_result)
    if target.shape != result.shape or target.ndim != 3 or target.shape[2] != 3:
        return result

    height, width = target.shape[:2]
    changed = _changed_mask(target, result)
    if float(changed.mean()) <= 0.00001:
        return result

    face_mask = _face_shape_mask(height, width, target_face, bbox_dilation=bbox_dilation, crop_factor=crop_factor)
    face_mask = _expand_face_mask(face_mask, target_face, bbox_dilation=bbox_dilation, crop_factor=crop_factor)
    # The changed area from InsightFace may include a broad pasted region.
    # Intersect it with a tighter face mask so cheeks/neck/background stay crisp.
    mask = changed * face_mask
    soft_mask, core_mask = _refine_mask(mask, height, width, feather=feather)

    color_result = _match_color_subtle(target, result, core_mask)
    color_result = _sharpen_face_subtle(color_result, core_mask)
    alpha = soft_mask[..., None]
    out = target.astype(np.float32) * (1.0 - alpha) + color_result.astype(np.float32) * alpha
    return _clip_rgb(out)
