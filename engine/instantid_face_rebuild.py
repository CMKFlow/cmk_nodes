from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceRebuildGeometry:
    face_bbox: tuple[int, int, int, int]
    roi: tuple[int, int, int, int]
    working_size: tuple[int, int]
    feather: int


def shape_strength_from_balance(source_target_balance: int) -> float:
    """Couple the five-position SOURCE/TARGET control to shape transfer."""
    balance = max(-2, min(2, int(source_target_balance)))
    return float(balance + 2) / 4.0


def dense_face_landmarks(face) -> np.ndarray | None:
    """Return the densest stable 2D landmark set exposed by InsightFace."""
    raw = face.get("raw") if isinstance(face, dict) else face
    for name in ("landmark_2d_106", "landmark_3d_68", "kps"):
        value = getattr(raw, name, None)
        if value is None and isinstance(face, dict):
            value = face.get(name)
        if value is None:
            continue
        points = np.asarray(value, dtype=np.float32)
        if points.ndim == 2 and points.shape[0] >= 5 and points.shape[1] >= 2:
            return points[:, :2]
    return None


def source_shape_in_target_pose(source_face, target_face) -> tuple[np.ndarray, np.ndarray] | None:
    """Normalize dense source geometry to the target's scale, rotation and position."""
    source_points = dense_face_landmarks(source_face)
    target_points = dense_face_landmarks(target_face)
    source_kps = dense_face_landmarks({"kps": source_face.get("kps")}) if isinstance(source_face, dict) else None
    target_kps = dense_face_landmarks({"kps": target_face.get("kps")}) if isinstance(target_face, dict) else None
    if source_points is None or target_points is None or source_kps is None or target_kps is None:
        return None
    if source_points.shape != target_points.shape or source_points.shape[0] <= 5:
        return None
    matrix = None
    try:
        import cv2

        matrix, _inliers = cv2.estimateAffinePartial2D(
            source_kps, target_kps, method=cv2.LMEDS,
        )
    except Exception:
        pass
    if matrix is None:
        # Dependency-light similarity fallback used by contract tests and
        # environments where only the runtime sampling process ships OpenCV.
        rows = []
        rhs = []
        for (sx, sy), (tx, ty) in zip(source_kps, target_kps):
            rows.extend(((sx, -sy, 1.0, 0.0), (sy, sx, 0.0, 1.0)))
            rhs.extend((tx, ty))
        try:
            a, b, shift_x, shift_y = np.linalg.lstsq(
                np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64), rcond=None,
            )[0]
            matrix = np.asarray(((a, -b, shift_x), (b, a, shift_y)), dtype=np.float32)
        except Exception:
            matrix = None
    if matrix is None:
        return None
    homogeneous = np.concatenate(
        (source_points, np.ones((source_points.shape[0], 1), dtype=np.float32)), axis=1,
    )
    desired = homogeneous @ np.asarray(matrix, dtype=np.float32).T

    # Five-point alignment can make the whole source landmark hull larger or
    # smaller when expression and mouth geometry differ. Shape transfer must
    # change proportions, not face area, so match the dense hull area to the
    # target before extracting its residual deformation.
    target_span = np.ptp(target_points, axis=0)
    desired_span = np.ptp(desired, axis=0)
    target_area = max(float(target_span[0] * target_span[1]), 1.0)
    desired_area = max(float(desired_span[0] * desired_span[1]), 1.0)
    area_scale = float(np.sqrt(target_area / desired_area))
    target_center = np.mean(target_points, axis=0)
    desired_center = np.mean(desired, axis=0)
    desired = (desired - desired_center) * area_scale + target_center

    # A similarity transform removes global rotation and scale but a residual
    # from a differently yawed source is still asymmetric. Project that
    # residual onto bilateral shape changes in the target's eye-line frame.
    eye_vector = target_kps[1] - target_kps[0]
    eye_distance = max(float(np.linalg.norm(eye_vector)), 1.0)
    axis_x = eye_vector / eye_distance
    axis_y = np.asarray((-axis_x[1], axis_x[0]), dtype=np.float32)
    rotation = np.stack((axis_x, axis_y), axis=1)
    center = (target_kps[0] + target_kps[1]) * 0.5
    local_points = (target_points - center) @ rotation
    local_delta = (desired - target_points) @ rotation
    neutral_delta = np.zeros_like(local_delta)
    for index, point in enumerate(local_points):
        mirror_error = (
            (local_points[:, 0] + point[0]) ** 2
            + (local_points[:, 1] - point[1]) ** 2
        )
        partner = int(np.argmin(mirror_error))
        partner_point = local_points[partner]
        # Reject unrelated nearest neighbours. Midline landmarks are their own
        # partner and must never introduce a horizontal pose shift.
        if float(mirror_error[partner]) > (eye_distance * 0.22) ** 2:
            neutral_delta[index] = (0.0, local_delta[index, 1])
        elif abs(float(point[0])) < eye_distance * 0.08:
            neutral_delta[index] = (0.0, local_delta[index, 1])
        else:
            neutral_delta[index, 0] = (
                local_delta[index, 0] - local_delta[partner, 0]
            ) * 0.5
            neutral_delta[index, 1] = (
                local_delta[index, 1] + local_delta[partner, 1]
            ) * 0.5
    desired_neutral = target_points + neutral_delta @ rotation.T

    # The target five-point constellation is the explicit pose contract.
    # Adding it as zero-displacement anchors prevents the dense field from
    # rotating or frontalising eyes, nose and mouth as shape strength rises.
    anchored_target = np.concatenate((target_points, target_kps), axis=0).astype(np.float32)
    anchored_desired = np.concatenate((desired_neutral, target_kps), axis=0).astype(np.float32)
    return anchored_target, anchored_desired


def warp_roi_towards_shape(
    target_roi: np.ndarray,
    target_points: np.ndarray,
    desired_points: np.ndarray,
    geometry: FaceRebuildGeometry,
    strength: float,
) -> np.ndarray:
    """Smoothly deform a target ROI toward normalized source face landmarks."""
    amount = max(0.0, min(1.0, float(strength)))
    if amount <= 0.0:
        return target_roi
    try:
        import cv2
    except Exception:
        return target_roi

    left, top, right, bottom = geometry.roi
    work_w, work_h = geometry.working_size
    scale = np.asarray(
        [work_w / max(1.0, right - left), work_h / max(1.0, bottom - top)],
        dtype=np.float32,
    )
    origin = np.asarray([left, top], dtype=np.float32)
    target = (np.asarray(target_points, dtype=np.float32) - origin) * scale
    desired = (np.asarray(desired_points, dtype=np.float32) - origin) * scale
    movement = (desired - target) * amount

    # Guard against detector outliers. A face-shape transfer should alter
    # proportions, never fling a landmark across the ROI.
    max_shift = max(2.0, min(work_w, work_h) * 0.12 * amount)
    lengths = np.linalg.norm(movement, axis=1, keepdims=True)
    movement *= np.minimum(1.0, max_shift / np.maximum(lengths, 1e-6))

    weight_seed = np.zeros((work_h, work_w), dtype=np.float32)
    dx_seed = np.zeros_like(weight_seed)
    dy_seed = np.zeros_like(weight_seed)
    radius = max(2, int(round(min(work_w, work_h) * 0.008)))
    for point, delta in zip(target, movement):
        x, y = (int(round(float(point[0]))), int(round(float(point[1]))))
        if 0 <= x < work_w and 0 <= y < work_h:
            cv2.circle(weight_seed, (x, y), radius, 1.0, -1)
            cv2.circle(dx_seed, (x, y), radius, float(delta[0]), -1)
            cv2.circle(dy_seed, (x, y), radius, float(delta[1]), -1)

    sigma = max(8.0, min(work_w, work_h) * 0.045)
    weight = cv2.GaussianBlur(weight_seed, (0, 0), sigmaX=sigma, sigmaY=sigma)
    dx = cv2.GaussianBlur(dx_seed, (0, 0), sigmaX=sigma, sigmaY=sigma) / np.maximum(weight, 1e-6)
    dy = cv2.GaussianBlur(dy_seed, (0, 0), sigmaX=sigma, sigmaY=sigma) / np.maximum(weight, 1e-6)
    support = np.clip(weight / max(float(weight.max()) * 0.08, 1e-6), 0.0, 1.0)

    # Do not let the smoothed displacement leak into ears, hair or the ROI
    # boundary. The dense landmark hull covers facial geometry; a small soft
    # dilation gives the jaw room to move while tapering back to the untouched
    # target before external anatomy begins.
    face_support = np.zeros((work_h, work_w), dtype=np.uint8)
    hull = cv2.convexHull(np.round(target).astype(np.int32))
    cv2.fillConvexPoly(face_support, hull, 255)
    taper = max(3, int(round(min(work_w, work_h) * 0.018)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (taper * 2 + 1, taper * 2 + 1))
    face_support = cv2.dilate(face_support, kernel).astype(np.float32) / 255.0
    face_support = cv2.GaussianBlur(face_support, (0, 0), sigmaX=taper, sigmaY=taper)
    support *= np.clip(face_support, 0.0, 1.0)
    dx *= support
    dy *= support

    grid_x, grid_y = np.meshgrid(
        np.arange(work_w, dtype=np.float32), np.arange(work_h, dtype=np.float32),
    )
    return cv2.remap(
        target_roi,
        grid_x - dx,
        grid_y - dy,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def balanced_sampling_start(
    face_bbox,
    target_size,
    total_steps: int,
    source_target_balance: int = 0,
) -> tuple[int, int, str]:
    """Map relative face size and a narrow Source/Target offset to a start step."""
    steps = max(1, int(total_steps))
    try:
        x1, y1, x2, y2 = (float(value) for value in face_bbox[:4])
        target_w, target_h = (max(1.0, float(value)) for value in target_size[:2])
        relative_short_side = min(x2 - x1, y2 - y1) / min(target_w, target_h)
    except Exception:
        relative_short_side = 0.5
    if relative_short_side <= 0.22:
        base_at_20, size_class = 10, "small"
    elif relative_short_side <= 0.37:
        base_at_20, size_class = 11, "medium"
    else:
        base_at_20, size_class = 12, "large"
    balance = max(-2, min(2, int(source_target_balance)))
    base = int(round(base_at_20 * steps / 20.0))
    start = int(round((base_at_20 + balance) * steps / 20.0))
    return max(0, min(steps - 1, start)), max(0, min(steps - 1, base)), size_class


def largest_face(faces):
    """Return the face with the largest valid bbox, without changing detector order."""
    valid = []
    for face in faces or []:
        bbox = face.get("bbox") if isinstance(face, dict) else getattr(face, "bbox", None)
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox[:4])
        except Exception:
            continue
        if x2 > x1 and y2 > y1:
            valid.append(((x2 - x1) * (y2 - y1), face))
    if not valid:
        raise RuntimeError("InstantID Face Rebuild: no valid target face detected")
    return max(valid, key=lambda item: item[0])[1]


def select_target_face(faces, selection: str, image_width: int, image_height: int):
    """Select one valid detected face using the established CMK modes."""
    valid = []
    for order, face in enumerate(faces or []):
        bbox = face.get("bbox") if isinstance(face, dict) else getattr(face, "bbox", None)
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox[:4])
        except Exception:
            continue
        if x2 > x1 and y2 > y1:
            valid.append((order, face, x1, y1, x2, y2))
    if not valid:
        raise RuntimeError("InstantID Face Rebuild: no valid target face detected")

    mode = str(selection or "Largest")
    center_x = float(image_width) * 0.5
    center_y = float(image_height) * 0.5
    key_functions = {
        "Largest": lambda item: (-((item[4] - item[2]) * (item[5] - item[3])), item[0]),
        "Leftmost": lambda item: (((item[2] + item[4]) * 0.5), item[0]),
        "Rightmost": lambda item: (-((item[2] + item[4]) * 0.5), item[0]),
        "Topmost": lambda item: (((item[3] + item[5]) * 0.5), item[0]),
        "Bottommost": lambda item: (-((item[3] + item[5]) * 0.5), item[0]),
        "Center": lambda item: (
            (((item[2] + item[4]) * 0.5) - center_x) ** 2
            + (((item[3] + item[5]) * 0.5) - center_y) ** 2,
            item[0],
        ),
    }
    if mode not in key_functions:
        mode = "Largest"
    return min(valid, key=key_functions[mode])[1]


def build_geometry(
    bbox,
    image_width: int,
    image_height: int,
    *,
    head_area: float = 0.75,
    neck_area: float = 0.40,
    feather: int = 32,
    working_resolution: int = 1024,
) -> FaceRebuildGeometry:
    """Derive a deterministic, clipped head/neck ROI from an InsightFace bbox."""
    if image_width < 1 or image_height < 1:
        raise ValueError("InstantID Face Rebuild: invalid target dimensions")
    x1, y1, x2, y2 = (float(value) for value in np.asarray(bbox).reshape(-1)[:4])
    if x2 <= x1 or y2 <= y1:
        raise ValueError("InstantID Face Rebuild: invalid face bbox")

    head = max(0.0, min(2.0, float(head_area)))
    neck = max(0.0, min(2.0, float(neck_area)))
    face_w = max(1.0, x2 - x1)
    face_h = max(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5

    left = int(np.floor(cx - face_w * (0.5 + head * 0.55)))
    right = int(np.ceil(cx + face_w * (0.5 + head * 0.55)))
    top = int(np.floor(y1 - face_h * (0.35 + head * 0.75)))
    bottom = int(np.ceil(y2 + face_h * (0.20 + neck)))
    left = max(0, min(image_width - 1, left))
    top = max(0, min(image_height - 1, top))
    right = max(left + 1, min(image_width, right))
    bottom = max(top + 1, min(image_height, bottom))

    roi_w, roi_h = right - left, bottom - top
    requested = max(512, min(2048, int(working_resolution)))
    scale = requested / float(max(roi_w, roi_h))
    work_w = max(64, int(round((roi_w * scale) / 64.0)) * 64)
    work_h = max(64, int(round((roi_h * scale) / 64.0)) * 64)
    work_w = min(2048, work_w)
    work_h = min(2048, work_h)

    face_box = (
        max(0, min(image_width - 1, int(np.floor(x1)))),
        max(0, min(image_height - 1, int(np.floor(y1)))),
        max(1, min(image_width, int(np.ceil(x2)))),
        max(1, min(image_height, int(np.ceil(y2)))),
    )
    return FaceRebuildGeometry(
        face_bbox=face_box,
        roi=(left, top, right, bottom),
        working_size=(work_w, work_h),
        feather=max(0, min(64, int(feather))),
    )


def _distance_feather(binary: np.ndarray, feather: int) -> np.ndarray:
    """Feather inward while keeping every pixel outside the geometry at zero."""
    binary = np.asarray(binary, dtype=bool)
    if feather <= 0:
        return binary.astype(np.float32)
    try:
        import cv2

        distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
        return np.clip(distance / float(max(1, feather)), 0.0, 1.0).astype(np.float32)
    except Exception:
        # Fallback retains strict binary support. Repeated neighbourhood
        # erosion provides a small inward distance approximation.
        current = binary.copy()
        out = np.zeros(binary.shape, dtype=np.float32)
        steps = max(1, min(int(feather), 64))
        for index in range(steps):
            out[current] = max(float(index + 1) / float(steps), 0.0)
            padded = np.pad(out, 1, mode="edge")
            neighbours = np.pad(current, 1, mode="constant", constant_values=False)
            current = (
                neighbours[:-2, 1:-1]
                & neighbours[2:, 1:-1]
                & neighbours[1:-1, :-2]
                & neighbours[1:-1, 2:]
                & current
            )
            if not np.any(current):
                break
        out[~binary] = 0.0
        return np.clip(out, 0.0, 1.0)


def _geometric_mask(
    geometry: FaceRebuildGeometry,
    *,
    pasteback: bool,
    include_neck: bool = True,
) -> np.ndarray:
    left, top, right, bottom = geometry.roi
    fx1, fy1, fx2, fy2 = geometry.face_bbox
    height, width = bottom - top, right - left
    yy, xx = np.ogrid[:height, :width]
    face_w = max(1.0, fx2 - fx1)
    face_h = max(1.0, fy2 - fy1)
    cx = ((fx1 + fx2) * 0.5) - left
    cy = ((fy1 + fy2) * 0.5) - top

    # A guaranteed zero border prevents a clipped neck ellipse or resized ROI
    # edge from becoming a visible rectangular paste boundary.
    border = max(3, min(int(round(min(width, height) * 0.08)), geometry.feather + 8))
    head_cy = cy - face_h * 0.18
    head_rx_requested = face_w * (0.72 if pasteback else 0.94)
    head_ry_requested = face_h * (1.16 if pasteback else 1.30)
    head_rx = max(1.0, min(head_rx_requested, cx - border, width - border - cx))
    # Limit the upper and lower halves independently. A single symmetric
    # radius made an ROI clipped at the top shrink at both ends, so its final
    # pasteback mask could stop above the mouth and chin even though there was
    # ample room below the face.
    head_ry_top = max(1.0, min(head_ry_requested, head_cy - border))
    if pasteback:
        # Keep the final composite around the actual face. The detector bbox
        # already reaches the chin; a small margin protects its feathered
        # contour without allowing the head mask to consume the upper neck.
        desired_bottom = (fy2 - top) + face_h * 0.08
        head_ry_bottom_requested = max(1.0, desired_bottom - head_cy)
    else:
        head_ry_bottom_requested = head_ry_requested
    head_ry_bottom = max(
        1.0,
        min(head_ry_bottom_requested, height - border - head_cy),
    )
    head_dy = yy - head_cy
    head_ry = np.where(head_dy < 0.0, head_ry_top, head_ry_bottom)
    head = ((xx - cx) / head_rx) ** 2 + (head_dy / head_ry) ** 2 <= 1.0

    binary = head
    if include_neck:
        neck_rx_requested = face_w * (0.34 if pasteback else 0.46)
        neck_ry_requested = face_h * (0.42 if pasteback else 0.58)
        neck_cy = (fy2 - top) + neck_ry_requested * 0.20
        neck_rx = max(1.0, min(neck_rx_requested, cx - border, width - border - cx))
        neck_ry = max(1.0, min(neck_ry_requested, neck_cy - border, height - border - neck_cy))
        neck = ((xx - cx) / neck_rx) ** 2 + ((yy - neck_cy) / neck_ry) ** 2 <= 1.0
        binary = np.logical_or(head, neck)
    binary[:border, :] = False
    binary[-border:, :] = False
    binary[:, :border] = False
    binary[:, -border:] = False
    return _distance_feather(binary, geometry.feather)


def make_inpaint_mask(geometry: FaceRebuildGeometry) -> np.ndarray:
    """Broad reconstruction mask with strict zero support outside its geometry."""
    return _geometric_mask(geometry, pasteback=False, include_neck=True)


def make_pasteback_mask(
    geometry: FaceRebuildGeometry,
    *,
    include_neck: bool = False,
) -> np.ndarray:
    """Tighter final composite mask, inset from both Inpaint and ROI borders."""
    return _geometric_mask(geometry, pasteback=True, include_neck=include_neck)


def make_roi_mask(geometry: FaceRebuildGeometry) -> np.ndarray:
    """Compatibility alias for the final pasteback mask."""
    return make_pasteback_mask(geometry)


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = (int(size[0]), int(size[1]))
    try:
        import cv2

        interpolation = cv2.INTER_LANCZOS4 if width > image.shape[1] else cv2.INTER_AREA
        return cv2.resize(image, (width, height), interpolation=interpolation)
    except Exception:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = (int(size[0]), int(size[1]))
    try:
        import cv2

        return np.clip(cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    except Exception:
        from PIL import Image

        image = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8))
        return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def pasteback_exact(target: np.ndarray, rebuilt_roi: np.ndarray, mask: np.ndarray, roi):
    """Composite into a copy and prove that every pixel outside alpha support is unchanged."""
    left, top, right, bottom = (int(value) for value in roi)
    roi_h, roi_w = bottom - top, right - left
    rebuilt = resize_rgb(rebuilt_roi, (roi_w, roi_h))
    if mask.shape != (roi_h, roi_w):
        mask = resize_mask(mask, (roi_w, roi_h))
    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)
    output = target.copy()
    original_crop = target[top:bottom, left:right]
    blended = np.rint(
        original_crop.astype(np.float32) * (1.0 - alpha[..., None])
        + rebuilt.astype(np.float32) * alpha[..., None]
    ).clip(0, 255).astype(np.uint8)
    support = alpha > 0.0
    output_crop = output[top:bottom, left:right]
    output_crop[support] = blended[support]

    full_support = np.zeros(target.shape[:2], dtype=bool)
    full_support[top:bottom, left:right] = support
    outside_unchanged = bool(np.array_equal(output[~full_support], target[~full_support]))
    if not outside_unchanged:
        raise AssertionError("InstantID Face Rebuild integrity failure: pixels changed outside pasteback mask")
    return output, outside_unchanged
