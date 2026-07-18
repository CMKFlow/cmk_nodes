from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def draw_face_boxes(
    image_rgb: np.ndarray,
    faces: List[Dict[str, Any]],
    thickness: int = 2,
    draw_boxes: bool = True,
    draw_landmarks: bool = True,
) -> np.ndarray:
    """Draw simple white boxes and landmark crosses without adding heavy dependencies."""
    out = image_rgb.copy()
    h, w = out.shape[:2]
    thickness = max(1, int(thickness))

    for face in faces:
        bbox = face.get("bbox")
        if bbox is None:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        if draw_boxes:
            out[y1 : min(h, y1 + thickness), x1:x2 + 1] = 255
            out[max(0, y2 - thickness + 1) : y2 + 1, x1:x2 + 1] = 255
            out[y1:y2 + 1, x1 : min(w, x1 + thickness)] = 255
            out[y1:y2 + 1, max(0, x2 - thickness + 1) : x2 + 1] = 255

        kps = face.get("kps")
        if draw_landmarks and kps is not None:
            for x, y in np.asarray(kps)[:, :2]:
                cx = max(0, min(w - 1, int(round(float(x)))))
                cy = max(0, min(h - 1, int(round(float(y)))))
                radius = max(2, thickness)
                half = max(1, thickness // 2)

                # Landmark crosses use the same visual weight as bounding boxes.
                # The old implementation only made the cross longer; the stroke
                # stayed 1px and was almost invisible on high-resolution previews.
                out[
                    max(0, cy - radius) : min(h, cy + radius + 1),
                    max(0, cx - half) : min(w, cx + half + 1),
                ] = 255
                out[
                    max(0, cy - half) : min(h, cy + half + 1),
                    max(0, cx - radius) : min(w, cx + radius + 1),
                ] = 255

    return out
