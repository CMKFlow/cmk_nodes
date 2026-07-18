from __future__ import annotations

import numpy as np
import torch

from ...utils.face_set_utils import face_bbox, normalize_selected_face
from ...utils.preview_utils import draw_face_boxes
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...utils.tensor_utils import tensor_to_uint8_rgb, uint8_rgb_to_tensor


class CMKFaceCrop:
    """Crop the selected face from an image."""

    CATEGORY = "CMK/Toolbox/Face"
    RETURN_TYPES = ("IMAGE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("face_image", "diagnostic")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "selected_face": ("CMK_SELECTED_FACE",),
                "padding": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
                "square_crop": ("BOOLEAN", {"default": True}),
            }
        }

    def run(self, image: torch.Tensor, selected_face, padding: float, square_crop: bool):
        selected = normalize_selected_face(selected_face)
        image_index = int(selected.get("image_index", 0) or 0)

        if image_index < 0 or image_index >= int(image.shape[0]):
            raise RuntimeError(f"image_index {image_index} out of range. Image batch size: {int(image.shape[0])}")

        face = selected.get("face") or {}
        if not face:
            raise RuntimeError("CMK Face Crop requires a valid CMK_SELECTED_FACE with face data.")

        rgb = tensor_to_uint8_rgb(image[image_index])
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = face_bbox(face)

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if square_crop:
            side = max(bw, bh) * (1.0 + float(padding) * 2.0)
            crop_w = crop_h = side
        else:
            crop_w = bw * (1.0 + float(padding) * 2.0)
            crop_h = bh * (1.0 + float(padding) * 2.0)

        left = int(np.floor(cx - crop_w / 2.0))
        top = int(np.floor(cy - crop_h / 2.0))
        right = int(np.ceil(cx + crop_w / 2.0))
        bottom = int(np.ceil(cy + crop_h / 2.0))

        left = max(0, min(w - 1, left))
        top = max(0, min(h - 1, top))
        right = max(left + 1, min(w, right))
        bottom = max(top + 1, min(h, bottom))

        crop = rgb[top:bottom, left:right].copy()
        preview = draw_face_boxes(rgb, [face], thickness=8, draw_boxes=True, draw_landmarks=True)

        summary = "\n".join([
            f"Selection    : {selected.get('selection', '')}",
            f"Padding      : {float(padding):.2f}",
            f"Square Crop  : {bool(square_crop)}",
            f"Crop Size    : {crop.shape[1]} × {crop.shape[0]}",
        ])

        diagnostic = make_diagnostic_payload(
            title="Face Crop",
            node="CMK Face Crop",
            stages=[
                {"title": "01 Source", "subtitle": "selected face", "image": preview},
                {"title": "02 Cropped Image", "subtitle": f"{crop.shape[1]} × {crop.shape[0]}", "image": crop},
            ],
            previews=[preview, crop],
            summary=summary,
            details=summary,
            mode="Crop",
            metadata={
                "selection": str(selected.get("selection", "")),
                "selected_index": int(selected.get("selected_index", -1)),
                "image_index": image_index,
                "crop_width": int(crop.shape[1]),
                "crop_height": int(crop.shape[0]),
                "padding": float(padding),
                "square_crop": bool(square_crop),
            },
        )

        return (uint8_rgb_to_tensor(crop).unsqueeze(0), diagnostic)


NODE_CLASS_MAPPINGS = {
    "CMKFaceCrop": CMKFaceCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMKFaceCrop": "CMK Face Crop",
}
