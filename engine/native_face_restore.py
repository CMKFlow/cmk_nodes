from __future__ import annotations

import numpy as np
import torch

from .detector_engine import get_face_analyser
from .enhance_backends import GFPGANEnhancerBackend, get_enhancer_backend
from .pasteback import pasteback_native
from ..utils.tensor_utils import tensor_to_uint8_rgb, uint8_rgb_to_tensor


def _sort_faces(faces, sort_by: str, reverse: bool):
    def box(face):
        values = np.asarray(getattr(face, "bbox", [0, 0, 0, 0]), dtype=np.float32)
        return values if values.size >= 4 else np.zeros(4, dtype=np.float32)

    def metric(face):
        x1, y1, x2, y2 = box(face)[:4]
        if sort_by == "x_position":
            return float((x1 + x2) * 0.5)
        if sort_by == "y_position":
            return float((y1 + y2) * 0.5)
        if sort_by == "detection_confidence":
            return float(getattr(face, "det_score", 0.0) or 0.0)
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    return sorted(list(faces), key=metric, reverse=bool(reverse))


def _select_faces(faces, selection, sort_by, reverse_order, take_start, take_count):
    ordered = _sort_faces(faces, str(sort_by), bool(reverse_order))
    if str(selection) == "all":
        return ordered
    if str(selection) == "largest":
        return _sort_faces(faces, "area", True)[:1]
    start = max(0, int(take_start or 0))
    count = max(1, int(take_count or 1))
    return ordered[start:start + count]


def _enhancer_mode(model_name: str) -> str:
    text = str(model_name or "").lower()
    return "CodeFormer" if "codeformer" in text else "GPEN"


def _enhancer(model_name: str):
    if "gfpgan" in str(model_name or "").lower():
        return GFPGANEnhancerBackend(str(model_name))
    return get_enhancer_backend(_enhancer_mode(str(model_name)))


class RestoreFaceAdvanced:
    """CMK-owned face-restoration adapter with the former FaceProcess contract."""

    def execute(
        self,
        image,
        model,
        visibility=1.0,
        codeformer_weight=0.5,
        facedetection="retinaface_resnet50",
        face_selection="all",
        sort_by="area",
        reverse_order=False,
        take_start=0,
        take_count=1,
    ):
        del codeformer_weight, facedetection
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("CMK native face restore requires a ComfyUI IMAGE tensor")
        analyser = get_face_analyser("buffalo_l", 640)
        enhancer = _enhancer(str(model))
        visibility = max(0.0, min(1.0, float(visibility)))
        outputs = []

        for item in image:
            original = tensor_to_uint8_rgb(item)
            result = original.copy()
            faces = analyser.get(original)
            selected = _select_faces(
                faces,
                face_selection,
                sort_by,
                reverse_order,
                take_start,
                take_count,
            )
            for face in selected:
                try:
                    from insightface.utils import face_align
                except Exception as exc:
                    raise ImportError(
                        "CMK native face restore requires the Python package 'insightface'."
                    ) from exc
                size = int(getattr(enhancer, "model_size", 512))
                aligned, matrix = face_align.norm_crop2(result, face.kps, image_size=size)
                enhanced = enhancer.enhance_aligned(aligned)
                restored = pasteback_native(
                    result,
                    enhanced,
                    face,
                    affine_matrix=matrix,
                    crop_factor=1.0,
                    feather=8,
                )
                if isinstance(restored, tuple):
                    restored = restored[0]
                result = np.clip(
                    result.astype(np.float32) * (1.0 - visibility)
                    + np.asarray(restored, dtype=np.float32) * visibility,
                    0,
                    255,
                ).astype(np.uint8)
            outputs.append(uint8_rgb_to_tensor(result))
        return (torch.cat(outputs, dim=0),)
