from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from ...utils.face_set_utils import (
    normalize_face_set,
    select_face,
    summarize_selected_face,
)
from ...utils.preview_utils import draw_face_boxes
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...utils.tensor_utils import tensor_to_uint8_rgb, uint8_rgb_to_tensor


_SELECTION_MODES = ["By Index", "Largest", "Leftmost", "Rightmost", "Topmost", "Bottommost", "Center"]
_SOURCE_MODES = ["Best Match", "By Index", "Largest", "Leftmost", "Rightmost", "Topmost", "Bottommost", "Center"]


def _embedding(face: Dict[str, Any]) -> np.ndarray:
    raw = face.get("raw")
    value = None
    if raw is not None:
        value = getattr(raw, "normed_embedding", None)
        if value is None:
            value = getattr(raw, "embedding", None)
    if value is None:
        value = face.get("normed_embedding") or face.get("embedding")
    if value is None:
        raise RuntimeError(
            "Selected face has no embedding. Use CMK Detect Faces with an InsightFace model "
            "that provides recognition embeddings, for example buffalo_l."
        )
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise RuntimeError("Selected face embedding has zero length")
    return arr / norm


def _cosine_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ea = _embedding(a)
    eb = _embedding(b)
    if ea.shape != eb.shape:
        raise RuntimeError(f"Embedding size mismatch: {ea.shape} vs {eb.shape}")
    return float(np.dot(ea, eb))


def _selected_payload(face_set: Dict[str, Any], selected: Dict[str, Any], selection_name: str) -> Dict[str, Any]:
    face = selected["face"]
    return {
        "type": "CMK_SELECTED_FACE",
        "source_type": face_set.get("type", "CMK_FACE_SET"),
        "detector_model": face_set.get("detector_model", ""),
        "detector_size": face_set.get("detector_size", ""),
        "image_index": int(selected.get("image_index", 0)),
        "image_width": selected.get("image_width"),
        "image_height": selected.get("image_height"),
        "selection": str(selection_name),
        "selected_index": int(face.get("index", selected.get("selected_index", 0))),
        "face": face,
    }


def _best_source_for_target(
    source_face_set: Dict[str, Any],
    source_image_index: int,
    target_face: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    batch = source_face_set.get("batch", [])
    if source_image_index < 0 or source_image_index >= len(batch):
        raise RuntimeError(f"source_image_index {source_image_index} out of range. Batch size: {len(batch)}")
    faces = batch[source_image_index].get("faces", [])
    if not faces:
        raise RuntimeError(f"No source faces found for image_index {source_image_index}")

    best_face = None
    best_similarity = -999.0
    for face in faces:
        similarity = _cosine_similarity(target_face, face)
        if similarity > best_similarity:
            best_similarity = similarity
            best_face = face

    if best_face is None:
        raise RuntimeError("No matching source face found")

    entry = batch[source_image_index]
    return (
        {
            "image_index": int(source_image_index),
            "image_width": int(entry.get("width", 0) or 0),
            "image_height": int(entry.get("height", 0) or 0),
            "selection": "Best Match",
            "selected_index": int(best_face.get("index", 0)),
            "face": best_face,
        },
        float(best_similarity),
    )


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == height:
        return image
    out = np.zeros((height, w, 3), dtype=np.uint8)
    out[:h, :w] = image
    return out


def _side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = max(int(left.shape[0]), int(right.shape[0]))
    left_p = _pad_to_height(left, height)
    right_p = _pad_to_height(right, height)
    separator = np.zeros((height, 12, 3), dtype=np.uint8)
    return np.concatenate([left_p, separator, right_p], axis=1)


class CMKFaceMatch:
    """Select a target face and match/select the corresponding source face."""

    CATEGORY = "CMK/Toolbox/Face"
    RETURN_TYPES = ("CMK_MATCHED_FACE_PAIR", "CMK_SELECTED_FACE", "CMK_SELECTED_FACE", "FLOAT", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("matched_pair", "target_face", "source_face", "similarity", "diagnostic")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "target_face_set": ("CMK_FACE_SET",),
                "source_image": ("IMAGE",),
                "source_face_set": ("CMK_FACE_SET",),
                "target_selection": (_SELECTION_MODES, {"default": "Largest"}),
                "target_face_index": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "source_selection": (_SOURCE_MODES, {"default": "Best Match"}),
                "source_face_index": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }

    def run(
        self,
        target_image: torch.Tensor,
        target_face_set,
        source_image: torch.Tensor,
        source_face_set,
        target_selection: str,
        target_face_index: int,
        source_selection: str,
        source_face_index: int,
    ):
        target_fs = normalize_face_set(target_face_set)
        source_fs = normalize_face_set(source_face_set)
        target_image_index = 0
        source_image_index = 0

        if target_image_index < 0 or target_image_index >= int(target_image.shape[0]):
            raise RuntimeError(f"target_image_index {target_image_index} out of range. Batch size: {int(target_image.shape[0])}")
        if source_image_index < 0 or source_image_index >= int(source_image.shape[0]):
            raise RuntimeError(f"source_image_index {source_image_index} out of range. Batch size: {int(source_image.shape[0])}")

        target_rgb = tensor_to_uint8_rgb(target_image[target_image_index])
        source_rgb = tensor_to_uint8_rgb(source_image[source_image_index])

        target_selected = select_face(target_fs, target_image_index, str(target_selection), int(target_face_index))
        target_payload = _selected_payload(target_fs, target_selected, str(target_selection))
        target_face = target_payload["face"]

        if str(source_selection) == "Best Match":
            source_selected, similarity = _best_source_for_target(source_fs, source_image_index, target_face)
        else:
            source_selected = select_face(source_fs, source_image_index, str(source_selection), int(source_face_index))
            similarity = _cosine_similarity(target_face, source_selected["face"])

        source_payload = _selected_payload(source_fs, source_selected, str(source_selection))

        matched_pair = {
            "type": "CMK_MATCHED_FACE_PAIR",
            "target_face": target_payload,
            "source_face": source_payload,
            "similarity": float(similarity),
            "target_selection": str(target_selection),
            "source_selection": str(source_selection),
        }

        target_preview = draw_face_boxes(
            target_rgb,
            [target_payload["face"]],
            thickness=8,
            draw_boxes=True,
            draw_landmarks=True,
        )
        source_preview = draw_face_boxes(
            source_rgb,
            [source_payload["face"]],
            thickness=8,
            draw_boxes=True,
            draw_landmarks=True,
        )
        preview = _side_by_side(target_preview, source_preview)

        summary = "\n".join([
            "operation: Face Match",
            f"similarity: {float(similarity):.4f}",
            "",
            "TARGET",
            summarize_selected_face(target_payload),
            "",
            "SOURCE",
            summarize_selected_face(source_payload),
        ])

        diagnostic = make_diagnostic_payload(
            title="Face Match",
            node="CMK Face Match",
            previews=[preview],
            summary=summary,
            details=summary,
            mode=f"{target_selection} → {source_selection}",
            metadata={"similarity": float(similarity)},
            metrics={"similarity": float(similarity)},
        )
        return (
            matched_pair,
            target_payload,
            source_payload,
            float(similarity),
            diagnostic,
        )
