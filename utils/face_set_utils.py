from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def normalize_face_set(face_set: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(face_set, dict):
        raise ValueError("face_set must be a valid face_set dictionary")
    if face_set.get("type") not in ("CMK_FACE_SET", "CMK_FACES"):
        raise ValueError("unsupported face_set payload")
    if "batch" not in face_set:
        raise ValueError("face_set is missing 'batch'")
    return face_set


def get_batch_entry(face_set: Dict[str, Any], image_index: int) -> Dict[str, Any]:
    fs = normalize_face_set(face_set)
    batch = fs.get("batch", [])
    if not batch:
        raise RuntimeError("face_set contains no batch entries")
    if image_index < 0 or image_index >= len(batch):
        raise RuntimeError(f"image_index {image_index} out of range. Batch size: {len(batch)}")
    return batch[image_index]


def get_face(face_set: Dict[str, Any], image_index: int, face_index: int) -> Dict[str, Any]:
    entry = get_batch_entry(face_set, image_index)
    faces = entry.get("faces", [])
    if not faces:
        raise RuntimeError(f"No faces found for image_index {image_index}")
    if face_index < 0 or face_index >= len(faces):
        raise RuntimeError(
            f"face_index {face_index} out of range for image_index {image_index}. Faces: {len(faces)}"
        )
    return faces[face_index]


def face_bbox(face: Dict[str, Any]) -> Tuple[float, float, float, float]:
    bbox = face.get("bbox")
    if bbox is None:
        raise RuntimeError("selected face has no bbox")
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        raise RuntimeError(f"invalid bbox: {bbox!r}")
    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])


def summarize_face_set(face_set: Dict[str, Any]) -> str:
    fs = normalize_face_set(face_set)
    lines: List[str] = []
    lines.append("object: face_set")
    lines.append(f"detector_model: {fs.get('detector_model', '')}")
    lines.append(f"detector_size: {fs.get('detector_size', '')}")
    lines.append(f"total_faces: {fs.get('total_faces', 0)}")
    for entry in fs.get("batch", []):
        idx = entry.get("image_index", "?")
        width = entry.get("width", "?")
        height = entry.get("height", "?")
        faces = entry.get("faces", [])
        lines.append(f"image[{idx}]: {width}x{height}, faces={len(faces)}")
        for face in faces:
            bbox = face.get("bbox")
            score = face.get("det_score")
            if bbox is not None:
                b = [round(float(v), 1) for v in np.asarray(bbox).reshape(-1)[:4]]
            else:
                b = None
            lines.append(f"  face[{face.get('index', '?')}]: bbox={b}, score={score}")
    return "\n".join(lines)


def _face_center(face: Dict[str, Any]) -> Tuple[float, float]:
    x1, y1, x2, y2 = face_bbox(face)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _face_area(face: Dict[str, Any]) -> float:
    x1, y1, x2, y2 = face_bbox(face)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_face(face_set: Dict[str, Any], image_index: int, selection: str, face_index: int = 0) -> Dict[str, Any]:
    entry = get_batch_entry(face_set, image_index)
    faces = entry.get("faces", [])
    if not faces:
        raise RuntimeError(f"No faces found for image_index {image_index}")

    selection = str(selection or "By Index")

    if selection == "By Index":
        face = get_face(face_set, image_index, int(face_index))
    elif selection == "Largest":
        face = max(faces, key=_face_area)
    elif selection == "Leftmost":
        face = min(faces, key=lambda f: _face_center(f)[0])
    elif selection == "Rightmost":
        face = max(faces, key=lambda f: _face_center(f)[0])
    elif selection == "Topmost":
        face = min(faces, key=lambda f: _face_center(f)[1])
    elif selection == "Bottommost":
        face = max(faces, key=lambda f: _face_center(f)[1])
    elif selection == "Center":
        width = float(entry.get("width", 0) or 0)
        height = float(entry.get("height", 0) or 0)
        cx = width / 2.0
        cy = height / 2.0
        face = min(faces, key=lambda f: (_face_center(f)[0] - cx) ** 2 + (_face_center(f)[1] - cy) ** 2)
    else:
        raise RuntimeError(f"Unsupported face selection mode: {selection!r}")

    return {
        "image_index": int(image_index),
        "image_width": int(entry.get("width", 0) or 0),
        "image_height": int(entry.get("height", 0) or 0),
        "selection": selection,
        "selected_index": int(face.get("index", 0)),
        "face": face,
    }


def normalize_selected_face(selected_face: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(selected_face, dict):
        raise ValueError("selected_face must be a valid selected_face dictionary")
    if selected_face.get("type") != "CMK_SELECTED_FACE":
        raise ValueError("unsupported selected_face payload")
    if "face" not in selected_face:
        raise ValueError("selected_face is missing 'face'")
    return selected_face


def summarize_selected_face(selected_face: Dict[str, Any]) -> str:
    sf = normalize_selected_face(selected_face)
    face = sf.get("face", {})
    x1, y1, x2, y2 = face_bbox(face)
    fw = max(0, int(round(x2 - x1)))
    fh = max(0, int(round(y2 - y1)))
    score = face.get("det_score")
    try:
        confidence = f"{float(score) * 100:.0f}%"
    except Exception:
        confidence = "n/a"
    selection = sf.get("selection", "")
    selected_index = sf.get("selected_index", "?")
    image_w = sf.get("image_width")
    image_h = sf.get("image_height")
    return "\n".join([
        "Selected Face",
        f"{selection} (#{selected_index})",
        f"Confidence: {confidence}",
        f"Face size: {fw} × {fh}px",
        f"Image size: {image_w} × {image_h}px",
    ])


def summarize_selected_face_details(selected_face: Dict[str, Any]) -> str:
    sf = normalize_selected_face(selected_face)
    face = sf.get("face", {})
    bbox = face.get("bbox")
    if bbox is not None:
        bbox_text = [round(float(v), 1) for v in np.asarray(bbox).reshape(-1)[:4]]
    else:
        bbox_text = None
    return "\n".join([
        "object: selected_face",
        f"image_index: {sf.get('image_index')}",
        f"image_size: {sf.get('image_width')}x{sf.get('image_height')}",
        f"selection: {sf.get('selection')}",
        f"selected_index: {sf.get('selected_index')}",
        f"bbox: {bbox_text}",
        f"score: {face.get('det_score')}",
    ])
