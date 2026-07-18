from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .detector_engine import CMKDetectorEngine, DetectorSettings
from .swap_selected_engine import CMKSelectedSwapEngine, SelectedSwapSettings


@dataclass
class TrackState:
    bbox: np.ndarray
    embedding: np.ndarray | None
    missing_frames: int = 0


def _bbox(face: dict) -> np.ndarray:
    return np.asarray(face.get("bbox"), dtype=np.float32).reshape(-1)[:4]


def _embedding(face: dict) -> np.ndarray | None:
    raw = face.get("raw")
    value = getattr(raw, "normed_embedding", None) if raw is not None else None
    if value is None:
        value = getattr(raw, "embedding", None) if raw is not None else None
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-8 else None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, float(a[2]-a[0])) * max(0.0, float(a[3]-a[1]))
    bb = max(0.0, float(b[2]-b[0])) * max(0.0, float(b[3]-b[1]))
    return inter / max(1e-6, aa + bb - inter)


def _center_distance(a: np.ndarray, b: np.ndarray, width: int, height: int) -> float:
    ac = np.array([(a[0]+a[2])*0.5, (a[1]+a[3])*0.5], dtype=np.float32)
    bc = np.array([(b[0]+b[2])*0.5, (b[1]+b[3])*0.5], dtype=np.float32)
    return float(np.linalg.norm(ac-bc) / max(1.0, math.hypot(width, height)))


def _select_initial(faces: list[dict], mode: str, width: int, height: int) -> dict:
    if not faces:
        raise ValueError("No face candidates")
    def key(face):
        b = _bbox(face); cx=(b[0]+b[2])*0.5; cy=(b[1]+b[3])*0.5; area=max(0,b[2]-b[0])*max(0,b[3]-b[1])
        if mode == "Leftmost": return cx
        if mode == "Rightmost": return -cx
        if mode == "Topmost": return cy
        if mode == "Bottommost": return -cy
        if mode == "Center": return (cx-width*0.5)**2 + (cy-height*0.5)**2
        return -area
    return min(faces, key=key)


class CMKVideoSwapEngine:
    def __init__(self, detector_model: str, detector_size: int = 640):
        self.detector = CMKDetectorEngine()
        self.detector_settings = DetectorSettings(detector_model=detector_model, detector_size=detector_size)
        self.swapper = CMKSelectedSwapEngine()

    def detect_filtered(self, image_rgb: np.ndarray, drop_size: int) -> list[dict]:
        faces = self.detector.detect_image(image_rgb, self.detector_settings)
        out=[]
        for face in faces:
            b=_bbox(face)
            if max(float(b[2]-b[0]), float(b[3]-b[1])) >= max(1, int(drop_size)):
                out.append(face)
        return out

    def select_source(self, image_rgb: np.ndarray, selection: str, drop_size: int) -> dict:
        faces=self.detect_filtered(image_rgb, drop_size)
        if not faces:
            raise RuntimeError("CMK FaceSwap Video: no source face detected")
        return _select_initial(faces, selection, image_rgb.shape[1], image_rgb.shape[0])

    def select_target(self, image_rgb: np.ndarray, faces: list[dict], selection: str, state: TrackState | None,
                      max_missing_frames: int, tracking_iou_threshold: float, tracking_embedding_threshold: float):
        if not faces:
            if state is not None: state.missing_frames += 1
            return None, state
        if state is None or state.missing_frames > max_missing_frames:
            chosen=_select_initial(faces, selection, image_rgb.shape[1], image_rgb.shape[0])
            return chosen, TrackState(_bbox(chosen), _embedding(chosen), 0)
        best=None; best_score=-1e9
        for face in faces:
            b=_bbox(face); iou=_iou(state.bbox,b); dist=_center_distance(state.bbox,b,image_rgb.shape[1],image_rgb.shape[0])
            emb=_embedding(face); sim=-1.0
            if state.embedding is not None and emb is not None: sim=float(np.dot(state.embedding, emb))
            plausible = iou >= tracking_iou_threshold or dist <= 0.18 or sim >= tracking_embedding_threshold
            if not plausible: continue
            score = iou*2.0 + max(0.0, 1.0-dist*4.0) + (max(0.0, sim)*2.0 if sim >= 0 else 0.0)
            if score > best_score: best_score=score; best=face
        if best is None:
            state.missing_frames += 1
            return None, state
        emb=_embedding(best)
        if state.embedding is not None and emb is not None:
            emb = state.embedding*0.8 + emb*0.2; emb /= max(1e-8, float(np.linalg.norm(emb)))
        return best, TrackState(_bbox(best), emb if emb is not None else state.embedding, 0)

    def swap_frame(self, image_rgb: np.ndarray, source_face: dict, target_face: dict, settings: SelectedSwapSettings) -> np.ndarray:
        return self.swapper.swap_faces(image_rgb, source_face["raw"], target_face["raw"], settings)
