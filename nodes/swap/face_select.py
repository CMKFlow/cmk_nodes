from __future__ import annotations

import torch

from ...engine.detector_engine import CMKDetectorEngine, DetectorSettings
from ...models.model_manager import list_detector_models
from ...utils.face_set_utils import (
    select_face,
    summarize_selected_face,
    summarize_selected_face_details,
)
from ...utils.preview_utils import draw_face_boxes
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...utils.tensor_utils import tensor_to_uint8_rgb


_SELECTION_MODES = ["Largest", "Leftmost", "Rightmost", "Topmost", "Bottommost", "Center"]


class CMKFaceSelect:
    """Detect faces and select one face by a semantic selection rule."""

    CATEGORY = "CMK/Toolbox/Face"
    RETURN_TYPES = ("IMAGE", "CMK_SELECTED_FACE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("image", "selected_face", "diagnostic")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detector_model": (list_detector_models(),),
                "detector_size": ("INT", {"default": 640, "min": 128, "max": 1280, "step": 64}),
                "selection": (_SELECTION_MODES, {"default": "Largest"}),
            }
        }

    def run(self, image: torch.Tensor, detector_model: str, detector_size: int, selection: str):
        image_index = 0
        if image_index < 0 or image_index >= int(image.shape[0]):
            raise RuntimeError(f"image_index {image_index} out of range. Image batch size: {int(image.shape[0])}")

        selection = str(selection)
        if selection not in _SELECTION_MODES:
            selection = "Largest"

        rgb = tensor_to_uint8_rgb(image[image_index])

        detector_settings = DetectorSettings(detector_model=str(detector_model), detector_size=int(detector_size))
        detector = CMKDetectorEngine()
        faces = detector.detect_image(rgb, detector_settings)

        if not faces:
            preview = rgb
            summary = "\n".join([
                f"Detector Model : {detector_model}",
                f"Detector Size  : {int(detector_size)}",
                f"Selection      : {selection}",
                "Faces          : 0",
                "Status         : No Detection",
            ])
            diagnostic = make_diagnostic_payload(
                title="Face Select",
                node="CMK Face Select",
                stages=[{"title": "01 No Detection", "subtitle": selection, "image": preview}],
                previews=[preview],
                summary=summary,
                details=summary,
                mode=selection,
                metadata={
                    "selection": selection,
                    "detector_model": str(detector_model),
                    "detector_size": int(detector_size),
                    "total_faces": 0,
                    "status": "No Detection",
                },
                warnings=["no face detected"],
                metrics={"faces": 0},
            )
            empty_selected_face = {
                "type": "CMK_SELECTED_FACE",
                "source_type": "CMK_FACE_SET",
                "detector_model": str(detector_model),
                "detector_size": int(detector_size),
                "image_index": image_index,
                "image_width": int(rgb.shape[1]),
                "image_height": int(rgb.shape[0]),
                "selection": selection,
                "selected_index": -1,
                "face": {},
            }
            return (image, empty_selected_face, diagnostic)

        face_set = {
            "type": "CMK_FACE_SET",
            "detector_model": str(detector_model),
            "detector_size": int(detector_size),
            "total_faces": int(len(faces)),
            "batch": [
                {
                    "image_index": image_index,
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "faces": faces,
                }
            ],
        }

        selected = select_face(face_set, image_index, selection, 0)
        face = selected["face"]

        selected_payload = {
            "type": "CMK_SELECTED_FACE",
            "source_type": "CMK_FACE_SET",
            "detector_model": str(detector_model),
            "detector_size": int(detector_size),
            "image_index": image_index,
            "image_width": selected.get("image_width"),
            "image_height": selected.get("image_height"),
            "selection": selection,
            "selected_index": int(face.get("index", selected.get("selected_index", 0))),
            "face": face,
        }

        preview = draw_face_boxes(rgb, [face], thickness=8, draw_boxes=True, draw_landmarks=True)
        summary = "\n".join([
            f"Detector Model : {detector_model}",
            f"Detector Size  : {int(detector_size)}",
            f"Selection      : {selection}",
            f"Faces          : {int(len(faces))}",
            "",
            summarize_selected_face(selected_payload),
        ])
        details = "\n".join([
            summary,
            "",
            summarize_selected_face_details(selected_payload),
        ])

        diagnostic = make_diagnostic_payload(
            title="Face Select",
            node="CMK Face Select",
            stages=[{"title": "01 Selected Face", "subtitle": selection, "image": preview}],
            previews=[preview],
            summary=summary,
            details=details,
            mode=selection,
            metadata={
                "selection": selection,
                "selected_index": int(selected_payload["selected_index"]),
                "detector_model": str(detector_model),
                "detector_size": int(detector_size),
                "total_faces": int(len(faces)),
                "status": "Selected",
            },
            metrics={"faces": int(len(faces))},
        )

        return (image, selected_payload, diagnostic)


NODE_CLASS_MAPPINGS = {
    "CMKFaceSelect": CMKFaceSelect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMKFaceSelect": "CMK Face Select",
}


def resolve_legacy_face_selection(image: torch.Tensor, selection: str) -> dict:
    """Translate CMK's semantic face choice to ReActor's legacy controls.

    The public UI stays intention-oriented. ReActor's ``all/filter/largest``
    selection contract remains an internal implementation detail.
    """
    selection = str(selection or "Largest")
    if selection == "All Faces":
        return {
            "select_face_selection": "all",
            "select_sort_by": "area",
            "select_reverse_order": False,
            "select_take_start": 0,
            "select_take_count": 100,
        }
    if selection not in _SELECTION_MODES:
        selection = "Largest"

    direct = {
        "Largest": {
            "select_face_selection": "largest",
            "select_sort_by": "area",
            "select_reverse_order": False,
            "select_take_start": 0,
            "select_take_count": 1,
        },
        "Leftmost": {
            "select_face_selection": "filter",
            "select_sort_by": "x_position",
            "select_reverse_order": False,
            "select_take_start": 0,
            "select_take_count": 1,
        },
        "Rightmost": {
            "select_face_selection": "filter",
            "select_sort_by": "x_position",
            "select_reverse_order": True,
            "select_take_start": 0,
            "select_take_count": 1,
        },
        "Topmost": {
            "select_face_selection": "filter",
            "select_sort_by": "y_position",
            "select_reverse_order": False,
            "select_take_start": 0,
            "select_take_count": 1,
        },
        "Bottommost": {
            "select_face_selection": "filter",
            "select_sort_by": "y_position",
            "select_reverse_order": True,
            "select_take_start": 0,
            "select_take_count": 1,
        },
    }
    if selection in direct:
        return direct[selection]

    # ReActor has no semantic "Center" mode. CMK resolves the centered face
    # with the same detector used by CMK Face Select, then translates that face
    # to its left-to-right rank for ReActor's filter interface.
    fallback = dict(direct["Largest"])
    try:
        rgb = tensor_to_uint8_rgb(image[0])
        detector_model = "buffalo_l"
        detector_size = 640
        detector = CMKDetectorEngine()
        faces = detector.detect_image(
            rgb,
            DetectorSettings(detector_model=detector_model, detector_size=detector_size),
        )
        if not faces:
            return fallback

        face_set = {
            "type": "CMK_FACE_SET",
            "detector_model": detector_model,
            "detector_size": detector_size,
            "total_faces": len(faces),
            "batch": [{
                "image_index": 0,
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "faces": faces,
            }],
        }
        selected = select_face(face_set, 0, "Center", 0)
        chosen = selected.get("face", {})
        chosen_index = int(chosen.get("index", selected.get("selected_index", 0)))

        def center_x(face):
            bbox = face.get("bbox") or face.get("box") or [0, 0, 0, 0]
            return (float(bbox[0]) + float(bbox[2])) * 0.5

        ordered = sorted(faces, key=center_x)
        rank = next(
            (i for i, face in enumerate(ordered) if int(face.get("index", i)) == chosen_index),
            0,
        )
        return {
            "select_face_selection": "filter",
            "select_sort_by": "x_position",
            "select_reverse_order": False,
            "select_take_start": int(rank),
            "select_take_count": 1,
        }
    except Exception:
        return fallback
