from __future__ import annotations

import torch

from ...engine.detector_engine import CMKDetectorEngine, DetectorSettings
from ...models.model_manager import list_detector_models
from ...utils.preview_utils import draw_face_boxes
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...utils.tensor_utils import tensor_to_uint8_rgb, uint8_rgb_to_tensor


class CMKDetectFaces:
    """CMK Detect Faces - native face detection foundation node."""

    CATEGORY = "CMK/Toolbox/Face"
    RETURN_TYPES = ("IMAGE", "CMK_FACE_SET", "INT", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("image", "face_set", "face_count", "diagnostic")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detector_model": (list_detector_models(),),
                "detector_size": ("INT", {"default": 640, "min": 128, "max": 1280, "step": 64}),
            }
        }

    def run(self, image: torch.Tensor, detector_model: str, detector_size: int):
        settings = DetectorSettings(detector_model=detector_model, detector_size=int(detector_size))

        engine = CMKDetectorEngine()
        preview_images = []
        batch_faces = []
        total_faces = 0

        for i in range(int(image.shape[0])):
            rgb = tensor_to_uint8_rgb(image[i])
            faces = engine.detect_image(rgb, settings)
            total_faces += len(faces)
            batch_faces.append({"image_index": i, "width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "faces": faces})

            preview_rgb = draw_face_boxes(
                rgb,
                faces,
                thickness=8,
                draw_boxes=True,
                draw_landmarks=True,
            )
            preview_images.append(uint8_rgb_to_tensor(preview_rgb))

        faces_payload = {
            "type": "CMK_FACE_SET",
            "detector_model": detector_model,
            "detector_size": int(detector_size),
            "batch": batch_faces,
            "total_faces": int(total_faces),
        }

        summary = "\n".join([
            f"Detector Model : {detector_model}",
            f"Detector Size  : {int(detector_size)}",
            f"Faces          : {int(total_faces)}",
            "Preview        : All",
        ])
        detection_preview = [t.detach().cpu().numpy() * 255.0 for t in preview_images]
        stages = []
        if detection_preview:
            stages.append({"title": "01 Detection", "subtitle": f"{int(total_faces)} face(s)", "image": detection_preview[0]})
        diagnostic = make_diagnostic_payload(
            title="Detect Faces",
            node="CMK Detect Faces",
            previews=detection_preview,
            stages=stages,
            summary=summary,
            details=summary,
            mode="All",
            metadata={
                "detector_model": detector_model,
                "detector_size": int(detector_size),
                "total_faces": int(total_faces),
            },
            metrics={"faces": int(total_faces)},
        )
        return (image, faces_payload, int(total_faces), diagnostic)
