from __future__ import annotations

import numpy as np
import torch

from ...engine.detector_engine import CMKDetectorEngine, DetectorSettings
from ...engine.swap_selected_engine import CMKSelectedSwapEngine, SelectedSwapSettings
from ...engine.enhance_backends import (
    get_available_enhancer_modes,
    get_default_enhancer_mode,
    validate_enhancer_mode,
)
from ...models.model_manager import list_detector_models, list_swap_models
from ...utils.face_set_utils import (
    normalize_selected_face,
    select_face,
    summarize_selected_face,
    summarize_selected_face_details,
)
from ...utils.preview_utils import draw_face_boxes
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...pipe.cmk_log_pipe import cmk_add_block, cmk_block_to_string
from ...utils.tensor_utils import tensor_to_uint8_rgb, uint8_rgb_to_tensor
from ...utils.stable_segs import CMKStableSEGS, image_signature


_SELECTION_MODES = ["Largest", "Leftmost", "Rightmost", "Topmost", "Bottommost", "Center"]


def _clamp_float(value, minimum: float, maximum: float, fallback: float) -> float:
    try:
        v = float(value)
    except Exception:
        return float(fallback)
    if not np.isfinite(v):
        return float(fallback)
    return max(float(minimum), min(float(maximum), v))


def _difference_image(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).astype(np.uint8)
    return np.clip(diff.astype(np.float32) * 3.0, 0, 255).astype(np.uint8)


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == height:
        return image
    out = np.zeros((height, w, 3), dtype=np.uint8)
    out[:h, :w] = image
    return out


def _side_by_side(*images: np.ndarray) -> np.ndarray:
    if not images:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    height = max(int(img.shape[0]) for img in images)
    parts = []
    for idx, img in enumerate(images):
        if idx:
            parts.append(np.zeros((height, 8, 3), dtype=np.uint8))
        parts.append(_pad_to_height(img, height))
    return np.concatenate(parts, axis=1)


def _face_set_from_detection(*, image_rgb: np.ndarray, faces: list[dict], detector_model: str, detector_size: int) -> dict:
    return {
        "type": "CMK_FACE_SET",
        "detector_model": str(detector_model),
        "detector_size": int(detector_size),
        "total_faces": int(len(faces)),
        "batch": [{"image_index": 0, "width": int(image_rgb.shape[1]), "height": int(image_rgb.shape[0]), "faces": faces}],
    }


def _select_detected_face(*, image_rgb: np.ndarray, faces: list[dict], detector_model: str, detector_size: int, selection: str, role: str) -> dict:
    face_set = _face_set_from_detection(image_rgb=image_rgb, faces=faces, detector_model=detector_model, detector_size=detector_size)
    selected = select_face(face_set, 0, str(selection), 0)
    face = selected["face"]
    return {
        "type": "CMK_SELECTED_FACE",
        "source_type": "CMK_FACE_SET",
        "role": str(role),
        "detector_model": str(detector_model),
        "detector_size": int(detector_size),
        "image_index": 0,
        "image_width": selected.get("image_width"),
        "image_height": selected.get("image_height"),
        "selection": str(selection),
        "selected_index": int(face.get("index", selected.get("selected_index", 0))),
        "face": face,
    }




def _face_extent(face: dict) -> tuple[float, float]:
    bbox = face.get("bbox")
    try:
        arr = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
        width = max(0.0, float(arr[2] - arr[0]))
        height = max(0.0, float(arr[3] - arr[1]))
        return width, height
    except Exception:
        return 0.0, 0.0


def _filter_faces(faces: list[dict], *, drop_size: int) -> list[dict]:
    filtered = []
    minimum = max(1, int(drop_size))
    for face in faces or []:
        width, height = _face_extent(face)
        if max(width, height) < float(minimum):
            continue
        filtered.append(face)
    return filtered


def _detect_faces_filtered(
    *,
    image_rgb: np.ndarray,
    detector: CMKDetectorEngine,
    detector_settings: DetectorSettings,
    drop_size: int,
) -> list[dict]:
    faces = detector.detect_image(image_rgb, detector_settings)
    return _filter_faces(faces, drop_size=drop_size)


def _resolve_selected_face(
    *,
    explicit_face,
    image_rgb: np.ndarray,
    detector: CMKDetectorEngine,
    detector_settings: DetectorSettings,
    detector_model: str,
    detector_size: int,
    selection: str,
    role: str,
    drop_size: int = 10,
) -> tuple[dict, str, int]:
    if explicit_face is not None:
        payload = normalize_selected_face(explicit_face)
        return payload, "Input", -1
    faces = _detect_faces_filtered(image_rgb=image_rgb, detector=detector, detector_settings=detector_settings, drop_size=drop_size)
    payload = _select_detected_face(
        image_rgb=image_rgb,
        faces=faces,
        detector_model=detector_model,
        detector_size=detector_size,
        selection=selection,
        role=role,
    )
    return payload, "Internal", len(faces)


def _mark_face(image: np.ndarray, selected_face: dict) -> np.ndarray:
    face = selected_face.get("face", {})
    return draw_face_boxes(image, [face], thickness=8, draw_boxes=True, draw_landmarks=True)


def _blend(target: np.ndarray, swapped: np.ndarray, blend: float) -> np.ndarray:
    blend = _clamp_float(blend, 0.0, 1.0, 1.0)
    if blend >= 0.999:
        return swapped
    out = target.astype(np.float32) * (1.0 - blend) + swapped.astype(np.float32) * blend
    return np.clip(out, 0, 255).astype(np.uint8)


class CMKFaceSwapImage:
    """CMK FaceSwap Image - comfort node with automatic face detection and optional selected-face overrides."""

    CATEGORY = "CMK/Toolbox/Face"
    RETURN_TYPES = ("IMAGE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("image", "diagnostic")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "swap_model": (list_swap_models(),),
                "detector_model": (list_detector_models(),),
                "face_enhancer": (get_available_enhancer_modes(), {"default": get_default_enhancer_mode()}),
                "target_selection": (_SELECTION_MODES, {"default": "Largest"}),
                "source_selection": (_SELECTION_MODES, {"default": "Largest"}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "bbox_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1, "advanced": True}),
                "crop_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "advanced": True}),
                "drop_size": ("INT", {"default": 10, "min": 1, "max": 8192, "step": 1, "advanced": True}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "advanced": True}),
            },
            "optional": {
                "opt_selected_target_face": ("CMK_SELECTED_FACE", {"forceInput": True}),
                "opt_selected_source_face": ("CMK_SELECTED_FACE", {"forceInput": True}),
            },
        }

    def run(
        self,
        target_image: torch.Tensor,
        source_image: torch.Tensor,
        enabled: bool,
        swap_model: str,
        detector_model: str,
        face_enhancer: str,
        target_selection: str,
        source_selection: str,
        blend: float,
        bbox_dilation: int = 0,
        crop_factor: float = 1.5,
        drop_size: int = 10,
        feather: int = 0,
        opt_selected_target_face=None,
        opt_selected_source_face=None,
    ):
        detector_size = 640
        blend = _clamp_float(blend, 0.0, 1.0, 1.0)
        target_selection = str(target_selection) if str(target_selection) in _SELECTION_MODES else "Largest"
        source_selection = str(source_selection) if str(source_selection) in _SELECTION_MODES else "Largest"
        requested_face_enhancer = str(face_enhancer or get_default_enhancer_mode())
        face_enhancer = requested_face_enhancer
        if bool(enabled):
            face_enhancer = validate_enhancer_mode(face_enhancer)
        elif face_enhancer not in get_available_enhancer_modes():
            face_enhancer = "Off"
        bbox_dilation = int(bbox_dilation)
        crop_factor = min(3.0, max(1.0, float(crop_factor)))
        drop_size = max(1, int(drop_size))
        feather = min(100, max(0, int(feather)))

        detector = CMKDetectorEngine()
        detector_settings = DetectorSettings(detector_model=str(detector_model), detector_size=int(detector_size))
        engine = CMKSelectedSwapEngine()
        settings = SelectedSwapSettings(
            swap_model=str(swap_model),
            enhancer_mode=face_enhancer,
            bbox_dilation=bbox_dilation,
            crop_factor=crop_factor,
            feather=feather,
        )

        outputs = []
        preview_images = []
        diagnostic_stages = []
        changed_values = []
        target_summary_payload = None
        source_summary_payload = None
        target_source_kind = "Internal"
        source_source_kind = "Internal"
        target_detect_count = 0
        source_detect_count = 0

        source_count = int(source_image.shape[0])
        for i in range(int(target_image.shape[0])):
            target_rgb = tensor_to_uint8_rgb(target_image[i])
            source_rgb = tensor_to_uint8_rgb(source_image[min(i, source_count - 1)])

            target_payload, target_source_kind, target_count = _resolve_selected_face(
                explicit_face=opt_selected_target_face,
                image_rgb=target_rgb,
                detector=detector,
                detector_settings=detector_settings,
                detector_model=str(detector_model),
                detector_size=int(detector_size),
                selection=target_selection,
                role="target",
                drop_size=drop_size,
            )
            source_payload, source_source_kind, source_count_detected = _resolve_selected_face(
                explicit_face=opt_selected_source_face,
                image_rgb=source_rgb,
                detector=detector,
                detector_settings=detector_settings,
                detector_model=str(detector_model),
                detector_size=int(detector_size),
                selection=source_selection,
                role="source",
                drop_size=drop_size,
            )
            if target_count >= 0:
                target_detect_count = target_count
            if source_count_detected >= 0:
                source_detect_count = source_count_detected

            if bool(enabled):
                raw_result = engine.swap_selected(
                    target_rgb=target_rgb,
                    source_rgb=source_rgb,
                    source_selected_face=source_payload,
                    target_selected_face=target_payload,
                    settings=settings,
                )
                result_rgb = _blend(target_rgb, raw_result, blend)
            else:
                result_rgb = target_rgb.copy()

            diff_rgb = _difference_image(target_rgb, result_rgb)
            changed = float(np.mean(np.abs(result_rgb.astype(np.float32) - target_rgb.astype(np.float32))) / 255.0)
            changed_values.append(changed)

            target_marked = _mark_face(target_rgb, target_payload)
            source_marked = _mark_face(source_rgb, source_payload)
            result_marked = _mark_face(result_rgb, target_payload)
            preview_images.append(_side_by_side(target_marked, source_marked, result_marked, diff_rgb))

            if i == 0:
                diagnostic_stages = [
                    {
                        "title": "01 Target",
                        "subtitle": f"{target_source_kind.lower()} face / {target_selection.lower()}",
                        "image": target_rgb,
                    },
                    {
                        "title": "02 Source",
                        "subtitle": f"{source_source_kind.lower()} face / {source_selection.lower()}",
                        "image": source_rgb,
                    },
                    {
                        "title": "03 Target Detection",
                        "subtitle": f"faces detected: {int(target_detect_count)}",
                        "image": target_marked,
                    },
                    {
                        "title": "04 Source Detection",
                        "subtitle": f"faces detected: {int(source_detect_count)}",
                        "image": source_marked,
                    },
                    {
                        "title": "05 Swap Result",
                        "subtitle": f"enhancer: {face_enhancer}",
                        "image": result_marked,
                    },
                    {
                        "title": "06 Difference",
                        "subtitle": f"changed: {changed:.4f}",
                        "image": diff_rgb,
                    },
                    {
                        "title": "07 Final",
                        "subtitle": "final output",
                        "image": result_rgb,
                    },
                ]

            outputs.append(uint8_rgb_to_tensor(result_rgb))

            if target_summary_payload is None:
                target_summary_payload = target_payload
            if source_summary_payload is None:
                source_summary_payload = source_payload

        changed_avg = float(np.mean(changed_values)) if changed_values else 0.0
        summary = "\n".join([
            f"Status           : {'Enabled' if bool(enabled) else 'Disabled'}",
            f"Model            : {swap_model}",
            f"Detector         : {detector_model}",
            f"Enhancer         : {face_enhancer}",
            *( [f"Enhancer fallback: {requested_face_enhancer} → {face_enhancer}"] if requested_face_enhancer != face_enhancer else [] ),
            f"Target Source    : {target_source_kind}",
            f"Target Selection : {target_selection}",
            f"Source Source    : {source_source_kind}",
            f"Source Selection : {source_selection}",
            f"Blend            : {blend:.2f}",
            f"bbox_dilation    : {bbox_dilation}",
            f"crop_factor      : {crop_factor:g}",
            f"drop_size        : {drop_size}",
            f"feather          : {feather}",
            f"Changed          : {changed_avg:.4f}",
            "",
            "Target",
            summarize_selected_face(target_summary_payload) if target_summary_payload else "n/a",
            "",
            "Source",
            summarize_selected_face(source_summary_payload) if source_summary_payload else "n/a",
        ])
        details = "\n".join([
            summary,
            "",
            f"detector_size: {int(detector_size)}",
            f"target_detected_faces: {int(target_detect_count)}",
            f"source_detected_faces: {int(source_detect_count)}",
            "",
            "TARGET DETAILS",
            summarize_selected_face_details(target_summary_payload) if target_summary_payload else "n/a",
            "",
            "SOURCE DETAILS",
            summarize_selected_face_details(source_summary_payload) if source_summary_payload else "n/a",
        ])

        diagnostic = make_diagnostic_payload(
            title="FaceSwap Image",
            node="CMK FaceSwap Image",
            previews=preview_images,
            stages=diagnostic_stages,
            summary=summary,
            details=details,
            mode="Comparison",
            metadata={
                "enabled": bool(enabled),
                "swap_model": str(swap_model),
                "detector_model": str(detector_model),
                "face_enhancer": face_enhancer,
                "detector_size": int(detector_size),
                "target_source": target_source_kind,
                "source_source": source_source_kind,
                "target_selection": target_selection,
                "source_selection": source_selection,
                "blend": float(blend),
                "bbox_dilation": int(bbox_dilation),
                "crop_factor": float(crop_factor),
                "drop_size": int(drop_size),
                "feather": int(feather),
                "changed_avg": changed_avg,
            },
            metrics={"changed_avg": changed_avg},
        )

        return (torch.stack(outputs, dim=0), diagnostic)


class CMKFaceSwapImagePipe:
    """Flow-safe FaceSwap execution without pass-through inputs or outputs."""

    CATEGORY = "CMK/Developer/Pipe/Execute"
    RETURN_TYPES = ("IMAGE", "SEGS", "CMK_LOG_BLOCK", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("IMAGE PROCEED", "SEGS PROCESSED", "LOG BLOCK", "diagnostic")
    FUNCTION = "run_pipe"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE_TARGET": ("IMAGE",),
                "IMAGE_SOURCE": ("IMAGE", {"lazy": True}),
                "GLOBAL ENABLE": ("BOOLEAN", {"default": True}),
                "ENABLE": ("BOOLEAN", {"default": True}),
                "SWAP MODEL": (list_swap_models(),),
                "DETECT MODEL": (list_detector_models(),),
                "TARGET FACE": (_SELECTION_MODES, {"default": "Largest"}),
                "SOURCE FACE": (_SELECTION_MODES, {"default": "Largest"}),
                "FACE ENHANCER": (get_available_enhancer_modes(), {"default": get_default_enhancer_mode()}),
                "BLEND": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "bbox_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1, "advanced": True}),
                "crop_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "advanced": True}),
                "drop_size": ("INT", {"default": 10, "min": 1, "max": 8192, "step": 1, "advanced": True}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "advanced": True}),
                "IDENTITY STRENGTH": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.05, "advanced": True}),
            }
        }

    def check_lazy_status(self, IMAGE_SOURCE=None, ENABLE=True, **kwargs):
        global_enable = bool(kwargs.get("GLOBAL ENABLE", True))
        if global_enable and bool(ENABLE) and IMAGE_SOURCE is None:
            return ["IMAGE_SOURCE"]
        return []

    def run_pipe(self, **inputs):
        target_image = inputs.get("IMAGE_TARGET")
        if target_image is None:
            raise ValueError("CMK FaceSwap Image -Pipe-: IMAGE_TARGET is required")

        global_enable = bool(inputs.get("GLOBAL ENABLE", True))
        local_enable = bool(inputs.get("ENABLE", True))
        enabled = global_enable and local_enable
        swap_model = str(inputs.get("SWAP MODEL", "") or "")
        detector_model = str(inputs.get("DETECT MODEL", "") or "")
        target_selection = str(inputs.get("TARGET FACE", "Largest") or "Largest")
        source_selection = str(inputs.get("SOURCE FACE", "Largest") or "Largest")
        if target_selection not in _SELECTION_MODES:
            target_selection = "Largest"
        if source_selection not in _SELECTION_MODES:
            source_selection = "Largest"
        requested_face_enhancer = str(
            inputs.get("FACE ENHANCER", get_default_enhancer_mode())
            or get_default_enhancer_mode()
        )
        face_enhancer = requested_face_enhancer
        if enabled:
            face_enhancer = validate_enhancer_mode(face_enhancer)
        elif face_enhancer not in get_available_enhancer_modes():
            face_enhancer = "Off"
        blend = _clamp_float(inputs.get("BLEND", 1.0), 0.0, 1.0, 1.0)
        bbox_dilation = int(inputs.get("bbox_dilation", 0) or 0)
        crop_factor = min(3.0, max(1.0, float(inputs.get("crop_factor", 1.5) or 1.5)))
        drop_size = max(1, int(inputs.get("drop_size", 10) or 10))
        feather = min(100, max(0, int(inputs.get("feather", 0) or 0)))
        identity_strength = _clamp_float(inputs.get("IDENTITY STRENGTH", 1.0), 0.5, 1.5, 1.0)

        if not enabled:
            log_lines = [
                f"GLOBAL ENABLE   : {'ON' if global_enable else 'OFF'}",
                f"ENABLE          : {'ON' if local_enable else 'OFF'}",
                "STATUS          : DISABLED",
                "RESULT          : IMAGE TARGET PASSTHROUGH",
                "SOURCE LOAD     : SKIPPED",
                "FACE DETECTION  : SKIPPED",
            ]
            log_block = cmk_block_to_string("FaceSwap Image", 60, log_lines, True)
            summary = "\n".join(log_lines)
            diagnostic = make_diagnostic_payload(
                title="FaceSwap Image -Pipe-",
                node="CMK FaceSwap Image -Pipe-",
                previews=[target_image],
                stages=[
                    {
                        "title": "01 Passthrough",
                        "subtitle": "module disabled",
                        "image": target_image,
                    }
                ],
                summary=summary,
                details=summary,
                mode="Disabled / Passthrough",
                metadata={
                    "global_enabled": global_enable,
                    "local_enabled": local_enable,
                    "enabled": False,
                    "source_requested": False,
                    "face_detection_executed": False,
                },
                metrics={"changed_avg": 0.0},
            )
            _, height, width, _ = target_image.shape
            empty_segs = ((int(width), int(height)), [])
            return target_image, empty_segs, log_block, diagnostic

        source_image = inputs.get("IMAGE_SOURCE")
        if source_image is None:
            raise ValueError("CMK FaceSwap Image -Pipe-: IMAGE_SOURCE is required when ENABLE is ON")

        detector_size = 640
        detector = CMKDetectorEngine()
        detector_settings = DetectorSettings(detector_model=detector_model, detector_size=int(detector_size))
        engine = CMKSelectedSwapEngine()
        settings = SelectedSwapSettings(
            swap_model=swap_model,
            enhancer_mode=face_enhancer,
            bbox_dilation=bbox_dilation,
            crop_factor=crop_factor,
            feather=feather,
            identity_strength=identity_strength,
        )

        outputs = []
        branch_masks = []
        preview_images = []
        diagnostic_stages = []
        changed_values = []
        target_summary_payload = None
        source_summary_payload = None
        target_detect_count = 0
        source_detect_count = 0

        source_count = int(source_image.shape[0])
        for i in range(int(target_image.shape[0])):
            target_rgb = tensor_to_uint8_rgb(target_image[i])
            source_rgb = tensor_to_uint8_rgb(source_image[min(i, source_count - 1)])

            target_payload, _, target_count = _resolve_selected_face(
                explicit_face=None,
                image_rgb=target_rgb,
                detector=detector,
                detector_settings=detector_settings,
                detector_model=detector_model,
                detector_size=int(detector_size),
                selection=target_selection,
                role="target",
                drop_size=drop_size,
            )
            source_payload, _, source_count_detected = _resolve_selected_face(
                explicit_face=None,
                image_rgb=source_rgb,
                detector=detector,
                detector_settings=detector_settings,
                detector_model=detector_model,
                detector_size=int(detector_size),
                selection=source_selection,
                role="source",
                drop_size=drop_size,
            )
            target_detect_count = int(target_count)
            source_detect_count = int(source_count_detected)

            if enabled:
                raw_result, paste_mask = engine.swap_selected_with_mask(
                    target_rgb=target_rgb,
                    source_rgb=source_rgb,
                    source_selected_face=source_payload,
                    target_selected_face=target_payload,
                    settings=settings,
                )
                result_rgb = _blend(target_rgb, raw_result, blend)
                effective_mask = np.clip(paste_mask * float(blend), 0.0, 1.0)
            else:
                result_rgb = target_rgb.copy()

            diff_rgb = _difference_image(target_rgb, result_rgb)
            changed = float(np.mean(np.abs(result_rgb.astype(np.float32) - target_rgb.astype(np.float32))) / 255.0)
            changed_values.append(changed)

            target_marked = _mark_face(target_rgb, target_payload)
            source_marked = _mark_face(source_rgb, source_payload)
            result_marked = _mark_face(result_rgb, target_payload)
            preview_images.append(_side_by_side(target_marked, source_marked, result_marked, diff_rgb))

            if i == 0:
                diagnostic_stages = [
                    {
                        "title": "01 Target",
                        "subtitle": f"{target_selection.lower()} / faces: {int(target_detect_count)}",
                        "image": target_marked,
                    },
                    {
                        "title": "02 Source",
                        "subtitle": f"{source_selection.lower()} / faces: {int(source_detect_count)}",
                        "image": source_marked,
                    },
                    {
                        "title": "03 Swap Result",
                        "subtitle": f"enhancer: {face_enhancer}",
                        "image": result_marked,
                    },
                    {
                        "title": "04 Difference",
                        "subtitle": f"changed: {changed:.4f}",
                        "image": diff_rgb,
                    },
                    {
                        "title": "05 Final",
                        "subtitle": "final output",
                        "image": result_rgb,
                    },
                ]

            outputs.append(uint8_rgb_to_tensor(result_rgb))
            branch_masks.append(
                torch.from_numpy(effective_mask.astype(np.float32))[None, ..., None]
            )
            if target_summary_payload is None:
                target_summary_payload = target_payload
            if source_summary_payload is None:
                source_summary_payload = source_payload

        changed_avg = float(np.mean(changed_values)) if changed_values else 0.0
        log_lines = [
            f"GLOBAL ENABLE   : {'ON' if global_enable else 'OFF'}",
            f"ENABLE          : {'ON' if local_enable else 'OFF'}",
            f"STATUS          : {'ENABLED' if enabled else 'DISABLED'}",
            f"SWAP MODEL      : {swap_model}",
            f"DETECT MODEL    : {detector_model}",
            f"FACE ENHANCER   : {face_enhancer}",
            *( [f"ENHANCER FALLBACK: {requested_face_enhancer} → {face_enhancer}"] if requested_face_enhancer != face_enhancer else [] ),
            f"TARGET FACE     : {target_selection}",
            f"SOURCE FACE     : {source_selection}",
            f"BLEND           : {blend:.2f}",
            f"BBOX DILATION   : {bbox_dilation}",
            f"CROP FACTOR     : {crop_factor:g}",
            f"DROP SIZE       : {drop_size}",
            f"FEATHER         : {feather}",
            f"IDENTITY STRENGTH: {identity_strength:.2f}",
            f"TARGET DETECTED : {int(target_detect_count)}",
            f"SOURCE DETECTED : {int(source_detect_count)}",
            f"CHANGED         : {changed_avg:.4f}",
        ]
        log_block = cmk_block_to_string("FaceSwap Image", 60, log_lines, True)

        summary = "\n".join(
            log_lines
            + [
                "",
                "TARGET",
                summarize_selected_face(target_summary_payload) if target_summary_payload else "n/a",
                "",
                "SOURCE",
                summarize_selected_face(source_summary_payload) if source_summary_payload else "n/a",
            ]
        )
        details = "\n".join(
            [
                summary,
                "",
                "TARGET DETAILS",
                summarize_selected_face_details(target_summary_payload) if target_summary_payload else "n/a",
                "",
                "SOURCE DETAILS",
                summarize_selected_face_details(source_summary_payload) if source_summary_payload else "n/a",
            ]
        )

        diagnostic = make_diagnostic_payload(
            title="FaceSwap Image -Pipe-",
            node="CMK FaceSwap Image -Pipe-",
            previews=preview_images,
            stages=diagnostic_stages,
            summary=summary,
            details=details,
            mode="Target + Source → Swap",
            metadata={
                "global_enabled": global_enable,
                "local_enabled": local_enable,
                "enabled": enabled,
                "swap_model": swap_model,
                "detector_model": detector_model,
                "face_enhancer": face_enhancer,
                "target_selection": target_selection,
                "source_selection": source_selection,
                "blend": float(blend),
                "bbox_dilation": int(bbox_dilation),
                "crop_factor": float(crop_factor),
                "drop_size": int(drop_size),
                "feather": int(feather),
                "identity_strength": float(identity_strength),
                "target_detected_faces": int(target_detect_count),
                "source_detected_faces": int(source_detect_count),
                "changed_avg": changed_avg,
            },
            metrics={"changed_avg": changed_avg},
        )

        branch_image = torch.stack(outputs, dim=0)
        branch_mask = torch.cat(branch_masks, dim=0).to(dtype=torch.bool)
        _, height, width, _ = target_image.shape
        processed_segs = CMKStableSEGS(
            (int(width), int(height)),
            [],
            branch_image=branch_image.detach().to(device="cpu").contiguous(),
            branch_mask=branch_mask.detach().to(device="cpu").contiguous(),
            source_signature=image_signature(target_image),
        )
        return branch_image, processed_segs, log_block, diagnostic
