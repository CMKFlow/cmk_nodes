import torch
import numpy as np
import comfy.samplers
import folder_paths

from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...engine.detailer_limits import clamp_detailer_denoise
from ...engine.native_detailer import (
    core,
    SimpleDetectorForEach,
    SEGSDetailer,
    SEGSPaste,
    UltralyticsDetectorProvider,
)
from ...engine.native_face_restore import RestoreFaceAdvanced


class CMKAnyType(str):
    def __ne__(self, other):
        return False


CMK_ANY = CMKAnyType("*")


_PROCESS_MODES = ["Off", "Restore", "Detailer"]
_REFINE_MODES = ["Off", "Detail", "Sharpen", "Smooth"]
_MODE_ALIASES = {
    "off": "Off",
    "bypass": "Off",
    "disabled": "Off",
    "none": "Off",
    "refine": "Off",
    "restore": "Restore",
    "detailer": "Detailer",
}
_REFINE_ALIASES = {
    "off": "Off",
    "none": "Off",
    "disabled": "Off",
    "detail": "Detail",
    "refine": "Detail",
    "sharpen": "Sharpen",
    "smooth": "Smooth",
}


def _is_legacy_off_mode(value):
    return str(value or "").strip().lower() in ("", "off", "bypass", "disabled", "none")


def _normalize_process_mode(value):
    key = str(value or "Off").strip()
    if key in _PROCESS_MODES:
        return key
    return _MODE_ALIASES.get(key.lower(), "Off")


def _normalize_refine_mode(value):
    key = str(value or "Off").strip()
    if key in _REFINE_MODES:
        return key
    return _REFINE_ALIASES.get(key.lower(), "Off")


def _face_detector_rank(name):
    """Rank detector names for face workflows. Lower is better."""
    text = str(name or "").lower()
    rank = 100
    preferred_tokens = ("face", "person", "head", "portrait")
    if any(token in text for token in preferred_tokens):
        rank = 0
    if text.startswith("bbox/"):
        rank += 0
    elif text.startswith("segm/"):
        rank += 10
    deprioritized_tokens = ("breast", "hand", "feet", "foot", "pose", "clothes", "clothing")
    if any(token in text for token in deprioritized_tokens):
        rank += 50
    return (rank, text)


def _sort_face_detectors(models):
    return sorted(models, key=_face_detector_rank)


def _preferred_face_detector(models):
    models = [m for m in models if m not in (None, "none")]
    if not models:
        return None
    return _sort_face_detectors(models)[0]


def _is_likely_non_face_detector(name):
    text = str(name or "").lower()
    return any(token in text for token in ("breast", "hand", "feet", "foot", "pose", "clothes", "clothing"))


class CMK_FaceProcess:
    @classmethod
    def INPUT_TYPES(cls):
        bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]
        detect_models = _sort_face_detectors(bboxs + segms)
        if not detect_models:
            detect_models = ["none"]

        facerestore_models = ["none"] + folder_paths.get_filename_list("facerestore_models")

        # Only the source image is a hard requirement. The dynamic frontend rebuilds
        # the visible widget list per mode; therefore every mode-specific control must
        # be optional from ComfyUI's validation perspective. Otherwise hidden widgets
        # are interpreted as missing required inputs when the prompt is queued.
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {"lazy": True}),
                "face_pipe": (CMK_ANY, {"lazy": True}),
                 "boolean_faceprocess_enable": ("BOOLEAN", {"default": True}),
                "enable": ("BOOLEAN", {"default": True}),
                "process_mode": (["Off", "Restore", "Detailer"], {"default": "Restore"}),
                "refine_mode": (["Off", "Detail", "Sharpen", "Smooth"], {"default": "Off"}),

                "detect_model": (detect_models,),
                "detect_bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detect_bbox_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "detect_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "detect_drop_size": ("INT", {"default": 10, "min": 1, "max": 8192, "step": 1}),

                "restore_model": (facerestore_models,),
                "restore_facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "restore_visibility": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "restore_codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),

                "select_face_selection": (["all", "filter", "largest"], {"default": "all"}),
                "select_sort_by": (["area", "x_position", "y_position", "detection_confidence"], {"default": "area"}),
                "select_reverse_order": ("BOOLEAN", {"default": False}),
                "select_take_start": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "select_take_count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),

                "detail_guide_size": ("FLOAT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "detail_guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                "detail_max_size": ("FLOAT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                "detail_denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 0.5, "step": 0.01}),
                "detail_noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "detail_force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "detail_paste_feather": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "SEGS", "SEGS", "STRING", "BOOLEAN", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("image", "image_proceed", "segs_detected", "segs_processed", "log_pipe", "enabled", "diagnostic")
    FUNCTION = "run"
    CATEGORY = "CMK/Toolbox/Face"

    @classmethod
    def IS_CHANGED(
        cls,
        image=None,
        face_pipe=None,
        boolean_faceprocess_enable=True,
        enable=True,
        process_mode="Restore",
        refine_mode="Off",
        **kwargs,
    ):
        # Normal graph dependencies track real changes.
        return False

    def check_lazy_status(
        self,
        image=None,
        face_pipe=None,
        boolean_faceprocess_enable=True,
        enable=True,
        process_mode="Restore",
        refine_mode="Off",
        **kwargs,
    ):
        """Request only data required by the active runtime branch.

        Disabled states are passthrough states, therefore they still request
        the image. Active modes additionally request face_pipe when needed.
        """
        needed = []

        if image is None:
            needed.append("image")

        if not bool(enable):
            return needed

        mode = _normalize_process_mode(process_mode)

        # Active primary modes use face_pipe as a defaults/basic-pipe source.
        # Refine runs as a post-process stage and does not need face_pipe.
        if mode in ("Restore", "Detailer") and face_pipe is None:
            needed.append("face_pipe")

        return needed

    def _is_torch_image(self, value):
        return isinstance(value, torch.Tensor) and len(value.shape) >= 3

    def _numpy_to_tensor(self, value):
        if not isinstance(value, np.ndarray):
            return None

        arr = value.astype(np.float32)

        if arr.max() > 1.0:
            arr = arr / 255.0

        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)

        if arr.ndim != 4:
            return None

        return torch.from_numpy(arr)

    def _find_image_tensor(self, value):
        if self._is_torch_image(value):
            return value

        converted = self._numpy_to_tensor(value)
        if converted is not None:
            return converted

        if isinstance(value, (tuple, list)):
            for item in value:
                found = self._find_image_tensor(item)
                if found is not None:
                    return found

        return None

    def _is_valid_segs(self, value):
        if not isinstance(value, tuple):
            return False
        if len(value) != 2:
            return False
        if not isinstance(value[0], tuple):
            return False
        if len(value[0]) != 2:
            return False
        if not isinstance(value[1], list):
            return False

        for seg in value[1]:
            if not hasattr(seg, "cropped_image"):
                return False

        return True

    def _find_valid_segs(self, value):
        if self._is_valid_segs(value):
            return value

        if isinstance(value, (tuple, list)):
            for item in value:
                found = self._find_valid_segs(item)
                if found is not None:
                    return found

        return None

    def _empty_segs(self, image):
        image = self._find_image_tensor(image)
        if image is None:
            return ((0, 0), [])
        return ((image.shape[2], image.shape[1]), [])

    def _detect_segs(
        self,
        image,
        detect_model,
        detect_bbox_threshold,
        detect_bbox_dilation,
        detect_crop_factor,
        detect_drop_size,
        opt_sam_model=None,
    ):
        image = self._find_image_tensor(image)

        if image is None:
            return ((0, 0), [])

        bbox_detector, segm_detector = UltralyticsDetectorProvider().doit(detect_model)

        if detect_model.startswith("bbox/"):
            segm_detector = None

        segs = SimpleDetectorForEach.detect(
            bbox_detector=bbox_detector,
            image=image,
            bbox_threshold=detect_bbox_threshold,
            bbox_dilation=detect_bbox_dilation,
            crop_factor=detect_crop_factor,
            drop_size=detect_drop_size,
            sub_threshold=detect_bbox_threshold,
            sub_dilation=0,
            sub_bbox_expansion=0,
            sam_mask_hint_threshold=0.7,
            post_dilation=0,
            sam_model_opt=opt_sam_model,
            segm_detector_opt=segm_detector,
        )[0]

        if self._is_valid_segs(segs):
            return segs

        return self._empty_segs(image)

    @staticmethod
    def _cmk_bool(value):
        return "Enabled" if bool(value) else "Disabled"

    @staticmethod
    def _format_summary(rows):
        rows = [(str(k), str(v)) for k, v in rows if v is not None]
        width = max((len(k) for k, _ in rows), default=0)
        return "\n".join(f"{k:<{width}} : {v}" for k, v in rows)

    @staticmethod
    def _seg_count(segs):
        try:
            return len(segs[1])
        except Exception:
            return 0

    @staticmethod
    def _tensor_batch_to_uint8(value):
        if value is None:
            return None
        try:
            tensor = value.detach().cpu() if hasattr(value, "detach") else value
            arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            if arr.ndim != 3:
                return None
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.moveaxis(arr, 0, -1)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            return np.clip(arr.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
        except Exception:
            return None

    @staticmethod
    def _uint8_to_tensor_image(arr):
        if arr is None:
            return None
        try:
            out = np.clip(arr, 0, 255).astype(np.float32) / 255.0
            return torch.from_numpy(out)[None,]
        except Exception:
            return None

    @staticmethod
    def _seg_bbox(seg):
        for attr in ("bbox", "crop_region"):
            box = getattr(seg, attr, None)
            if box is None:
                continue
            try:
                if len(box) >= 4:
                    return [int(round(float(box[0]))), int(round(float(box[1]))), int(round(float(box[2]))), int(round(float(box[3])))]
            except Exception:
                pass
        return None


    def _filter_segs_by_selection(
        self,
        segs,
        selection="all",
        sort_by="area",
        reverse=False,
        take_start=0,
        take_count=1,
    ):
        """Apply the legacy face-selection contract directly to Impact SEGS.

        RestoreFaceAdvanced already consumes these controls internally. The
        detailer path works on detector SEGS, so it must apply the same
        selection before SEGSDetailer is called.
        """
        if not self._is_valid_segs(segs):
            return segs
        items = list(segs[1])
        if not items or str(selection or "all") == "all":
            return segs

        def confidence(seg):
            for attr in ("confidence", "score", "detection_confidence"):
                value = getattr(seg, attr, None)
                if value is not None:
                    try:
                        return float(value)
                    except Exception:
                        pass
            return 0.0

        def metric(seg):
            box = self._seg_bbox(seg) or [0, 0, 0, 0]
            x1, y1, x2, y2 = [float(v) for v in box]
            if sort_by == "x_position":
                return (x1 + x2) * 0.5
            if sort_by == "y_position":
                return (y1 + y2) * 0.5
            if sort_by == "detection_confidence":
                return confidence(seg)
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        ordered = sorted(items, key=metric, reverse=bool(reverse))
        if str(selection) == "largest":
            ordered = sorted(items, key=lambda seg: metric(seg), reverse=True)
            chosen = ordered[:1]
        else:
            start = max(0, int(take_start or 0))
            count = max(1, int(take_count or 1))
            chosen = ordered[start:start + count]
        return (segs[0], chosen)

    def _make_detection_preview(self, image, segs):
        arr = self._tensor_batch_to_uint8(image)
        if arr is None:
            return None
        try:
            items = segs[1] if self._is_valid_segs(segs) else []
        except Exception:
            items = []
        if not items:
            return self._uint8_to_tensor_image(arr)
        out = arr.copy()
        h, w = out.shape[:2]
        thickness = max(2, int(round(max(h, w) / 350)))
        for seg in items:
            box = self._seg_bbox(seg)
            if not box:
                continue
            x1, y1, x2, y2 = box
            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            out[y1:y1+thickness, x1:x2+1] = 255
            out[max(y2-thickness+1, y1):y2+1, x1:x2+1] = 255
            out[y1:y2+1, x1:x1+thickness] = 255
            out[y1:y2+1, max(x2-thickness+1, x1):x2+1] = 255
        return self._uint8_to_tensor_image(out)

    def _make_difference_preview(self, before, after):
        a = self._tensor_batch_to_uint8(before)
        b = self._tensor_batch_to_uint8(after)
        if a is None or b is None:
            return None
        if a.shape != b.shape:
            return None
        diff = np.abs(b.astype(np.int16) - a.astype(np.int16)).astype(np.float32)
        diff = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
        return self._uint8_to_tensor_image(diff)

    def _mean_abs_change(self, before, after):
        a = self._tensor_batch_to_uint8(before)
        b = self._tensor_batch_to_uint8(after)
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        return float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32))) / 255.0)

    def _label_preview(self, image, title: str, subtitle: str = ""):
        arr = self._tensor_batch_to_uint8(image)
        if arr is None:
            return image
        try:
            from PIL import Image, ImageDraw, ImageFont
            h, w = arr.shape[:2]
            bar = max(34, min(72, int(round(h * 0.075))))
            canvas = np.zeros((h + bar, w, 3), dtype=np.uint8)
            canvas[:bar, :, :] = 18
            canvas[bar:, :, :] = arr
            pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil)
            try:
                font_title = ImageFont.truetype("Arial.ttf", max(14, int(bar * 0.38)))
                font_sub = ImageFont.truetype("Arial.ttf", max(10, int(bar * 0.24)))
            except Exception:
                font_title = ImageFont.load_default()
                font_sub = ImageFont.load_default()
            draw.text((10, 5), str(title), fill=(245, 245, 245), font=font_title)
            if subtitle:
                draw.text((10, max(20, int(bar * 0.56))), str(subtitle), fill=(190, 190, 190), font=font_sub)
            return self._uint8_to_tensor_image(np.asarray(pil))
        except Exception:
            return image


    @staticmethod
    def _pad_rgb_to_size(arr: np.ndarray, width: int, height: int) -> np.ndarray:
        if arr is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        h, w = arr.shape[:2]
        out = np.zeros((height, width, 3), dtype=np.uint8)
        out[:min(h, height), :min(w, width)] = arr[:min(h, height), :min(w, width), :3]
        return out

    def _compose_pipeline_preview(self, panels):
        """Create one diagnostic overview image from all pipeline panels.

        CMK Preview Render currently displays the first preview image prominently.
        Therefore the first diagnostic preview must be a complete pipeline board,
        while the individual stage panels remain available to Preview Board /
        future expert tools.
        """
        arrays = []
        for panel in panels or []:
            arr = self._tensor_batch_to_uint8(panel)
            if arr is not None:
                arrays.append(arr)
        if not arrays:
            return None
        try:
            gap = 16
            margin = 18
            max_w = max(int(a.shape[1]) for a in arrays)
            total_h = margin + sum(int(a.shape[0]) for a in arrays) + gap * (len(arrays) - 1) + margin
            canvas = np.zeros((total_h, max_w + margin * 2, 3), dtype=np.uint8)
            canvas[:, :, :] = 10
            y = margin
            for arr in arrays:
                h, w = arr.shape[:2]
                x = margin + max(0, (max_w - w) // 2)
                canvas[y:y+h, x:x+w] = arr[:, :, :3]
                y += h + gap
            return self._uint8_to_tensor_image(canvas)
        except Exception:
            return panels[0] if panels else None


    @staticmethod
    def _blur_rgb(rgb: np.ndarray, radius: int) -> np.ndarray:
        radius = max(1, int(radius))
        try:
            import cv2
            k = max(3, radius * 2 + 1)
            if k % 2 == 0:
                k += 1
            return cv2.GaussianBlur(rgb, (k, k), 0)
        except Exception:
            out = rgb.astype(np.float32)
            for _ in range(max(1, radius)):
                padded = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
                out = (
                    padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                    padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                    padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
                ) / 9.0
            return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _soften_mask(mask: np.ndarray, radius: int = 8) -> np.ndarray:
        mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
        if radius <= 0:
            return mask
        try:
            import cv2
            k = max(3, int(radius) * 2 + 1)
            if k % 2 == 0:
                k += 1
            return np.clip(cv2.GaussianBlur(mask, (k, k), 0), 0.0, 1.0).astype(np.float32)
        except Exception:
            out = mask.astype(np.float32)
            for _ in range(max(1, int(radius) // 4)):
                padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
                out = (
                    padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                    padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
                    padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
                ) / 9.0
            return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _refine_rgb(self, rgb: np.ndarray, mode: str, strength: float = 0.65) -> np.ndarray:
        mode = _normalize_refine_mode(mode)
        if mode == "Off":
            return rgb
        strength = max(0.0, min(2.0, float(strength)))
        base = rgb.astype(np.float32)

        if mode == "Smooth":
            blurred = self._blur_rgb(rgb, radius=max(1, int(2 + 4 * strength))).astype(np.float32)
            out = base * (1.0 - 0.65 * strength) + blurred * (0.65 * strength)
            return np.clip(out, 0, 255).astype(np.uint8)

        if mode == "Sharpen":
            blurred = self._blur_rgb(rgb, radius=max(1, int(1 + 2 * strength))).astype(np.float32)
            detail = base - blurred
            out = base + detail * (1.25 * strength)
            return np.clip(out, 0, 255).astype(np.uint8)

        small_blur = self._blur_rgb(rgb, radius=max(1, int(1 + strength))).astype(np.float32)
        large_blur = self._blur_rgb(rgb, radius=max(2, int(3 + 4 * strength))).astype(np.float32)
        fine = base - small_blur
        local = small_blur - large_blur
        out = base + fine * (0.85 * strength) + local * (0.35 * strength)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _mask_from_seg(self, height: int, width: int, seg, padding: float = 0.25) -> np.ndarray | None:
        box = self._seg_bbox(seg)
        if not box:
            return None
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        left = max(0.0, x1 - bw * padding)
        top = max(0.0, y1 - bh * padding)
        right = min(float(width), x2 + bw * padding)
        bottom = min(float(height), y2 + bh * padding)
        yy, xx = np.ogrid[:height, :width]
        cx = (left + right) * 0.5
        cy = (top + bottom) * 0.5
        rx = max(1.0, (right - left) * 0.5)
        ry = max(1.0, (bottom - top) * 0.5)
        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(np.float32)
        return self._soften_mask(mask, 10)

    def _apply_refine_stage(self, image, segs, refine_mode: str):
        refine_mode = _normalize_refine_mode(refine_mode)
        if refine_mode == "Off":
            return image, 0.0, "refine off"

        if not self._is_valid_segs(segs) or not segs[1]:
            return image, 0.0, f"refine skipped | no detection | mode: {refine_mode}"

        try:
            tensor = image.detach().cpu() if hasattr(image, "detach") else image
            arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
            if arr.ndim == 3:
                arr = arr[None, ...]
            if arr.ndim != 4:
                return image, 0.0, f"refine skipped | invalid image tensor | mode: {refine_mode}"

            out_items = []
            changed_values = []
            for batch_index in range(arr.shape[0]):
                rgb = np.clip(arr[batch_index].astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
                h, w = rgb.shape[:2]
                combined_mask = np.zeros((h, w), dtype=np.float32)
                for seg in segs[1]:
                    mask = self._mask_from_seg(h, w, seg, padding=0.25)
                    if mask is not None:
                        combined_mask = np.maximum(combined_mask, mask)
                if float(np.max(combined_mask)) <= 0.0:
                    result = rgb
                else:
                    refined_full = self._refine_rgb(rgb, refine_mode, strength=0.65)
                    alpha = np.clip(combined_mask, 0.0, 1.0)[..., None] * 0.75
                    result = np.clip(rgb.astype(np.float32) * (1.0 - alpha) + refined_full.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                changed_values.append(float(np.mean(np.abs(result.astype(np.float32) - rgb.astype(np.float32))) / 255.0))
                out_items.append(torch.from_numpy(result.astype(np.float32) / 255.0))
            changed_avg = float(np.mean(changed_values)) if changed_values else 0.0
            return torch.stack(out_items, dim=0), changed_avg, f"refine applied | mode: {refine_mode} | changed: {changed_avg:.4f}"
        except Exception as exc:
            return image, 0.0, f"refine skipped | error: {exc} | mode: {refine_mode}"

    def _make_diagnostic(
        self,
        *,
        mode,
        refine_mode="Off",
        enabled,
        status,
        global_enabled=None,
        local_enabled=None,
        image=None,
        primary_image=None,
        processed_image=None,
        segs_detected=None,
        segs_processed=None,
        reason="",
        detect_model=None,
        restore_model=None,
        restore_facedetection=None,
        restore_visibility=None,
        restore_codeformer_weight=None,
        steps=0,
        cfg=0.0,
        denoise=0.0,
        force_inpaint=False,
        guide_size=0.0,
        max_size=0.0,
        paste_feather=None,
        metadata=None,
        warnings=None,
    ):
        detected_count = self._seg_count(segs_detected)
        processed_count = self._seg_count(segs_processed)

        mode_text = str(mode or "").strip() or "unknown"
        refine_text = _normalize_refine_mode(refine_mode)
        if mode_text == "Detailer":
            model_text = detect_model or "none"
        elif mode_text == "Restore":
            model_text = restore_model or "none"
        else:
            model_text = detect_model or restore_model or "none"

        rows = [
            ("Primary Process", mode_text),
            ("Refine", refine_text),
            ("Status", status),
            ("Global Enable", self._cmk_bool(global_enabled) if global_enabled is not None else None),
            ("Local Enable", self._cmk_bool(local_enabled) if local_enabled is not None else None),
            ("Model", model_text),
            ("Input Detections", detected_count),
            ("Output Detections", processed_count),
        ]

        if mode_text == "Restore":
            rows.extend([
                ("Face Detection", restore_facedetection),
                ("Visibility", restore_visibility),
                ("CodeFormer Weight", restore_codeformer_weight),
            ])
        else:
            rows.extend([
                ("Steps", steps),
                ("CFG", cfg),
                ("Denoise", denoise),
                ("Force Inpaint", self._cmk_bool(force_inpaint)),
                ("Guide Size", f"{float(guide_size or 0.0):g}"),
                ("Max Size", f"{float(max_size or 0.0):g}"),
            ])
            if paste_feather is not None:
                rows.append(("Paste Feather", paste_feather))

        if reason:
            rows.append(("Reason", reason))

        summary = self._format_summary(rows)

        previews = []
        preview_panels = []
        preview_steps = []
        stage_previews = []

        def _add_preview(label, tensor, subtitle=""):
            if tensor is None:
                return
            # ``previews`` keeps labelled panels for legacy/single-card renderers.
            # ``preview_steps`` intentionally stores the raw stage image; the
            # universal flow renderer owns card layout, labels, and horizontal
            # composition. Do not pass pre-rendered cards here.
            panel = self._label_preview(tensor, label, subtitle)
            stage_previews.append(panel)
            preview_steps.append({"title": label, "subtitle": subtitle, "image": tensor})
            preview_panels.append(label)

        primary_change = self._mean_abs_change(image, primary_image) if image is not None and primary_image is not None else 0.0
        refine_change = self._mean_abs_change(primary_image, processed_image) if primary_image is not None and processed_image is not None else 0.0
        total_change = self._mean_abs_change(image, processed_image) if image is not None and processed_image is not None else 0.0

        _add_preview("01 Source", image, "original input / passthrough output")

        detection_preview = self._make_detection_preview(image, segs_detected) if image is not None else None
        if detection_preview is not None:
            _add_preview("02 Detected Faces", detection_preview, f"faces detected: {detected_count}")

        if primary_image is not None:
            if mode_text == "Off":
                primary_label = "03 Primary Off"
                primary_subtitle = "no primary processing"
            else:
                primary_label = f"03 Primary {mode_text}"
                primary_subtitle = f"change vs source: {primary_change:.4f}"
            if mode_text != "Off" or primary_change > 0.00001:
                _add_preview(primary_label, primary_image, primary_subtitle)

        primary_diff = self._make_difference_preview(image, primary_image) if image is not None and primary_image is not None and primary_change > 0.00001 else None
        if primary_diff is not None:
            _add_preview("04 Difference Source→Primary", primary_diff, f"primary change: {primary_change:.4f}")

        if processed_image is not None and refine_text != "Off":
            _add_preview(f"05 Refine {refine_text}", processed_image, f"change vs primary: {refine_change:.4f}")

        refine_diff = self._make_difference_preview(primary_image, processed_image) if primary_image is not None and processed_image is not None and refine_change > 0.00001 else None
        if refine_diff is not None:
            _add_preview("06 Difference Primary→Refine", refine_diff, f"refine change: {refine_change:.4f}")

        total_diff = self._make_difference_preview(image, processed_image) if image is not None and processed_image is not None and total_change > 0.00001 else None
        if total_diff is not None and (primary_diff is None or refine_diff is not None):
            _add_preview("07 Difference Source→Final", total_diff, f"total change: {total_change:.4f}")

        if processed_image is not None and total_change > 0.00001:
            _add_preview("08 Final", processed_image, "final output / image_proceed")

        # Keep previews as individual pipeline stages. The diagnostic renderer
        # is responsible for arranging preview_steps horizontally.
        previews.extend(stage_previews)

        diagnostic_metadata = {
            "primary_process": mode_text,
            "refine_mode": refine_text,
            "mode": mode_text,
            "enabled": bool(enabled),
            "global_enabled": bool(global_enabled) if global_enabled is not None else None,
            "local_enabled": bool(local_enabled) if local_enabled is not None else None,
            "status": status,
            "model": model_text,
            "detect_model": detect_model,
            "restore_model": restore_model,
            "input_detections": detected_count,
            "output_detections": processed_count,
            "steps": steps,
            "cfg": cfg,
            "denoise": denoise,
            "force_inpaint": bool(force_inpaint),
            "guide_size": guide_size,
            "max_size": max_size,
            "paste_feather": paste_feather,
            "reason": reason,
            "preview_order": "source, detections, primary, primary_difference, refine, refine_difference, final_difference",
            "preview_panels": preview_panels,
            "preview_steps": preview_steps,
            "primary_change": primary_change,
            "refine_change": refine_change,
            "total_change": total_change,
        }
        diagnostic_metadata.update(dict(metadata or {}))

        warning_items = list(warnings or [])
        if reason and reason not in warning_items and status not in ("Processed", "Enabled"):
            warning_items.append(reason)

        return make_diagnostic_payload(
            title="FaceProcess",
            node="CMK FaceProcess",
            previews=previews,
            stages=preview_steps,
            summary=summary,
            details=summary,
            mode=mode_text,
            metadata=diagnostic_metadata,
            warnings=warning_items,
            metrics={
                "primary_change": float(primary_change or 0.0),
                "refine_change": float(refine_change or 0.0),
                "total_change": float(total_change or 0.0),
            },
        )

    def run(
        self,
        image,
        face_pipe=None,
        boolean_faceprocess_enable=True,
        enable=True,
        process_mode="Restore",
        refine_mode="Off",
        detect_model=None,
        detect_bbox_threshold=0.5,
        detect_bbox_dilation=0,
        detect_crop_factor=3.0,
        detect_drop_size=10,
        restore_model="none",
        restore_facedetection="retinaface_resnet50",
        restore_visibility=1.0,
        restore_codeformer_weight=0.5,
        select_face_selection="all",
        select_sort_by="area",
        select_reverse_order=False,
        select_take_start=0,
        select_take_count=1,
        detail_guide_size=512,
        detail_guide_size_for=True,
        detail_max_size=768,
        detail_denoise=0.5,
        detail_noise_mask=True,
        detail_force_inpaint=True,
        detail_paste_feather=20,
    ):
        legacy_off = _is_legacy_off_mode(process_mode)
        process_mode = _normalize_process_mode(process_mode)
        refine_mode = _normalize_refine_mode(refine_mode)
        empty_segs = ((0, 0), [])

        global_enabled = bool(boolean_faceprocess_enable)
        local_enabled = bool(enable)
        faceprocess_enabled = global_enabled and local_enabled

        # Disabled states are handled after image normalization so output remains a valid passthrough image.

        image = self._find_image_tensor(image)

        if image is None:
            face_log = "error | no valid image tensor"
            diagnostic = self._make_diagnostic(
                mode=process_mode,
                refine_mode=refine_mode,
                enabled=False,
                status="Error",
                global_enabled=global_enabled,
                local_enabled=local_enabled,
                segs_detected=empty_segs,
                segs_processed=empty_segs,
                reason="no valid image tensor",
                detect_model=detect_model,
                restore_model=restore_model,
                restore_facedetection=restore_facedetection,
                restore_visibility=restore_visibility,
                restore_codeformer_weight=restore_codeformer_weight,
                steps=20,
                cfg=8.0,
                denoise=detail_denoise,
                force_inpaint=detail_force_inpaint,
                guide_size=detail_guide_size,
                max_size=detail_max_size,
                warnings=["no valid image tensor"],
            )
            return (None, None, empty_segs, empty_segs, face_log, False, diagnostic)

        empty_segs = self._empty_segs(image)

        if not faceprocess_enabled:
            reason = []
            if not global_enabled:
                reason.append("global disabled")
            if not local_enabled:
                reason.append("local disabled")
            reason_text = " + ".join(reason) or "disabled"
            face_log = f"{reason_text} | image passed through unchanged"
            diagnostic = self._make_diagnostic(
                mode=process_mode,
                refine_mode=refine_mode,
                enabled=False,
                status="Disabled",
                global_enabled=global_enabled,
                local_enabled=local_enabled,
                image=image,
                processed_image=image,
                segs_detected=empty_segs,
                segs_processed=empty_segs,
                reason=reason_text,
                detect_model=detect_model,
                restore_model=restore_model,
                restore_facedetection=restore_facedetection,
                restore_visibility=restore_visibility,
                restore_codeformer_weight=restore_codeformer_weight,
                steps=20,
                cfg=8.0,
                denoise=detail_denoise,
                force_inpaint=detail_force_inpaint,
                guide_size=detail_guide_size,
                max_size=detail_max_size,
            )
            return (image, image, empty_segs, empty_segs, face_log, False, diagnostic)

        if not isinstance(face_pipe, dict):
            face_pipe = {}

        all_detect_models = _sort_face_detectors(
            ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
            + ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]
        )

        def _pipe_value(name, current=None, none_values=(None, "none")):
            if current not in none_values:
                return current
            return face_pipe.get(name, current)

        basic_pipe = (
            face_pipe.get("face_model"),
            face_pipe.get("face_clip"),
            face_pipe.get("face_vae"),
            face_pipe.get("face_conditioning_pos"),
            face_pipe.get("face_conditioning_neg"),
        )

        detect_model = _pipe_value("face_detect_model", detect_model)
        detect_bbox_threshold = _pipe_value("face_detect_bbox_threshold", detect_bbox_threshold, none_values=(None,))
        detect_bbox_dilation = _pipe_value("face_detect_bbox_dilation", detect_bbox_dilation, none_values=(None,))
        detect_crop_factor = _pipe_value("face_detect_crop_factor", detect_crop_factor, none_values=(None,))
        detect_drop_size = _pipe_value("face_detect_drop_size", detect_drop_size, none_values=(None,))

        restore_model = _pipe_value("face_restore_model", restore_model)
        restore_facedetection = _pipe_value("face_restore_facedetection", restore_facedetection, none_values=(None,))
        restore_visibility = _pipe_value("face_restore_visibility", restore_visibility, none_values=(None,))
        restore_codeformer_weight = _pipe_value("face_restore_codeformer_weight", restore_codeformer_weight, none_values=(None,))

        select_face_selection = _pipe_value("face_select_face_selection", select_face_selection, none_values=(None,))
        select_sort_by = _pipe_value("face_select_sort_by", select_sort_by, none_values=(None,))
        select_reverse_order = _pipe_value("face_select_reverse_order", select_reverse_order, none_values=(None,))
        select_take_start = _pipe_value("face_select_take_start", select_take_start, none_values=(None,))
        select_take_count = _pipe_value("face_select_take_count", select_take_count, none_values=(None,))

        detail_guide_size = _pipe_value("face_detail_guide_size", detail_guide_size, none_values=(None,))
        detail_guide_size_for = _pipe_value("face_detail_guide_size_for", detail_guide_size_for, none_values=(None,))
        detail_max_size = _pipe_value("face_detail_max_size", detail_max_size, none_values=(None,))
        detail_denoise = _pipe_value("face_detail_denoise", detail_denoise, none_values=(None,))
        detail_denoise = clamp_detailer_denoise(detail_denoise)
        detail_noise_mask = _pipe_value("face_detail_noise_mask", detail_noise_mask, none_values=(None,))
        detail_force_inpaint = _pipe_value("face_detail_force_inpaint", detail_force_inpaint, none_values=(None,))
        detail_paste_feather = _pipe_value("face_detail_paste_feather", detail_paste_feather, none_values=(None,))
        detail_paste_feather = min(200, max(0, int(detail_paste_feather)))

        detail_seed = face_pipe.get("face_seed", 0)
        detail_seed = 0 if detail_seed is None else int(detail_seed)
        detail_steps = face_pipe.get("face_steps", 20)
        detail_steps = 20 if detail_steps is None else int(detail_steps)
        detail_cfg = face_pipe.get("face_cfg", 8.0)
        detail_sampler = face_pipe.get("face_sampler")
        detail_scheduler = face_pipe.get("face_scheduler")

        opt_sam_model = face_pipe.get("face_sam_model")

        segs_detected = empty_segs
        seg_count_in = 0
        processed_image = image
        process_info = "none"

        if process_mode == "Restore":
            # CMK face restoration has its own InsightFace detector. CMK SEGS
            # detection is useful for diagnostics only and must never block restore.
            if detect_model not in (None, "none"):
                segs_detected = self._detect_segs(
                    image=image,
                    detect_model=detect_model,
                    detect_bbox_threshold=detect_bbox_threshold,
                    detect_bbox_dilation=detect_bbox_dilation,
                    detect_crop_factor=detect_crop_factor,
                    detect_drop_size=detect_drop_size,
                    opt_sam_model=opt_sam_model,
                )
                seg_count_in = len(segs_detected[1])

            if restore_model in (None, "none"):
                face_log = "restore skipped | no restore model selected"
                diagnostic = self._make_diagnostic(
                    mode=process_mode,
                    refine_mode=refine_mode,
                    enabled=False,
                    status="Skipped",
                    image=image,
                    processed_image=image,
                    segs_detected=segs_detected,
                    segs_processed=empty_segs,
                    reason="no restore model selected",
                    detect_model=detect_model,
                    restore_model=restore_model,
                    restore_facedetection=restore_facedetection,
                    restore_visibility=restore_visibility,
                    restore_codeformer_weight=restore_codeformer_weight,
                    metadata={"restore_model": restore_model},
                    warnings=["no restore model selected"],
                )
                return (image, image, segs_detected, empty_segs, face_log, False, diagnostic)

            restore_result = RestoreFaceAdvanced().execute(
                image=image,
                model=restore_model,
                visibility=restore_visibility,
                codeformer_weight=restore_codeformer_weight,
                facedetection=restore_facedetection,
                face_selection=select_face_selection,
                sort_by=select_sort_by,
                reverse_order=select_reverse_order,
                take_start=select_take_start,
                take_count=select_take_count,
            )

            found_image = self._find_image_tensor(restore_result)
            processed_image = found_image if found_image is not None else image

            process_info = (
                f"restore | model: {restore_model} | "
                f"facedetection: {restore_facedetection} | visibility: {restore_visibility} | "
                f"codeformer_weight: {restore_codeformer_weight}"
            )

        elif process_mode == "Off":
            processed_image = image
            process_info = "primary off | passthrough"

        elif process_mode == "Detailer":
            if detect_model in (None, "none"):
                face_log = "detailer skipped | no detection model selected"
                diagnostic = self._make_diagnostic(
                    mode=process_mode,
                    refine_mode=refine_mode,
                    enabled=False,
                    status="Skipped",
                    image=image,
                    processed_image=image,
                    segs_detected=empty_segs,
                    segs_processed=empty_segs,
                    reason="no detection model selected",
                    detect_model=detect_model,
                    steps=detail_steps,
                    cfg=detail_cfg,
                    denoise=detail_denoise,
                    force_inpaint=detail_force_inpaint,
                    guide_size=detail_guide_size,
                    max_size=detail_max_size,
                    paste_feather=detail_paste_feather,
                    metadata={"detect_model": detect_model},
                    warnings=["no detection model selected"],
                )
                return (image, image, empty_segs, empty_segs, face_log, False, diagnostic)

            used_detect_model = detect_model
            segs_detected = self._detect_segs(
                image=image,
                detect_model=used_detect_model,
                detect_bbox_threshold=detect_bbox_threshold,
                detect_bbox_dilation=detect_bbox_dilation,
                detect_crop_factor=detect_crop_factor,
                detect_drop_size=detect_drop_size,
                opt_sam_model=opt_sam_model,
            )
            seg_count_in = len(segs_detected[1])

            # Face Process should not silently fail just because ComfyUI picked a
            # non-face detector as the first default model, e.g. breast/hand models.
            # If the selected/default detector finds nothing, retry once with the
            # most face-like detector available. Explicit successful selections are
            # left untouched.
            fallback_detect_model = _preferred_face_detector(all_detect_models)
            if (
                seg_count_in == 0
                and fallback_detect_model
                and fallback_detect_model != used_detect_model
                and _is_likely_non_face_detector(used_detect_model)
            ):
                fallback_segs = self._detect_segs(
                    image=image,
                    detect_model=fallback_detect_model,
                    detect_bbox_threshold=detect_bbox_threshold,
                    detect_bbox_dilation=detect_bbox_dilation,
                    detect_crop_factor=detect_crop_factor,
                    detect_drop_size=detect_drop_size,
                    opt_sam_model=opt_sam_model,
                )
                fallback_count = len(fallback_segs[1])
                if fallback_count > 0:
                    segs_detected = fallback_segs
                    seg_count_in = fallback_count
                    used_detect_model = fallback_detect_model

            # Apply the same face selection used by the restore path before
            # the detected segments enter SEGSDetailer. This guarantees that
            # selected_face, the processed segments, and the actual image
            # modification all describe the same target.
            segs_detected = self._filter_segs_by_selection(
                segs_detected,
                selection=select_face_selection,
                sort_by=select_sort_by,
                reverse=select_reverse_order,
                take_start=select_take_start,
                take_count=select_take_count,
            )
            seg_count_in = self._seg_count(segs_detected)

            if seg_count_in == 0:
                processed_image = image
                process_info = f"detailer skipped | no selected detection | detect_model: {used_detect_model}"
            else:
                detailer_result = SEGSDetailer().doit(
                    image=image,
                    segs=segs_detected,
                    guide_size=detail_guide_size,
                    guide_size_for=detail_guide_size_for,
                    max_size=detail_max_size,
                    seed=detail_seed,
                    steps=detail_steps,
                    cfg=detail_cfg,
                    sampler_name=detail_sampler,
                    scheduler=detail_scheduler,
                    denoise=detail_denoise,
                    noise_mask=detail_noise_mask,
                    force_inpaint=detail_force_inpaint,
                    basic_pipe=basic_pipe,
                    refiner_ratio=0.2,
                    batch_size=1,
                    cycle=1,
                    refiner_basic_pipe_opt=None,
                    inpaint_model=False,
                    noise_mask_feather=20,
                    scheduler_func_opt=None,
                )

                segs_detailed = self._find_valid_segs(detailer_result)

                if segs_detailed is not None:
                    paste_result = SEGSPaste.doit(
                        image,
                        segs_detailed,
                        detail_paste_feather,
                        alpha=255,
                    )
                    pasted_image = self._find_image_tensor(paste_result)
                    processed_image = pasted_image if pasted_image is not None else image
                else:
                    processed_image = image
                    segs_detailed = segs_detected

                process_info = (
                    f"detailer + paste | detections: {seg_count_in} | detect_model: {used_detect_model} | "
                    f"steps: {detail_steps} | cfg: {detail_cfg} | denoise: {detail_denoise} | "
                    f"paste_feather: {detail_paste_feather}"
                )

        else:
            processed_image = image
            process_info = f"unknown mode fallback: {process_mode}"

        primary_image = self._find_image_tensor(processed_image)
        if primary_image is None:
            primary_image = image

        refine_changed = 0.0
        if refine_mode != "Off":
            refine_segs = segs_detected
            if self._seg_count(refine_segs) == 0 and detect_model not in (None, "none"):
                refine_segs = self._detect_segs(
                    image=primary_image,
                    detect_model=detect_model,
                    detect_bbox_threshold=detect_bbox_threshold,
                    detect_bbox_dilation=detect_bbox_dilation,
                    detect_crop_factor=detect_crop_factor,
                    detect_drop_size=detect_drop_size,
                    opt_sam_model=opt_sam_model,
                )
                if self._seg_count(segs_detected) == 0:
                    segs_detected = refine_segs
                    seg_count_in = self._seg_count(segs_detected)
            processed_image, refine_changed, refine_info = self._apply_refine_stage(primary_image, refine_segs, refine_mode)
            process_info = f"{process_info} | {refine_info}"

        processed_image = self._find_image_tensor(processed_image)
        if processed_image is None:
            processed_image = image

        if detect_model is None:
            segs_processed = empty_segs
        else:
            segs_processed = self._detect_segs(
                image=processed_image,
                detect_model=detect_model,
                detect_bbox_threshold=detect_bbox_threshold,
                detect_bbox_dilation=detect_bbox_dilation,
                detect_crop_factor=detect_crop_factor,
                detect_drop_size=detect_drop_size,
                opt_sam_model=opt_sam_model,
            )

        seg_count_out = len(segs_processed[1])

        face_log = (
            f"enabled | primary: {process_mode} | refine: {refine_mode} | "
            f"input detections: {seg_count_in} | "
            f"output detections: {seg_count_out} | "
            f"{process_info}"
        )

        print(f"[CMK Face Process] {face_log}")

        diagnostic = self._make_diagnostic(
            mode=process_mode,
            refine_mode=refine_mode,
            enabled=True,
            status="Enabled",
            global_enabled=global_enabled,
            local_enabled=local_enabled,
            image=image,
            primary_image=primary_image,
            processed_image=processed_image,
            segs_detected=segs_detected,
            segs_processed=segs_processed,
            reason="",
            detect_model=detect_model,
            restore_model=restore_model,
            restore_facedetection=restore_facedetection,
            restore_visibility=restore_visibility,
            restore_codeformer_weight=restore_codeformer_weight,
            steps=detail_steps,
            cfg=detail_cfg,
            denoise=detail_denoise,
            force_inpaint=detail_force_inpaint,
            guide_size=detail_guide_size,
            max_size=detail_max_size,
            paste_feather=detail_paste_feather,
            metadata={
                "detect_model": detect_model,
                "restore_model": restore_model,
                "restore_facedetection": restore_facedetection,
                "restore_visibility": restore_visibility,
                "restore_codeformer_weight": restore_codeformer_weight,
                "selection": select_face_selection,
                "sort_by": select_sort_by,
                "take_start": select_take_start,
                "take_count": select_take_count,
                "detail_guide_size": detail_guide_size,
                "detail_max_size": detail_max_size,
                "detail_denoise": detail_denoise,
                "detail_force_inpaint": bool(detail_force_inpaint),
                "detail_paste_feather": detail_paste_feather,
                "refine_mode": refine_mode,
                "refine_changed": refine_changed,
            },
        )

        return (
            image,
            processed_image,
            segs_detected,
            segs_processed,
            face_log,
            True,
            diagnostic,
        )


NODE_CLASS_MAPPINGS = {
    "CMK_FaceProcess": CMK_FaceProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_FaceProcess": "CMK FaceProcess",
}
