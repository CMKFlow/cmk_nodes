from __future__ import annotations

from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import comfy.samplers
import folder_paths
import nodes


CMKSEG = namedtuple(
    "CMKSEG",
    (
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
        "batch_index",
    ),
    defaults=(None, None),
)


def _register_model_folders() -> None:
    models_root = Path(folder_paths.models_dir)
    supported = set(getattr(folder_paths, "supported_pt_extensions", {".pt", ".pth"}))
    supported.update({".onnx"})
    ultralytics_roots = []
    base_registration = folder_paths.folder_names_and_paths.get("ultralytics")
    if base_registration:
        ultralytics_roots = [Path(path) for path in base_registration[0]]
    for key, path in (
        ("ultralytics_bbox", models_root / "ultralytics" / "bbox"),
        ("ultralytics_segm", models_root / "ultralytics" / "segm"),
        ("sams", models_root / "sams"),
    ):
        paths = [str(path)]
        if key.startswith("ultralytics_"):
            kind = key.removeprefix("ultralytics_")
            paths.extend(str(root / kind) for root in ultralytics_roots)
            # Some established CMK installations store YOLO segmentation
            # checkpoints below ultralytics/bbox. Expose those files through
            # the segmentation category as well, without copying models.
            if kind == "segm":
                paths.extend(str(root / "bbox") for root in ultralytics_roots)
        current = folder_paths.folder_names_and_paths.get(key)
        if current:
            paths.extend(str(item) for item in current[0])
            supported.update(current[1])
        folder_paths.folder_names_and_paths[key] = (list(dict.fromkeys(paths)), supported)


_register_model_folders()


def get_schedulers():
    return comfy.samplers.KSampler.SCHEDULERS


def _image_tensor(value):
    if isinstance(value, torch.Tensor) and value.ndim == 4:
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _image_tensor(item)
            if found is not None:
                return found
    return None


def _mask_tensor(mask, *, batch: int, height: int, width: int, device, dtype):
    if mask is None:
        return torch.ones((batch, height, width), device=device, dtype=dtype)
    value = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
    value = value.detach().to(device=device, dtype=dtype)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    elif value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    elif value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        return torch.ones((batch, height, width), device=device, dtype=dtype)
    if value.shape[0] == 1 and batch > 1:
        value = value.expand(batch, -1, -1)
    if tuple(value.shape[-2:]) != (height, width):
        value = F.interpolate(value.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    return value.clamp(0.0, 1.0)


def _soft_mask(mask: torch.Tensor, feather: int) -> torch.Tensor:
    radius = max(0, int(feather))
    if radius <= 0:
        return mask
    kernel = max(3, radius * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    value = mask.unsqueeze(1)
    value = F.avg_pool2d(value, kernel_size=kernel, stride=1, padding=kernel // 2)
    return value[:, 0].clamp(0.0, 1.0)


class SEGSPaste:
    @staticmethod
    def doit(image, segs, feather=5, alpha=255):
        source = _image_tensor(image)
        if source is None:
            raise ValueError("CMK native SEGSPaste requires a ComfyUI IMAGE tensor")
        if not isinstance(segs, tuple) or len(segs) != 2:
            raise ValueError("CMK native SEGSPaste received invalid SEGS")

        result = source.clone()
        _, image_h, image_w, _ = result.shape
        alpha_scale = max(0.0, min(1.0, float(alpha) / 255.0))

        for seg in segs[1]:
            crop = _image_tensor(getattr(seg, "cropped_image", None))
            region = getattr(seg, "crop_region", None)
            if crop is None or region is None or len(region) < 4:
                continue
            x1 = max(0, min(image_w, int(round(float(region[0])))))
            y1 = max(0, min(image_h, int(round(float(region[1])))))
            x2 = max(0, min(image_w, int(round(float(region[2])))))
            y2 = max(0, min(image_h, int(round(float(region[3])))))
            if x2 <= x1 or y2 <= y1:
                continue
            height, width = y2 - y1, x2 - x1
            batch_index = getattr(seg, "batch_index", None)
            if batch_index is None:
                target = result
            else:
                index = max(0, min(result.shape[0] - 1, int(batch_index)))
                target = result[index:index + 1]
            candidate = crop.to(device=result.device, dtype=result.dtype)
            if candidate.shape[0] == 1 and target.shape[0] > 1:
                candidate = candidate.expand(target.shape[0], -1, -1, -1)
            if tuple(candidate.shape[1:3]) != (height, width):
                candidate = F.interpolate(
                    candidate.movedim(-1, 1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).movedim(1, -1)
            mask = _mask_tensor(
                getattr(seg, "cropped_mask", None),
                batch=target.shape[0],
                height=height,
                width=width,
                device=result.device,
                dtype=result.dtype,
            )
            mask = (_soft_mask(mask, int(feather)) * alpha_scale).unsqueeze(-1)
            base = target[:, y1:y2, x1:x2, :]
            target[:, y1:y2, x1:x2, :] = base * (1.0 - mask) + candidate * mask
        return (result,)


def _crop_region(bbox, image_w: int, image_h: int, crop_factor: float):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    factor = max(1.0, float(crop_factor))
    side_w = width * factor
    side_h = height * factor
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    left = max(0, int(round(cx - side_w * 0.5)))
    top = max(0, int(round(cy - side_h * 0.5)))
    right = min(image_w, int(round(cx + side_w * 0.5)))
    bottom = min(image_h, int(round(cy + side_h * 0.5)))
    return (left, top, max(left + 1, right), max(top + 1, bottom))


@lru_cache(maxsize=8)
def _load_yolo(path: str):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise ImportError(
            "CMK detection requires the Python package 'ultralytics'. "
            "Install CMK Flow requirements in the ComfyUI Python environment."
        ) from exc
    return YOLO(path)


class _CMKUltralyticsDetector:
    def __init__(self, model_name: str, kind: str):
        self.model_name = str(model_name)
        self.kind = str(kind)

    def detect(self, image, threshold, dilation, crop_factor, drop_size, sam_model_opt=None):
        tensor = _image_tensor(image)
        if tensor is None:
            return ((0, 0), [])
        model_path = folder_paths.get_full_path(f"ultralytics_{self.kind}", self.model_name)
        if not model_path:
            raise FileNotFoundError(f"CMK detector model not found: {self.kind}/{self.model_name}")
        model = _load_yolo(str(model_path))
        _, image_h, image_w, _ = tensor.shape
        items = []

        for batch_index in range(tensor.shape[0]):
            rgb = np.clip(tensor[batch_index].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            predictions = model.predict(source=rgb, conf=float(threshold), verbose=False)
            if not predictions:
                continue
            prediction = predictions[0]
            boxes = getattr(prediction, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            masks = getattr(prediction, "masks", None)
            mask_data = None if masks is None else masks.data.detach().cpu().float()
            if mask_data is not None and tuple(mask_data.shape[-2:]) != (image_h, image_w):
                mask_data = F.interpolate(mask_data.unsqueeze(1), size=(image_h, image_w), mode="bilinear", align_corners=False)[:, 0]

            for index, raw_box in enumerate(xyxy):
                x1, y1, x2, y2 = [float(v) for v in raw_box]
                dilation_px = float(dilation)
                bbox = (
                    max(0, int(round(x1 - dilation_px))),
                    max(0, int(round(y1 - dilation_px))),
                    min(image_w, int(round(x2 + dilation_px))),
                    min(image_h, int(round(y2 + dilation_px))),
                )
                if bbox[2] - bbox[0] < int(drop_size) or bbox[3] - bbox[1] < int(drop_size):
                    continue
                region = _crop_region(bbox, image_w, image_h, crop_factor)
                rx1, ry1, rx2, ry2 = region
                # Each detection belongs to one IMAGE batch item. Keeping the
                # full batch here would paste it into unrelated images.
                crop = tensor[batch_index:batch_index + 1, ry1:ry2, rx1:rx2, :].detach().cpu()
                if mask_data is not None and index < mask_data.shape[0]:
                    crop_mask = mask_data[index:index + 1, ry1:ry2, rx1:rx2]
                elif sam_model_opt is not None and hasattr(sam_model_opt, "mask_for_box"):
                    full_mask = sam_model_opt.mask_for_box(rgb, bbox)
                    crop_mask = torch.from_numpy(full_mask[ry1:ry2, rx1:rx2]).float().unsqueeze(0)
                else:
                    crop_mask = torch.zeros((1, ry2 - ry1, rx2 - rx1), dtype=torch.float32)
                    bx1, by1 = max(0, bbox[0] - rx1), max(0, bbox[1] - ry1)
                    bx2, by2 = min(rx2 - rx1, bbox[2] - rx1), min(ry2 - ry1, bbox[3] - ry1)
                    crop_mask[:, by1:by2, bx1:bx2] = 1.0
                class_id = int(classes[index]) if index < len(classes) else -1
                names = getattr(prediction, "names", {}) or {}
                label = str(names.get(class_id, class_id))
                items.append(
                    CMKSEG(
                        crop,
                        crop_mask,
                        float(confidences[index]),
                        region,
                        bbox,
                        label,
                        None,
                        batch_index,
                    )
                )
        return ((image_w, image_h), items)


class UltralyticsDetectorProvider:
    def doit(self, model_name):
        text = str(model_name or "")
        if "/" not in text:
            raise ValueError(f"Invalid CMK detector model: {text}")
        prefix, name = text.split("/", 1)
        if prefix not in ("bbox", "segm"):
            raise ValueError(f"Unsupported CMK detector type: {prefix}")
        detector = _CMKUltralyticsDetector(name, prefix)
        return (detector, detector if prefix == "segm" else None)


class SimpleDetectorForEach:
    @staticmethod
    def detect(
        bbox_detector,
        image,
        bbox_threshold,
        bbox_dilation,
        crop_factor,
        drop_size,
        sam_model_opt=None,
        **_kwargs,
    ):
        return (
            bbox_detector.detect(
                image,
                bbox_threshold,
                bbox_dilation,
                crop_factor,
                drop_size,
                sam_model_opt=sam_model_opt,
            ),
        )


class _CMKSAMModel:
    def __init__(self, checkpoint: str, device: str):
        self.checkpoint = str(checkpoint)
        self.device = str(device)
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:
            raise ImportError(
                "CMK SAM masking requires the Python package 'segment-anything'."
            ) from exc
        name = Path(self.checkpoint).name.lower()
        model_type = "vit_h" if "vit_h" in name else "vit_l" if "vit_l" in name else "vit_b"
        model = sam_model_registry[model_type](checkpoint=self.checkpoint)
        model.to(device=self.device)
        self._predictor = SamPredictor(model)
        return self._predictor

    def mask_for_box(self, rgb: np.ndarray, bbox):
        predictor = self._load()
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.asarray(bbox, dtype=np.float32),
            multimask_output=True,
        )
        index = int(np.argmax(scores)) if len(scores) else 0
        return np.asarray(masks[index], dtype=np.float32)


class CMKSAMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("sams")
        return {
            "required": {
                "model_name": (models if models else ["No SAM model installed"],),
                "device_mode": (("AUTO", "Prefer GPU", "CPU"), {"default": "CPU"}),
            }
        }

    RETURN_TYPES = ("SAM_MODEL",)
    FUNCTION = "load"
    CATEGORY = "CMK/Internal"

    def load(self, model_name, device_mode="CPU"):
        path = folder_paths.get_full_path("sams", str(model_name))
        if not path:
            raise FileNotFoundError(
                f"CMK SAM model not found: {model_name}. Place it in ComfyUI/models/sams."
            )
        mode = str(device_mode or "CPU").lower()
        device = "cpu"
        if mode != "cpu":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        return (_CMKSAMModel(path, device),)


def _resize_image(image: torch.Tensor, height: int, width: int):
    return F.interpolate(
        image.movedim(-1, 1), size=(height, width), mode="bicubic", align_corners=False
    ).movedim(1, -1).clamp(0.0, 1.0)


class SEGSDetailer:
    def doit(
        self,
        image,
        segs,
        guide_size,
        guide_size_for,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        noise_mask,
        force_inpaint,
        basic_pipe,
        **_kwargs,
    ):
        source = _image_tensor(image)
        if source is None or not isinstance(segs, tuple) or len(segs) != 2:
            return segs
        if not isinstance(basic_pipe, (tuple, list)) or len(basic_pipe) < 5:
            raise ValueError("CMK native detailer requires BASIC_PIPE (model, clip, vae, positive, negative)")
        model, _clip, vae, positive, negative = basic_pipe[:5]
        detailed = []

        for index, seg in enumerate(segs[1]):
            crop = _image_tensor(getattr(seg, "cropped_image", None))
            if crop is None:
                detailed.append(seg)
                continue
            crop_h, crop_w = int(crop.shape[1]), int(crop.shape[2])
            bbox = getattr(seg, "bbox", (0, 0, crop_w, crop_h))
            basis = min(max(1, int(bbox[2] - bbox[0])), max(1, int(bbox[3] - bbox[1]))) if guide_size_for else min(crop_h, crop_w)
            scale = max(1.0, float(guide_size) / float(max(1, basis)))
            if max(crop_h, crop_w) * scale > float(max_size):
                scale = float(max_size) / float(max(crop_h, crop_w))
            target_h = max(8, int(round(crop_h * scale / 8.0)) * 8)
            target_w = max(8, int(round(crop_w * scale / 8.0)) * 8)
            work = _resize_image(crop, target_h, target_w)
            latent = {"samples": vae.encode(work)}
            if bool(noise_mask) or bool(force_inpaint):
                mask = _mask_tensor(
                    getattr(seg, "cropped_mask", None),
                    batch=work.shape[0],
                    height=target_h,
                    width=target_w,
                    device=work.device,
                    dtype=work.dtype,
                )
                latent["noise_mask"] = mask
            sampled = nodes.common_ksampler(
                model,
                int(seed) + index,
                int(steps),
                float(cfg),
                str(sampler_name),
                str(scheduler),
                positive,
                negative,
                latent,
                denoise=float(denoise),
            )[0]
            decoded = vae.decode(sampled["samples"])
            restored = _resize_image(decoded, crop_h, crop_w).detach().cpu()
            detailed.append(seg._replace(cropped_image=restored))
        return (segs[0], detailed)


class core:
    get_schedulers = staticmethod(get_schedulers)
