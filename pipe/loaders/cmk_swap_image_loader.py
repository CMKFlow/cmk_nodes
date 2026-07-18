from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths

from ..cmk_log_pipe import cmk_add_block
from ..cmk_pipe_image import (
    RESOLUTION_PRESETS,
    UPSCALE_METHODS,
    get_image_size,
    parse_resolution,
    resize_image_tensor,
)
from ...utils.cmk_diagnostic import make_diagnostic_payload
from .cmk_image_load_resize import CROP_POSITIONS, calculate_crop_box


_IMAGE_PAIR_DEFAULT = json.dumps(
    {"target": "", "source": ""},
    ensure_ascii=False,
    separators=(",", ":"),
)


def _parse_image_pair(value: Any) -> tuple[str, str]:
    """Return target/source names from the serialized dual-loader widget."""
    payload: Any = value
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            payload = {}
        else:
            try:
                payload = json.loads(text)
            except Exception as exc:
                raise ValueError(
                    "CMK Swap Image Loader -Pipe-: IMAGE PAIR is invalid"
                ) from exc

    if not isinstance(payload, dict):
        raise ValueError("CMK Swap Image Loader -Pipe-: IMAGE PAIR is invalid")

    target = str(payload.get("target", "") or "").strip()
    source = str(payload.get("source", "") or "").strip()
    if not target:
        raise ValueError("CMK Swap Image Loader -Pipe-: TARGET IMAGE is missing")
    if not source:
        raise ValueError("CMK Swap Image Loader -Pipe-: SOURCE IMAGE is missing")
    return target, source


def _resolve_image_path(image_name: str) -> str:
    try:
        return folder_paths.get_annotated_filepath(image_name)
    except Exception:
        return os.path.join(folder_paths.get_input_directory(), image_name)


def _probe_image(image_path: str) -> tuple[int, int, str]:
    with Image.open(image_path) as image:
        source_format = str(image.format or "unknown")
        image.seek(0)
        frame = ImageOps.exif_transpose(image.copy())
        width, height = frame.size
    return int(width), int(height), source_format


def _load_rgb_frames(
    image_path: str,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    crop_enabled: bool = False,
    crop_position: str = "center",
) -> tuple[torch.Tensor, int, int, tuple[int, int, int, int]]:
    output_images: list[torch.Tensor] = []
    source_width: int | None = None
    source_height: int | None = None
    first_crop_box: tuple[int, int, int, int] | None = None

    with Image.open(image_path) as image:
        for raw_frame in ImageSequence.Iterator(image):
            frame = ImageOps.exif_transpose(raw_frame)
            frame_width, frame_height = frame.size

            if source_width is None or source_height is None:
                source_width, source_height = int(frame_width), int(frame_height)

            if crop_enabled:
                if target_width is None or target_height is None:
                    raise ValueError(
                        "CMK Swap Image Loader -Pipe-: crop requires a target size"
                    )
                crop_box = calculate_crop_box(
                    frame_width,
                    frame_height,
                    int(target_width),
                    int(target_height),
                    crop_position,
                )
                frame = frame.crop(crop_box)
                if first_crop_box is None:
                    first_crop_box = tuple(int(value) for value in crop_box)
            elif first_crop_box is None:
                first_crop_box = (0, 0, int(frame_width), int(frame_height))

            array = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
            output_images.append(torch.from_numpy(array)[None, ...])

    if not output_images:
        raise RuntimeError(
            "CMK Swap Image Loader -Pipe-: no image frames could be loaded"
        )

    width = int(source_width or 0)
    height = int(source_height or 0)
    return (
        torch.cat(output_images, dim=0),
        width,
        height,
        tuple(first_crop_box or (0, 0, width, height)),
    )


def _file_digest(image_name: str) -> str:
    image_path = _resolve_image_path(image_name)
    digest = hashlib.sha256()
    with open(image_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CMKSwapImageLoaderPipe:
    """Dual image source for the closed CMK FaceSwap image module.

    The target image follows the complete CMK Image Load and Resize contract,
    including optional aspect-ratio crop. The source/reference image is loaded
    without crop or resize. IMAGE TARGET and IMAGE SOURCE remain independent
    authoritative pixel transports; PROCESS contains metadata only.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "RESOLUTION": (RESOLUTION_PRESETS, {"default": "SDXL 1152x832"}),
                "SWAP DIMENSIONS": ("BOOLEAN", {"default": False}),
                "RESIZE METHOD": (UPSCALE_METHODS, {"default": "lanczos"}),
                "CROP": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "ON",
                        "label_off": "OFF",
                        "advanced": True,
                    },
                ),
                "CROP POSITION": (
                    CROP_POSITIONS,
                    {
                        "default": "center",
                        "advanced": True,
                    },
                ),
                "IMAGE PAIR": (
                    "STRING",
                    {
                        "default": _IMAGE_PAIR_DEFAULT,
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "CMK_PIPE",
        "IMAGE",
        "IMAGE",
        "CMK_LOG_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = (
        "PROCESS",
        "IMAGE TARGET",
        "IMAGE SOURCE",
        "LOG",
        "diagnostic",
    )
    FUNCTION = "load_swap_images"
    CATEGORY = "CMK/Flow/Input"

    def load_swap_images(self, **inputs):
        target_name, source_name = _parse_image_pair(inputs.get("IMAGE PAIR"))
        resolution = str(
            inputs.get("RESOLUTION", "SDXL 1152x832") or "SDXL 1152x832"
        )
        swap_dimensions = bool(inputs.get("SWAP DIMENSIONS", False))
        resize_method = str(inputs.get("RESIZE METHOD", "lanczos") or "lanczos")
        crop_enabled = bool(inputs.get("CROP", False))
        crop_position = str(inputs.get("CROP POSITION", "center") or "center").lower()
        if crop_position not in CROP_POSITIONS:
            crop_position = "center"

        target_path = _resolve_image_path(target_name)
        source_path = _resolve_image_path(source_name)

        target_probe_w, target_probe_h, target_format = _probe_image(target_path)
        source_probe_w, source_probe_h, source_format = _probe_image(source_path)

        target_width, target_height = parse_resolution(
            resolution,
            fallback_width=int(target_probe_w or 1024),
            fallback_height=int(target_probe_h or 1024),
        )
        if swap_dimensions:
            target_width, target_height = target_height, target_width

        target_loaded, target_file_w, target_file_h, crop_box = _load_rgb_frames(
            target_path,
            target_width=int(target_width),
            target_height=int(target_height),
            crop_enabled=crop_enabled,
            crop_position=crop_position,
        )
        target_pre_resize_w, target_pre_resize_h = get_image_size(target_loaded)
        target_image = resize_image_tensor(
            target_loaded,
            int(target_width),
            int(target_height),
            resize_method,
        )

        source_image, source_file_w, source_file_h, _ = _load_rgb_frames(source_path)

        crop_left, crop_top, crop_right, crop_bottom = [int(value) for value in crop_box]
        crop_width = max(0, crop_right - crop_left)
        crop_height = max(0, crop_bottom - crop_top)
        target_frames = int(target_image.shape[0])
        source_frames = int(source_image.shape[0])

        process = {
            "width": int(target_width),
            "height": int(target_height),
            "source_width": int(target_file_w or target_probe_w or target_width),
            "source_height": int(target_file_h or target_probe_h or target_height),
            "target_width": int(target_width),
            "target_height": int(target_height),
            "resolution": resolution,
            "swap_dimensions": swap_dimensions,
            "upscale_method": resize_method,
            "crop_enabled": crop_enabled,
            "crop_position": crop_position,
            "crop_left": crop_left,
            "crop_top": crop_top,
            "crop_right": crop_right,
            "crop_bottom": crop_bottom,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "filename_string": target_name,
            "file_name": target_name,
            "target_filename": target_name,
            "target_source_width": int(target_file_w or target_probe_w),
            "target_source_height": int(target_file_h or target_probe_h),
            "target_frames": target_frames,
            "source_filename": source_name,
            "reference_filename": source_name,
            "reference_width": int(source_file_w or source_probe_w),
            "reference_height": int(source_file_h or source_probe_h),
            "reference_frames": source_frames,
            "pipe_origin": "CMK Swap Image Loader -Pipe-",
        }

        log_lines = [
            f"TARGET FILE     : {target_name}",
            f"TARGET SOURCE   : {int(target_file_w or target_probe_w)} × {int(target_file_h or target_probe_h)}",
            f"TARGET OUTPUT   : {int(target_width)} × {int(target_height)}",
            f"TARGET FRAMES   : {target_frames}",
            f"TARGET FORMAT   : {target_format}",
            f"RESIZE METHOD   : {resize_method}",
            f"SWAP DIMENSIONS : {'ON' if swap_dimensions else 'OFF'}",
            f"TARGET CROP     : {'ON' if crop_enabled else 'OFF'}",
        ]
        if crop_enabled:
            log_lines.extend(
                [
                    f"CROP POSITION   : {crop_position}",
                    f"CROP SIZE       : {crop_width} × {crop_height}",
                    f"CROP BOX        : {crop_left}, {crop_top}, {crop_right}, {crop_bottom}",
                ]
            )
        log_lines.extend(
            [
                f"SOURCE FILE     : {source_name}",
                f"SOURCE SIZE     : {int(source_file_w or source_probe_w)} × {int(source_file_h or source_probe_h)}",
                f"SOURCE FRAMES   : {source_frames}",
                f"SOURCE FORMAT   : {source_format}",
                "SOURCE RESIZE   : NONE",
                "SOURCE CROP     : NONE",
            ]
        )

        log_pipe = cmk_add_block(
            {
                "blocks": [],
                "filename_string": target_name,
                "file_name": target_name,
                "target_filename": target_name,
                "source_filename": source_name,
            },
            "Swap Image Loader",
            1,
            log_lines,
            True,
        )

        summary = "\n".join(log_lines)
        diagnostic = make_diagnostic_payload(
            title="Swap Image Loader -Pipe-",
            node="CMK Swap Image Loader -Pipe-",
            previews=[target_image, source_image],
            stages=[
                {
                    "title": "TARGET",
                    "subtitle": f"{int(target_width)} × {int(target_height)}",
                    "image": target_image,
                },
                {
                    "title": "SOURCE",
                    "subtitle": f"{int(source_file_w or source_probe_w)} × {int(source_file_h or source_probe_h)}",
                    "image": source_image,
                },
            ],
            summary=summary,
            details=summary,
            mode="Target Load + Crop + Resize / Source Load"
            if crop_enabled
            else "Target Load + Resize / Source Load",
            metadata={
                "target_filename": target_name,
                "target_source_width": int(target_file_w or target_probe_w),
                "target_source_height": int(target_file_h or target_probe_h),
                "target_pre_resize_width": int(target_pre_resize_w or crop_width),
                "target_pre_resize_height": int(target_pre_resize_h or crop_height),
                "target_width": int(target_width),
                "target_height": int(target_height),
                "target_frames": target_frames,
                "target_format": target_format,
                "resolution": resolution,
                "swap_dimensions": swap_dimensions,
                "resize_method": resize_method,
                "crop_enabled": crop_enabled,
                "crop_position": crop_position,
                "crop_box": [crop_left, crop_top, crop_right, crop_bottom],
                "source_filename": source_name,
                "source_width": int(source_file_w or source_probe_w),
                "source_height": int(source_file_h or source_probe_h),
                "source_frames": source_frames,
                "source_format": source_format,
                "source_resized": False,
                "source_cropped": False,
            },
        )

        return process, target_image, source_image, log_pipe, diagnostic

    @classmethod
    def IS_CHANGED(cls, **inputs):
        try:
            target_name, source_name = _parse_image_pair(inputs.get("IMAGE PAIR"))
            digest = hashlib.sha256()
            digest.update(_file_digest(target_name).encode("ascii"))
            digest.update(b"\0")
            digest.update(_file_digest(source_name).encode("ascii"))
            return digest.hexdigest()
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **inputs):
        try:
            target_name, source_name = _parse_image_pair(inputs.get("IMAGE PAIR"))
        except Exception as exc:
            return str(exc)

        for label, image_name in (("TARGET IMAGE", target_name), ("SOURCE IMAGE", source_name)):
            try:
                if not folder_paths.exists_annotated_filepath(image_name):
                    return f"Invalid {label.lower()} file: {image_name}"
            except Exception:
                image_path = os.path.join(folder_paths.get_input_directory(), image_name)
                if not os.path.isfile(image_path):
                    return f"Invalid {label.lower()} file: {image_name}"
        return True
