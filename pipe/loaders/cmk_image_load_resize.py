from __future__ import annotations

import hashlib
import os

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


CROP_POSITIONS = ["center", "top", "bottom", "left", "right"]


def calculate_crop_box(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    position: str = "center",
) -> tuple[int, int, int, int]:
    """Return an aspect-ratio crop box anchored at the requested position.

    The crop always remains inside the source image. Positions that cannot
    affect the currently cropped axis fall back to centering on that axis:
    ``left``/``right`` apply to a horizontal crop, while ``top``/``bottom``
    apply to a vertical crop.
    """
    source_width = max(1, int(source_width))
    source_height = max(1, int(source_height))
    target_width = max(1, int(target_width))
    target_height = max(1, int(target_height))
    position = str(position or "center").strip().lower()
    if position not in CROP_POSITIONS:
        position = "center"

    source_ratio = source_width / source_height
    target_ratio = target_width / target_height

    if abs(source_ratio - target_ratio) <= 1e-9:
        return 0, 0, source_width, source_height

    if source_ratio > target_ratio:
        # Source is wider than the requested target aspect ratio.
        crop_width = int(round(source_height * target_ratio))
        crop_width = max(1, min(source_width, crop_width))
        excess = source_width - crop_width
        if position == "left":
            left = 0
        elif position == "right":
            left = excess
        else:
            left = excess // 2
        return left, 0, left + crop_width, source_height

    # Source is taller than the requested target aspect ratio.
    crop_height = int(round(source_width / target_ratio))
    crop_height = max(1, min(source_height, crop_height))
    excess = source_height - crop_height
    if position == "top":
        top = 0
    elif position == "bottom":
        top = excess
    else:
        top = excess // 2
    return 0, top, source_width, top + crop_height


class CMKImageLoadAndResizePipe:
    """Compact standalone image source for pixel-based CMK modules.

    Public contract:
        image file + resize/crop parameters -> PROCESS + IMAGE + LOG + diagnostic

    IMAGE is the only authoritative pixel transport. PROCESS contains only
    source/target/crop metadata required by downstream CMK Prepare nodes. This
    node deliberately provides no mask, prompt, LoRA, inpaint, outpaint or
    latent preparation.
    """

    @classmethod
    def _available_images(cls):
        input_dir = folder_paths.get_input_directory()
        try:
            files = [
                name
                for name in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, name))
            ]
        except Exception:
            files = []
        return sorted(files)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE": (cls._available_images(), {"image_upload": True}),
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
            }
        }

    RETURN_TYPES = ("CMK_PIPE", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("PROCESS", "IMAGE", "LOG", "diagnostic")
    FUNCTION = "load_and_resize"
    CATEGORY = "CMK/Flow/Input"

    @staticmethod
    def _resolve_image_path(image: str) -> str:
        try:
            return folder_paths.get_annotated_filepath(image)
        except Exception:
            return os.path.join(folder_paths.get_input_directory(), image)

    @staticmethod
    def _probe_image(image_path: str) -> tuple[int, int, str]:
        with Image.open(image_path) as img:
            source_format = str(img.format or "unknown")
            img.seek(0)
            frame = ImageOps.exif_transpose(img.copy())
            width, height = frame.size
        return int(width), int(height), source_format

    @staticmethod
    def _load_frames(
        image_path: str,
        *,
        target_width: int,
        target_height: int,
        crop_enabled: bool,
        crop_position: str,
    ):
        output_images = []
        source_width = None
        source_height = None
        first_crop_box = None

        with Image.open(image_path) as img:
            for frame in ImageSequence.Iterator(img):
                frame = ImageOps.exif_transpose(frame)
                frame_width, frame_height = frame.size

                if source_width is None or source_height is None:
                    source_width, source_height = frame_width, frame_height

                if crop_enabled:
                    crop_box = calculate_crop_box(
                        frame_width,
                        frame_height,
                        target_width,
                        target_height,
                        crop_position,
                    )
                    frame = frame.crop(crop_box)
                    if first_crop_box is None:
                        first_crop_box = crop_box
                elif first_crop_box is None:
                    first_crop_box = (0, 0, frame_width, frame_height)

                rgb = frame.convert("RGB")
                array = np.asarray(rgb, dtype=np.float32) / 255.0
                output_images.append(torch.from_numpy(array)[None, ...])

        if not output_images:
            raise RuntimeError(
                "CMK Image Load and Resize -Pipe-: no image frames could be loaded"
            )

        return (
            torch.cat(output_images, dim=0),
            int(source_width or 0),
            int(source_height or 0),
            tuple(first_crop_box or (0, 0, int(source_width or 0), int(source_height or 0))),
        )

    def load_and_resize(self, **inputs):
        image_name = str(inputs.get("IMAGE", "") or "")
        resolution = str(inputs.get("RESOLUTION", "SDXL 1152x832") or "SDXL 1152x832")
        swap_dimensions = bool(inputs.get("SWAP DIMENSIONS", False))
        resize_method = str(inputs.get("RESIZE METHOD", "lanczos") or "lanczos")
        crop_enabled = bool(inputs.get("CROP", False))
        crop_position = str(inputs.get("CROP POSITION", "center") or "center").lower()
        if crop_position not in CROP_POSITIONS:
            crop_position = "center"

        image_path = self._resolve_image_path(image_name)
        probed_width, probed_height, source_format = self._probe_image(image_path)

        target_width, target_height = parse_resolution(
            resolution,
            fallback_width=int(probed_width or 1024),
            fallback_height=int(probed_height or 1024),
        )
        if swap_dimensions:
            target_width, target_height = target_height, target_width

        loaded_image, file_width, file_height, crop_box = self._load_frames(
            image_path,
            target_width=int(target_width),
            target_height=int(target_height),
            crop_enabled=crop_enabled,
            crop_position=crop_position,
        )

        pre_resize_width, pre_resize_height = get_image_size(loaded_image)
        resized_image = resize_image_tensor(
            loaded_image,
            int(target_width),
            int(target_height),
            resize_method,
        )

        crop_left, crop_top, crop_right, crop_bottom = [int(value) for value in crop_box]
        crop_width = max(0, crop_right - crop_left)
        crop_height = max(0, crop_bottom - crop_top)

        process = {
            "width": int(target_width),
            "height": int(target_height),
            "source_width": int(file_width or probed_width or target_width),
            "source_height": int(file_height or probed_height or target_height),
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
            "filename_string": image_name,
            "file_name": image_name,
            "pipe_origin": "CMK Image Load and Resize -Pipe-",
        }

        frame_count = int(resized_image.shape[0])
        log_lines = [
            f"FILE NAME       : {image_name}",
            f"SOURCE SIZE     : {int(file_width or probed_width)} × {int(file_height or probed_height)}",
            f"TARGET SIZE     : {int(target_width)} × {int(target_height)}",
            f"FRAMES          : {frame_count}",
            f"FORMAT          : {source_format}",
            f"RESIZE METHOD   : {resize_method}",
            f"SWAP DIMENSIONS : {'ON' if swap_dimensions else 'OFF'}",
            f"CROP            : {'ON' if crop_enabled else 'OFF'}",
        ]
        if crop_enabled:
            log_lines.extend(
                [
                    f"CROP POSITION   : {crop_position}",
                    f"CROP SIZE       : {crop_width} × {crop_height}",
                    f"CROP BOX        : {crop_left}, {crop_top}, {crop_right}, {crop_bottom}",
                ]
            )

        log_pipe = cmk_add_block(
            {
                "blocks": [],
                "filename_string": image_name,
                "file_name": image_name,
            },
            "Image Load and Resize",
            1,
            log_lines,
            True,
        )

        summary = "\n".join(log_lines)
        diagnostic = make_diagnostic_payload(
            title="Image Load and Resize -Pipe-",
            node="CMK Image Load and Resize -Pipe-",
            previews=[resized_image],
            summary=summary,
            details=summary,
            mode="Load + Crop + Resize" if crop_enabled else "Load + Resize",
            metadata={
                "source_width": int(file_width or probed_width),
                "source_height": int(file_height or probed_height),
                "pre_resize_width": int(pre_resize_width or crop_width),
                "pre_resize_height": int(pre_resize_height or crop_height),
                "target_width": int(target_width),
                "target_height": int(target_height),
                "frames": frame_count,
                "format": source_format,
                "resolution": resolution,
                "swap_dimensions": swap_dimensions,
                "resize_method": resize_method,
                "crop_enabled": crop_enabled,
                "crop_position": crop_position,
                "crop_box": [crop_left, crop_top, crop_right, crop_bottom],
                "crop_width": crop_width,
                "crop_height": crop_height,
            },
        )

        return process, resized_image, log_pipe, diagnostic

    @classmethod
    def IS_CHANGED(cls, **inputs):
        image_name = str(inputs.get("IMAGE", "") or "")
        try:
            image_path = folder_paths.get_annotated_filepath(image_name)
            with open(image_path, "rb") as handle:
                return hashlib.sha256(handle.read()).hexdigest()
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **inputs):
        image_name = str(inputs.get("IMAGE", "") or "")
        try:
            if not folder_paths.exists_annotated_filepath(image_name):
                return f"Invalid image file: {image_name}"
        except Exception:
            image_path = os.path.join(folder_paths.get_input_directory(), image_name)
            if not os.path.isfile(image_path):
                return f"Invalid image file: {image_name}"
        return True
