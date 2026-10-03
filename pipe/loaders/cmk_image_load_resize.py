from __future__ import annotations

import hashlib
import os
from pathlib import Path

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
    resize_mask_tensor,
)
from ...utils.cmk_diagnostic import make_diagnostic_payload


CROP_POSITIONS = ["center", "top", "bottom", "left", "right"]
CMK_PACKAGED_REFERENCES = {
    f"CMK Package · {filename}": filename
    for filename in (
        "face_reference.png",
        "controlnet_reference.png",
        "detailer_reference.png",
        "face_identity_reference.png",
        "face_reference2.png",
        "faceswap_reference.png",
        "inpaint_reference.png",
        "inpaint_reference2.png",
        "inpaint_reference3.png",
        "mask_detailer_reference.png",
        "instantid_reference.png",
        "portrait_reference_00002.png",
        "remove_refrence.png",
    )
}
_CMK_REFERENCE_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "references"


def _packaged_reference_path(image: str):
    filename = CMK_PACKAGED_REFERENCES.get(str(image or ""))
    if filename is None:
        return None
    path = (_CMK_REFERENCE_ASSETS / filename).resolve()
    try:
        path.relative_to(_CMK_REFERENCE_ASSETS.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


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
        image file + resize/crop parameters
        -> optional MODEL SDXL (opt) input, then MODEL + neutral PROCESS + IMAGE + LOG
           + diagnostic + MASK + image_file

    IMAGE is the only authoritative pixel transport. PROCESS contains only
    source/target/crop metadata required by downstream CMK Prepare nodes, but
    carries the family-neutral result contract required by standalone
    postprocessors. An
    optionally connected MODEL SDXL (opt) is passed
    through unchanged as the ordinary downstream MODEL. Without it, pixel-only
    modules use PROCESS, IMAGE and LOG and no artificial model placeholder is
    created. A mask painted in the native image editor is preserved through
    the same crop/resize transform and exposed both in PROCESS and as MASK.
    image_file carries only the selected ComfyUI file reference so identity
    modules can load the same source through one visible wire. This node
    provides no prompt, LoRA, inpaint or latent preparation.
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
        return list(CMK_PACKAGED_REFERENCES) + sorted(files)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    cls._available_images(),
                    {"image_upload": True, "label": "IMAGE"},
                ),
                "RESOLUTION": (RESOLUTION_PRESETS, {"default": "SDXL 1152x832"}),
                "SWAP DIMENSIONS": ("BOOLEAN", {"default": False}),
                "RESIZE METHOD": (UPSCALE_METHODS, {"default": "lanczos"}),
                "CROP": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "ON",
                        "label_off": "OFF",
                    },
                ),
                "CROP POSITION": (
                    CROP_POSITIONS,
                    {
                        "default": "center",
                    },
                ),
            },
            "optional": {
                "MODEL SDXL (opt)": ("CMK_MODEL_PIPE",),
            },
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "CMK_RESULT_PROCESS",
        "IMAGE",
        "CMK_LOG_PIPE",
        "CMK_DIAGNOSTIC",
        "MASK",
        "STRING",
    )
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "diagnostic", "MASK", "image_file")
    FUNCTION = "load_and_resize"
    CATEGORY = "CMK/Toolbox/Image"

    @staticmethod
    def _resolve_image_path(image: str) -> str:
        packaged_path = _packaged_reference_path(image)
        if packaged_path is not None:
            return str(packaged_path)
        try:
            path = Path(folder_paths.get_annotated_filepath(image))
        except Exception:
            path = Path(folder_paths.get_input_directory()) / image

        # ComfyUI's mask editor may hand custom upload nodes the composited
        # preview (painted-masked), whose RGB pixels contain the visible mask
        # overlay. The sibling clipspace-mask file carries the untouched RGB
        # source and the exact same alpha mask and is therefore authoritative.
        prefix = "clipspace-painted-masked-"
        if path.name.startswith(prefix):
            clean_name = "clipspace-mask-" + path.name[len(prefix):]
            clean_path = path.with_name(clean_name)
            if clean_path.is_file():
                path = clean_path
        return str(path)

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
        output_masks = []
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

                if "A" in frame.getbands():
                    alpha = np.asarray(frame.getchannel("A"), dtype=np.float32) / 255.0
                    mask_array = 1.0 - alpha
                else:
                    mask_array = np.zeros((frame.height, frame.width), dtype=np.float32)
                rgb = frame.convert("RGB")
                array = np.asarray(rgb, dtype=np.float32) / 255.0
                output_images.append(torch.from_numpy(array)[None, ...])
                output_masks.append(torch.from_numpy(mask_array)[None, ...])

        if not output_images:
            raise RuntimeError(
                "CMK Image Load and Resize -Pipe-: no image frames could be loaded"
            )

        return (
            torch.cat(output_images, dim=0),
            torch.cat(output_masks, dim=0),
            int(source_width or 0),
            int(source_height or 0),
            tuple(first_crop_box or (0, 0, int(source_width or 0), int(source_height or 0))),
        )

    def load_and_resize(self, **inputs):
        model_sdxl = inputs.get("MODEL SDXL (opt)")
        image_name = str(inputs.get("image", inputs.get("IMAGE", "")) or "")
        resolution = str(inputs.get("RESOLUTION", "SDXL 1152x832") or "SDXL 1152x832")
        swap_dimensions = bool(inputs.get("SWAP DIMENSIONS", False))
        resize_method = str(inputs.get("RESIZE METHOD", "lanczos") or "lanczos")
        crop_enabled = bool(inputs.get("CROP", True))
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

        loaded_image, loaded_mask, file_width, file_height, crop_box = self._load_frames(
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
        resized_mask = resize_mask_tensor(
            loaded_mask,
            int(target_width),
            int(target_height),
        )

        crop_left, crop_top, crop_right, crop_bottom = [int(value) for value in crop_box]
        crop_width = max(0, crop_right - crop_left)
        crop_height = max(0, crop_bottom - crop_top)

        process = {
            "image": resized_image,
            "image_original": resized_image,
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
            "result_contract": "family_neutral",
            "source_model_family": "image",
            "mask": resized_mask,
            "mask_original": loaded_mask,
            "boolean_inpaint_mode": bool(float(resized_mask.max().detach().cpu()) > 0.0),
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

        if model_sdxl is not None:
            if not isinstance(model_sdxl, dict):
                raise TypeError("CMK Load Image: MODEL SDXL must be a CMK model pipe")
            if str(model_sdxl.get("model_family", "sdxl")).strip().lower() != "sdxl":
                raise ValueError("CMK Load Image accepts only MODEL SDXL")
        return model_sdxl, process, resized_image, log_pipe, diagnostic, resized_mask, image_name

    @classmethod
    def IS_CHANGED(cls, **inputs):
        image_name = str(inputs.get("image", inputs.get("IMAGE", "")) or "")
        try:
            packaged_path = _packaged_reference_path(image_name)
            image_path = (
                str(packaged_path)
                if packaged_path is not None
                else folder_paths.get_annotated_filepath(image_name)
            )
            with open(image_path, "rb") as handle:
                return hashlib.sha256(handle.read()).hexdigest()
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **inputs):
        image_name = str(inputs.get("image", inputs.get("IMAGE", "")) or "")
        if _packaged_reference_path(image_name) is not None:
            return True
        try:
            if not folder_paths.exists_annotated_filepath(image_name):
                return f"Invalid image file: {image_name}"
        except Exception:
            image_path = os.path.join(folder_paths.get_input_directory(), image_name)
            if not os.path.isfile(image_path):
                return f"Invalid image file: {image_name}"
        return True
