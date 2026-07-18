from __future__ import annotations

import hashlib
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths

from ..cmk_log_pipe import cmk_add_block


class CMKLoadImage:
    """CMK pipe-only image loader.

    This node is the pixel-workflow start point for CMK pipes. It mirrors the
    native ComfyUI image picker, but exposes only CMK transport outputs:

    - PROCESS: CMK process context for downstream CMK modules
    - IMAGE: native image payload for the open image-processing layer
    - MASK: alpha-derived mask matching the loaded image
    - FILENAME_STRING: selected source filename for Create Image or save modules
    - LOG: documentation context containing source metadata

    IMAGE and MASK remain available inside PROCESS as well. Their explicit
    outputs allow direct use by Create Image and native ComfyUI nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        try:
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        except Exception:
            files = []
        files = sorted(files)
        return {
            "required": {
                "image": (files, {"image_upload": True}),
            }
        }

    RETURN_TYPES = ("CMK_PIPE", "IMAGE", "MASK", "STRING", "CMK_LOG_PIPE")
    RETURN_NAMES = ("PROCESS", "IMAGE", "MASK", "FILENAME_STRING", "LOG")
    FUNCTION = "load_image"
    CATEGORY = "CMK/Flow/Input"

    def _resolve_image_path(self, image: str) -> str:
        try:
            return folder_paths.get_annotated_filepath(image)
        except Exception:
            return os.path.join(folder_paths.get_input_directory(), image)

    def load_image(self, image):
        image_path = self._resolve_image_path(image)

        output_images = []
        output_masks = []
        source_width = None
        source_height = None

        with Image.open(image_path) as img:
            source_format = img.format or "unknown"
            frame_count = 0

            for frame in ImageSequence.Iterator(img):
                frame_count += 1
                frame = ImageOps.exif_transpose(frame)

                if source_width is None or source_height is None:
                    source_width, source_height = frame.size

                if "A" in frame.getbands():
                    alpha = frame.getchannel("A")
                    mask = np.array(alpha).astype(np.float32) / 255.0
                    mask = 1.0 - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((frame.size[1], frame.size[0]), dtype=torch.float32)

                rgb = frame.convert("RGB")
                arr = np.array(rgb).astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr)[None,]

                output_images.append(tensor)
                output_masks.append(mask.unsqueeze(0))

        if not output_images:
            raise RuntimeError("CMK Load Image -Pipe-: no image frames could be loaded")

        loaded_image = torch.cat(output_images, dim=0)
        loaded_mask = torch.cat(output_masks, dim=0)

        width = int(loaded_image.shape[2])
        height = int(loaded_image.shape[1])
        filename_string = str(image)

        pipe = {
            "image": loaded_image,
            "image_original": loaded_image,
            "mask": loaded_mask,
            "mask_original": loaded_mask,
            "width": width,
            "height": height,
            "source_width": int(source_width or width),
            "source_height": int(source_height or height),
            "target_width": width,
            "target_height": height,
            "filename_string": filename_string,
            "file_name": filename_string,
            "pipe_origin": "CMK Load Image -Pipe-",
        }

        log_lines = [
            f"filename_string : {filename_string}",
            f"Resolution      : {width}x{height}",
            f"Frames          : {int(loaded_image.shape[0])}",
            f"Format          : {source_format}",
        ]

        log_pipe = cmk_add_block(
            {"blocks": [], "filename_string": filename_string},
            "Load Image",
            1,
            log_lines,
            True,
        )

        return (pipe, loaded_image, loaded_mask, filename_string, log_pipe)

    @classmethod
    def IS_CHANGED(cls, image):
        try:
            image_path = folder_paths.get_annotated_filepath(image)
            with open(image_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        try:
            if not folder_paths.exists_annotated_filepath(image):
                return f"Invalid image file: {image}"
        except Exception:
            image_path = os.path.join(folder_paths.get_input_directory(), image)
            if not os.path.isfile(image_path):
                return f"Invalid image file: {image}"
        return True
