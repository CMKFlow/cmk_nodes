from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths

from ..cmk_log_pipe import cmk_add_block


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


class CMKLoadImage:
    """CMK pipe-only image loader.

    This node is the pixel-workflow start point for CMK pipes. It mirrors the
    native ComfyUI image picker, but exposes only CMK transport outputs:

    - PROCESS: CMK process context for downstream CMK modules
    - IMAGE: native image payload for the open image-processing layer
    - MASK: alpha-derived mask matching the loaded image
    - FILENAME_STRING: selected source filename for Create Image or save modules
    - LOG: documentation context containing source metadata

    An optional opt_LOG input lets the loader append its source-image block to
    an existing workflow log instead of starting a separate log chain.

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
        files = list(CMK_PACKAGED_REFERENCES) + sorted(files)
        return {
            "required": {
                "image": (files, {"image_upload": True}),
            },
            "optional": {
                "opt_LOG": ("CMK_LOG_PIPE",),
            },
        }

    RETURN_TYPES = ("CMK_PIPE", "IMAGE", "MASK", "STRING", "CMK_LOG_PIPE")
    RETURN_NAMES = ("PROCESS", "IMAGE", "MASK", "FILENAME_STRING", "LOG")
    FUNCTION = "load_image"
    CATEGORY = "CMK/Flow/Input"

    def _resolve_image_path(self, image: str) -> str:
        packaged_path = _packaged_reference_path(image)
        if packaged_path is not None:
            return str(packaged_path)
        try:
            return folder_paths.get_annotated_filepath(image)
        except Exception:
            return os.path.join(folder_paths.get_input_directory(), image)

    @staticmethod
    def _preview_descriptor(image: str) -> dict[str, str]:
        """Describe the image that was actually supplied to this execution.

        A subgraph input can override the loader's persisted widget value.  The
        execution result must therefore drive the preview instead of that
        potentially stale widget.
        """
        value = str(image or "")
        if value in CMK_PACKAGED_REFERENCES:
            return {"filename": value, "subfolder": "", "type": "input"}

        clean_name, base_dir = folder_paths.annotated_filepath(value)
        image_type = "input"
        if base_dir == folder_paths.get_output_directory():
            image_type = "output"
        elif base_dir == folder_paths.get_temp_directory():
            image_type = "temp"
        return {
            "filename": os.path.basename(clean_name),
            "subfolder": os.path.dirname(clean_name),
            "type": image_type,
        }

    def load_image(self, image, opt_LOG=None):
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

        log_base = dict(opt_LOG) if isinstance(opt_LOG, dict) else {"blocks": []}
        log_base["filename_string"] = filename_string
        log_pipe = cmk_add_block(
            log_base,
            "Load Image",
            1,
            log_lines,
            True,
        )

        return {
            "ui": {"images": [self._preview_descriptor(filename_string)]},
            "result": (pipe, loaded_image, loaded_mask, filename_string, log_pipe),
        }

    @classmethod
    def IS_CHANGED(cls, image):
        try:
            packaged_path = _packaged_reference_path(image)
            image_path = str(packaged_path) if packaged_path is not None else folder_paths.get_annotated_filepath(image)
            with open(image_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if _packaged_reference_path(image) is not None:
            return True
        try:
            if not folder_paths.exists_annotated_filepath(image):
                return f"Invalid image file: {image}"
        except Exception:
            image_path = os.path.join(folder_paths.get_input_directory(), image)
            if not os.path.isfile(image_path):
                return f"Invalid image file: {image}"
        return True
