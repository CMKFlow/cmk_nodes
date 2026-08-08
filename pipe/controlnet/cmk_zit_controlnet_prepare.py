from __future__ import annotations

import folder_paths
from comfy_execution.graph_utils import ExecutionBlocker

from ...nodes.controlnet.controlnet import (
    CONTROLNET_IMAGE_SOURCES,
    _apply_mask_to_image,
    _empty_controlnet_ui_placeholder,
    _get_input_files,
    _load_image_from_input,
    _tensor_image_to_temp_ui,
)
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ..cmk_log_pipe import cmk_add_block
from ..cmk_sampler_prepare import _call_node_kwargs, _unwrap_node_output


DEFAULT_ZIT_CONTROLNET_PATCH = "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"


def _patch_choices() -> list[str]:
    values = list(folder_paths.get_filename_list("model_patches"))
    if DEFAULT_ZIT_CONTROLNET_PATCH in values:
        values.remove(DEFAULT_ZIT_CONTROLNET_PATCH)
        values.insert(0, DEFAULT_ZIT_CONTROLNET_PATCH)
    return values or [DEFAULT_ZIT_CONTROLNET_PATCH]


class CMKZITControlNetPreparePipe:
    """Attach lazy ZIT ControlNet preparation data to the ZIT process pipe."""

    @classmethod
    def INPUT_TYPES(cls):
        input_files = _get_input_files() or [""]
        return {
            "required": {
                "PROCESS": ("CMK_PROCESS_Z_IMAGE",),
                "IMAGE": ("IMAGE", {"lazy": True}),
                "ENABLE": ("BOOLEAN", {"default": False}),
                "IMAGE SOURCE": (
                    CONTROLNET_IMAGE_SOURCES,
                    {"default": "Reference Image"},
                ),
                "REFERENCE IMAGE": (input_files,),
                "APPLY MASK": ("BOOLEAN", {"default": False}),
                "STRENGTH": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "resolution": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "advanced": True,
                        "label": "resolution",
                    },
                ),
                "low_threshold": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "advanced": True,
                    },
                ),
                "high_threshold": (
                    "FLOAT",
                    {
                        "default": 0.32,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "advanced": True,
                    },
                ),
                "MODEL PATCH": (
                    _patch_choices(),
                    {
                        "default": DEFAULT_ZIT_CONTROLNET_PATCH,
                        "advanced": True,
                    },
                ),
            },
            "optional": {
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
            },
        }

    RETURN_TYPES = (
        "CMK_PROCESS_Z_IMAGE",
        "CMK_LOG_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("PROCESS", "LOG", "diagnostic")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Flow/Process/ZIT"
    OUTPUT_NODE = True

    def check_lazy_status(
        self,
        PROCESS=None,
        **kwargs,
    ):
        if PROCESS is None:
            return ["PROCESS"]
        if not isinstance(PROCESS, dict) or not PROCESS.get("family_active", True):
            return []
        if not bool(kwargs.get("ENABLE", False)):
            return []
        requested = []
        if kwargs.get("IMAGE SOURCE", "Reference Image") == "Base Image" and kwargs.get("IMAGE") is None:
            requested.append("IMAGE")
        if kwargs.get("LOG") is None:
            requested.append("LOG")
        return requested

    def prepare(self, PROCESS=None, **kwargs):
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK ZIT ControlNet requires PROCESS ZIT.")
        if str(PROCESS.get("model_family", "z_image_turbo")).lower() != "z_image_turbo":
            raise ValueError("CMK ZIT ControlNet accepts only PROCESS ZIT.")

        enabled = bool(kwargs.get("ENABLE", False))
        if not PROCESS.get("family_active", True):
            blocked = ExecutionBlocker(None)
            return (PROCESS, blocked, blocked)

        process = dict(PROCESS)
        if not enabled:
            process.update(
                {
                    "boolean_controlnet_enable": False,
                    "zit_controlnet_image": None,
                    "zit_controlnet_patch": None,
                }
            )
            blocked = ExecutionBlocker(None)
            return {
                "ui": {"images": _empty_controlnet_ui_placeholder()},
                "result": (process, kwargs.get("LOG"), blocked),
            }

        image_source = str(kwargs.get("IMAGE SOURCE", "Reference Image"))
        reference_image = str(kwargs.get("REFERENCE IMAGE", ""))
        image = (
            kwargs.get("IMAGE")
            if image_source == "Base Image"
            else _load_image_from_input(reference_image)
        )
        if image is None:
            raise ValueError(
                "CMK ZIT ControlNet requires a valid base or reference image when enabled."
            )
        if bool(kwargs.get("APPLY MASK", False)):
            image = _apply_mask_to_image(image, PROCESS.get("mask"))

        image = _unwrap_node_output(
            _call_node_kwargs(
                ("ImageScaleToMaxDimension",),
                image=image,
                upscale_method="lanczos",
                largest_size=int(kwargs.get("resolution", 1024)),
            )
        )
        image = _unwrap_node_output(
            _call_node_kwargs(
                ("Canny",),
                image=image,
                low_threshold=float(kwargs.get("low_threshold", 0.1)),
                high_threshold=float(kwargs.get("high_threshold", 0.32)),
            )
        )

        strength = float(kwargs.get("STRENGTH", 1.0))
        patch_name = str(
            kwargs.get("MODEL PATCH", DEFAULT_ZIT_CONTROLNET_PATCH)
        ).strip()
        if not patch_name:
            raise ValueError("CMK ZIT ControlNet requires a MODEL PATCH.")

        process.update(
            {
                "boolean_controlnet_enable": True,
                "zit_controlnet_image": image,
                "zit_controlnet_strength": strength,
                "zit_controlnet_patch": patch_name,
                "zit_controlnet_type": "canny",
                "generation_mode": "controlnet",
            }
        )
        lines = [
            "STATUS          : PREPARED",
            "MODEL FAMILY    : ZIT",
            "CONTROL TYPE    : CANNY",
            f"STRENGTH        : {strength:g}",
            f"MODEL PATCH     : {patch_name}",
            f"IMAGE SOURCE    : {image_source}",
            f"RESOLUTION      : {int(kwargs.get('resolution', 1024))}",
        ]
        log = cmk_add_block(kwargs.get("LOG"), "ZIT ControlNet", 30, lines, True)
        diagnostic = make_diagnostic_payload(
            title="ZIT ControlNet",
            node="CMK ZIT ControlNet Prepare -Pipe-",
            previews=[image],
            summary=f"Canny | Strength {strength:g}",
            details="\n".join(lines),
            mode="ControlNet",
            metadata={
                "model_family": "z_image_turbo",
                "control_type": "canny",
                "strength": strength,
                "model_patch": patch_name,
            },
        )
        return {
            "ui": {"images": _tensor_image_to_temp_ui(image, "cmk_zit_controlnet")},
            "result": (process, log, diagnostic),
        }
