from __future__ import annotations

import math

import comfy.utils
import numpy as np
import torch

from ...pipe.cmk_log_pipe import cmk_add_block
from ...pipe.cmk_module_boundary_cache import (
    _faceswap_disk_available,
    _load_diagnostic,
    _load_image_log,
    _retain_latest_boundary_entry,
    _save_diagnostic,
    _save_image_log,
)
from ...pipe.cmk_persistent_cache import build_node_fingerprint, write_status
from ...pipe.cmk_visual import empty_visual, register_provider
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...utils.cmk_timing import cmk_timed_call


METHOD_MAX_DIMENSION = "Scale to Max Dimension"
METHOD_MEGAPIXELS = "Scale to Megapixels"
METHOD_RESIZE_PAD = "Resize and Pad"
METHODS = (METHOD_MAX_DIMENSION, METHOD_MEGAPIXELS, METHOD_RESIZE_PAD)
INTERPOLATION_METHODS = ("area", "lanczos", "bilinear", "nearest-exact", "bicubic")
PADDING_COLORS = ("black", "white")


class CMKImageResize:
    """Family-neutral, model-free image resize module for CMK Flow."""

    CATEGORY = "CMK/Flow/PostProcess"
    FUNCTION = "resize"
    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "CMK_RESULT_PROCESS",
        "IMAGE",
        "CMK_LOG_PIPE",
        "CMK_VISUAL_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic")

    _CACHE_SCOPE = "image_resize_boundary"
    _CACHE_SCHEMA = "cmk_image_resize_boundary_v1"
    _NODE_TYPE = "CMKImageResize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PROCESS": ("CMK_RESULT_PROCESS",),
                "IMAGE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
                "SCALE ENABLE": ("BOOLEAN", {"default": True}),
                "scale_method": (
                    METHODS,
                    {"default": METHOD_MAX_DIMENSION, "advanced": True},
                ),
                "interpolation": (
                    INTERPOLATION_METHODS,
                    {"default": "lanczos", "advanced": True},
                ),
            },
            "optional": {
                "largest_size": (
                    "INT",
                    {
                        "default": 1600,
                        "min": 1,
                        "max": 16384,
                        "step": 1,
                        "advanced": True,
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.01,
                        "max": 16.0,
                        "step": 0.01,
                        "advanced": True,
                    },
                ),
                "target_width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 16384,
                        "step": 1,
                        "advanced": True,
                    },
                ),
                "target_height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 16384,
                        "step": 1,
                        "advanced": True,
                    },
                ),
                "padding_color": (
                    PADDING_COLORS,
                    {"default": "black", "advanced": True},
                ),
                "MODEL (opt)": ("CMK_MODEL_PIPE",),
                "VISUAL": ("CMK_VISUAL_PIPE",),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    @staticmethod
    def _validate_transport(process, image, log):
        if not isinstance(process, dict):
            raise TypeError("CMK Image Resize: PROCESS must be a CMK result process pipe")
        if getattr(image, "ndim", None) != 4:
            raise TypeError("CMK Image Resize: IMAGE must be a BHWC tensor")
        if int(image.shape[1]) < 1 or int(image.shape[2]) < 1:
            raise ValueError("CMK Image Resize: IMAGE dimensions must be positive")
        if not isinstance(log, dict):
            raise TypeError("CMK Image Resize: LOG must be a CMK log pipe")

    @staticmethod
    def _dimensions(image):
        return int(image.shape[2]), int(image.shape[1])

    @staticmethod
    def _scale(image, width: int, height: int, interpolation: str):
        samples = image.movedim(-1, 1)
        resized = comfy.utils.common_upscale(
            samples,
            max(1, int(width)),
            max(1, int(height)),
            interpolation,
            "disabled",
        )
        return resized.movedim(1, -1)

    @classmethod
    def _scale_to_max_dimension(cls, image, largest_size: int, interpolation: str):
        width, height = cls._dimensions(image)
        largest = max(1, int(largest_size))
        if height > width:
            target_width = round((width / height) * largest)
            target_height = largest
        elif width > height:
            target_width = largest
            target_height = round((height / width) * largest)
        else:
            target_width = target_height = largest
        return cls._scale(image, target_width, target_height, interpolation)

    @classmethod
    def _scale_to_megapixels(cls, image, megapixels: float, interpolation: str):
        width, height = cls._dimensions(image)
        total_pixels = float(megapixels) * 1024.0 * 1024.0
        scale_by = math.sqrt(total_pixels / float(width * height))
        target_width = round(width * scale_by)
        target_height = round(height * scale_by)
        return cls._scale(image, target_width, target_height, interpolation)

    @classmethod
    def _resize_and_pad(
        cls,
        image,
        target_width: int,
        target_height: int,
        interpolation: str,
        padding_color: str,
    ):
        width, height = cls._dimensions(image)
        output_width = max(1, int(target_width))
        output_height = max(1, int(target_height))
        scale = min(output_width / width, output_height / height)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        resized = cls._scale(image, resized_width, resized_height, interpolation)
        padded = image.new_full(
            (int(image.shape[0]), output_height, output_width, int(image.shape[3])),
            0.0 if padding_color == "black" else 1.0,
        )
        x_offset = (output_width - resized_width) // 2
        y_offset = (output_height - resized_height) // 2
        padded[
            :,
            y_offset:y_offset + resized_height,
            x_offset:x_offset + resized_width,
            :,
        ] = resized
        return padded

    @classmethod
    def _cache_key(cls, prompt, unique_id, method: str):
        inactive = {
            METHOD_MAX_DIMENSION: ("megapixels", "target_width", "target_height", "padding_color"),
            METHOD_MEGAPIXELS: ("largest_size", "target_width", "target_height", "padding_color"),
            METHOD_RESIZE_PAD: ("largest_size", "megapixels"),
        }[method]
        # MODEL, PROCESS and VISUAL are transport-only. LOG and IMAGE remain in
        # the fingerprint because they are materialized together at the module
        # boundary. Inactive method settings are deliberately excluded.
        excluded = (*inactive, "MODEL (opt)", "PROCESS", "VISUAL")
        return build_node_fingerprint(
            prompt,
            unique_id,
            (cls._NODE_TYPE,),
            cls._CACHE_SCHEMA,
            exclude_inputs=excluded,
            include_node_identity=True,
        )

    @classmethod
    def _run_selected(cls, method: str, image, **settings):
        interpolation = str(settings["interpolation"])
        if method == METHOD_MAX_DIMENSION:
            return cls._scale_to_max_dimension(
                image, settings["largest_size"], interpolation,
            )
        if method == METHOD_MEGAPIXELS:
            return cls._scale_to_megapixels(
                image, settings["megapixels"], interpolation,
            )
        if method == METHOD_RESIZE_PAD:
            return cls._resize_and_pad(
                image,
                settings["target_width"],
                settings["target_height"],
                interpolation,
                str(settings["padding_color"]),
            )
        raise ValueError(f"CMK Image Resize: unsupported scale method {method!r}")

    @staticmethod
    def _setting_lines(method: str, settings: dict):
        if method == METHOD_MAX_DIMENSION:
            return [f"LARGEST SIZE    : {int(settings['largest_size'])}"]
        if method == METHOD_MEGAPIXELS:
            return [f"MEGAPIXELS      : {float(settings['megapixels']):.2f}"]
        return [
            f"TARGET SIZE     : {int(settings['target_width'])} x {int(settings['target_height'])}",
            f"PADDING COLOR   : {str(settings['padding_color']).upper()}",
        ]

    @classmethod
    def _result_metadata(cls, enabled, method, settings, input_size, output_size, cache_hit=False):
        metadata = {
            "enabled": bool(enabled),
            "bypassed": not bool(enabled),
            "method": method if enabled else "Bypass",
            "interpolation": str(settings["interpolation"]),
            "input_width": input_size[0],
            "input_height": input_size[1],
            "output_width": output_size[0],
            "output_height": output_size[1],
            "cache_hit": bool(cache_hit),
        }
        if enabled:
            if method == METHOD_MAX_DIMENSION:
                metadata["largest_size"] = int(settings["largest_size"])
            elif method == METHOD_MEGAPIXELS:
                metadata["megapixels"] = float(settings["megapixels"])
            elif method == METHOD_RESIZE_PAD:
                metadata.update({
                    "target_width": int(settings["target_width"]),
                    "target_height": int(settings["target_height"]),
                    "padding_color": str(settings["padding_color"]),
                })
        return metadata

    @classmethod
    def _diagnostic(cls, source, output, enabled, method, settings, cache_hit=False):
        input_size = cls._dimensions(source)
        output_size = cls._dimensions(output)
        mode = method if enabled else "Bypass"
        lines = [
            f"Status        : {'Executed' if enabled else 'Disabled / passthrough'}",
            f"Method        : {mode}",
            f"Input         : {input_size[0]} x {input_size[1]}",
            f"Output        : {output_size[0]} x {output_size[1]}",
        ]
        if enabled:
            lines.extend(cls._setting_lines(method, settings))
            lines.append(f"Interpolation : {settings['interpolation']}")
        summary = "\n".join(lines)
        stages = [{"title": "01 Source", "subtitle": f"{input_size[0]} x {input_size[1]}", "image": source}]
        if enabled:
            stages.append({"title": "02 Resized", "subtitle": f"{output_size[0]} x {output_size[1]}", "image": output})
        return make_diagnostic_payload(
            title="CMK Flow · Image Resize",
            node=cls._NODE_TYPE,
            previews=[source, output] if enabled else [source],
            stages=stages,
            summary=summary,
            details=summary,
            mode=mode,
            metadata=cls._result_metadata(
                enabled, method, settings, input_size, output_size, cache_hit,
            ),
            warnings=[] if enabled else ["image resize disabled; input image passed through unchanged"],
            metrics={
                "input_megapixels": input_size[0] * input_size[1] / 1_000_000.0,
                "output_megapixels": output_size[0] * output_size[1] / 1_000_000.0,
            },
        )

    @classmethod
    def _visual(cls, visual, source, output, enabled, unique_id):
        result = empty_visual() if visual is None else visual
        if enabled:
            result = register_provider(
                result,
                module_instance_id=unique_id or "image-resize",
                module_type=cls._NODE_TYPE,
                module_label="Image Resize",
                sequence=80,
                channels={"before": source, "after": output},
                status="completed",
                live_node_id=unique_id,
                branch="result",
                stage_key="result.image_resize",
            )
        return result

    @classmethod
    def _cache_safe_diagnostic(cls, value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {key: cls._cache_safe_diagnostic(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._cache_safe_diagnostic(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._cache_safe_diagnostic(item) for item in value)
        return value

    @cmk_timed_call("CMK IMAGE RESIZE")
    def resize(
        self,
        PROCESS,
        IMAGE,
        LOG,
        **inputs,
    ):
        model = inputs.get("MODEL (opt)")
        visual = inputs.get("VISUAL")
        prompt = inputs.get("prompt")
        unique_id = inputs.get("unique_id")
        enabled = bool(inputs.get("SCALE ENABLE", True))
        self._validate_transport(PROCESS, IMAGE, LOG)
        source = IMAGE
        input_size = self._dimensions(source)

        if not enabled:
            method = str(inputs.get("scale_method", METHOD_MAX_DIMENSION))
            settings = {"interpolation": "not used"}
            diagnostic = self._diagnostic(source, source, False, method, settings)
            return (
                model,
                PROCESS,
                source,
                LOG,
                self._visual(visual, source, source, False, unique_id),
                diagnostic,
            )

        method = str(inputs.get("scale_method", METHOD_MAX_DIMENSION))
        if method not in METHODS:
            raise ValueError(f"CMK Image Resize: unsupported scale method {method!r}")
        interpolation = str(inputs.get("interpolation", "lanczos"))
        if interpolation not in INTERPOLATION_METHODS:
            raise ValueError("CMK Image Resize: invalid interpolation method")

        settings = {"interpolation": interpolation}
        if method == METHOD_MAX_DIMENSION:
            settings["largest_size"] = int(inputs.get("largest_size", 1600))
        elif method == METHOD_MEGAPIXELS:
            settings["megapixels"] = float(inputs.get("megapixels", 4.0))
        else:
            padding_color = str(inputs.get("padding_color", "black"))
            if padding_color not in PADDING_COLORS:
                raise ValueError("CMK Image Resize: invalid padding color")
            settings.update({
                "target_width": int(inputs.get("target_width", 512)),
                "target_height": int(inputs.get("target_height", 512)),
                "padding_color": padding_color,
            })

        cache_key, cache_detail = self._cache_key(prompt, unique_id, method)
        if cache_key and _faceswap_disk_available(self._CACHE_SCOPE, cache_key):
            try:
                output, result_log = _load_image_log(self._CACHE_SCOPE, cache_key)
                diagnostic = _load_diagnostic(self._CACHE_SCOPE, cache_key)
                diagnostic = dict(diagnostic)
                diagnostic["metadata"] = dict(diagnostic.get("metadata") or {}, cache_hit=True)
                _retain_latest_boundary_entry(self._CACHE_SCOPE, cache_key, unique_id)
                write_status(
                    self._CACHE_SCOPE,
                    "HIT",
                    cache_key=cache_key,
                    detail=cache_detail,
                    unique_id=unique_id,
                )
                print(f"[CMK Boundary Cache / Image Resize] HIT {cache_key[:12]}")
                return (
                    model,
                    PROCESS,
                    output,
                    result_log,
                    self._visual(visual, source, output, True, unique_id),
                    diagnostic,
                )
            except Exception as exc:
                write_status(
                    self._CACHE_SCOPE,
                    "HIT_FAILED",
                    cache_key=cache_key,
                    detail=str(exc),
                    unique_id=unique_id,
                )

        output = self._run_selected(method, source, **settings)
        output_size = self._dimensions(output)
        result_log = cmk_add_block(
            LOG,
            "Image Resize",
            80,
            [
                "STATUS          : EXECUTED",
                f"METHOD          : {method}",
                f"INPUT           : {input_size[0]} x {input_size[1]}",
                f"OUTPUT          : {output_size[0]} x {output_size[1]}",
                *self._setting_lines(method, settings),
                f"INTERPOLATION   : {settings['interpolation']}",
            ],
            True,
        )
        diagnostic = self._diagnostic(source, output, True, method, settings)

        if cache_key:
            try:
                _save_image_log(
                    self._CACHE_SCOPE,
                    cache_key,
                    output,
                    result_log,
                    {"complete": True, "schema": self._CACHE_SCHEMA},
                )
                _save_diagnostic(
                    self._CACHE_SCOPE,
                    cache_key,
                    self._cache_safe_diagnostic(diagnostic),
                )
                _retain_latest_boundary_entry(self._CACHE_SCOPE, cache_key, unique_id)
                write_status(
                    self._CACHE_SCOPE,
                    "MISS_STORED",
                    cache_key=cache_key,
                    detail=cache_detail,
                    unique_id=unique_id,
                )
                print(f"[CMK Boundary Cache / Image Resize] MISS {cache_key[:12]} -> STORED")
            except Exception as exc:
                write_status(
                    self._CACHE_SCOPE,
                    "STORE_FAILED",
                    cache_key=cache_key,
                    detail=str(exc),
                    unique_id=unique_id,
                )

        return (
            model,
            PROCESS,
            output,
            result_log,
            self._visual(visual, source, output, True, unique_id),
            diagnostic,
        )
