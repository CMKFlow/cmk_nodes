from __future__ import annotations

from ...nodes.controlnet.controlnet import (
    CONTROLNET_IMAGE_SOURCES,
    _controlnet_preprocessor_input,
    _empty_controlnet_ui_placeholder,
    _get_controlnet_models,
    _get_input_files,
    _load_image_from_input,
    _tensor_image_to_temp_ui,
)
from .cmk_controlnet_prepare import CMKControlNetPreparePipe, _controlnet_visual
from .cmk_zit_controlnet_prepare import (
    CMKZITControlNetPreparePipe,
    DEFAULT_ZIT_CONTROLNET_PATCH,
    _patch_choices,
)


def _unpack_node_result(payload):
    if isinstance(payload, dict) and "result" in payload:
        return payload.get("ui", {}), tuple(payload["result"])
    return {}, tuple(payload)


def _combined_preview_ui(*, enabled, image_source, reference_image, image, ui):
    if not enabled:
        preview_image = (
            _load_image_from_input(reference_image)
            if str(image_source) == "Reference Image"
            else image
        )
        return {
            "images": _tensor_image_to_temp_ui(
                preview_image, "cmk_controlnet_reference"
            ) if preview_image is not None else []
        }
    return ui or {"images": _empty_controlnet_ui_placeholder()}


class CMKCombinedControlNetPreparePipe:
    """One ControlNet UI for parallel SDXL and Z-Image Turbo flow branches."""

    @classmethod
    def INPUT_TYPES(cls):
        input_files = _get_input_files() or [""]
        controlnet_models = _get_controlnet_models() or [""]
        return {
            "required": {
                "PROCESS SDXL": ("CMK_PROCESS_SDXL",),
                "PROCESS ZIT": ("CMK_PROCESS_Z_IMAGE",),
                "IMAGE": ("IMAGE",),
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
                "RESOLUTION": (
                    "INT",
                    {"default": 768, "min": 64, "max": 8192, "step": 8},
                ),
                "sdxl_controlnet_model": (
                    controlnet_models,
                    {"advanced": True, "label": "sdxl · controlnet_model"},
                ),
                "sdxl_preprocessor": (
                    _controlnet_preprocessor_input()[0],
                    {
                        **_controlnet_preprocessor_input()[1],
                        "advanced": True,
                        "label": "sdxl · preprocessor",
                    },
                ),
                "sdxl_start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 1.0,
                        "advanced": True,
                        "label": "sdxl · start_percent",
                    },
                ),
                "sdxl_end_percent": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 1.0,
                        "advanced": True,
                        "label": "sdxl · end_percent",
                    },
                ),
                "sdxl_invert_hint": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "advanced": True,
                        "label": "sdxl · invert_hint",
                    },
                ),
                "zit_low_threshold": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "advanced": True,
                        "label": "zit · low_threshold",
                    },
                ),
                "zit_high_threshold": (
                    "FLOAT",
                    {
                        "default": 0.32,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "advanced": True,
                        "label": "zit · high_threshold",
                    },
                ),
                "zit_model_patch": (
                    _patch_choices(),
                    {
                        "default": DEFAULT_ZIT_CONTROLNET_PATCH,
                        "advanced": True,
                        "label": "zit · model_patch",
                    },
                ),
            },
            "optional": {
                "LOG": ("CMK_LOG_PIPE",),
                "VISUAL": ("CMK_VISUAL_PIPE",),
                "REFERENCE IMAGE INPUT": ("IMAGE",),
                "REFERENCE IMAGE NAME": ("STRING",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "CMK_PROCESS_SDXL",
        "CMK_PROCESS_Z_IMAGE",
        "IMAGE",
        "CMK_LOG_PIPE",
        "CMK_VISUAL_PIPE",
        "CMK_DIAGNOSTIC",
        "IMAGE",
    )
    RETURN_NAMES = (
        "PROCESS SDXL", "PROCESS ZIT", "IMAGE", "LOG", "VISUAL", "diagnostic",
        "CONTROLNET IMAGE",
    )
    FUNCTION = "prepare"
    CATEGORY = "CMK/Flow/Process"
    OUTPUT_NODE = True

    def prepare(self, **kwargs):
        process_sdxl = kwargs["PROCESS SDXL"]
        process_zit = kwargs["PROCESS ZIT"]
        if not isinstance(process_sdxl, dict) or not isinstance(process_zit, dict):
            raise TypeError("CMK Combined ControlNet requires both SDXL and ZIT PROCESS inputs.")

        common = {
            "ENABLE": kwargs.get("ENABLE", False),
            "IMAGE SOURCE": kwargs.get("IMAGE SOURCE", "Reference Image"),
            "REFERENCE IMAGE": kwargs.get("REFERENCE IMAGE", ""),
            "APPLY MASK": kwargs.get("APPLY MASK", False),
            "STRENGTH": kwargs.get("STRENGTH", 1.0),
            "LOG": kwargs.get("LOG"),
            "REFERENCE IMAGE INPUT": kwargs.get("REFERENCE IMAGE INPUT"),
            "REFERENCE IMAGE NAME": kwargs.get("REFERENCE IMAGE NAME"),
            "VISUAL": kwargs.get("VISUAL"),
            "_register_visual": False,
        }
        image = kwargs.get("IMAGE")
        enabled = bool(common["ENABLE"])
        image_source = str(common["IMAGE SOURCE"])
        reference_image = str(common["REFERENCE IMAGE"])
        active_sdxl = bool(process_sdxl.get("family_active", True))
        active_zit = bool(process_zit.get("family_active", True))
        hybrid_mode = bool(
            process_sdxl.get("hybrid_mode", False)
            or process_zit.get("hybrid_mode", False)
        )
        if not active_sdxl and not active_zit:
            raise ValueError("CMK Combined ControlNet requires at least one active model family.")
        if active_sdxl and active_zit and not hybrid_mode:
            raise ValueError("CMK Combined ControlNet received two active non-Hybrid families.")

        # Hybrid uses SDXL for composition and ZIT only for the final natural
        # low-denoise finish. Applying ControlNet again in that finish would
        # steer the already established image a second time.
        prepare_zit = active_zit and not hybrid_mode

        prepared_sdxl = process_sdxl
        prepared_zit = process_zit
        log = common["LOG"]
        ui = {}
        controlnet_image = None
        diagnostics = []
        if active_sdxl:
            sdxl_common = dict(common)
            sdxl_common["LOG"] = log
            nested = CMKControlNetPreparePipe().prepare_controlnet_pipe(
                process_sdxl,
                image,
                int(kwargs.get("RESOLUTION", 768)),
                float(kwargs.get("sdxl_start_percent", 0.0)),
                float(kwargs.get("sdxl_end_percent", 30.0)),
                bool(kwargs.get("sdxl_invert_hint", False)),
                **sdxl_common,
                **{
                    "CONTROLNET MODEL": kwargs.get("sdxl_controlnet_model", ""),
                    "PREPROCESSOR": kwargs.get("sdxl_preprocessor", "none"),
                },
            )
            branch_ui, result = _unpack_node_result(nested)
            prepared_sdxl, _, log, _, diagnostic, branch_image = result
            ui = branch_ui or ui
            if enabled and branch_image is not None:
                controlnet_image = branch_image
            if isinstance(diagnostic, dict):
                diagnostics.append(diagnostic)

        if prepare_zit:
            zit_common = dict(common)
            zit_common["LOG"] = log
            nested = CMKZITControlNetPreparePipe().prepare(
                PROCESS=process_zit,
                IMAGE=image,
                resolution=int(kwargs.get("RESOLUTION", 768)),
                low_threshold=float(kwargs.get("zit_low_threshold", 0.1)),
                high_threshold=float(kwargs.get("zit_high_threshold", 0.32)),
                **zit_common,
                **{"MODEL PATCH": kwargs.get("zit_model_patch", DEFAULT_ZIT_CONTROLNET_PATCH)},
            )
            branch_ui, result = _unpack_node_result(nested)
            prepared_zit, _, log, _, diagnostic, branch_image = result
            ui = branch_ui or ui
            if enabled and branch_image is not None:
                controlnet_image = branch_image
            if isinstance(diagnostic, dict):
                diagnostics.append(diagnostic)

        diagnostic = diagnostics[0] if diagnostics else None
        visual = _controlnet_visual(
            kwargs.get("VISUAL"), enabled=enabled, image=controlnet_image,
            unique_id=kwargs.get("unique_id"),
            module_type="CMKCombinedControlNetPreparePipe",
            branch="", stage_key="controlnet",
        )
        return {
            "ui": _combined_preview_ui(
                enabled=enabled,
                image_source=image_source,
                reference_image=reference_image,
                image=image,
                ui=ui,
            ),
            "result": (
                prepared_sdxl, prepared_zit, image, log,
                visual, diagnostic, controlnet_image,
            ),
        }
