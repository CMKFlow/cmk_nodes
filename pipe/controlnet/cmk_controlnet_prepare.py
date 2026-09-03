from __future__ import annotations

from ...nodes.controlnet.controlnet import (
    CMKControlNetPrepare,
    CMKPipeSetControlNet,
    CONTROLNET_IMAGE_SOURCES,
    _get_input_files,
    _get_controlnet_models,
    _controlnet_preprocessor_input,
    _build_controlnet_log_lines,
    _load_image_from_input,
    _build_controlnet_diagnostic,
    _tensor_image_to_temp_ui,
    _empty_controlnet_ui_placeholder,
)
from ..cmk_log_pipe import cmk_add_block


class CMKControlNetPreparePipe:
    """Pipe wrapper for CMK ControlNet Prepare.

    Integrates the former Pipe Peek ControlNet Source -> ControlNet Prepare ->
    Pipe Set ControlNet chain into one frontend-friendly module.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_files = _get_input_files()
        if not input_files:
            input_files = [""]

        controlnet_models = _get_controlnet_models()
        if not controlnet_models:
            controlnet_models = [""]

        return {
            "required": {
                "PROCESS": ("CMK_PROCESS_SDXL",),
                "IMAGE": ("IMAGE",),
                "ENABLE": ("BOOLEAN", {"default": False}),
                "CONTROLNET MODEL": (controlnet_models,),
                "IMAGE SOURCE": (CONTROLNET_IMAGE_SOURCES, {"default": "Reference Image"}),
                "REFERENCE IMAGE": (
                    input_files,
                ),
                "APPLY MASK": ("BOOLEAN", {"default": False}),
                "PREPROCESSOR": _controlnet_preprocessor_input(),
                "STRENGTH": ("FLOAT", {
                    "default": 1.50, "min": 0.0, "max": 2.0, "step": 0.05
                }),
                "resolution": ("INT", {
                    "default": 768, "min": 64, "max": 8192, "step": 8,
                    "advanced": True
                }),
                "controlnet_start_percent": ("FLOAT", {
                    "default": 0.00, "min": 0.0, "max": 100.0, "step": 1.0,
                    "advanced": True,
                    "tooltip": "ControlNet start point in percent (0–100)."
                }),
                "controlnet_end_percent": ("FLOAT", {
                    "default": 30.00, "min": 0.0, "max": 100.0, "step": 1.0,
                    "advanced": True,
                    "tooltip": "ControlNet end point in percent (0–100)."
                }),
                "invert_hint": ("BOOLEAN", {
                    "default": False,
                    "advanced": True
                }),
            },
            "optional": {
                "LOG": ("CMK_LOG_PIPE",),
                "REFERENCE IMAGE INPUT": ("IMAGE",),
                "REFERENCE IMAGE NAME": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = (
        "CMK_PROCESS_SDXL",
        "IMAGE",
        "CMK_LOG_PIPE",
        "CMK_DIAGNOSTIC",
        "IMAGE",
    )
    RETURN_NAMES = ("PROCESS", "IMAGE", "LOG", "diagnostic", "CONTROLNET IMAGE")
    FUNCTION = "prepare_controlnet_pipe"
    CATEGORY = "CMK/Flow/Process"
    OUTPUT_NODE = True

    def prepare_controlnet_pipe(
        self,
        PROCESS,
        IMAGE,
        resolution,
        controlnet_start_percent,
        controlnet_end_percent,
        invert_hint,
        **kwargs,
    ):
        new_pipe = dict(PROCESS)

        enable = bool(kwargs.get("ENABLE", False))
        controlnet_model = str(kwargs.get("CONTROLNET MODEL", ""))
        image_source = str(kwargs.get("IMAGE SOURCE", "Reference Image"))
        reference_image = str(kwargs.get("REFERENCE IMAGE", ""))
        external_reference_image = kwargs.get("REFERENCE IMAGE INPUT")
        external_reference_name = str(kwargs.get("REFERENCE IMAGE NAME", "") or "")
        if external_reference_name:
            reference_image = external_reference_name
        apply_mask = bool(kwargs.get("APPLY MASK", False))
        preprocessor = kwargs.get("PREPROCESSOR")
        controlnet_strength = float(kwargs.get("STRENGTH", 1.50))
        invert_hint = bool(invert_hint)

        if image_source == "Base Image":
            source_image = IMAGE
            source_log = "Base Image"
        elif external_reference_image is not None:
            source_image = external_reference_image
            source_log = f"Reference Image | {reference_image}"
        else:
            source_image = _load_image_from_input(reference_image)
            source_log = f"Reference Image | {reference_image}"

        # InstantID Text2Image reference-latent path: module 05 supplies the
        # original RGB composition reference. Module 15 VAE-encodes it, so the
        # regular ControlNet model/preprocessor and module 10's partial sampling
        # are not needed for this branch.
        #
        # Inpaint must retain the regular ControlNet + module 10 path. Its x0 is
        # decoded/re-encoded at the configured Inpaint handoff before module 15
        # continues with InstantID. Letting the Text2Image shortcut win here
        # would bypass both Fooocus Inpaint sampling and that clean bridge.
        instantid_reference_latent_mode = bool(
            new_pipe.get("instantid_enabled", False)
            and str(new_pipe.get("instantid_mode", "HYBRID") or "HYBRID").strip().upper()
            == "REFERENCE-LATENT"
            and not new_pipe.get("boolean_inpaint_mode", False)
            and enable
            and source_image is not None
        )

        # IMAGE is transported only through the public IMAGE socket.
        # Never mirror the main image payload into PROCESS or LOG.
        new_pipe.pop("image", None)
        new_pipe.pop("image_original", None)

        # DIAGNOSTIC: reconstruct the former, proven standalone chain exactly:
        #
        #   CMK ControlNet Prepare
        #       -> CONTROL_NET + processed IMAGE
        #   CMK Pipe Set ControlNet
        #       -> PROCESS
        #
        # The public standalone method is used instead of calling _prepare_core
        # directly, so source selection, preprocessing and result wrapping follow
        # the old node path rather than the merged Pipe implementation.
        if instantid_reference_latent_mode:
            control_net = None
            controlnet_image = source_image
            enabled = True
            log = (
                "InstantID reference latent | original RGB source forwarded | "
                f"{source_log} | regular ControlNet load/preprocess bypassed"
            )
        else:
            prepare_image_source = (
                "Base Image" if external_reference_image is not None else image_source
            )
            standalone_result = CMKControlNetPrepare().prepare_controlnet(
                controlnet_model=controlnet_model,
                image_source=prepare_image_source,
                reference_image=reference_image,
                apply_mask=apply_mask,
                preprocessor=preprocessor,
                resolution=resolution,
                opt_base_image=(
                    external_reference_image
                    if external_reference_image is not None
                    else IMAGE
                ),
                opt_mask=new_pipe.get("mask"),
                **{"USE CONTROLNET": enable},
            )

            if not isinstance(standalone_result, dict) or "result" not in standalone_result:
                raise TypeError(
                    "CMK ControlNet Prepare -Pipe-: standalone prepare returned invalid output"
                )

            result_payload = standalone_result["result"]
            if not isinstance(result_payload, (tuple, list)) or len(result_payload) < 4:
                raise TypeError(
                    "CMK ControlNet Prepare -Pipe-: standalone prepare result is incomplete"
                )

            control_net = result_payload[0]
            controlnet_image = result_payload[1]
            enabled = bool(result_payload[2])
            log = str(result_payload[3])

        strength = float(controlnet_strength)
        start_percent_ui = min(100.0, max(0.0, float(controlnet_start_percent)))
        end_percent_ui = min(100.0, max(0.0, float(controlnet_end_percent)))
        start_percent = start_percent_ui / 100.0
        end_percent = end_percent_ui / 100.0
        
        if enabled and control_net is not None and controlnet_image is not None:
            new_pipe = CMKPipeSetControlNet().set_controlnet(
                new_pipe,
                control_net,
                controlnet_image,
            )[0]
        else:
            new_pipe["control_net"] = None
            new_pipe["controlnet_image"] = None

        new_pipe["instantid_reference_latent_mode"] = instantid_reference_latent_mode
        new_pipe["instantid_reference_image"] = (
            source_image if instantid_reference_latent_mode else None
        )

        new_pipe["boolean_controlnet_enable"] = bool(enabled)
        new_pipe["controlnet_prepare_log"] = (
            log
        )
        new_pipe["controlnet_strength"] = strength
        new_pipe["controlnet_start_percent"] = start_percent
        new_pipe["controlnet_end_percent"] = end_percent
        new_pipe["controlnet_invert_hint"] = invert_hint

        log_lines = _build_controlnet_log_lines(
            use_controlnet=enable,
            controlnet_model=controlnet_model,
            image_source=image_source,
            reference_image=reference_image,
            apply_mask=apply_mask,
            preprocessor=preprocessor,
            resolution=resolution,
            enabled=enabled,
            log=log,
        )
        if enable:
            log_lines.extend([
                f"STRENGTH        : {strength:.3f}",
                f"START PERCENT   : {start_percent_ui:.0f}%",
                f"END PERCENT     : {end_percent_ui:.0f}%",
                f"INVERT HINT     : {'ON' if invert_hint else 'OFF'}",
            ])
        if instantid_reference_latent_mode:
            log_lines.extend([
                "INSTANTID MODE  : REFERENCE LATENT",
                "MODULE 10       : SAMPLING BYPASS",
                "MODULE 15       : VAE ENCODE + NOISE",
            ])
        log_pipe = cmk_add_block(kwargs.get("LOG"), "ControlNet", 30, log_lines, True)

        diagnostic_source_image = source_image

        diagnostic = _build_controlnet_diagnostic(
            title="ControlNet Prepare",
            node="CMK ControlNet Prepare -Pipe-",
            use_controlnet=enable,
            controlnet_model=controlnet_model,
            image_source=image_source,
            reference_image=reference_image,
            apply_mask=apply_mask,
            preprocessor=preprocessor,
            resolution=resolution,
            base_image=diagnostic_source_image,
            mask=new_pipe.get("mask"),
            controlnet_image=controlnet_image,
            enabled=enabled,
            log=log,
        )

        if enabled and controlnet_image is not None:
            ui_images = _tensor_image_to_temp_ui(controlnet_image)
        else:
            ui_images = []

        return {
            "ui": {"images": ui_images},
            "result": (
                new_pipe,
                IMAGE,
                log_pipe,
                diagnostic,
                controlnet_image if enabled else None,
            ),
        }
