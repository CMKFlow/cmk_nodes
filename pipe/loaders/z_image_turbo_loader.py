from __future__ import annotations

import folder_paths
from comfy_execution.graph_utils import ExecutionBlocker

from ..cmk_log_pipe import cmk_add_block
from ..cmk_sampler_prepare import _call_node_kwargs, _unwrap_node_output


DEFAULT_UNET = "z_image_turbo_bf16.safetensors"
DEFAULT_TEXT_ENCODER = "qwen_3_4b.safetensors"
DEFAULT_VAE = "ae.safetensors"


def _choices(folder_name: str, preferred: str) -> list[str]:
    values = list(folder_paths.get_filename_list(folder_name))
    if preferred in values:
        values.remove(preferred)
        values.insert(0, preferred)
    return values or [preferred]


class CMKZImageTurboLoaderPipe:
    """Load the complete Z-Image Turbo model family through ComfyUI Core."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PROCESS": ("CMK_PROCESS_Z_IMAGE",),
                "diffusion_model": (
                    _choices("diffusion_models", DEFAULT_UNET),
                    {"default": DEFAULT_UNET, "label": "DIFFUSION MODEL"},
                ),
                "text_encoder": (
                    _choices("text_encoders", DEFAULT_TEXT_ENCODER),
                    {"default": DEFAULT_TEXT_ENCODER, "label": "TEXT ENCODER"},
                ),
                "vae": (
                    _choices("vae", DEFAULT_VAE),
                    {"default": DEFAULT_VAE, "label": "VAE"},
                ),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    {"default": "default", "advanced": True},
                ),
                "clip_device": (
                    ["default", "cpu"],
                    {"default": "default", "advanced": True},
                ),
            },
            "optional": {
                "LOG": ("CMK_LOG_PIPE",),
            },
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_LOG_PIPE")
    RETURN_NAMES = ("MODEL", "LOG")
    FUNCTION = "load"
    CATEGORY = "CMK/Developer/Pipe/Load"

    def load(
        self,
        PROCESS,
        diffusion_model,
        text_encoder,
        vae,
        weight_dtype="default",
        clip_device="default",
        LOG=None,
    ):
        if PROCESS is None or (
            isinstance(PROCESS, dict) and not PROCESS.get("family_active", True)
        ):
            blocked = ExecutionBlocker(None)
            return (blocked, blocked)
        if not isinstance(PROCESS, dict):
            raise TypeError(
                "CMK Z-Image Turbo Loader: PROCESS must be a CMK process pipe"
            )
        family = str(PROCESS.get("model_family", "z_image_turbo")).strip().lower()
        if family != "z_image_turbo":
            raise ValueError(
                "CMK Z-Image Turbo Loader requires PROCESS ZIT"
            )
        model = _unwrap_node_output(
            _call_node_kwargs(
                ("UNETLoader",),
                unet_name=diffusion_model,
                weight_dtype=weight_dtype,
            )
        )
        clip = _unwrap_node_output(
            _call_node_kwargs(
                ("CLIPLoader",),
                clip_name=text_encoder,
                type="lumina2",
                device=clip_device,
            )
        )
        loaded_vae = _unwrap_node_output(
            _call_node_kwargs(("VAELoader",), vae_name=vae)
        )
        if model is None or clip is None or loaded_vae is None:
            raise RuntimeError(
                "CMK Z-Image Turbo Loader: ComfyUI Core did not return model, "
                "text encoder and VAE."
            )

        model_pipe = {
            "model": model,
            "clip": clip,
            "vae": loaded_vae,
            "model_family": "z_image_turbo",
            "diffusion_model_name": str(diffusion_model),
            "text_encoder_name": str(text_encoder),
            "vae_name": str(vae),
            "weight_dtype": str(weight_dtype),
            "clip_device": str(clip_device),
            "model_pipe_source": "CMK Z-Image Turbo Loader -Pipe-",
        }
        log = cmk_add_block(
            LOG,
            "Z-Image Turbo Loader",
            15,
            [
                "MODEL FAMILY    : Z-IMAGE TURBO",
                f"DIFFUSION MODEL: {diffusion_model}",
                f"TEXT ENCODER    : {text_encoder}",
                f"VAE             : {vae}",
                f"WEIGHT DTYPE    : {weight_dtype}",
                f"CLIP DEVICE     : {clip_device}",
            ],
            True,
        )
        return (model_pipe, log)
