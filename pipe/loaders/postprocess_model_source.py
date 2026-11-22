"""Common model source for standalone PostProcess workflows (2.7.1).

One node, two independent outputs: ``MODEL SDXL`` and ``MODEL ZIT``. Both are
the cheap specifications of the existing single sources
(``make_postprocess_model_spec`` / ``make_zit_postprocess_model_spec``);
nothing is loaded here. Each family is materialised only by its own first
real consumer (``resolve_postprocess_model`` / ``resolve_zit_postprocess_model``).
"""

from __future__ import annotations

from .checkpoint_vae_loader import CMKPostProcessModelPipe, make_postprocess_model_spec
from .z_image_turbo_loader import (
    CMKZImageTurboPostProcessModelPipe,
    make_zit_postprocess_model_spec,
)

SDXL_INPUTS = ("ckpt_name", "vae_name", "checkpoint_vae")
ZIT_INPUTS = ("diffusion_model", "text_encoder", "vae", "weight_dtype", "clip_device")


class CMKPostProcessModelSourcePipe:
    """SDXL and Z-Image Turbo model specifications from one node.

    The settings are exactly those of CMK PostProcess Model SDXL and CMK
    Z-Image Turbo PostProcess Model; both families stay configurable at the
    same time and there is no family switch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        sdxl = CMKPostProcessModelPipe.INPUT_TYPES()["required"]
        zit = CMKZImageTurboPostProcessModelPipe.INPUT_TYPES()["required"]
        return {"required": {**sdxl, **zit}}

    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_MODEL_PIPE")
    RETURN_NAMES = ("MODEL SDXL", "MODEL ZIT")
    FUNCTION = "load"
    CATEGORY = "CMK/Flow/Input"

    def load(
        self,
        ckpt_name,
        vae_name,
        checkpoint_vae,
        diffusion_model,
        text_encoder,
        vae,
        weight_dtype="default",
        clip_device="default",
    ):
        return (
            make_postprocess_model_spec(ckpt_name, vae_name, checkpoint_vae),
            make_zit_postprocess_model_spec(diffusion_model, text_encoder, vae, weight_dtype, clip_device),
        )
