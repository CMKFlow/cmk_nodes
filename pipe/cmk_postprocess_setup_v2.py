"""CMK Flow · 30 PostProcess Setup v2 (2.8): the PostProcess Setup for SDXL and ZIT.

The generation completion, its cache, the release of the generation models and
the SDXL PostProcess MODEL are those of CMK Flow · 30 PostProcess Setup
(inherited unchanged).  v2 adds the ZIT PostProcess MODEL from the settings of
CMK PostProcess Model (SDXL/ZIT); like the SDXL MODEL it is the cheap
specification, materialised only by its first ZIT consumer.

Outputs follow the CMK port contract: MODEL ZIT, MODEL SDXL, PROCESS, IMAGE,
LOG, VISUAL.
"""

from __future__ import annotations

from .cmk_family_result import CMKPostProcessSetupPipe
from .cmk_log_pipe import cmk_add_block
from .loaders.postprocess_model_source import ZIT_INPUTS, CMKPostProcessModelSourcePipe

# The ZIT settings of CMK PostProcess Model, named like the SDXL settings of
# the Setup (postprocess_checkpoint, postprocess_vae, use_checkpoint_vae).
ZIT_WIDGETS = {
    "postprocess_zit_model": "diffusion_model",
    "postprocess_zit_text_encoder": "text_encoder",
    "postprocess_zit_vae": "vae",
    "postprocess_zit_weight_dtype": "weight_dtype",
    "postprocess_zit_clip_device": "clip_device",
}
assert tuple(ZIT_WIDGETS.values()) == ZIT_INPUTS


class CMKPostProcessSetupV2Pipe(CMKPostProcessSetupPipe):
    """Complete any supported generation family and provide the SDXL and ZIT
    PostProcess models (neither is loaded here)."""

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        source = CMKPostProcessModelSourcePipe.INPUT_TYPES()["required"]
        for name, source_name in ZIT_WIDGETS.items():
            widget = source[source_name]
            types["required"][name] = (widget[0], {**(widget[1] if len(widget) > 1 else {}), "advanced": True})
        return types

    RETURN_TYPES = ("CMK_MODEL_PIPE", *CMKPostProcessSetupPipe.RETURN_TYPES)
    RETURN_NAMES = ("MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL")

    def merge(self, postprocess_checkpoint, postprocess_vae, use_checkpoint_vae, **inputs):
        zit = [inputs.pop(name) for name in ZIT_WIDGETS]
        model, process, image, log, visual = super().merge(
            postprocess_checkpoint, postprocess_vae, use_checkpoint_vae, **inputs
        )
        zit_model = CMKPostProcessModelSourcePipe().load(
            postprocess_checkpoint, postprocess_vae, use_checkpoint_vae, *zit,
        )[1]
        if process.get("result_contract") == "preview_handoff":
            return zit_model, model, process, image, log, visual
        for key in ("source_model_family", "generation_family"):
            zit_model[key] = process.get(key)
        process = {**process, "postprocess_zit_model": zit_model["diffusion_model_name"]}
        log = cmk_add_block(log, "PostProcess Model ZIT", 55, [
            f"DIFFUSION MODEL : {zit_model['diffusion_model_name']}",
            f"TEXT ENCODER    : {zit_model['text_encoder_name']}",
            f"VAE             : {zit_model['vae_name']}",
            "MODEL LOAD      : DEFERRED UNTIL REQUIRED",
        ], True)
        return zit_model, model, process, image, log, visual
