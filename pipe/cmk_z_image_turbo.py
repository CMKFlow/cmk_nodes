from __future__ import annotations

import folder_paths

from comfy_execution.graph_utils import ExecutionBlocker

from ..cmk_common import SAMPLERS, SCHEDULERS
from ..utils.cmk_diagnostic import make_diagnostic_payload
from .cmk_log_pipe import cmk_add_block
from ..utils.cmk_translation import translate_prompt
from .cmk_sampler_prepare import (
    CMK_FIXED_SEED_WIDGET,
    _call_node,
    _clean_text,
    _int,
    _unwrap_node_output,
    _validate_conditioning,
    _call_node_kwargs,
)


def _preferred(values, name):
    return {"default": name} if name in values else {}


DEFAULT_ZIT_INPAINT_PATCH = (
    "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors"
)


def _inpaint_patch_choices():
    values = list(folder_paths.get_filename_list("model_patches"))
    if DEFAULT_ZIT_INPAINT_PATCH in values:
        values.remove(DEFAULT_ZIT_INPAINT_PATCH)
        values.insert(0, DEFAULT_ZIT_INPAINT_PATCH)
    return values or [DEFAULT_ZIT_INPAINT_PATCH]


class CMKSamplerPrepareZImageTurboPipe:
    """Prepare the proven ComfyUI-Core Z-Image Turbo sampling contract."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE",),
                "PROCESS": ("CMK_PROCESS_Z_IMAGE",),
                "seed": ("INT", dict(CMK_FIXED_SEED_WIDGET)),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "sampler": (
                    SAMPLERS,
                    _preferred(SAMPLERS, "res_multistep")
                    | {"advanced": True},
                ),
                "scheduler": (
                    SCHEDULERS,
                    _preferred(SCHEDULERS, "simple")
                    | {"advanced": True},
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.1,
                        "advanced": True,
                    },
                ),
                "model_shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "advanced": True,
                    },
                ),
                "inpaint_model_patch": (
                    _inpaint_patch_choices(),
                    {
                        "default": DEFAULT_ZIT_INPAINT_PATCH,
                        "advanced": True,
                        "label": "INPAINT MODEL PATCH",
                    },
                ),
            },
            "optional": {
                "LOG": ("CMK_LOG_PIPE",),
                "IMAGE": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("CMK_SAMPLER_PIPE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("SAMPLER", "LOG", "diagnostic")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Developer/Pipe/Prepare"

    def check_lazy_status(self, PROCESS=None, IMAGE=None, **kwargs):
        if PROCESS is None:
            return ["PROCESS"]
        if not isinstance(PROCESS, dict) or not PROCESS.get("family_active", True):
            return []
        if bool(PROCESS.get("boolean_inpaint_mode", False)) and IMAGE is None:
            return ["IMAGE"]
        return []

    def prepare(
        self,
        MODEL,
        PROCESS,
        seed,
        steps=8,
        sampler="res_multistep",
        scheduler="simple",
        cfg=1.0,
        model_shift=3.0,
        inpaint_model_patch=DEFAULT_ZIT_INPAINT_PATCH,
        LOG=None,
        IMAGE=None,
    ):
        if PROCESS is None or (
            isinstance(PROCESS, dict) and not PROCESS.get("family_active", True)
        ):
            blocked = ExecutionBlocker(None)
            return (blocked, blocked, blocked)
        if not isinstance(MODEL, dict):
            raise TypeError(
                "CMK Sampler Prepare Z-Image Turbo -Pipe-: MODEL must be a CMK model pipe"
            )
        if not isinstance(PROCESS, dict):
            raise TypeError(
                "CMK Sampler Prepare Z-Image Turbo -Pipe-: PROCESS must be a CMK process pipe"
            )
        if str(MODEL.get("model_family", "")).lower() != "z_image_turbo":
            raise ValueError(
                "CMK Sampler Prepare Z-Image Turbo -Pipe- requires the "
                "CMK Z-Image Turbo Loader."
            )
        process_family = str(PROCESS.get("model_family", "z_image_turbo")).lower()
        if process_family != "z_image_turbo":
            raise ValueError(
                "CMK Sampler Prepare Z-Image Turbo -Pipe- requires "
                "MODEL FAMILY = Z-Image Turbo in Create Image."
            )

        model = MODEL.get("model")
        clip = MODEL.get("clip")
        vae = MODEL.get("vae")
        if model is None or clip is None or vae is None:
            raise ValueError(
                "CMK Sampler Prepare Z-Image Turbo -Pipe-: MODEL must provide "
                "model, text encoder and VAE."
            )

        width = _int(PROCESS.get("width", PROCESS.get("target_width")), 1024)
        height = _int(PROCESS.get("height", PROCESS.get("target_height")), 1024)
        prompt = _clean_text(PROCESS.get("prompt_pos"), "")
        if not prompt:
            raise ValueError(
                "CMK Sampler Prepare Z-Image Turbo -Pipe- requires PROMPT POS."
            )
        translation = translate_prompt(prompt)

        positive = _validate_conditioning(
            _unwrap_node_output(_call_node(("CLIPTextEncode",), clip, translation.text)),
            "z_image_conditioning_pos",
        )
        negative = _validate_conditioning(
            _unwrap_node_output(_call_node(("ConditioningZeroOut",), positive)),
            "z_image_conditioning_neg",
        )
        inpaint_enabled = bool(PROCESS.get("boolean_inpaint_mode", False))
        mask = PROCESS.get("mask")
        controlnet_enabled = bool(PROCESS.get("boolean_controlnet_enable", False))
        if controlnet_enabled:
            control_image = PROCESS.get("zit_controlnet_image")
            patch_name = str(PROCESS.get("zit_controlnet_patch", "")).strip()
            if control_image is None or not patch_name:
                raise ValueError(
                    "CMK ZIT ControlNet preparation is incomplete: CONTROL IMAGE "
                    "and MODEL PATCH are required."
                )
            model_patch = _unwrap_node_output(
                _call_node_kwargs(("ModelPatchLoader",), name=patch_name)
            )
            model = _unwrap_node_output(
                _call_node_kwargs(
                    ("QwenImageDiffsynthControlnet", "ZImageFunControlnet"),
                    model=model,
                    model_patch=model_patch,
                    vae=vae,
                    image=control_image,
                    strength=float(PROCESS.get("zit_controlnet_strength", 1.0)),
                )
            )
            width = int(control_image.shape[2])
            height = int(control_image.shape[1])

        if inpaint_enabled:
            if IMAGE is None:
                raise ValueError(
                    "CMK Z-Image Turbo Inpaint requires IMAGE from Create Image."
                )
            if mask is None:
                raise ValueError(
                    "CMK Z-Image Turbo Inpaint requires MASK in PROCESS."
                )
            patch_name = str(inpaint_model_patch or "").strip()
            if not patch_name:
                raise ValueError(
                    "CMK Z-Image Turbo Inpaint requires an INPAINT MODEL PATCH."
                )
            model_patch = _unwrap_node_output(
                _call_node_kwargs(("ModelPatchLoader",), name=patch_name)
            )
            model = _unwrap_node_output(
                _call_node_kwargs(
                    ("ZImageFunControlnet",),
                    model=model,
                    model_patch=model_patch,
                    vae=vae,
                    inpaint_image=IMAGE,
                    mask=mask,
                    strength=1.0,
                )
            )
            conditioned = _call_node_kwargs(
                ("InpaintModelConditioning",),
                positive=positive,
                negative=negative,
                vae=vae,
                pixels=IMAGE,
                mask=mask,
                noise_mask=True,
            )
            if not isinstance(conditioned, (tuple, list)) or len(conditioned) < 3:
                raise TypeError(
                    "CMK Z-Image Turbo Inpaint: InpaintModelConditioning "
                    "returned an invalid result."
                )
            positive = _validate_conditioning(
                conditioned[0], "z_image_inpaint_conditioning_pos"
            )
            negative = _validate_conditioning(
                conditioned[1], "z_image_inpaint_conditioning_neg"
            )
            latent = conditioned[2]
            width = int(IMAGE.shape[2])
            height = int(IMAGE.shape[1])
        else:
            latent = _unwrap_node_output(
                _call_node(("EmptySD3LatentImage",), width, height, 1)
            )

        patched_model = _unwrap_node_output(
            _call_node(("ModelSamplingAuraFlow",), model, float(model_shift))
        )
        if not isinstance(latent, dict) or "samples" not in latent:
            raise TypeError(
                "CMK Sampler Prepare Z-Image Turbo -Pipe-: "
                "EmptySD3LatentImage returned an invalid LATENT."
            )

        sampler_pipe = dict(PROCESS)
        sampler_pipe.update(
            {
                "model": model,
                "model_patched": patched_model,
                "clip": clip,
                "vae": vae,
                "conditioning_pos": positive,
                "conditioning_neg": negative,
                "latent_image": latent,
                "latent_original": latent,
                "steps_1st_pass": max(1, int(steps)),
                "steps": max(1, int(steps)),
                "cfg": float(cfg),
                "sampler": str(sampler),
                "scheduler": str(scheduler),
                "seed": int(seed),
                "denoise": 1.0,
                "model_family": "z_image_turbo",
                "model_shift": float(model_shift),
                "boolean_inpaint_mode": inpaint_enabled,
                "inpaint_process_mode": (
                    str(PROCESS.get("inpaint_process_mode", "custom"))
                    if inpaint_enabled else "text2image"
                ),
                "inpaint_model_patch": (
                    str(inpaint_model_patch) if inpaint_enabled else None
                ),
                "boolean_controlnet_enable": controlnet_enabled,
            }
        )
        lines = [
            "STATUS          : PREPARED",
            "MODEL FAMILY    : ZIT",
            f"MODE            : {'INPAINT (EXPERIMENTAL)' if inpaint_enabled else ('CONTROLNET' if controlnet_enabled else 'TEXT2IMAGE')}",
            f"SIZE            : {width} × {height}",
            f"STEPS           : {int(steps)}",
            f"CFG             : {float(cfg):g}",
            f"SAMPLER         : {sampler}",
            f"SCHEDULER       : {scheduler}",
            f"MODEL SHIFT     : {float(model_shift):g}",
            f"SEED            : {int(seed)}",
            "",
            "POSITIVE PROMPT:",
            prompt,
        ]
        if translation.status not in {"empty", "disabled", "not_configured"}:
            lines.insert(9, translation.log_line("POS"))
        if inpaint_enabled:
            lines.insert(3, f"INPAINT PATCH   : {inpaint_model_patch}")
        log = cmk_add_block(LOG, "Z-Image Turbo Prepare", 40, lines, True)
        summary = "\n".join(lines)
        diagnostic = make_diagnostic_payload(
            title="Z-Image Turbo Prepare",
            node="CMK Sampler Prepare Z-Image Turbo -Pipe-",
            previews=[],
            summary=f"{width}x{height} | {int(steps)} steps | CFG {float(cfg):g}",
            details=summary,
            mode=(
                "Inpaint (Experimental)" if inpaint_enabled
                else ("ControlNet" if controlnet_enabled else "Text2Image")
            ),
            metadata={
                "model_family": "z_image_turbo",
                "width": width,
                "height": height,
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler": str(sampler),
                "scheduler": str(scheduler),
                "model_shift": float(model_shift),
                "seed": int(seed),
                "inpaint": inpaint_enabled,
                "inpaint_model_patch": (
                    str(inpaint_model_patch) if inpaint_enabled else None
                ),
            },
        )
        return (sampler_pipe, log, diagnostic)


class CMKZImageTurboFinalizePipe:
    """Decode a sampled Z-Image latent and rejoin the public IMAGE flow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS": ("CMK_PROCESS_Z_IMAGE",),
                "SAMPLED": ("CMK_SAMPLED_PIPE", {"lazy": True}),
            },
            "optional": {
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
            },
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "CMK_PROCESS_Z_IMAGE",
        "IMAGE",
        "CMK_LOG_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "diagnostic")
    FUNCTION = "finalize"
    CATEGORY = "CMK/Developer/Pipe/Finalize"

    def check_lazy_status(self, MODEL=None, PROCESS=None, SAMPLED=None, LOG=None):
        if PROCESS is None:
            return ["PROCESS"]
        if isinstance(PROCESS, dict) and not PROCESS.get("family_active", True):
            return []
        return [
            name for name, value in (
                ("MODEL", MODEL), ("SAMPLED", SAMPLED), ("LOG", LOG)
            ) if value is None
        ]

    def finalize(self, MODEL=None, PROCESS=None, SAMPLED=None, LOG=None):
        if isinstance(PROCESS, dict) and not PROCESS.get("family_active", True):
            blocked = ExecutionBlocker(None)
            return (blocked, PROCESS, blocked, blocked, blocked)
        if not isinstance(MODEL, dict) or not isinstance(PROCESS, dict):
            raise TypeError(
                "CMK Z-Image Turbo Finalize -Pipe- requires MODEL and PROCESS pipes."
            )
        if not isinstance(SAMPLED, dict):
            raise TypeError(
                "CMK Z-Image Turbo Finalize -Pipe- requires a SAMPLED pipe."
            )
        vae = MODEL.get("vae")
        latent = SAMPLED.get(
            "samples",
            SAMPLED.get("latent_1st_pass", SAMPLED.get("latent_image")),
        )
        if vae is None or latent is None:
            raise ValueError(
                "CMK Z-Image Turbo Finalize -Pipe- requires MODEL['vae'] "
                "and a sampled latent."
            )
        image = SAMPLED.get("image")
        if image is None:
            image = _unwrap_node_output(_call_node(("VAEDecode",), vae, latent))
        if image is None:
            raise RuntimeError(
                "CMK Z-Image Turbo Finalize -Pipe-: VAEDecode returned no IMAGE."
            )

        process = dict(PROCESS)
        generation_mode = (
            "inpaint" if bool(PROCESS.get("boolean_inpaint_mode", False))
            else "text2image"
        )
        process.update(
            {
                "model_family": "z_image_turbo",
                "generation_mode": generation_mode,
                "z_image_sampled": True,
            }
        )
        lines = [
            "STATUS          : DECODED",
            "MODEL FAMILY    : Z-IMAGE TURBO",
            f"SIZE            : {process.get('width')} × {process.get('height')}",
        ]
        log = cmk_add_block(LOG, "Z-Image Turbo Decode", 45, lines, True)
        diagnostic = make_diagnostic_payload(
            title="Z-Image Turbo",
            node="CMK Z-Image Turbo Finalize -Pipe-",
            previews=[image],
            stages=[
                {
                    "title": "Z-Image Turbo Result",
                    "subtitle": "Decoded image",
                    "image": image,
                }
            ],
            summary="Z-Image Turbo sampled and decoded",
            details="\n".join(lines),
            mode=(
                "Inpaint (Experimental)"
                if generation_mode == "inpaint"
                else "Text2Image"
            ),
            metadata={"model_family": "z_image_turbo"},
        )
        return (MODEL, process, image, log, diagnostic)
