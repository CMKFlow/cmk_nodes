from __future__ import annotations

import math

import folder_paths

from comfy_execution.graph_utils import ExecutionBlocker

from ..cmk_common import SAMPLERS, SCHEDULERS
from ..utils.cmk_diagnostic import make_diagnostic_payload
from .cmk_log_pipe import cmk_add_block
from ..utils.cmk_translation import translate_prompt
from ..loader.cmk_lora_text_loader import CMKLoRATextLoader
from .cmk_sampler_prepare import (
    CMK_FIXED_SEED_WIDGET,
    _call_node,
    _clean_text,
    _int,
    _unwrap_node_output,
    _validate_conditioning,
    _call_node_kwargs,
)
from .loaders.z_image_turbo_loader import (
    ensure_zit_text_encoder,
    evict_zit_text_encoder,
)


def _preferred(values, name):
    return {"default": name} if name in values else {}


DEFAULT_ZIT_INPAINT_PATCH = (
    "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors"
)


def _hybrid_finish_steps(base_steps, denoise):
    """Limit Hybrid ZIT to the denoise-sized tail of its normal step budget."""
    base_steps = max(1, int(base_steps))
    denoise = min(1.0, max(0.0, float(denoise)))
    return max(1, min(base_steps, math.ceil(base_steps * denoise)))


def _hybrid_masked_composite(source, result, mask):
    """Keep the authoritative source pixel-exact outside the process mask."""
    import torch
    import torch.nn.functional as F

    if not isinstance(source, torch.Tensor) or not isinstance(result, torch.Tensor):
        raise TypeError("CMK Z-Image Turbo INPAINT requires tensor images.")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("CMK Z-Image Turbo INPAINT requires a tensor mask.")

    target_h, target_w = int(result.shape[1]), int(result.shape[2])
    base = source.to(device=result.device, dtype=result.dtype)
    if tuple(base.shape[1:3]) != (target_h, target_w):
        base = F.interpolate(
            base.movedim(-1, 1),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)

    alpha = mask.to(device=result.device, dtype=result.dtype)
    if alpha.ndim == 2:
        alpha = alpha.unsqueeze(0)
    elif alpha.ndim == 4:
        alpha = alpha[:, 0] if alpha.shape[1] == 1 else alpha[..., 0]
    if alpha.ndim != 3:
        raise ValueError("CMK Z-Image Turbo INPAINT mask must be HxW or BxHxW.")
    if tuple(alpha.shape[-2:]) != (target_h, target_w):
        alpha = F.interpolate(
            alpha.unsqueeze(1),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    if base.shape[0] == 1 and result.shape[0] > 1:
        base = base.expand(result.shape[0], -1, -1, -1)
    if alpha.shape[0] == 1 and result.shape[0] > 1:
        alpha = alpha.expand(result.shape[0], -1, -1)
    if base.shape[0] != result.shape[0] or alpha.shape[0] != result.shape[0]:
        raise ValueError("CMK Z-Image Turbo INPAINT batch sizes do not match.")

    alpha = alpha.clamp(0.0, 1.0).unsqueeze(-1)
    return base * (1.0 - alpha) + result * alpha


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
        if (
            bool(PROCESS.get("boolean_inpaint_mode", False))
            or bool(PROCESS.get("hybrid_mode", False))
        ) and IMAGE is None:
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
        if clip is None:
            clip, _ = ensure_zit_text_encoder(MODEL)
        if model is None or clip is None or vae is None:
            raise ValueError(
                "CMK Sampler Prepare Z-Image Turbo -Pipe-: MODEL must provide "
                "model, text encoder and VAE."
            )

        model, clip, _, loaded_loras = CMKLoRATextLoader().load_loras(
            model,
            clip,
            opt_lora_syntax=str(PROCESS.get("lora_syntax", "") or ""),
            opt_lora_stack=PROCESS.get("lora_stack"),
        )

        width = _int(PROCESS.get("width", PROCESS.get("target_width")), 1024)
        height = _int(PROCESS.get("height", PROCESS.get("target_height")), 1024)
        prompt = _clean_text(
            PROCESS.get("effective_prompt_pos"),
            _clean_text(PROCESS.get("prompt_pos"), ""),
        )
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
        unload_requested = bool(
            PROCESS.get(
                "unload_models_after_use",
                PROCESS.get("unload_zit_after", True),
            )
        )
        text_encoder_status = "KEPT LOADED"
        if unload_requested:
            text_encoder_status = evict_zit_text_encoder(MODEL)
            # The optional LoRA path returns a CLIP clone. Do not retain that
            # clone in this frame or in the sampler pipe after conditioning.
            clip = None
            import gc

            gc.collect()
            try:
                import comfy.model_management

                comfy.model_management.cleanup_models_gc()
                comfy.model_management.soft_empty_cache(force=True)
            except Exception:
                pass
        inpaint_enabled = bool(PROCESS.get("boolean_inpaint_mode", False))
        hybrid_mode = bool(PROCESS.get("hybrid_mode", False))
        hybrid_inpaint_mode = bool(
            hybrid_mode and PROCESS.get("hybrid_inpaint_mode", False)
        )
        mask = PROCESS.get("mask")
        controlnet_enabled = bool(PROCESS.get("boolean_controlnet_enable", False))
        controlnet_model_patch = None
        inpaint_model_patch_resource = None
        inpaint_source_image = None
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
            controlnet_model_patch = model_patch
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
            inpaint_model_patch_resource = model_patch
            inpaint_source_image = PROCESS.get("inpaint_source_image")
            if inpaint_source_image is None:
                # Compatibility for externally constructed PROCESS pipes. The
                # CMK Create Image contract always supplies the unfilled source.
                inpaint_source_image = IMAGE
            model = _unwrap_node_output(
                _call_node_kwargs(
                    ("ZImageFunControlnet",),
                    model=model,
                    model_patch=model_patch,
                    vae=vae,
                    inpaint_image=inpaint_source_image,
                    mask=mask,
                    strength=1.0,
                )
            )
            latent = _unwrap_node_output(
                _call_node(("VAEEncode",), vae, inpaint_source_image)
            )
            if not isinstance(latent, dict) or "samples" not in latent:
                raise TypeError(
                    "CMK Z-Image Turbo Inpaint: VAEEncode returned no LATENT."
                )
            width = int(inpaint_source_image.shape[2])
            height = int(inpaint_source_image.shape[1])
        elif hybrid_mode:
            if IMAGE is None:
                raise ValueError(
                    "CMK Z-Image Turbo HYBRID requires the SDXL handoff IMAGE."
                )
            latent = _unwrap_node_output(
                _call_node(("VAEEncode",), vae, IMAGE)
            )
            if not isinstance(latent, dict) or "samples" not in latent:
                raise TypeError(
                    "CMK Z-Image Turbo HYBRID: VAEEncode returned no LATENT."
                )
            if bool(PROCESS.get("hybrid_inpaint_mode", False)):
                if mask is None:
                    raise ValueError(
                        "CMK Z-Image Turbo HYBRID INPAINT requires MASK in PROCESS."
                    )
                latent = _unwrap_node_output(
                    _call_node(("SetLatentNoiseMask",), latent, mask)
                )
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
                "zit_controlnet_model_patch": controlnet_model_patch,
                "zit_inpaint_model_patch": inpaint_model_patch_resource,
                "loaded_loras": loaded_loras,
                "text_encoder_status": text_encoder_status,
            }
        )
        if inpaint_enabled:
            sampler_pipe.update(
                {
                    "zit_inpaint_source_image": inpaint_source_image,
                    "zit_inpaint_finish_mask": mask,
                    "zit_inpaint_masked_finish": True,
                }
            )
        if hybrid_mode:
            hybrid_denoise = min(
                0.30,
                max(0.10, float(PROCESS.get("hybrid_zit_denoise", 0.20))),
            )
            hybrid_steps = _hybrid_finish_steps(steps, hybrid_denoise)
            sampler_pipe.update(
                {
                    "denoise": hybrid_denoise,
                    "steps_1st_pass": hybrid_steps,
                    "steps": hybrid_steps,
                    "hybrid_zit_base_steps": max(1, int(steps)),
                    "hybrid_zit_steps": hybrid_steps,
                    # Preserve the decoded SDXL handoff as the authoritative
                    # Before image for the Hybrid ZIT compare provider.
                    "hybrid_source_image": IMAGE,
                }
            )
            if bool(PROCESS.get("hybrid_inpaint_mode", False)):
                sampler_pipe.update(
                    {
                        "hybrid_finish_mask": mask,
                        "hybrid_masked_finish": True,
                    }
                )
        lines = [
            "STATUS          : PREPARED",
            "MODEL FAMILY    : ZIT",
            f"MODE            : {'HYBRID INPAINT FINISH' if hybrid_inpaint_mode else ('HYBRID FINISH' if hybrid_mode else ('INPAINT (EXPERIMENTAL)' if inpaint_enabled else ('CONTROLNET' if controlnet_enabled else 'TEXT2IMAGE')))}",
            f"SIZE            : {width} × {height}",
            f"STEPS           : {int(sampler_pipe['steps'])}",
            f"CFG             : {float(cfg):g}",
            f"SAMPLER         : {sampler}",
            f"SCHEDULER       : {scheduler}",
            f"MODEL SHIFT     : {float(model_shift):g}",
            f"SEED            : {int(seed)}",
            f"TEXT ENCODER    : {text_encoder_status}",
            "",
            "POSITIVE PROMPT:",
            prompt,
        ]
        if hybrid_mode:
            lines[3:3] = [
                f"SDXL HANDOFF   : {int(PROCESS.get('hybrid_sdxl_handoff', 90))}%",
                f"DENOISE         : {sampler_pipe['denoise']:.2f}",
                f"ZIT FINISH STEPS: {sampler_pipe['hybrid_zit_steps']} / {sampler_pipe['hybrid_zit_base_steps']}",
            ]
            if bool(PROCESS.get("hybrid_inpaint_mode", False)):
                lines[6:6] = [
                    "NOISE MASK      : ENABLED",
                    "VISIBLE RESULT  : MASK ONLY",
                ]
        if loaded_loras:
            lines[9:9] = ["", "LORAS:", loaded_loras]
        if translation.status not in {"empty", "disabled", "not_configured"}:
            lines.insert(9, translation.log_line("POS"))
        if inpaint_enabled:
            lines[3:3] = [
                f"INPAINT PATCH   : {inpaint_model_patch}",
                "SAMPLING SCOPE  : FULL FRAME",
                "VISIBLE RESULT  : MASK ONLY",
            ]
        log = cmk_add_block(LOG, "Z-Image Turbo Prepare", 40, lines, True)
        summary = "\n".join(lines)
        diagnostic = make_diagnostic_payload(
            title="Z-Image Turbo Prepare",
            node="CMK Sampler Prepare Z-Image Turbo -Pipe-",
            previews=[],
            summary=f"{width}x{height} | {int(sampler_pipe['steps'])} steps | CFG {float(cfg):g}",
            details=summary,
            mode=(
                "Hybrid Inpaint Finish" if hybrid_inpaint_mode
                else "Hybrid Finish" if hybrid_mode
                else "Inpaint (Experimental)" if inpaint_enabled
                else ("ControlNet" if controlnet_enabled else "Text2Image")
            ),
            metadata={
                "model_family": "z_image_turbo",
                "width": width,
                "height": height,
                "steps": int(sampler_pipe["steps"]),
                "base_steps": max(1, int(steps)),
                "cfg": float(cfg),
                "sampler": str(sampler),
                "scheduler": str(scheduler),
                "model_shift": float(model_shift),
                "seed": int(seed),
                "inpaint": inpaint_enabled,
                "hybrid": hybrid_mode,
                "denoise": sampler_pipe["denoise"],
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

        hybrid_masked_finish = bool(
            PROCESS.get("hybrid_inpaint_mode", False)
            or SAMPLED.get("hybrid_masked_finish", False)
        )
        direct_masked_finish = bool(
            SAMPLED.get("zit_inpaint_masked_finish", False)
        ) and not hybrid_masked_finish
        if hybrid_masked_finish:
            source_image = SAMPLED.get("hybrid_source_image")
            finish_mask = SAMPLED.get("hybrid_finish_mask")
            if source_image is None:
                raise ValueError(
                    "CMK Z-Image Turbo HYBRID INPAINT lost the SDXL handoff image."
                )
            if finish_mask is None:
                raise ValueError(
                    "CMK Z-Image Turbo HYBRID INPAINT lost the process mask."
                )
            image = _hybrid_masked_composite(source_image, image, finish_mask)
        elif direct_masked_finish:
            source_image = SAMPLED.get("zit_inpaint_source_image")
            finish_mask = SAMPLED.get("zit_inpaint_finish_mask")
            if source_image is None:
                raise ValueError(
                    "CMK Z-Image Turbo INPAINT lost the unfilled source image."
                )
            if finish_mask is None:
                raise ValueError(
                    "CMK Z-Image Turbo INPAINT lost the authoritative process mask."
                )
            image = _hybrid_masked_composite(source_image, image, finish_mask)

        process = dict(PROCESS)
        hybrid_mode = bool(PROCESS.get("hybrid_mode", False))
        # ``hybrid_mode`` describes the active generation family, whereas
        # ``generation_mode`` describes the user-visible operation and output
        # class.  Hybrid Inpaint deliberately disables boolean_inpaint_mode in
        # the ZIT branch because ZIT performs a normal masked Img2Img finish;
        # that execution detail must not turn the finished image into
        # Text2Image provenance.
        hybrid_inpaint_mode = bool(
            hybrid_mode and PROCESS.get("hybrid_inpaint_mode", False)
        )
        generation_mode = str(
            PROCESS.get("generation_mode", "") or ""
        ).strip().casefold()
        if hybrid_inpaint_mode:
            generation_mode = "inpaint"
        elif generation_mode not in {"inpaint", "text2image"}:
            generation_mode = (
                "inpaint"
                if bool(PROCESS.get("boolean_inpaint_mode", False))
                else "text2image"
            )
        process.update(
            {
                "model_family": "z_image_turbo",
                "generation_mode": generation_mode,
                "z_image_sampled": True,
            }
        )
        unload_requested = bool(
            PROCESS.get(
                "unload_models_after_use",
                PROCESS.get("unload_zit_after", True),
            )
        )
        unload_status = "KEPT LOADED"
        if unload_requested:
            diffusion_model = MODEL.get("model_patched") or MODEL.get("model")
            if diffusion_model is None:
                unload_status = "NOT AVAILABLE"
            else:
                from .loaders.checkpoint_vae_loader import unload_all_phase_models

                unload_status = unload_all_phase_models("10 ZIT complete")
        lines = [
            "STATUS          : DECODED",
            "MODEL FAMILY    : Z-IMAGE TURBO",
            f"SIZE            : {process.get('width')} × {process.get('height')}",
            f"ZIT MODEL       : {unload_status}",
        ]
        controlnet_model_status = str(
            SAMPLED.get("controlnet_model_status", "NOT ACTIVE")
        )
        lines.append(f"CONTROLNET MODEL: {controlnet_model_status}")
        if hybrid_masked_finish:
            lines.extend(
                [
                    "HYBRID COMPOSITE: MASK ONLY",
                    "OUTSIDE MASK    : SDXL HANDOFF (UNCHANGED)",
                ]
            )
        elif direct_masked_finish:
            lines.extend(
                [
                    "INPAINT COMPOSITE: MASK ONLY",
                    "OUTSIDE MASK    : SOURCE IMAGE (UNCHANGED)",
                ]
            )
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
                "Hybrid Inpaint Finish" if hybrid_inpaint_mode
                else "Hybrid Finish" if hybrid_mode
                else "Inpaint (Experimental)" if generation_mode == "inpaint"
                else "Text2Image"
            ),
            metadata={
                "model_family": "z_image_turbo",
                "unload_models_after_use": unload_requested,
                "zit_model_status": unload_status,
                "hybrid_masked_finish": hybrid_masked_finish,
                "direct_masked_finish": direct_masked_finish,
                "hybrid_inpaint_mode": hybrid_inpaint_mode,
                "generation_mode": generation_mode,
                "controlnet_model_status": controlnet_model_status,
            },
        )
        return (MODEL, process, image, log, diagnostic)
