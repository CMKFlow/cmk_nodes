from __future__ import annotations

from ..cmk_common import SAMPLERS, SCHEDULERS
from ..loader.cmk_lora_text_loader import CMKLoRATextLoader
from .cmk_log_pipe import cmk_add_block, cmk_format_loras
from .cmk_sampler_prepare import (
    SAMPLING_MODES,
    _clean_text,
    _float,
    _int,
    _safe_lora_list,
    _unwrap_node_output,
    _validate_conditioning,
    CMKSamplerPrepareSDXLPipe,
)
from ..utils.cmk_diagnostic import make_diagnostic_payload


def _safe_default(items, preferred):
    if preferred in items:
        return preferred
    return items[0] if items else "None"


class CMKRefinerPrepareSDXLPipe:
    """Create the isolated REFINER working pipe.

    Public contract:
        MODEL + PROCESS + SAMPLED + LOG -> REFINER + LOG + diagnostic

    MODEL is read-only. The sampler result is consumed exclusively through
    SAMPLED; no IMAGE input is required because the Refiner works on latent data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        available_loras = _safe_lora_list()
        loras = ["None"] + [
            item for item in available_loras
            if str(item).strip().lower() != "none"
        ]
        # A Refiner LoRA must be an explicit user choice. Applying an SDXL base
        # LoRA to an incompatible Refiner architecture can produce extensive
        # shape-mismatch errors and unnecessary model patching.
        default_lora = "None"

        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE",),
                "PROCESS": ("CMK_PIPE",),
                "SAMPLED": ("CMK_SAMPLED_PIPE",),
                "use_prompt_lora_from_sampler": ("BOOLEAN", {"default": False}),
                "lora_name": (loras, {"default": default_lora}),
                "strength_model": ("FLOAT", {"default": 1.00, "min": -20.0, "max": 20.0, "step": 0.01, "advanced": True}),
                "strength_clip": ("FLOAT", {"default": 1.00, "min": -20.0, "max": 20.0, "step": 0.01, "advanced": True}),
                "prompt_pos": ("STRING", {"default": "", "multiline": True}),
                "prompt_neg": ("STRING", {"default": "", "multiline": True}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
                "start_percent": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 100.0, "step": 1.0, "advanced": True}),
                "cfg": ("FLOAT", {"default": 4.8, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler": (SAMPLERS, {"default": "euler"} if "euler" in SAMPLERS else {}),
                "sampling": (SAMPLING_MODES, {"default": "lcm", "advanced": True}),
                "zsnr": ("BOOLEAN", {"default": False, "advanced": True}),
                "pag_scale": ("FLOAT", {"default": 2.50, "min": 0.0, "max": 20.0, "step": 0.05, "advanced": True}),
                "scheduler": (SCHEDULERS, {"default": "simple"} if "simple" in SCHEDULERS else {}),
            },
            "optional": {
                "LOG": ("CMK_LOG_PIPE",),
            },
        }

    RETURN_TYPES = ("CMK_REFINER_PIPE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("REFINER", "LOG", "diagnostic")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Developer/Pipe/Prepare"

    def prepare(
        self,
        MODEL,
        PROCESS,
        SAMPLED,
        use_prompt_lora_from_sampler,
        lora_name,
        strength_model,
        strength_clip,
        prompt_pos,
        prompt_neg,
        steps,
        start_percent,
        cfg,
        sampler,
        sampling,
        zsnr,
        pag_scale,
        scheduler,
        LOG=None,
    ):
        if not isinstance(MODEL, dict):
            raise TypeError("CMK Refiner Prepare SDXL -Pipe-: MODEL must be a CMK model pipe")
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Refiner Prepare SDXL -Pipe-: PROCESS must be a CMK process pipe")
        if not isinstance(SAMPLED, dict):
            raise TypeError("CMK Refiner Prepare SDXL -Pipe-: SAMPLED must be a CMK sampled pipe")

        latent = SAMPLED.get("latent_1st_pass", SAMPLED.get("latent_image"))
        if latent is None:
            raise ValueError("CMK Refiner Prepare SDXL -Pipe-: SAMPLED contains no sampler latent")

        model = MODEL.get("model")
        clip = MODEL.get("clip")
        vae = MODEL.get("vae")
        if model is None or clip is None or vae is None:
            raise ValueError("CMK Refiner Prepare SDXL -Pipe-: MODEL must provide model, clip and vae")

        width = _int(SAMPLED.get("width", PROCESS.get("width", PROCESS.get("target_width"))), 1024)
        height = _int(SAMPLED.get("height", PROCESS.get("height", PROCESS.get("target_height"))), 1024)
        size_cond_factor = _int(SAMPLED.get("size_cond_factor", PROCESS.get("size_cond_factor")), 4)

        helper = CMKSamplerPrepareSDXLPipe()
        use_sampler_context = bool(use_prompt_lora_from_sampler)

        if use_sampler_context:
            selected_prompt_pos = _clean_text(SAMPLED.get("prompt_pos", PROCESS.get("prompt_pos")), "")
            selected_prompt_neg = _clean_text(SAMPLED.get("prompt_neg", PROCESS.get("prompt_neg")), "")
            lora_syntax = _clean_text(
                SAMPLED.get("lora_syntax", SAMPLED.get("active_loras", PROCESS.get("lora_syntax", PROCESS.get("active_loras")))),
                "",
            )
            lora_stack = SAMPLED.get("lora_stack", PROCESS.get("lora_stack"))
            model, clip, _, loaded_loras = CMKLoRATextLoader().load_loras(
                model, clip, lora_syntax, lora_stack
            )
            local_lora = ""
        else:
            selected_prompt_pos = _clean_text(prompt_pos, "")
            selected_prompt_neg = _clean_text(prompt_neg, "")
            model, clip, local_lora = helper._apply_single_lora(
                model,
                clip,
                lora_name,
                _float(strength_model, 1.0),
                _float(strength_clip, 1.0),
            )
            loaded_loras = []
            lora_stack = None

        model = _unwrap_node_output(model)
        clip = _unwrap_node_output(clip)
        model = _unwrap_node_output(helper._apply_pag(model, _float(pag_scale, 2.5)))
        model = _unwrap_node_output(helper._apply_sampling(model, str(sampling), bool(zsnr)))

        conditioning_pos = _validate_conditioning(
            helper._encode_sdxl_plus(clip, width, height, size_cond_factor, selected_prompt_pos),
            "refiner_conditioning_pos",
        )
        conditioning_neg = _validate_conditioning(
            helper._encode_sdxl_plus(clip, width, height, size_cond_factor, selected_prompt_neg),
            "refiner_conditioning_neg",
        )

        refiner_steps = max(1, _int(steps, 25))
        start_pct = min(100.0, max(0.0, _float(start_percent, 80.0)))
        start_at_step = int(refiner_steps * start_pct / 100.0)
        end_at_step = refiner_steps

        refiner_pipe = dict(SAMPLED)
        refiner_pipe.update({
            "refiner_checkpoint_name": str(MODEL.get("ckpt_name", "")),
            "refiner_vae_name": str(MODEL.get("vae_name", "")),
            "refiner_model": model,
            "refiner_clip": clip,
            "refiner_vae": vae,
            "refiner_conditioning_pos": conditioning_pos,
            "refiner_conditioning_neg": conditioning_neg,
            "refiner_latent_image": latent,
            "refiner_seed": _int(SAMPLED.get("seed", PROCESS.get("seed")), 0),
            "refiner_steps": refiner_steps,
            "refiner_cfg": _float(cfg, 4.8),
            "refiner_sampler": sampler,
            "refiner_scheduler": scheduler,
            "refiner_start_percent": start_pct,
            "refiner_start_at_step": start_at_step,
            "refiner_end_at_step": end_at_step,
            "refiner_prompt_pos": selected_prompt_pos,
            "refiner_prompt_neg": selected_prompt_neg,
            "refiner_use_prompt_lora_from_sampler": use_sampler_context,
            "refiner_active_loras": _clean_text(
                SAMPLED.get("lora_syntax", SAMPLED.get("active_loras")), ""
            ) if use_sampler_context else local_lora,
            "refiner_lora_stack": lora_stack,
            "refiner_loaded_loras": loaded_loras,
        })

        details = (
            "CMK Refiner Prepare SDXL -Pipe- | "
            f"checkpoint={MODEL.get('ckpt_name', '')} | vae={MODEL.get('vae_name', '')} | "
            f"{width}x{height} | steps={refiner_steps} | cfg={_float(cfg, 4.8)} | "
            f"sampler={sampler} | scheduler={scheduler} | start={start_pct:.1f}% | "
            f"prompt_lora_from_sampler={use_sampler_context}"
        )
        refiner_pipe["refiner_prepare_log"] = details

        source_label = "SOURCE" if use_sampler_context else "LOCAL"
        log_lines = [
            "STATUS          : PREPARED",
            "MODEL SOURCE    : MODEL",
            f"CHECKPOINT      : {MODEL.get('ckpt_name', '')}",
            f"VAE             : {MODEL.get('vae_name', '')}",
            f"SIZE            : {width} × {height}",
            f"STEPS           : {refiner_steps}",
            f"CFG             : {_float(cfg, 4.8)}",
            f"SAMPLER         : {sampler}",
            f"SCHEDULER       : {scheduler}",
            f"REFINER START   : {start_pct:.1f}%",
            f"PROMPT SOURCE   : {source_label}",
            f"LORA SOURCE     : {source_label}",
        ]
        if not use_sampler_context:
            log_lines.extend(["", "LOCAL LORAS:", cmk_format_loras(local_lora)])
            if selected_prompt_pos:
                log_lines.extend(["", "POSITIVE PROMPT:", selected_prompt_pos])
            if selected_prompt_neg:
                log_lines.extend(["", "NEGATIVE PROMPT:", selected_prompt_neg])
        log_pipe = cmk_add_block(LOG, "Refiner Prepare", 50, log_lines, True)
        diagnostic = make_diagnostic_payload(
            title="Refiner Prepare",
            node="CMK Refiner Prepare SDXL -Pipe-",
            previews=[],
            summary=f"{width}x{height} | {refiner_steps} steps | start {start_pct:.1f}%",
            details=details,
            mode="refiner",
            metadata={
                "checkpoint": MODEL.get("ckpt_name", ""),
                "vae": MODEL.get("vae_name", ""),
                "sampler": sampler,
                "scheduler": scheduler,
                "seed": refiner_pipe["refiner_seed"],
            },
        )
        return (refiner_pipe, log_pipe, diagnostic)
