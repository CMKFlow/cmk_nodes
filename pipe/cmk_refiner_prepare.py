from __future__ import annotations

from ..cmk_common import SAMPLERS, SCHEDULERS
from ..loader.cmk_lora_text_loader import CMKLoRATextLoader
from .cmk_log_pipe import cmk_add_block, cmk_format_loras
from ..utils.cmk_translation import translate_prompt
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
from ..utils.cmk_timing import cmk_timed_call


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
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS": ("CMK_PROCESS_SDXL", {"lazy": True}),
                "SAMPLED": ("CMK_SAMPLED_PIPE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "use_prompt_lora_from_sampler": ("BOOLEAN", {"default": True}),
                "use_lora_from_1st_pass": ("BOOLEAN", {"default": False}),
                "lora_name": (loras, {"default": default_lora}),
                "strength_model": ("FLOAT", {"default": 1.00, "min": -20.0, "max": 20.0, "step": 0.01, "advanced": True}),
                "strength_clip": ("FLOAT", {"default": 1.00, "min": -20.0, "max": 20.0, "step": 0.01, "advanced": True}),
                "sampling_source": (["1st pass", "Local settings"], {
                    "default": "1st pass", "advanced": True,
                    "tooltip": "Inherit steps, sampler and scheduler from the first pass.",
                }),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
                "start_without_instantid": ("FLOAT", {
                    "default": 80.0, "min": 0.0, "max": 100.0, "step": 1.0, "advanced": True,
                    "tooltip": "Refiner start used when InstantID is disabled.",
                }),
                "start_with_instantid": ("FLOAT", {
                    "default": 90.0, "min": 0.0, "max": 100.0, "step": 1.0, "advanced": True,
                    "tooltip": "Refiner start used automatically when InstantID is enabled.",
                }),
                "cfg": ("FLOAT", {"default": 4.8, "min": 0.0, "max": 10.0, "step": 0.1}),
                "sampler": (SAMPLERS, {
                    **({"default": "euler"} if "euler" in SAMPLERS else {}),
                    "advanced": True,
                }),
                "sampling": (SAMPLING_MODES, {"default": "lcm", "advanced": True}),
                "zsnr": ("BOOLEAN", {"default": False, "advanced": True}),
                "pag_scale": ("FLOAT", {"default": 2.50, "min": 0.0, "max": 20.0, "step": 0.05, "advanced": True}),
                "scheduler": (SCHEDULERS, {
                    **({"default": "simple"} if "simple" in SCHEDULERS else {}),
                    "advanced": True,
                }),
            },
            "optional": {
                "prompt_pos_input": ("STRING", {"forceInput": True}),
                "prompt_neg_input": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CMK_REFINER_PIPE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("REFINER", "LOG", "diagnostic")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Developer/Pipe/Prepare"

    def check_lazy_status(
        self,
        MODEL=None,
        PROCESS=None,
        SAMPLED=None,
        LOG=None,
        **kwargs,
    ):
        # Complete the first pass before loading the separate Refiner model.
        # On unified-memory systems, resolving both branches in one lazy round
        # retains two multi-gigabyte SDXL model families at the same time.
        if SAMPLED is None:
            return ["SAMPLED"]
        if LOG is None:
            return ["LOG"]
        if PROCESS is None:
            return ["PROCESS"]
        if MODEL is None:
            return ["MODEL"]
        return []

    @cmk_timed_call("20 REFINER PREPARE")
    def prepare(
        self,
        MODEL,
        PROCESS,
        SAMPLED,
        use_prompt_lora_from_sampler,
        use_lora_from_1st_pass,
        lora_name,
        strength_model,
        strength_clip,
        sampling_source,
        steps,
        start_without_instantid,
        start_with_instantid,
        cfg,
        sampler,
        sampling,
        zsnr,
        pag_scale,
        scheduler,
        LOG=None,
        prompt_pos_input=None,
        prompt_neg_input=None,
        prompt_pos="",
        prompt_neg="",
    ):
        if not isinstance(SAMPLED, dict):
            raise TypeError("CMK Refiner Prepare SDXL -Pipe-: SAMPLED must be a CMK sampled pipe")

        latent = SAMPLED.get("latent_1st_pass", SAMPLED.get("latent_image"))
        if latent is None:
            raise ValueError("CMK Refiner Prepare SDXL -Pipe-: SAMPLED contains no sampler latent")

        if not isinstance(MODEL, dict):
            raise TypeError("CMK Refiner Prepare SDXL -Pipe-: MODEL must be a CMK model pipe")
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Refiner Prepare SDXL -Pipe-: PROCESS must be a CMK process pipe")

        model = MODEL.get("model")
        clip = MODEL.get("clip")
        vae = MODEL.get("vae")
        if model is None or clip is None or vae is None:
            raise ValueError("CMK Refiner Prepare SDXL -Pipe-: MODEL must provide model, clip and vae")

        width = _int(SAMPLED.get("width", PROCESS.get("width", PROCESS.get("target_width"))), 1024)
        height = _int(SAMPLED.get("height", PROCESS.get("height", PROCESS.get("target_height"))), 1024)
        size_cond_factor = _int(SAMPLED.get("size_cond_factor", PROCESS.get("size_cond_factor")), 4)

        helper = CMKSamplerPrepareSDXLPipe()
        inherit_prompt = bool(use_prompt_lora_from_sampler)
        inherit_lora = bool(use_lora_from_1st_pass)
        local_prompt_pos = prompt_pos if prompt_pos_input is None else prompt_pos_input
        local_prompt_neg = prompt_neg if prompt_neg_input is None else prompt_neg_input

        if inherit_prompt:
            selected_prompt_pos = _clean_text(SAMPLED.get("prompt_pos", PROCESS.get("prompt_pos")), "")
            selected_prompt_neg = _clean_text(SAMPLED.get("prompt_neg", PROCESS.get("prompt_neg")), "")
        else:
            selected_prompt_pos = _clean_text(local_prompt_pos, "")
            selected_prompt_neg = _clean_text(local_prompt_neg, "")

        if inherit_lora:
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

        translation_pos = translate_prompt(selected_prompt_pos)
        translation_neg = translate_prompt(selected_prompt_neg)
        conditioning_pos = _validate_conditioning(
            helper._encode_sdxl_plus(clip, width, height, size_cond_factor, translation_pos.text),
            "refiner_conditioning_pos",
        )
        conditioning_neg = _validate_conditioning(
            helper._encode_sdxl_plus(clip, width, height, size_cond_factor, translation_neg.text),
            "refiner_conditioning_neg",
        )

        inherit_sampling = (
            sampling_source is True
            or str(sampling_source).strip().lower() == "1st pass"
        )
        local_steps = max(1, _int(steps, 25))
        refiner_steps = max(
            1,
            _int(SAMPLED.get("steps_1st_pass", SAMPLED.get("steps")), local_steps),
        ) if inherit_sampling else local_steps
        instantid_active = bool(PROCESS.get("instantid_enabled", False))
        start_without_id = min(100.0, max(0.0, _float(start_without_instantid, 80.0)))
        start_with_id = min(100.0, max(0.0, _float(start_with_instantid, 90.0)))
        start_pct = start_with_id if instantid_active else start_without_id
        start_source = "InstantID enabled" if instantid_active else "InstantID disabled"
        start_at_step = int(refiner_steps * start_pct / 100.0)
        end_at_step = refiner_steps
        selected_sampler = (
            str(SAMPLED.get("sampler", sampler)) if inherit_sampling else str(sampler)
        )
        selected_scheduler = (
            str(SAMPLED.get("scheduler", scheduler)) if inherit_sampling else str(scheduler)
        )

        refiner_pipe = dict(SAMPLED)
        refiner_pipe.update({
            "refiner_checkpoint_name": str(MODEL.get("ckpt_name", "")),
            "refiner_vae_name": str(MODEL.get("vae_name", "")),
            "refiner_model": model,
            "refiner_clip": clip,
            "refiner_vae": vae,
            "refiner_vae_source": "refiner",
            "refiner_conditioning_pos": conditioning_pos,
            "refiner_conditioning_neg": conditioning_neg,
            "refiner_latent_image": latent,
            "refiner_source_image": SAMPLED.get("image_1st_pass"),
            "refiner_seed": _int(SAMPLED.get("seed", PROCESS.get("seed")), 0),
            "refiner_steps": refiner_steps,
            "refiner_cfg": _float(cfg, 4.8),
            "refiner_sampler": selected_sampler,
            "refiner_scheduler": selected_scheduler,
            "refiner_use_1st_pass_sampling": inherit_sampling,
            "refiner_steps_source": "sampled" if inherit_sampling else "local",
            "refiner_start_percent": start_pct,
            "refiner_start_percent_without_instantid": start_without_id,
            "refiner_start_percent_with_instantid": start_with_id,
            "refiner_start_percent_source": start_source,
            "refiner_start_at_step": start_at_step,
            "refiner_end_at_step": end_at_step,
            "refiner_prompt_pos": selected_prompt_pos,
            "refiner_prompt_neg": selected_prompt_neg,
            "refiner_use_prompt_lora_from_sampler": inherit_prompt and inherit_lora,
            "refiner_use_prompt_from_1st_pass": inherit_prompt,
            "refiner_use_lora_from_1st_pass": inherit_lora,
            "refiner_active_loras": _clean_text(
                SAMPLED.get("lora_syntax", SAMPLED.get("active_loras")), ""
            ) if inherit_lora else local_lora,
            "refiner_lora_stack": lora_stack,
            "refiner_loaded_loras": loaded_loras,
        })

        details = (
            "CMK Refiner Prepare SDXL -Pipe- | "
            f"checkpoint={MODEL.get('ckpt_name', '')} | vae={MODEL.get('vae_name', '')} | "
            f"{width}x{height} | steps={refiner_steps} | cfg={_float(cfg, 4.8)} | "
            f"sampler={selected_sampler} | scheduler={selected_scheduler} | "
            f"start={start_pct:.1f}% ({start_source}) | "
            f"sampling_from_1st_pass={inherit_sampling} | steps_source={'SAMPLED' if inherit_sampling else 'LOCAL'} | "
            f"prompt_from_1st_pass={inherit_prompt} | lora_from_1st_pass={inherit_lora}"
        )
        refiner_pipe["refiner_prepare_log"] = details

        prompt_source_label = "1ST PASS" if inherit_prompt else "LOCAL"
        lora_source_label = "1ST PASS" if inherit_lora else "LOCAL"
        log_lines = [
            "STATUS          : PREPARED",
            "MODEL SOURCE    : MODEL",
            f"CHECKPOINT      : {MODEL.get('ckpt_name', '')}",
            f"VAE             : {MODEL.get('vae_name', '')}",
            f"SIZE            : {width} × {height}",
            f"STEPS           : {refiner_steps}",
            f"STEPS SOURCE    : {'SAMPLED' if inherit_sampling else 'LOCAL'}",
            f"CFG             : {_float(cfg, 4.8)}",
            f"SAMPLER         : {selected_sampler}",
            f"SCHEDULER       : {selected_scheduler}",
            f"SAMPLING SOURCE : {'1ST PASS' if inherit_sampling else 'LOCAL'}",
            f"REFINER START   : {start_pct:.1f}%",
            f"START SOURCE    : {start_source}",
            f"START NO ID     : {start_without_id:.1f}%",
            f"START INSTANTID : {start_with_id:.1f}%",
            f"PROMPT SOURCE   : {prompt_source_label}",
            f"LORA SOURCE     : {lora_source_label}",
        ]
        for label, result in (("POS", translation_pos), ("NEG", translation_neg)):
            if result.status not in {"empty", "disabled", "not_configured"}:
                log_lines.append(result.log_line(label))
        if not inherit_lora:
            log_lines.extend(["", "LOCAL LORAS:", cmk_format_loras(local_lora)])
        if not inherit_prompt:
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
                "sampler": selected_sampler,
                "scheduler": selected_scheduler,
                "sampling_source": "1st_pass" if inherit_sampling else "local",
                "prompt_source": "1st_pass" if inherit_prompt else "local",
                "lora_source": "1st_pass" if inherit_lora else "local",
                "seed": refiner_pipe["refiner_seed"],
            },
        )
        return (refiner_pipe, log_pipe, diagnostic)
