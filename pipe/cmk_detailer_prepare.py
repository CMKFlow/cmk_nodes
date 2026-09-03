from __future__ import annotations

from ..cmk_common import SAMPLERS, SCHEDULERS
from ..loader.cmk_lora_text_loader import CMKLoRATextLoader
from .cmk_sampler_prepare import (
    SAMPLING_MODES,
    _clean_text,
    _float,
    _int,
    _safe_lora_list,
    _unwrap_node_output,
    _call_node_kwargs,
    _validate_conditioning,
    CMKSamplerPrepareSDXLPipe,
)
from .cmk_log_pipe import cmk_add_block, cmk_format_loras
from ..utils.cmk_translation import translate_prompt
from ..utils.cmk_diagnostic import make_diagnostic_payload
from ..engine.native_detailer import CMKSAMLoader



def _sam_loader_input_specs():
    required = CMKSAMLoader.INPUT_TYPES()["required"]
    return required["model_name"], required["device_mode"]

def _safe_default(items, preferred):
    if preferred in items:
        return preferred
    return items[0] if items else "None"


class CMKDetailerPreparePipe:
    """Create the isolated DETAILER working pipe.

    Public contract:
        MODEL + PROCESS + IMAGE + LOG -> DETAILER + LOG + diagnostic

    MODEL is read-only. IMAGE is the authoritative image payload. PROCESS is
    read only for shared CMK context such as prompt/LoRA offers and seed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        available_loras = _safe_lora_list()
        loras = ["None"] + [
            item for item in available_loras
            if str(item).strip().lower() != "none"
        ]
        # Detailer LoRAs are optional module-local resources. They must never be
        # forced merely because at least one LoRA exists on disk.
        default_lora = "None"
        sam_model_spec, sam_device_spec = _sam_loader_input_specs()

        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS": ("CMK_PROCESS_SDXL", {"lazy": True}),
                "IMAGE": ("IMAGE", {"lazy": True}),

                "sam_model_name": sam_model_spec,
                "sam_device_mode": sam_device_spec,

                "detailer_global_enable": ("BOOLEAN", {"default": False}),
                # The legacy key remains stable for saved workflows, but now
                # controls prompt inheritance only.
                "use_prompt_lora_from_sampler": ("BOOLEAN", {"default": True}),
                "use_lora_from_1st_pass": ("BOOLEAN", {"default": False}),
                "use_1st_pass_sampling": ("BOOLEAN", {"default": True}),

                "lora_name": (loras, {"default": default_lora}),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.00, "min": -20.0, "max": 20.0, "step": 0.01, "advanced": True},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.00, "min": -20.0, "max": 20.0, "step": 0.01, "advanced": True},
                ),

                "prompt_pos": ("STRING", {"default": "", "multiline": True}),
                "prompt_neg": ("STRING", {"default": "", "multiline": True}),

                "steps": ("INT", {"default": 30, "min": 1, "max": 10000, "step": 1}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler": (SAMPLERS, {
                    **({"default": "euler"} if "euler" in SAMPLERS else {}),
                    "advanced": True,
                }),
                "scheduler": (SCHEDULERS, {
                    **({"default": "simple"} if "simple" in SCHEDULERS else {}),
                    "advanced": True,
                }),

                "stop_at_clip_layer": (
                    "INT",
                    {"default": -2, "min": -24, "max": -1, "step": 1, "advanced": True},
                ),
                "pag_scale": (
                    "FLOAT",
                    {"default": 2.50, "min": 0.0, "max": 20.0, "step": 0.05, "advanced": True},
                ),
                "sampling": (SAMPLING_MODES, {"default": "v_prediction", "advanced": True}),
                "zsnr": ("BOOLEAN", {"default": True, "advanced": True}),

                "freeu_enabled": ("BOOLEAN", {"default": True}),
                "freeu_b1": (
                    "FLOAT",
                    {"default": 1.30, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True},
                ),
                "freeu_b2": (
                    "FLOAT",
                    {"default": 1.40, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True},
                ),
                "freeu_s1": (
                    "FLOAT",
                    {"default": 0.90, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True},
                ),
                "freeu_s2": (
                    "FLOAT",
                    {"default": 0.20, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True},
                ),
            },
            "optional": {
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("CMK_DETAILER_PIPE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("DETAILER", "LOG", "diagnostic")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Developer/Pipe/Prepare"

    def check_lazy_status(self, MODEL=None, PROCESS=None, IMAGE=None, LOG=None, detailer_global_enable=False, **kwargs):
        # Resolve the authoritative upstream result first. Requesting MODEL in
        # the same lazy round as IMAGE lets ComfyUI materialize SDXL/Refiner
        # model branches while upstream sampling is still active, causing a
        # severe unified-memory peak on Apple Silicon.
        upstream_needed = []
        if IMAGE is None:
            upstream_needed.append("IMAGE")
        if LOG is None:
            upstream_needed.append("LOG")
        if upstream_needed:
            return upstream_needed

        if not bool(detailer_global_enable):
            return []

        model_needed = []
        if PROCESS is None:
            model_needed.append("PROCESS")
        if MODEL is None:
            model_needed.append("MODEL")
        return model_needed

    @staticmethod
    def _encode(clip, text):
        helper = CMKSamplerPrepareSDXLPipe()
        # Standard CLIP Text Encode is the intended encoder for this SDXL-only module.
        try:
            from .cmk_sampler_prepare import _call_node

            result = _call_node(("CLIPTextEncode",), clip, text)
            return _validate_conditioning(_unwrap_node_output(result), "detailer_conditioning")
        except Exception as exc:
            raise RuntimeError(f"CMK Detailer Prepare -Pipe-: CLIP Text Encode failed: {exc}") from exc

    def prepare(
        self,
        MODEL,
        PROCESS,
        IMAGE,
        sam_model_name,
        sam_device_mode,
        detailer_global_enable,
        use_prompt_lora_from_sampler,
        use_lora_from_1st_pass,
        use_1st_pass_sampling,
        lora_name,
        strength_model,
        strength_clip,
        prompt_pos,
        prompt_neg,
        steps,
        cfg,
        sampler,
        scheduler,
        stop_at_clip_layer,
        pag_scale,
        sampling,
        zsnr,
        freeu_enabled,
        freeu_b1,
        freeu_b2,
        freeu_s1,
        freeu_s2,
        LOG=None,
    ):
        if IMAGE is None:
            raise ValueError("CMK Detailer Prepare -Pipe-: IMAGE is missing")
        global_enable=bool(detailer_global_enable)
        image=IMAGE
        if not global_enable:
            detailer_pipe={
                "source_pipe": {}, "_source_pipe": {},
                "detailer_image": image,
                "detailer_global_enable": False,
                "boolean_detailer_enable": False,
                "detailer_prepare_bypassed": True,
            }
            lines=["STATUS          : GLOBAL DISABLED","RESULT          : LIGHTWEIGHT DETAILER PASSTHROUGH","MODEL REQUEST   : SKIPPED","PROCESS REQUEST : SKIPPED","SAM LOAD        : SKIPPED","LORA / CLIP     : SKIPPED","CONDITIONING    : SKIPPED"]
            print(
                "[CMK Detailer Prepare -Pipe-] GLOBAL DISABLED "
                "-> MODEL / PROCESS / SAM / LORA / CONDITIONING SKIPPED"
            )
            log_pipe=cmk_add_block(LOG,"Detailer Prepare",70,lines,True)
            diagnostic=make_diagnostic_payload(title="Detailer Prepare",node="CMK Detailer Prepare -Pipe-",previews=[image],summary="global disabled | expensive preparation skipped",details="\n".join(lines),mode="disabled / passthrough",metadata={"global_enabled":False})
            return (detailer_pipe,log_pipe,diagnostic)
        if not isinstance(MODEL, dict): raise TypeError("CMK Detailer Prepare -Pipe-: MODEL must be a CMK model pipe")
        if not isinstance(PROCESS, dict): raise TypeError("CMK Detailer Prepare -Pipe-: PROCESS must be a CMK process pipe")
        source_pipe=dict(PROCESS)
        model_base=MODEL.get("model")
        clip_base = MODEL.get("clip")
        vae = MODEL.get("vae")

        missing = [
            name
            for name, value in (("model", model_base), ("clip", clip_base), ("vae", vae))
            if value is None
        ]
        if missing:
            raise ValueError(
                "CMK Detailer Prepare -Pipe-: required data missing in MODEL: "
                + ", ".join(missing)
                + ". Connect CMK Checkpoint VAE Loader -Pipe-."
            )

        try:
            sam_model = _unwrap_node_output(CMKSAMLoader().load(sam_model_name, sam_device_mode))
        except Exception as exc:
            raise RuntimeError(
                "CMK Detailer Prepare -Pipe-: native CMK SAM loader failed. "
                "Check the selected SAM model and device mode. "
                f"Original error: {exc}"
            ) from exc

        helper = CMKSamplerPrepareSDXLPipe()
        model = model_base
        clip = helper._clip_set_last_layer(clip_base, _int(stop_at_clip_layer, -2))
        clip = _unwrap_node_output(clip)

        inherit_prompt = bool(use_prompt_lora_from_sampler)
        inherit_lora = bool(use_lora_from_1st_pass)
        inherit_sampling = bool(use_1st_pass_sampling)

        if inherit_prompt:
            selected_prompt_pos = _clean_text(source_pipe.get("prompt_pos"), "")
            selected_prompt_neg = _clean_text(source_pipe.get("prompt_neg"), "")
        else:
            selected_prompt_pos = _clean_text(prompt_pos, "")
            selected_prompt_neg = _clean_text(prompt_neg, "")

        if inherit_lora:
            # Both inputs of CMK LoRA Text Loader are optional by contract.
            # Missing syntax/stack therefore remain a clean model/clip throughpass.
            opt_lora_syntax = _clean_text(source_pipe.get("lora_syntax", source_pipe.get("active_loras")), "")
            opt_lora_stack = source_pipe.get("lora_stack")
            model, clip, _, loaded_loras = CMKLoRATextLoader().load_loras(
                model=model,
                clip=clip,
                opt_lora_syntax=opt_lora_syntax,
                opt_lora_stack=opt_lora_stack,
            )
            active_loras = opt_lora_syntax
            lora_stack = opt_lora_stack
        else:
            model, clip, local_lora = helper._apply_single_lora(
                model,
                clip,
                lora_name,
                _float(strength_model, 1.0),
                _float(strength_clip, 1.0),
            )
            loaded_loras = local_lora
            active_loras = local_lora
            lora_stack = None

        selected_sampler = source_pipe.get("sampler", sampler) if inherit_sampling else sampler
        selected_scheduler = source_pipe.get("scheduler", scheduler) if inherit_sampling else scheduler
        selected_sampling = source_pipe.get("sampling", sampling) if inherit_sampling else sampling
        selected_zsnr = source_pipe.get("zsnr", zsnr) if inherit_sampling else zsnr
        selected_sampler = str(selected_sampler or sampler)
        selected_scheduler = str(selected_scheduler or scheduler)
        selected_sampling = str(selected_sampling or sampling)
        selected_zsnr = bool(selected_zsnr)

        model = _unwrap_node_output(model)
        clip = _unwrap_node_output(clip)
        model = _unwrap_node_output(helper._apply_pag(model, _float(pag_scale, 2.5)))
        model = _unwrap_node_output(
            helper._apply_sampling(model, selected_sampling, selected_zsnr)
        )
        model = _unwrap_node_output(
            helper._apply_freeu(
                model,
                bool(freeu_enabled),
                _float(freeu_b1, 1.3),
                _float(freeu_b2, 1.4),
                _float(freeu_s1, 0.9),
                _float(freeu_s2, 0.2),
            )
        )

        translation_pos = translate_prompt(selected_prompt_pos)
        translation_neg = translate_prompt(selected_prompt_neg)
        conditioning_pos = self._encode(clip, translation_pos.text)
        conditioning_neg = self._encode(clip, translation_neg.text)

        detailer_pipe = {
            # Immutable hand-off back to the main CMK pipeline. Finalize copies
            # this dictionary and commits the detailer result exactly once.
            "source_pipe": source_pipe,
            "_source_pipe": source_pipe,
            "detailer_model": model,
            "detailer_clip": clip,
            "detailer_vae": vae,
            "detailer_conditioning_pos": conditioning_pos,
            "detailer_conditioning_neg": conditioning_neg,
            "detailer_image": image,
            "detailer_seed": _int(source_pipe.get("seed"), 0),
            "detailer_global_enable": global_enable,
            # Legacy/runtime key consumed by CMK Pipe Peek Detailer and CMK Smart Detailer.
            "boolean_detailer_enable": global_enable,
            "detailer_steps": max(1, _int(steps, 30)),
            "detailer_cfg": _float(cfg, 7.0),
            "detailer_sampler": selected_sampler,
            "detailer_scheduler": selected_scheduler,
            "detailer_sam_model": sam_model,
            "detailer_sam_model_name": str(sam_model_name),
            "detailer_use_prompt_lora_from_sampler": inherit_prompt,
            "detailer_use_prompt_from_1st_pass": inherit_prompt,
            "detailer_use_lora_from_1st_pass": inherit_lora,
            "detailer_use_1st_pass_sampling": inherit_sampling,
            "detailer_prompt_pos": selected_prompt_pos,
            "detailer_prompt_neg": selected_prompt_neg,
            "detailer_active_loras": active_loras,
            "detailer_lora_stack": lora_stack,
            "detailer_loaded_loras": loaded_loras,
            "detailer_stop_at_clip_layer": _int(stop_at_clip_layer, -2),
            "detailer_pag_scale": _float(pag_scale, 2.5),
            "detailer_sampling": selected_sampling,
            "detailer_zsnr": selected_zsnr,
            "detailer_freeu_enabled": bool(freeu_enabled),
            "detailer_freeu_b1": _float(freeu_b1, 1.3),
            "detailer_freeu_b2": _float(freeu_b2, 1.4),
            "detailer_freeu_s1": _float(freeu_s1, 0.9),
            "detailer_freeu_s2": _float(freeu_s2, 0.2),
        }

        details = (
            "CMK Detailer Prepare -Pipe- | "
            f"checkpoint={MODEL.get('ckpt_name', '')} | vae={MODEL.get('vae_name', '')} | "
            f"steps={detailer_pipe['detailer_steps']} | cfg={detailer_pipe['detailer_cfg']} | "
            f"sampler={selected_sampler} | scheduler={selected_scheduler} | "
            f"sampling={selected_sampling} | zsnr={selected_zsnr} | "
            f"prompt_from_1st_pass={inherit_prompt} | "
            f"lora_from_1st_pass={inherit_lora} | "
            f"sampling_from_1st_pass={inherit_sampling} | enabled={global_enable}"
        )
        detailer_pipe["detailer_prepare_log"] = details

        prompt_source_label = "1ST PASS" if inherit_prompt else "LOCAL"
        lora_source_label = "1ST PASS" if inherit_lora else "LOCAL"
        sampling_source_label = "1ST PASS" if inherit_sampling else "LOCAL"
        log_lines = [
            "STATUS          : PREPARED",
            "MODEL SOURCE    : MODEL",
            f"CHECKPOINT      : {MODEL.get('ckpt_name', '')}",
            f"VAE             : {MODEL.get('vae_name', '')}",
            f"GLOBAL ENABLE   : {global_enable}",
            f"STEPS           : {detailer_pipe['detailer_steps']}",
            f"CFG             : {detailer_pipe['detailer_cfg']}",
            f"SAMPLER         : {selected_sampler}",
            f"SCHEDULER       : {selected_scheduler}",
            f"SAMPLING        : {selected_sampling}",
            f"ZSNR            : {selected_zsnr}",
            f"SAMPLING SOURCE : {sampling_source_label}",
            f"PROMPT SOURCE   : {prompt_source_label}",
            f"LORA SOURCE     : {lora_source_label}",
        ]
        for label, result in (("POS", translation_pos), ("NEG", translation_neg)):
            if result.status not in {"empty", "disabled", "not_configured"}:
                log_lines.append(result.log_line(label))
        if not inherit_lora:
            log_lines.extend(["", "LOCAL LORAS:", cmk_format_loras(loaded_loras)])
        if not inherit_prompt:
            if selected_prompt_pos:
                log_lines.extend(["", "POSITIVE PROMPT:", selected_prompt_pos])
            if selected_prompt_neg:
                log_lines.extend(["", "NEGATIVE PROMPT:", selected_prompt_neg])
        log_pipe = cmk_add_block(LOG, "Detailer Prepare", 60, log_lines, True)
        diagnostic = make_diagnostic_payload(
            title="Detailer Prepare",
            node="CMK Detailer Prepare -Pipe-",
            previews=[image],
            summary=(
                f"enabled={global_enable} | {detailer_pipe['detailer_steps']} steps | "
                f"{selected_sampler} / {selected_scheduler}"
            ),
            details=details,
            mode="detailer",
            metadata={
                "checkpoint": MODEL.get("ckpt_name", ""),
                "vae": MODEL.get("vae_name", ""),
                "sampler": selected_sampler,
                "scheduler": selected_scheduler,
                "sampling": selected_sampling,
                "zsnr": selected_zsnr,
                "sampling_source": sampling_source_label,
                "seed": detailer_pipe["detailer_seed"],
            },
        )
        return (detailer_pipe, log_pipe, diagnostic)
