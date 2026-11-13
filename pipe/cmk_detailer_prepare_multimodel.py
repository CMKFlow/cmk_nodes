from __future__ import annotations

from ..cmk_common import SAMPLERS, SCHEDULERS
from ..loader.cmk_lora_text_loader import CMKLoRATextLoader
from .cmk_sampler_prepare import (
    _call_node,
    _clean_text,
    _float,
    _int,
    _unwrap_node_output,
    _validate_conditioning,
)
from .cmk_log_pipe import cmk_add_block, cmk_format_loras
from ..utils.cmk_translation import translate_prompt
from ..utils.cmk_diagnostic import make_diagnostic_payload
from ..engine.native_detailer import CMKSAMLoader
from .cmk_detailer_prepare import CMKDetailerPreparePipe
from .loaders.z_image_turbo_loader import (
    ensure_zit_text_encoder,
    evict_zit_text_encoder,
    release_zit_resources,
    resolve_zit_postprocess_model,
)


NODE_LABEL = "CMK Detailer Prepare Multi-Model -Pipe-"
DETAILER_MULTIMODEL_PIPE = "CMK_DETAILER_MULTIMODEL_PIPE"
DETAILER_MODEL_FAMILIES = ["None", "SDXL", "Z-Image Turbo"]
SDXL_SAMPLING_ENTRY = 95.0
# Programmatic keyword (never a ComfyUI input): the Advanced Detailer passes a
# per-prompt conditioning memo to the unchanged ZIT preparation.
CONDITIONING_MEMO_ARGUMENT = "_conditioning_memo"
ZIT_SAMPLING_ENTRY = 85.0


_CONDITIONING_MEMO = {"prompt": None, "entries": {}}


def prompt_conditioning_memo(prompt=None):
    """Conditioning memo of the running prompt (Advanced Detailer only).

    Holds conditioning tensors, never models; replaced when the prompt changes.
    """
    try:
        from .cmk_prompt_result_state import active_prompt_id

        identity = active_prompt_id()
    except Exception:
        identity = None
    if identity is None:
        identity = ("prompt-object", id(prompt))
    if _CONDITIONING_MEMO["prompt"] != identity:
        _CONDITIONING_MEMO["prompt"] = identity
        _CONDITIONING_MEMO["entries"] = {}
    return _CONDITIONING_MEMO["entries"]


def detailer_model_family(value):
    """Map the model_family widget to the CMK family id ("none" = bypass)."""
    name = str(value or "").strip().lower()
    if name == "sdxl":
        return "sdxl"
    return "z_image_turbo" if name == "z-image turbo" else "none"


def _sampling_entry_spec(default):
    # Same range and meaning as the SAMPLING ENTRY AT % of CMK Smart Detailer -Pipe-.
    return ("FLOAT", {
        "default": default,
        "min": 50.0,
        "max": 100.0,
        "step": 1.0,
        "round": 1.0,
        "tooltip": (
            "Sampling entry in percent. Later entry preserves the original "
            "content; earlier entry rebuilds it more strongly."
        ),
    })


class CMKDetailerFamilyEnable:
    """Model family control of one Detailer.

    ENABLE: the Boolean module switch for CMK Module Bypass Gate and CMK
    Preview & Compare (None is off, SDXL and Z-Image Turbo are on).
    MODEL FAMILY: the unchanged selection for the model_family input of the
    Multi-Model Prepare. The prompt handler cmk_prompt_family_inline puts the
    literal selection in place of that link, so gates and boundaries see the
    same prompt as with the Prepare dropdown. The node reads nothing else.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_family": (DETAILER_MODEL_FAMILIES, {"default": "None"})}}

    RETURN_TYPES = ("BOOLEAN", DETAILER_MODEL_FAMILIES)
    RETURN_NAMES = ("ENABLE", "MODEL FAMILY")
    FUNCTION = "read"
    CATEGORY = "CMK/Flow/PostProcess"

    @staticmethod
    def read(model_family="None"):
        return (detailer_model_family(model_family) != "none", model_family)


class CMKDetailerPrepareMultiModelPipe(CMKDetailerPreparePipe):
    """DETAILER preparation for SDXL or Z-Image Turbo.

    Public contract:
        MODEL ZIT (opt) + MODEL SDXL + PROCESS + IMAGE + LOG (+ LoRA ZIT)
            -> DETAILER (multi-model) + LOG + diagnostic

    MODEL FAMILY = None is the complete bypass. The SDXL path is the unchanged
    CMK Detailer Prepare -Pipe- preparation; each family carries its own
    steps, CFG, sampler, scheduler and sampling entry. The selected sampling
    entry travels in the DETAILER pipe to CMK Smart Detailer Multi-Model -Pipe-.
    """

    @classmethod
    def INPUT_TYPES(cls):
        legacy = CMKDetailerPreparePipe.INPUT_TYPES()["required"]
        sdxl = {name: legacy[name] for name in legacy if name != "detailer_global_enable"}
        # The frontend shows the optional MODEL ZIT above MODEL SDXL
        # (cmk_detailer_multimodel_ui.js); saved "MODEL" links are renamed there.
        required = {
            "MODEL SDXL": sdxl["MODEL"],
            "PROCESS": sdxl["PROCESS"],
            "IMAGE": sdxl["IMAGE"],
            "model_family": (DETAILER_MODEL_FAMILIES, {"default": "None"}),
            "sam_model_name": sdxl["sam_model_name"],
            "sam_device_mode": sdxl["sam_device_mode"],
            "use_prompt_lora_from_sampler": sdxl["use_prompt_lora_from_sampler"],
            "use_lora_from_1st_pass": sdxl["use_lora_from_1st_pass"],
            "use_1st_pass_sampling": sdxl["use_1st_pass_sampling"],
            "lora_name": sdxl["lora_name"],
            "strength_model": sdxl["strength_model"],
            "strength_clip": sdxl["strength_clip"],
            "prompt_pos": sdxl["prompt_pos"],
            "prompt_neg": sdxl["prompt_neg"],
            "steps": sdxl["steps"],
            "cfg": sdxl["cfg"],
            "sampling_entry": _sampling_entry_spec(SDXL_SAMPLING_ENTRY),
            "zit_steps": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
            "zit_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            "zit_sampling_entry": _sampling_entry_spec(ZIT_SAMPLING_ENTRY),
            "sampler": sdxl["sampler"],
            "scheduler": sdxl["scheduler"],
            "zit_sampler": (SAMPLERS, {
                **({"default": "res_multistep"} if "res_multistep" in SAMPLERS else {}),
                "advanced": True,
            }),
            "zit_scheduler": (SCHEDULERS, {
                **({"default": "simple"} if "simple" in SCHEDULERS else {}),
                "advanced": True,
            }),
            "stop_at_clip_layer": sdxl["stop_at_clip_layer"],
            "pag_scale": sdxl["pag_scale"],
            "sampling": sdxl["sampling"],
            "zsnr": sdxl["zsnr"],
            "zit_model_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "advanced": True}),
            "freeu_enabled": sdxl["freeu_enabled"],
            "freeu_b1": sdxl["freeu_b1"],
            "freeu_b2": sdxl["freeu_b2"],
            "freeu_s1": sdxl["freeu_s1"],
            "freeu_s2": sdxl["freeu_s2"],
        }
        return {
            "required": required,
            "optional": {
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "MODEL ZIT": ("CMK_MODEL_PIPE", {"lazy": True}),
                "LoRA ZIT": ("CMK_LORA_ZIT_PIPE", {"lazy": True}),
            },
        }

    RETURN_TYPES = (DETAILER_MULTIMODEL_PIPE, "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("DETAILER", "LOG", "diagnostic")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Flow/PostProcess"

    def check_lazy_status(self, PROCESS=None, IMAGE=None, LOG=None, model_family="None", **kwargs):
        # IMAGE and LOG first, exactly like the SDXL Detailer Prepare.
        upstream = super().check_lazy_status(
            PROCESS=PROCESS, IMAGE=IMAGE, LOG=LOG, detailer_global_enable=False,
        )
        if upstream:
            return upstream
        family = detailer_model_family(model_family)
        if family == "none":
            return []
        needed = ["PROCESS"] if PROCESS is None else []
        if family == "z_image_turbo":
            # Unconnected optional inputs are absent; requesting them would
            # fail, so prepare() reports a missing ZIT loader instead.
            for name in ("MODEL ZIT", "LoRA ZIT"):
                if name in kwargs and kwargs[name] is None:
                    needed.append(name)
        elif kwargs.get("MODEL SDXL") is None:
            needed.append("MODEL SDXL")
        return needed

    def prepare(
        self,
        PROCESS,
        IMAGE,
        model_family,
        sam_model_name,
        sam_device_mode,
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
        sampling_entry,
        zit_steps,
        zit_cfg,
        zit_sampling_entry,
        sampler,
        scheduler,
        zit_sampler,
        zit_scheduler,
        stop_at_clip_layer,
        pag_scale,
        sampling,
        zsnr,
        zit_model_shift,
        freeu_enabled,
        freeu_b1,
        freeu_b2,
        freeu_s1,
        freeu_s2,
        LOG=None,
        **zit_inputs,
    ):
        family = detailer_model_family(model_family)
        if family == "z_image_turbo":
            if IMAGE is None:
                raise ValueError(f"{NODE_LABEL}: IMAGE is missing")
            if not isinstance(PROCESS, dict):
                raise TypeError(f"{NODE_LABEL}: PROCESS must be a CMK process pipe")
            if PROCESS.get("result_contract") != "family_neutral":
                raise ValueError(f"{NODE_LABEL} requires the PostProcess Boundary")
            return self._prepare_zit(
                zit_inputs.get("MODEL ZIT"),
                PROCESS,
                IMAGE,
                sam_model_name=sam_model_name,
                sam_device_mode=sam_device_mode,
                use_prompt_lora_from_sampler=use_prompt_lora_from_sampler,
                use_lora_from_1st_pass=use_lora_from_1st_pass,
                lora_zit=zit_inputs.get("LoRA ZIT"),
                prompt_pos=prompt_pos,
                steps=zit_steps,
                cfg=zit_cfg,
                sampling_entry=zit_sampling_entry,
                model_shift=zit_model_shift,
                sampler=zit_sampler,
                scheduler=zit_scheduler,
                LOG=LOG,
                conditioning_memo=zit_inputs.get(CONDITIONING_MEMO_ARGUMENT),
            )

        model_sdxl = zit_inputs.get("MODEL SDXL")
        if (
            family == "sdxl"
            and IMAGE is not None
            and isinstance(model_sdxl, dict)
            and isinstance(PROCESS, dict)
            and PROCESS.get("result_contract") == "family_neutral"
        ):
            # SDXL is about to be materialised: drop CMK's own ZIT references
            # unless another consumer still uses them (no-op when empty).
            release_zit_resources("ZIT -> SDXL Detailer")

        # None is the SDXL Detailer Prepare with detailer_global_enable = False:
        # the identical lightweight bypass without MODEL, PROCESS, SAM or LoRA.
        detailer_pipe, log_pipe, diagnostic = super().prepare(
            MODEL=zit_inputs.get("MODEL SDXL"),
            PROCESS=PROCESS,
            IMAGE=IMAGE,
            sam_model_name=sam_model_name,
            sam_device_mode=sam_device_mode,
            detailer_global_enable=family == "sdxl",
            use_prompt_lora_from_sampler=use_prompt_lora_from_sampler,
            use_lora_from_1st_pass=use_lora_from_1st_pass,
            use_1st_pass_sampling=use_1st_pass_sampling,
            lora_name=lora_name,
            strength_model=strength_model,
            strength_clip=strength_clip,
            prompt_pos=prompt_pos,
            prompt_neg=prompt_neg,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            scheduler=scheduler,
            stop_at_clip_layer=stop_at_clip_layer,
            pag_scale=pag_scale,
            sampling=sampling,
            zsnr=zsnr,
            freeu_enabled=freeu_enabled,
            freeu_b1=freeu_b1,
            freeu_b2=freeu_b2,
            freeu_s1=freeu_s1,
            freeu_s2=freeu_s2,
            LOG=LOG,
        )
        detailer_pipe["detailer_model_family"] = family
        if family == "sdxl":
            detailer_pipe["detailer_sampling_entry"] = _float(sampling_entry, SDXL_SAMPLING_ENTRY)
        return (detailer_pipe, log_pipe, diagnostic)

    def _prepare_zit(
        self,
        MODEL,
        PROCESS,
        image,
        *,
        sam_model_name,
        sam_device_mode,
        use_prompt_lora_from_sampler,
        use_lora_from_1st_pass,
        lora_zit,
        prompt_pos,
        steps,
        cfg,
        sampling_entry,
        model_shift,
        sampler,
        scheduler,
        LOG,
        conditioning_memo=None,
    ):
        """Z-Image Turbo preparation on the CMK Sampler Prepare Z-Image Turbo contract.

        SDXL-only operations (CLIP skip, PAG, v_pred/zsnr, FreeU, local SDXL
        LoRA) are not applied. LoRAs come from the ZIT LoRA bundle or from a
        ZIT first pass, exactly as Create Image hands them to the ZIT sampler.
        """
        if not isinstance(MODEL, dict):
            raise ValueError(
                f"{NODE_LABEL}: MODEL FAMILY = Z-Image Turbo requires "
                "CMK Z-Image Turbo Loader -Pipe- connected to MODEL ZIT."
            )
        if str(MODEL.get("model_family", "")).strip().lower() != "z_image_turbo":
            raise ValueError(
                f"{NODE_LABEL}: MODEL ZIT must come from CMK Z-Image Turbo Loader -Pipe- "
                "or CMK Z-Image Turbo PostProcess Model."
            )
        # A PostProcess spec is materialised here, never upstream.
        MODEL = resolve_zit_postprocess_model(MODEL)
        model = MODEL.get("model")
        vae = MODEL.get("vae")
        clip = MODEL.get("clip")

        # Prompt and LoRA selection (no side effects).
        source_pipe = dict(PROCESS)
        source_family = str(source_pipe.get("source_model_family", "sdxl")).strip().lower()
        inherit_prompt = bool(use_prompt_lora_from_sampler) and any(
            key in source_pipe for key in ("prompt_pos", "prompt_neg")
        )
        inherit_lora = bool(use_lora_from_1st_pass) and source_family == "z_image_turbo"
        selected_prompt_pos = _clean_text(
            source_pipe.get("prompt_pos") if inherit_prompt else prompt_pos, ""
        )

        if inherit_lora:
            lora_syntax = _clean_text(source_pipe.get("lora_syntax", source_pipe.get("active_loras")), "")
            lora_stack = source_pipe.get("lora_stack")
            trigger_words = ""
        else:
            bundle = lora_zit if isinstance(lora_zit, dict) else {}
            if bundle and bundle.get("family") != "z_image_turbo":
                raise ValueError(f"{NODE_LABEL}: LoRA ZIT does not match its model family.")
            lora_syntax = str(bundle.get("active_loras", "") or "").strip()
            lora_stack = bundle.get("lora_stack")
            trigger_words = str(bundle.get("trigger_words", "") or "").strip()
            if trigger_words and trigger_words in selected_prompt_pos:
                trigger_words = ""
        positive_text = "\n".join(part for part in (selected_prompt_pos, trigger_words) if part)

        # Advanced Detailer only: reuse conditioning produced in this prompt by
        # an identical configuration (same encoder, device and text, no LoRA).
        # The Standard Prepare passes no memo and always encodes.
        memo_key = None
        reused = None
        if conditioning_memo is not None and not lora_syntax and not lora_stack:
            memo_key = (
                "z_image_turbo",
                str(MODEL.get("text_encoder_name", "")),
                str(MODEL.get("clip_device", "")),
                positive_text,
            )
            reused = conditioning_memo.get(memo_key)

        if clip is None and reused is None:
            clip, _ = ensure_zit_text_encoder(MODEL)
        missing = [
            name
            for name, value in (
                ("model", model),
                ("text encoder", clip if reused is None else True),
                ("vae", vae),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"{NODE_LABEL}: required data missing in MODEL ZIT: " + ", ".join(missing)
            )

        try:
            sam_model = _unwrap_node_output(CMKSAMLoader().load(sam_model_name, sam_device_mode))
        except Exception as exc:
            raise RuntimeError(
                f"{NODE_LABEL}: native CMK SAM loader failed. "
                "Check the selected SAM model and device mode. "
                f"Original error: {exc}"
            ) from exc

        if reused is None:
            model, clip, _, loaded_loras = CMKLoRATextLoader().load_loras(
                model,
                clip,
                opt_lora_syntax=lora_syntax,
                opt_lora_stack=lora_stack,
            )

            translation_pos = translate_prompt(positive_text)
            conditioning_pos = self._encode(clip, translation_pos.text)
            conditioning_neg = _validate_conditioning(
                _unwrap_node_output(_call_node(("ConditioningZeroOut",), conditioning_pos)),
                "detailer_conditioning_neg",
            )
            # Same lifecycle as the ZIT sampler: the LoRA path returns a CLIP clone,
            # so drop it together with the loader's text encoder after encoding.
            clip = None
            text_encoder_status = evict_zit_text_encoder(MODEL)
            if memo_key is not None:
                conditioning_memo[memo_key] = (conditioning_pos, conditioning_neg, translation_pos)
        else:
            conditioning_pos, conditioning_neg, translation_pos = reused
            loaded_loras = ""
            text_encoder_status = "CONDITIONING REUSED"
        model = _unwrap_node_output(
            _call_node(("ModelSamplingAuraFlow",), _unwrap_node_output(model), float(model_shift))
        )

        selected_steps = max(1, _int(steps, 8))
        selected_cfg = _float(cfg, 1.0)
        selected_shift = _float(model_shift, 3.0)
        selected_entry = _float(sampling_entry, ZIT_SAMPLING_ENTRY)
        model_identity = {
            name: str(MODEL.get(name, ""))
            for name in (
                "diffusion_model_name", "text_encoder_name", "vae_name",
                "weight_dtype", "clip_device",
            )
        }
        detailer_pipe = {
            "source_pipe": source_pipe,
            "_source_pipe": source_pipe,
            "detailer_model_family": "z_image_turbo",
            "detailer_model_identity": model_identity,
            "detailer_model": model,
            "detailer_clip": None,
            "detailer_vae": vae,
            "detailer_conditioning_pos": conditioning_pos,
            "detailer_conditioning_neg": conditioning_neg,
            "detailer_image": image,
            "detailer_seed": _int(source_pipe.get("seed"), 0),
            "detailer_global_enable": True,
            "boolean_detailer_enable": True,
            "detailer_steps": selected_steps,
            "detailer_cfg": selected_cfg,
            "detailer_sampler": str(sampler),
            "detailer_scheduler": str(scheduler),
            "detailer_sampling_entry": selected_entry,
            "detailer_sam_model": sam_model,
            "detailer_sam_model_name": str(sam_model_name),
            "detailer_use_prompt_lora_from_sampler": inherit_prompt,
            "detailer_use_prompt_from_1st_pass": inherit_prompt,
            "detailer_use_lora_from_1st_pass": inherit_lora,
            "detailer_use_1st_pass_sampling": False,
            "detailer_prompt_pos": selected_prompt_pos,
            "detailer_prompt_neg": "",
            "detailer_active_loras": lora_syntax,
            "detailer_lora_stack": lora_stack,
            "detailer_loaded_loras": loaded_loras,
            "detailer_zit_trigger_words": trigger_words,
            "detailer_zit_model_shift": selected_shift,
            "detailer_text_encoder_status": text_encoder_status,
            # SDXL-only settings do not apply to Z-Image Turbo.
            "detailer_stop_at_clip_layer": None,
            "detailer_pag_scale": None,
            "detailer_sampling": None,
            "detailer_zsnr": None,
            "detailer_freeu_enabled": None,
            "detailer_freeu_b1": None,
            "detailer_freeu_b2": None,
            "detailer_freeu_s1": None,
            "detailer_freeu_s2": None,
        }

        prompt_source_label = "1ST PASS" if inherit_prompt else "LOCAL"
        lora_source_label = "1ST PASS · ZIT" if inherit_lora else ("LoRA ZIT" if lora_syntax or lora_stack else "NONE")
        details = (
            f"{NODE_LABEL} | family=z_image_turbo | "
            f"diffusion_model={model_identity['diffusion_model_name']} | "
            f"steps={selected_steps} | cfg={selected_cfg} | shift={selected_shift} | "
            f"sampler={sampler} | scheduler={scheduler} | sampling_entry={selected_entry:g}% | "
            f"prompt_from_1st_pass={inherit_prompt} | lora_from_1st_pass={inherit_lora} | enabled=True"
        )
        detailer_pipe["detailer_prepare_log"] = details
        log_lines = [
            "STATUS          : PREPARED",
            "MODEL FAMILY    : Z-IMAGE TURBO",
            f"DIFFUSION MODEL : {model_identity['diffusion_model_name']}",
            f"TEXT ENCODER    : {model_identity['text_encoder_name']} · {text_encoder_status}",
            f"VAE             : {model_identity['vae_name']}",
            f"STEPS           : {selected_steps}",
            f"CFG             : {selected_cfg}",
            f"MODEL SHIFT     : {selected_shift:g}",
            f"SAMPLER         : {sampler}",
            f"SCHEDULER       : {scheduler}",
            f"SAMPLING ENTRY  : {selected_entry:g} %",
            "NEGATIVE        : CONDITIONING ZERO OUT",
            f"PROMPT SOURCE   : {prompt_source_label}",
            f"LORA SOURCE     : {lora_source_label}",
        ]
        if translation_pos.status not in {"empty", "disabled", "not_configured"}:
            log_lines.append(translation_pos.log_line("POS"))
        log_lines.extend(["", "ZIT LORAS:", cmk_format_loras(loaded_loras)])
        if not inherit_prompt and selected_prompt_pos:
            log_lines.extend(["", "POSITIVE PROMPT:", selected_prompt_pos])
        if trigger_words:
            log_lines.extend(["", "TRIGGER WORDS:", trigger_words])
        log_pipe = cmk_add_block(LOG, "Detailer Prepare", 60, log_lines, True)
        diagnostic = make_diagnostic_payload(
            title="Detailer Prepare",
            node=NODE_LABEL,
            previews=[image],
            summary=f"Z-Image Turbo | {selected_steps} steps | {sampler} / {scheduler}",
            details=details,
            mode="detailer",
            metadata={
                "model_family": "z_image_turbo",
                "diffusion_model": model_identity["diffusion_model_name"],
                "vae": model_identity["vae_name"],
                "sampler": str(sampler),
                "scheduler": str(scheduler),
                "sampling_entry": selected_entry,
                "model_shift": selected_shift,
                "seed": detailer_pipe["detailer_seed"],
            },
        )
        return (detailer_pipe, log_pipe, diagnostic)
