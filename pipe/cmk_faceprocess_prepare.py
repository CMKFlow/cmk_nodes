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
from ..utils.cmk_diagnostic import make_diagnostic_payload



def _sam_loader_input_specs():
    """Reuse Impact Pack SAMLoader input definitions when available."""
    try:
        import nodes  # type: ignore

        node_cls = (getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}).get("SAMLoader")
        if node_cls is None or not hasattr(node_cls, "INPUT_TYPES"):
            raise RuntimeError("SAMLoader unavailable")

        required = (node_cls.INPUT_TYPES() or {}).get("required", {}) or {}
        model_spec = required.get("model_name")
        device_spec = required.get("device_mode")
        if model_spec is None or device_spec is None:
            raise RuntimeError("SAMLoader input definition incomplete")
        return model_spec, device_spec
    except Exception:
        # Keeps the node loadable so the runtime error can clearly identify a
        # missing/outdated Impact Pack instead of hiding the entire CMK package.
        return (("SAMLoader unavailable",),), (("AUTO", "Prefer GPU", "CPU"), {"default": "CPU"})

def _safe_default(items, preferred):
    if preferred in items:
        return preferred
    return items[0] if items else "None"


class CMKFaceProcessPreparePipe:
    """Create the isolated FACE working pipe.

    Public contract:
        MODEL + PROCESS + IMAGE + LOG -> FACE + LOG + diagnostic

    MODEL is read-only. IMAGE is the authoritative image payload. PROCESS is
    read only for shared CMK context such as prompt/LoRA offers and seed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = _safe_lora_list()
        default_lora = _safe_default(loras, "refiner/Hyper-SDXL-1step-lora.safetensors")
        sam_model_spec, sam_device_spec = _sam_loader_input_specs()

        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS": ("CMK_PIPE", {"lazy": True}),
                "IMAGE": ("IMAGE", {"lazy": True}),

                "sam_model_name": sam_model_spec,
                "sam_device_mode": sam_device_spec,

                "face_global_enable": ("BOOLEAN", {"default": False}),
                "use_prompt_lora_from_sampler": ("BOOLEAN", {"default": False}),

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
                "sampler": (SAMPLERS, {"default": "euler"} if "euler" in SAMPLERS else {}),
                "scheduler": (SCHEDULERS, {"default": "simple"} if "simple" in SCHEDULERS else {}),

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

    RETURN_TYPES = ("CMK_FACE_PIPE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("FACE", "LOG", "diagnostic")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Developer/Pipe/Prepare"

    def check_lazy_status(
        self,
        MODEL=None,
        PROCESS=None,
        IMAGE=None,
        LOG=None,
        face_global_enable=False,
        **kwargs,
    ):
        needed = []

        if IMAGE is None:
            needed.append("IMAGE")
        if LOG is None:
            needed.append("LOG")

        if bool(face_global_enable):
            if MODEL is None:
                needed.append("MODEL")
            if PROCESS is None:
                needed.append("PROCESS")

        return needed

    @staticmethod
    def _encode(clip, text):
        helper = CMKSamplerPrepareSDXLPipe()
        # Standard CLIP Text Encode on purpose: FaceProcess is not SDXL-specific.
        try:
            from .cmk_sampler_prepare import _call_node

            result = _call_node(("CLIPTextEncode",), clip, text)
            return _validate_conditioning(_unwrap_node_output(result), "face_conditioning")
        except Exception as exc:
            raise RuntimeError(f"CMK FaceProcess Prepare -Pipe-: CLIP Text Encode failed: {exc}") from exc

    def prepare(
        self,
        MODEL,
        PROCESS,
        IMAGE,
        sam_model_name,
        sam_device_mode,
        face_global_enable,
        use_prompt_lora_from_sampler,
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
            raise ValueError("CMK FaceProcess Prepare -Pipe-: IMAGE is missing")

        global_enable = bool(face_global_enable)
        image = IMAGE

        if not global_enable:
            source_pipe = dict(PROCESS) if isinstance(PROCESS, dict) else {}
            face_pipe = {
                "source_pipe": source_pipe,
                "_source_pipe": source_pipe,
                "face_image": image,
                "face_global_enable": False,
                "boolean_face_enable": False,
                "face_prepare_bypassed": True,
                "face_prepare_log": (
                    "CMK FaceProcess Prepare -Pipe- | "
                    "GLOBAL ENABLE=False | expensive preparation skipped"
                ),
            }

            log_lines = [
                "STATUS          : GLOBAL DISABLED",
                "RESULT          : LIGHTWEIGHT FACE PASSTHROUGH",
                "MODEL REQUEST   : SKIPPED",
                "PROCESS REQUEST : SKIPPED",
                "SAM LOAD        : SKIPPED",
                "LORA / CLIP     : SKIPPED",
                "CONDITIONING    : SKIPPED",
            ]
            log_pipe = cmk_add_block(
                LOG,
                "FaceProcess Prepare",
                80,
                log_lines,
                True,
            )
            diagnostic = make_diagnostic_payload(
                title="FaceProcess Prepare",
                node="CMK FaceProcess Prepare -Pipe-",
                previews=[image],
                summary="global disabled | expensive preparation skipped",
                details="\n".join(log_lines),
                mode="disabled / passthrough",
                metadata={
                    "global_enabled": False,
                    "model_requested": False,
                    "process_requested": False,
                    "sam_loaded": False,
                    "conditioning_created": False,
                },
            )
            return (face_pipe, log_pipe, diagnostic)

        if not isinstance(MODEL, dict):
            raise TypeError("CMK FaceProcess Prepare -Pipe-: MODEL must be a CMK model pipe")
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK FaceProcess Prepare -Pipe-: PROCESS must be a CMK process pipe")

        source_pipe = dict(PROCESS)
        model_base = MODEL.get("model")
        clip_base = MODEL.get("clip")
        vae = MODEL.get("vae")

        missing = [
            name
            for name, value in (("model", model_base), ("clip", clip_base), ("vae", vae))
            if value is None
        ]
        if missing:
            raise ValueError(
                "CMK FaceProcess Prepare -Pipe-: required data missing in MODEL: "
                + ", ".join(missing)
                + ". Connect CMK Checkpoint VAE Loader -Pipe-."
            )

        try:
            sam_model = _unwrap_node_output(
                _call_node_kwargs(
                    ("SAMLoader",),
                    model_name=sam_model_name,
                    device_mode=sam_device_mode,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "CMK FaceProcess Prepare -Pipe-: Impact Pack SAMLoader failed. "
                "Check the selected SAM model and device mode. "
                f"Original error: {exc}"
            ) from exc

        helper = CMKSamplerPrepareSDXLPipe()
        model = model_base
        clip = helper._clip_set_last_layer(clip_base, _int(stop_at_clip_layer, -2))
        clip = _unwrap_node_output(clip)

        inherit = bool(use_prompt_lora_from_sampler)
        if inherit:
            selected_prompt_pos = _clean_text(source_pipe.get("prompt_pos"), "")
            selected_prompt_neg = _clean_text(source_pipe.get("prompt_neg"), "")

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
            selected_prompt_pos = _clean_text(prompt_pos, "")
            selected_prompt_neg = _clean_text(prompt_neg, "")
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

        model = _unwrap_node_output(model)
        clip = _unwrap_node_output(clip)
        model = _unwrap_node_output(helper._apply_pag(model, _float(pag_scale, 2.5)))
        model = _unwrap_node_output(helper._apply_sampling(model, str(sampling), bool(zsnr)))
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

        conditioning_pos = self._encode(clip, selected_prompt_pos)
        conditioning_neg = self._encode(clip, selected_prompt_neg)

        face_pipe = {
            # Immutable hand-off back to the main CMK pipeline. Finalize copies
            # this dictionary and commits the FaceProcess result exactly once.
            "source_pipe": source_pipe,
            "_source_pipe": source_pipe,
            "face_model": model,
            "face_clip": clip,
            "face_vae": vae,
            "face_conditioning_pos": conditioning_pos,
            "face_conditioning_neg": conditioning_neg,
            "face_image": image,
            "face_seed": _int(source_pipe.get("seed"), 0),
            "face_global_enable": global_enable,
            # Runtime compatibility key consumed by FaceProcess nodes.
            "boolean_face_enable": global_enable,
            "face_steps": max(1, _int(steps, 30)),
            "face_cfg": _float(cfg, 7.0),
            "face_sampler": sampler,
            "face_scheduler": scheduler,
            "face_sam_model": sam_model,
            "face_use_prompt_lora_from_sampler": inherit,
            "face_prompt_pos": selected_prompt_pos,
            "face_prompt_neg": selected_prompt_neg,
            "face_active_loras": active_loras,
            "face_lora_stack": lora_stack,
            "face_loaded_loras": loaded_loras,
            "face_stop_at_clip_layer": _int(stop_at_clip_layer, -2),
            "face_pag_scale": _float(pag_scale, 2.5),
            "face_sampling": str(sampling),
            "face_zsnr": bool(zsnr),
            "face_freeu_enabled": bool(freeu_enabled),
        }

        details = (
            "CMK FaceProcess Prepare -Pipe- | "
            f"checkpoint={MODEL.get('ckpt_name', '')} | vae={MODEL.get('vae_name', '')} | "
            f"steps={face_pipe['face_steps']} | cfg={face_pipe['face_cfg']} | "
            f"sampler={sampler} | scheduler={scheduler} | "
            f"prompt_lora_from_sampler={inherit} | enabled={global_enable}"
        )
        face_pipe["face_prepare_log"] = details

        source_label = "SOURCE" if inherit else "LOCAL"
        log_lines = [
            "STATUS          : PREPARED",
            "MODEL SOURCE    : MODEL",
            f"CHECKPOINT      : {MODEL.get('ckpt_name', '')}",
            f"VAE             : {MODEL.get('vae_name', '')}",
            f"GLOBAL ENABLE   : {global_enable}",
            f"STEPS           : {face_pipe['face_steps']}",
            f"CFG             : {face_pipe['face_cfg']}",
            f"SAMPLER         : {sampler}",
            f"SCHEDULER       : {scheduler}",
            f"PROMPT SOURCE   : {source_label}",
            f"LORA SOURCE     : {source_label}",
        ]
        if not inherit:
            log_lines.extend(["", "LOCAL LORAS:", cmk_format_loras(loaded_loras)])
            if selected_prompt_pos:
                log_lines.extend(["", "POSITIVE PROMPT:", selected_prompt_pos])
            if selected_prompt_neg:
                log_lines.extend(["", "NEGATIVE PROMPT:", selected_prompt_neg])
        log_pipe = cmk_add_block(LOG, "FaceProcess Prepare", 80, log_lines, True)
        diagnostic = make_diagnostic_payload(
            title="FaceProcess Prepare",
            node="CMK FaceProcess Prepare -Pipe-",
            previews=[image],
            summary=(
                f"enabled={global_enable} | {face_pipe['face_steps']} steps | "
                f"{sampler} / {scheduler}"
            ),
            details=details,
            mode="faceprocess",
            metadata={
                "checkpoint": MODEL.get("ckpt_name", ""),
                "vae": MODEL.get("vae_name", ""),
                "sampler": sampler,
                "scheduler": scheduler,
                "seed": face_pipe["face_seed"],
            },
        )
        return (face_pipe, log_pipe, diagnostic)
