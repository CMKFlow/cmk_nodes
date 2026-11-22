"""Mask Detailer 2.8: discrete two-model setup (SDXL / Z-Image Turbo).

Built like the Multi-Model Detailer: CMK Detailer Family Enable selects the
family, the Prepare requests and materialises only that family's model and the
Process samples with that family's settings (sampling entry per family, seed
from PROCESS).  Mask Detailer specific: the painted mask instead of a
detector, the source prompt plus an own prompt addition, one local LoRA per
family, and its own sampling-entry range: inpainting a painted area needs
more than the Detailer limit (denoise <= 0.5); the range is the one of the
existing Mask Detailer (denoise 0.01..1.0).

The public outputs are those of CMK Mask Detailer Prepare / Process; the
legacy nodes are unchanged.
"""

from __future__ import annotations

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:  # Unit tests outside a running ComfyUI installation.
    class ExecutionBlocker:
        def __init__(self, message):
            self.message = message

from ...loader.cmk_lora_text_loader import CMKLoRATextLoader
from ...pipe.cmk_detailer_prepare import CMKDetailerPreparePipe
from ...pipe.cmk_detailer_prepare_multimodel import (
    DETAILER_MODEL_FAMILIES,
    CMKDetailerPrepareMultiModelPipe,
    detailer_model_family,
)
from ...pipe.cmk_log_pipe import cmk_add_block
from ...pipe.cmk_sampler_prepare import _call_node, _unwrap_node_output
from ...pipe.cmk_visual import empty_visual
from ...pipe.loaders.checkpoint_vae_loader import resolve_postprocess_model
from ...pipe.loaders.z_image_turbo_loader import (
    ensure_zit_text_encoder,
    evict_zit_text_encoder,
    release_zit_resources,
    resolve_zit_postprocess_model,
)
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...utils.cmk_translation import translate_prompt
from ...utils.prepare_card import footer_card
from ...utils.segs_diagnostic import detailer_spatial_stages
from ..instantid_face_rebuild_v2 import CMKFaceRebuildModeEnable
from ..utils.diagnostic_concat import CMKDiagnosticConcat
from .smart_detailer import CMK_SmartDetailer
from .mask_detailer import (
    MASK_DETAILER_PROCESS,
    MASK_DETAILER_SOURCE,
    CMKMaskDetailerFlowGate,
    CMKMaskDetailerPrepare,
    CMKMaskDetailerProcess,
    _node,
    _positive_text,
    _prepare_diagnostic,
    _prepare_source,
    _prepared_mask,
    _process_pipe,
)

PREPARE_LABEL = "CMK Mask Detailer · Prepare Multi-Model"
PROCESS_LABEL = "CMK Mask Detailer · Process Multi-Model"
FAMILY_LABELS = {"sdxl": "SDXL", "z_image_turbo": "Z-IMAGE TURBO", "none": "NONE"}
# The text encoder of Z-Image Turbo is released after encoding; the sampling
# engine never reads CLIP, the basic pipe only needs a value.
ZIT_TEXT_ENCODER = "released after Mask Detailer Prepare"
# Denoise 0.55, the default of CMK Mask Detailer Process.
MASK_SAMPLING_ENTRY = 45.0


def _translation_lines(translations):
    """Log lines of the central prompt translation, as CMK Detailer Prepare writes them."""
    return [result.log_line(label) for label, result in translations
            if result.status not in {"empty", "disabled", "not_configured"}]


def _intake_pending(terminal):
    """A Mask Detailer Intake source before the mask is painted (S0-S2)."""
    return isinstance(terminal, dict) and not bool(terminal.get("ready", True))


def _seed(source, process):
    """Seed of the source; behind an Intake the snapshot carries none, the
    flow's PROCESS does."""
    seed = source.get("seed")
    if seed is None and isinstance(process, dict):
        seed = process.get("seed")
    return int(seed or 0)


def _mask_sampling_entry_spec():
    return ("FLOAT", {
        "default": MASK_SAMPLING_ENTRY,
        "min": 0.0,
        "max": 99.0,
        "step": 1.0,
        "round": 1.0,
        "tooltip": (
            "Sampling entry in percent (denoise = 1 - entry). Later entry preserves "
            "the painted area; earlier entry repaints it more strongly."
        ),
    })


def mask_entry_to_denoise(entry):
    """Sampling entry in percent -> denoise, in the Mask Detailer range 0.01..1.0."""
    return min(1.0, max(0.01, (100.0 - float(entry)) / 100.0))


def _lora_syntax(name, strength):
    name = str(name or "None")
    return "" if name.lower() == "none" else f"<lora:{name}:{float(strength):g}>"


class CMKMaskDetailerPrepareMultiModel:
    """Mask Detailer preparation for SDXL or Z-Image Turbo.

    Public contract (as CMK Mask Detailer · Prepare):
        MODEL ZIT (opt) + MODEL SDXL + PROCESS / MASK DETAILER SOURCE + IMAGE + LOG
            -> MODEL + MASK DETAILER PROCESS + LOG + VISUAL + diagnostic

    MODEL FAMILY = None is the bypass: no model, no conditioning; the Process
    passes the image on unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        legacy = CMKMaskDetailerPrepare.INPUT_TYPES()["required"]
        detailer = CMKDetailerPrepareMultiModelPipe.INPUT_TYPES()["required"]
        return {
            "required": {
                "MODEL SDXL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "model_family": (DETAILER_MODEL_FAMILIES, {"default": "None"}),
                "fill_mask_holes": legacy["fill_mask_holes"],
                "solid_mask": legacy["solid_mask"],
                "lora_name": legacy["lora_name"],
                "lora_strength": legacy["lora_strength"],
                "zit_lora_name": legacy["lora_name"],
                "zit_lora_strength": legacy["lora_strength"],
                "prompt_pos": legacy["prompt_pos"],
                "steps": detailer["steps"],
                "cfg": detailer["cfg"],
                "sampling_entry": _mask_sampling_entry_spec(),
                "zit_steps": detailer["zit_steps"],
                "zit_cfg": detailer["zit_cfg"],
                "zit_sampling_entry": _mask_sampling_entry_spec(),
                "sampler": detailer["sampler"],
                "scheduler": detailer["scheduler"],
                "zit_sampler": detailer["zit_sampler"],
                "zit_scheduler": detailer["zit_scheduler"],
                "zit_model_shift": detailer["zit_model_shift"],
            },
            "optional": {
                "MODEL ZIT": ("CMK_MODEL_PIPE", {"lazy": True}),
                "MASK DETAILER SOURCE": (MASK_DETAILER_SOURCE,),
                "PROCESS": ("CMK_RESULT_PROCESS",),
                "IMAGE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
                "VISUAL": ("CMK_VISUAL_PIPE",),
            },
        }

    RETURN_TYPES = CMKMaskDetailerPrepare.RETURN_TYPES
    RETURN_NAMES = CMKMaskDetailerPrepare.RETURN_NAMES
    FUNCTION = "prepare"
    CATEGORY = "CMK/Flow/Mask Detailer"

    def check_lazy_status(self, model_family="None", **kwargs):
        if _intake_pending(kwargs.get("MASK DETAILER SOURCE")):
            return []
        family = detailer_model_family(model_family)
        if family == "sdxl" and kwargs.get("MODEL SDXL") is None:
            return ["MODEL SDXL"]
        # An unconnected optional input is absent from kwargs and never requested.
        if family == "z_image_turbo" and "MODEL ZIT" in kwargs and kwargs["MODEL ZIT"] is None:
            return ["MODEL ZIT"]
        return []

    def prepare(self, model_family, fill_mask_holes, solid_mask, lora_name, lora_strength, zit_lora_name,
                zit_lora_strength, prompt_pos, steps, cfg, sampling_entry, zit_steps, zit_cfg, zit_sampling_entry,
                sampler, scheduler, zit_sampler, zit_scheduler, zit_model_shift, LOG=None, VISUAL=None, **kwargs):
        family = detailer_model_family(model_family)
        terminal = kwargs.get("MASK DETAILER SOURCE")
        # Intake S0-S2 (no painted mask yet): no model, no sampling.
        pending = _intake_pending(terminal)
        if pending:
            family = "none"
        source, image, LOG = _prepare_source(kwargs, LOG, PREPARE_LABEL)
        # The Intake hands over an empty mask until it is painted.
        mask = image.new_zeros(image.shape[0:3]) if pending else _prepared_mask(source, image, fill_mask_holes, solid_mask)
        own_prompt = str(prompt_pos or "").strip()
        sampling = {"steps": steps, "cfg": cfg, "entry": sampling_entry, "sampler": sampler, "scheduler": scheduler}
        if family == "z_image_turbo":
            sampling = {"steps": zit_steps, "cfg": zit_cfg, "entry": zit_sampling_entry, "sampler": zit_sampler,
                        "scheduler": zit_scheduler}
        sampling = {"steps": max(1, int(sampling["steps"])), "cfg": float(sampling["cfg"]),
                    "entry": float(sampling["entry"]), "sampler": str(sampling["sampler"]),
                    "scheduler": str(sampling["scheduler"]), "seed": _seed(source, kwargs.get("PROCESS"))}
        model_pipe = None
        lines = [f"MODEL FAMILY     : {FAMILY_LABELS[family]}"]
        if family == "none":
            process = _process_pipe(source, image, mask, None, None, None, None, None, "", "", "", fill_mask_holes,
                                    model_family="none")
            lines.append("STATUS           : INTAKE PENDING · NO MASK" if pending else "STATUS           : NONE · BYPASS")
        elif family == "sdxl":
            model_pipe = kwargs.get("MODEL SDXL")
            process, model_lines = self._prepare_sdxl(model_pipe, source, image, mask, own_prompt, lora_name,
                                                      lora_strength, fill_mask_holes)
            lines += model_lines
        else:
            model_pipe = kwargs.get("MODEL ZIT")
            process, model_lines = self._prepare_zit(model_pipe, source, image, mask, own_prompt, zit_lora_name,
                                                     zit_lora_strength, zit_model_shift, fill_mask_holes)
            lines += model_lines
        process["mask_detailer_sampling"] = sampling
        if family != "none":
            lines += [
                f"STEPS / CFG      : {sampling['steps']} / {sampling['cfg']:g}",
                f"SAMPLER          : {sampling['sampler']} / {sampling['scheduler']}",
                f"SAMPLING ENTRY   : {sampling['entry']:g} % · DENOISE {mask_entry_to_denoise(sampling['entry']):g}",
                f"SEED             : {sampling['seed']} (PROCESS)",
            ]
        lines += [
            f"MASK             : {int(mask.shape[-1])} × {int(mask.shape[-2])}",
            f"FILL MASK HOLES  : {'ON' if bool(fill_mask_holes) else 'OFF'}",
            f"PROMPT ADDITION  : {own_prompt or 'None'}",
        ]
        # 2.8 rule: a Prepare reports as a text card (like CMK Detailer Prepare
        # Multi-Model); the Process shows the mask where it is used.
        intake = []
        if isinstance(terminal, dict):
            intake = [f"Intake          : {'PENDING' if pending else 'READY'}",
                      f"Snapshot        : {terminal.get('snapshot', '')}"]
            lines.insert(1, f"SOURCE           : INTAKE · {'PENDING' if pending else 'READY'} · {terminal.get('snapshot', '')}")
        log = cmk_add_block(LOG, "Mask Detailer Prepare", 50, lines, True)
        diagnostic = _prepare_diagnostic(image, mask, fill_mask_holes, solid_mask, node="CMKMaskDetailerPrepareMultiModel",
                                         metadata={"model_family": family, "sampling": dict(sampling)}, with_stages=False)
        summary = "\n".join([diagnostic["summary"], f"Model family    : {FAMILY_LABELS[family]}", *intake])
        diagnostic.update(summary=summary, details=summary)
        rows = [(label.strip(), value.strip()) for label, sep, value in (line.partition(":") for line in summary.splitlines()) if sep]
        diagnostic = footer_card(diagnostic, rows)
        return (model_pipe, process, log, empty_visual() if VISUAL is None else VISUAL, diagnostic)

    @staticmethod
    def _prepare_sdxl(spec, source, image, mask, own_prompt, lora_name, lora_strength, fill_mask_holes):
        if not isinstance(spec, dict):
            raise TypeError(f"{PREPARE_LABEL}: MODEL FAMILY = SDXL requires MODEL SDXL from CMK PostProcess Model.")
        # SDXL is about to load: drop CMK's own ZIT references unless still in use.
        release_zit_resources("ZIT -> SDXL Mask Detailer")
        model_pipe = resolve_postprocess_model(spec)
        model, clip, vae = model_pipe.get("model"), model_pipe.get("clip"), model_pipe.get("vae")
        if any(item is None for item in (model, clip, vae)):
            raise ValueError(f"{PREPARE_LABEL}: MODEL SDXL is missing model, clip or vae.")
        loaded_loras = ""
        syntax = _lora_syntax(lora_name, lora_strength)
        if syntax:
            model, clip, _trigger, loaded_loras = CMKLoRATextLoader().load_loras(model, clip, opt_lora_syntax=syntax)
        positive_text = _positive_text(source, own_prompt)
        negative_text = str(source.get("prompt_neg") or "").strip()
        translations = (("POS", translate_prompt(positive_text)), ("NEG", translate_prompt(negative_text)))
        positive = _node("CLIPTextEncode", "encode", clip=clip, text=translations[0][1].text)[0]
        negative = _node("CLIPTextEncode", "encode", clip=clip, text=translations[1][1].text)[0]
        process = _process_pipe(source, image, mask, model, clip, vae, positive, negative, positive_text,
                                negative_text, loaded_loras, fill_mask_holes)
        return process, [
            f"CHECKPOINT       : {model_pipe.get('ckpt_name', 'Unknown')}",
            f"VAE              : {model_pipe.get('vae_name', 'Unknown')} ({model_pipe.get('vae_source', 'unknown')})",
            f"LORA             : {loaded_loras or 'None'}",
            *_translation_lines(translations),
        ]

    @staticmethod
    def _prepare_zit(spec, source, image, mask, own_prompt, lora_name, lora_strength, model_shift, fill_mask_holes):
        if not isinstance(spec, dict) or str(spec.get("model_family", "")).strip().lower() != "z_image_turbo":
            raise ValueError(f"{PREPARE_LABEL}: MODEL FAMILY = Z-Image Turbo requires MODEL ZIT from CMK PostProcess Model.")
        model_pipe = resolve_zit_postprocess_model(spec)
        model, vae, clip = model_pipe.get("model"), model_pipe.get("vae"), model_pipe.get("clip")
        if clip is None:
            clip, _ = ensure_zit_text_encoder(model_pipe)
        if any(item is None for item in (model, clip, vae)):
            raise ValueError(f"{PREPARE_LABEL}: MODEL ZIT is missing model, text encoder or vae.")
        loaded_loras = ""
        syntax = _lora_syntax(lora_name, lora_strength)
        if syntax:
            model, clip, _trigger, loaded_loras = CMKLoRATextLoader().load_loras(model, clip, opt_lora_syntax=syntax)
        # Z-Image Turbo: positive prompt only, the negative is the zeroed positive
        # (as the Multi-Model Detailer); the text encoder is released after encoding.
        positive_text = _positive_text(source, own_prompt)
        translations = (("POS", translate_prompt(positive_text)),)
        positive = CMKDetailerPreparePipe._encode(clip, translations[0][1].text)
        negative = _unwrap_node_output(_call_node(("ConditioningZeroOut",), positive))
        clip = None
        text_encoder_status = evict_zit_text_encoder(model_pipe)
        model = _unwrap_node_output(_call_node(("ModelSamplingAuraFlow",), _unwrap_node_output(model), float(model_shift)))
        process = _process_pipe(source, image, mask, model, ZIT_TEXT_ENCODER, vae, positive, negative, positive_text,
                                "", loaded_loras, fill_mask_holes, model_family="z_image_turbo")
        return process, [
            f"DIFFUSION MODEL  : {model_pipe.get('diffusion_model_name', 'Unknown')}",
            f"TEXT ENCODER     : {model_pipe.get('text_encoder_name', 'Unknown')} · {text_encoder_status}",
            f"VAE              : {model_pipe.get('vae_name', 'Unknown')}",
            f"MODEL SHIFT      : {float(model_shift):g}",
            "NEGATIVE         : CONDITIONING ZERO OUT",
            f"LORA             : {loaded_loras or 'None'}",
            *_translation_lines(translations),
        ]


class CMKMaskDetailerProcessMultiModel:
    """Sample the painted mask with the family's settings of the Prepare.

    The sampling itself is CMK Mask Detailer · Process (SEGS from the mask,
    SEGSDetailer, SEGSPaste with MASK FEATHER); MODEL FAMILY = None passes the
    image on unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        legacy = CMKMaskDetailerProcess.INPUT_TYPES()
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE",),
                "MASK DETAILER PROCESS": (MASK_DETAILER_PROCESS,),
                "mask_expand": legacy["required"]["mask_expand"],
                "mask_feather": legacy["required"]["mask_feather"],
            },
            # opt_diagnostic: the Prepare diagnostic, concatenated into the module
            # timeline like CMK Smart Detailer's opt_diagnostic.
            "optional": {**legacy["optional"], "opt_diagnostic": ("CMK_DIAGNOSTIC",)},
            "hidden": legacy["hidden"],
        }

    RETURN_TYPES = CMKMaskDetailerProcess.RETURN_TYPES
    RETURN_NAMES = CMKMaskDetailerProcess.RETURN_NAMES
    FUNCTION = "process"
    CATEGORY = "CMK/Flow/Mask Detailer"

    @staticmethod
    def process(MODEL, mask_expand, mask_feather, LOG=None, VISUAL=None, unique_id=None, opt_diagnostic=None, **kwargs):
        pipe = kwargs.get("MASK DETAILER PROCESS")
        if not isinstance(pipe, dict) or pipe.get("type") != MASK_DETAILER_PROCESS:
            raise TypeError(f"{PROCESS_LABEL}: invalid process family.")
        family = str(pipe.get("model_family", "none"))
        sampling = pipe.get("mask_detailer_sampling") or {}
        if family == "none":
            image = pipe["image"]
            mask = pipe.get("mask")
            # An empty mask behind an Intake: the snapshot is not painted yet.
            status = "NONE · BYPASS" if mask is None or float(mask.max()) > 0.0 else "NO PAINTED MASK · BYPASS"
            log = cmk_add_block(LOG, "Mask Detailer Process", 51, [
                f"STATUS           : {status}",
                "SAMPLING         : SKIPPED",
                "RESULT           : PASSTHROUGH",
            ], True)
            diagnostic = make_diagnostic_payload(
                title="CMK Mask Detailer · Process", node="CMKMaskDetailerProcessMultiModel", previews=[image],
                summary=f"{status} · no sampling, image unchanged", details=f"{status} · no sampling",
                mode="Process", metadata={"model_family": "none"},
            )
            return (MODEL, dict(pipe), image, log, empty_visual() if VISUAL is None else VISUAL,
                    _timeline(opt_diagnostic, diagnostic))
        denoise = mask_entry_to_denoise(sampling["entry"])
        result, run = CMKMaskDetailerProcess._run(
            MODEL, sampling["seed"], sampling["steps"], sampling["cfg"], sampling["sampler"], sampling["scheduler"],
            denoise, mask_expand, mask_feather, LOG=LOG, VISUAL=VISUAL, unique_id=unique_id,
            **{"MASK DETAILER PROCESS": pipe},
        )
        log = cmk_add_block(result[3], "Mask Detailer Multi-Model", 52, [
            f"MODEL FAMILY     : {FAMILY_LABELS.get(family, family.upper())}",
            f"SAMPLING ENTRY   : {float(sampling['entry']):g} %",
            f"DENOISE          : {denoise:g}",
            f"SEED             : {sampling['seed']} (PROCESS)",
        ], True)
        diagnostic = process_diagnostic(result[5], run, family=family, entry=float(sampling["entry"]),
                                        mask_expand=mask_expand, mask_feather=mask_feather)
        return (*result[:3], log, result[4], _timeline(opt_diagnostic, diagnostic))


def _timeline(prepare_diagnostic, process_diagnostic):
    """Module timeline: Prepare text card + Process stages (CMK Smart Detailer)."""
    return CMKDiagnosticConcat().concat("CMK Flow · MaskDetailer", prepare_diagnostic, diagnostic_2=process_diagnostic)[0]


def process_diagnostic(legacy, run, *, family, entry, mask_expand, mask_feather):
    """Sampling ROI, Mask, Pasteback and Difference of the executed run.

    Geometry from the SEGS SEGSPaste pasted back (``crop_region`` /
    ``cropped_mask`` of the effective mask after MASK EXPAND); the pasteback
    weight uses MASK FEATHER and alpha 255 like the paste.  No Source and no
    Detection stage: the input is visible at the loader and nothing is
    detected.  Difference only when the run changed the image.
    """
    before, after = run["before"], run["after"]
    segs = run["detailed"] if isinstance(run.get("detailed"), tuple) else run["segs"]
    stages = detailer_spatial_stages(
        before, segs, feather=int(mask_feather), alpha=255,
        bbox_label="PAINTED MASK BOUNDS", mask_label=f"SAMPLING MASK (expand {int(mask_expand)} px)", result=after,
    )
    detailer = CMK_SmartDetailer()
    changed = detailer._mean_abs_change(before, after)
    difference = detailer._make_difference_preview(before, after) if changed > 0.00001 else None
    if difference is not None:
        stages.append({"title": "Difference", "image": difference, "subtitle": f"source→result: {changed:.4f}"})
    stages = [dict(stage, title=f"{index:02d} {stage['title']}") for index, stage in enumerate(stages, start=1)]
    height, width = int(before.shape[1]), int(before.shape[2])
    regions = [list(map(int, seg.crop_region)) for seg in segs[1]] if isinstance(segs, tuple) else []
    bounds = [list(map(int, seg.bbox)) for seg in segs[1]] if isinstance(segs, tuple) else []
    summary = "\n".join([
        legacy.get("summary", ""),
        f"Model family     : {FAMILY_LABELS.get(family, family.upper())}",
        f"Sampling entry   : {entry:g} %",
        f"Image            : {width} x {height}",
        f"Sampling ROI     : {regions}",
        f"Mask bounds      : {bounds}",
        f"Change           : {changed:.4f}",
    ])
    return make_diagnostic_payload(
        title="CMK Mask Detailer · Process",
        node="CMKMaskDetailerProcessMultiModel",
        stages=stages,
        summary=summary,
        details=summary,
        mode="Process",
        metadata={**legacy.get("metadata", {}), "model_family": family, "sampling_entry": entry,
                  "image_size": [width, height], "sampling_roi": regions, "mask_bounds": bounds,
                  "change": changed},
        metrics=legacy.get("metrics"),
    )


class CMKMaskDetailerFlowGateV2(CMKMaskDetailerFlowGate):
    """CMK Mask Detailer Flow Gate with the module diagnostic behind the gate.

    IMAGE, PROCESS, MODEL, LOG and VISUAL are selected exactly as by the Flow
    Gate.  DIAGNOSTIC (the Process diagnostic) is requested whenever the
    module is enabled, independent of READY, so the Intake states S0-S2 keep
    their pending diagnostic and S3 its spatial one.  Disabled, it is not
    requested and the diagnostic output is blocked: neither the Preview Render
    nor a consumer of the public diagnostic pulls Process, Prepare or Intake.
    """

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["optional"]["DIAGNOSTIC"] = ("CMK_DIAGNOSTIC", {"lazy": True})
        return types

    RETURN_TYPES = (*CMKMaskDetailerFlowGate.RETURN_TYPES, "CMK_DIAGNOSTIC")
    RETURN_NAMES = (*CMKMaskDetailerFlowGate.RETURN_NAMES, "diagnostic")

    def check_lazy_status(self, enabled=True, READY=None, prompt=None, unique_id=None, **inputs):
        needed = super().check_lazy_status(enabled=enabled, READY=READY, prompt=prompt, unique_id=unique_id, **inputs)
        if bool(enabled) and "DIAGNOSTIC" not in needed:
            needed = needed + self._pending_connected_visual("DIAGNOSTIC", prompt, unique_id, inputs)
        return needed

    def gate(self, enabled=True, READY=None, unique_id=None, DIAGNOSTIC=None, **inputs):
        result = super().gate(enabled=enabled, READY=READY, unique_id=unique_id, **inputs)
        return (*result, DIAGNOSTIC if bool(enabled) else ExecutionBlocker(None))


MASKDETAILER_MODE_WIDGET = "maskdetailer_mode"


class CMKMaskDetailerModeEnable(CMKFaceRebuildModeEnable):
    """MODE of the 2.8 Mask Detailer subgraph (the FaceRebuild v2 MODE
    control): ENABLE activates the module, DISABLE is its bypass.  The model
    family is chosen inside the subgraph (CMK Detailer Family Enable)."""

    @classmethod
    def INPUT_TYPES(cls):
        modes, options = CMKFaceRebuildModeEnable.INPUT_TYPES()["required"]["facerebuild_mode"]
        return {"required": {MASKDETAILER_MODE_WIDGET: (modes, dict(options))}}

    @staticmethod
    def read(maskdetailer_mode="ENABLE"):
        return CMKFaceRebuildModeEnable.read(maskdetailer_mode)
