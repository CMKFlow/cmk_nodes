"""Advanced Multi-Model Detailer (2.8.0): one Prepare, three Detailers.

CMK Detailer Prepare Advanced Multi-Model -Pipe- builds three deferred
DETAILER pipes: model specifications, the settings of each Detailer and the
arguments of the unchanged Multi-Model Prepare, but no model, VAE, SAM model
or conditioning.  CMK Smart Detailer Multi-Model -Pipe- materialises its pipe
on a branch-cache MISS (``_materialize_detailer``), so models exist only while
a Detailer samples and no node output holds them.

CMK Detailer Merge Multi-Model requests the Detailer branches lazily, the
Z-Image Turbo branches first and SDXL after them, and merges their SEGS in
slot order.  Of a None branch it requests only the diagnostic, last.  The
guarded ZIT release in the SDXL path of the Multi-Model Prepare can therefore
free ZIT before SDXL loads.

The merge also passes each Detailer diagnostic on unchanged (outputs
``diagnostic 1..3``, 2.8.0 Plan B).  The single Detailer previews hang behind
the merge instead of directly on the Detailers: an output node on a Detailer
would be an independent output root and run that Detailer outside the
family-ordered requests.
"""

from __future__ import annotations

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:  # Unit tests outside a running ComfyUI installation.
    class ExecutionBlocker:
        def __init__(self, message):
            self.message = message

from ..utils.cmk_diagnostic import make_diagnostic_payload
from ..utils.prepare_card import footer_card, log_block_rows
from ..utils.segs_branch_merge import merge_segs_collections_masked, valid_segs
from ..utils.segs_diagnostic import merge_diagnostic_stages
from ..utils.stable_segs import stable_branch_components
from .cmk_detailer_prepare_multimodel import (
    DETAILER_MODEL_FAMILIES,
    DETAILER_MULTIMODEL_PIPE,
    SDXL_SAMPLING_ENTRY,
    ZIT_SAMPLING_ENTRY,
    CMKDetailerPrepareMultiModelPipe,
    detailer_model_family,
)
from .cmk_log_pipe import cmk_add_block, cmk_make_block
from .cmk_sampler_prepare import _clean_text, _float, _int
from .cmk_visual import empty_visual, register_provider

PREPARE_LABEL = "CMK Detailer Prepare Advanced Multi-Model -Pipe-"
MERGE_LABEL = "CMK Detailer Merge Multi-Model"
SLOTS = (1, 2, 3)
SHARED_INPUTS = (
    "sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler",
    "stop_at_clip_layer", "pag_scale", "sampling", "zsnr",
    "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2",
)
SDXL_SLOT_INPUTS = ("steps", "cfg", "sampling_entry", "sampler", "scheduler")
ZIT_SLOT_INPUTS = ("zit_steps", "zit_cfg", "zit_sampling_entry", "zit_sampler", "zit_scheduler", "zit_model_shift")
FAMILY_LABELS = {"none": "NONE", "sdxl": "SDXL", "z_image_turbo": "Z-IMAGE TURBO"}
GLOBAL_ENABLE_VALUES = ["ENABLE", "DISABLE"]


def global_enabled(inputs):
    """The module switch of the Advanced Prepare (ENABLE / DISABLE dropdown)."""
    return inputs.get("detailer_global_enable", "ENABLE") != "DISABLE"


def slot_name(name, slot):
    return f"{name}_{slot}"


def slot_excluded_inputs(inputs, slot):
    """Advanced Prepare inputs that do not affect Detailer ``slot``.

    The other slots, the settings of the family this slot does not use, the
    LOG, and the model source of the unused family.
    """
    family = detailer_model_family(inputs.get(slot_name("model_family", slot)))
    # The global switch only decides whether the module runs; an executed
    # Detailer always has it on.
    excluded = ["LOG", "detailer_global_enable"]
    for other in SLOTS:
        if other == slot:
            continue
        excluded.append(slot_name("model_family", other))
        excluded.extend(slot_name(name, other) for name in (*SDXL_SLOT_INPUTS, *ZIT_SLOT_INPUTS))
    if family != "sdxl":
        excluded.extend(slot_name(name, slot) for name in SDXL_SLOT_INPUTS)
        excluded.extend(("MODEL SDXL", "stop_at_clip_layer", "pag_scale", "sampling", "zsnr",
                         "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2"))
    if family != "z_image_turbo":
        excluded.extend(slot_name(name, slot) for name in ZIT_SLOT_INPUTS)
        excluded.append("MODEL ZIT")
    return tuple(excluded)


class CMKDetailerPrepareAdvancedMultiModelPipe:
    """Three independently configured Detailers from one preparation.

    Promptless: the Multi-Model Prepare runs with an empty prompt (or the
    first-pass prompt) and without local LoRAs, exactly as the Standard
    Detailer does without a prompt entry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        source = CMKDetailerPrepareMultiModelPipe.INPUT_TYPES()
        base = source["required"]
        required = {
            "MODEL SDXL": base["MODEL SDXL"],
            "PROCESS": base["PROCESS"],
            "IMAGE": base["IMAGE"],
        }
        # Widget order follows the sections of the node: shared settings, then
        # per Detailer its MODEL FAMILY and its settings.
        for name in SHARED_INPUTS:
            required[name] = base[name]
        for slot in SLOTS:
            required[slot_name("model_family", slot)] = (DETAILER_MODEL_FAMILIES, {"default": "None"})
            for name in (*SDXL_SLOT_INPUTS, *ZIT_SLOT_INPUTS):
                required[slot_name(name, slot)] = base[name]
        # Global module switch, last so saved widget values keep their positions.
        required["detailer_global_enable"] = (GLOBAL_ENABLE_VALUES, {"default": "ENABLE"})
        return {
            "required": required,
            "optional": {
                "LOG": source["optional"]["LOG"],
                "MODEL ZIT": source["optional"]["MODEL ZIT"],
            },
        }

    RETURN_TYPES = (
        DETAILER_MULTIMODEL_PIPE, DETAILER_MULTIMODEL_PIPE, DETAILER_MULTIMODEL_PIPE,
        "BOOLEAN", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("DETAILER 1", "DETAILER 2", "DETAILER 3", "ENABLE", "LOG", "diagnostic")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Flow/PostProcess"

    @staticmethod
    def families(inputs):
        """Family per Detailer that runs: all None while the module is off."""
        if not global_enabled(inputs):
            return {slot: "none" for slot in SLOTS}
        return {slot: detailer_model_family(inputs.get(slot_name("model_family", slot))) for slot in SLOTS}

    def check_lazy_status(self, **kwargs):
        families = set(self.families(kwargs).values())
        needed = []
        # Unconnected optional inputs are absent from kwargs and never requested.
        if "LOG" in kwargs and kwargs["LOG"] is None:
            needed.append("LOG")
        if "sdxl" in families and kwargs.get("MODEL SDXL") is None:
            needed.append("MODEL SDXL")
        # An unconnected optional input is absent from kwargs and never requested.
        if "z_image_turbo" in families and "MODEL ZIT" in kwargs and kwargs["MODEL ZIT"] is None:
            needed.append("MODEL ZIT")
        return needed

    @staticmethod
    def _prepare_arguments(inputs, slot, family, process, image):
        values = {name: inputs[name] for name in SHARED_INPUTS}
        for name in (*SDXL_SLOT_INPUTS, *ZIT_SLOT_INPUTS):
            values[name] = inputs[slot_name(name, slot)]
        return {
            "PROCESS": process,
            "IMAGE": image,
            "model_family": {"sdxl": "SDXL", "z_image_turbo": "Z-Image Turbo"}.get(family, "None"),
            "use_lora_from_1st_pass": False,
            "use_1st_pass_sampling": False,
            "lora_name": "None",
            "strength_model": 1.0,
            "strength_clip": 1.0,
            "prompt_pos": "",
            "prompt_neg": "",
            "LOG": None,
            "MODEL SDXL": inputs.get("MODEL SDXL"),
            "MODEL ZIT": inputs.get("MODEL ZIT"),
            **values,
        }

    @staticmethod
    def _cache_fields(inputs, slot, family, process):
        """The settings the Smart Detailer branch-cache key reads from a pipe."""
        inherit = bool(inputs["use_prompt_lora_from_sampler"]) and any(
            key in process for key in ("prompt_pos", "prompt_neg")
        )
        sdxl = family == "sdxl"
        fields = {
            "detailer_seed": _int(process.get("seed"), 0),
            "detailer_sam_model_name": str(inputs["sam_model_name"]),
            "detailer_use_prompt_lora_from_sampler": inherit,
            "detailer_use_prompt_from_1st_pass": inherit,
            "detailer_use_lora_from_1st_pass": False,
            "detailer_use_1st_pass_sampling": False,
            "detailer_prompt_pos": _clean_text(process.get("prompt_pos") if inherit else "", ""),
            "detailer_prompt_neg": _clean_text(process.get("prompt_neg") if inherit and sdxl else "", ""),
            "detailer_active_loras": "",
            "detailer_model_family": family,
            "detailer_loaded_loras": "",
            "detailer_zit_trigger_words": "",
        }
        if sdxl:
            fields.update({
                "detailer_steps": _int(inputs[slot_name("steps", slot)], 30),
                "detailer_cfg": _float(inputs[slot_name("cfg", slot)], 7.0),
                "detailer_sampler": str(inputs[slot_name("sampler", slot)]),
                "detailer_scheduler": str(inputs[slot_name("scheduler", slot)]),
                "detailer_sampling_entry": _float(inputs[slot_name("sampling_entry", slot)], SDXL_SAMPLING_ENTRY),
                "detailer_model_identity": None,
                "detailer_zit_model_shift": None,
                **{f"detailer_{name}": inputs[name] for name in (
                    "stop_at_clip_layer", "pag_scale", "sampling", "zsnr",
                    "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2",
                )},
            })
        else:
            spec = inputs.get("MODEL ZIT") or {}
            fields.update({
                "detailer_steps": max(1, _int(inputs[slot_name("zit_steps", slot)], 8)),
                "detailer_cfg": _float(inputs[slot_name("zit_cfg", slot)], 1.0),
                "detailer_sampler": str(inputs[slot_name("zit_sampler", slot)]),
                "detailer_scheduler": str(inputs[slot_name("zit_scheduler", slot)]),
                "detailer_sampling_entry": _float(inputs[slot_name("zit_sampling_entry", slot)], ZIT_SAMPLING_ENTRY),
                "detailer_model_identity": {
                    name: str(spec.get(name, "")) for name in (
                        "diffusion_model_name", "text_encoder_name", "vae_name", "weight_dtype", "clip_device",
                    )
                },
                "detailer_zit_model_shift": _float(inputs[slot_name("zit_model_shift", slot)], 3.0),
                **{f"detailer_{name}": None for name in (
                    "stop_at_clip_layer", "pag_scale", "sampling", "zsnr",
                    "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2",
                )},
            })
        return fields

    def prepare(self, PROCESS, IMAGE, LOG=None, **inputs):
        if IMAGE is None:
            raise ValueError(f"{PREPARE_LABEL}: IMAGE is missing")
        families = self.families(inputs)
        active = [slot for slot, family in families.items() if family != "none"]
        if active:
            if not isinstance(PROCESS, dict):
                raise TypeError(f"{PREPARE_LABEL}: PROCESS must be a CMK process pipe")
            if PROCESS.get("result_contract") != "family_neutral":
                raise ValueError(f"{PREPARE_LABEL} requires the PostProcess Boundary")
        process = dict(PROCESS) if isinstance(PROCESS, dict) else {}
        pipes = []
        lines = []
        for slot in SLOTS:
            family = families[slot]
            pipe = {
                "source_pipe": process,
                "_source_pipe": process,
                "detailer_image": IMAGE,
                "detailer_slot": slot,
                "detailer_model_family": family,
                "detailer_global_enable": family != "none",
                "boolean_detailer_enable": family != "none",
            }
            if family == "none":
                pipe["detailer_prepare_bypassed"] = True
                configured = detailer_model_family(inputs.get(slot_name("model_family", slot)))
                lines.append(
                    f"DETAILER {slot}      : {FAMILY_LABELS[configured]} · GLOBAL DISABLE"
                    if configured != "none" else f"DETAILER {slot}      : NONE · BYPASS"
                )
            else:
                pipe.update(self._cache_fields(inputs, slot, family, process))
                pipe["detailer_model_deferred"] = True
                pipe["detailer_prepare_arguments"] = self._prepare_arguments(inputs, slot, family, process, IMAGE)
                lines.append(
                    f"DETAILER {slot}      : {FAMILY_LABELS[family]} · STEPS {pipe['detailer_steps']} · "
                    f"CFG {pipe['detailer_cfg']:g} · {pipe['detailer_sampler']} / {pipe['detailer_scheduler']} · "
                    f"ENTRY {pipe['detailer_sampling_entry']:g} %"
                )
            pipes.append(pipe)
        if not global_enabled(inputs):
            lines.insert(0, "GLOBAL          : DISABLE · MODULE BYPASS")
        lines.append("PROMPT          : " + ("FROM 1ST PASS" if inputs.get("use_prompt_lora_from_sampler") else "NONE (PROMPTLESS)"))
        lines.append("MATERIALISATION : PER DETAILER ON CACHE MISS · ZIT BEFORE SDXL")
        log_pipe = cmk_add_block(LOG, "Detailer Prepare Advanced", 70, lines, True)
        diagnostic = make_diagnostic_payload(
            title="Detailer Prepare Advanced",
            node=PREPARE_LABEL,
            previews=[IMAGE],
            summary=" | ".join(f"D{slot}={FAMILY_LABELS[families[slot]]}" for slot in SLOTS),
            details="\n".join(lines),
            mode="deferred",
            metadata={"families": {str(slot): families[slot] for slot in SLOTS}},
        )
        diagnostic = footer_card(diagnostic, log_block_rows(log_pipe, "Detailer Prepare Advanced"))
        return (*pipes, bool(active), log_pipe, diagnostic)


class CMKDetailerMergeMultiModel:
    """Merge three Detailer branches in slot order, requested family-grouped.

    Branches are alpha-blended by their effective pasteback masks
    (``merge_segs_collections_masked``), not by their sampling ROIs.
    """

    # Part of the Detailer boundary key: a result of an earlier merge rule is
    # never reused for this one.
    MERGE_SEMANTICS = "pasteback_mask_v1"

    @classmethod
    def INPUT_TYPES(cls):
        lazy = {"lazy": True}
        optional = {"LOG": ("CMK_LOG_PIPE",), "VISUAL": ("CMK_VISUAL_PIPE",)}
        for slot in SLOTS:
            optional[f"SEGS {slot}"] = ("SEGS", lazy)
            optional[f"LOG {slot}"] = ("CMK_LOG_PIPE", lazy)
            optional[f"diagnostic {slot}"] = ("CMK_DIAGNOSTIC", lazy)
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1, "advanced": True}),
                "alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "advanced": True}),
            },
            "optional": optional,
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    # The slot diagnostics are appended, so the slots of the existing outputs
    # and every saved link to them stay unchanged.
    RETURN_TYPES = (
        "IMAGE", "CMK_LOG_PIPE", "CMK_VISUAL_PIPE", "CMK_DIAGNOSTIC",
        "CMK_DIAGNOSTIC", "CMK_DIAGNOSTIC", "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("IMAGE", "LOG", "VISUAL", "diagnostic", "diagnostic 1", "diagnostic 2", "diagnostic 3")
    FUNCTION = "merge"
    CATEGORY = "CMK/Flow/PostProcess"

    @staticmethod
    def slot_families(prompt, unique_id):
        """Family per connected slot, read from the Advanced Prepare in the prompt.

        SEGS i -> Smart Detailer -> DETAILER -> Advanced Prepare output k ->
        model_family_{k+1}.  A slot that cannot be resolved is "unknown" and
        is requested last.
        """
        node = prompt.get(str(unique_id)) if isinstance(prompt, dict) else None
        inputs = (node or {}).get("inputs", {}) if isinstance(node, dict) else {}
        result = {}
        for slot in SLOTS:
            link = inputs.get(f"SEGS {slot}")
            if not (isinstance(link, (list, tuple)) and len(link) == 2):
                continue
            family = "unknown"
            detailer = prompt.get(str(link[0]))
            detailer_link = ((detailer or {}).get("inputs") or {}).get("DETAILER") if isinstance(detailer, dict) else None
            if isinstance(detailer_link, (list, tuple)) and len(detailer_link) == 2:
                prepare = prompt.get(str(detailer_link[0]))
                if isinstance(prepare, dict) and prepare.get("class_type") == "CMKDetailerPrepareAdvancedMultiModelPipe":
                    value = (prepare.get("inputs") or {}).get(slot_name("model_family", int(detailer_link[1]) + 1))
                    if isinstance(value, str):
                        family = detailer_model_family(value)
            result[slot] = family
        return result

    def check_lazy_status(self, prompt=None, unique_id=None, **kwargs):
        families = self.slot_families(prompt, unique_id)
        branch = ("SEGS", "LOG", "diagnostic")
        # A slot's diagnostic joins the request of its own family group; of a
        # None slot only the diagnostic (its bypass) is requested, last.
        for wanted, names in (("z_image_turbo", branch), ("sdxl", branch), ("unknown", branch), ("none", ("diagnostic",))):
            missing = [
                f"{name} {slot}"
                for slot, family in sorted(families.items()) if family == wanted
                for name in names
                if f"{name} {slot}" in kwargs and kwargs[f"{name} {slot}"] is None
            ]
            if missing:
                return missing
        return []

    @staticmethod
    def _slot_blocks(base, slot_log):
        """Blocks a Detailer added on top of the shared Prepare LOG."""
        base_blocks = list((base or {}).get("blocks", [])) if isinstance(base, dict) else []
        blocks = list((slot_log or {}).get("blocks", [])) if isinstance(slot_log, dict) else []
        if blocks[: len(base_blocks)] == base_blocks:
            return blocks[len(base_blocks):]
        return blocks

    def merge(self, IMAGE, feather=5, alpha=255, LOG=None, VISUAL=None, prompt=None, unique_id=None, **kwargs):
        if IMAGE is None:
            raise ValueError(f"{MERGE_LABEL}: IMAGE is missing")
        families = self.slot_families(prompt, unique_id)
        height, width = int(IMAGE.shape[1]), int(IMAGE.shape[2])
        segs_collections = []
        blocks = list((LOG or {}).get("blocks", [])) if isinstance(LOG, dict) else []
        for slot in SLOTS:
            family = families.get(slot, "none")
            segs = kwargs.get(f"SEGS {slot}")
            if family in ("none",) or segs is None:
                segs_collections.append(((width, height), []))
                lines = [f"STATUS          : {'NONE · BYPASS' if family == 'none' else 'NOT CONNECTED'}"]
                blocks.append(cmk_make_block(f"Detailer {slot}", 70, lines, True))
                continue
            segs_collections.append(segs)
            for block in self._slot_blocks(LOG, kwargs.get(f"LOG {slot}")):
                block = dict(block)
                block["title"] = f"Detailer {slot} · {block.get('title', '')}".rstrip(" ·")
                blocks.append(block)
        result_image = merge_segs_collections_masked(
            authoritative_image=IMAGE,
            segs_collections=segs_collections,
            feather=int(feather),
            alpha=int(alpha),
        )
        log_pipe = dict(LOG) if isinstance(LOG, dict) else {}
        log_pipe["blocks"] = blocks

        def has_content(value):
            if not valid_segs(value):
                return False
            stable = stable_branch_components(value)
            if stable is not None:
                return bool(stable[1].any().item())
            return bool(value[1])

        result_visual = empty_visual() if VISUAL is None else VISUAL
        if any(has_content(value) for value in segs_collections):
            result_visual = register_provider(
                result_visual,
                module_instance_id=unique_id or "detailer-advanced",
                module_type="CMKDetailerMergeMultiModel",
                module_label="Detailer",
                sequence=23,
                channels={"before": IMAGE, "after": result_image},
                status="completed",
                branch="sdxl",
                stage_key="sdxl.detailer.advanced",
            )
        diagnostic = self._merge_diagnostic(IMAGE, result_image, segs_collections, families, feather, alpha)
        # Passed on as received; an unconnected slot blocks its preview.
        slot_diagnostics = tuple(
            kwargs.get(f"diagnostic {slot}") if kwargs.get(f"diagnostic {slot}") is not None else ExecutionBlocker(None)
            for slot in SLOTS
        )
        return (result_image, log_pipe, result_visual, diagnostic, *slot_diagnostics)

    @staticmethod
    def _merge_diagnostic(image, result_image, segs_collections, families, feather, alpha):
        """The merge's own spatial diagnostic (2.8.0), image data only.

        The single Detailer diagnostics are not part of it; they leave the
        merge unchanged on the outputs ``diagnostic 1..3``.
        """
        stages, metrics = merge_diagnostic_stages(
            image, result_image, segs_collections, families,
            feather=int(feather), alpha=int(alpha),
        )
        labels = {"sdxl": "SDXL", "z_image_turbo": "ZIT", "none": "NONE", "unknown": "?"}
        lines = ["ORDER           : D1 → D2 → D3 (alpha blend by pasteback mask)"]
        for slot in SLOTS:
            lines.append(
                f"D{slot}              : {labels.get(families.get(slot, 'unknown'), '?'):<5}"
                f" · {metrics['share_px'][f'D{slot}']:.0f} px share in result"
            )
        lines.append(f"ROI OVERLAP     : {metrics['roi_overlap_px']} px (no effect)")
        overlaps = metrics["mask_overlap_px"]
        lines.append(
            "MASK OVERLAP    : "
            + (", ".join(f"{key} {value} px" for key, value in overlaps.items()) or "none")
        )
        summary = "\n".join(lines)
        warnings = [f"Pasteback masks overlap, later slot on top: {key} {value} px" for key, value in overlaps.items()]
        return make_diagnostic_payload(
            title="CMK Flow · Detailer Advanced · Merge",
            node=MERGE_LABEL,
            stages=stages,
            summary=summary,
            details=summary,
            mode="Merge",
            metadata={"slot_order": [1, 2, 3], "families": {f"D{slot}": families.get(slot, "unknown") for slot in SLOTS}},
            metrics=metrics,
            warnings=warnings,
        )
