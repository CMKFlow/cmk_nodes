"""Advanced FaceProcess 2.8 (F5): one Prepare, three FaceProcess branches.

CMK FaceProcess Prepare Advanced Multi-Model -Pipe- builds three FACE
specifications on the unchanged Standard contract (``FACE_CONTRACT``), so each
branch is the approved CMK FaceProcess Multi-Model -Pipe- with its own branch
cache.  Every branch has its own mode (NONE / RESTORE / SDXL / ZIT), Refine,
restore model and sampling settings; the shared settings and the promptless
behaviour are those of the Advanced Multi-Model Detailer.  Nothing is loaded
here; a branch materialises on its own cache MISS.

CMK FaceProcess Merge Multi-Model requests the branches lazily in resource
order RESTORE -> ZIT -> SDXL (ZIT is released before SDXL loads) and composes
them in slot order D1 -> D2 -> D3 from their F4 processing layers
(``compose_layers``): the actual restore, detailer and refine masks with the
blend arithmetic of their paths, each branch applied onto the result of the
previous ones.  The single branch diagnostics leave the merge unchanged, so
the previews hang behind the merge and add no output root on a branch.
"""

from __future__ import annotations

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:  # Unit tests outside a running ComfyUI installation.
    class ExecutionBlocker:
        def __init__(self, message):
            self.message = message

import torch

from ..utils.cmk_diagnostic import make_diagnostic_payload
from ..utils.prepare_card import footer_card, log_block_rows
from ..utils.face_diagnostic import merge_layer_stages
from ..utils.face_layers import LAYER_SCHEMA, compose_layers
from ..utils.stable_segs import image_signature, stable_branch_components
from .cmk_detailer_advanced_multimodel import SDXL_SLOT_INPUTS, SHARED_INPUTS, ZIT_SLOT_INPUTS, CMKDetailerMergeMultiModel
from .cmk_faceprocess_mode import FACEPROCESS_MODES, faceprocess_mode
from .cmk_faceprocess_multimodel import (
    ADVANCED_GLOBAL_ENABLE,
    ADVANCED_SLOTS,
    FACE_MULTIMODEL_PIPE,
    MODE_LABELS,
    RESTORE_VISIBILITY_DEFAULT,
    CMKFaceProcessPrepareMultiModelPipe,
    advanced_slot_mode,
    branch_mode,
)
from .cmk_log_pipe import cmk_add_block, cmk_make_block
from .cmk_visual import empty_visual, register_provider

PREPARE_LABEL = "CMK FaceProcess Prepare Advanced Multi-Model -Pipe-"
MERGE_LABEL = "CMK FaceProcess Merge Multi-Model"
GLOBAL_ENABLE_VALUES = ["ENABLE", "DISABLE"]
# Lazy request order of the merge: no model, then ZIT before SDXL.
RESOURCE_ORDER = ("restore", "z_image_turbo", "sdxl", "unknown")
MODE_VALUES = {faceprocess_mode(value): value for value in FACEPROCESS_MODES}


class CMKFaceProcessPrepareAdvancedMultiModelPipe:
    """Three independently configured FaceProcess branches from one preparation.

    Promptless like the Advanced Multi-Model Detailer: SDXL/ZIT branches use
    an empty prompt (or the first-pass prompt) and no local LoRA.
    """

    @classmethod
    def INPUT_TYPES(cls):
        standard = CMKFaceProcessPrepareMultiModelPipe.INPUT_TYPES()
        base = standard["required"]
        required = {name: base[name] for name in ("MODEL SDXL", "PROCESS", "IMAGE")}
        for name in SHARED_INPUTS:
            required[name] = base[name]
        for slot in ADVANCED_SLOTS:
            required[f"process_mode_{slot}"] = (FACEPROCESS_MODES, {"default": "None"})
            for name in ("refine_mode", "restore_model", "restore_visibility", *SDXL_SLOT_INPUTS, *ZIT_SLOT_INPUTS):
                required[f"{name}_{slot}"] = base[name]
        # Global module switch, last so saved widget values keep their positions.
        required[ADVANCED_GLOBAL_ENABLE] = (GLOBAL_ENABLE_VALUES, {"default": "ENABLE"})
        return {"required": required,
                "optional": {"LOG": standard["optional"]["LOG"], "MODEL ZIT": standard["optional"]["MODEL ZIT"]}}

    RETURN_TYPES = (FACE_MULTIMODEL_PIPE, FACE_MULTIMODEL_PIPE, FACE_MULTIMODEL_PIPE,
                    "BOOLEAN", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("FACE 1", "FACE 2", "FACE 3", "ENABLE", "LOG", "diagnostic")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Flow/PostProcess"

    @staticmethod
    def modes(inputs):
        return {slot: advanced_slot_mode(inputs, slot) or "none" for slot in ADVANCED_SLOTS}

    def check_lazy_status(self, IMAGE=None, PROCESS=None, **kwargs):
        modes = set(self.modes(kwargs).values())
        needed = []
        if IMAGE is None:
            needed.append("IMAGE")
        # An unconnected optional input is absent from kwargs and never requested.
        if "LOG" in kwargs and kwargs["LOG"] is None:
            needed.append("LOG")
        if modes & {"sdxl", "z_image_turbo"} and PROCESS is None:
            needed.append("PROCESS")
        if "sdxl" in modes and kwargs.get("MODEL SDXL") is None:
            needed.append("MODEL SDXL")
        if "z_image_turbo" in modes and "MODEL ZIT" in kwargs and kwargs["MODEL ZIT"] is None:
            needed.append("MODEL ZIT")
        return needed

    @staticmethod
    def _slot_settings(inputs, slot):
        """The Standard Prepare's settings for FACE ``slot`` (promptless, no local LoRA)."""
        settings = {name: inputs[name] for name in SHARED_INPUTS}
        settings.update({name: inputs[f"{name}_{slot}"] for name in (*SDXL_SLOT_INPUTS, *ZIT_SLOT_INPUTS)})
        settings.update({
            "use_lora_from_1st_pass": False, "use_1st_pass_sampling": False, "lora_name": "None",
            "strength_model": 1.0, "strength_clip": 1.0, "prompt_pos": "", "prompt_neg": "",
            "MODEL SDXL": inputs.get("MODEL SDXL"), "MODEL ZIT": inputs.get("MODEL ZIT"),
        })
        return settings

    def prepare(self, IMAGE, PROCESS=None, LOG=None, **inputs):
        if IMAGE is None:
            raise ValueError(f"{PREPARE_LABEL}: IMAGE is missing")
        modes = self.modes(inputs)
        standard = CMKFaceProcessPrepareMultiModelPipe()
        faces, lines = [], []
        for slot in ADVANCED_SLOTS:
            mode = modes[slot]
            face = standard.prepare(
                IMAGE, process_mode=MODE_VALUES[mode],
                refine_mode=inputs[f"refine_mode_{slot}"], restore_model=inputs[f"restore_model_{slot}"],
                restore_visibility=inputs.get(f"restore_visibility_{slot}", RESTORE_VISIBILITY_DEFAULT),
                PROCESS=PROCESS if mode in ("sdxl", "z_image_turbo") else None, LOG=None,
                **self._slot_settings(inputs, slot),
            )[0]
            face["face_slot"] = slot
            faces.append(face)
            if mode == "none":
                configured = advanced_slot_mode({**inputs, ADVANCED_GLOBAL_ENABLE: "ENABLE"}, slot) or "none"
                lines.append(f"FACEPROCESS {slot}   : "
                             + (f"{MODE_LABELS[configured]} · GLOBAL DISABLE" if configured != "none" else "NONE · BYPASS"))
            elif mode == "restore":
                lines.append(f"FACEPROCESS {slot}   : RESTORE · {face['face_restore_model']} · VISIBILITY "
                             f"{face['face_restore_visibility']:.2f} · REFINE {face['face_refine_mode']}")
            else:
                prefix = "" if mode == "sdxl" else "zit_"
                lines.append(f"FACEPROCESS {slot}   : {MODE_LABELS[mode]} · STEPS {inputs[f'{prefix}steps_{slot}']} · "
                             f"CFG {float(inputs[f'{prefix}cfg_{slot}']):g} · ENTRY "
                             f"{float(inputs[f'{prefix}sampling_entry_{slot}']):g} % · REFINE {face['face_refine_mode']}")
        if inputs.get(ADVANCED_GLOBAL_ENABLE, "ENABLE") == "DISABLE":
            lines.insert(0, "GLOBAL          : DISABLE · MODULE BYPASS")
        lines.append("PROMPT          : " + ("FROM 1ST PASS" if inputs.get("use_prompt_lora_from_sampler") else "NONE (PROMPTLESS)"))
        lines.append("MATERIALISATION : PER BRANCH ON CACHE MISS · RESTORE → ZIT → SDXL")
        log_pipe = cmk_add_block(LOG, "FaceProcess Prepare Advanced", 80, lines, True)
        diagnostic = make_diagnostic_payload(
            title="FaceProcess Prepare Advanced", node=PREPARE_LABEL, previews=[IMAGE],
            summary=" | ".join(f"D{slot}={MODE_LABELS[modes[slot]]}" for slot in ADVANCED_SLOTS),
            details="\n".join(lines), mode="deferred",
            metadata={"modes": {str(slot): modes[slot] for slot in ADVANCED_SLOTS}, "models_loaded": False},
        )
        diagnostic = footer_card(diagnostic, log_block_rows(log_pipe, "FaceProcess Prepare Advanced"))
        return (*faces, any(mode != "none" for mode in modes.values()), log_pipe, diagnostic)


class CMKFaceProcessMergeMultiModel:
    """Compose three FaceProcess branches from their processing layers in slot
    order, requested in resource order RESTORE -> ZIT -> SDXL."""

    @classmethod
    def INPUT_TYPES(cls):
        lazy = {"lazy": True}
        optional = {"LOG": ("CMK_LOG_PIPE",), "VISUAL": ("CMK_VISUAL_PIPE",)}
        for slot in ADVANCED_SLOTS:
            optional[f"SEGS {slot}"] = ("SEGS", lazy)
            optional[f"LOG {slot}"] = ("CMK_LOG_PIPE", lazy)
            optional[f"diagnostic {slot}"] = ("CMK_DIAGNOSTIC", lazy)
        return {"required": {"IMAGE": ("IMAGE",)}, "optional": optional,
                "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "CMK_LOG_PIPE", "CMK_VISUAL_PIPE", "CMK_DIAGNOSTIC",
                    "CMK_DIAGNOSTIC", "CMK_DIAGNOSTIC", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("IMAGE", "LOG", "VISUAL", "diagnostic", "diagnostic 1", "diagnostic 2", "diagnostic 3")
    FUNCTION = "merge"
    CATEGORY = "CMK/Flow/PostProcess"

    @staticmethod
    def slot_modes(prompt, unique_id):
        """Mode per connected slot: SEGS i -> FaceProcess branch -> its Prepare
        FACE slot.  A slot that cannot be resolved is "unknown"."""
        node = prompt.get(str(unique_id)) if isinstance(prompt, dict) else None
        inputs = (node or {}).get("inputs", {}) if isinstance(node, dict) else {}
        result = {}
        for slot in ADVANCED_SLOTS:
            link = inputs.get(f"SEGS {slot}")
            if isinstance(link, (list, tuple)) and len(link) == 2:
                result[slot] = branch_mode(prompt, link[0]) or "unknown"
        return result

    def check_lazy_status(self, prompt=None, unique_id=None, **kwargs):
        modes = self.slot_modes(prompt, unique_id)
        branch = ("SEGS", "LOG", "diagnostic")
        # A slot's diagnostic joins the request of its own group; of a NONE
        # slot only the diagnostic (its bypass) is requested, last.
        for wanted, names in (*((mode, branch) for mode in RESOURCE_ORDER), ("none", ("diagnostic",))):
            missing = [f"{name} {slot}" for slot, mode in sorted(modes.items()) if mode == wanted for name in names
                       if f"{name} {slot}" in kwargs and kwargs[f"{name} {slot}"] is None]
            if missing:
                return missing
        return []

    def merge(self, IMAGE, LOG=None, VISUAL=None, prompt=None, unique_id=None, **kwargs):
        if IMAGE is None:
            raise ValueError(f"{MERGE_LABEL}: IMAGE is missing")
        modes = self.slot_modes(prompt, unique_id)
        signature = image_signature(IMAGE)
        result = IMAGE.detach().cpu().to(torch.float32)
        blocks = list((LOG or {}).get("blocks", [])) if isinstance(LOG, dict) else []
        slot_info = {}
        for slot in ADVANCED_SLOTS:
            mode = modes.get(slot, "none")
            segs = kwargs.get(f"SEGS {slot}")
            if mode == "none" or segs is None:
                status = "NONE · BYPASS" if mode == "none" else "NOT CONNECTED"
                blocks.append(cmk_make_block(f"FaceProcess {slot}", 90, [f"STATUS          : {status}"], True))
                slot_info[slot] = {"mode": mode, "layers": [], "changed_px": 0}
                continue
            if getattr(segs, "cmk_layer_schema", None) != LAYER_SCHEMA:
                raise ValueError(f"{MERGE_LABEL}: SEGS {slot} carry no FaceProcess processing layers")
            stable = stable_branch_components(segs)
            if stable is not None and stable[2] and stable[2] != signature:
                raise RuntimeError(f"{MERGE_LABEL}: SEGS {slot} were processed from a different image; "
                                   "the branch cache must be recomputed")
            layers = list(segs.cmk_layers)
            composed = compose_layers(result, layers) if layers else result
            changed = int((composed != result).any(dim=-1).sum().item())
            result = composed
            slot_info[slot] = {"mode": mode, "layers": layers, "changed_px": changed}
            for block in CMKDetailerMergeMultiModel._slot_blocks(LOG, kwargs.get(f"LOG {slot}")):
                block = dict(block)
                block["title"] = f"FaceProcess {slot} · {block.get('title', '')}".rstrip(" ·")
                blocks.append(block)
        log_pipe = dict(LOG) if isinstance(LOG, dict) else {}
        log_pipe["blocks"] = blocks

        result_visual = empty_visual() if VISUAL is None else VISUAL
        if any(info["changed_px"] for info in slot_info.values()):
            result_visual = register_provider(
                result_visual, module_instance_id=unique_id or "faceprocess-advanced",
                module_type="CMKFaceProcessMergeMultiModel", module_label="FaceProcess", sequence=30,
                channels={"before": IMAGE, "after": result}, status="completed", branch="sdxl",
                stage_key="sdxl.faceprocess.advanced",
            )
        diagnostic = self._merge_diagnostic(IMAGE, result, slot_info)
        # Passed on as received; an unconnected slot blocks its preview.
        slot_diagnostics = tuple(
            kwargs.get(f"diagnostic {slot}") if kwargs.get(f"diagnostic {slot}") is not None else ExecutionBlocker(None)
            for slot in ADVANCED_SLOTS
        )
        return (result, log_pipe, result_visual, diagnostic, *slot_diagnostics)

    @staticmethod
    def _merge_diagnostic(image, result, slot_info):
        """The merge's own spatial diagnostic from the branch layers it composed
        (INPUT LAYERS, OVERLAP, COMPOSITE, FINAL); the image is not touched."""
        modes = {slot: slot_info[slot]["mode"] for slot in ADVANCED_SLOTS}
        stages, metrics = merge_layer_stages(image, result, {slot: slot_info[slot]["layers"] for slot in ADVANCED_SLOTS}, modes)
        lines = ["ORDER           : D1 → D2 → D3 (processing layers on the previous result)",
                 "REQUESTS        : RESTORE → ZIT → SDXL"]
        for slot in ADVANCED_SLOTS:
            kinds = [layer["kind"] for layer in slot_info[slot]["layers"]]
            lines.append(f"D{slot}              : {MODE_LABELS.get(modes[slot], '?'):<7} · {slot_info[slot]['changed_px']} px changed"
                         f" · {metrics['share_px'][f'D{slot}']:.0f} px share in result · layers {', '.join(kinds) or 'none'}")
        overlaps = metrics["alpha_overlap_px"]
        lines.append(f"REGION OVERLAP  : {metrics['region_overlap_px']} px (frames only)")
        lines.append("ALPHA OVERLAP   : " + (", ".join(f"{key} {value} px" for key, value in overlaps.items()) or "none"))
        summary = "\n".join(lines)
        return make_diagnostic_payload(
            title="CMK Flow · FaceProcess Advanced · Merge", node=MERGE_LABEL, stages=stages,
            summary=summary, details=summary, mode="Merge",
            metadata={"slot_order": list(ADVANCED_SLOTS), "request_order": list(RESOURCE_ORDER),
                      "modes": {f"D{slot}": mode for slot, mode in modes.items()}, "layers": metrics["layers"]},
            metrics={**metrics, "changed_px": {f"D{slot}": slot_info[slot]["changed_px"] for slot in ADVANCED_SLOTS}},
            warnings=[f"Effective alphas overlap, later branch on top: {key} {value} px" for key, value in overlaps.items()],
        )
