"""FaceProcess 2.8 Standard (F2): deferred Prepare and Multi-Model execution.

CMK FaceProcess Prepare Multi-Model -Pipe- builds a FACE specification only:
the operating mode, the authoritative image and the arguments a branch needs
to prepare itself.  It never loads a model, SAM or conditioning, and it
requests a model source only for the mode that needs it:

    NONE     IMAGE + LOG                    (complete bypass, also no Refine)
    RESTORE  IMAGE + LOG                    (no diffusion model, no PROCESS)
    SDXL     IMAGE + LOG + PROCESS + MODEL SDXL
    ZIT      IMAGE + LOG + PROCESS + MODEL ZIT

CMK FaceProcess Multi-Model -Pipe- materialises on a branch-cache MISS only:
SDXL and ZIT run the unchanged CMK Detailer Prepare Multi-Model with the
Prepare's settings (same parameters and semantics as the Multi-Model
Detailer: sampling entry per family, prompt/seed/LoRA resolution, LoRA ZIT,
guarded ZIT release before SDXL); the 2.7 detailer step then samples with the
family's sampling entry.  RESTORE (F3)
needs no diffusion model or SAM: YOLO selects the faces, InsightFace assigns
its complete face objects (``utils.face_matching``) and the resolved restore
backend restores exactly those (``cmk_faceprocess_restore``).  Models exist
only while the branch runs; no output holds them.

Legacy CMK FaceProcess Prepare -Pipe- / CMK FaceProcess -Pipe- are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import folder_paths
import numpy as np
import torch

from ..engine.detector_engine import get_face_analyser
from ..engine.native_detailer import SEGSDetailer, SEGSPaste
from ..engine.native_face_restore import RestoreFaceAdvanced
from ..nodes.image.smart_detailer import _sampling_entry_to_denoise as detailer_entry_to_denoise
from ..nodes.swap.face_process import (
    _is_likely_non_face_detector,
    _normalize_refine_mode,
    _preferred_face_detector,
    _sort_face_detectors,
)
from ..utils.cmk_diagnostic import make_diagnostic_payload
from ..utils.face_diagnostic import detection_view, with_face_stages
from ..utils.face_layers import LAYER_SCHEMA, layers_roi, paste_layers, quantize_layer, refine_layer, restore_layer, with_layers
from ..utils.face_matching import MATCH_RULE, OK, assign_faces
from ..utils.stable_segs import make_stable_segs, segs_with_image_crops
from ..utils.tensor_utils import tensor_to_uint8_rgb
from .cmk_detailer_prepare_multimodel import CMKDetailerPrepareMultiModelPipe
from .cmk_faceprocess import CMKFaceProcessPipe
from .cmk_faceprocess_mode import FACEPROCESS_MODES, faceprocess_mode
from .cmk_faceprocess_restore import RestoreConfigurationError, backend_identity, resolve_restore_backend
from .cmk_log_pipe import cmk_add_block, cmk_block_to_string
from .cmk_persistent_cache import DISABLED_REVISION, build_node_fingerprint, save_pickle, write_status

PREPARE_CLASS = "CMKFaceProcessPrepareMultiModelPipe"
ADVANCED_PREPARE_CLASS = "CMKFaceProcessPrepareAdvancedMultiModelPipe"
EXECUTE_CLASS = "CMKFaceProcessMultiModelPipe"
PREPARE_LABEL = "CMK FaceProcess Prepare Multi-Model -Pipe-"
EXECUTE_LABEL = "CMK FaceProcess Multi-Model -Pipe-"
FACE_MULTIMODEL_PIPE = "CMK_FACE_MULTIMODEL_PIPE"
FACE_CONTRACT = "faceprocess_multimodel_v1"
MODE_LABELS = {"none": "NONE", "restore": "RESTORE", "sdxl": "SDXL", "z_image_turbo": "ZIT"}

# Prepare inputs by the mode that reads them.
SDXL_SETTINGS = (
    "use_1st_pass_sampling", "lora_name", "strength_model", "strength_clip", "prompt_neg",
    "steps", "cfg", "sampling_entry", "sampler", "scheduler", "stop_at_clip_layer", "pag_scale", "sampling",
    "zsnr", "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2",
)
ZIT_SETTINGS = ("zit_steps", "zit_cfg", "zit_sampling_entry", "zit_sampler", "zit_scheduler", "zit_model_shift")
DIFFUSION_SHARED = ("sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler",
                    "use_lora_from_1st_pass", "prompt_pos")
TRANSPORT = ("LOG",)

# Restore configuration of the Prepare (FACE carries it to the execution).
RESTORE_INPUTS = ("restore_model", "restore_visibility")
RESTORE_VISIBILITY_DEFAULT = 0.5
# Inputs of the legacy execution the 2.8 node does not offer:
# restore_facedetection / restore_codeformer_weight have no effect (E5);
# detail_denoise is replaced by the sampling entry per family of the Prepare;
# refine_mode and the restore configuration belong to the Prepare.
DROPPED_EXECUTE_INPUTS = ("restore_facedetection", "restore_codeformer_weight", "detail_denoise", "refine_mode",
                          *RESTORE_INPUTS)
REFINE_MODES = ["Off", "Detail", "Sharpen", "Smooth"]
STATUS_TEXT = {"OK": "RESTORED", "UNMATCHED": "SKIPPED · NO INSIGHTFACE ASSIGNMENT",
               "AMBIGUOUS": "SKIPPED · AMBIGUOUS ASSIGNMENT", "CONFLICT": "SKIPPED · AMBIGUOUS ASSIGNMENT (CONFLICT)"}
DETAIL_INPUTS = ("detail_guide_size", "detail_guide_size_for", "detail_max_size",
                 "detail_noise_mask", "detail_force_inpaint", "detail_paste_feather")


def prepare_excluded_inputs(mode):
    """Prepare inputs outside a mode's identity (never part of its branch key)."""
    if mode == "restore":
        return (*TRANSPORT, "MODEL ZIT", "MODEL SDXL", "LoRA ZIT", "PROCESS", *DIFFUSION_SHARED, *SDXL_SETTINGS, *ZIT_SETTINGS)
    if mode == "sdxl":
        return (*TRANSPORT, "MODEL ZIT", "LoRA ZIT", *RESTORE_INPUTS, *ZIT_SETTINGS)
    if mode == "z_image_turbo":
        return (*TRANSPORT, "MODEL SDXL", *RESTORE_INPUTS, *SDXL_SETTINGS)
    return (*TRANSPORT,)


def execute_excluded_inputs(mode):
    # output_image_proceed only switches an output, like on the Smart Detailer.
    transport = ("opt_log", "VISUAL", "opt_diagnostic", "FACE", "output_image_proceed")
    if mode == "restore":
        return (*transport, *DETAIL_INPUTS)
    return transport


# CMK FaceProcess Prepare Advanced Multi-Model: per FACE slot the suffix _1/_2/_3.
ADVANCED_SLOTS = (1, 2, 3)
ADVANCED_SLOT_INPUTS = ("process_mode", "refine_mode", *RESTORE_INPUTS, "steps", "cfg", "sampling_entry", "sampler",
                        "scheduler", *ZIT_SETTINGS)
ADVANCED_SDXL_SLOT = ("steps", "cfg", "sampling_entry", "sampler", "scheduler")
ADVANCED_SDXL_SHARED = ("stop_at_clip_layer", "pag_scale", "sampling", "zsnr",
                        "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2")
ADVANCED_DIFFUSION_SHARED = ("sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler", *ADVANCED_SDXL_SHARED)
ADVANCED_GLOBAL_ENABLE = "faceprocess_global_enable"


def advanced_slot_mode(inputs, slot):
    """Mode of FACE ``slot`` of the Advanced Prepare; ``None`` while it is a link."""
    if inputs.get(ADVANCED_GLOBAL_ENABLE, "ENABLE") == "DISABLE":
        return "none"
    value = inputs.get(f"process_mode_{slot}")
    return faceprocess_mode(value) if isinstance(value, str) else None


def advanced_prepare_excluded_inputs(mode, slot):
    """Advanced Prepare inputs outside FACE ``slot``'s identity in ``mode``."""
    excluded = [*TRANSPORT, ADVANCED_GLOBAL_ENABLE]
    excluded += [f"{name}_{other}" for other in ADVANCED_SLOTS if other != slot for name in ADVANCED_SLOT_INPUTS]
    own = lambda names: [f"{name}_{slot}" for name in names]
    if mode == "restore":
        return (*excluded, "MODEL ZIT", "MODEL SDXL", "PROCESS", *ADVANCED_DIFFUSION_SHARED,
                *own(ADVANCED_SDXL_SLOT), *own(ZIT_SETTINGS))
    if mode == "sdxl":
        return (*excluded, "MODEL ZIT", *own(RESTORE_INPUTS), *own(ZIT_SETTINGS))
    if mode == "z_image_turbo":
        return (*excluded, "MODEL SDXL", *ADVANCED_SDXL_SHARED, *own(RESTORE_INPUTS), *own(ADVANCED_SDXL_SLOT))
    return tuple(excluded)


def _prepare_link(prompt, unique_id):
    """(prepare id, prepare node, FACE slot) of this branch; the slot is
    ``None`` for the Standard Prepare."""
    node = prompt.get(str(unique_id)) if isinstance(prompt, dict) else None
    link = ((node or {}).get("inputs") or {}).get("FACE") if isinstance(node, dict) else None
    if not (isinstance(link, (list, tuple)) and len(link) == 2):
        return None, None, None
    prepare = prompt.get(str(link[0]))
    class_type = prepare.get("class_type") if isinstance(prepare, dict) else None
    if class_type == PREPARE_CLASS:
        return str(link[0]), prepare, None
    if class_type == ADVANCED_PREPARE_CLASS:
        return str(link[0]), prepare, int(link[1]) + 1
    return None, None, None


def branch_mode(prompt, unique_id):
    """Mode of the Prepare feeding this branch: an id, or ``None`` if unresolvable.

    A Mode Enable control arrives as its literal selection (prompt inline
    handler); a selection that is still a link cannot be resolved statically.
    """
    _prepare_id, prepare, slot = _prepare_link(prompt, unique_id)
    if prepare is None:
        return None
    inputs = prepare.get("inputs") or {}
    if slot is not None:
        return advanced_slot_mode(inputs, slot)
    value = inputs.get("process_mode")
    return faceprocess_mode(value) if isinstance(value, str) else None


class CMKFaceProcessPrepareMultiModelPipe:
    """Deferred FACE preparation for RESTORE, SDXL or Z-Image Turbo (2.8.0).

    Public contract:
        MODEL ZIT (opt) + MODEL SDXL + PROCESS + IMAGE + LOG (+ LoRA ZIT) -> FACE + LOG + diagnostic
    """

    @classmethod
    def INPUT_TYPES(cls):
        # The parameters of CMK Detailer Prepare Multi-Model, unchanged; the
        # operating mode replaces its model family.
        detailer = CMKDetailerPrepareMultiModelPipe.INPUT_TYPES()
        required = {name: detailer["required"][name] for name in ("MODEL SDXL", "PROCESS", "IMAGE")}
        required["process_mode"] = (FACEPROCESS_MODES, {"default": "None"})
        # Refine belongs to the branch configuration of every active mode,
        # the restore model and its visibility to RESTORE.
        required["refine_mode"] = (REFINE_MODES, {"default": "Off"})
        legacy = CMKFaceProcessPipe.INPUT_TYPES()["optional"]
        required["restore_model"] = legacy["restore_model"]
        visibility_type, visibility = legacy["restore_visibility"]
        required["restore_visibility"] = (visibility_type, {**visibility, "default": RESTORE_VISIBILITY_DEFAULT})
        required.update({name: spec for name, spec in detailer["required"].items() if name not in required and name != "model_family"})
        return {"required": required, "optional": dict(detailer["optional"])}

    RETURN_TYPES = (FACE_MULTIMODEL_PIPE, "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("FACE", "LOG", "diagnostic")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Flow/PostProcess"

    def check_lazy_status(self, IMAGE=None, PROCESS=None, process_mode="None", **kwargs):
        needed = []
        if IMAGE is None:
            needed.append("IMAGE")
        # An unconnected optional input is absent from kwargs and never requested.
        if "LOG" in kwargs and kwargs["LOG"] is None:
            needed.append("LOG")
        mode = faceprocess_mode(process_mode)
        if mode in ("sdxl", "z_image_turbo"):
            if PROCESS is None:
                needed.append("PROCESS")
            names = ("MODEL SDXL",) if mode == "sdxl" else ("MODEL ZIT", "LoRA ZIT")
            needed += [name for name in names if name in kwargs and kwargs[name] is None]
        return needed

    def prepare(self, IMAGE, process_mode, refine_mode="Off", restore_model="none",
                restore_visibility=RESTORE_VISIBILITY_DEFAULT, PROCESS=None, LOG=None, **settings):
        if IMAGE is None:
            raise ValueError(f"{PREPARE_LABEL}: IMAGE is missing")
        mode = faceprocess_mode(process_mode)
        face = {
            "face_contract": FACE_CONTRACT,
            "face_deferred": True,
            "face_mode": mode,
            "face_process_mode": str(process_mode),
            "face_image": IMAGE,
            "face_global_enable": mode != "none",
            "boolean_face_enable": mode != "none",
            # NONE is the complete bypass: no Refine either.
            "face_refine_mode": "Off" if mode == "none" else _normalize_refine_mode(refine_mode),
        }
        if mode == "restore":
            face.update(face_restore_model=restore_model, face_restore_visibility=float(restore_visibility))
        model_source = "NOT REQUESTED"
        if mode in ("sdxl", "z_image_turbo"):
            if not isinstance(PROCESS, dict):
                raise TypeError(f"{PREPARE_LABEL}: PROCESS must be a CMK process pipe")
            if PROCESS.get("result_contract") != "family_neutral":
                raise ValueError(f"{PREPARE_LABEL} requires the PostProcess Boundary")
            name = "MODEL SDXL" if mode == "sdxl" else "MODEL ZIT"
            model = settings.pop("MODEL SDXL", None) if mode == "sdxl" else settings.pop("MODEL ZIT", None)
            if not isinstance(model, dict):
                raise ValueError(f"{PREPARE_LABEL}: MODE {MODE_LABELS[mode]} requires {name} from CMK PostProcess Model.")
            settings.pop("MODEL SDXL", None)
            settings.pop("MODEL ZIT", None)
            if mode == "sdxl":
                settings.pop("LoRA ZIT", None)
            face.update({
                "source_pipe": dict(PROCESS),
                "_source_pipe": dict(PROCESS),
                "face_seed": int(PROCESS.get("seed", 0) or 0),
                "face_model_spec": model,
                "face_prepare_settings": dict(settings),
            })
            model_source = f"{name} · DEFERRED (branch MISS)"
        lines = [
            f"STATUS          : {'NONE · BYPASS' if mode == 'none' else 'PREPARED · DEFERRED'}",
            f"MODE            : {MODE_LABELS[mode]}",
            f"MODEL SOURCE    : {model_source}",
            "SAM / LORA      : NOT LOADED",
            "CONDITIONING    : NOT CREATED",
        ]
        log_pipe = cmk_add_block(LOG, "FaceProcess Prepare", 80, lines, True)
        diagnostic = make_diagnostic_payload(
            title="FaceProcess Prepare",
            node=PREPARE_LABEL,
            previews=[IMAGE],
            summary=f"mode={MODE_LABELS[mode]} | deferred, nothing loaded",
            details="\n".join(lines),
            mode=MODE_LABELS[mode].lower(),
            metadata={"mode": mode, "model_requested": mode in ("sdxl", "z_image_turbo"),
                      "models_loaded": False, "sam_loaded": False, "conditioning_created": False},
        )
        return (face, log_pipe, diagnostic)


class CMKFaceProcessMultiModelPipe(CMKFaceProcessPipe):
    """FaceProcess execution for CMK FaceProcess Prepare Multi-Model (2.8.0).

    The operating mode comes from the Prepare; this node has no own enable or
    mode widget.  Cache entries use their own schema and never meet the legacy
    CMK FaceProcess -Pipe- entries.
    """

    # v2 (F3): YOLO-led RESTORE with assigned InsightFace faces, resolved backends.
    # v3 (F4): processing layers in SEGS PROCESSED; Refine only on the branch's faces.
    # v4: Multi-Model Detailer parameters (sampling entry per family, LoRA ZIT).
    _CACHE_SCHEMA = "cmk_faceprocess_multimodel_branch_v4"
    # The legacy outputs keep their positions; FACE IMAGE is appended. Like
    # DETAILER IMAGE of the Smart Detailer it always carries the processed
    # image, independent of OUTPUT IMAGE PROCEED, and feeds the boundary.
    RETURN_TYPES = (*CMKFaceProcessPipe.RETURN_TYPES, "IMAGE")
    RETURN_NAMES = (*CMKFaceProcessPipe.RETURN_NAMES, "FACE IMAGE")
    _ZIT_TEXT_ENCODER = "released after FaceProcess Prepare"

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        rest = {name: value for name, value in spec["required"].items() if name not in ("enable", "process_mode", "FACE")}
        # OUTPUT IMAGE PROCEED as on CMK Smart Detailer -Pipe-: off leaves the
        # IMAGE PROCEED output empty (SEGS PROCESSED carries the result).
        required = {"FACE": (FACE_MULTIMODEL_PIPE, {"lazy": True}),
                    "output_image_proceed": ("BOOLEAN", {"default": True}), **rest}
        optional = {name: value for name, value in spec["optional"].items() if name not in DROPPED_EXECUTE_INPUTS}
        return {**spec, "required": required, "optional": optional}

    # ---- cache identity -------------------------------------------------
    @classmethod
    def dependency_cache_key(cls, prompt, unique_id):
        mode = branch_mode(prompt, unique_id)
        prepare_id, prepare, slot = _prepare_link(prompt, unique_id)
        suffix = "" if slot is None else f"_{slot}"
        extra = {"mode": mode or "unknown"}
        if mode in ("restore", None):
            restore_model = (prepare.get("inputs") or {}).get(f"restore_model{suffix}") if prepare is not None else None
            extra.update(match_rule=MATCH_RULE, restore_backend=backend_identity(restore_model))
        if prepare is not None and slot is None:
            extra["prepare"], _detail = build_node_fingerprint(
                prompt,
                prepare_id,
                (PREPARE_CLASS,),
                "cmk_faceprocess_multimodel_prepare_v1",
                exclude_inputs=prepare_excluded_inputs(mode),
            )
        elif prepare is not None:
            # Only this FACE slot's settings and the shared ones its mode reads.
            extra["prepare"], _detail = build_node_fingerprint(
                prompt,
                prepare_id,
                (ADVANCED_PREPARE_CLASS,),
                "cmk_faceprocess_advanced_prepare_v1",
                exclude_inputs=advanced_prepare_excluded_inputs(mode, slot),
            )
        return build_node_fingerprint(
            prompt,
            unique_id,
            (EXECUTE_CLASS,),
            cls._CACHE_SCHEMA,
            exclude_inputs=execute_excluded_inputs(mode) if prepare is not None else ("opt_log", "VISUAL", "opt_diagnostic"),
            include_node_identity=True,
            extra_identity=extra,
        )

    @classmethod
    def dependency_revision(cls, prompt, unique_id, cache_key):
        # NONE never runs and stores nothing, like a switched-off branch.
        if branch_mode(prompt, unique_id) == "none":
            return DISABLED_REVISION
        return super().dependency_revision(prompt, unique_id, cache_key)

    def check_lazy_status(self, FACE=None, prompt=None, unique_id=None, **kwargs):
        if branch_mode(prompt, unique_id) == "none":
            # The bypass only needs the image of the (model-free) FACE spec.
            return ["FACE"] if FACE is None else []
        return super().check_lazy_status(FACE=FACE, prompt=prompt, unique_id=unique_id, **kwargs)

    # ---- execution ------------------------------------------------------
    def run_pipe(self, FACE=None, output_image_proceed=True, select_face="Largest", opt_log=None, VISUAL=None,
                 opt_diagnostic=None, prompt=None, unique_id=None, **kwargs):
        return self._image_output(
            self._run_branch(FACE, select_face, opt_log, VISUAL, opt_diagnostic, prompt, unique_id, kwargs),
            output_image_proceed,
        )

    @staticmethod
    def _image_output(result, output_image_proceed):
        # Like the Smart Detailer: without OUTPUT IMAGE PROCEED the IMAGE PROCEED
        # output stays empty; SEGS, VISUAL and diagnostic are unchanged. FACE
        # IMAGE (like DETAILER IMAGE) always carries the processed image, so the
        # boundary can store a MISS whatever the switch says.
        image = result[0]
        return (image if bool(output_image_proceed) else None, *result[1:], image)

    @staticmethod
    def _merge_transport(image_proceed, selected_face, segs_processed, enabled, log_block, diagnostic, *, before, **kwargs):
        # A branch-cache entry written before the spatial stages gets them on
        # its HIT, from its cached SEGS and layers; nothing is processed again.
        if bool(enabled) and isinstance(diagnostic, dict):
            restore = (diagnostic.get("metadata") or {}).get("primary_process") == "Restore"
            diagnostic = with_face_stages(diagnostic, before, "restore" if restore else "detailer", selected_face,
                                          getattr(segs_processed, "cmk_layers", []))
        return CMKFaceProcessPipe._merge_transport(image_proceed, selected_face, segs_processed, enabled, log_block,
                                                   diagnostic, before=before, **kwargs)

    def _run_branch(self, FACE, select_face, opt_log, VISUAL, opt_diagnostic, prompt, unique_id, kwargs):
        from .cmk_persistent_cache import pickle_available

        mode = branch_mode(prompt, unique_id)
        if mode is None and isinstance(FACE, dict):
            mode = FACE.get("face_mode")
        if mode == "none":
            return self._bypass(FACE, VISUAL, unique_id, opt_log, opt_diagnostic)
        transport = dict(select_face=select_face, opt_log=opt_log, VISUAL=VISUAL, opt_diagnostic=opt_diagnostic,
                         prompt=prompt, unique_id=unique_id, **kwargs)
        cache_key, _detail = self.dependency_cache_key(prompt, unique_id)
        if cache_key and pickle_available(self._CACHE_SCOPE, cache_key):
            # Branch HIT: the legacy path loads the payload; FACE is not read.
            return super().run_pipe(FACE=None, enable=True, process_mode="restore", **transport)
        face_pipe = self._materialize_face(FACE, prompt=prompt)
        params = {**kwargs, "refine_mode": face_pipe.get("face_refine_mode", "Off")}
        run = self._run_restore if face_pipe.get("face_mode") == "restore" else self._run_detailer
        return run(face_pipe, select_face, params, opt_log=opt_log, VISUAL=VISUAL, opt_diagnostic=opt_diagnostic,
                   unique_id=unique_id, cache_key=cache_key, cache_detail=_detail)

    # ---- shared ---------------------------------------------------------
    def _detection(self, params):
        detect_model = params.get("detect_model")
        values = {name: params.get(f"detect_{name}") for name in ("bbox_threshold", "bbox_dilation", "crop_factor", "drop_size")}
        defaults = {"bbox_threshold": 0.5, "bbox_dilation": 0, "crop_factor": 3.0, "drop_size": 10}
        values = {name: defaults[name] if value is None else value for name, value in values.items()}
        return detect_model, values

    def _yolo(self, image, detect_model, detect, sam_model=None):
        return self._detect_segs(image=image, detect_model=detect_model, detect_bbox_threshold=detect["bbox_threshold"],
                                 detect_bbox_dilation=detect["bbox_dilation"], detect_crop_factor=detect["crop_factor"],
                                 detect_drop_size=detect["drop_size"], opt_sam_model=sam_model)

    def _refine(self, image, segs, refine_mode):
        """Refine of the 2.7 path (fixed strength, alpha and padding) on ``segs``
        only, as image and layer.  No targets: no refine and no quantisation."""
        refine_mode = _normalize_refine_mode(refine_mode)
        if refine_mode == "Off":
            return image, [], "refine off"
        if not self._seg_count(segs):
            return image, [], f"refine skipped | no selected face | mode: {refine_mode}"
        arr = image.detach().cpu().numpy()
        out, layers, changes = [], [], []
        for batch_index in range(arr.shape[0]):
            rgb = np.clip(arr[batch_index].astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
            height, width = rgb.shape[:2]
            mask = np.zeros((height, width), dtype=np.float32)
            for seg in segs[1]:
                seg_mask = self._mask_from_seg(height, width, seg, padding=0.25)
                if seg_mask is not None:
                    mask = np.maximum(mask, seg_mask)
            refined = self._refine_rgb(rgb, refine_mode, strength=0.65)
            alpha = np.clip(mask, 0.0, 1.0) * 0.75
            result = np.clip(rgb.astype(np.float32) * (1.0 - alpha[..., None]) + refined.astype(np.float32) * alpha[..., None],
                             0, 255).astype(np.uint8)
            changes.append(float(np.mean(np.abs(result.astype(np.float32) - rgb.astype(np.float32))) / 255.0))
            out.append(torch.from_numpy(result.astype(np.float32) / 255.0))
            layer = refine_layer(batch_index, refined, alpha)
            if layer is not None:
                layers.append(layer)
        change = float(np.mean(changes)) if changes else 0.0
        return torch.stack(out, dim=0), layers, f"refine applied | mode: {refine_mode} | changed: {change:.4f}"

    def _finish(self, *, mode_label, image, primary, primary_layers, quantize, refine_targets, coverage, full_crop_mask,
                select_face, params, log_lines, diagnostic_args, opt_log, VISUAL, opt_diagnostic, unique_id, cache_key,
                cache_detail, selected=None):
        refine_mode = params.get("refine_mode", "Off")
        processed, refine_layers, refine_info = self._refine(primary, refine_targets, refine_mode)
        layers = list(primary_layers)
        if refine_layers and quantize == "trunc":
            # The refine input conversion of 2.7 (truncating) over the branch region.
            layers.append(quantize_layer("trunc", layers + refine_layers))
        layers += refine_layers
        if quantize == "round" and layers:
            layers.insert(0, quantize_layer("round", layers))

        segs_processed = (segs_with_image_crops(coverage, processed, full_crop_mask=full_crop_mask)
                          if coverage[1] else self._empty_segs(processed))
        segs_processed = with_layers(make_stable_segs(segs_processed, image, processed, coverage_segs=coverage), layers)
        log_lines = [*log_lines, f"REFINE          : {refine_info}",
                     f"LAYERS          : {', '.join(layer['kind'] for layer in layers) or 'none'}"]
        log_block = cmk_block_to_string("FaceProcess", 90, log_lines, True)
        roi = layers_roi(layers)
        diagnostic = self._make_diagnostic(
            mode=mode_label, refine_mode=refine_mode, enabled=True, image=image, primary_image=primary,
            processed_image=processed, segs_processed=segs_processed, detect_model=params.get("detect_model"),
            **diagnostic_args,
        )
        diagnostic["metadata"]["layers"] = {"schema": LAYER_SCHEMA, "kinds": [layer["kind"] for layer in layers],
                                            "roi": list(roi) if roi else None}
        restore = mode_label == "Restore"
        records = diagnostic_args.get("metadata", {}).get("faces") if restore else None
        diagnostic = with_face_stages(
            diagnostic, image, "restore" if restore else "detailer", coverage, layers,
            detection=detection_view(image, diagnostic_args.get("segs_detected"),
                                     coverage if selected is None else selected, records),
        )
        payload = {"before_image": image, "image_proceed": processed, "selected_face": coverage,
                   "segs_processed": segs_processed, "enabled": True, "diagnostic": diagnostic, "log_block": log_block}
        if cache_key:
            try:
                save_pickle(self._CACHE_SCOPE, cache_key, payload, max_entries=32)
                write_status(self._CACHE_SCOPE, "MISS_STORED", cache_key=cache_key, detail=cache_detail, unique_id=unique_id)
                print(f"[CMK FaceProcess -Pipe-] CACHE MISS {cache_key[:12]} -> STORED")
            except Exception as exc:
                write_status(self._CACHE_SCOPE, "STORE_FAILED", cache_key=cache_key, detail=str(exc), unique_id=unique_id)
                print(f"[CMK FaceProcess -Pipe-] CACHE STORE FAILED: {exc}")
        return self._merge_transport(processed, coverage, segs_processed, True, log_block, diagnostic, before=image,
                                     visual=VISUAL, unique_id=unique_id, opt_log=opt_log, opt_diagnostic=opt_diagnostic)

    # ---- RESTORE (F3, layers F4) -----------------------------------------
    @staticmethod
    def _insightface_faces(rgb):
        """Complete InsightFace face objects (bbox, kps, 106 landmarks) of one image."""
        return list(get_face_analyser("buffalo_l", 640).get(rgb))

    @staticmethod
    def _face_area(face):
        x1, y1, x2, y2 = [float(v) for v in face.bbox[:4]]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _assign(self, image, selected):
        """Per batch item: assignment records and the assigned face objects."""
        records, assigned = [], []
        for batch_index in range(int(image.shape[0])):
            items = [seg for seg in selected[1] if getattr(seg, "batch_index", None) in (None, batch_index)]
            faces = self._insightface_faces(tensor_to_uint8_rgb(image[batch_index])) if items else []
            rows = assign_faces([seg.bbox for seg in items], faces)
            kept = []
            for seg, row in zip(items, rows):
                records.append({**row, "batch_index": batch_index, "yolo_bbox": [float(v) for v in seg.bbox],
                                "landmarks_106": row["status"] == OK and getattr(faces[row["face_index"]], "landmark_2d_106", None) is not None})
                if row["status"] == OK:
                    kept.append((seg, faces[row["face_index"]]))
            assigned.append(kept)
        return records, assigned

    def _run_restore(self, face_pipe, select_face, params, **transport):
        image = face_pipe["face_image"]
        detect_model, detect = self._detection(params)
        if detect_model in (None, "", "none"):
            raise RestoreConfigurationError(f"{EXECUTE_LABEL}: RESTORE requires a YOLO face detector (detect_model)")
        backend, backend_info = resolve_restore_backend(face_pipe.get("face_restore_model"))
        visibility = face_pipe.get("face_restore_visibility", RESTORE_VISIBILITY_DEFAULT)

        # RESTORE uses the YOLO boxes only; a SAM mask would be replaced by the
        # full crop mask of the coverage, so SAM is not loaded.
        detected = self._yolo(image, detect_model, detect)
        selected = self._select_output_segs(detected, image, select_face) if self._seg_count(detected) else detected
        records, assigned = self._assign(image, selected)
        primary, captured = image, []
        if any(assigned):
            # Restore order of 2.7 (ascending face area): faces are pasted one
            # after another and GFPGAN draws noise per face.
            order = [sorted((face for _seg, face in kept), key=self._face_area) for kept in assigned]
            primary = RestoreFaceAdvanced().execute(
                image=image, model=backend_info["file"], visibility=visibility, assigned_faces=order, enhancer=backend,
                layers=captured,
            )[0]
        primary_layers = [layer for layer in (restore_layer(item["batch_index"], item["warped"], item["alpha"], item["visibility"])
                                              for item in captured) if layer is not None]
        restored_segs = (detected[0], [seg for kept in assigned for seg, _face in kept])

        restored = sum(1 for row in records if row["status"] == OK)
        face_lines = [f"FACE {index:<11}: {STATUS_TEXT[row['status']]} · IoU {row['iou']:.2f} · kps {row['keypoints']}/5"
                      for index, row in enumerate(records, 1)]
        if not self._seg_count(detected):
            face_lines = ["FACES           : NOT DETECTED BY YOLO"]
        warnings = [f"face {index}: {STATUS_TEXT[row['status']].lower()}" for index, row in enumerate(records, 1)
                    if row["status"] != OK]
        if not self._seg_count(detected):
            warnings.append("no face detected by YOLO")
        return self._finish(
            mode_label="Restore", image=image, primary=primary, primary_layers=primary_layers, quantize="round",
            # Refine only on assigned and restored faces (E9).
            refine_targets=restored_segs, coverage=restored_segs, full_crop_mask=True, select_face=select_face,
            params=params, selected=selected,
            log_lines=["STATUS          : EXECUTED", "PROCESS MODE    : RESTORE", f"SELECTION       : {select_face} (YOLO)",
                       f"RESTORE BACKEND : {backend_info['backend']} · {Path(backend_info['file']).name}",
                       f"FACES DETECTED  : {self._seg_count(detected)}", f"FACES SELECTED  : {self._seg_count(selected)}",
                       *face_lines, f"FACES RESTORED  : {restored}"],
            diagnostic_args={
                "status": "Processed" if restored else ("No face detected" if not self._seg_count(detected) else "No face assigned"),
                "segs_detected": detected, "restore_model": Path(backend_info["file"]).name, "restore_visibility": visibility,
                "metadata": {"selection": select_face, "match_rule": MATCH_RULE, "faces": records,
                             "restore_backend": {**backend_info, "file": Path(backend_info["file"]).name},
                             "faces_restored": restored, "faces_skipped": len(records) - restored},
                "warnings": warnings},
            **transport)

    # ---- SDXL / ZIT detailer (F4: the 2.7 detailer step with layers) -------
    def _run_detailer(self, face_pipe, select_face, params, **transport):
        image = face_pipe["face_image"]
        detect_model, detect = self._detection(params)
        if detect_model in (None, "", "none"):
            raise ValueError(f"{EXECUTE_LABEL}: the detailer requires a YOLO face detector (detect_model)")
        sam_model = face_pipe.get("face_sam_model")
        used_detector = detect_model
        detected = self._yolo(image, detect_model, detect, sam_model)
        # 2.7: a non-face default detector that finds nothing is retried once
        # with the most face-like detector installed.
        fallback = _preferred_face_detector(_sort_face_detectors(
            ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
            + ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]))
        if (not self._seg_count(detected) and fallback and fallback != detect_model
                and _is_likely_non_face_detector(detect_model)):
            retry = self._yolo(image, fallback, detect, sam_model)
            if self._seg_count(retry):
                detected, used_detector = retry, fallback
        selected = self._select_output_segs(detected, image, select_face) if self._seg_count(detected) else detected

        guide_size, max_size = params.get("detail_guide_size", 512), params.get("detail_max_size", 768)
        # Sampling entry of the family, converted exactly like CMK Smart Detailer Multi-Model.
        entry = face_pipe.get("face_sampling_entry")
        entry = 80.0 if entry is None else float(entry)
        denoise = detailer_entry_to_denoise(entry)
        feather = min(200, max(0, int(params.get("detail_paste_feather", 20))))
        primary, primary_layers = image, []
        steps, cfg = face_pipe.get("face_steps", 20), face_pipe.get("face_cfg", 8.0)
        if self._seg_count(selected):
            basic_pipe = (face_pipe.get("face_model"), face_pipe.get("face_clip"), face_pipe.get("face_vae"),
                          face_pipe.get("face_conditioning_pos"), face_pipe.get("face_conditioning_neg"))
            detailed = self._find_valid_segs(SEGSDetailer().doit(
                image=image, segs=selected, guide_size=guide_size, guide_size_for=params.get("detail_guide_size_for", True),
                max_size=max_size, seed=int(face_pipe.get("face_seed", 0) or 0), steps=steps, cfg=cfg,
                sampler_name=face_pipe.get("face_sampler"), scheduler=face_pipe.get("face_scheduler"), denoise=denoise,
                noise_mask=params.get("detail_noise_mask", True), force_inpaint=params.get("detail_force_inpaint", True),
                basic_pipe=basic_pipe, refiner_ratio=0.2, batch_size=1, cycle=1, refiner_basic_pipe_opt=None,
                inpaint_model=False, noise_mask_feather=20, scheduler_func_opt=None,
            ))
            if detailed is not None:
                primary = self._find_image_tensor(SEGSPaste.doit(image, detailed, feather, alpha=255))
                primary_layers = paste_layers(image, detailed, feather)
        mode_label = "Detailer"
        family = MODE_LABELS.get(face_pipe.get("face_mode"), "SDXL")
        return self._finish(
            mode_label=mode_label, image=image, primary=primary, primary_layers=primary_layers, quantize="trunc",
            refine_targets=selected, coverage=selected, full_crop_mask=False, select_face=select_face, params=params,
            log_lines=["STATUS          : EXECUTED", f"PROCESS MODE    : {family} DETAILER",
                       f"SELECTION       : {select_face} (YOLO)", f"DETECTOR        : {used_detector}",
                       f"FACES DETECTED  : {self._seg_count(detected)}", f"FACES SELECTED  : {self._seg_count(selected)}",
                       f"SAMPLING ENTRY  : {entry:g} %", f"DENOISE         : {denoise:g}",
                       f"STEPS / CFG     : {steps} / {cfg}", f"PASTE FEATHER   : {feather}",
                       f"PROMPT SOURCE   : {'1ST PASS' if face_pipe.get('face_use_prompt_from_1st_pass') else 'LOCAL'}",
                       f"LORAS           : {face_pipe.get('face_active_loras') or 'none'}"],
            diagnostic_args={"status": "Processed" if self._seg_count(selected) else "No face detected",
                             "segs_detected": detected, "steps": steps, "cfg": cfg, "denoise": denoise,
                             "force_inpaint": params.get("detail_force_inpaint", True), "guide_size": guide_size,
                             "max_size": max_size, "paste_feather": feather,
                             "metadata": {"selection": select_face, "detector": used_detector, "family": family,
                                          "sampling_entry": entry},
                             "warnings": [] if self._seg_count(detected) else ["no face detected by YOLO"]},
            **transport)

    def _bypass(self, FACE, VISUAL, unique_id, opt_log, opt_diagnostic):
        image = FACE.get("face_image") if isinstance(FACE, dict) else None
        if image is None:
            raise ValueError(f"{EXECUTE_LABEL}: FACE carries no image")
        empty = self._empty_segs(image)
        lines = ["STATUS          : NONE · BYPASS", "FACE DETECTION  : SKIPPED", "PROCESSING      : SKIPPED",
                 "REFINE          : SKIPPED", "CACHE           : SKIPPED", "RESULT          : PASSTHROUGH"]
        diagnostic = make_diagnostic_payload(
            title="FaceProcess -Pipe-", node=EXECUTE_LABEL, previews=[image], summary="NONE · bypass",
            details="\n".join(lines), mode="none / passthrough", metadata={"mode": "none"},
        )
        return self._merge_transport(
            image, empty, empty, False, cmk_block_to_string("FaceProcess", 90, lines, True), diagnostic,
            before=image, visual=VISUAL, unique_id=unique_id, opt_log=opt_log, opt_diagnostic=opt_diagnostic,
        )

    def _materialize_face(self, face, prompt=None):
        """FACE spec -> FACE pipe of the legacy execution, on a branch MISS only."""
        if not isinstance(face, dict) or face.get("face_contract") != FACE_CONTRACT:
            raise ValueError(f"{EXECUTE_LABEL}: FACE must come from {PREPARE_LABEL}")
        mode = face.get("face_mode")
        image = face.get("face_image")
        if mode == "restore":
            # Restore needs no diffusion model, conditioning or SAM.
            return {"face_mode": "restore", "face_image": image, "face_global_enable": True,
                    "boolean_face_enable": True, "face_sam_model": None, "face_seed": 0,
                    "face_refine_mode": face.get("face_refine_mode", "Off"),
                    "face_restore_model": face.get("face_restore_model"),
                    "face_restore_visibility": face.get("face_restore_visibility", RESTORE_VISIBILITY_DEFAULT)}
        if mode not in ("sdxl", "z_image_turbo"):
            raise ValueError(f"{EXECUTE_LABEL}: unsupported FACE mode {mode!r}")
        # The Multi-Model Detailer preparation, exactly as its Prepare runs it
        # (SDXL: guarded ZIT release first; ZIT: LoRA ZIT / ZIT first pass).
        arguments = dict(face.get("face_prepare_settings") or {})
        arguments["MODEL SDXL" if mode == "sdxl" else "MODEL ZIT"] = face.get("face_model_spec")
        detailer = CMKDetailerPrepareMultiModelPipe().prepare(
            PROCESS=face.get("source_pipe"), IMAGE=image,
            model_family="SDXL" if mode == "sdxl" else "Z-Image Turbo", **arguments,
        )[0]
        return {
            "face_mode": mode,
            "face_refine_mode": face.get("face_refine_mode", "Off"),
            "source_pipe": detailer.get("source_pipe"),
            "_source_pipe": detailer.get("source_pipe"),
            "face_model": detailer.get("detailer_model"),
            # The sampling engine never reads CLIP; ZIT releases its text
            # encoder after encoding, the BASIC_PIPE check needs a value.
            "face_clip": detailer.get("detailer_clip") or self._ZIT_TEXT_ENCODER,
            "face_vae": detailer.get("detailer_vae"),
            "face_conditioning_pos": detailer.get("detailer_conditioning_pos"),
            "face_conditioning_neg": detailer.get("detailer_conditioning_neg"),
            "face_image": image,
            "face_seed": detailer.get("detailer_seed", 0),
            "face_global_enable": True,
            "boolean_face_enable": True,
            "face_steps": detailer.get("detailer_steps"),
            "face_cfg": detailer.get("detailer_cfg"),
            "face_sampler": detailer.get("detailer_sampler"),
            "face_scheduler": detailer.get("detailer_scheduler"),
            "face_sampling_entry": detailer.get("detailer_sampling_entry"),
            "face_sam_model": detailer.get("detailer_sam_model"),
            "face_prompt_pos": detailer.get("detailer_prompt_pos"),
            "face_active_loras": detailer.get("detailer_active_loras"),
            "face_use_prompt_from_1st_pass": detailer.get("detailer_use_prompt_from_1st_pass"),
            "face_use_lora_from_1st_pass": detailer.get("detailer_use_lora_from_1st_pass"),
            "face_model_identity": detailer.get("detailer_model_identity"),
            "face_zit_model_shift": detailer.get("detailer_zit_model_shift"),
        }

