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
SDXL runs the unchanged legacy FaceProcess preparation (after the guarded ZIT
release); ZIT runs the Z-Image Turbo preparation of CMK Detailer Prepare
Multi-Model; both then use the unchanged legacy detailer path.  RESTORE (F3)
needs no diffusion model or SAM: YOLO selects the faces, InsightFace assigns
its complete face objects (``utils.face_matching``) and the resolved restore
backend restores exactly those (``cmk_faceprocess_restore``).  Models exist
only while the branch runs; no output holds them.

Legacy CMK FaceProcess Prepare -Pipe- / CMK FaceProcess -Pipe- are unchanged.
"""

from __future__ import annotations

from pathlib import Path

from .cmk_detailer_prepare_multimodel import CMKDetailerPrepareMultiModelPipe, ZIT_SAMPLING_ENTRY
from .cmk_faceprocess import CMKFaceProcessPipe
from .cmk_faceprocess_mode import FACEPROCESS_MODES, faceprocess_mode
from .cmk_faceprocess_prepare import CMKFaceProcessPreparePipe
from .cmk_log_pipe import cmk_add_block, cmk_block_to_string
from .cmk_faceprocess_restore import RestoreConfigurationError, backend_identity, resolve_restore_backend
from .cmk_persistent_cache import DISABLED_REVISION, build_node_fingerprint, save_pickle, write_status
from ..engine.detector_engine import get_face_analyser
from ..engine.native_face_restore import RestoreFaceAdvanced
from ..utils.cmk_diagnostic import make_diagnostic_payload
from ..utils.face_matching import MATCH_RULE, OK, assign_faces
from ..utils.stable_segs import make_stable_segs, segs_with_image_crops
from ..utils.tensor_utils import tensor_to_uint8_rgb

PREPARE_CLASS = "CMKFaceProcessPrepareMultiModelPipe"
EXECUTE_CLASS = "CMKFaceProcessMultiModelPipe"
PREPARE_LABEL = "CMK FaceProcess Prepare Multi-Model -Pipe-"
EXECUTE_LABEL = "CMK FaceProcess Multi-Model -Pipe-"
FACE_MULTIMODEL_PIPE = "CMK_FACE_MULTIMODEL_PIPE"
FACE_CONTRACT = "faceprocess_multimodel_v1"
MODE_LABELS = {"none": "NONE", "restore": "RESTORE", "sdxl": "SDXL", "z_image_turbo": "ZIT"}

# Prepare inputs by the mode that reads them.
SDXL_SETTINGS = (
    "use_1st_pass_sampling", "lora_name", "strength_model", "strength_clip", "prompt_neg",
    "steps", "cfg", "sampler", "scheduler", "stop_at_clip_layer", "pag_scale", "sampling",
    "zsnr", "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2",
)
ZIT_SETTINGS = ("zit_steps", "zit_cfg", "zit_sampler", "zit_scheduler", "zit_model_shift")
DIFFUSION_SHARED = ("sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler",
                    "use_lora_from_1st_pass", "prompt_pos")
TRANSPORT = ("LOG",)

# Execute inputs by the path that reads them.
RESTORE_INPUTS = ("restore_model", "restore_visibility")
# Restore controls without effect (E5): not offered by the 2.8 node.
DROPPED_RESTORE_INPUTS = ("restore_facedetection", "restore_codeformer_weight")
STATUS_TEXT = {"OK": "RESTORED", "UNMATCHED": "SKIPPED · NO INSIGHTFACE ASSIGNMENT",
               "AMBIGUOUS": "SKIPPED · AMBIGUOUS ASSIGNMENT", "CONFLICT": "SKIPPED · AMBIGUOUS ASSIGNMENT (CONFLICT)"}
DETAIL_INPUTS = ("detail_guide_size", "detail_guide_size_for", "detail_max_size", "detail_denoise",
                 "detail_noise_mask", "detail_force_inpaint", "detail_paste_feather")


def prepare_excluded_inputs(mode):
    """Prepare inputs outside a mode's identity (never part of its branch key)."""
    if mode == "restore":
        return (*TRANSPORT, "MODEL ZIT", "MODEL SDXL", "PROCESS", *DIFFUSION_SHARED, *SDXL_SETTINGS, *ZIT_SETTINGS)
    if mode == "sdxl":
        return (*TRANSPORT, "MODEL ZIT", *ZIT_SETTINGS)
    if mode == "z_image_turbo":
        return (*TRANSPORT, "MODEL SDXL", *SDXL_SETTINGS)
    return (*TRANSPORT,)


def execute_excluded_inputs(mode):
    transport = ("opt_log", "VISUAL", "opt_diagnostic", "FACE")
    if mode == "restore":
        return (*transport, *DETAIL_INPUTS)
    if mode in ("sdxl", "z_image_turbo"):
        return (*transport, *RESTORE_INPUTS)
    return transport


def _prepare_link(prompt, unique_id):
    node = prompt.get(str(unique_id)) if isinstance(prompt, dict) else None
    link = ((node or {}).get("inputs") or {}).get("FACE") if isinstance(node, dict) else None
    if not (isinstance(link, (list, tuple)) and len(link) == 2):
        return None, None
    prepare = prompt.get(str(link[0]))
    if not isinstance(prepare, dict) or prepare.get("class_type") != PREPARE_CLASS:
        return None, None
    return str(link[0]), prepare


def branch_mode(prompt, unique_id):
    """Mode of the Prepare feeding this branch: an id, or ``None`` if unresolvable.

    A Mode Enable control arrives as its literal selection (prompt inline
    handler); a selection that is still a link cannot be resolved statically.
    """
    _prepare_id, prepare = _prepare_link(prompt, unique_id)
    if prepare is None:
        return None
    value = (prepare.get("inputs") or {}).get("process_mode")
    return faceprocess_mode(value) if isinstance(value, str) else None


class CMKFaceProcessPrepareMultiModelPipe:
    """Deferred FACE preparation for RESTORE, SDXL or Z-Image Turbo (2.8.0).

    Public contract:
        MODEL ZIT (opt) + MODEL SDXL + PROCESS + IMAGE + LOG -> FACE + LOG + diagnostic
    """

    @classmethod
    def INPUT_TYPES(cls):
        legacy = CMKFaceProcessPreparePipe.INPUT_TYPES()["required"]
        detailer = CMKDetailerPrepareMultiModelPipe.INPUT_TYPES()["required"]
        required = {
            "MODEL SDXL": legacy["MODEL"],
            "PROCESS": legacy["PROCESS"],
            "IMAGE": legacy["IMAGE"],
            "process_mode": (FACEPROCESS_MODES, {"default": "None"}),
        }
        required.update({name: legacy[name] for name in (
            "sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler", "use_lora_from_1st_pass",
            "use_1st_pass_sampling", "lora_name", "strength_model", "strength_clip", "prompt_pos", "prompt_neg",
            "steps", "cfg", "sampler", "scheduler",
        )})
        required.update({name: detailer[name] for name in ZIT_SETTINGS})
        required.update({name: legacy[name] for name in (
            "stop_at_clip_layer", "pag_scale", "sampling", "zsnr",
            "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2",
        )})
        return {
            "required": required,
            "optional": {
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "MODEL ZIT": ("CMK_MODEL_PIPE", {"lazy": True}),
            },
        }

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
            name = "MODEL SDXL" if mode == "sdxl" else "MODEL ZIT"
            if name in kwargs and kwargs[name] is None:
                needed.append(name)
        return needed

    def prepare(self, IMAGE, process_mode, PROCESS=None, LOG=None, **settings):
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
        }
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
    _CACHE_SCHEMA = "cmk_faceprocess_multimodel_branch_v2"
    _ZIT_TEXT_ENCODER = "released after FaceProcess Prepare"

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        required = {name: value for name, value in spec["required"].items() if name not in ("enable", "process_mode")}
        required["FACE"] = (FACE_MULTIMODEL_PIPE, {"lazy": True})
        optional = {name: value for name, value in spec["optional"].items() if name not in DROPPED_RESTORE_INPUTS}
        return {**spec, "required": required, "optional": optional}

    # ---- cache identity -------------------------------------------------
    @classmethod
    def dependency_cache_key(cls, prompt, unique_id):
        mode = branch_mode(prompt, unique_id)
        prepare_id, prepare = _prepare_link(prompt, unique_id)
        extra = {"mode": mode or "unknown"}
        if mode in ("restore", None):
            node = prompt.get(str(unique_id)) if isinstance(prompt, dict) else None
            restore_model = ((node or {}).get("inputs") or {}).get("restore_model") if isinstance(node, dict) else None
            extra.update(match_rule=MATCH_RULE, restore_backend=backend_identity(restore_model))
        if prepare is not None:
            extra["prepare"], _detail = build_node_fingerprint(
                prompt,
                prepare_id,
                (PREPARE_CLASS,),
                "cmk_faceprocess_multimodel_prepare_v1",
                exclude_inputs=prepare_excluded_inputs(mode),
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
    def run_pipe(self, FACE=None, select_face="Largest", opt_log=None, VISUAL=None, opt_diagnostic=None,
                 prompt=None, unique_id=None, **kwargs):
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
        if face_pipe.get("face_mode") == "restore":
            return self._run_restore(face_pipe, select_face, kwargs, opt_log=opt_log, VISUAL=VISUAL,
                                     opt_diagnostic=opt_diagnostic, unique_id=unique_id, cache_key=cache_key,
                                     cache_detail=_detail)
        return super().run_pipe(FACE=face_pipe, enable=True, process_mode="detailer", **transport)

    # ---- RESTORE (F3) ---------------------------------------------------
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

    def _run_restore(self, face_pipe, select_face, params, *, opt_log, VISUAL, opt_diagnostic, unique_id,
                     cache_key, cache_detail):
        image = face_pipe["face_image"]
        refine_mode = params.get("refine_mode", "Off")
        detect_model = params.get("detect_model")
        if detect_model in (None, "", "none"):
            raise RestoreConfigurationError(f"{EXECUTE_LABEL}: RESTORE requires a YOLO face detector (detect_model)")
        backend, backend_info = resolve_restore_backend(params.get("restore_model"))
        visibility = params.get("restore_visibility", 1.0)
        detect = {name: params.get(f"detect_{name}") for name in ("bbox_threshold", "bbox_dilation", "crop_factor", "drop_size")}
        defaults = {"bbox_threshold": 0.5, "bbox_dilation": 0, "crop_factor": 3.0, "drop_size": 10}
        detect = {name: defaults[name] if value is None else value for name, value in detect.items()}

        def yolo(target):
            # RESTORE uses the YOLO boxes only; a SAM mask would be replaced by
            # the full crop mask of the coverage, so SAM is not loaded.
            return self._detect_segs(image=target, detect_model=detect_model, detect_bbox_threshold=detect["bbox_threshold"],
                                     detect_bbox_dilation=detect["bbox_dilation"], detect_crop_factor=detect["crop_factor"],
                                     detect_drop_size=detect["drop_size"], opt_sam_model=None)

        detected = yolo(image)
        selected = self._select_output_segs(detected, image, select_face) if self._seg_count(detected) else detected
        records, assigned = self._assign(image, selected)
        primary = image
        if any(assigned):
            # Restore order of 2.7 (ascending face area): faces are pasted one
            # after another and GFPGAN draws noise per face.
            order = [sorted((face for _seg, face in kept), key=self._face_area) for kept in assigned]
            primary = RestoreFaceAdvanced().execute(
                image=image, model=backend_info["file"], visibility=visibility, assigned_faces=order, enhancer=backend,
            )[0]

        # Refine keeps its 2.7 targets until F4 (all YOLO faces; a new
        # detection on the restored image when YOLO found none).
        refine_segs = detected
        if refine_mode != "Off" and self._seg_count(refine_segs) == 0:
            refine_segs = yolo(primary)
        processed, refine_changed, refine_info = self._apply_refine_stage(primary, refine_segs, refine_mode)
        processed = self._find_image_tensor(processed)
        processed = image if processed is None else processed

        coverage = (detected[0], [seg for kept in assigned for seg, _face in kept])
        segs_processed = (segs_with_image_crops(coverage, processed, full_crop_mask=True)
                          if coverage[1] else self._empty_segs(processed))
        segs_processed = make_stable_segs(segs_processed, image, processed, coverage_segs=coverage)

        restored = sum(1 for row in records if row["status"] == OK)
        skipped = [row for row in records if row["status"] != OK]
        face_lines = [f"FACE {index:<11}: {STATUS_TEXT[row['status']]} · IoU {row['iou']:.2f} · kps {row['keypoints']}/5"
                      for index, row in enumerate(records, 1)]
        if not self._seg_count(detected):
            face_lines = ["FACES           : NOT DETECTED BY YOLO"]
        log_lines = [
            "STATUS          : EXECUTED",
            "PROCESS MODE    : RESTORE",
            f"SELECTION       : {select_face} (YOLO)",
            f"RESTORE BACKEND : {backend_info['backend']} · {Path(backend_info['file']).name}",
            f"FACES DETECTED  : {self._seg_count(detected)}",
            f"FACES SELECTED  : {self._seg_count(selected)}",
            *face_lines,
            f"FACES RESTORED  : {restored}",
            f"REFINE          : {refine_info}",
            f"RESULT          : {'IMAGE PROCESSED' if restored or refine_changed else 'UNCHANGED'}",
        ]
        log_block = cmk_block_to_string("FaceProcess", 90, log_lines, True)
        warnings = [f"face {index}: {STATUS_TEXT[row['status']].lower()}" for index, row in enumerate(records, 1)
                    if row["status"] != OK]
        if not self._seg_count(detected):
            warnings.append("no face detected by YOLO")
        diagnostic = self._make_diagnostic(
            mode="Restore", refine_mode=refine_mode, enabled=True,
            status="Processed" if restored else ("No face detected" if not self._seg_count(detected) else "No face assigned"),
            image=image, primary_image=primary, processed_image=processed, segs_detected=detected,
            segs_processed=segs_processed, detect_model=detect_model, restore_model=Path(backend_info["file"]).name,
            restore_visibility=visibility,
            metadata={"selection": select_face, "match_rule": MATCH_RULE, "faces": records,
                      "restore_backend": {**backend_info, "file": Path(backend_info["file"]).name},
                      "faces_restored": restored, "faces_skipped": len(skipped)},
            warnings=warnings,
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
                    "boolean_face_enable": True, "face_sam_model": None, "face_seed": 0}
        settings = dict(face.get("face_prepare_settings") or {})
        process = face.get("source_pipe")
        if mode == "sdxl":
            from .loaders.z_image_turbo_loader import release_zit_resources

            # SDXL is about to be materialised: drop CMK's own ZIT references
            # unless another consumer still uses them (no-op when empty).
            release_zit_resources("ZIT -> SDXL FaceProcess")
            arguments = {name: settings[name] for name in (
                "sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler", "use_lora_from_1st_pass",
                *SDXL_SETTINGS, "prompt_pos",
            )}
            pipe = CMKFaceProcessPreparePipe().prepare(
                MODEL=face.get("face_model_spec"), PROCESS=process, IMAGE=image, face_global_enable=True, **arguments,
            )[0]
            return {**pipe, "face_mode": "sdxl"}
        if mode == "z_image_turbo":
            detailer = CMKDetailerPrepareMultiModelPipe().prepare(
                PROCESS=process,
                IMAGE=image,
                model_family="Z-Image Turbo",
                sam_model_name=settings["sam_model_name"],
                sam_device_mode=settings["sam_device_mode"],
                use_prompt_lora_from_sampler=settings["use_prompt_lora_from_sampler"],
                use_lora_from_1st_pass=settings["use_lora_from_1st_pass"],
                use_1st_pass_sampling=False,
                lora_name="None",
                strength_model=1.0,
                strength_clip=1.0,
                prompt_pos=settings["prompt_pos"],
                prompt_neg="",
                steps=settings["steps"],
                cfg=settings["cfg"],
                sampling_entry=95.0,
                zit_steps=settings["zit_steps"],
                zit_cfg=settings["zit_cfg"],
                zit_sampling_entry=ZIT_SAMPLING_ENTRY,
                sampler=settings["sampler"],
                scheduler=settings["scheduler"],
                zit_sampler=settings["zit_sampler"],
                zit_scheduler=settings["zit_scheduler"],
                stop_at_clip_layer=settings["stop_at_clip_layer"],
                pag_scale=settings["pag_scale"],
                sampling=settings["sampling"],
                zsnr=settings["zsnr"],
                zit_model_shift=settings["zit_model_shift"],
                freeu_enabled=settings["freeu_enabled"],
                freeu_b1=settings["freeu_b1"],
                freeu_b2=settings["freeu_b2"],
                freeu_s1=settings["freeu_s1"],
                freeu_s2=settings["freeu_s2"],
                **{"MODEL ZIT": face.get("face_model_spec")},
            )[0]
            return {
                "face_mode": "z_image_turbo",
                "source_pipe": detailer.get("source_pipe"),
                "_source_pipe": detailer.get("source_pipe"),
                "face_model": detailer.get("detailer_model"),
                # The sampling engine never reads CLIP; the text encoder is
                # released after encoding, the BASIC_PIPE check needs a value.
                "face_clip": self._ZIT_TEXT_ENCODER,
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
                "face_sam_model": detailer.get("detailer_sam_model"),
                "face_model_identity": detailer.get("detailer_model_identity"),
                "face_zit_model_shift": detailer.get("detailer_zit_model_shift"),
            }
        raise ValueError(f"{EXECUTE_LABEL}: unsupported FACE mode {mode!r}")
