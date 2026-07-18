from __future__ import annotations

from ..nodes.swap.face_process import CMK_FaceProcess
from ..nodes.swap.face_select import resolve_legacy_face_selection
from ..utils.stable_segs import make_stable_segs, segs_with_image_crops
from .cmk_log_pipe import cmk_block_to_string
from .cmk_persistent_cache import (
    build_node_fingerprint,
    load_pickle,
    pickle_available,
    save_pickle,
    write_status,
)


class CMKFaceProcessPipe(CMK_FaceProcess):
    """Pipe-native FaceProcess.

    The image and global enable state are read exclusively from CMK_FACE_PIPE.
    UI finalisation is intentionally deferred until the functional contract has
    been validated.
    """

    @classmethod
    def INPUT_TYPES(cls):
        parent = CMK_FaceProcess.INPUT_TYPES()
        optional = dict((parent or {}).get("optional", {}) or {})

        # The pipe-native node owns these values through face_pipe.
        for name in (
            "image",
            "face_pipe",
            "boolean_faceprocess_enable",
            "enable",
            "process_mode",
        ):
            optional.pop(name, None)

        # The legacy ReActor selection controls remain an implementation detail.
        for name in (
            "select_face_selection",
            "select_sort_by",
            "select_reverse_order",
            "select_take_start",
            "select_take_count",
        ):
            optional.pop(name, None)

        # Fixed CMK default for the integrated face detector. Fall back to the
        # first available face-ranked model when the exact file is not installed.
        detect_spec = optional.get("detect_model")
        if detect_spec:
            choices = list(detect_spec[0]) if isinstance(detect_spec[0], (list, tuple)) else []
            preferred = "bbox/face/face_yolov8n.pt"
            default_model = preferred if preferred in choices else (choices[0] if choices else "none")
            optional["detect_model"] = (choices or ["none"], {"default": default_model})

        # Native ComfyUI Advanced metadata. Mode-dependent visibility is handled
        # by the CMK frontend extension; these flags define the second UI layer
        # within the currently active mode.
        advanced_names = {
            "detect_bbox_threshold",
            "detect_bbox_dilation",
            "detect_crop_factor",
            "detect_drop_size",
            "restore_codeformer_weight",
            "detail_guide_size_for",
            "detail_max_size",
            "detail_noise_mask",
            "detail_force_inpaint",
            "detail_paste_feather",
        }
        for name in advanced_names:
            spec = optional.get(name)
            if not spec:
                continue
            values = list(spec)
            options = dict(values[1]) if len(values) > 1 and isinstance(values[1], dict) else {}
            options["advanced"] = True
            if len(values) > 1:
                values[1] = options
            else:
                values.append(options)
            optional[name] = tuple(values)

        return {
            "required": {
                "FACE": ("CMK_FACE_PIPE", {"lazy": True}),
                "enable": ("BOOLEAN", {"default": True}),
                "process_mode": (["restore", "detailer"], {"default": "restore"}),
                "select_face": (
                    ["Largest", "Leftmost", "Rightmost", "Topmost", "Bottommost", "Center", "All Faces"],
                    {"default": "Largest"},
                ),
            },
            "optional": optional,
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "SEGS",
        "SEGS",
        "BOOLEAN",
        "CMK_DIAGNOSTIC",
        "CMK_LOG_BLOCK",
    )
    RETURN_NAMES = (
        "IMAGE PROCEED",
        "SELECTED FACE",
        "SEGS PROCESSED",
        "ENABLED",
        "diagnostic",
        "LOG BLOCK",
    )
    FUNCTION = "run_pipe"
    CATEGORY = "CMK/Developer/Pipe/Execute"

    _CACHE_SCOPE = "faceprocess_branch"
    _CACHE_SCHEMA = "cmk_faceprocess_branch_v4"

    def _cache_key(self, prompt, unique_id):
        return build_node_fingerprint(
            prompt,
            unique_id,
            ("CMKFaceProcessPipe",),
            self._CACHE_SCHEMA,
            include_node_identity=True,
        )

    def check_lazy_status(
        self,
        FACE=None,
        prompt=None,
        unique_id=None,
        **kwargs,
    ):
        if not bool(kwargs.get("enable", True)):
            return ["FACE"] if FACE is None else []
        cache_key, detail = self._cache_key(prompt, unique_id)

        if cache_key and pickle_available(self._CACHE_SCOPE, cache_key):
            write_status(
                self._CACHE_SCOPE,
                "HIT_READY",
                cache_key=cache_key,
                detail=detail,
                unique_id=unique_id,
            )
            return []

        if cache_key is None:
            write_status(
                self._CACHE_SCOPE,
                "NO_FINGERPRINT",
                detail=detail,
                unique_id=unique_id,
            )

        return ["FACE"] if FACE is None else []

    def run_pipe(
        self,
        FACE,
        enable=True,
        process_mode="restore",
        select_face="Largest",
        prompt=None,
        unique_id=None,
        **kwargs,
    ):
        face_pipe=FACE
        if not isinstance(face_pipe,dict): raise ValueError("CMK FaceProcess -Pipe-: FACE is missing")
        image=face_pipe.get("face_image")
        if image is None: raise ValueError("CMK FaceProcess -Pipe-: no image found in face_pipe")
        global_enable=bool(face_pipe.get("face_global_enable",False))
        effective_enable=bool(global_enable and enable)
        normalized_mode="Detailer" if str(process_mode).strip().lower()=="detailer" else "Restore"
        if not effective_enable:
            empty=self._empty_segs(image)
            status="LOCAL DISABLED" if not enable else "GLOBAL DISABLED"
            lines=[f"STATUS          : {status}",f"PROCESS MODE    : {normalized_mode.upper()}","FACE DETECTION  : SKIPPED","PROCESSING      : SKIPPED","CACHE           : SKIPPED","RESULT          : PASSTHROUGH"]
            block=cmk_block_to_string("FaceProcess",90,lines,True)
            diagnostic={"title":"FaceProcess -Pipe-","summary":"disabled passthrough","details":"\n".join(lines),"metadata":{"global_enabled":global_enable,"local_enabled":bool(enable)}}
            return (image,empty,empty,False,diagnostic,block)
        cache_key, cache_detail=self._cache_key(prompt,unique_id)

        if (
            cache_key
            and pickle_available(self._CACHE_SCOPE, cache_key)
        ):
            try:
                payload = load_pickle(
                    self._CACHE_SCOPE,
                    cache_key,
                )
                if not isinstance(payload, dict):
                    raise TypeError(
                        "cached FaceProcess payload is invalid"
                    )

                write_status(
                    self._CACHE_SCOPE,
                    "HIT",
                    cache_key=cache_key,
                    unique_id=unique_id,
                )
                print(
                    "[CMK FaceProcess -Pipe-] CACHE HIT "
                    f"{cache_key[:12]}"
                )
                return (
                    payload.get("image_proceed"),
                    payload.get("selected_face"),
                    payload.get("segs_processed"),
                    bool(payload.get("enabled", False)),
                    payload.get("diagnostic"),
                    payload.get("log_block", ""),
                )
            except Exception as exc:
                write_status(
                    self._CACHE_SCOPE,
                    "HIT_FAILED",
                    cache_key=cache_key,
                    detail=str(exc),
                    unique_id=unique_id,
                )
                print(
                    "[CMK FaceProcess -Pipe-] CACHE HIT FAILED "
                    f"{cache_key[:12]}: {exc}"
                )

        refine_mode = kwargs.pop("refine_mode", "Off")

        selection = resolve_legacy_face_selection(image, select_face)
        kwargs.update(selection)

        result = super().run(
            image=image,
            face_pipe=face_pipe,
            boolean_faceprocess_enable=global_enable,
            enable=bool(enable),
            process_mode=normalized_mode,
            refine_mode=refine_mode,
            **kwargs,
        )

        # Parent result:
        # image, image_proceed, segs_detected, segs_processed, log_text, enabled, diagnostic
        image_out, image_proceed, segs_detected, segs_processed, log_text, enabled, diagnostic = result
        selected_face = self._select_output_segs(segs_detected, image, select_face)

        # A FaceProcess branch must claim only the face(s) it actually owns.
        # The parent implementation re-detects the complete processed image and
        # therefore also returns unchanged sibling faces. Those sibling SEGS
        # caused later branches to paste source pixels over earlier results.
        coverage_segs = selected_face
        if not self._is_valid_segs(coverage_segs) or not coverage_segs[1]:
            coverage_segs = self._select_output_segs(
                segs_processed,
                image_proceed,
                select_face,
            )
        if self._is_valid_segs(coverage_segs):
            selected_face = coverage_segs
            segs_processed = segs_with_image_crops(
                coverage_segs,
                image_proceed,
                full_crop_mask=(normalized_mode == "Restore"),
            )
        else:
            segs_processed = self._empty_segs(image_proceed)

        segs_processed = make_stable_segs(
            segs_processed,
            image,
            image_proceed,
            coverage_segs=coverage_segs,
        )

        detected_count = len(segs_detected[1]) if self._is_valid_segs(segs_detected) else 0
        processed_count = len(segs_processed[1]) if self._is_valid_segs(segs_processed) else 0
        effective_enable = bool(enabled)
        status = "EXECUTED" if effective_enable else ("LOCAL DISABLED" if not enable else "DISABLED")
        log_lines = [
            f"STATUS          : {status}",
            f"PROCESS MODE    : {str(normalized_mode).upper()}",
            f"SELECTION       : {select_face}",
            f"FACES DETECTED  : {detected_count}",
            f"FACES PROCESSED : {processed_count}",
            f"RESULT          : {'PASSTHROUGH' if not effective_enable else 'IMAGE PROCESSED'}",
        ]
        log_block = cmk_block_to_string("FaceProcess", 90, log_lines, True)

        payload = {
            "image_proceed": image_proceed,
            "selected_face": selected_face,
            "segs_processed": segs_processed,
            "enabled": bool(enabled),
            "diagnostic": diagnostic,
            "log_block": log_block,
        }

        if cache_key:
            try:
                save_pickle(
                    self._CACHE_SCOPE,
                    cache_key,
                    payload,
                    max_entries=32,
                )
                write_status(
                    self._CACHE_SCOPE,
                    "MISS_STORED",
                    cache_key=cache_key,
                    detail=cache_detail,
                    unique_id=unique_id,
                )
                print(
                    "[CMK FaceProcess -Pipe-] CACHE MISS "
                    f"{cache_key[:12]} -> STORED"
                )
            except Exception as exc:
                write_status(
                    self._CACHE_SCOPE,
                    "STORE_FAILED",
                    cache_key=cache_key,
                    detail=str(exc),
                    unique_id=unique_id,
                )
                print(
                    "[CMK FaceProcess -Pipe-] CACHE STORE FAILED: "
                    f"{exc}"
                )

        return (
            image_proceed,
            selected_face,
            segs_processed,
            bool(enabled),
            diagnostic,
            log_block,
        )

    def _select_output_segs(self, segs, image, selection):
        if not self._is_valid_segs(segs) or not segs[1]:
            return segs
        if str(selection or "Largest") == "All Faces":
            return segs

        items = list(segs[1])
        if len(items) == 1:
            return segs

        height = int(image.shape[1]) if getattr(image, "shape", None) is not None and len(image.shape) >= 3 else 0
        width = int(image.shape[2]) if getattr(image, "shape", None) is not None and len(image.shape) >= 3 else 0

        def metrics(item):
            box = self._seg_bbox(item)
            if not box:
                return {"area": -1.0, "cx": float("inf"), "cy": float("inf"), "center": float("inf")}
            x1, y1, x2, y2 = [float(v) for v in box]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            center = (cx - width * 0.5) ** 2 + (cy - height * 0.5) ** 2
            return {"area": area, "cx": cx, "cy": cy, "center": center}

        mode = str(selection or "Largest")
        if mode == "Leftmost":
            chosen = min(items, key=lambda item: metrics(item)["cx"])
        elif mode == "Rightmost":
            chosen = max(items, key=lambda item: metrics(item)["cx"])
        elif mode == "Topmost":
            chosen = min(items, key=lambda item: metrics(item)["cy"])
        elif mode == "Bottommost":
            chosen = max(items, key=lambda item: metrics(item)["cy"])
        elif mode == "Center":
            chosen = min(items, key=lambda item: metrics(item)["center"])
        else:
            chosen = max(items, key=lambda item: metrics(item)["area"])

        return (segs[0], [chosen])
