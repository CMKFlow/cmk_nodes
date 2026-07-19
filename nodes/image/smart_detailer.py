import numpy as np
import comfy.samplers
import folder_paths

from ...engine.native_detailer import (
    core,
    SimpleDetectorForEach,
    SEGSDetailer,
    SEGSPaste,
    UltralyticsDetectorProvider,
)

from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...engine.detailer_limits import clamp_detailer_denoise
from ...utils.stable_segs import make_stable_segs
from ...pipe.cmk_log_pipe import cmk_block_to_string
from ...pipe.cmk_persistent_cache import (
    build_node_fingerprint,
    load_pickle,
    pickle_available,
    save_pickle,
    write_status,
)


class CMK_SmartDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]

        return {
            "required": {
                "image": ("IMAGE",),
                "basic_pipe": ("BASIC_PIPE",),
                "boolean_detailer_enable": ("BOOLEAN", {"default": True}),
                "enable": ("BOOLEAN", {"default": True}),
                "model_name": (bboxs + segms,),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "drop_size": ("INT", {"default": 10, "min": 1, "max": 8192, "step": 1}),
                "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                "max_size": ("FLOAT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (core.get_schedulers(),),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 0.5, "step": 0.01}),
                "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            },
            "optional": {
                "sam_model_opt": ("SAM_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE", "SEGS", "SEGS", "STRING", "BOOLEAN", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("image", "segs_detected", "segs_detailed", "detailer_log", "enabled", "diagnostic")
    FUNCTION = "run"
    CATEGORY = "CMK/Toolbox/Image"

    def _is_image_tensor(self, value):
        return hasattr(value, "shape") and len(value.shape) >= 3

    def _find_image_tensor(self, value):
        if self._is_image_tensor(value):
            return value

        if isinstance(value, (tuple, list)):
            for item in value:
                found = self._find_image_tensor(item)
                if found is not None:
                    return found

        return None

    def _is_valid_segs(self, value):
        if not isinstance(value, tuple):
            return False
        if len(value) != 2:
            return False
        if not isinstance(value[0], tuple):
            return False
        if len(value[0]) != 2:
            return False
        if not isinstance(value[1], list):
            return False

        for seg in value[1]:
            if not hasattr(seg, "cropped_image"):
                return False

        return True

    def _find_valid_segs(self, value):
        if self._is_valid_segs(value):
            return value

        if isinstance(value, (tuple, list)):
            for item in value:
                found = self._find_valid_segs(item)
                if found is not None:
                    return found

        return None

    def _empty_segs(self, image):
        image = self._find_image_tensor(image)

        if image is None:
            return ((0, 0), [])

        return ((image.shape[2], image.shape[1]), [])

    @staticmethod
    def _tensor_batch_to_uint8(value):
        if value is None:
            return None
        try:
            tensor = value.detach().cpu() if hasattr(value, "detach") else value
            arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            if arr.ndim != 3:
                return None
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.moveaxis(arr, 0, -1)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            arr = arr.astype(np.float32)
            if float(np.nanmax(arr)) <= 1.5:
                arr = arr * 255.0
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
            return np.clip(arr, 0, 255).astype(np.uint8)
        except Exception:
            return None

    @staticmethod
    def _uint8_to_tensor_image(arr):
        if arr is None:
            return None
        try:
            out = np.clip(arr, 0, 255).astype(np.float32) / 255.0
            import torch
            return torch.from_numpy(out)[None,]
        except Exception:
            return None

    @staticmethod
    def _seg_bbox(seg):
        for attr in ("bbox", "crop_region"):
            box = getattr(seg, attr, None)
            if box is None:
                continue
            try:
                if len(box) >= 4:
                    return [int(round(float(box[0]))), int(round(float(box[1]))), int(round(float(box[2]))), int(round(float(box[3])))]
            except Exception:
                pass
        return None

    def _make_detection_preview(self, image, segs):
        arr = self._tensor_batch_to_uint8(image)
        if arr is None:
            return None
        try:
            items = segs[1] if self._is_valid_segs(segs) else []
        except Exception:
            items = []
        if not items:
            return self._uint8_to_tensor_image(arr)
        out = arr.copy()
        h, w = out.shape[:2]
        thickness = max(2, int(round(max(h, w) / 350)))
        for seg in items:
            box = self._seg_bbox(seg)
            if not box:
                continue
            x1, y1, x2, y2 = box
            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            out[y1:y1+thickness, x1:x2+1] = 255
            out[max(y2-thickness+1, y1):y2+1, x1:x2+1] = 255
            out[y1:y2+1, x1:x1+thickness] = 255
            out[y1:y2+1, max(x2-thickness+1, x1):x2+1] = 255
        return self._uint8_to_tensor_image(out)

    def _make_difference_preview(self, before, after):
        a = self._tensor_batch_to_uint8(before)
        b = self._tensor_batch_to_uint8(after)
        if a is None or b is None or a.shape != b.shape:
            return None
        diff = np.abs(b.astype(np.int16) - a.astype(np.int16)).astype(np.float32)
        diff = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
        return self._uint8_to_tensor_image(diff)

    def _mean_abs_change(self, before, after):
        a = self._tensor_batch_to_uint8(before)
        b = self._tensor_batch_to_uint8(after)
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        return float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32))) / 255.0)

    @staticmethod
    def _enabled_text(value):
        return "Enabled" if bool(value) else "Disabled"

    @staticmethod
    def _summary_lines(
        *,
        status,
        model_name,
        detections,
        steps,
        cfg,
        denoise,
        force_inpaint,
        guide_size,
        max_size,
        global_enabled=None,
        local_enabled=None,
        changed=0.0,
        reason="",
    ):
        lines = [
            f"Status        : {status}",
        ]
        if global_enabled is not None:
            lines.append(f"Global Enable : {CMK_SmartDetailer._enabled_text(global_enabled)}")
        if local_enabled is not None:
            lines.append(f"Local Enable  : {CMK_SmartDetailer._enabled_text(local_enabled)}")
        lines.extend([
            f"Model         : {model_name}",
            f"Detections    : {detections}",
            f"Changed       : {float(changed or 0.0):.4f}",
            f"Steps         : {steps}",
            f"CFG           : {cfg}",
            f"Denoise       : {denoise}",
            f"Force Inpaint : {CMK_SmartDetailer._enabled_text(force_inpaint)}",
            f"Guide Size    : {guide_size:g}",
            f"Max Size      : {max_size:g}",
        ])
        if reason:
            lines.append(f"Reason        : {reason}")
        return lines

    def _make_diagnostic(
        self,
        *,
        title="Smart Detailer",
        status,
        image_before=None,
        image_after=None,
        segs_detected=None,
        model_name="",
        detections=0,
        steps=0,
        cfg=0.0,
        denoise=0.0,
        force_inpaint=False,
        guide_size=0.0,
        max_size=0.0,
        reason="",
        enabled=False,
        global_enabled=None,
        local_enabled=None,
    ):
        final_image = image_after if image_after is not None else image_before
        changed = self._mean_abs_change(image_before, final_image) if image_before is not None and final_image is not None else 0.0

        stages = []
        previews = []

        def add_stage(title_text, tensor, subtitle=""):
            if tensor is None:
                return
            stages.append({"title": title_text, "subtitle": subtitle, "image": tensor})
            previews.append(tensor)

        add_stage("01 Source", image_before, "original input / passthrough")

        detection_preview = self._make_detection_preview(image_before, segs_detected) if image_before is not None else None
        if detection_preview is not None:
            add_stage("02 Detection", detection_preview, f"detections: {int(detections or 0)}")

        if image_after is not None and bool(enabled) and int(detections or 0) > 0:
            add_stage("03 Detailer", image_after, f"change: {changed:.4f}")

        diff_preview = self._make_difference_preview(image_before, final_image) if changed > 0.00001 else None
        if diff_preview is not None:
            add_stage("04 Difference", diff_preview, f"source→detailer: {changed:.4f}")

        add_stage("05 Final", final_image, "final output")

        summary = "\n".join(CMK_SmartDetailer._summary_lines(
            status=status,
            model_name=model_name,
            detections=detections,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            force_inpaint=force_inpaint,
            guide_size=float(guide_size or 0.0),
            max_size=float(max_size or 0.0),
            global_enabled=global_enabled,
            local_enabled=local_enabled,
            changed=changed,
            reason=reason,
        ))

        warnings = []
        if reason:
            warnings.append(reason)

        return make_diagnostic_payload(
            title=title,
            node="CMK Smart Detailer",
            previews=previews,
            stages=stages,
            summary=summary,
            details=summary,
            mode="Detailer",
            metadata={
                "status": status,
                "enabled": bool(enabled),
                "global_enabled": bool(global_enabled) if global_enabled is not None else None,
                "local_enabled": bool(local_enabled) if local_enabled is not None else None,
                "model_name": model_name,
                "detections": int(detections or 0),
                "steps": int(steps or 0),
                "cfg": float(cfg or 0.0),
                "denoise": float(denoise or 0.0),
                "force_inpaint": bool(force_inpaint),
                "guide_size": float(guide_size or 0.0),
                "max_size": float(max_size or 0.0),
                "changed": float(changed or 0.0),
                "reason": reason,
            },
            warnings=warnings,
            metrics={
                "detections": int(detections or 0),
                "changed": float(changed or 0.0),
            },
        )

    def run(
        self,
        image,
        basic_pipe,
        boolean_detailer_enable,
        enable,
        model_name,
        bbox_threshold,
        bbox_dilation,
        crop_factor,
        drop_size,
        guide_size,
        guide_size_for,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        noise_mask,
        force_inpaint,
        sam_model_opt=None,
    ):
        denoise = clamp_detailer_denoise(denoise)
        image = self._find_image_tensor(image)

        if image is None:
            empty_segs = ((0, 0), [])
            detailer_log = "error | no valid image tensor"
            diagnostic = self._make_diagnostic(
                status="Error",
                image_before=None,
                image_after=None,
                segs_detected=empty_segs,
                model_name=model_name,
                detections=0,
                steps=steps,
                cfg=cfg,
                denoise=denoise,
                force_inpaint=force_inpaint,
                guide_size=guide_size,
                max_size=max_size,
                reason="no valid image tensor",
                enabled=False,
                global_enabled=boolean_detailer_enable,
                local_enabled=enable,
            )
            return (
                image,
                empty_segs,
                empty_segs,
                detailer_log,
                False,
                diagnostic,
            )

        empty_segs = self._empty_segs(image)

        detailer_enabled = bool(boolean_detailer_enable) and bool(enable)

        if not detailer_enabled:
            reason = []
            if not boolean_detailer_enable:
                reason.append("global disabled")
            if not enable:
                reason.append("local disabled")

            reason_text = " + ".join(reason)
            detailer_log = f"{reason_text} | model skipped: {model_name}"
            diagnostic = self._make_diagnostic(
                status="Disabled",
                image_before=image,
                image_after=image,
                segs_detected=empty_segs,
                model_name=model_name,
                detections=0,
                steps=steps,
                cfg=cfg,
                denoise=denoise,
                force_inpaint=force_inpaint,
                guide_size=guide_size,
                max_size=max_size,
                reason=reason_text,
                enabled=False,
                global_enabled=boolean_detailer_enable,
                local_enabled=enable,
            )
            return (
                image,
                empty_segs,
                empty_segs,
                detailer_log,
                False,
                diagnostic,
            )

        bbox_detector, segm_detector = UltralyticsDetectorProvider().doit(model_name)

        if model_name.startswith("bbox/"):
            segm_detector = None

        segs_detected = SimpleDetectorForEach.detect(
            bbox_detector=bbox_detector,
            image=image,
            bbox_threshold=bbox_threshold,
            bbox_dilation=bbox_dilation,
            crop_factor=crop_factor,
            drop_size=drop_size,
            sub_threshold=bbox_threshold,
            sub_dilation=0,
            sub_bbox_expansion=0,
            sam_mask_hint_threshold=0.7,
            post_dilation=0,
            sam_model_opt=sam_model_opt,
            segm_detector_opt=segm_detector,
        )[0]

        seg_count = len(segs_detected[1])

        if seg_count == 0:
            detailer_log = f"no detection | model: {model_name}"
            diagnostic = self._make_diagnostic(
                status="No Detection",
                image_before=image,
                image_after=image,
                segs_detected=segs_detected,
                model_name=model_name,
                detections=0,
                steps=steps,
                cfg=cfg,
                denoise=denoise,
                force_inpaint=force_inpaint,
                guide_size=guide_size,
                max_size=max_size,
                reason="no detection",
                enabled=True,
                global_enabled=boolean_detailer_enable,
                local_enabled=enable,
            )
            return (
                image,
                segs_detected,
                segs_detected,
                detailer_log,
                True,
                diagnostic,
            )

        # Impact Pack intentionally skips detailing when force_inpaint is disabled
        # and the detected region is already at/above guide_size. CMK Smart Detailer
        # is expected to detail whenever it is enabled, so in non-force mode we lift
        # guide_size/max_size just enough to keep the normal upscale-based detailer
        # path active without switching to Impact's force-inpaint path.
        effective_guide_size = float(guide_size)
        effective_max_size = float(max_size)

        if not force_inpaint:
            required_guide_size = effective_guide_size
            required_max_size = effective_max_size

            for seg in segs_detected[1]:
                try:
                    bbox_w = float(seg.bbox[2] - seg.bbox[0])
                    bbox_h = float(seg.bbox[3] - seg.bbox[1])
                    crop_w = float(seg.crop_region[2] - seg.crop_region[0])
                    crop_h = float(seg.crop_region[3] - seg.crop_region[1])
                except Exception:
                    continue

                basis = min(bbox_w, bbox_h) if guide_size_for else min(crop_w, crop_h)
                if basis > 0:
                    required_guide_size = max(required_guide_size, basis + 8.0)
                    upscale = required_guide_size / basis
                    required_max_size = max(required_max_size, max(crop_w * upscale, crop_h * upscale))

            effective_guide_size = required_guide_size
            effective_max_size = required_max_size

        seed = 0 if seed is None else int(seed)
        steps = 20 if steps is None else int(steps)

        detailer_result = SEGSDetailer().doit(
            image=image,
            segs=segs_detected,
            guide_size=effective_guide_size,
            guide_size_for=guide_size_for,
            max_size=effective_max_size,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            noise_mask=noise_mask,
            force_inpaint=force_inpaint,
            basic_pipe=basic_pipe,
            refiner_ratio=0.2,
            batch_size=1,
            cycle=1,
            refiner_basic_pipe_opt=None,
            inpaint_model=False,
            noise_mask_feather=20,
            scheduler_func_opt=None,
        )

        # SEGSDetailer primarily returns the processed SEGS.  Those crops must
        # be pasted back into the full source image before image/diagnostic
        # outputs represent the actual detailer result.
        segs_detailed = self._find_valid_segs(detailer_result)

        if segs_detailed is None:
            segs_detailed = segs_detected

        try:
            pasted_result = SEGSPaste.doit(
                image=image,
                segs=segs_detailed,
                feather=5,
                alpha=255,
            )
            image_detailed = self._find_image_tensor(pasted_result)
        except Exception:
            image_detailed = None

        # Compatibility fallback for Impact Pack variants that already return
        # a composed image from SEGSDetailer.
        if image_detailed is None:
            image_detailed = self._find_image_tensor(detailer_result)

        if image_detailed is None:
            image_detailed = image

        detailer_log = (
            f"enabled | model: {model_name} | detections: {seg_count} | "
            f"steps: {steps} | cfg: {cfg} | denoise: {denoise} | "
            f"force_inpaint: {force_inpaint} | guide_size: {effective_guide_size:g} | max_size: {effective_max_size:g}"
        )

        diagnostic = self._make_diagnostic(
            status="Enabled",
            image_before=image,
            image_after=image_detailed,
            segs_detected=segs_detected,
            model_name=model_name,
            detections=seg_count,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            force_inpaint=force_inpaint,
            guide_size=effective_guide_size,
            max_size=effective_max_size,
            reason="",
            enabled=True,
            global_enabled=boolean_detailer_enable,
            local_enabled=enable,
        )

        return (
            image_detailed,
            segs_detected,
            segs_detailed,
            detailer_log,
            True,
            diagnostic,
        )


class CMK_SmartDetailerPipe(CMK_SmartDetailer):
    """Execute Smart Detailer directly from CMK_DETAILER_PIPE."""

    @classmethod
    def INPUT_TYPES(cls):
        bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]
        return {
            "required": {
                "DETAILER": ("CMK_DETAILER_PIPE", {"lazy": True}),
                "enable": ("BOOLEAN", {"default": True}),
                "output_image_proceed": ("BOOLEAN", {"default": True}),
                "model_name": (bboxs + segms,),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1, "advanced": True}),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "drop_size": ("INT", {"default": 10, "min": 1, "max": 8192, "step": 1, "advanced": True}),
                "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region", "advanced": True}),
                "max_size": ("FLOAT", {"default": 768, "min": 64, "max": 8192, "step": 8, "advanced": True}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 0.5, "step": 0.01}),
                "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled", "advanced": True}),
                "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled", "advanced": True}),
            }
       ,
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("SEGS", "SEGS", "IMAGE", "CMK_DIAGNOSTIC", "CMK_LOG_BLOCK")
    RETURN_NAMES = ("SEGS DETECTED", "SEGS PROCEED", "IMAGE PROCEED", "diagnostic", "LOG BLOCK")
    FUNCTION = "run_pipe"
    CATEGORY = "CMK/Developer/Pipe/Execute"

    _CACHE_SCOPE = "detailer_branch"
    _CACHE_SCHEMA = "cmk_detailer_branch_v4"

    def _cache_key(self, prompt, unique_id):
        return build_node_fingerprint(
            prompt,
            unique_id,
            ("CMK_SmartDetailerPipe",),
            self._CACHE_SCHEMA,
            exclude_inputs=("output_image_proceed",),
            include_node_identity=True,
        )

    def check_lazy_status(
        self,
        DETAILER=None,
        prompt=None,
        unique_id=None,
        **kwargs,
    ):
        if not bool(kwargs.get("enable", True)):
            return ["DETAILER"] if DETAILER is None else []
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

        return ["DETAILER"] if DETAILER is None else []

    def _load_cached_result(
        self,
        cache_key,
        output_image_proceed,
    ):
        payload = load_pickle(self._CACHE_SCOPE, cache_key)
        if not isinstance(payload, dict):
            raise TypeError("cached detailer payload is invalid")

        result = (
            payload.get("segs_detected"),
            payload.get("segs_proceed"),
            (
                payload.get("image_proceed")
                if bool(output_image_proceed)
                else None
            ),
            payload.get("diagnostic"),
            payload.get("log_block", ""),
        )
        preview = payload.get("detection_preview")

        write_status(
            self._CACHE_SCOPE,
            "HIT",
            cache_key=cache_key,
        )
        print(
            "[CMK Smart Detailer -Pipe-] CACHE HIT "
            f"{cache_key[:12]}"
        )
        return self._preview_result(preview, result)

    @staticmethod
    def _required(detailer_pipe, key):
        value = detailer_pipe.get(key) if isinstance(detailer_pipe, dict) else None
        if value is None:
            raise ValueError(f"CMK Smart Detailer -Pipe-: detailer_pipe['{key}'] is missing")
        return value

    def _preview_result(self, preview, result):
        if preview is None:
            return result
        try:
            from nodes import PreviewImage
            payload = PreviewImage().save_images(
                preview,
                filename_prefix="CMK_Smart_Detailer_Detected",
                prompt=None,
                extra_pnginfo=None,
            )
            ui = payload.get("ui", {}) if isinstance(payload, dict) else {}
            return {"ui": ui, "result": result}
        except Exception:
            return result

    def run_pipe(
        self,
        DETAILER,
        enable,
        output_image_proceed,
        model_name,
        bbox_threshold,
        bbox_dilation,
        crop_factor,
        drop_size,
        guide_size,
        guide_size_for,
        max_size,
        denoise,
        noise_mask,
        force_inpaint,
        prompt=None,
        unique_id=None,
):
        denoise = clamp_detailer_denoise(denoise)
        cache_key, cache_detail = self._cache_key(prompt,unique_id)

        if (
            cache_key
            and pickle_available(self._CACHE_SCOPE, cache_key)
        ):
            try:
                return self._load_cached_result(
                    cache_key,
                    output_image_proceed,
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
                    "[CMK Smart Detailer -Pipe-] CACHE HIT FAILED "
                    f"{cache_key[:12]}: {exc}"
                )

        # On a persistent cache hit ComfyUI's lazy execution intentionally does
        # not materialise the shared DETAILER input. Therefore the cached result
        # must be resolved before validating or reading that input.
        detailer_pipe=DETAILER
        if not isinstance(detailer_pipe,dict): raise ValueError("CMK Smart Detailer -Pipe-: DETAILER is missing")
        source_image=self._required(detailer_pipe,"detailer_image")
        detailer_global_enable=bool(detailer_pipe.get("detailer_global_enable",detailer_pipe.get("boolean_detailer_enable",False)))
        effective_enable=bool(detailer_global_enable and enable)
        if not effective_enable:
            _,h,w,_=source_image.shape
            empty=((int(w),int(h)),[])
            status="LOCAL DISABLED" if not enable else "GLOBAL DISABLED"
            lines=[f"STATUS          : {status}","DETECTION       : SKIPPED","DETAILING       : SKIPPED","CACHE           : SKIPPED","RESULT          : PASSTHROUGH"]
            block=cmk_block_to_string("Smart Detailer",70,lines,True)
            diagnostic=make_diagnostic_payload(title="Smart Detailer -Pipe-",node="CMK Smart Detailer -Pipe-",previews=[source_image],summary="disabled passthrough",details="\n".join(lines),mode="disabled / passthrough",metadata={"global_enabled":detailer_global_enable,"local_enabled":bool(enable)})
            return (empty,empty,source_image if bool(output_image_proceed) else None,diagnostic,block)

        basic_pipe = (
            self._required(detailer_pipe, "detailer_model"),
            self._required(detailer_pipe, "detailer_clip"),
            self._required(detailer_pipe, "detailer_vae"),
            self._required(detailer_pipe, "detailer_conditioning_pos"),
            self._required(detailer_pipe, "detailer_conditioning_neg"),
        )

        image_proceed, segs_detected, segs_proceed, log_pipe, _enabled, diagnostic = super().run(
            image=source_image,
            basic_pipe=basic_pipe,
            boolean_detailer_enable=detailer_global_enable,
            enable=enable,
            model_name=model_name,
            bbox_threshold=bbox_threshold,
            bbox_dilation=bbox_dilation,
            crop_factor=crop_factor,
            drop_size=drop_size,
            guide_size=guide_size,
            guide_size_for=guide_size_for,
            max_size=max_size,
            seed=detailer_pipe.get("detailer_seed", 0),
            steps=detailer_pipe.get("detailer_steps", 20),
            cfg=detailer_pipe.get("detailer_cfg", 8.0),
            sampler_name=detailer_pipe.get("detailer_sampler", "euler"),
            scheduler=detailer_pipe.get("detailer_scheduler", "simple"),
            denoise=denoise,
            noise_mask=noise_mask,
            force_inpaint=force_inpaint,
            sam_model_opt=detailer_pipe.get("detailer_sam_model"),
        )

        # The public value remains ordinary Impact-compatible SEGS, but carries
        # a full cache-stable branch composition for CMK SEGS CONCAT. This makes
        # a cache hit pixel-identical to a fresh execution and avoids re-pasting
        # pickled SEG cropped_image tensors.
        segs_proceed = make_stable_segs(
            segs_proceed,
            source_image,
            image_proceed,
            coverage_segs=segs_proceed,
        )
        opt_image_proceed = image_proceed if bool(output_image_proceed) else None
        detection_preview = self._make_detection_preview(source_image, segs_detected)

        detected_count = len(segs_detected[1]) if isinstance(segs_detected, tuple) and len(segs_detected) > 1 else 0
        proceed_count = len(segs_proceed[1]) if isinstance(segs_proceed, tuple) and len(segs_proceed) > 1 else 0
        effective_enable = bool(detailer_global_enable and enable)
        status = "EXECUTED" if effective_enable else ("LOCAL DISABLED" if not enable else "DISABLED")
        log_lines = [
            f"STATUS          : {status}",
            f"DETECT MODEL    : {model_name}",
            f"DETECTIONS      : {detected_count}",
            f"PROCESSED       : {proceed_count}",
            f"RESULT          : {'PASSTHROUGH' if not effective_enable else 'IMAGE PROCESSED'}",
        ]
        log_block = cmk_block_to_string("Smart Detailer", 70, log_lines, True)

        payload = {
            "segs_detected": segs_detected,
            "segs_proceed": segs_proceed,
            "image_proceed": image_proceed,
            "diagnostic": diagnostic,
            "log_block": log_block,
            "detection_preview": detection_preview,
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
                    "[CMK Smart Detailer -Pipe-] CACHE MISS "
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
                    "[CMK Smart Detailer -Pipe-] CACHE STORE FAILED: "
                    f"{exc}"
                )

        result = (
            segs_detected,
            segs_proceed,
            opt_image_proceed,
            diagnostic,
            log_block,
        )
        return self._preview_result(detection_preview, result)


NODE_CLASS_MAPPINGS = {
    "CMK_SmartDetailer": CMK_SmartDetailer,
    "CMK_SmartDetailerPipe": CMK_SmartDetailerPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_SmartDetailer": "CMK Smart Detailer",
    "CMK_SmartDetailerPipe": "CMK Smart Detailer -Pipe-",
}
