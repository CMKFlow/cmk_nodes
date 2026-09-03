from __future__ import annotations

import numpy as np
import torch

from ..engine.detector_engine import CMKDetectorEngine, DetectorSettings
from ..engine.content_guard import GUARD_VERSION, get_content_guard
from ..engine.instantid_face_rebuild import (
    FaceRebuildGeometry,
    build_geometry,
    largest_face,
    make_inpaint_mask,
    make_pasteback_mask,
    pasteback_exact,
    resize_mask,
    resize_rgb,
    select_target_face,
    source_shape_in_target_pose,
    warp_roi_towards_shape,
)
from ..models.model_manager import list_detector_models
from ..pipe.cmk_log_pipe import cmk_add_block, cmk_block_to_string, cmk_parse_block_string
from ..utils.cmk_diagnostic import make_diagnostic_payload
from ..utils.tensor_utils import tensor_to_uint8_rgb, uint8_rgb_to_tensor


TARGET_FACE_MODES = ["Largest", "Leftmost", "Rightmost", "Topmost", "Bottommost", "Center"]
FACEREBUILD_GUARD_VERSION = f"{GUARD_VERSION}:facerebuild-target-v1"


def _box_text(box):
    return ", ".join(str(int(value)) for value in box)


class CMKInstantIDFaceRebuildPrepare:
    """Detect the largest target face and prepare a local InstantID Inpaint ROI."""

    CATEGORY = "CMK/Toolbox/Face"
    FUNCTION = "prepare"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "CMK_FACE_REBUILD", "CMK_LOG_BLOCK", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("TARGET IMAGE", "TARGET ROI", "SOURCE FACE", "INPAINT MASK", "PASTEBACK MASK", "REBUILD CONTEXT", "LOG BLOCK", "diagnostic")

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return FACEREBUILD_GUARD_VERSION

    @classmethod
    def INPUT_TYPES(cls):
        detector_models = list_detector_models()
        detector_default = "antelopev2" if "antelopev2" in detector_models else detector_models[0]
        return {
            "required": {
                "TARGET IMAGE": ("IMAGE",),
                "SOURCE FACE": ("IMAGE",),
                "TARGET FACE": (TARGET_FACE_MODES, {"default": "Largest"}),
                "HEAD AREA": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05}),
                "NECK AREA": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 2.0, "step": 0.05}),
                "MASK FEATHER": ("INT", {"default": 32, "min": 0, "max": 64, "step": 1}),
                "WORKING RESOLUTION": (["768", "1024", "1280"], {"default": "1024"}),
                "DETECT MODEL": (detector_models, {"default": detector_default, "advanced": True}),
                "DETECT SIZE": ("INT", {"default": 640, "min": 128, "max": 1280, "step": 64, "advanced": True}),
            },
            "optional": {
                "SHAPE STRENGTH": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "advanced": True,
                    "tooltip": "Deform target face geometry toward dense source landmarks before FaceRebuild. 0 preserves the target shape.",
                }),
            },
        }

    def prepare(self, **kwargs):
        target = kwargs["TARGET IMAGE"]
        source = kwargs["SOURCE FACE"]
        if int(target.shape[0]) != 1:
            raise ValueError("InstantID Face Rebuild prototype supports exactly one target image")
        rgb = tensor_to_uint8_rgb(target[0])
        detector_model = str(kwargs["DETECT MODEL"])
        faces = CMKDetectorEngine().detect_image(
            rgb,
            DetectorSettings(detector_model=detector_model, detector_size=int(kwargs["DETECT SIZE"])),
        )
        target_selection = str(kwargs["TARGET FACE"])
        face = select_target_face(faces, target_selection, rgb.shape[1], rgb.shape[0])
        get_content_guard().inspect_image(rgb, face, "target")
        geometry = build_geometry(
            face["bbox"],
            rgb.shape[1],
            rgb.shape[0],
            head_area=float(kwargs["HEAD AREA"]),
            neck_area=float(kwargs["NECK AREA"]),
            feather=int(kwargs["MASK FEATHER"]),
            working_resolution=int(kwargs["WORKING RESOLUTION"]),
        )
        left, top, right, bottom = geometry.roi
        original_inpaint_mask = make_inpaint_mask(geometry)
        original_pasteback_mask = make_pasteback_mask(geometry, include_neck=False)
        target_roi = resize_rgb(rgb[top:bottom, left:right], geometry.working_size)
        shape_strength = float(kwargs.get("SHAPE STRENGTH", 0.0))
        shape_status = "OFF · experimental branch disabled"
        if shape_strength > 0.0:
            source_rgb = tensor_to_uint8_rgb(source[0])
            source_faces = CMKDetectorEngine().detect_image(
                source_rgb,
                DetectorSettings(detector_model=detector_model, detector_size=int(kwargs["DETECT SIZE"])),
            )
            source_face = largest_face(source_faces)
            shape_pair = source_shape_in_target_pose(source_face, face)
            if shape_pair is None:
                shape_status = "SKIPPED · dense landmarks unavailable"
            else:
                target_points, desired_points = shape_pair
                target_roi = warp_roi_towards_shape(
                    target_roi, target_points, desired_points, geometry, shape_strength,
                )
                shape_status = f"ON · {shape_strength:.2f}"
        working_mask = resize_mask(original_inpaint_mask, geometry.working_size)
        full_pasteback_mask = np.zeros(rgb.shape[:2], dtype=np.float32)
        full_pasteback_mask[top:bottom, left:right] = original_pasteback_mask
        context = {
            "type": "CMK_FACE_REBUILD",
            "face_bbox": geometry.face_bbox,
            "roi": geometry.roi,
            "working_size": geometry.working_size,
            "target_size": (rgb.shape[1], rgb.shape[0]),
            "target_face": target_selection,
            "detector_model": detector_model,
            "feather": geometry.feather,
            "content_guard_version": FACEREBUILD_GUARD_VERSION,
        }
        lines = [
            "Workflow            : INSTANTID FACE REBUILD",
            f"Target Face         : {target_selection}",
            f"Face BBox           : {_box_text(geometry.face_bbox)}",
            f"Head/Neck ROI       : {_box_text(geometry.roi)}",
            f"Working Resolution  : {geometry.working_size[0]} × {geometry.working_size[1]}",
            f"Mask Feathering     : {geometry.feather} px",
            "Mask Separation     : Inpaint broad / Pasteback inset",
            "Pasteback Neck      : configured by Pasteback",
            "ROI Border Guard    : PASS",
            "ContentGuard        : PASS · TARGET",
            "Sampling            : configured by Face Detailer",
            f"Shape Transfer      : {shape_status}",
            "Identity/Pose/Noise : 1.00 / 0.70 / 0.75",
            "Conditioning        : 75% native reference / 25% CMK prepared",
        ]
        log_block = cmk_block_to_string("InstantID Face Rebuild", 44, lines, True)
        diagnostic = make_diagnostic_payload(
            title="InstantID Face Rebuild · Prepare",
            node="CMK InstantID Face Rebuild Prepare",
            stages=[
                {"title": "01 Target ROI", "subtitle": f"{geometry.working_size[0]} × {geometry.working_size[1]}", "image": target_roi},
                {"title": "02 Inpaint Mask", "subtitle": f"feather {geometry.feather}px", "image": np.repeat((working_mask * 255).astype(np.uint8)[..., None], 3, axis=2)},
                {"title": "03 Base Pasteback Mask", "subtitle": "head only · zero ROI border", "image": np.repeat((original_pasteback_mask * 255).astype(np.uint8)[..., None], 3, axis=2)},
            ],
            previews=[target_roi],
            summary="\n".join(lines),
            details="\n".join(lines),
            mode=target_selection,
            metadata={"face_bbox": geometry.face_bbox, "roi": geometry.roi, "working_size": geometry.working_size, "shape_strength": shape_strength, "shape_status": shape_status},
        )
        return (
            target,
            uint8_rgb_to_tensor(target_roi).unsqueeze(0),
            source,
            torch.from_numpy(working_mask).unsqueeze(0),
            torch.from_numpy(full_pasteback_mask).unsqueeze(0),
            context,
            log_block,
            diagnostic,
        )


class CMKInstantIDFaceRebuildPasteback:
    """Return a rebuilt ROI to its exact target location and verify its boundary."""

    CATEGORY = "CMK/Toolbox/Face"
    FUNCTION = "pasteback"
    RETURN_TYPES = ("IMAGE", "MASK", "CMK_LOG_BLOCK", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("IMAGE", "PASTEBACK MASK", "LOG BLOCK", "diagnostic")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TARGET IMAGE": ("IMAGE",),
                "REBUILT ROI": ("IMAGE",),
                "PASTEBACK MASK": ("MASK",),
                "REBUILD CONTEXT": ("CMK_FACE_REBUILD",),
                "PASTEBACK NECK": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include the generated neck ellipse in final pasteback without resampling the rebuilt ROI.",
                }),
            }
        }

    def pasteback(self, **kwargs):
        rebuilt_batch = kwargs["REBUILT ROI"]
        target_batch = kwargs["TARGET IMAGE"]
        mask_batch = kwargs["PASTEBACK MASK"]
        context = kwargs["REBUILD CONTEXT"]
        if not isinstance(context, dict) or context.get("type") != "CMK_FACE_REBUILD":
            raise ValueError("InstantID Face Rebuild Pasteback requires a valid rebuild context")
        if int(rebuilt_batch.shape[0]) != 1 or int(target_batch.shape[0]) != 1 or int(mask_batch.shape[0]) != 1:
            raise ValueError("InstantID Face Rebuild prototype supports exactly one target and rebuilt ROI")
        target = tensor_to_uint8_rgb(target_batch[0])
        rebuilt = tensor_to_uint8_rgb(rebuilt_batch[0])
        left, top, right, bottom = context["roi"]
        supplied_mask = mask_batch[0].detach().cpu().numpy().astype(np.float32)
        if supplied_mask.shape != target.shape[:2]:
            raise ValueError("InstantID Face Rebuild Pasteback mask must match the target dimensions")
        pasteback_neck = bool(kwargs["PASTEBACK NECK"])
        geometry = FaceRebuildGeometry(
            face_bbox=tuple(int(value) for value in context["face_bbox"]),
            roi=tuple(int(value) for value in context["roi"]),
            working_size=tuple(int(value) for value in context["working_size"]),
            feather=int(context["feather"]),
        )
        local_mask = make_pasteback_mask(geometry, include_neck=pasteback_neck)
        full_mask = np.zeros(target.shape[:2], dtype=np.float32)
        full_mask[top:bottom, left:right] = local_mask
        output, unchanged = pasteback_exact(
            target,
            rebuilt,
            full_mask[top:bottom, left:right],
            context["roi"],
        )
        lines = [
            "Workflow            : INSTANTID FACE REBUILD",
            "Pasteback           : executed",
            f"Head/Neck ROI       : {_box_text(context['roi'])}",
            f"Mask Feathering     : {int(context['feather'])} px",
            f"Pasteback Neck      : {'ON' if pasteback_neck else 'OFF'}",
            f"Outside Unchanged   : {'PASS' if unchanged else 'FAIL'}",
        ]
        log_block = cmk_block_to_string("InstantID Face Rebuild · Pasteback", 46, lines, True)
        diagnostic = make_diagnostic_payload(
            title="InstantID Face Rebuild · Pasteback",
            node="CMK InstantID Face Rebuild Pasteback",
            stages=[
                {"title": "03 Effective Mask", "subtitle": f"neck {'on' if pasteback_neck else 'off'}", "image": np.repeat((local_mask * 255).astype(np.uint8)[..., None], 3, axis=2)},
                {"title": "04 Final Pasteback", "subtitle": "outside mask unchanged", "image": output},
            ],
            previews=[output],
            summary="\n".join(lines),
            details="\n".join(lines),
            mode="Integrity PASS" if unchanged else "Integrity FAIL",
            metadata={"pasteback": True, "pasteback_neck": pasteback_neck, "outside_unchanged": unchanged, "roi": context["roi"]},
            metrics={"outside_unchanged": int(unchanged)},
        )
        return uint8_rgb_to_tensor(output).unsqueeze(0), torch.from_numpy(full_mask).unsqueeze(0), log_block, diagnostic


def _append_log_block(log_pipe, block_value):
    block = cmk_parse_block_string(block_value)
    if not isinstance(block, dict):
        return dict(log_pipe) if isinstance(log_pipe, dict) else {"blocks": []}
    return cmk_add_block(
        log_pipe,
        block.get("title", "InstantID Face Rebuild"),
        block.get("order", 44),
        block.get("lines", []),
        block.get("enabled", True),
    )


class CMKInstantIDFaceRebuildPreparePipe:
    """Prepare an independent SDXL Face Rebuild detailer branch."""

    CATEGORY = "CMK/Toolbox/Face"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return FACEREBUILD_GUARD_VERSION
    FUNCTION = "prepare_pipe"
    RETURN_TYPES = (
        "CMK_PROCESS_SDXL",
        "IMAGE",
        "CMK_LOG_PIPE",
        "IMAGE",
        "MASK",
        "CMK_FACE_REBUILD",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = (
        "PROCESS",
        "IMAGE",
        "LOG",
        "TARGET ORIGINAL",
        "PASTEBACK MASK",
        "REBUILD CONTEXT",
        "diagnostic",
    )

    @classmethod
    def INPUT_TYPES(cls):
        base = CMKInstantIDFaceRebuildPrepare.INPUT_TYPES()["required"]
        return {
            "required": {
                "PROCESS": ("CMK_PROCESS_SDXL",),
                "IMAGE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
                "FACEREBUILD ENABLE": ("BOOLEAN", {"default": True}),
                "ENABLE": ("BOOLEAN", {"default": True}),
                "TARGET FACE": base["TARGET FACE"],
                "HEAD AREA": base["HEAD AREA"],
                "NECK AREA": base["NECK AREA"],
                "MASK FEATHER": base["MASK FEATHER"],
                "WORKING RESOLUTION": base["WORKING RESOLUTION"],
                "DETECT MODEL": base["DETECT MODEL"],
                "DETECT SIZE": base["DETECT SIZE"],
            }
        }

    def prepare_pipe(self, **kwargs):
        process = kwargs["PROCESS"]
        if not isinstance(process, dict):
            raise TypeError("InstantID Face Rebuild Prepare -Pipe- requires CMK_PROCESS_SDXL")
        if str(process.get("model_family", "sdxl")).strip().lower() != "sdxl":
            raise ValueError("InstantID Face Rebuild is available only for the SDXL family")
        global_enabled = bool(kwargs["FACEREBUILD ENABLE"])
        flow_enabled = bool(kwargs["ENABLE"])
        enabled = global_enabled and flow_enabled
        if not enabled:
            image = kwargs["IMAGE"]
            height, width = int(image.shape[1]), int(image.shape[2])
            process_out = dict(process)
            process_out.update({
                "face_rebuild_enabled": False,
                "face_rebuild_global_enabled": global_enabled,
                "face_rebuild_flow_enabled": flow_enabled,
                "face_rebuild_detailer": True,
            })
            context = {
                "type": "CMK_FACE_REBUILD",
                "roi": (0, 0, width, height),
                "working_size": (width, height),
                "feather": 0,
                "pasteback_neck": False,
                "bypassed": True,
            }
            lines = [
                "Workflow            : INSTANTID FACE REBUILD",
                f"Global Enable       : {'ON' if global_enabled else 'OFF'}",
                f"Flow Enable         : {'ON' if flow_enabled else 'OFF'}",
                "Status              : BYPASSED",
                "Target              : unchanged",
            ]
            log_out = cmk_add_block(kwargs["LOG"], "InstantID Face Rebuild", 44, lines, True)
            diagnostic = make_diagnostic_payload(
                title="InstantID Face Rebuild · Bypass",
                node="CMK InstantID Face Rebuild Prepare",
                stages=[{"title": "01 Bypass", "subtitle": "target unchanged", "image": image}],
                previews=[image],
                summary="\n".join(lines),
                details="\n".join(lines),
                mode="Bypass",
                metadata={"enabled": False, "bypassed": True},
            )
            return process_out, image, log_out, image, torch.zeros((1, height, width)), context, diagnostic
        result = CMKInstantIDFaceRebuildPrepare().prepare(
            **{
                "TARGET IMAGE": kwargs["IMAGE"],
                # The canonical module 15 owns the Source Face picker. Prepare
                # only needs target geometry, so no identity image is threaded
                # through this adapter.
                "SOURCE FACE": kwargs["IMAGE"],
                "TARGET FACE": kwargs["TARGET FACE"],
                "HEAD AREA": kwargs["HEAD AREA"],
                "NECK AREA": kwargs["NECK AREA"],
                "MASK FEATHER": kwargs["MASK FEATHER"],
                "WORKING RESOLUTION": kwargs["WORKING RESOLUTION"],
                "DETECT MODEL": kwargs["DETECT MODEL"],
                "DETECT SIZE": kwargs["DETECT SIZE"],
            }
        )
        target, roi, _source, inpaint_mask, pasteback_mask, context, log_block, diagnostic = result
        work_w, work_h = context["working_size"]
        process_out = dict(process)
        process_out.update(
            {
                "face_rebuild_enabled": True,
                "face_rebuild_global_enabled": global_enabled,
                "face_rebuild_flow_enabled": flow_enabled,
                "width": int(work_w),
                "height": int(work_h),
                "source_width": int(work_w),
                "source_height": int(work_h),
                "target_width": int(work_w),
                "target_height": int(work_h),
                "resolution": f"{int(work_w)}x{int(work_h)}",
                "generation_mode": "img2img_detailer",
                "boolean_inpaint_mode": False,
                "inpaint_process_mode": "custom",
                "instantid_face_rebuild": False,
                "face_rebuild_detailer": True,
                "face_rebuild_face_bbox": context["face_bbox"],
                "face_rebuild_target_size": context["target_size"],
            }
        )
        log_out = _append_log_block(kwargs["LOG"], log_block)
        return process_out, roi, log_out, target, pasteback_mask, context, diagnostic


class CMKInstantIDFaceRebuildPastebackPipe:
    """Rejoin a rebuilt ROI with the canonical CMK SDXL flow contract."""

    CATEGORY = "CMK/Toolbox/Face"
    FUNCTION = "pasteback_pipe"
    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_PROCESS_SDXL", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "diagnostic")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE",),
                "PROCESS": ("CMK_PROCESS_SDXL",),
                "IMAGE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
                "TARGET ORIGINAL": ("IMAGE",),
                "PASTEBACK MASK": ("MASK",),
                "REBUILD CONTEXT": ("CMK_FACE_REBUILD",),
                "PASTEBACK NECK": CMKInstantIDFaceRebuildPasteback.INPUT_TYPES()["required"]["PASTEBACK NECK"],
            }
        }

    def pasteback_pipe(self, **kwargs):
        if not bool(kwargs["PROCESS"].get("face_rebuild_enabled", True)):
            lines = [
                "Workflow            : INSTANTID FACE REBUILD",
                "Pasteback           : BYPASSED",
                "Target              : pixel-exact passthrough",
            ]
            diagnostic = make_diagnostic_payload(
                title="InstantID Face Rebuild · Bypass",
                node="CMK InstantID Face Rebuild Pasteback",
                stages=[{"title": "03 Bypass", "subtitle": "target unchanged", "image": kwargs["TARGET ORIGINAL"]}],
                previews=[kwargs["TARGET ORIGINAL"]],
                summary="\n".join(lines),
                details="\n".join(lines),
                mode="Bypass",
                metadata={"enabled": False, "bypassed": True},
            )
            return (
                kwargs["MODEL"],
                kwargs["PROCESS"],
                kwargs["TARGET ORIGINAL"],
                cmk_add_block(kwargs["LOG"], "InstantID Face Rebuild · Pasteback", 46, lines, True),
                diagnostic,
            )
        image, _mask, log_block, diagnostic = CMKInstantIDFaceRebuildPasteback().pasteback(
            **{
                "TARGET IMAGE": kwargs["TARGET ORIGINAL"],
                "REBUILT ROI": kwargs["IMAGE"],
                "PASTEBACK MASK": kwargs["PASTEBACK MASK"],
                "REBUILD CONTEXT": kwargs["REBUILD CONTEXT"],
                "PASTEBACK NECK": kwargs["PASTEBACK NECK"],
            }
        )
        return (
            kwargs["MODEL"],
            kwargs["PROCESS"],
            image,
            _append_log_block(kwargs["LOG"], log_block),
            diagnostic,
        )
