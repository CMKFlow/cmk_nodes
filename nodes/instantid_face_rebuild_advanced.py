from __future__ import annotations

from .instantid_face_detailer import CMKInstantIDFaceDetailerSDXL
from .instantid_face_rebuild import (
    CMKInstantIDFaceRebuildPasteback,
    CMKInstantIDFaceRebuildPrepare,
    FACEREBUILD_GUARD_VERSION,
    TARGET_FACE_MODES,
    select_target_face,
)
from ..engine.detector_engine import CMKDetectorEngine, DetectorSettings
from .utils.diagnostic_concat import CMKDiagnosticConcat
from ..pipe.cmk_log_pipe import cmk_add_block, cmk_parse_block_string
from ..utils.cmk_diagnostic import make_diagnostic_payload
from ..utils.tensor_utils import tensor_to_uint8_rgb


def _append_structured_log_block(log_pipe, block_value, title_prefix=None):
    """Append a serialized CMK log block without exposing its wire format."""
    block = cmk_parse_block_string(block_value)
    if not isinstance(block, dict):
        return cmk_add_block(
            log_pipe, title_prefix or "FaceRebuild", 46,
            [str(block_value)], True,
        )
    title = str(block.get("title", "FaceRebuild"))
    if title_prefix:
        title = f"{title_prefix} · {title}"
    return cmk_add_block(
        log_pipe,
        title,
        int(block.get("order", 46)),
        block.get("lines", []),
        bool(block.get("enabled", True)),
    )


class CMKInstantIDFaceRebuildAdvancedSDXL:
    """One sequential executor for up to three independently configured faces."""

    CATEGORY = "CMK/Toolbox/Face"
    DIAGNOSTIC_NAME = "CMK Flow · 42 FaceRebuild Advanced"
    LOG_LABEL = "FaceRebuild Advanced"
    FUNCTION = "rebuild"
    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_PROCESS_SDXL", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "diagnostic")

    def __init__(self):
        self._advanced_cache_key = None
        self._advanced_cache_value = None

    @staticmethod
    def _advanced_cache_token(value):
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        data_ptr = getattr(value, "data_ptr", None)
        if callable(data_ptr):
            try:
                return (id(value), int(data_ptr()), tuple(value.shape), getattr(value, "_version", None))
            except (RuntimeError, TypeError):
                pass
        return id(value)

    def _advanced_sampling_key(self, values):
        return (("content_guard_version", FACEREBUILD_GUARD_VERSION),) + tuple(
            (name, self._advanced_cache_token(value))
            for name, value in sorted(values.items())
            if not name.startswith("PASTEBACK NECK ")
        )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "MODEL": ("CMK_MODEL_PIPE",), "PROCESS": ("CMK_PROCESS_SDXL",),
            "IMAGE": ("IMAGE",), "LOG": ("CMK_LOG_PIPE",),
            "FACEREBUILD ENABLE": ("BOOLEAN", {"default": True}),
        }
        defaults = ("Leftmost", "Center", "Rightmost")
        for index, default in enumerate(defaults, start=1):
            required.update({
                f"FACE {index} ENABLE": ("BOOLEAN", {"default": index == 1}),
                f"TARGET FACE {index}": (TARGET_FACE_MODES, {"default": default}),
                f"SOURCE FACE {index}": ("IMAGE",),
                f"PASTEBACK NECK {index}": ("BOOLEAN", {"default": False}),
                f"SAMPLING START {index}": ("INT", {
                    "default": 0, "min": -2, "max": 2, "step": 1,
                    "tooltip": "SOURCE (-2) to TARGET (+2). Zero selects the face-size balance automatically.",
                }),
                f"IDENTITY STRENGTH {index}": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                f"POSE STRENGTH {index}": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            })
        required.update({
            "TOTAL STEPS": ("INT", {"default": 20, "min": 1, "max": 200}),
            "HEAD AREA": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05}),
            "NECK AREA": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
            "MASK FEATHER": ("INT", {"default": 32, "min": 0, "max": 64}),
            "WORKING RESOLUTION": (["768", "1024", "1280"], {"default": "1024"}),
            "DETECT MODEL": CMKInstantIDFaceRebuildPrepare.INPUT_TYPES()["required"]["DETECT MODEL"],
            "DETECT SIZE": CMKInstantIDFaceRebuildPrepare.INPUT_TYPES()["required"]["DETECT SIZE"],
            "instantid_model": CMKInstantIDFaceDetailerSDXL.INPUT_TYPES()["required"]["instantid_model"],
            "controlnet_model": CMKInstantIDFaceDetailerSDXL.INPUT_TYPES()["required"]["controlnet_model"],
            "provider": CMKInstantIDFaceDetailerSDXL.INPUT_TYPES()["required"]["provider"],
            "cfg": CMKInstantIDFaceDetailerSDXL.INPUT_TYPES()["required"]["cfg"],
            "identity_noise": CMKInstantIDFaceDetailerSDXL.INPUT_TYPES()["required"]["identity_noise"],
            "sampler": CMKInstantIDFaceDetailerSDXL.INPUT_TYPES()["required"]["sampler"],
            "scheduler": CMKInstantIDFaceDetailerSDXL.INPUT_TYPES()["required"]["scheduler"],
        })
        return {"required": required}

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return FACEREBUILD_GUARD_VERSION

    def rebuild(self, **values):
        model, process, image, log = values["MODEL"], values["PROCESS"], values["IMAGE"], values["LOG"]
        active = [i for i in range(1, 4) if values[f"FACE {i} ENABLE"]]
        if not values["FACEREBUILD ENABLE"] or not active:
            diagnostic = make_diagnostic_payload(
                title=f"{self.DIAGNOSTIC_NAME} · Bypass", node=self.__class__.__name__,
                stages=[{"title": "Bypass", "subtitle": "target unchanged", "image": image}],
                previews=[image], summary="No active FaceRebuild slot", details="No active FaceRebuild slot",
                mode="Bypass", metadata={"bypassed": True},
            )
            return model, process, image, log, diagnostic

        cache_key = self._advanced_sampling_key(values)
        cached = self._advanced_cache_value if cache_key == self._advanced_cache_key else None
        if cached is None:
            # Resolve selectors once against the unchanged target. Slot numbers are
            # configuration containers; larger faces are sampled first,
            # independently of the final neck pasteback mode.
            rgb = tensor_to_uint8_rgb(image[0])
            faces = CMKDetectorEngine().detect_image(
                rgb,
                DetectorSettings(
                    detector_model=str(values["DETECT MODEL"]),
                    detector_size=int(values["DETECT SIZE"]),
                ),
            )
            resolved = []
            for index in active:
                face = select_target_face(
                    faces, str(values[f"TARGET FACE {index}"]), rgb.shape[1], rgb.shape[0],
                )
                raw_bbox = face.get("bbox") if isinstance(face, dict) else getattr(face, "bbox", None)
                bbox = tuple(int(round(float(value))) for value in raw_bbox[:4])
                resolved.append((index, bbox))
            bboxes = [bbox for _index, bbox in resolved]
            if len(set(bboxes)) != len(bboxes):
                raise ValueError(
                    "FaceRebuild Advanced: two active slots resolve to the same target face; "
                    "choose a different TARGET FACE in one of those slots"
                )
            resolved.sort(
                key=lambda item: -max(0, item[1][2] - item[1][0]) * max(0, item[1][3] - item[1][1])
            )

            stages = []
            sampling_image = image
            for index, _bbox in resolved:
                absolute_key = f"SAMPLING START ABSOLUTE {index}"
                if absolute_key in values:
                    sampling_start = max(
                        0,
                        min(int(values["TOTAL STEPS"]) - 1, int(values[absolute_key])),
                    )
                else:
                    total_steps = max(1, int(values["TOTAL STEPS"]))
                    neutral_start = int(round(total_steps * 0.5))
                    sampling_start = max(
                        0,
                        min(
                            total_steps - 1,
                            neutral_start + int(values[f"SAMPLING START {index}"]),
                        ),
                    )
                prepared = CMKInstantIDFaceRebuildPrepare().prepare(**{
                    "TARGET IMAGE": sampling_image, "SOURCE FACE": values[f"SOURCE FACE {index}"],
                    "TARGET FACE": values[f"TARGET FACE {index}"], "HEAD AREA": values["HEAD AREA"],
                    "NECK AREA": values["NECK AREA"], "MASK FEATHER": values["MASK FEATHER"],
                    "WORKING RESOLUTION": values["WORKING RESOLUTION"], "DETECT MODEL": values["DETECT MODEL"],
                    "DETECT SIZE": values["DETECT SIZE"],
                    # Experimental landmark warp is deliberately suspended.
                    # SOURCE/TARGET controls sampling only during this test phase.
                    "SHAPE STRENGTH": 0.0,
                })
                target, roi, source, _inpaint, mask, context, block, prepare_diag = prepared
                process_slot = dict(
                    process,
                    face_rebuild_enabled=True,
                    face_rebuild_face_bbox=context["face_bbox"],
                    face_rebuild_target_size=context["target_size"],
                )
                detailed = CMKInstantIDFaceDetailerSDXL().detail(**{
                    "MODEL": model, "PROCESS": process_slot, "TARGET ROI": roi, "SOURCE FACE": source,
                    "LOG": log, "TOTAL STEPS": values["TOTAL STEPS"],
                    "SAMPLING START": sampling_start,
                    "IDENTITY STRENGTH": values[f"IDENTITY STRENGTH {index}"],
                    "POSE STRENGTH": values[f"POSE STRENGTH {index}"],
                    **{name: values[name] for name in ("instantid_model", "controlnet_model", "provider", "cfg", "identity_noise", "sampler", "scheduler")},
                })
                model, process_slot, rebuilt, log, detail_diag = detailed
                stages.append((index, rebuilt, mask, context, block, prepare_diag, detail_diag))
                # A deterministic head-only interim image feeds subsequent slots;
                # user-selectable neck pasteback is replayed only after sampling.
                sampling_image = CMKInstantIDFaceRebuildPasteback().pasteback(**{
                    "TARGET IMAGE": sampling_image, "REBUILT ROI": rebuilt,
                    "PASTEBACK MASK": mask, "REBUILD CONTEXT": context,
                    "PASTEBACK NECK": False,
                })[0]
                process = process_slot
            cached = (model, process, log, stages)
            self._advanced_cache_key = cache_key
            self._advanced_cache_value = cached

        model, process, detail_log, stages = cached
        image = values["IMAGE"]
        log = detail_log
        diagnostics = []
        for index, rebuilt, mask, context, block, prepare_diag, detail_diag in stages:
            image, _mask, paste_block, paste_diag = CMKInstantIDFaceRebuildPasteback().pasteback(**{
                "TARGET IMAGE": image, "REBUILT ROI": rebuilt, "PASTEBACK MASK": mask,
                "REBUILD CONTEXT": context, "PASTEBACK NECK": values[f"PASTEBACK NECK {index}"],
            })
            prefix = f"{self.LOG_LABEL} · Slot {index}"
            log = _append_structured_log_block(log, block, prefix)
            log = _append_structured_log_block(log, paste_block, prefix)
            diagnostics.extend((prepare_diag, detail_diag, paste_diag))

        diagnostic = CMKDiagnosticConcat().concat(self.DIAGNOSTIC_NAME, diagnostics[0], **{
            f"diagnostic_{i}": item for i, item in enumerate(diagnostics[1:], start=2)
        })[0]
        return model, process, image, log, diagnostic


class CMKInstantIDFaceRebuildSDXL:
    """Advanced FaceRebuild copied with face branches two and three removed."""

    CATEGORY = "CMK/Toolbox/Face"
    DIAGNOSTIC_NAME = "CMK Flow · 25 FaceRebuild SDXL"
    LOG_LABEL = "FaceRebuild"
    FUNCTION = "rebuild"
    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_PROCESS_SDXL", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "diagnostic")

    def __init__(self):
        self._cache_key = None
        self._cache_value = None

    @staticmethod
    def _cache_token(value):
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        data_ptr = getattr(value, "data_ptr", None)
        if callable(data_ptr):
            try:
                return (id(value), int(data_ptr()), tuple(value.shape), getattr(value, "_version", None))
            except (RuntimeError, TypeError):
                pass
        return id(value)

    def _sampling_key(self, values):
        return (("content_guard_version", FACEREBUILD_GUARD_VERSION),) + tuple(
            (name, self._cache_token(value))
            for name, value in sorted(values.items())
            if name != "PASTEBACK NECK"
        )

    @classmethod
    def INPUT_TYPES(cls):
        advanced = CMKInstantIDFaceRebuildAdvancedSDXL.INPUT_TYPES()["required"]
        required = {
            "MODEL": advanced["MODEL"],
            "PROCESS": advanced["PROCESS"],
            "IMAGE": advanced["IMAGE"],
            "LOG": advanced["LOG"],
            "FACEREBUILD ENABLE": advanced["FACEREBUILD ENABLE"],
            "TARGET FACE": advanced["TARGET FACE 1"],
            "SOURCE FACE": advanced["SOURCE FACE 1"],
            "PASTEBACK NECK": advanced["PASTEBACK NECK 1"],
            "SAMPLING START": advanced["SAMPLING START 1"],
            "IDENTITY STRENGTH": advanced["IDENTITY STRENGTH 1"],
            "POSE STRENGTH": advanced["POSE STRENGTH 1"],
        }
        for name in (
            "TOTAL STEPS", "HEAD AREA", "NECK AREA", "MASK FEATHER",
            "WORKING RESOLUTION", "DETECT MODEL", "DETECT SIZE",
            "instantid_model", "controlnet_model", "provider", "cfg",
            "identity_noise", "sampler", "scheduler",
        ):
            required[name] = advanced[name]
        return {"required": required}

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return FACEREBUILD_GUARD_VERSION

    def rebuild(self, **values):
        model, process, image, log = values["MODEL"], values["PROCESS"], values["IMAGE"], values["LOG"]
        if not values["FACEREBUILD ENABLE"]:
            diagnostic = make_diagnostic_payload(
                title=f"{self.DIAGNOSTIC_NAME} · Bypass", node=self.__class__.__name__,
                stages=[{"title": "Bypass", "subtitle": "target unchanged", "image": image}],
                previews=[image], summary="FaceRebuild disabled", details="FaceRebuild disabled",
                mode="Bypass", metadata={"bypassed": True},
            )
            return model, process, image, log, diagnostic

        cache_key = self._sampling_key(values)
        cached = self._cache_value if cache_key == self._cache_key else None
        if cached is None:
            total_steps = max(1, int(values["TOTAL STEPS"]))
            neutral_start = int(round(total_steps * 0.5))
            sampling_start = max(0, min(
                total_steps - 1,
                neutral_start + int(values["SAMPLING START"]),
            ))
            prepared = CMKInstantIDFaceRebuildPrepare().prepare(**{
                "TARGET IMAGE": image, "SOURCE FACE": values["SOURCE FACE"],
                "TARGET FACE": values["TARGET FACE"], "HEAD AREA": values["HEAD AREA"],
                "NECK AREA": values["NECK AREA"], "MASK FEATHER": values["MASK FEATHER"],
                "WORKING RESOLUTION": values["WORKING RESOLUTION"], "DETECT MODEL": values["DETECT MODEL"],
                "DETECT SIZE": values["DETECT SIZE"], "SHAPE STRENGTH": 0.0,
            })
            target, roi, source, _inpaint, mask, context, block, prepare_diag = prepared
            process_out = dict(
                process,
                face_rebuild_enabled=True,
                face_rebuild_face_bbox=context["face_bbox"],
                face_rebuild_target_size=context["target_size"],
            )
            detailed = CMKInstantIDFaceDetailerSDXL().detail(**{
                "MODEL": model, "PROCESS": process_out, "TARGET ROI": roi, "SOURCE FACE": source,
                "LOG": log, "TOTAL STEPS": values["TOTAL STEPS"], "SAMPLING START": sampling_start,
                "IDENTITY STRENGTH": values["IDENTITY STRENGTH"],
                "POSE STRENGTH": values["POSE STRENGTH"],
                **{name: values[name] for name in (
                    "instantid_model", "controlnet_model", "provider", "cfg",
                    "identity_noise", "sampler", "scheduler",
                )},
            })
            model, process_out, rebuilt, detail_log, detail_diag = detailed
            cached = (
                model, process_out, rebuilt, detail_log, mask, context,
                block, prepare_diag, detail_diag,
            )
            self._cache_key = cache_key
            self._cache_value = cached

        model, process, rebuilt, detail_log, mask, context, block, prepare_diag, detail_diag = cached
        image, _mask, paste_block, paste_diag = CMKInstantIDFaceRebuildPasteback().pasteback(**{
            "TARGET IMAGE": values["IMAGE"], "REBUILT ROI": rebuilt,
            "PASTEBACK MASK": mask, "REBUILD CONTEXT": context,
            "PASTEBACK NECK": values["PASTEBACK NECK"],
        })
        log = _append_structured_log_block(detail_log, block, self.LOG_LABEL)
        log = _append_structured_log_block(log, paste_block, self.LOG_LABEL)
        diagnostics = (prepare_diag, detail_diag, paste_diag)
        diagnostic = CMKDiagnosticConcat().concat(self.DIAGNOSTIC_NAME, diagnostics[0], **{
            f"diagnostic_{i}": item for i, item in enumerate(diagnostics[1:], start=2)
        })[0]
        return model, process, image, log, diagnostic
