from __future__ import annotations

from ..utils.cmk_diagnostic import make_diagnostic_payload
from ..utils.segs_branch_merge import merge_segs_collections


_MAX_SEGS_INPUTS = 32


class CMKDetailerFinalizePipe:
    """Finalize an isolated CMK detailer module.

    Every Smart Detailer receives the same immutable detailer context and source
    image. This node performs the single final SEGS paste and commits the result
    to a fresh main CMK pipe. The incoming source pipe is never mutated in place.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"segs_{index}": ("SEGS",)
            for index in range(2, _MAX_SEGS_INPUTS + 1)
        }
        optional.update({
            f"log_pipe_{index}": ("STRING", {"forceInput": True})
            for index in range(2, _MAX_SEGS_INPUTS + 1)
        })

        return {
            "required": {
                "detailer_pipe": ("CMK_DETAILER_PIPE",),
                "segs": ("SEGS",),
                "feather": (
                    "INT",
                    {"default": 5, "min": 0, "max": 100, "step": 1, "advanced": True},
                ),
                "alpha": (
                    "INT",
                    {"default": 255, "min": 0, "max": 255, "step": 1, "advanced": True},
                ),
            },
            "optional": {
                "log_pipe": ("STRING", {"forceInput": True}),
                **optional,
            },
        }

    RETURN_TYPES = ("CMK_PIPE", "IMAGE", "IMAGE", "CMK_DIAGNOSTIC", "STRING")
    RETURN_NAMES = ("pipe", "image", "image_proceed", "diagnostic", "log_pipe")
    FUNCTION = "finalize"
    CATEGORY = "CMK/Developer/Pipe/Finalize"

    @staticmethod
    def _extract_image(value):
        if value is None:
            return None
        if hasattr(value, "shape") and len(getattr(value, "shape", ())) == 4:
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                found = CMKDetailerFinalizePipe._extract_image(item)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _valid_segs(value):
        return (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], tuple)
            and len(value[0]) == 2
            and isinstance(value[1], list)
        )

    @staticmethod
    def _segment_count(value):
        return len(value[1]) if CMKDetailerFinalizePipe._valid_segs(value) else 0

    @staticmethod
    def _clean_log(value):
        if value is None:
            return ""
        return str(value).strip()

    def finalize(self, detailer_pipe, segs, feather=5, alpha=255, log_pipe=None, **kwargs):
        if not isinstance(detailer_pipe, dict):
            raise ValueError("CMK Detailer Finalize -Pipe-: detailer_pipe is missing")

        image = detailer_pipe.get("detailer_image")
        if image is None:
            raise ValueError(
                "CMK Detailer Finalize -Pipe-: detailer_image is missing in detailer_pipe. "
                "Recreate the context with CMK Detailer Prepare -Pipe-."
            )

        source_pipe = detailer_pipe.get("source_pipe")
        if not isinstance(source_pipe, dict):
            source_pipe = detailer_pipe.get("_source_pipe")
        if not isinstance(source_pipe, dict):
            raise ValueError(
                "CMK Detailer Finalize -Pipe-: source_pipe is missing in detailer_pipe. "
                "Recreate the context with CMK Detailer Prepare -Pipe-."
            )

        seg_collections = [segs]
        for index in range(2, _MAX_SEGS_INPUTS + 1):
            value = kwargs.get(f"segs_{index}")
            if value is not None:
                seg_collections.append(value)

        valid_collections = []
        used_collections = 0
        total_segments = 0

        for current_segs in seg_collections:
            if not self._valid_segs(current_segs):
                continue
            count = self._segment_count(current_segs)
            total_segments += count
            used_collections += 1
            valid_collections.append(current_segs)

        result_image = merge_segs_collections(
            authoritative_image=image,
            segs_collections=valid_collections,
            feather=int(feather),
            alpha=int(alpha),
        )

        # Commit exactly once into a fresh main pipe. Never mutate the source
        # pipe stored in the isolated detailer context.
        result_pipe = dict(source_pipe)
        result_pipe["image_detailer"] = result_image
        result_pipe["image"] = result_image
        result_pipe["image_final"] = result_image
        result_pipe["detailer_finalized"] = True
        result_pipe["detailer_segment_collections"] = used_collections
        result_pipe["detailer_segments_total"] = total_segments

        incoming_logs = []
        first_log = self._clean_log(log_pipe)
        if first_log:
            incoming_logs.append(first_log)
        for index in range(2, _MAX_SEGS_INPUTS + 1):
            text = self._clean_log(kwargs.get(f"log_pipe_{index}"))
            if text:
                incoming_logs.append(text)

        finalize_log = (
            "CMK Detailer Finalize | "
            f"collections: {used_collections} | segments: {total_segments} | "
            f"feather: {int(feather)} | alpha: {int(alpha)}"
        )
        combined_log = "\n".join([*incoming_logs, finalize_log])
        result_pipe["detailer_log_pipe"] = combined_log

        summary = "\n".join([
            "Status      : Finalized",
            f"Collections : {used_collections}",
            f"Segments    : {total_segments}",
            f"Feather     : {int(feather)}",
            f"Alpha       : {int(alpha)}",
        ])
        diagnostic = make_diagnostic_payload(
            title="Detailer Finalize",
            node="CMK Detailer Finalize -Pipe-",
            previews=(image, result_image),
            stages=(
                {"title": "01 Source", "subtitle": "detailer module input", "image": image},
                {
                    "title": "02 Final",
                    "subtitle": f"{total_segments} segments from {used_collections} collections",
                    "image": result_image,
                },
            ),
            summary=summary,
            details=combined_log,
            mode="Detailer Finalize",
            metadata={
                "collections": used_collections,
                "segments": total_segments,
                "feather": int(feather),
                "alpha": int(alpha),
            },
            metrics={"collections": used_collections, "segments": total_segments},
        )

        return result_pipe, image, result_image, diagnostic, combined_log
