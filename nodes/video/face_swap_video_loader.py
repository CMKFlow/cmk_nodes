from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ...pipe.cmk_log_pipe import cmk_add_block
from ...pipe.loaders.cmk_swap_image_loader import _load_rgb_frames, _resolve_image_path
from .split_video_segments import (
    CMKSplitVideoIntoSegments,
    _PRESETS,
    _VIDEO_CODECS,
    _list_video_files,
)

_MEDIA_DEFAULT = json.dumps(
    {"video": "", "source": ""},
    ensure_ascii=False,
    separators=(",", ":"),
)
_PROJECT_METADATA_NAME = "cmk_video_project.json"


def _write_project_metadata(segments: dict[str, Any], video_name: str, source_name: str) -> None:
    manifest_path = Path(str(segments.get("manifest_path", "") or ""))
    if not manifest_path.is_file():
        return
    path = manifest_path.parent / _PROJECT_METADATA_NAME
    payload = {
        "type": "CMK_VIDEO_PROJECT",
        "version": 1,
        "source_video": video_name,
        "source_image": source_name,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_media_sources(value: Any) -> tuple[str, str]:
    payload: Any = value
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            payload = {}
        else:
            try:
                payload = json.loads(text)
            except Exception as exc:
                raise ValueError("CMK FaceSwap Video Loader: MEDIA SOURCES is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("CMK FaceSwap Video Loader: MEDIA SOURCES is invalid")

    video_name = str(payload.get("video", "") or "").strip()
    source_name = str(payload.get("source", "") or "").strip()
    if not video_name:
        raise ValueError("CMK FaceSwap Video Loader: SOURCE VIDEO is missing")
    if not source_name:
        raise ValueError("CMK FaceSwap Video Loader: SOURCE IMAGE is missing")
    return video_name, source_name


def _source_image_state(image_name: str) -> dict[str, Any]:
    path = Path(_resolve_image_path(image_name)).resolve()
    if not path.is_file():
        raise ValueError(f"CMK FaceSwap Video Loader: SOURCE IMAGE not found: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


class CMKFaceSwapVideoLoader:
    """Closed FaceSwap-video entry node over the universal Split backend.

    Segment creation, manifests and reuse are delegated unchanged to
    CMKSplitVideoIntoSegments. The additional IMAGE SOURCE output is an
    independent native IMAGE transport and never becomes part of
    CMK_VIDEO_SEGMENTS.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MAX FRAMES 720P": (
                    "INT",
                    {"default": 540, "min": 1, "max": 100000, "step": 1},
                ),
                "MAX FRAMES 1080P": (
                    "INT",
                    {"default": 240, "min": 1, "max": 100000, "step": 1},
                ),
                "OVERLAP": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.5},
                ),
                "VIDEO CODEC": (_VIDEO_CODECS, {"default": "libx264"}),
                "VIDEO BITRATE": (
                    "STRING",
                    {"default": "8000k", "multiline": False},
                ),
                "PRESET": (_PRESETS, {"default": "fast"}),
                "MEDIA SOURCES": (
                    "STRING",
                    {"default": _MEDIA_DEFAULT, "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("CMK_VIDEO_SEGMENTS", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("SEGMENTS", "IMAGE SOURCE", "LOG", "diagnostic")
    FUNCTION = "load_and_split"
    CATEGORY = "CMK/Toolbox/Video"
    OUTPUT_NODE = True

    def load_and_split(self, **inputs):
        video_name, source_name = _parse_media_sources(inputs.get("MEDIA SOURCES"))
        source_path = Path(_resolve_image_path(source_name)).resolve()
        if not source_path.is_file():
            raise ValueError(f"CMK FaceSwap Video Loader: SOURCE IMAGE not found: {source_path}")

        source_image, source_width, source_height, _ = _load_rgb_frames(str(source_path))

        split_inputs = {
            "VIDEO": video_name,
            "MAX FRAMES 720P": inputs.get("MAX FRAMES 720P", 540),
            "MAX FRAMES 1080P": inputs.get("MAX FRAMES 1080P", 240),
            "OVERLAP": inputs.get("OVERLAP", 0.0),
            "VIDEO CODEC": inputs.get("VIDEO CODEC", "libx264"),
            "VIDEO BITRATE": inputs.get("VIDEO BITRATE", "8000k"),
            "PRESET": inputs.get("PRESET", "fast"),
        }
        split_result = CMKSplitVideoIntoSegments().split_video(**split_inputs)
        if not isinstance(split_result, dict) or "result" not in split_result:
            raise RuntimeError("CMK FaceSwap Video Loader: Split backend returned an invalid result")

        segments, log_pipe, diagnostic = split_result["result"]
        segments["source_image_name"] = source_name
        segments["source_video_name"] = video_name
        _write_project_metadata(segments, video_name, source_name)
        log_pipe = cmk_add_block(
            log_pipe,
            "FaceSwap Video Loader",
            2,
            [
                f"SOURCE VIDEO      : {video_name}",
                f"SOURCE IMAGE      : {source_name}",
                f"SOURCE IMAGE SIZE : {source_width} × {source_height}",
            ],
            True,
        )

        return {
            "ui": split_result.get("ui", {}),
            "result": (segments, source_image, log_pipe, diagnostic),
        }

    @classmethod
    def IS_CHANGED(cls, **inputs):
        try:
            video_name, source_name = _parse_media_sources(inputs.get("MEDIA SOURCES"))
            split_state = CMKSplitVideoIntoSegments.IS_CHANGED(
                VIDEO=video_name,
                **{
                    "MAX FRAMES 720P": inputs.get("MAX FRAMES 720P", 540),
                    "MAX FRAMES 1080P": inputs.get("MAX FRAMES 1080P", 240),
                    "OVERLAP": inputs.get("OVERLAP", 0.0),
                    "VIDEO CODEC": inputs.get("VIDEO CODEC", "libx264"),
                    "VIDEO BITRATE": inputs.get("VIDEO BITRATE", "8000k"),
                    "PRESET": inputs.get("PRESET", "fast"),
                },
            )
            state = {
                "split": split_state,
                "source_image": _source_image_state(source_name),
            }
            return hashlib.sha256(
                json.dumps(state, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            ).hexdigest()
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **inputs):
        try:
            video_name, source_name = _parse_media_sources(inputs.get("MEDIA SOURCES"))
            _source_image_state(source_name)
            split_validation = CMKSplitVideoIntoSegments.VALIDATE_INPUTS(
                VIDEO=video_name,
                **{
                    "MAX FRAMES 720P": inputs.get("MAX FRAMES 720P", 540),
                    "MAX FRAMES 1080P": inputs.get("MAX FRAMES 1080P", 240),
                    "OVERLAP": inputs.get("OVERLAP", 0.0),
                    "VIDEO CODEC": inputs.get("VIDEO CODEC", "libx264"),
                    "VIDEO BITRATE": inputs.get("VIDEO BITRATE", "8000k"),
                    "PRESET": inputs.get("PRESET", "fast"),
                },
            )
            if split_validation is not True:
                return split_validation
        except Exception as exc:
            return str(exc)
        return True


NODE_CLASS_MAPPINGS = {
    "CMKFaceSwapVideoLoader": CMKFaceSwapVideoLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMKFaceSwapVideoLoader": "CMK FaceSwap Video Loader",
}
