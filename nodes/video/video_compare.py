from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import folder_paths

from ...pipe.cmk_log_pipe import cmk_add_block
from ...utils.cmk_diagnostic import make_diagnostic_payload
from .split_video_segments import _executable, _optional_ffprobe, _probe_video, _run




def _video_ui_descriptor(path: Path) -> dict[str, Any]:
    """Build a valid /view descriptor from the file's actual ComfyUI root."""
    path = path.resolve()
    roots = (
        ("input", Path(folder_paths.get_input_directory()).resolve()),
        ("output", Path(folder_paths.get_output_directory()).resolve()),
        ("temp", Path(folder_paths.get_temp_directory()).resolve()),
    )
    for folder_type, base in roots:
        try:
            relative_parent = path.parent.relative_to(base).as_posix()
            subfolder = "" if relative_parent == "." else relative_parent
            break
        except ValueError:
            continue
    else:
        raise ValueError(
            f"CMK Video Compare: video is outside ComfyUI input/output/temp and cannot be previewed: {path}"
        )
    extension = path.suffix.lower().lstrip(".") or "mp4"
    return {
        "filename": path.name,
        "subfolder": subfolder,
        "type": folder_type,
        "format": f"video/{extension}",
    }

def _normalize_segments(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "CMK_VIDEO_SEGMENTS":
        raise TypeError("CMK Video Compare: SEGMENTS is not CMK_VIDEO_SEGMENTS")
    source_path = Path(str(value.get("source_path", "") or "")).resolve()
    if not source_path.is_file():
        raise ValueError(f"CMK Video Compare: source video is missing: {source_path}")
    return dict(value)


def _normalize_video(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "CMK_VIDEO":
        raise TypeError("CMK Video Compare: VIDEO is not CMK_VIDEO")
    result_path = Path(str(value.get("video_path", "") or "")).resolve()
    if not result_path.is_file():
        raise ValueError(f"CMK Video Compare: result video is missing: {result_path}")
    return dict(value)


def _fraction(value: Any) -> float:
    text = str(value or "0").strip()
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        return float(text)
    except Exception:
        return 0.0


def _ffprobe_details(path: Path) -> dict[str, Any]:
    ffprobe = _optional_ffprobe()
    if not ffprobe:
        return {}
    result = _run(
        [
            ffprobe,
            "-v", "error",
            "-show_streams",
            "-show_format",
            "-of", "json",
            str(path),
        ],
        capture_stdout=True,
    )
    try:
        payload = json.loads(result.stdout.decode("utf-8", errors="replace"))
    except Exception:
        return {}
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    video = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"), {})
    fmt = payload.get("format", {}) if isinstance(payload.get("format"), dict) else {}

    def number(value: Any) -> float:
        try:
            out = float(value)
            return out if math.isfinite(out) else 0.0
        except Exception:
            return 0.0

    audio_duration = number(audio.get("duration")) or number(fmt.get("duration")) if audio else 0.0
    return {
        "video_codec": str(video.get("codec_name") or "unknown"),
        "audio_codec": str(audio.get("codec_name") or "none") if audio else "none",
        "audio_duration": float(audio_duration),
        "has_audio": bool(audio),
        "exact_frame_count": int(video.get("nb_frames") or 0) if str(video.get("nb_frames") or "").isdigit() else 0,
        "avg_frame_rate": _fraction(video.get("avg_frame_rate")),
    }



def _ffmpeg_stream_details(path: Path) -> dict[str, Any]:
    ffmpeg = _executable("ffmpeg")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    text = result.stderr.decode("utf-8", errors="replace")
    video_match = re.search(r"Stream #.*?Video:\s*([^,\s]+)", text)
    audio_match = re.search(r"Stream #.*?Audio:\s*([^,\s]+)", text)
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    duration = 0.0
    if duration_match:
        duration = int(duration_match.group(1)) * 3600 + int(duration_match.group(2)) * 60 + float(duration_match.group(3))
    return {
        "video_codec": video_match.group(1) if video_match else "unknown",
        "audio_codec": audio_match.group(1) if audio_match else "none",
        "audio_duration": float(duration) if audio_match else 0.0,
        "has_audio": bool(audio_match),
    }

def _probe(path: Path) -> dict[str, Any]:
    base = dict(_probe_video(path))
    extra = _ffprobe_details(path)
    if not extra:
        extra = _ffmpeg_stream_details(path)
    base.update({key: value for key, value in extra.items() if value not in (None, "")})
    if int(base.get("exact_frame_count", 0) or 0) > 0:
        base["frame_count"] = int(base["exact_frame_count"])
    return base


def _severity_rank(value: str) -> int:
    return {"PASS": 0, "WARNING": 1, "FAIL": 2}.get(str(value), 2)


def _metric_status(*, pass_condition: bool, warning_condition: bool) -> str:
    if pass_condition:
        return "PASS"
    if warning_condition:
        return "WARNING"
    return "FAIL"


def _timeline(segments: dict[str, Any], video: dict[str, Any]) -> list[dict[str, Any]]:
    items = [dict(item) for item in segments.get("segments", []) if isinstance(item, dict)]
    trims = list(video.get("trim_starts", []) or [])
    output = []
    previous_end = None
    merged_cursor = 0.0
    for index, item in enumerate(items):
        start = float(item.get("start", 0.0) or 0.0)
        end = float(item.get("end", start) or start)
        trim = float(trims[index]) if index < len(trims) else (0.0 if previous_end is None else max(0.0, float(previous_end) - start))
        used_duration = max(0.0, (end - start) - trim)
        output.append(
            {
                "index": index,
                "source_start": start,
                "source_end": end,
                "trim_start": trim,
                "merged_start": merged_cursor,
                "merged_end": merged_cursor + used_duration,
            }
        )
        merged_cursor += used_duration
        previous_end = end
    return output


class CMKVideoCompare:
    """Objectively validate Original → Split → Merge roundtrips."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SEGMENTS": ("CMK_VIDEO_SEGMENTS",),
                "VIDEO": ("CMK_VIDEO",),
                "LOG": ("CMK_LOG_PIPE",),
            }
        }

    RETURN_TYPES = ("CMK_VIDEO_METRICS", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("METRICS", "LOG", "diagnostic")
    FUNCTION = "compare"
    CATEGORY = "CMK/Toolbox/Video"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, SEGMENTS, VIDEO, LOG):
        """Include effective preview files in the cache key so UI descriptors are resent."""
        try:
            source = Path(str(SEGMENTS.get("compare_source_path") or SEGMENTS.get("source_path") or "")).resolve()
            result = Path(str(VIDEO.get("video_path") or "")).resolve()
            state = []
            for path in (source, result):
                stat = path.stat()
                state.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
            return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()
        except Exception:
            return float("nan")

    def compare(self, SEGMENTS, VIDEO, LOG):
        segments = _normalize_segments(SEGMENTS)
        video = _normalize_video(VIDEO)
        if not isinstance(LOG, dict):
            raise TypeError("CMK Video Compare: LOG is not a CMK log pipe")

        source_path = Path(str(segments.get("compare_source_path") or segments.get("source_path"))).resolve()
        result_path = Path(str(video.get("video_path"))).resolve()
        source = _probe(source_path)
        result = _probe(result_path)

        source_fps = float(source.get("fps", 0.0) or 0.0)
        result_fps = float(result.get("fps", 0.0) or 0.0)
        source_duration = float(source.get("duration", 0.0) or 0.0)
        result_duration = float(result.get("duration", 0.0) or 0.0)
        source_frames = int(source.get("frame_count", 0) or 0)
        result_frames = int(result.get("frame_count", 0) or 0)
        duration_delta = result_duration - source_duration
        frame_delta = result_frames - source_frames
        fps_delta = result_fps - source_fps
        frame_time = 1.0 / source_fps if source_fps > 0 else 0.04

        resolution_match = (
            int(source.get("width", 0)) == int(result.get("width", 0))
            and int(source.get("height", 0)) == int(result.get("height", 0))
        )
        fps_status = _metric_status(
            pass_condition=abs(fps_delta) <= 0.001,
            warning_condition=abs(fps_delta) <= 0.05,
        )
        # Roundtrip comparison must account for container timestamp and GOP
        # rounding. Use the source FPS as the authoritative tolerance basis:
        # PASS through 2 frames, WARNING through 6 frames, FAIL beyond that.
        duration_status = _metric_status(
            pass_condition=abs(duration_delta) <= max(frame_time * 2.0, 0.01),
            warning_condition=abs(duration_delta) <= max(frame_time * 6.0, 0.05),
        )
        frame_status = _metric_status(
            pass_condition=abs(frame_delta) <= 2,
            warning_condition=abs(frame_delta) <= 6,
        )
        resolution_status = "PASS" if resolution_match else "FAIL"

        source_has_audio = source.get("has_audio")
        result_has_audio = result.get("has_audio")
        audio_duration_source = float(source.get("audio_duration", 0.0) or 0.0)
        audio_duration_result = float(result.get("audio_duration", 0.0) or 0.0)
        audio_duration_delta = audio_duration_result - audio_duration_source
        if source_has_audio is None or result_has_audio is None:
            audio_status = "WARNING"
            audio_reason = "audio metadata unavailable without ffprobe"
        elif bool(source_has_audio) != bool(result_has_audio):
            audio_status = "FAIL"
            audio_reason = "audio stream presence differs"
        elif not bool(source_has_audio):
            audio_status = "PASS"
            audio_reason = "both videos have no audio"
        else:
            audio_status = _metric_status(
                pass_condition=abs(audio_duration_delta) <= max(frame_time * 2.0, 0.01),
                warning_condition=abs(audio_duration_delta) <= max(frame_time * 6.0, 0.05),
            )
            audio_reason = f"audio duration delta {audio_duration_delta:+.4f}s"

        statuses = [resolution_status, fps_status, duration_status, frame_status, audio_status]
        overall = max(statuses, key=_severity_rank)
        timeline = _timeline(segments, video)

        metrics = {
            "type": "CMK_VIDEO_METRICS",
            "version": 1,
            "status": overall,
            "source_path": str(source_path),
            "result_path": str(result_path),
            "source": source,
            "result": result,
            "resolution_match": resolution_match,
            "resolution_status": resolution_status,
            "fps_delta": fps_delta,
            "fps_status": fps_status,
            "duration_delta": duration_delta,
            "duration_status": duration_status,
            "frame_delta": frame_delta,
            "frame_status": frame_status,
            "audio_duration_delta": audio_duration_delta,
            "audio_status": audio_status,
            "audio_reason": audio_reason,
            "timeline": tuple(timeline),
            "segments_compared": len(timeline),
        }

        log_lines = [
            f"STATUS             : {overall}",
            f"SOURCE             : {source_path}",
            f"RESULT             : {result_path}",
            f"RESOLUTION         : {source.get('width')}×{source.get('height')} → {result.get('width')}×{result.get('height')} [{resolution_status}]",
            f"FPS                : {source_fps:.6f} → {result_fps:.6f} / Δ {fps_delta:+.6f} [{fps_status}]",
            f"DURATION           : {source_duration:.4f}s → {result_duration:.4f}s / Δ {duration_delta:+.4f}s [{duration_status}]",
            f"FRAMES             : {source_frames} → {result_frames} / Δ {frame_delta:+d} [{frame_status}]",
            f"AUDIO              : {audio_reason} [{audio_status}]",
            f"SEGMENTS           : {len(timeline)}",
        ]
        log_pipe = cmk_add_block(LOG, "Video Compare", 100, log_lines, True)

        timeline_lines = [
            (
                f"segment {item['index']:03d}: source {item['source_start']:.3f}–{item['source_end']:.3f}s | "
                f"trim {item['trim_start']:.3f}s | merged {item['merged_start']:.3f}–{item['merged_end']:.3f}s"
            )
            for item in timeline
        ]
        summary = "\n".join(log_lines)
        details = summary + "\n\nSEGMENT TIMELINE\n----------------\n" + ("\n".join(timeline_lines) if timeline_lines else "No timeline data")
        diagnostic = make_diagnostic_payload(
            title="Video Compare",
            node="CMK Video Compare",
            previews=[],
            stages=[],
            summary=summary,
            details=details,
            mode="Original vs merged roundtrip",
            metadata=metrics,
            metrics={
                "status": overall,
                "duration_delta": duration_delta,
                "frame_delta": frame_delta,
                "fps_delta": fps_delta,
                "audio_duration_delta": audio_duration_delta,
            },
        )
        return {
            "ui": {
                "cmk_compare_videos": [
                    _video_ui_descriptor(source_path),
                    _video_ui_descriptor(result_path),
                ]
            },
            "result": (metrics, log_pipe, diagnostic),
        }


NODE_CLASS_MAPPINGS = {"CMKVideoCompare": CMKVideoCompare}
NODE_DISPLAY_NAME_MAPPINGS = {"CMKVideoCompare": "CMK Video Compare"}
