from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import folder_paths

from ...pipe.cmk_log_pipe import cmk_add_block
from ...utils.cmk_diagnostic import make_diagnostic_payload


_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
}
_VIDEO_CODECS = ["libx264", "hevc"]
_PRESETS = [
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
]
_MANIFEST_NAME = "segments.json"
_SCHEMA = "cmk_video_segments_v1"
_MIN_SEGMENT_SECONDS = 3.0
_MIN_SEGMENT_FRAMES = 180
_AUDIO_CODEC = "aac"
_AUDIO_BITRATE = "192k"


def _input_root() -> Path:
    return Path(folder_paths.get_input_directory()).resolve()


def _video_root() -> Path:
    return (_input_root() / "video").resolve()


def _output_root() -> Path:
    return (Path(folder_paths.get_output_directory()).resolve() / "video" / "segments")


def _list_video_files() -> list[str]:
    input_root = _input_root()
    video_root = _video_root()
    if not video_root.exists():
        return [""]

    files: list[str] = []
    for path in video_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _VIDEO_EXTENSIONS:
            continue
        try:
            files.append(path.resolve().relative_to(input_root).as_posix())
        except Exception:
            continue
    return sorted(files, key=str.casefold) or [""]


def _resolve_video_path(value: str) -> Path:
    name = str(value or "").strip()
    if not name:
        raise ValueError("CMK Split Video into Segments: VIDEO is missing")

    try:
        candidate = Path(folder_paths.get_annotated_filepath(name)).resolve()
    except Exception:
        candidate = (_input_root() / name).resolve()

    input_root = _input_root()
    try:
        candidate.relative_to(input_root)
    except ValueError as exc:
        raise ValueError(
            "CMK Split Video into Segments: VIDEO must be inside the ComfyUI input directory"
        ) from exc

    if not candidate.is_file():
        raise ValueError(f"CMK Split Video into Segments: video file not found: {candidate}")
    if candidate.suffix.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(
            f"CMK Split Video into Segments: unsupported video extension: {candidate.suffix}"
        )
    return candidate


def _executable(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path

    # Desktop applications on macOS commonly inherit a minimal PATH which does
    # not include Homebrew or MacPorts, even though the executable is installed.
    # Check the conventional absolute locations before falling back to a Python
    # package that bundles FFmpeg.
    executable_name = f"{name}.exe" if os.name == "nt" else name
    conventional_dirs = (
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/opt/local/bin",
        "/usr/bin",
        "/bin",
    )
    for directory in conventional_dirs:
        candidate = Path(directory) / executable_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            path = str(imageio_ffmpeg.get_ffmpeg_exe())
            if path and Path(path).is_file():
                return path
        except Exception:
            pass

    raise RuntimeError(
        f"CMK Split Video into Segments: required executable '{name}' was not found"
    )


def _optional_ffprobe() -> str | None:
    """Return ffprobe when available, otherwise use the FFmpeg metadata fallback."""
    path = shutil.which("ffprobe")
    if path:
        return path

    # Some installations keep ffprobe next to their explicitly bundled ffmpeg.
    try:
        ffmpeg_path = Path(_executable("ffmpeg"))
        candidates = [
            ffmpeg_path.with_name("ffprobe"),
            ffmpeg_path.with_name("ffprobe.exe"),
        ]
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    except Exception:
        pass
    return None


def _run(command: list[str], *, capture_stdout: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        if len(stderr) > 4000:
            stderr = stderr[-4000:]
        raise RuntimeError(
            "CMK Split Video into Segments: ffmpeg/ffprobe failed"
            + (f"\n{stderr}" if stderr else "")
        )
    return result


def _fraction(value: Any) -> float:
    text = str(value or "").strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        except Exception:
            return 0.0
    try:
        return float(text)
    except Exception:
        return 0.0


def _number(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return 0.0
    return result if math.isfinite(result) else 0.0


def _rotation(stream: dict[str, Any]) -> int:
    tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}
    try:
        value = int(round(float(tags.get("rotate", 0))))
        if value:
            return value % 360
    except Exception:
        pass

    for item in stream.get("side_data_list", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            value = int(round(float(item.get("rotation", 0))))
            if value:
                return value % 360
        except Exception:
            continue
    return 0


def _probe_video_ffprobe(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    result = _run(command, capture_stdout=True)
    try:
        payload = json.loads(result.stdout.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError(
            "CMK Split Video into Segments: ffprobe returned invalid metadata"
        ) from exc

    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    video_stream = next(
        (item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"),
        None,
    )
    if not isinstance(video_stream, dict):
        raise ValueError("CMK Split Video into Segments: no video stream found")

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    rotation = _rotation(video_stream)
    if rotation in {90, 270}:
        width, height = height, width

    fps = _fraction(video_stream.get("avg_frame_rate"))
    if fps <= 0:
        fps = _fraction(video_stream.get("r_frame_rate"))
    if fps <= 0:
        raise ValueError("CMK Split Video into Segments: video FPS could not be determined")

    format_info = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    duration = _number(video_stream.get("duration")) or _number(format_info.get("duration"))
    if duration <= 0:
        raise ValueError("CMK Split Video into Segments: video duration could not be determined")

    frame_count_raw = str(video_stream.get("nb_frames", "") or "").strip()
    try:
        frame_count = int(frame_count_raw) if frame_count_raw not in {"", "N/A"} else 0
    except Exception:
        frame_count = 0
    if frame_count <= 0:
        frame_count = max(1, int(round(duration * fps)))

    return {
        "width": width,
        "height": height,
        "fps": float(fps),
        "duration": float(duration),
        "frame_count": int(frame_count),
        "has_audio": any(
            isinstance(item, dict) and item.get("codec_type") == "audio"
            for item in streams
        ),
        "source_codec": str(video_stream.get("codec_name") or "unknown"),
        "rotation": int(rotation),
        "probe_backend": "ffprobe",
    }


def _probe_video_ffmpeg(path: Path) -> dict[str, Any]:
    """Read metadata through imageio-ffmpeg when no ffprobe binary exists.

    ComfyUI Desktop commonly exposes a bundled FFmpeg executable through the
    Python package but does not ship a separate ffprobe binary. ``read_frames``
    yields metadata before decoding the first frame, so closing the generator
    immediately keeps this probe lightweight.
    """
    try:
        import imageio_ffmpeg
    except Exception as exc:
        raise RuntimeError(
            "CMK Split Video into Segments: neither ffprobe nor imageio_ffmpeg is available"
        ) from exc

    generator = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(generator)
    except Exception as exc:
        raise RuntimeError(
            "CMK Split Video into Segments: FFmpeg could not read video metadata"
        ) from exc
    finally:
        try:
            generator.close()
        except Exception:
            pass

    source_size = metadata.get("source_size") or metadata.get("size") or (0, 0)
    try:
        width, height = int(source_size[0]), int(source_size[1])
    except Exception:
        width, height = 0, 0

    rotation = int(round(_number(metadata.get("rotate")))) % 360
    if rotation in {90, 270}:
        width, height = height, width

    fps = _number(metadata.get("fps"))
    duration = _number(metadata.get("duration"))
    if width <= 0 or height <= 0:
        raise ValueError("CMK Split Video into Segments: video dimensions could not be determined")
    if fps <= 0:
        raise ValueError("CMK Split Video into Segments: video FPS could not be determined")
    if duration <= 0:
        raise ValueError("CMK Split Video into Segments: video duration could not be determined")

    frame_count = max(1, int(round(duration * fps)))
    return {
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "duration": float(duration),
        "frame_count": int(frame_count),
        # Audio is mapped with ``0:a?`` during splitting, so probing it is not
        # required for a safe encode path.
        "has_audio": None,
        "source_codec": str(metadata.get("codec") or "unknown"),
        "rotation": int(rotation),
        "probe_backend": "ffmpeg",
    }


def _probe_video(path: Path) -> dict[str, Any]:
    ffprobe = _optional_ffprobe()
    if ffprobe:
        try:
            return _probe_video_ffprobe(path, ffprobe)
        except Exception as exc:
            print(
                "[CMK Split Video into Segments] ffprobe failed; "
                f"using FFmpeg metadata fallback: {exc}"
            )
    return _probe_video_ffmpeg(path)

def _sanitize_stem(path: Path) -> str:
    value = re.sub(r"[^\w.-]+", "_", path.stem, flags=re.UNICODE).strip("._")
    return value or "video"


def _encoder_name(value: str) -> str:
    return "libx265" if str(value) == "hevc" else "libx264"


def _build_ranges(duration: float, segment_length: float, overlap: float) -> list[tuple[float, float]]:
    step = float(segment_length) - float(overlap)
    if step <= 0:
        raise ValueError(
            "CMK Split Video into Segments: OVERLAP must be smaller than the effective segment length"
        )

    ranges: list[tuple[float, float]] = []
    start = 0.0
    guard = 0
    while start < duration - 1e-6:
        guard += 1
        if guard > 100000:
            raise RuntimeError("CMK Split Video into Segments: segmentation guard triggered")

        end = min(start + segment_length, duration)
        remaining = duration - end
        # Preserve the full source without creating a tiny final fragment. The
        # current segment may grow by less than three seconds at the tail.
        if 0 < remaining < _MIN_SEGMENT_SECONDS:
            end = duration

        ranges.append((round(start, 6), round(end, 6)))
        if end >= duration - 1e-6:
            break
        start += step

    return ranges


def _format_fps(value: float) -> str:
    rounded = round(value)
    return str(int(rounded)) if abs(value - rounded) < 0.001 else f"{value:.3f}".rstrip("0").rstrip(".")


def _thumbnail(path: Path, duration: float) -> np.ndarray | None:
    timestamp = min(max(duration * 0.05, 0.0), 1.0)
    command = [
        _executable("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-vf",
        "scale='min(960,iw)':-2",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    try:
        result = _run(command, capture_stdout=True)
        with Image.open(io.BytesIO(result.stdout)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def _settings_identity(
    *,
    source: Path,
    max_frames_720p: int,
    max_frames_1080p: int,
    overlap: float,
    video_codec: str,
    video_bitrate: str,
    preset: str,
) -> dict[str, Any]:
    stat = source.stat()
    return {
        "schema": _SCHEMA,
        "source_path": str(source),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "max_frames_720p": int(max_frames_720p),
        "max_frames_1080p": int(max_frames_1080p),
        "overlap": float(overlap),
        "video_codec": str(video_codec),
        "video_bitrate": str(video_bitrate),
        "preset": str(preset),
        "audio_codec": _AUDIO_CODEC,
        "audio_bitrate": _AUDIO_BITRATE,
    }


def _identity_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_reusable_manifest(
    path: Path,
    identity_hash: str,
    *,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, str]:
    """Return a validated reusable manifest and a precise cache state reason."""
    if not path.is_file():
        return None, "NO_MANIFEST"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "MANIFEST_UNREADABLE"

    if not isinstance(payload, dict):
        return None, "MANIFEST_INVALID"
    if payload.get("type") != "CMK_VIDEO_SEGMENTS":
        return None, "TYPE_MISMATCH"
    if payload.get("schema") != _SCHEMA or int(payload.get("version", 0) or 0) != 1:
        return None, "SCHEMA_MISMATCH"
    if payload.get("identity_hash") != identity_hash:
        return None, "SETTINGS_CHANGED"

    manifest_output = Path(str(payload.get("output_directory", "") or ""))
    try:
        if manifest_output.resolve() != output_dir.resolve():
            return None, "OUTPUT_DIRECTORY_CHANGED"
    except Exception:
        return None, "OUTPUT_DIRECTORY_INVALID"

    segments = payload.get("segments")
    declared_count = int(payload.get("segments_created", 0) or 0)
    if not isinstance(segments, list) or not segments:
        return None, "SEGMENT_LIST_MISSING"
    if declared_count != len(segments):
        return None, "SEGMENT_COUNT_MISMATCH"

    previous_end = -1.0
    seen_paths: set[str] = set()
    for expected_index, item in enumerate(segments):
        if not isinstance(item, dict):
            return None, "SEGMENT_ENTRY_INVALID"
        try:
            index = int(item.get("index"))
            start = float(item.get("start"))
            end = float(item.get("end"))
            duration = float(item.get("duration"))
            segment_path = Path(str(item.get("path", "") or "")).resolve()
        except Exception:
            return None, "SEGMENT_METADATA_INVALID"

        if index != expected_index:
            return None, "SEGMENT_INDEX_MISMATCH"
        if start < 0 or end <= start or duration <= 0:
            return None, "SEGMENT_RANGE_INVALID"
        if abs((end - start) - duration) > 0.05:
            return None, "SEGMENT_DURATION_MISMATCH"
        if previous_end >= 0 and start > previous_end + 60.0:
            return None, "SEGMENT_SEQUENCE_INVALID"
        previous_end = max(previous_end, end)

        try:
            segment_path.relative_to(output_dir.resolve())
        except ValueError:
            return None, "SEGMENT_OUTSIDE_OUTPUT"

        key = str(segment_path)
        if key in seen_paths:
            return None, "SEGMENT_DUPLICATE"
        seen_paths.add(key)

        if not segment_path.is_file():
            return None, "SEGMENT_MISSING"
        try:
            if segment_path.stat().st_size <= 0:
                return None, "SEGMENT_EMPTY"
        except Exception:
            return None, "SEGMENT_UNREADABLE"

        filename = str(item.get("filename", "") or "")
        if filename and filename != segment_path.name:
            return None, "SEGMENT_FILENAME_MISMATCH"

    segment_paths = payload.get("segment_paths")
    if not isinstance(segment_paths, list) or len(segment_paths) != len(segments):
        return None, "SEGMENT_PATHS_MISMATCH"
    if [str(Path(str(value)).resolve()) for value in segment_paths] != [
        str(Path(str(item.get("path", ""))).resolve()) for item in segments
    ]:
        return None, "SEGMENT_PATHS_MISMATCH"

    return payload, "VALID"


def _video_preview_ui(path: Path, folder_type: str) -> dict[str, Any]:
    """Build a native ComfyUI video-preview descriptor."""
    path = path.resolve()
    if folder_type == "input":
        base = _input_root()
    elif folder_type == "output":
        base = Path(folder_paths.get_output_directory()).resolve()
    else:
        base = Path(folder_paths.get_temp_directory()).resolve()
    try:
        relative_parent = path.parent.relative_to(base).as_posix()
    except Exception:
        relative_parent = ""
    extension = path.suffix.lower().lstrip(".") or "mp4"
    return {
        "filename": path.name,
        "subfolder": "" if relative_parent == "." else relative_parent,
        "type": folder_type,
        "format": f"video/{extension}",
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


class CMKSplitVideoIntoSegments:
    """Split one input video into persistent, workflow-ready video segments."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "VIDEO": (_list_video_files(),),
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
            }
        }

    RETURN_TYPES = ("CMK_VIDEO_SEGMENTS", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("SEGMENTS", "LOG", "diagnostic")
    FUNCTION = "split_video"
    CATEGORY = "CMK/Toolbox/Video"
    OUTPUT_NODE = True

    def split_video(self, **inputs):
        video_name = str(inputs.get("VIDEO", "") or "")
        max_frames_720p = max(1, int(inputs.get("MAX FRAMES 720P", 540) or 540))
        max_frames_1080p = max(1, int(inputs.get("MAX FRAMES 1080P", 240) or 240))
        overlap = max(0.0, float(inputs.get("OVERLAP", 0.0) or 0.0))
        video_codec = str(inputs.get("VIDEO CODEC", "libx264") or "libx264")
        video_bitrate = str(inputs.get("VIDEO BITRATE", "8000k") or "8000k").strip()
        preset = str(inputs.get("PRESET", "fast") or "fast")

        if video_codec not in _VIDEO_CODECS:
            raise ValueError(f"CMK Split Video into Segments: unsupported VIDEO CODEC: {video_codec}")
        if preset not in _PRESETS:
            raise ValueError(f"CMK Split Video into Segments: unsupported PRESET: {preset}")
        if not video_bitrate:
            raise ValueError("CMK Split Video into Segments: VIDEO BITRATE is empty")

        source_path = _resolve_video_path(video_name)
        probe = _probe_video(source_path)
        width = int(probe["width"])
        height = int(probe["height"])
        fps = float(probe["fps"])
        duration = float(probe["duration"])
        frame_count = int(probe["frame_count"])

        resolution_class = "720p" if height <= 720 else "1080p+"
        target_frames = max_frames_720p if height <= 720 else max_frames_1080p
        minimum_frames_for_duration = int(math.ceil(fps * _MIN_SEGMENT_SECONDS))
        real_frames = max(target_frames, _MIN_SEGMENT_FRAMES, minimum_frames_for_duration)
        segment_length = round(real_frames / fps, 1)
        segment_length = max(_MIN_SEGMENT_SECONDS, segment_length)
        real_frames = max(real_frames, int(round(segment_length * fps)))

        if overlap >= segment_length:
            raise ValueError(
                "CMK Split Video into Segments: OVERLAP must be smaller than "
                f"{segment_length:.1f}s"
            )

        ranges = _build_ranges(duration, segment_length, overlap)
        if not ranges:
            raise RuntimeError("CMK Split Video into Segments: no segments were calculated")

        stem = _sanitize_stem(source_path)
        output_dir = (_output_root() / stem).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / _MANIFEST_NAME

        identity = _settings_identity(
            source=source_path,
            max_frames_720p=max_frames_720p,
            max_frames_1080p=max_frames_1080p,
            overlap=overlap,
            video_codec=video_codec,
            video_bitrate=video_bitrate,
            preset=preset,
        )
        identity_hash = _identity_hash(identity)
        reusable, reuse_reason = _load_reusable_manifest(
            manifest_path,
            identity_hash,
            output_dir=output_dir,
        )
        reused = reusable is not None
        invalidated = manifest_path.is_file() and not reused
        cache_status = "REUSED" if reused else ("INVALIDATED" if invalidated else "SPLIT")

        if reusable is not None:
            manifest = reusable
            segment_items = list(manifest["segments"])
        else:
            for old_path in output_dir.glob(f"{stem}_segment_*.mp4"):
                try:
                    old_path.unlink()
                except Exception:
                    pass

            try:
                from comfy.utils import ProgressBar

                progress = ProgressBar(len(ranges))
            except Exception:
                progress = None

            ffmpeg = _executable("ffmpeg")
            encoder = _encoder_name(video_codec)
            segment_items: list[dict[str, Any]] = []

            for index, (start, end) in enumerate(ranges):
                segment_duration = max(0.0, end - start)
                output_path = output_dir / (
                    f"{stem}_segment_{index:03d}_{start:05.1f}-{end:05.1f}.mp4"
                )
                command = [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_path),
                    "-ss",
                    f"{start:.6f}",
                    "-t",
                    f"{segment_duration:.6f}",
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-map_metadata",
                    "0",
                    "-c:v",
                    encoder,
                    "-b:v",
                    video_bitrate,
                    "-preset",
                    preset,
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    _AUDIO_CODEC,
                    "-b:a",
                    _AUDIO_BITRATE,
                    "-movflags",
                    "+faststart",
                    "-avoid_negative_ts",
                    "make_zero",
                    str(output_path),
                ]

                print(
                    "[CMK Split Video into Segments] "
                    f"{index + 1}/{len(ranges)}: {start:.1f}s–{end:.1f}s"
                )
                _run(command)

                item = {
                    "index": int(index),
                    "filename": output_path.name,
                    "path": str(output_path),
                    "start": float(start),
                    "end": float(end),
                    "duration": float(segment_duration),
                    "estimated_frames": int(round(segment_duration * fps)),
                }
                segment_items.append(item)
                if progress is not None:
                    progress.update(1)

            manifest = {
                "type": "CMK_VIDEO_SEGMENTS",
                "version": 1,
                "schema": _SCHEMA,
                "identity_hash": identity_hash,
                "identity": identity,
                "source_file": source_path.name,
                "source_relative": video_name,
                "source_path": str(source_path),
                "probe_backend": str(probe.get("probe_backend", "unknown")),
                "output_directory": str(output_dir),
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "resolution_class": resolution_class,
                "target_frames": int(target_frames),
                "real_frames": int(real_frames),
                "segment_length": float(segment_length),
                "overlap": float(overlap),
                "video_codec": video_codec,
                "ffmpeg_video_encoder": encoder,
                "video_bitrate": video_bitrate,
                "audio_codec": _AUDIO_CODEC,
                "audio_bitrate": _AUDIO_BITRATE,
                "preset": preset,
                "segments_created": len(segment_items),
                "segments": segment_items,
                "segment_paths": [str(item["path"]) for item in segment_items],
            }
            _write_manifest(manifest_path, manifest)

        # Rebuild the public payload from the persistent manifest and use tuples
        # for sequence fields to communicate the immutable transport contract.
        segments_payload = dict(manifest)
        segments_payload["segments"] = tuple(dict(item) for item in segment_items)
        segments_payload["segment_paths"] = tuple(str(item["path"]) for item in segment_items)
        segments_payload["manifest_path"] = str(manifest_path)
        segments_payload["cache_reused"] = bool(reused)
        segments_payload["cache_status"] = cache_status
        segments_payload["cache_reason"] = reuse_reason

        log_lines = [
            f"VIDEO FILE       : {source_path.name}",
            f"SOURCE           : {resolution_class}/{_format_fps(fps)}fps",
            f"METADATA PROBE   : {probe.get('probe_backend', 'unknown')}",
            f"RESOLUTION       : {width} × {height}",
            f"FRAMES           : {frame_count}",
            f"DURATION         : {duration:.3f}s",
            f"TARGET FRAMES    : {target_frames}",
            f"REAL FRAMES      : {real_frames}",
            f"SEGMENT LENGTH   : {segment_length:.1f}s",
            f"SEGMENTS CREATED : {len(segment_items)}",
            f"OVERLAP          : {overlap:g}s",
            f"VIDEO CODEC      : {video_codec}",
            f"VIDEO BITRATE    : {video_bitrate}",
            f"PRESET           : {preset}",
            f"AUDIO            : {_AUDIO_CODEC}/{_AUDIO_BITRATE}",
            f"OUTPUT           : {output_dir}",
            f"CACHE STATUS     : {cache_status}",
            f"CACHE REASON     : {reuse_reason}",
        ]
        log_pipe = cmk_add_block(
            {
                "blocks": [],
                "filename_string": source_path.name,
                "video_source": str(source_path),
                "video_segments_directory": str(output_dir),
                "video_segments_manifest": str(manifest_path),
            },
            "Split Video into Segments",
            1,
            log_lines,
            True,
        )

        summary = "\n".join(
            [
                "INPUT",
                "-----",
                f"video_file           : {source_path.name}",
                f"video_path           : {source_path}",
                f"source               : {resolution_class}/{_format_fps(fps)}fps",
                f"metadata_probe       : {probe.get('probe_backend', 'unknown')}",
                "",
                "SEGMENTATION",
                "------------",
                f"target_frames        : {target_frames} frames/segment",
                f"real_frames          : {real_frames} frames/segment",
                f"segment_length       : {segment_length:.1f}s",
                f"segments_created     : {len(segment_items)}",
                f"overlap              : {overlap:g}s",
                "",
                "ENCODING",
                "--------",
                f"video_codec          : {video_codec}",
                f"video_bitrate        : {video_bitrate}",
                f"preset               : {preset}",
                f"audio                : {_AUDIO_CODEC}/{_AUDIO_BITRATE}",
                "",
                "OUTPUT",
                "------",
                f"directory            : {output_dir}",
                f"manifest             : {manifest_path}",
                f"cache_status          : {cache_status.lower()}",
                f"cache_reason          : {reuse_reason.lower()}",
            ]
        )
        preview = _thumbnail(source_path, duration)
        diagnostic = make_diagnostic_payload(
            title="Split Video into Segments",
            node="CMK Split Video into Segments",
            previews=[preview] if preview is not None else [],
            stages=(
                [
                    {
                        "title": "SOURCE VIDEO",
                        "subtitle": f"{width} × {height} / {_format_fps(fps)} fps / {duration:.1f}s",
                        "image": preview,
                    }
                ]
                if preview is not None
                else []
            ),
            summary=summary,
            details=summary,
            mode="Persistent Video Segmentation",
            metadata={
                "source_file": source_path.name,
                "source_path": str(source_path),
                "probe_backend": str(probe.get("probe_backend", "unknown")),
                "output_directory": str(output_dir),
                "manifest_path": str(manifest_path),
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "resolution_class": resolution_class,
                "target_frames": int(target_frames),
                "real_frames": int(real_frames),
                "segment_length": float(segment_length),
                "overlap": float(overlap),
                "segments_created": len(segment_items),
                "video_codec": video_codec,
                "video_bitrate": video_bitrate,
                "preset": preset,
                "cache_reused": bool(reused),
                "cache_status": cache_status,
                "cache_reason": reuse_reason,
            },
            metrics={
                "segments_created": len(segment_items),
                "segment_length": float(segment_length),
                "duration": duration,
                "fps": fps,
            },
        )

        result = (segments_payload, log_pipe, diagnostic)
        return {
            "ui": {"cmk_video_player": [_video_preview_ui(source_path, "input")]},
            "result": result,
        }

    @classmethod
    def IS_CHANGED(cls, **inputs):
        try:
            source = _resolve_video_path(str(inputs.get("VIDEO", "") or ""))
            identity = _settings_identity(
                source=source,
                max_frames_720p=max(1, int(inputs.get("MAX FRAMES 720P", 540) or 540)),
                max_frames_1080p=max(1, int(inputs.get("MAX FRAMES 1080P", 240) or 240)),
                overlap=max(0.0, float(inputs.get("OVERLAP", 0.0) or 0.0)),
                video_codec=str(inputs.get("VIDEO CODEC", "libx264") or "libx264"),
                video_bitrate=str(inputs.get("VIDEO BITRATE", "8000k") or "8000k"),
                preset=str(inputs.get("PRESET", "fast") or "fast"),
            )
            stem = _sanitize_stem(source)
            manifest = _output_root() / stem / _MANIFEST_NAME
            state = {
                "identity": identity,
                "manifest_exists": manifest.is_file(),
                "manifest_size": manifest.stat().st_size if manifest.is_file() else 0,
                "manifest_mtime_ns": manifest.stat().st_mtime_ns if manifest.is_file() else 0,
            }
            return _identity_hash(state)
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **inputs):
        try:
            _resolve_video_path(str(inputs.get("VIDEO", "") or ""))
            overlap = max(0.0, float(inputs.get("OVERLAP", 0.0) or 0.0))
            if overlap > 60.0:
                return "OVERLAP exceeds 60 seconds"
            if str(inputs.get("VIDEO CODEC", "libx264")) not in _VIDEO_CODECS:
                return "Invalid VIDEO CODEC"
            if str(inputs.get("PRESET", "fast")) not in _PRESETS:
                return "Invalid PRESET"
            if not str(inputs.get("VIDEO BITRATE", "8000k") or "").strip():
                return "VIDEO BITRATE is empty"
            _executable("ffmpeg")
        except Exception as exc:
            return str(exc)
        return True


NODE_CLASS_MAPPINGS = {
    "CMKSplitVideoIntoSegments": CMKSplitVideoIntoSegments,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMKSplitVideoIntoSegments": "CMK Split Video into Segments",
}
