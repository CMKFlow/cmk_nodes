from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import folder_paths

from ...pipe.cmk_log_pipe import cmk_add_block
from ...utils.cmk_diagnostic import make_diagnostic_payload
from .split_video_segments import _executable, _encoder_name, _run


_SCHEMA = "cmk_merged_video_v1"
_MANIFEST_NAME = "merged.json"
_VIDEO_CODECS = ["libx264", "hevc"]
_PRESETS = [
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
]
_AUDIO_CODEC = "aac"
_AUDIO_BITRATE = "192k"


def _output_root() -> Path:
    return Path(folder_paths.get_output_directory()).resolve() / "video" / "merged"


def _safe_stem(value: str) -> str:
    stem = Path(str(value or "video")).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "video"


def _normalize_segments(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "CMK_VIDEO_SEGMENTS":
        raise TypeError("CMK Merge and Save Video: SEGMENTS is not CMK_VIDEO_SEGMENTS")
    items = value.get("segments")
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("CMK Merge and Save Video: SEGMENTS contains no segment entries")
    normalized = dict(value)
    normalized["segments"] = [dict(item) for item in items]
    return normalized


def _validate_segment_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload["segments"]
    validated: list[dict[str, Any]] = []
    previous_start = -1.0
    for expected, item in enumerate(items):
        try:
            index = int(item.get("index"))
            start = float(item.get("start"))
            end = float(item.get("end"))
            path = Path(str(item.get("path", "") or "")).resolve()
        except Exception as exc:
            raise ValueError(f"CMK Merge and Save Video: invalid segment {expected}") from exc
        if index != expected:
            raise ValueError("CMK Merge and Save Video: segment indices are not contiguous")
        if start < 0 or end <= start or start < previous_start:
            raise ValueError("CMK Merge and Save Video: invalid segment chronology")
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"CMK Merge and Save Video: missing or empty segment: {path}")
        previous_start = start
        validated.append({**item, "index": index, "start": start, "end": end, "path": str(path)})
    return validated


def _has_audio(path: Path) -> bool:
    ffmpeg = _executable("ffmpeg")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    text = result.stderr.decode("utf-8", errors="replace")
    return bool(re.search(r"Stream #.*Audio:", text))


def _segment_fingerprint(items: list[dict[str, Any]], settings: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    files = []
    for item in items:
        path = Path(item["path"])
        stat = path.stat()
        files.append({
            "index": int(item["index"]),
            "path": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "start": float(item["start"]),
            "end": float(item["end"]),
        })
    raw = json.dumps({"schema": _SCHEMA, "files": files, "settings": settings}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), files


def _load_reusable(path: Path, fingerprint: str, output_path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, "NO_MANIFEST"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "MANIFEST_UNREADABLE"
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA or int(payload.get("version", 0)) != 1:
        return None, "SCHEMA_MISMATCH"
    if payload.get("fingerprint") != fingerprint:
        return None, "SEGMENTS_OR_SETTINGS_CHANGED"
    try:
        if Path(str(payload.get("video_path", ""))).resolve() != output_path.resolve():
            return None, "OUTPUT_PATH_CHANGED"
    except Exception:
        return None, "OUTPUT_PATH_INVALID"
    if not output_path.is_file():
        return None, "OUTPUT_MISSING"
    if output_path.stat().st_size <= 0:
        return None, "OUTPUT_EMPTY"
    return payload, "VALID"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _thumbnail(video_path: Path) -> np.ndarray | None:
    ffmpeg = _executable("ffmpeg")
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", "0", "-i", str(video_path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        import io
        with Image.open(io.BytesIO(result.stdout)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def _video_preview_ui(path: Path) -> dict[str, Any]:
    """Build a native ComfyUI output-video preview descriptor."""
    path = path.resolve()
    base = Path(folder_paths.get_output_directory()).resolve()
    try:
        relative_parent = path.parent.relative_to(base).as_posix()
    except Exception:
        relative_parent = ""
    extension = path.suffix.lower().lstrip(".") or "mp4"
    return {
        "filename": path.name,
        "subfolder": "" if relative_parent == "." else relative_parent,
        "type": "output",
        "format": f"video/{extension}",
    }


def _safe_output_folder(value: str) -> Path:
    base = Path(folder_paths.get_output_directory()).resolve()
    relative = str(value or "video/final").strip().replace("\\", "/").strip("/")
    parts = [part for part in relative.split("/") if part not in {"", ".", ".."}]
    target = (base.joinpath(*parts) if parts else base / "video" / "final").resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError("CMK Merge and Save Video: OUTPUT FOLDER must remain inside output/") from exc
    target.mkdir(parents=True, exist_ok=True)
    return target


def _unique_copy(source: Path, target_dir: Path, prefix: str) -> Path:
    extension = source.suffix.lower() or ".mp4"
    stem = _safe_stem(prefix or source.stem)
    candidate = target_dir / f"{stem}{extension}"
    counter = 1
    while candidate.exists():
        candidate = target_dir / f"{stem}_{counter:03d}{extension}"
        counter += 1
    shutil.copy2(source, candidate)
    return candidate.resolve()


class CMKMergeAndSaveVideo:
    """Merge CMK_VIDEO_SEGMENTS into one persistent video with overlap removal."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "SEGMENTS": ("CMK_VIDEO_SEGMENTS",),
                "LOG": ("CMK_LOG_PIPE",),
                "SAVE ENABLED": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
                "FILENAME PREFIX": ("STRING", {"default": "video", "multiline": False}),
                "OUTPUT FOLDER": ("STRING", {"default": "video/final", "multiline": False}),
                "VIDEO CODEC": (_VIDEO_CODECS, {"default": "libx264"}),
                "VIDEO BITRATE": ("STRING", {"default": "8000k", "multiline": False}),
                "PRESET": (_PRESETS, {"default": "fast"}),
            }
        }

    RETURN_TYPES = ("CMK_VIDEO", "STRING", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("VIDEO", "FULLPATH", "LOG", "diagnostic")
    FUNCTION = "merge_segments"
    CATEGORY = "CMK/Toolbox/Video"
    OUTPUT_NODE = True

    def merge_segments(self, **inputs):
        segments_payload = _normalize_segments(inputs.get("SEGMENTS"))
        segment_items = _validate_segment_items(segments_payload)
        log_in = inputs.get("LOG")
        if not isinstance(log_in, dict):
            raise TypeError("CMK Merge and Save Video: LOG is not a CMK log pipe")

        save_enabled = bool(inputs.get("SAVE ENABLED", False))
        filename_prefix = str(inputs.get("FILENAME PREFIX", "video") or "video")
        output_folder = str(inputs.get("OUTPUT FOLDER", "video/final") or "video/final")

        video_codec = str(inputs.get("VIDEO CODEC", segments_payload.get("video_codec", "libx264")) or "libx264")
        video_bitrate = str(inputs.get("VIDEO BITRATE", segments_payload.get("video_bitrate", "8000k")) or "8000k").strip()
        preset = str(inputs.get("PRESET", segments_payload.get("preset", "fast")) or "fast")
        if video_codec not in _VIDEO_CODECS:
            raise ValueError(f"CMK Merge and Save Video: unsupported VIDEO CODEC: {video_codec}")
        if preset not in _PRESETS:
            raise ValueError(f"CMK Merge and Save Video: unsupported PRESET: {preset}")
        if not video_bitrate:
            raise ValueError("CMK Merge and Save Video: VIDEO BITRATE is empty")

        source_name = str(segments_payload.get("source_file", "video.mp4") or "video.mp4")
        stem = _safe_stem(source_name)
        output_dir = (_output_root() / stem).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{stem}_merged.mp4"
        manifest_path = output_dir / _MANIFEST_NAME

        settings = {
            "video_codec": video_codec,
            "video_bitrate": video_bitrate,
            "preset": preset,
            "audio_codec": _AUDIO_CODEC,
            "audio_bitrate": _AUDIO_BITRATE,
        }
        fingerprint, file_state = _segment_fingerprint(segment_items, settings)
        reusable, reason = _load_reusable(manifest_path, fingerprint, output_path)
        reused = reusable is not None
        invalidated = manifest_path.is_file() and not reused
        status = "REUSED" if reused else ("INVALIDATED" if invalidated else "MERGED")

        trims = []
        previous_end = None
        for item in segment_items:
            trim = 0.0 if previous_end is None else max(0.0, float(previous_end) - float(item["start"]))
            trims.append(trim)
            previous_end = float(item["end"])

        if not reused:
            ffmpeg = _executable("ffmpeg")
            has_audio = all(_has_audio(Path(item["path"])) for item in segment_items)
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
            for item in segment_items:
                command += ["-i", str(item["path"])]

            filters = []
            concat_inputs = []
            for index, trim_start in enumerate(trims):
                filters.append(f"[{index}:v:0]trim=start={trim_start:.6f},setpts=PTS-STARTPTS[v{index}]")
                concat_inputs.append(f"[v{index}]")
                if has_audio:
                    filters.append(f"[{index}:a:0]atrim=start={trim_start:.6f},asetpts=PTS-STARTPTS[a{index}]")
                    concat_inputs.append(f"[a{index}]")

            if has_audio:
                filters.append("".join(concat_inputs) + f"concat=n={len(segment_items)}:v=1:a=1[vout][aout]")
            else:
                filters.append("".join(concat_inputs) + f"concat=n={len(segment_items)}:v=1:a=0[vout]")

            command += [
                "-filter_complex", ";".join(filters),
                "-map", "[vout]",
            ]
            if has_audio:
                command += ["-map", "[aout]"]
            command += [
                "-c:v", _encoder_name(video_codec),
                "-b:v", video_bitrate,
                "-preset", preset,
                "-pix_fmt", "yuv420p",
            ]
            if has_audio:
                command += ["-c:a", _AUDIO_CODEC, "-b:a", _AUDIO_BITRATE]
            command += ["-movflags", "+faststart", str(output_path)]

            print(f"[CMK Merge and Save Video] merging {len(segment_items)} segments -> {output_path}")
            _run(command)
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                raise RuntimeError("CMK Merge and Save Video: merged output was not created")

            reusable = {
                "type": "CMK_VIDEO",
                "version": 1,
                "schema": _SCHEMA,
                "fingerprint": fingerprint,
                "source_file": source_name,
                "source_segments_manifest": str(segments_payload.get("manifest_path", "")),
                "video_path": str(output_path),
                "output_directory": str(output_dir),
                "segments_merged": len(segment_items),
                "overlap": float(segments_payload.get("overlap", 0.0) or 0.0),
                "trim_starts": trims,
                "width": int(segments_payload.get("width", 0) or 0),
                "height": int(segments_payload.get("height", 0) or 0),
                "fps": float(segments_payload.get("fps", 0.0) or 0.0),
                "duration": float(segments_payload.get("duration", 0.0) or 0.0),
                "frame_count": int(segments_payload.get("frame_count", 0) or 0),
                "video_codec": video_codec,
                "video_bitrate": video_bitrate,
                "audio_codec": _AUDIO_CODEC if has_audio else "none",
                "audio_bitrate": _AUDIO_BITRATE if has_audio else "none",
                "preset": preset,
                "segment_files": file_state,
            }
            _write_json(manifest_path, reusable)

        video_payload = dict(reusable)
        video_payload["manifest_path"] = str(manifest_path)
        video_payload["cache_reused"] = bool(reused)
        video_payload["cache_status"] = status
        video_payload["cache_reason"] = reason

        shown_path = output_path
        fullpath = ""
        publish_status = "PREVIEW"
        if save_enabled:
            shown_path = _unique_copy(output_path, _safe_output_folder(output_folder), filename_prefix)
            fullpath = str(shown_path)
            publish_status = "SAVED"

        video_payload["preview_path"] = str(shown_path)
        video_payload["published_path"] = fullpath
        video_payload["preview_status"] = publish_status

        log_lines = [
            f"SOURCE FILE      : {source_name}",
            f"SEGMENTS MERGED  : {len(segment_items)}",
            f"OVERLAP REMOVED  : {float(segments_payload.get('overlap', 0.0) or 0.0):g}s",
            f"VIDEO CODEC      : {video_codec}",
            f"VIDEO BITRATE    : {video_bitrate}",
            f"PRESET           : {preset}",
            f"OUTPUT           : {output_path}",
            f"CACHE STATUS     : {status}",
            f"CACHE REASON     : {reason}",
            f"SAVE ENABLED     : {'ON' if save_enabled else 'OFF'}",
            f"PLAYER           : {shown_path}",
        ]
        if save_enabled:
            log_lines.extend([
                f"OUTPUT FOLDER    : {shown_path.parent}",
                f"FULLPATH         : {shown_path}",
            ])
        log_pipe = cmk_add_block(log_in, "Merge and Save Video", 90, log_lines, True)
        log_pipe["video_output"] = str(output_path)
        log_pipe["video_merge_manifest"] = str(manifest_path)

        summary = "\n".join(log_lines)
        preview = _thumbnail(output_path)
        diagnostic = make_diagnostic_payload(
            title="Merge and Save Video",
            node="CMK Merge and Save Video",
            previews=[preview] if preview is not None else [],
            stages=[{"title": "MERGED VIDEO", "subtitle": f"{len(segment_items)} segments / {status}", "image": preview}] if preview is not None else [],
            summary=summary,
            details=summary + "\n\nTRIM STARTS\n" + "\n".join(f"segment {i:03d}: {value:.3f}s" for i, value in enumerate(trims)),
            mode="Overlap-aware concat",
            metadata={
                "status": status,
                "reason": reason,
                "video_path": str(output_path),
                "segments_merged": len(segment_items),
                "overlap": float(segments_payload.get("overlap", 0.0) or 0.0),
                "trim_starts": trims,
                "video_codec": video_codec,
                "video_bitrate": video_bitrate,
                "preset": preset,
            },
        )
        result = (video_payload, fullpath, log_pipe, diagnostic)
        return {
            "ui": {"cmk_video_player": [_video_preview_ui(shown_path)]},
            "result": result,
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **inputs):
        try:
            _executable("ffmpeg")
        except Exception as exc:
            return str(exc).replace("CMK Split Video into Segments", "CMK Merge and Save Video")
        bitrate = str(inputs.get("VIDEO BITRATE", "8000k") or "").strip()
        if not bitrate:
            return "CMK Merge and Save Video: VIDEO BITRATE is empty"
        return True
