import asyncio
import hashlib
import json
import os
import subprocess
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import folder_paths
from aiohttp import web
from server import PromptServer

from .cmk_mappings import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
_SHOWCASE_WORKFLOWS = Path(__file__).resolve().parent / "workflows" / "showcase"
_SHOWCASE_METADATA = _SHOWCASE_WORKFLOWS / "metadata"
_VIDEO_WORKFLOW_TEMPLATE = _SHOWCASE_WORKFLOWS / "CMK FaceSwap Video.json"
_PROJECT_WORKFLOW_NAME = "cmk_project_workflow.json"
_PROJECT_METADATA_NAME = "cmk_video_project.json"


def _video_storage_roots():
    video_root = Path(folder_paths.get_output_directory()).resolve() / "video"
    return {
        "segments": video_root / "segments",
        "merged": video_root / "merged",
    }


def _storage_stats(path):
    files = media_files = directories = size = 0
    latest = 0.0
    if not path.exists():
        return {"files": 0, "media_files": 0, "directories": 0, "bytes": 0, "latest": 0.0}
    for item in path.rglob("*"):
        try:
            if item.is_symlink():
                files += 1
            elif item.is_dir():
                directories += 1
            elif item.is_file():
                files += 1
                if item.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
                    media_files += 1
                stat = item.stat()
                size += stat.st_size
                latest = max(latest, stat.st_mtime)
        except OSError:
            continue
    return {"files": files, "media_files": media_files, "directories": directories, "bytes": size, "latest": latest}


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _project_id(key):
    return hashlib.sha256(f"cmk-video-project-v1\0{key}".encode("utf-8")).hexdigest()[:24]


def _project_key_from_manifest_reference(value):
    parts = Path(str(value or "")).parts
    try:
        face_swap = parts.index("face_swap")
        if face_swap + 1 < len(parts) and parts[face_swap + 1] != "_selection_state":
            return parts[face_swap + 1]
    except ValueError:
        pass
    for index, part in enumerate(parts):
        if part == "segments" and index + 1 < len(parts):
            candidate = parts[index + 1]
            if candidate != "video":
                return candidate
    return None


def _media_sources_from_workflow(workflow):
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    if not isinstance(nodes, list):
        return None
    for node in nodes:
        if node.get("type") != "CMKFaceSwapVideoLoader":
            continue
        values = node.get("widgets_values")
        if not isinstance(values, list) or len(values) < 7:
            continue
        media = values[6]
        if isinstance(media, str):
            try:
                media = json.loads(media)
            except json.JSONDecodeError:
                continue
        if not isinstance(media, dict):
            continue
        video = str(media.get("video", "") or "").strip()
        source = str(media.get("source", "") or "").strip()
        if video:
            return {"video": video, "source": source}
    return None


def _saved_video_workflows():
    """Index user workflows by their exact video input, newest first."""
    getter = getattr(folder_paths, "get_user_directory", None)
    if not callable(getter):
        return {}
    root = Path(getter()).resolve()
    if not root.is_dir():
        return {}
    matches = {}
    for path in root.rglob("*.json"):
        if path.is_symlink() or path.name == _PROJECT_WORKFLOW_NAME:
            continue
        workflow = _read_json(path)
        media = _media_sources_from_workflow(workflow)
        if not media:
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        matches.setdefault(media["video"], []).append({
            "path": path,
            "source_image": media["source"],
            "modified": modified,
        })
    for candidates in matches.values():
        candidates.sort(key=lambda item: item["modified"], reverse=True)
    return matches


def _discover_video_projects():
    roots = _video_storage_roots()
    segments_root = roots["segments"]
    merged_root = roots["merged"]
    projects = {}
    saved_workflows = _saved_video_workflows()

    def ensure(key):
        project = projects.setdefault(key, {
            "key": key,
            "id": _project_id(key),
            "display_name": key,
            "segment_paths": [],
            "merged_paths": [],
            "state_paths": [],
            "source_relative": "",
            "source_image": "",
            "split_settings": {},
            "workflow_path": None,
        })
        return project

    if segments_root.is_dir() and not segments_root.is_symlink():
        for path in segments_root.iterdir():
            if not path.is_dir() or path.is_symlink() or path.name == "video":
                continue
            project = ensure(path.name)
            project["segment_paths"].append(path)
            manifest = _read_json(path / "segments.json")
            identity = manifest.get("identity") if isinstance(manifest.get("identity"), dict) else {}
            project["display_name"] = str(manifest.get("source_file") or project["display_name"])
            project["source_relative"] = str(manifest.get("source_relative") or "")
            project["split_settings"] = {
                "max_frames_720p": identity.get("max_frames_720p", 540),
                "max_frames_1080p": identity.get("max_frames_1080p", 240),
                "overlap": manifest.get("overlap", 0.0),
                "video_codec": manifest.get("video_codec", "libx264"),
                "video_bitrate": manifest.get("video_bitrate", "8000k"),
                "preset": manifest.get("preset", "fast"),
            }
            metadata = _read_json(path / _PROJECT_METADATA_NAME)
            if metadata.get("source_image"):
                project["source_image"] = str(metadata["source_image"])
            workflow_path = path / _PROJECT_WORKFLOW_NAME
            if workflow_path.is_file() and not workflow_path.is_symlink():
                project["workflow_path"] = workflow_path

            if not project["workflow_path"]:
                candidates = saved_workflows.get(project["source_relative"], [])
                if candidates:
                    project["workflow_path"] = candidates[0]["path"]
                    if not project["source_image"]:
                        project["source_image"] = candidates[0]["source_image"]

        face_swap_root = segments_root / "video" / "face_swap"
        if face_swap_root.is_dir() and not face_swap_root.is_symlink():
            for path in face_swap_root.iterdir():
                if not path.is_dir() or path.is_symlink():
                    continue
                if path.name == "_selection_state":
                    for state_path in path.glob("*.json"):
                        state = _read_json(state_path)
                        key = _project_key_from_manifest_reference(state.get("manifest_path"))
                        if key:
                            ensure(key)["state_paths"].append(state_path)
                    continue
                ensure(path.name)["segment_paths"].append(path)

    if merged_root.is_dir() and not merged_root.is_symlink():
        for path in merged_root.iterdir():
            if not path.is_dir() or path.is_symlink():
                continue
            manifest = _read_json(path / "merged.json")
            key = _project_key_from_manifest_reference(manifest.get("source_segments_manifest")) or path.name
            project = ensure(key)
            project["merged_paths"].append(path)
            project["display_name"] = str(manifest.get("source_file") or project["display_name"])

    public = []
    by_id = {}
    for project in projects.values():
        segment_stats = [_storage_stats(path) for path in project["segment_paths"]]
        merged_stats = [_storage_stats(path) for path in project["merged_paths"]]
        segment_files = sum(item["media_files"] for item in segment_stats)
        segment_bytes = sum(item["bytes"] for item in segment_stats)
        merged_files = sum(item["media_files"] for item in merged_stats)
        merged_bytes = sum(item["bytes"] for item in merged_stats)
        latest = max([0.0, *(item["latest"] for item in segment_stats), *(item["latest"] for item in merged_stats)])
        entry = {
            "id": project["id"],
            "display_name": project["display_name"],
            "segment_files": segment_files,
            "segment_bytes": segment_bytes,
            "merged_files": merged_files,
            "merged_bytes": merged_bytes,
            "total_files": segment_files + merged_files,
            "total_bytes": segment_bytes + merged_bytes,
            "last_modified": datetime.fromtimestamp(latest, timezone.utc).isoformat() if latest else None,
            "has_segments": bool(segment_files),
            "has_merged": bool(merged_files),
            "has_workflow": bool(project["workflow_path"]),
        }
        public.append(entry)
        by_id[project["id"]] = project
    public.sort(key=lambda item: item["last_modified"] or "", reverse=True)
    return public, by_id


def _delete_video_project(project):
    targets = [*project["segment_paths"], *project["merged_paths"]]
    files = size = 0
    failures = []
    for path in targets:
        try:
            if path.is_symlink():
                raise OSError("symbolic-link project directory refused")
            stats = _storage_stats(path)
            shutil.rmtree(path)
            files += stats["files"]
            size += stats["bytes"]
        except OSError as error:
            failures.append({"item": path.name, "error": str(error)})
    for path in project["state_paths"]:
        try:
            if path.is_symlink():
                raise OSError("symbolic-link state file refused")
            stat = path.stat()
            path.unlink()
            files += 1
            size += stat.st_size
        except FileNotFoundError:
            continue
        except OSError as error:
            failures.append({"item": path.name, "error": str(error)})
    return {
        "deleted_files": files,
        "freed_bytes": size,
        "failures": failures,
        "complete": not failures,
    }


def _project_workflow(project):
    assigned = project.get("workflow_path")
    if assigned:
        workflow = _read_json(assigned)
        if isinstance(workflow.get("nodes"), list):
            return workflow, "assigned"

    workflow = _read_json(_VIDEO_WORKFLOW_TEMPLATE)
    if not isinstance(workflow.get("nodes"), list):
        raise OSError("CMK video workflow template is unavailable")
    settings = project.get("split_settings") or {}
    media_sources = json.dumps(
        {
            "video": project.get("source_relative", ""),
            "source": project.get("source_image", ""),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    for node in workflow["nodes"]:
        if node.get("type") != "CMKFaceSwapVideoLoader":
            continue
        node["widgets_values"] = [
            int(settings.get("max_frames_720p", 540) or 540),
            int(settings.get("max_frames_1080p", 240) or 240),
            float(settings.get("overlap", 0.0) or 0.0),
            str(settings.get("video_codec", "libx264") or "libx264"),
            str(settings.get("video_bitrate", "8000k") or "8000k"),
            str(settings.get("preset", "fast") or "fast"),
            media_sources,
        ]
        break
    return workflow, "template"


def _open_project_folder(project):
    candidates = [*project.get("segment_paths", []), *project.get("merged_paths", [])]
    folder = next((path for path in candidates if path.is_dir() and not path.is_symlink()), None)
    if folder is None:
        raise OSError("CMK video project folder is unavailable")
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)], close_fds=True)
    elif os.name == "nt":
        os.startfile(str(folder))
    else:
        subprocess.Popen(["xdg-open", str(folder)], close_fds=True)
    return folder


def _clear_storage_root(path):
    before = _storage_stats(path)
    if path.is_symlink():
        raise OSError(f"refusing to clear symbolic-link storage root: {path}")
    if path.exists():
        for item in path.iterdir():
            if item.is_symlink() or item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    path.mkdir(parents=True, exist_ok=True)
    return before


@PromptServer.instance.routes.get("/cmk/video-storage")
async def cmk_video_storage(request):
    roots = _video_storage_roots()
    locations = await asyncio.gather(*(
        asyncio.to_thread(_storage_stats, path) for path in roots.values()
    ))
    projects, _ = await asyncio.to_thread(_discover_video_projects)
    location_map = dict(zip(roots, locations))
    return web.json_response({
        "locations": location_map,
        "totals": {
            "segments": location_map["segments"],
            "merged": location_map["merged"],
            "all_bytes": location_map["segments"]["bytes"] + location_map["merged"]["bytes"],
        },
        "projects": projects,
    })


@PromptServer.instance.routes.post("/cmk/video-storage/clear")
async def cmk_clear_video_storage(request):
    roots = _video_storage_roots()
    try:
        results = await asyncio.gather(*(
            asyncio.to_thread(_clear_storage_root, path) for path in roots.values()
        ))
        removed = dict(zip(roots, results))
    except OSError as error:
        raise web.HTTPInternalServerError(text=f"CMK video storage could not be cleared: {error}") from error
    return web.json_response({"cleared": True, "removed": removed})


@PromptServer.instance.routes.post("/cmk/video-storage/project/clear")
async def cmk_clear_video_project(request):
    try:
        payload = await request.json()
    except (json.JSONDecodeError, TypeError):
        raise web.HTTPBadRequest(text="Invalid JSON request")
    project_id = str(payload.get("id", "") if isinstance(payload, dict) else "")
    _, projects = await asyncio.to_thread(_discover_video_projects)
    project = projects.get(project_id)
    if project is None:
        raise web.HTTPNotFound(text="CMK video project not found")
    result = await asyncio.to_thread(_delete_video_project, project)
    return web.json_response(result, status=200 if result["complete"] else 207)


@PromptServer.instance.routes.get("/cmk/video-storage/project/workflow")
async def cmk_video_project_workflow(request):
    project_id = str(request.query.get("id", ""))
    _, projects = await asyncio.to_thread(_discover_video_projects)
    project = projects.get(project_id)
    if project is None:
        raise web.HTTPNotFound(text="CMK video project not found")
    try:
        workflow, source = await asyncio.to_thread(_project_workflow, project)
    except OSError as error:
        raise web.HTTPInternalServerError(text=str(error)) from error
    return web.json_response({
        "workflow": workflow,
        "source": source,
        "name": project["display_name"],
    })


@PromptServer.instance.routes.post("/cmk/video-storage/project/open-folder")
async def cmk_open_video_project_folder(request):
    try:
        payload = await request.json()
    except (json.JSONDecodeError, TypeError):
        raise web.HTTPBadRequest(text="Invalid JSON request")
    project_id = str(payload.get("id", "") if isinstance(payload, dict) else "")
    _, projects = await asyncio.to_thread(_discover_video_projects)
    project = projects.get(project_id)
    if project is None:
        raise web.HTTPNotFound(text="CMK video project not found")
    try:
        folder = await asyncio.to_thread(_open_project_folder, project)
    except OSError as error:
        raise web.HTTPInternalServerError(text=str(error)) from error
    return web.json_response({"opened": True, "folder": folder.name})


@PromptServer.instance.routes.get("/cmk/showcase-workflows")
async def cmk_showcase_workflows(request):
    filename = request.query.get("file")
    if filename:
        path = _SHOWCASE_WORKFLOWS / Path(filename).name
        if path.parent != _SHOWCASE_WORKFLOWS or path.suffix.lower() != ".json" or not path.is_file():
            raise web.HTTPNotFound(text="Showcase workflow not found")
        try:
            return web.json_response(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise web.HTTPInternalServerError(text=f"Showcase workflow could not be read: {error}") from error

    entries = []
    for path in _SHOWCASE_WORKFLOWS.glob("*.json"):
        name = path.stem
        metadata_path = _SHOWCASE_METADATA / path.name
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        if not metadata.get("published", False):
            continue
        entries.append({
            "filename": path.name,
            "name": metadata.get("displayName", name),
            "kind": metadata.get("kind", "standard"),
            "order": metadata.get("order", 999),
            "category": metadata.get("category", "Beispiel"),
            "category_en": metadata.get("category_en", "Example"),
            "description": metadata.get("description", ""),
            "description_en": metadata.get("description_en", ""),
            "cmkHighlight": metadata.get("cmkHighlight", ""),
            "cmkHighlight_en": metadata.get("cmkHighlight_en", ""),
            "previews": metadata.get("previews", []),
        })
    entries.sort(key=lambda item: (item["order"], item["name"].casefold()))
    return web.json_response({"workflows": entries})

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
