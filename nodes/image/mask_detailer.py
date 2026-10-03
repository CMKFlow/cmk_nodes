from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import comfy.samplers
import folder_paths
import nodes

from ...engine.native_detailer import CMKSEG, SEGSDetailer, SEGSPaste
from ...loader.cmk_lora_text_loader import CMKLoRATextLoader
from ...pipe.cmk_log_pipe import cmk_add_block
from ...pipe.cmk_pipe_image import fill_mask_holes as fill_mask_holes_fn
from ...pipe.cmk_visual import empty_visual, register_provider
from ...pipe.loaders.checkpoint_vae_loader import resolve_postprocess_model


MASK_DETAILER_PROCESS = "CMK_MASK_DETAILER_PROCESS"
MASK_DETAILER_SOURCE = "CMK_MASK_DETAILER_SOURCE"


class _CMKAnyType(str):
    def __ne__(self, other):
        return False


CMK_TERMINAL_INPUT = _CMKAnyType("*")


def _node(name, method, **kwargs):
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"CMK Mask Detailer: required ComfyUI node is unavailable: {name}")
    result = getattr(cls(), method)(**kwargs)
    if isinstance(result, dict):
        result = result.get("result", result)
    return result if isinstance(result, (tuple, list)) else (result,)


def _mask_bhw(mask, image):
    if mask is None:
        raise ValueError(
            "CMK Mask Detailer: no mask found. Draw the mask in CMK Load Image "
            "or on the locked Mask Detailer Intake snapshot."
        )
    value = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
    value = value.to(device=image.device, dtype=image.dtype)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    elif value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    elif value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"CMK Mask Detailer: unsupported mask shape {tuple(value.shape)}")
    if value.shape[0] == 1 and image.shape[0] > 1:
        value = value.expand(image.shape[0], -1, -1)
    if tuple(value.shape[-2:]) != tuple(image.shape[1:3]):
        value = F.interpolate(value.unsqueeze(1), size=image.shape[1:3], mode="bilinear", align_corners=False)[:, 0]
    value = value.clamp(0.0, 1.0)
    if float(value.max().detach().cpu()) <= 0.0:
        raise ValueError(
            "CMK Mask Detailer: the selected mask is empty. Paint the locked Intake image first."
        )
    return value


def _grow(mask, pixels):
    amount = max(0, int(pixels))
    if amount == 0:
        return mask
    return F.max_pool2d(mask.unsqueeze(1), amount * 2 + 1, stride=1, padding=amount)[:, 0]


def _feather(mask, pixels):
    amount = max(0, int(pixels))
    if amount == 0:
        return mask
    kernel = amount * 2 + 1
    return F.avg_pool2d(mask.unsqueeze(1), kernel, stride=1, padding=amount)[:, 0].clamp(0.0, 1.0)


def _mask_to_segs(image, mask, crop_factor=2.0):
    """Create one Detailer segment per image from the authoritative mask."""
    height, width = int(image.shape[1]), int(image.shape[2])
    items = []
    for batch_index in range(int(image.shape[0])):
        frame_mask = mask[min(batch_index, int(mask.shape[0]) - 1)]
        active = torch.nonzero(frame_mask > 0.01, as_tuple=False)
        if active.numel() == 0:
            continue
        y1 = int(active[:, 0].min().item())
        y2 = int(active[:, 0].max().item()) + 1
        x1 = int(active[:, 1].min().item())
        x2 = int(active[:, 1].max().item()) + 1
        bbox_w, bbox_h = max(1, x2 - x1), max(1, y2 - y1)
        crop_w = min(width, max(bbox_w, int(round(bbox_w * float(crop_factor)))))
        crop_h = min(height, max(bbox_h, int(round(bbox_h * float(crop_factor)))))
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        crop_x1 = max(0, min(width - crop_w, center_x - crop_w // 2))
        crop_y1 = max(0, min(height - crop_h, center_y - crop_h // 2))
        crop_x2, crop_y2 = crop_x1 + crop_w, crop_y1 + crop_h
        items.append(CMKSEG(
            cropped_image=image[batch_index:batch_index + 1, crop_y1:crop_y2, crop_x1:crop_x2, :],
            cropped_mask=frame_mask[crop_y1:crop_y2, crop_x1:crop_x2].unsqueeze(0),
            confidence=1.0,
            crop_region=(crop_x1, crop_y1, crop_x2, crop_y2),
            bbox=(x1, y1, x2, y2),
            label="mask",
            control_net_wrapper=None,
            batch_index=batch_index,
        ))
    return ((width, height), items)


def _snapshot_path(value):
    root = Path(folder_paths.get_input_directory()).resolve()
    try:
        path = Path(folder_paths.get_annotated_filepath(str(value or ""))).resolve()
    except Exception:
        path = (root / str(value or "")).resolve()
    if not path.is_relative_to(root):
        raise ValueError("CMK Mask Detailer Intake: snapshot must be inside ComfyUI/input.")
    prefix = "clipspace-painted-masked-"
    if path.name.startswith(prefix):
        clean = path.with_name("clipspace-mask-" + path.name[len(prefix):])
        if clean.is_file():
            path = clean
    return path


def _load_snapshot(value):
    path = _snapshot_path(value)
    if not path.is_file():
        raise ValueError("CMK Mask Detailer Intake: no locked snapshot. Use CAPTURE CURRENT IMAGE first.")
    with Image.open(path) as opened:
        frame = ImageOps.exif_transpose(opened.copy())
    alpha = (
        np.asarray(frame.getchannel("A"), dtype=np.float32) / 255.0
        if "A" in frame.getbands()
        else np.ones((frame.height, frame.width), dtype=np.float32)
    )
    image = torch.from_numpy(np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0)[None, ...]
    mask = torch.from_numpy(1.0 - alpha)[None, ...]
    return image, mask


def _safe_node_id(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "mask-detailer")).strip("-") or "mask-detailer"


def _snapshot_files(unique_id):
    folder = Path(folder_paths.get_input_directory()) / "cmk_mask_detailer"
    node_id = _safe_node_id(unique_id)
    snapshot_stem = f"snapshot-{node_id}"
    incoming_stem = f"incoming-{node_id}"
    return (
        folder,
        folder / f"{snapshot_stem}.png",
        folder / f"{snapshot_stem}.json",
        folder / f"{incoming_stem}.png",
        folder / f"{incoming_stem}.json",
    )


def _image_pixels(image):
    pixels = image[0:1].detach().cpu().clamp(0.0, 1.0).numpy()[0]
    return np.rint(pixels * 255.0).astype(np.uint8)


def _image_fingerprint(image):
    pixels = _image_pixels(image)
    digest = hashlib.sha256()
    digest.update(str(tuple(pixels.shape)).encode("ascii"))
    digest.update(pixels.tobytes())
    return digest.hexdigest()


def _read_metadata(path):
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return stored if isinstance(stored, dict) else {}


def _write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_image_pixels(image), mode="RGB").save(path)


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if not isinstance(item, torch.Tensor)}
    return str(value)


def _source_metadata(process):
    source = process if isinstance(process, dict) else {}
    return {
        key: _json_safe(source.get(key))
        for key in (
            "type", "version", "prompt_pos", "prompt_neg", "source_model_family",
            "model_family", "filename_string", "file_name",
        )
        if source.get(key) is not None
    }


def _terminal_source(image, mask, metadata, log, snapshot, ready=True):
    return {
        "type": MASK_DETAILER_SOURCE,
        "version": 1,
        "image": image,
        "mask": mask,
        "prompt_pos": str(metadata.get("prompt_pos") or ""),
        "prompt_neg": str(metadata.get("prompt_neg") or ""),
        "source_model_family": metadata.get("source_model_family", "image"),
        "model_family": metadata.get("model_family", "sdxl"),
        "filename_string": metadata.get("filename_string", ""),
        "file_name": metadata.get("file_name", metadata.get("filename_string", "")),
        "snapshot": str(snapshot),
        "terminal": True,
        "ready": bool(ready),
        "log": log,
    }


def _terminal_process(metadata, snapshot):
    process = dict(metadata) if isinstance(metadata, dict) else {}
    process.update({
        "terminal": True,
        "snapshot": str(snapshot),
        "result_contract": "mask_detailer_intake",
    })
    return process


class CMKMaskDetailerIntake:
    """Terminal, user-controlled hand-off from a completed Flow into Mask Detailer."""

    @classmethod
    def _available_snapshots(cls):
        root = Path(folder_paths.get_input_directory())
        try:
            files = [str(path.relative_to(root)) for path in root.rglob("*.png") if path.is_file()]
        except OSError:
            files = []
        # ComfyUI's mask editor persists its generated media with an
        # `` [input]`` annotation.  Keep both spellings selectable so a
        # locked, painted snapshot survives a workflow rebuild.
        values = {value for name in files for value in (name, f"{name} [input]")}
        return ["None"] + sorted(values)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PROCESS": (CMK_TERMINAL_INPUT, {"lazy": True}),
                "IMAGE": ("IMAGE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "capture_current": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # ComfyUI's native mask editor updates a widget named
                # `image` to its generated clipspace-painted-masked file.
                "image": (cls._available_snapshots(), {"image_upload": True, "label": "LOCKED IMAGE"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (CMK_TERMINAL_INPUT, "IMAGE", "CMK_LOG_PIPE", MASK_DETAILER_SOURCE, "BOOLEAN")
    RETURN_NAMES = ("PROCESS", "IMAGE", "LOG", "MASK DETAILER SOURCE", "READY")
    FUNCTION = "intake"
    CATEGORY = "CMK/Toolbox/Image"

    def check_lazy_status(self, capture_current=False, PROCESS=None, IMAGE=None, LOG=None, **kwargs):
        if bool(capture_current):
            return []
        return [name for name, value in (("PROCESS", PROCESS), ("IMAGE", IMAGE), ("LOG", LOG)) if value is None]

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, image="None", capture_current=False, **kwargs):
        if bool(capture_current):
            return float("nan")
        path = _snapshot_path(image)
        if not path.is_file():
            return str(image)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def intake(PROCESS=None, IMAGE=None, LOG=None, image="None", capture_current=False, unique_id=None):
        folder, image_path, metadata_path, incoming_path, incoming_metadata_path = _snapshot_files(unique_id)
        snapshot = image
        if bool(capture_current):
            if not incoming_path.is_file():
                raise ValueError("CMK Mask Detailer Intake: no current input image is waiting for capture.")
            with Image.open(incoming_path) as opened:
                frame = ImageOps.exif_transpose(opened.copy()).convert("RGB")
            folder.mkdir(parents=True, exist_ok=True)
            frame.save(image_path)
            metadata = _read_metadata(incoming_metadata_path)
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            snapshot = str(image_path.relative_to(Path(folder_paths.get_input_directory())))
            image, _unused_mask = _load_snapshot(snapshot)
            mask = torch.zeros(image.shape[0:3], dtype=image.dtype, device=image.device)
            base_log = metadata.get("log")
            status = "CAPTURED · MASK NOW"
        else:
            if not isinstance(IMAGE, torch.Tensor):
                raise ValueError("CMK Mask Detailer Intake: IMAGE must be connected.")
            current_image = IMAGE[0:1].detach().cpu()
            current_fingerprint = _image_fingerprint(current_image)
            metadata = _read_metadata(metadata_path)
            locked_fingerprint = metadata.get("fingerprint")
            if not locked_fingerprint and image_path.is_file():
                locked_image, _unused_mask = _load_snapshot(
                    str(image_path.relative_to(Path(folder_paths.get_input_directory())))
                )
                locked_fingerprint = _image_fingerprint(locked_image)
                metadata["fingerprint"] = locked_fingerprint
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

            snapshot_value = str(snapshot or "").strip()
            # The frontend stores the first live preview in the hidden image
            # widget so ComfyUI's mask editor has a file anchor. That incoming
            # file is still only a candidate, never a locked snapshot.
            is_pending_snapshot = Path(snapshot_value).name.startswith("incoming-")
            has_locked_image = (
                snapshot_value.lower() not in ("", "none")
                and not is_pending_snapshot
            )
            if not has_locked_image or current_fingerprint != locked_fingerprint:
                _write_image(incoming_path, current_image)
                incoming_metadata = {
                    "fingerprint": current_fingerprint,
                    "process": _source_metadata(PROCESS),
                    "log": _json_safe(LOG),
                }
                incoming_metadata_path.write_text(
                    json.dumps(incoming_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                pending = str(incoming_path.relative_to(Path(folder_paths.get_input_directory())))
                descriptor = {
                    "filename": incoming_path.name,
                    "subfolder": str(incoming_path.parent.relative_to(Path(folder_paths.get_input_directory()))),
                    "type": "input",
                }
                mask = torch.zeros(current_image.shape[0:3], dtype=current_image.dtype, device=current_image.device)
                source = _terminal_source(
                    current_image, mask, incoming_metadata["process"], LOG, pending, ready=False,
                )
                process = PROCESS if isinstance(PROCESS, dict) else _terminal_process(
                    incoming_metadata["process"], pending,
                )
                return {
                    "ui": {
                        "images": [descriptor],
                        "cmk_mask_preview": [pending],
                        "cmk_mask_status": ["LIVE · CAPTURE CURRENT"],
                        "cmk_mask_lock": ["unlocked"],
                    },
                    "result": (process, current_image, LOG, source, False),
                }

            image, mask = _load_snapshot(snapshot)
            base_log = LOG if LOG is not None else metadata.get("log")
            if float(mask.max().detach().cpu()) <= 0.0:
                descriptor = {
                    "filename": Path(snapshot).name,
                    "subfolder": str(Path(snapshot).parent) if str(Path(snapshot).parent) != "." else "",
                    "type": "input",
                }
                source = _terminal_source(
                    image, mask, metadata.get("process", {}), base_log, snapshot, ready=False,
                )
                process = PROCESS if isinstance(PROCESS, dict) else _terminal_process(
                    metadata.get("process", {}), snapshot,
                )
                return {
                    "ui": {
                        "images": [descriptor],
                        "cmk_mask_snapshot": [snapshot],
                        "cmk_mask_status": ["LOCKED · PAINT MASK"],
                        "cmk_mask_lock": ["locked"],
                    },
                    "result": (process, image, base_log, source, False),
                }
            status = "LOCKED"

        process_metadata = metadata.get("process") if isinstance(metadata.get("process"), dict) else {}
        log = cmk_add_block(base_log, "Mask Detailer Intake", 49, [
            f"STATUS           : {status}",
            f"SNAPSHOT         : {snapshot}",
            "FLOW CONTRACT    : TERMINATED",
            "NEXT             : MASK DETAILER / VISUALIZER ONLY",
        ], True)
        ready = not bool(capture_current)
        source = _terminal_source(image, mask, process_metadata, log, snapshot, ready=ready)
        process = PROCESS if isinstance(PROCESS, dict) else _terminal_process(process_metadata, snapshot)
        descriptor = {
            "filename": Path(snapshot).name,
            "subfolder": str(Path(snapshot).parent) if str(Path(snapshot).parent) != "." else "",
            "type": "input",
        }
        return {
            "ui": {
                "images": [descriptor],
                "cmk_mask_snapshot": [snapshot],
                "cmk_mask_status": [status],
                "cmk_mask_lock": ["locked"],
            },
            "result": (process, image, log, source, ready),
        }


class CMKMaskDetailerFlowGate:
    """Lazily select the untouched terminal image or the completed detailer result."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"enabled": ("BOOLEAN", {"default": True})},
            "optional": {
                "READY": ("BOOLEAN", {"lazy": True}),
                "MODEL BYPASS": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS BYPASS": (CMK_TERMINAL_INPUT, {"lazy": True}),
                "IMAGE BYPASS": ("IMAGE", {"lazy": True}),
                "LOG BYPASS": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL BYPASS": ("CMK_VISUAL_PIPE", {"lazy": True}),
                "MODEL ACTIVE": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS ACTIVE": (MASK_DETAILER_PROCESS, {"lazy": True}),
                "IMAGE ACTIVE": ("IMAGE", {"lazy": True}),
                "LOG ACTIVE": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL ACTIVE": ("CMK_VISUAL_PIPE", {"lazy": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE", MASK_DETAILER_PROCESS, "IMAGE",
        "CMK_LOG_PIPE", "CMK_VISUAL_PIPE",
    )
    RETURN_NAMES = ("MODEL", "MASK DETAILER PROCESS", "IMAGE", "LOG", "VISUAL")
    FUNCTION = "gate"
    CATEGORY = "CMK/Developer/Internal"
    DEV_ONLY = True

    @staticmethod
    def _bypass_names():
        return (
            "MODEL BYPASS", "PROCESS BYPASS", "IMAGE BYPASS",
            "LOG BYPASS",
        )

    @staticmethod
    def _active_names():
        return (
            "MODEL ACTIVE", "PROCESS ACTIVE", "IMAGE ACTIVE",
            "LOG ACTIVE", "VISUAL ACTIVE",
        )

    @staticmethod
    def _pending_connected_visual(name, prompt, unique_id, inputs):
        current = prompt.get(str(unique_id), {}) if isinstance(prompt, dict) else {}
        connected = name in ((current.get("inputs") or {}) if isinstance(current, dict) else {})
        return [name] if connected and inputs.get(name) is None else []

    def check_lazy_status(
        self,
        enabled=True,
        READY=None,
        prompt=None,
        unique_id=None,
        **inputs,
    ):
        if not bool(enabled):
            missing = [name for name in self._bypass_names() if inputs.get(name) is None]
            return missing[:1] or self._pending_connected_visual(
                "VISUAL BYPASS", prompt, unique_id, inputs
            )
        if READY is None:
            return ["READY"]
        names = self._active_names() if bool(READY) else self._bypass_names()
        missing = [name for name in names if inputs.get(name) is None]
        if missing:
            return missing[:1]
        if not bool(READY):
            return self._pending_connected_visual(
                "VISUAL BYPASS", prompt, unique_id, inputs
            )
        return []

    @staticmethod
    def _bypass_process(process, image, reason):
        result = _source_metadata(process)
        result.update({
            "type": MASK_DETAILER_PROCESS,
            "version": 1,
            "image": image,
            "terminal": True,
            "mask_detailer_bypassed": True,
            "mask_detailer_bypass_reason": str(reason),
            "result_contract": "mask_detailer",
            "source_model_family": result.get("source_model_family", "image"),
            "model_family": "sdxl",
        })
        return result

    def gate(self, enabled=True, READY=None, unique_id=None, **inputs):
        active = bool(enabled) and bool(READY)
        if active:
            missing = [name for name in self._active_names() if inputs.get(name) is None]
            if missing:
                raise ValueError("CMK Mask Detailer Flow Gate is missing " + ", ".join(missing))
            return tuple(inputs[name] for name in self._active_names())

        missing = [name for name in self._bypass_names() if inputs.get(name) is None]
        if missing:
            raise ValueError("CMK Mask Detailer Flow Gate is missing " + ", ".join(missing))
        image = inputs["IMAGE BYPASS"]
        process = self._bypass_process(
            inputs["PROCESS BYPASS"], image,
            "disabled" if not bool(enabled) else "snapshot not ready",
        )
        reason = "disabled" if not bool(enabled) else "snapshot not ready"
        visual = register_provider(
            inputs.get("VISUAL BYPASS") or empty_visual(),
            module_instance_id=unique_id or "mask-detailer-bypass",
            module_type="CMKMaskDetailer",
            module_label="Mask Detailer",
            sequence=95,
            channels={"before": image, "after": image},
            status="disabled" if not bool(enabled) else "waiting",
            branch="mask-detailer",
            stage_key="mask-detailer.bypass",
        )
        return (
            inputs["MODEL BYPASS"], process, image,
            inputs["LOG BYPASS"], visual,
        )


class CMKMaskDetailerPrepare:
    """Create the isolated mask-detailer process from an image or terminal snapshot."""

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + [
            name for name in folder_paths.get_filename_list("loras")
            if str(name).strip().lower() != "none"
        ]
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE",),
                "lora_name": (loras, {"default": "None"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.05}),
                "prompt_pos": ("STRING", {"default": "", "multiline": False}),
                "fill_mask_holes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "MASK DETAILER SOURCE": (MASK_DETAILER_SOURCE,),
                "PROCESS": ("CMK_RESULT_PROCESS",),
                "IMAGE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
                "VISUAL": ("CMK_VISUAL_PIPE",),
            },
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", MASK_DETAILER_PROCESS, "CMK_LOG_PIPE", "CMK_VISUAL_PIPE")
    RETURN_NAMES = ("MODEL", "MASK DETAILER PROCESS", "LOG", "VISUAL")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Toolbox/Image"

    @staticmethod
    def prepare(MODEL, lora_name, lora_strength, prompt_pos, fill_mask_holes=False, LOG=None, VISUAL=None, **kwargs):
        terminal_source = kwargs.get("MASK DETAILER SOURCE")
        source = terminal_source if isinstance(terminal_source, dict) else kwargs.get(
            "PROCESS", kwargs.get("PROCESS SDXL")
        )
        IMAGE = source.get("image") if isinstance(terminal_source, dict) else kwargs.get("IMAGE")
        if not isinstance(MODEL, dict):
            raise TypeError("CMK Mask Detailer Prepare: MODEL must come from CMK Checkpoint & VAE.")
        MODEL = resolve_postprocess_model(MODEL)
        if not isinstance(source, dict):
            raise TypeError("CMK Mask Detailer Prepare: connect CMK Load Image or Mask Detailer Intake.")
        if not isinstance(IMAGE, torch.Tensor):
            raise TypeError("CMK Mask Detailer Prepare: IMAGE is missing.")
        if isinstance(terminal_source, dict):
            if terminal_source.get("type") != MASK_DETAILER_SOURCE or not terminal_source.get("terminal"):
                raise TypeError("CMK Mask Detailer Prepare: invalid terminal source.")
            if LOG is None:
                LOG = terminal_source.get("log")
        model, clip, vae = MODEL.get("model"), MODEL.get("clip"), MODEL.get("vae")
        if any(item is None for item in (model, clip, vae)):
            raise ValueError("CMK Mask Detailer Prepare: MODEL is missing model, clip or vae.")
        mask = _mask_bhw(source.get("mask"), IMAGE)
        if bool(fill_mask_holes):
            mask = fill_mask_holes_fn(mask)
        lora = str(lora_name or "None")
        loaded_loras = ""
        if lora.lower() != "none":
            syntax = f"<lora:{lora}:{float(lora_strength):g}>"
            model, clip, _trigger, loaded_loras = CMKLoRATextLoader().load_loras(
                model, clip, opt_lora_syntax=syntax
            )
        own_prompt = str(prompt_pos or "").strip()
        positive_text = "\n".join(
            text for text in (str(source.get("prompt_pos") or "").strip(), own_prompt) if text
        )
        negative_text = str(source.get("prompt_neg") or "").strip()
        positive = _node("CLIPTextEncode", "encode", clip=clip, text=positive_text)[0]
        negative = _node("CLIPTextEncode", "encode", clip=clip, text=negative_text)[0]
        process = {
            "type": MASK_DETAILER_PROCESS,
            "version": 1,
            "model": model,
            "clip": clip,
            "vae": vae,
            "image": IMAGE,
            "mask": mask,
            "positive": positive,
            "negative": negative,
            "prompt_pos": positive_text,
            "prompt_neg": negative_text,
            "lora": loaded_loras,
            "fill_mask_holes": bool(fill_mask_holes),
            "source_model_family": source.get("source_model_family", "image"),
            "model_family": "sdxl",
            "result_contract": "mask_detailer",
            "filename_string": source.get("filename_string", ""),
            "file_name": source.get("file_name", source.get("filename_string", "")),
        }
        log = cmk_add_block(LOG, "Mask Detailer Prepare", 50, [
            "MODEL SOURCE     : MODEL",
            f"CHECKPOINT       : {MODEL.get('ckpt_name', 'Unknown')}",
            f"VAE              : {MODEL.get('vae_name', 'Unknown')} ({MODEL.get('vae_source', 'unknown')})",
            f"MASK             : {int(mask.shape[-1])} × {int(mask.shape[-2])}",
            f"FILL MASK HOLES  : {'ON' if bool(fill_mask_holes) else 'OFF'}",
            f"LORA             : {loaded_loras or 'None'}",
            f"PROMPT ADDITION  : {own_prompt or 'None'}",
        ], True)
        return (MODEL, process, log, empty_visual() if VISUAL is None else VISUAL)


class CMKMaskDetailerProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE",),
                "MASK DETAILER PROCESS": (MASK_DETAILER_PROCESS,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": "fixed"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.45, "min": 0.01, "max": 1.0, "step": 0.01}),
                "mask_expand": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "mask_feather": ("INT", {"default": 12, "min": 0, "max": 128, "step": 1}),
            },
            "optional": {
                "LOG": ("CMK_LOG_PIPE",),
                "VISUAL": ("CMK_VISUAL_PIPE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", MASK_DETAILER_PROCESS, "IMAGE", "CMK_LOG_PIPE", "CMK_VISUAL_PIPE")
    RETURN_NAMES = ("MODEL", "MASK DETAILER PROCESS", "IMAGE", "LOG", "VISUAL")
    FUNCTION = "process"
    CATEGORY = "CMK/Toolbox/Image"

    @staticmethod
    def process(MODEL, seed, steps, cfg, sampler_name, scheduler, denoise, mask_expand, mask_feather, LOG=None, VISUAL=None, unique_id=None, **kwargs):
        pipe = kwargs.get("MASK DETAILER PROCESS")
        if not isinstance(pipe, dict) or pipe.get("type") != MASK_DETAILER_PROCESS:
            raise TypeError("CMK Mask Detailer Process: invalid process family.")
        if not isinstance(MODEL, dict):
            raise TypeError("CMK Mask Detailer Process: MODEL must come from Mask Detailer Prepare.")
        if any(pipe.get(key) is None for key in ("model", "clip", "vae")):
            raise ValueError("CMK Mask Detailer Process: process is missing model, clip or vae.")
        before = pipe["image"]
        visual = empty_visual() if VISUAL is None else VISUAL
        mask = _grow(_mask_bhw(pipe.get("mask"), before), mask_expand)
        segs = _mask_to_segs(before, mask, crop_factor=2.0)
        detailed = SEGSDetailer().doit(
            image=before,
            segs=segs,
            guide_size=512,
            guide_size_for=True,
            max_size=1024,
            seed=int(seed),
            steps=int(steps),
            cfg=float(cfg),
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=float(denoise),
            noise_mask=True,
            force_inpaint=True,
            basic_pipe=(pipe["model"], pipe["clip"], pipe["vae"], pipe["positive"], pipe["negative"]),
        )
        after = SEGSPaste.doit(
            image=before,
            segs=detailed,
            feather=int(mask_feather),
            alpha=255,
        )[0]
        result_pipe = dict(pipe)
        result_pipe.update({"image": after, "seed": int(seed), "steps": int(steps), "cfg": float(cfg), "denoise": float(denoise)})
        log = cmk_add_block(LOG, "Mask Detailer Process", 51, [
            "STATUS           : COMPLETED",
            f"SEGMENTS         : {len(segs[1])}",
            f"STEPS / CFG      : {int(steps)} / {float(cfg):g}",
            f"SAMPLER          : {sampler_name} / {scheduler}",
            f"DENOISE          : {float(denoise):g}",
            f"MASK EXPAND      : {int(mask_expand)} px",
            f"MASK FEATHER     : {int(mask_feather)} px",
            "PASTEBACK        : MASKED SEGS",
        ], True)
        visual = register_provider(
            visual,
            module_instance_id=unique_id or "mask-detailer",
            module_type="CMKMaskDetailer",
            module_label="Mask Detailer",
            sequence=50,
            channels={"before": before, "after": after},
            status="completed",
            branch="mask-detailer",
            stage_key="mask-detailer.result",
        )
        return (MODEL, result_pipe, after, log, visual)
