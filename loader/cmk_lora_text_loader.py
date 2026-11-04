from __future__ import annotations

import os
import re
from typing import Any, Iterable

import folder_paths
import comfy.sd
import comfy.utils

from ..engine.validation import validate_values


_LORA_PATTERN = re.compile(r"<lora:([^:>]+):([^:>]+)(?::([^:>]+))?>", re.IGNORECASE)


def _to_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clean_lora_name(name: str) -> str:
    name = str(name or "").strip()
    if name.startswith("loras" + os.sep):
        name = name[len("loras" + os.sep):]
    if name.startswith("loras/"):
        name = name[len("loras/"):]
    return name.replace("\\", os.sep).replace("/", os.sep)


def _resolve_lora_path(name: str) -> str:
    """Resolve a LoRA name/path against ComfyUI's loras folder.

    Accepts the common CMK/LoraManager forms:
    - folder/name.safetensors
    - folder/name
    - name only, even if the file is stored in a loras subfolder
    - absolute path
    """
    raw_name = str(name or "").strip()
    if not raw_name:
        raise ValueError("LoRA name is empty")

    if os.path.isabs(raw_name) and os.path.exists(raw_name):
        return raw_name

    lora_name = _clean_lora_name(raw_name)
    lora_name_posix = lora_name.replace(os.sep, "/")

    # 1) Exact ComfyUI-relative match.
    direct = folder_paths.get_full_path("loras", lora_name)
    if direct:
        return direct

    # 2) Same relative name with common LoRA file extensions.
    root, ext = os.path.splitext(lora_name)
    if not ext:
        for suffix in (".safetensors", ".pt", ".pth", ".ckpt"):
            direct = folder_paths.get_full_path("loras", root + suffix)
            if direct:
                return direct

    wanted_path = os.path.splitext(lora_name_posix)[0].lower()
    wanted_base = os.path.basename(wanted_path)
    matches: list[tuple[str, str]] = []

    # 3) Search the complete ComfyUI loras index. This covers names coming
    #    from LoraManager, where the UI often exposes only the display/file
    #    stem while the actual file lives in a nested subfolder.
    for candidate in folder_paths.get_filename_list("loras"):
        normalized = candidate.replace("\\", "/")
        normalized_lower = normalized.lower()
        candidate_stem = os.path.splitext(normalized)[0].lower()
        candidate_base = os.path.basename(candidate_stem)

        if (
            normalized_lower == lora_name_posix.lower()
            or candidate_stem == wanted_path
            or candidate_base == wanted_base
        ):
            resolved = folder_paths.get_full_path("loras", candidate)
            if resolved:
                matches.append((candidate, resolved))

    if len(matches) == 1:
        return matches[0][1]

    if len(matches) > 1:
        choices = "\n".join(f"• {candidate}" for candidate, _ in matches[:20])
        extra = "" if len(matches) <= 20 else f"\n• ... {len(matches) - 20} more"
        raise FileNotFoundError(
            "CMK LoRA Text Loader\n\n"
            "LoRA name is ambiguous.\n\n"
            f"Requested:\n• {raw_name}\n\n"
            f"Matches:\n{choices}{extra}"
        )

    raise FileNotFoundError(
        "CMK LoRA Text Loader\n\n"
        "LoRA file not found.\n\n"
        f"Requested:\n• {raw_name}\n\n"
        "Searched in ComfyUI loras folder, including subfolders."
    )


def _display_lora_name(name: str) -> str:
    name = _clean_lora_name(name)
    return name.replace(os.sep, "/")


def _parse_lora_syntax(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not text:
        return entries

    for name, model_strength, clip_strength in _LORA_PATTERN.findall(str(text)):
        model_strength_f = _to_float(model_strength, 1.0)
        clip_strength_f = _to_float(clip_strength, model_strength_f) if clip_strength else model_strength_f
        entries.append(
            {
                "name": _display_lora_name(name),
                "path": _resolve_lora_path(name),
                "model_strength": model_strength_f,
                "clip_strength": clip_strength_f,
            }
        )
    return entries


def _parse_lora_stack(lora_stack: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not lora_stack:
        return entries

    for item in lora_stack:
        if not item:
            continue
        try:
            name = item[0]
            model_strength = item[1] if len(item) > 1 else 1.0
            clip_strength = item[2] if len(item) > 2 else model_strength
        except Exception:
            continue

        entries.append(
            {
                "name": _display_lora_name(name),
                "path": _resolve_lora_path(name),
                "model_strength": _to_float(model_strength, 1.0),
                "clip_strength": _to_float(clip_strength, _to_float(model_strength, 1.0)),
            }
        )
    return entries


def _lora_entry_key(entry: dict[str, Any]) -> tuple[str, float, float]:
    """Return a canonical identity for one exact LoRA application.

    The resolved file path identifies the LoRA independently of whether the
    source used a display name, relative path or absolute path. Strengths are
    part of the identity so deliberately different applications remain valid.
    """
    path = os.path.normcase(os.path.realpath(str(entry["path"])))
    model_strength = round(float(entry["model_strength"]), 8)
    clip_strength = round(float(entry["clip_strength"]), 8)
    return path, model_strength, clip_strength


def _deduplicate_lora_entries(
    entries: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove exact duplicate LoRA applications while preserving source order.

    Stack entries are parsed first and therefore remain authoritative when the
    same LoRA with identical MODEL/CLIP strengths is also present in syntax.
    Unique syntax entries and entries with intentionally different strengths
    are preserved.
    """
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()

    for entry in entries:
        key = _lora_entry_key(entry)
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)

    return unique


def _format_loaded_loras(entries: Iterable[dict[str, Any]]) -> str:
    formatted: list[str] = []
    for entry in entries:
        name = entry["name"]
        model_strength = entry["model_strength"]
        clip_strength = entry["clip_strength"]
        if abs(float(model_strength) - float(clip_strength)) > 0.001:
            formatted.append(f"<lora:{name}:{model_strength:g}:{clip_strength:g}>")
        else:
            formatted.append(f"<lora:{name}:{model_strength:g}>")
    return " ".join(formatted)


class CMKLoRATextLoader:
    """CMK LoRA loader for expert subnodes.

    MODEL and CLIP are required. LoRA syntax and LoRA stack are optional.
    If no LoRA data is provided, the node is a pure model/clip throughpass.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
            "optional": {
                "opt_lora_syntax": (
                    "STRING",
                    {
                        "forceInput": True,
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional. Format: <lora:name:strength> or <lora:name:model_strength:clip_strength>.",
                    },
                ),
                "opt_lora_stack": ("LORA_STACK", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "trigger_words", "loaded_loras")
    FUNCTION = "load_loras"
    CATEGORY = "CMK/Toolbox/Model & LoRA"

    def load_loras(self, model, clip, opt_lora_syntax: str = "", opt_lora_stack=None):
        validate_values(
            "CMK LoRA Text Loader",
            "LoRA Loader",
            {"model": model, "clip": clip},
            ["model", "clip"],
        )

        entries = []
        entries.extend(_parse_lora_stack(opt_lora_stack))
        entries.extend(_parse_lora_syntax(opt_lora_syntax or ""))
        entries = _deduplicate_lora_entries(entries)

        if not entries:
            return (model, clip, "", "")

        for entry in entries:
            lora = comfy.utils.load_torch_file(entry["path"], safe_load=True)
            model, clip = comfy.sd.load_lora_for_models(
                model,
                clip,
                lora,
                float(entry["model_strength"]),
                float(entry["clip_strength"]),
            )

        # CMK v1 intentionally does not depend on LoraManager metadata.
        # Trigger words remain an empty string unless a future CMK-native metadata path is added.
        return (model, clip, "", _format_loaded_loras(entries))
