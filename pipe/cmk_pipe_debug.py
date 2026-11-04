from __future__ import annotations

from typing import Any

from ..utils.comfy_preview_output import render_summary_panel, save_preview_png, ui_images


# ComfyUI's frontend only supports exact socket type matching.  A Python-side
# pseudo-union is therefore not connectable in the graph editor.  Pipe Inspect
# intentionally uses the wildcard socket and validates the received value at
# runtime so every current and future CMK process/function pipe can be inspected.
CMK_PROCESS_PIPE = "*"


def _shape(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "-"
    try:
        return "×".join(str(int(x)) for x in shape)
    except Exception:
        return str(shape)


def _type_name(value: Any) -> str:
    if value is None:
        return "missing"
    return type(value).__name__


def _object_id(value: Any) -> str:
    if value is None:
        return "-"
    return hex(id(value))


def _present(value: Any) -> str:
    return "yes" if value is not None else "no"


def _short(value: Any, max_len: int = 96) -> str:
    if value is None:
        return "missing"
    text = str(value)
    text = text.replace("\n", "\\n")
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _len_or_dash(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        return str(len(value))
    except Exception:
        return "-"


def _image_source(pipe: dict) -> str:
    for key in ("image_face", "image_detailer", "image_final", "image_refiner", "image_1st_pass", "image"):
        if pipe.get(key) is not None:
            return key
    return "missing"


def _line(label: str, value: str) -> str:
    return f"{label:<22}: {value}"

INSPECT_KEYS = (
    "image", "image_1st_pass", "image_refiner", "image_detailer", "image_face", "image_final",
    "model", "model_patched", "clip", "vae",
    "seed", "noise_seed", "steps", "steps_1st_pass",
    "prompt_pos", "prompt_neg", "active_loras", "lora_stack",
    "latent_image", "samples",
)


def _value_detail(pipe: dict, key: str) -> str:
    value = pipe.get(key)
    if key.startswith("image") or key in ("latent_image", "samples"):
        return f"{_present(value)} | {_type_name(value)} | id {_object_id(value)} | shape {_shape(value)}"
    if key in ("model", "model_patched", "clip", "vae"):
        return f"{_present(value)} | {_type_name(value)} | id {_object_id(value)}"
    if key in ("prompt_pos", "prompt_neg", "active_loras", "lora_stack"):
        return f"{_present(value)} | {_type_name(value)} | id {_object_id(value)} | len {_len_or_dash(value)}"
    detail = f"{_present(value)} | {_type_name(value)}"
    if key in ("seed", "noise_seed", "steps", "steps_1st_pass"):
        detail += f" | {_short(value)}"
    return detail


def _table_row(columns: list[str], widths: list[int]) -> str:
    return " | ".join(str(col)[:widths[i]].ljust(widths[i]) for i, col in enumerate(columns))


def _summarize_two_pipes(pipe_a: Any, label_a: str, pipe_b: Any, label_b: str) -> list[str]:
    lines: list[str] = []
    la = label_a.strip() or "A"
    lb = label_b.strip() or "B"
    lines.append("Pipe Compare")
    lines.append("------------")

    if not isinstance(pipe_a, dict) or not isinstance(pipe_b, dict):
        lines.append(_line(la, _type_name(pipe_a)))
        lines.append(_line(lb, _type_name(pipe_b)))
        return lines

    keys_a = sorted(str(k) for k in pipe_a.keys())
    keys_b = sorted(str(k) for k in pipe_b.keys())
    rows = [
        ("keys", str(len(keys_a)), str(len(keys_b))),
        ("image_source", _image_source(pipe_a), _image_source(pipe_b)),
    ]
    rows.extend((key, _value_detail(pipe_a, key), _value_detail(pipe_b, key)) for key in INSPECT_KEYS)

    widths = [22, 58, 58]
    lines.append(_table_row(["key", la, lb], widths))
    lines.append(_table_row(["-" * 22, "-" * 42, "-" * 42], widths))
    for row in rows:
        lines.append(_table_row(list(row), widths))

    only_a = sorted(set(keys_a) - set(keys_b))
    only_b = sorted(set(keys_b) - set(keys_a))
    lines.append("")
    lines.append("key_diff")
    lines.append("--------")
    lines.append(_line(f"only {la}", ", ".join(only_a) if only_a else "-"))
    lines.append(_line(f"only {lb}", ", ".join(only_b) if only_b else "-"))
    return lines


def _summarize_pipe(pipe: Any, label: str) -> list[str]:
    lines: list[str] = []
    title = label.strip() or "Pipe"
    lines.append(title)
    lines.append("-" * len(title))

    if not isinstance(pipe, dict):
        lines.append(_line("type", _type_name(pipe)))
        lines.append("not a CMK pipe dict")
        return lines

    keys = sorted(str(k) for k in pipe.keys())
    lines.append(_line("keys", str(len(keys))))
    lines.append(_line("image_source", _image_source(pipe)))

    for key in INSPECT_KEYS:
        lines.append(_line(key, _value_detail(pipe, key)))

    lines.append("")
    lines.append("all_keys")
    lines.append("--------")
    lines.append(", ".join(keys) if keys else "empty")
    return lines


def _compare(pipe_a: Any, pipe_b: Any) -> list[str]:
    lines: list[str] = []
    if not isinstance(pipe_a, dict) or not isinstance(pipe_b, dict):
        return lines

    keys_a = set(str(k) for k in pipe_a.keys())
    keys_b = set(str(k) for k in pipe_b.keys())
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    lines.append("Compare")
    lines.append("-------")
    lines.append(_line("only A", ", ".join(only_a) if only_a else "-"))
    lines.append(_line("only B", ", ".join(only_b) if only_b else "-"))

    changed = []
    for key in common:
        a = pipe_a.get(key)
        b = pipe_b.get(key)
        if a is b:
            continue
        if type(a).__name__ != type(b).__name__ or _shape(a) != _shape(b) or _short(a, 48) != _short(b, 48):
            changed.append(key)
    lines.append(_line("different", ", ".join(changed) if changed else "-"))
    return lines


class CMKPipeInspect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe_a": (CMK_PROCESS_PIPE,),
                "label_a": ("STRING", {"default": "A", "multiline": False}),
                "label_b": ("STRING", {"default": "B", "multiline": False}),
            },
            "optional": {
                "opt_pipe_b": (CMK_PROCESS_PIPE, {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "inspect_pipe"
    CATEGORY = "CMK/Developer/Diagnostics"
    OUTPUT_NODE = True

    def inspect_pipe(self, pipe_a, label_a="A", label_b="B", opt_pipe_b=None):
        if not isinstance(pipe_a, dict):
            raise TypeError(
                "CMK Pipe Inspect: PROCESS PIPE A must be a CMK process/function pipe dictionary."
            )
        if opt_pipe_b is not None and not isinstance(opt_pipe_b, dict):
            raise TypeError(
                "CMK Pipe Inspect: PROCESS PIPE B must be a CMK process/function pipe dictionary."
            )

        lines = []
        if opt_pipe_b is not None:
            lines.extend(_summarize_two_pipes(pipe_a, label_a, opt_pipe_b, label_b))
            lines.append("")
            lines.extend(_compare(pipe_a, opt_pipe_b))
        else:
            lines.extend(_summarize_pipe(pipe_a, label_a))

        text = "\n".join(lines).rstrip()
        panel = render_summary_panel(title="CMK Pipe Inspect", text=text)
        image_info = save_preview_png(panel, prefix="CMK_pipe_inspect")
        return {
            "ui": ui_images([image_info], text)["ui"],
            "result": (text,),
        }
