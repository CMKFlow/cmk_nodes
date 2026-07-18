import json

from ..utils.comfy_preview_output import render_summary_panel, save_preview_png, ui_images

def cmk_bool(value):
    return "Enabled" if bool(value) else "Disabled"


def cmk_clean_text(value):
    if value is None:
        return ""
    return str(value).strip()




def cmk_format_loras(value):
    """Return a stable human-readable representation of effective LoRAs."""
    if value is None:
        return "None"
    if isinstance(value, str):
        text = value.strip()
        return text if text else "None"
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                name = item.get("name") or item.get("lora_name") or item.get("path") or ""
                model_strength = item.get("model_strength", item.get("strength_model", item.get("strength", 1.0)))
                clip_strength = item.get("clip_strength", item.get("strength_clip", model_strength))
                if name:
                    try:
                        if abs(float(model_strength) - float(clip_strength)) > 0.001:
                            text = f"<lora:{name}:{float(model_strength):g}:{float(clip_strength):g}>"
                        else:
                            text = f"<lora:{name}:{float(model_strength):g}>"
                    except Exception:
                        text = str(name)
                else:
                    text = str(item).strip()
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return " ".join(parts) if parts else "None"
    text = str(value).strip()
    return text if text else "None"


def cmk_make_block(title, order, lines=None, enabled=True):
    clean_lines = []
    for line in (lines or []):
        if line is None:
            continue
        text = str(line).rstrip()
        if text != "":
            clean_lines.append(text)
    return {
        "order": int(order),
        "title": str(title),
        "lines": clean_lines,
        "enabled": bool(enabled),
    }


def cmk_add_block(log_pipe, title, order, lines=None, enabled=True):
    base = dict(log_pipe) if isinstance(log_pipe, dict) else {"blocks": []}
    blocks = list(base.get("blocks", []))
    if enabled:
        block = cmk_make_block(title, order, lines, enabled)
        if block["lines"]:
            blocks.append(block)
    base["blocks"] = blocks
    return base


def cmk_block_to_string(title, order, lines=None, enabled=True):
    payload = {
        "cmk_log_block": True,
        "block": cmk_make_block(title, order, lines, enabled),
    }
    return "CMK_LOG_BLOCK::" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def cmk_parse_block_string(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("CMK_LOG_BLOCK::"):
        text = text.split("::", 1)[1]
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("cmk_log_block"):
        return None
    block = data.get("block")
    if not isinstance(block, dict):
        return None
    return cmk_make_block(
        block.get("title", ""),
        block.get("order", 9999),
        block.get("lines", []),
        block.get("enabled", True),
    )


_MAX_LOG_BLOCK_INPUTS = 32


class CMKLogConcat:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {"log_block": ("CMK_LOG_BLOCK",)}
        for index in range(2, _MAX_LOG_BLOCK_INPUTS + 1):
            optional[f"log_block_{index}"] = ("CMK_LOG_BLOCK",)
        return {
            "required": {"LOG": ("CMK_LOG_PIPE",)},
            "optional": optional,
        }

    RETURN_TYPES = ("CMK_LOG_PIPE",)
    RETURN_NAMES = ("LOG",)
    FUNCTION = "concat"
    CATEGORY = "CMK/Toolbox/Diagnostics"

    def concat(self, LOG, log_block=None, **kwargs):
        base = dict(LOG) if isinstance(LOG, dict) else {"blocks": []}
        blocks = list(base.get("blocks", []))

        values = [log_block]
        for index in range(2, _MAX_LOG_BLOCK_INPUTS + 1):
            values.append(kwargs.get(f"log_block_{index}"))

        for value in values:
            block = cmk_parse_block_string(value)
            if isinstance(block, dict) and block.get("enabled", True):
                lines = [str(line).rstrip() for line in block.get("lines", []) if str(line).strip() != ""]
                if lines:
                    blocks.append(cmk_make_block(block.get("title", ""), block.get("order", 9999), lines, True))

        base["blocks"] = blocks
        return (base,)


def cmk_render_log(log_pipe):
    if not isinstance(log_pipe, dict):
        return ""

    blocks = [
        block
        for block in log_pipe.get("blocks", [])
        if isinstance(block, dict) and block.get("enabled", True)
    ]
    blocks.sort(
        key=lambda block: (
            int(block.get("order", 9999)),
            str(block.get("title", "")),
        )
    )

    rendered = []
    for block in blocks:
        title = str(block.get("title", "")).strip()
        lines = [
            str(line).rstrip()
            for line in block.get("lines", [])
            if str(line).strip() != ""
        ]

        if not title and not lines:
            continue
        if title:
            rendered.append(title)
            rendered.append("-" * len(title))
        rendered.extend(lines)
        rendered.append("")

    return "\n".join(rendered).rstrip()


class CMKLogCreate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("CMK_LOG_PIPE",)
    RETURN_NAMES = ("log_pipe",)
    FUNCTION = "create_log"
    CATEGORY = "CMK/Toolbox/Diagnostics"

    def create_log(self):
        return ({"blocks": []},)


class CMKLogSetBlock:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "log_pipe": ("CMK_LOG_PIPE",),
                "title": ("STRING", {"default": "Module"}),
                "order": ("INT", {"default": 50, "min": 0, "max": 9999, "step": 1}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CMK_LOG_PIPE",)
    RETURN_NAMES = ("log_pipe",)
    FUNCTION = "set_block"
    CATEGORY = "CMK/Toolbox/Diagnostics"

    def set_block(self, log_pipe, title, order, text, enabled):
        lines = cmk_clean_text(text).splitlines()
        return (cmk_add_block(log_pipe, title, order, lines, enabled),)


class CMKLogExportText:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"log_pipe": ("CMK_LOG_PIPE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "export_text"
    CATEGORY = "CMK/Toolbox/Diagnostics"
    OUTPUT_NODE = True

    def export_text(self, log_pipe):
        log_text = cmk_render_log(log_pipe)
        preview_text = log_text if log_text else "CMK log_pipe is empty."

        panel = render_summary_panel(title="CMK Log Export Text", text=preview_text)
        image_info = save_preview_png(panel, prefix="CMK_log_export")

        return {
            "ui": ui_images([image_info], preview_text)["ui"],
            "result": (log_text,),
        }
