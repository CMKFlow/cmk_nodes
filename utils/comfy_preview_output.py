from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_CAPTION_MODES = ["Off", "Standard", "Details"]


# -----------------------------------------------------------------------------
# ComfyUI output plumbing
# -----------------------------------------------------------------------------

def _temp_dir() -> Path:
    try:
        import folder_paths
        return Path(folder_paths.get_temp_directory())
    except Exception:
        path = Path(__file__).resolve().parents[1] / "temp"
        path.mkdir(parents=True, exist_ok=True)
        return path


def _counter(prefix: str) -> int:
    directory = _temp_dir()
    existing = sorted(directory.glob(f"{prefix}_*.png"))
    if not existing:
        return 1
    max_id = 0
    for file in existing:
        try:
            max_id = max(max_id, int(file.stem.rsplit("_", 1)[-1]))
        except Exception:
            pass
    return max_id + 1


def _safe_prefix(prefix: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(prefix or "CMK_preview"))
    return cleaned[:80] or "CMK_preview"


def save_preview_png(image_rgb: np.ndarray, *, prefix: str = "CMK_preview") -> Dict[str, str]:
    directory = _temp_dir()
    directory.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(prefix)
    counter = _counter(prefix)
    filename = f"{prefix}_{counter:05d}.png"
    Image.fromarray(_as_rgb(image_rgb)).save(directory / filename)
    return {"filename": filename, "subfolder": "", "type": "temp"}


def ui_images(images: List[Dict[str, str]], summary: str = ""):
    ui = {"images": images}
    if summary:
        ui["text"] = [summary]
    return {"ui": ui}


# -----------------------------------------------------------------------------
# Design constants: image-first diagnostic card, v0.20
# -----------------------------------------------------------------------------

def _design() -> dict:
    return {
        "bg": (18, 18, 18),
        "card_bg": (18, 18, 18),
        "board_bg": (0, 0, 0),
        "text": (246, 246, 246),
        "muted": (170, 170, 170),
        "line": (72, 72, 72),
        "pad": 36,
        "gap": 24,
        "card_gap": 24,
        "title": 52,
        "label": 26,
        "value": 34,
        "preview_h": 640,
        "caption_rows_summary": 5,
        "caption_rows_details": 8,
        "max_title_lines": 2,
    }


def _load_font(size: int, *, bold: bool = False):
    size = max(10, int(size))
    candidates = []
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ])
    for path in candidates:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------

def _as_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return np.zeros((160, 520, 3), dtype=np.uint8)
    return np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    image = _as_rgb(image)
    height = max(64, int(height))
    h, w = image.shape[:2]
    if h == height:
        return image
    scale = height / float(max(1, h))
    new_w = max(1, int(round(w * scale)))
    try:
        return np.asarray(Image.fromarray(image).resize((new_w, height), Image.Resampling.LANCZOS), dtype=np.uint8)
    except Exception:
        return image


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    image = _as_rgb(image)
    h, w = image.shape[:2]
    if h == height:
        return image
    out = np.zeros((height, w, 3), dtype=np.uint8)
    out[:h, :w] = image
    return out


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------

def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    try:
        bb = draw.textbbox((0, 0), str(text), font=font)
        return max(1, int(bb[2] - bb[0])), max(1, int(bb[3] - bb[1]))
    except Exception:
        return max(1, len(str(text)) * 12), 24


def _text_w(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    return _text_bbox(draw, text, font)[0]


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int | None = None) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if _text_w(draw, candidate, font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
        if max_lines is not None and len(lines) >= max_lines:
            break
    if current and (max_lines is None or len(lines) < max_lines):
        lines.append(current)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines or [""]


def _normalize_label(label: str) -> str:
    raw = str(label or "").strip().replace("_", " ")
    if not raw:
        return "Info"
    mapping = {
        "bbox": "Box",
        "score": "Confidence",
        "face count": "Faces",
        "face count avg": "Faces",
        "changed avg": "Changed",
        "coverage avg": "Coverage",
        "restore strength": "Strength",
        "preview mode": "Preview",
        "selected index": "Face",
        "detector model": "Detector",
        "detector size": "Detector Size",
        "mask source": "Mask",
        "draw preview": "Preview",
        "total faces": "Faces",
    }
    low = raw.lower()
    if low in mapping:
        return mapping[low]
    return " ".join(part.capitalize() for part in raw.split())


def _format_value(value: Any, label: str = "") -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return "—"

    lower = text.lower()
    if lower in {"true", "false"}:
        return "Enabled" if lower == "true" else "Disabled"

    label_low = str(label or "").strip().replace("_", " ").lower()

    # Polished known values.
    value_map = {
        "input_mask": "Input",
        "fallback_face_ellipse": "Fallback",
        "buffalo_l": "Buffalo L",
    }
    if lower in value_map:
        return value_map[lower]

    try:
        number = float(text)
        if np.isfinite(number):
            # Labels that are identifiers/counts must never be formatted as percentages.
            if label_low in {"face", "selected face", "selected index", "index", "image index", "detector size", "faces", "total faces", "input detections", "output detections", "target detected faces", "source detected faces"}:
                if abs(number - round(number)) < 1e-6:
                    return str(int(round(number)))
                return f"{number:.2f}"
            if label_low in {"blend", "coverage", "coverage avg", "score", "confidence", "padding"}:
                if 0.0 <= number <= 1.0:
                    return f"{number * 100:.0f} %"
                return f"{number:.0f} %"
            if label_low in {"changed", "changed avg"}:
                # Native restore stores average per-channel difference. Display as compact metric.
                return f"{number:.2f}"
            if 0.0 <= number <= 1.0 and label_low not in {"strength", "restore strength"}:
                return f"{number * 100:.0f} %"
            if abs(number - round(number)) < 1e-6:
                return str(int(round(number)))
            return f"{number:.2f}"
    except Exception:
        pass

    if "." in text:
        import re
        def repl(match):
            try:
                return f"{float(match.group(0)):.2f}"
            except Exception:
                return match.group(0)
        text = re.sub(r"-?\d+\.\d{4,}", repl, text)
    return text


def _summary_items(title: str, summary: str, details: str, metadata: dict, caption: str) -> list[tuple[str, str]]:
    caption = str(caption or "Standard")
    if caption == "Off":
        return []

    max_items = _design()["caption_rows_details"] if caption == "Details" else _design()["caption_rows_summary"]
    source = str(details if caption == "Details" else summary or "")
    items: list[tuple[str, str]] = []

    skip_keys = {"type", "node", "version", "image index", "image_index", "operation"}
    preferred_order = ["enabled", "method", "restore_strength", "blend", "changed_avg", "shape", "padding", "feather", "invert", "coverage_avg", "detector_model", "detector_size", "total_faces", "selection", "selected_index", "score", "bbox", "mask_source"]
    raw_pairs: dict[str, str] = {}
    loose_lines: list[str] = []

    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key_clean = key.strip().replace("_", " ").lower()
            if key_clean in skip_keys:
                continue
            raw_pairs[key.strip()] = value.strip()
        else:
            if caption != "Details" and ("[" in line or "]" in line or len(line) > 80):
                continue
            loose_lines.append(line)

    def add_pair(key: str, value: Any):
        label = _normalize_label(key)
        items.append((label, _format_value(value, key)))

    # Preferred keys first, preserving concise product language.
    for wanted in preferred_order:
        for key in list(raw_pairs.keys()):
            if key.strip().lower() == wanted.lower():
                add_pair(key, raw_pairs.pop(key))
                break
        if len(items) >= max_items:
            return items[:max_items]

    for key, value in raw_pairs.items():
        add_pair(key, value)
        if len(items) >= max_items:
            return items[:max_items]

    if caption == "Details":
        for line in loose_lines:
            items.append(("Info", _format_value(line)))
            if len(items) >= max_items:
                return items[:max_items]

    if not items and metadata:
        for key, value in metadata.items():
            add_pair(str(key), value)
            if len(items) >= max_items:
                break

    if not items:
        items.append(("Status", "Ready"))
    return items[:max_items]




# -----------------------------------------------------------------------------
# Flow preview renderer
# -----------------------------------------------------------------------------

def _image_like_to_rgb(value: Any) -> np.ndarray | None:
    """Convert common CMK/Comfy preview image values to RGB uint8.

    Supports numpy RGB/HWC, Comfy IMAGE tensors (BHWC), single batch tensors,
    and already-rendered RGB panels. Returns None for non-image values.
    """
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "numpy"):
            arr = value.numpy()
        else:
            arr = np.asarray(value)
    except Exception:
        return None

    if arr is None:
        return None
    try:
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim != 3:
            return None
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        arr = arr.astype(np.float32)
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(arr, 0, 255).astype(np.uint8)
    except Exception:
        return None


def _flow_stage_card(image: Any, *, title: str, subtitle: str = "") -> np.ndarray | None:
    """Render one fixed-width CDDL stage card.

    Stage cards keep a stable width. Long labels wrap inside the fixed card
    instead of expanding the layout or being clipped.
    """
    rgb = _image_like_to_rgb(image)
    if rgb is None:
        return None

    spec = _design()
    bg = spec["card_bg"]
    text = spec["text"]
    muted = spec["muted"]
    line = spec["line"]

    pad = 12
    card_w = 260
    image_h = 320
    image_w = card_w - pad * 2
    header_h = 96

    title_font = _load_font(22, bold=True)
    sub_font = _load_font(15, bold=False)

    # Fixed card width: fit the image into the available box without changing
    # the card width. This keeps the whole flow visually disciplined.
    rgb = _as_rgb(rgb)
    ih0, iw0 = rgb.shape[:2]
    scale = min(image_w / float(max(1, iw0)), image_h / float(max(1, ih0)))
    new_w = max(1, int(round(iw0 * scale)))
    new_h = max(1, int(round(ih0 * scale)))
    try:
        rgb = np.asarray(Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)
    except Exception:
        rgb = _resize_to_height(rgb, min(image_h, ih0))
        new_h, new_w = rgb.shape[:2]

    total_h = pad + header_h + image_h + pad
    panel = Image.new("RGB", (card_w, total_h), bg)
    d = ImageDraw.Draw(panel)

    text_w = card_w - pad * 2
    title_lines = _wrap_text(d, str(title or "Stage"), title_font, text_w, max_lines=2)
    subtitle_lines = _wrap_text(d, str(subtitle or ""), sub_font, text_w, max_lines=2) if subtitle else []

    y = pad
    for line_text in title_lines:
        d.text((pad, y), line_text, fill=text, font=title_font)
        y += _text_bbox(d, line_text or "Ag", title_font)[1] + 3

    if subtitle_lines:
        y += 2
        for line_text in subtitle_lines:
            d.text((pad, y), line_text, fill=muted, font=sub_font)
            y += _text_bbox(d, line_text or "Ag", sub_font)[1] + 2

    y_line = pad + header_h - 7
    d.rectangle((pad, y_line, card_w - pad, y_line + 2), fill=line)

    x_img = pad + max(0, (image_w - new_w) // 2)
    y_img = pad + header_h + max(0, (image_h - new_h) // 2)
    panel.paste(Image.fromarray(rgb), (x_img, y_img))
    return np.asarray(panel, dtype=np.uint8)


def _flow_caption_panel(data: dict, *, caption: str, width: int) -> np.ndarray | None:
    if caption == "Off":
        return None

    spec = _design()
    bg = spec["card_bg"]
    text = spec["text"]
    muted = spec["muted"]
    line = spec["line"]
    pad = int(spec["pad"])
    gap = int(spec["gap"])

    width = max(720, int(width or 720))
    text_w = width - pad * 2
    label_font = _load_font(int(spec["label"]), bold=False)
    value_font = _load_font(int(spec["value"]), bold=True)

    summary = str(data.get("summary") or "")
    details = str(data.get("details") or summary or "")
    metadata = dict(data.get("metadata") or {})
    probe = Image.new("RGB", (width, 1), bg)
    draw = ImageDraw.Draw(probe)
    items = _summary_items("", summary, details, metadata, caption)
    caption_h = _draw_caption(draw, items, pad, 0, text_w, label_font, value_font, muted, text)
    total_h = pad + 4 + gap + caption_h + pad

    panel = Image.new("RGB", (width, total_h), bg)
    d = ImageDraw.Draw(panel)
    y = pad
    d.rectangle((pad, y, width - pad, y + 4), fill=line)
    y += 4 + gap
    _draw_caption(d, items, pad, y, text_w, label_font, value_font, muted, text)
    return np.asarray(panel, dtype=np.uint8)


def _preview_steps_from_payload(data: dict) -> list[dict]:
    metadata = dict(data.get("metadata") or {})
    raw_steps = data.get("stages") or data.get("preview_steps") or metadata.get("stages") or metadata.get("preview_steps")
    steps: list[dict] = []

    if isinstance(raw_steps, list):
        for idx, item in enumerate(raw_steps):
            if isinstance(item, dict):
                title = item.get("title") or item.get("label") or f"Stage {idx + 1}"
                subtitle = item.get("subtitle") or item.get("note") or ""
                summary = item.get("summary") or ""
                details = item.get("details") or summary
                step_metadata = dict(item.get("metadata") or {})
                # Do not use ``item.get("image") or ...`` here: torch tensors
                # cannot be evaluated as booleans when they contain more than one
                # value. Treat only None as missing.
                image = item.get("image")
                if image is None:
                    image = item.get("preview")
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                title = item[0]
                image = item[1]
                subtitle = item[2] if len(item) >= 3 else ""
                summary = ""
                details = ""
                step_metadata = {}
            else:
                continue
            if _image_like_to_rgb(image) is not None:
                steps.append({
                    "title": str(title),
                    "subtitle": str(subtitle),
                    "image": image,
                    "summary": str(summary),
                    "details": str(details),
                    "metadata": step_metadata,
                })

    if steps:
        return steps

    previews = data.get("previews") or data.get("preview") or []
    if not isinstance(previews, list):
        previews = [previews]
    labels = metadata.get("preview_panels") or []
    if not isinstance(labels, list):
        labels = []

    for idx, image in enumerate(previews):
        if _image_like_to_rgb(image) is None:
            continue
        title = labels[idx] if idx < len(labels) else f"Preview {idx + 1}"
        steps.append({"title": str(title), "subtitle": "", "image": image})
    return steps


def _render_stage_strip(
    cards: list[np.ndarray],
    *,
    title: str,
    show_title: bool = True,
) -> np.ndarray:
    spec = _design()
    bg = spec["board_bg"]
    text = spec["text"]
    line = spec["line"]
    pad = 28
    gap = 18
    arrow_w = 36

    cards = [_as_rgb(card) for card in cards if card is not None]
    if not cards:
        return np.zeros((240, 720, 3), dtype=np.uint8)

    title_font = _load_font(int(spec["title"]), bold=True)
    arrow_font = _load_font(30, bold=True)

    probe = Image.new("RGB", (1, 1), bg)
    d0 = ImageDraw.Draw(probe)
    title_line_h = _text_bbox(d0, "Ag", title_font)[1] + 8
    title_h = title_line_h * int(spec["max_title_lines"])
    card_h = max(card.shape[0] for card in cards)
    content_w = sum(card.shape[1] for card in cards) + gap * max(0, len(cards) - 1) + arrow_w * max(0, len(cards) - 1)
    width = max(720, content_w + pad * 2)
    header_h = pad + title_h + gap if show_title else pad
    height = header_h + card_h + pad

    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[:, :, :] = bg
    d = ImageDraw.Draw(Image.fromarray(out))

    # Need to draw on a PIL image and convert back, because ImageDraw works on PIL object.
    panel = Image.fromarray(out)
    d = ImageDraw.Draw(panel)
    if show_title:
        _draw_title(
            d,
            str(title or "CMK Preview"),
            pad,
            pad,
            width - pad * 2,
            title_font,
            text,
        )
    y = header_h
    x = pad
    for idx, card in enumerate(cards):
        card = _pad_to_height(card, card_h)
        panel.paste(Image.fromarray(card), (x, y))
        x += card.shape[1]
        if idx < len(cards) - 1:
            arrow_x = x + gap // 2
            arrow_y = y + card_h // 2 - 20
            d.text((arrow_x, arrow_y), "→", fill=line, font=arrow_font)
            x += gap + arrow_w
    return np.asarray(panel, dtype=np.uint8)


def _render_stage_grid(
    cards: list[np.ndarray],
    *,
    title: str,
    columns: int = 4,
    show_title: bool = True,
) -> np.ndarray:
    """Render a chronological diagnostic timeline in compact reading rows."""
    spec = _design()
    bg = spec["board_bg"]
    text = spec["text"]
    line = spec["line"]
    pad = 28
    gap = 18
    columns = max(1, min(int(columns or 4), 6))

    cards = [_as_rgb(card) for card in cards if card is not None]
    if not cards:
        return np.zeros((240, 720, 3), dtype=np.uint8)

    title_font = _load_font(int(spec["title"]), bold=True)
    probe = Image.new("RGB", (1, 1), bg)
    d0 = ImageDraw.Draw(probe)
    title_line_h = _text_bbox(d0, "Ag", title_font)[1] + 8
    title_h = title_line_h * int(spec["max_title_lines"])

    cell_w = max(card.shape[1] for card in cards)
    cell_h = max(card.shape[0] for card in cards)
    rows = (len(cards) + columns - 1) // columns
    used_columns = min(columns, len(cards))
    width = max(720, pad * 2 + used_columns * cell_w + max(0, used_columns - 1) * gap)
    header_h = pad + title_h + gap if show_title else pad
    height = header_h + rows * cell_h + max(0, rows - 1) * gap + pad

    panel = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(panel)
    if show_title:
        _draw_title(
            d,
            str(title or "CMK Preview"),
            pad,
            pad,
            width - pad * 2,
            title_font,
            text,
        )

    for index, card in enumerate(cards):
        row = index // columns
        column = index % columns
        x = pad + column * (cell_w + gap)
        y = header_h + row * (cell_h + gap)
        card = _as_rgb(card)
        x += max(0, (cell_w - card.shape[1]) // 2)
        panel.paste(Image.fromarray(card), (x, y))

    return np.asarray(panel, dtype=np.uint8)


def _stack_vertical(top: np.ndarray, bottom: np.ndarray | None, *, gap: int | None = None) -> np.ndarray:
    top = _as_rgb(top)
    if bottom is None:
        return top
    bottom = _as_rgb(bottom)
    gap = int(_design()["card_gap"] if gap is None else gap)
    width = max(top.shape[1], bottom.shape[1])
    height = top.shape[0] + gap + bottom.shape[0]
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[:, :, :] = _design()["board_bg"]
    out[:top.shape[0], :top.shape[1]] = top
    y = top.shape[0] + gap
    out[y:y + bottom.shape[0], :bottom.shape[1]] = bottom
    return out


def _render_flow_preview_panel(data: dict, *, caption: str = "Standard") -> np.ndarray | None:
    steps = _preview_steps_from_payload(data)
    if len(steps) < 2:
        return None

    cards: list[np.ndarray] = []
    for step in steps:
        card = _flow_stage_card(
            step.get("image"),
            title=str(step.get("title") or "Stage"),
            subtitle=str(step.get("subtitle") or ""),
        )
        if card is not None:
            cards.append(card)
    if len(cards) < 2:
        return None

    metadata = dict(data.get("metadata") or {})
    title = str(data.get("title") or data.get("node") or "CMK Preview")
    if metadata.get("preview_layout") == "grid":
        strip = _render_stage_grid(
            cards,
            title=title,
            columns=int(metadata.get("preview_columns") or 4),
        )
    else:
        strip = _render_stage_strip(cards, title=title)
    # Standard preview boards are image-first and keep one stable module card.
    # Detailed captions remain available explicitly without duplicating titles.
    if caption == "Details":
        caption_panel = _flow_caption_panel(data, caption=caption, width=int(strip.shape[1]))
        return _stack_vertical(strip, caption_panel)
    return strip


def _concat_stage_panels(data: dict) -> list[np.ndarray]:
    """Expand one diagnostic into full-size chronological stage cards.

    Concat is deliberately dumb: it preserves its input order.  A multi-stage
    diagnostic is expanded in place, with every stage rendered as an ordinary
    quartet-style card (title, image, information).  It must never be rendered
    as a miniature timeline inside another process card.
    """
    panels: list[np.ndarray] = []
    for step in _preview_steps_from_payload(data):
        image = _image_like_to_rgb(step.get("image"))
        if image is None:
            continue
        subtitle = str(step.get("subtitle") or "").strip()
        stage_text = str(step.get("summary") or "").strip()
        if not stage_text:
            stage_text = f"Info: {subtitle}" if subtitle else "Status: Ready"
        panels.append(
            _diagnostic_card(
                image,
                title=str(step.get("title") or "Stage"),
                summary=stage_text,
                details=str(step.get("details") or stage_text),
                metadata=dict(step.get("metadata") or {}),
                caption="Details",
            )
        )
    return panels


def _concat_detail_panels(data: dict, *, depth: int = 0) -> list[np.ndarray]:
    """Recursively flatten nested concat diagnostics into chronological cards."""
    if depth < 12 and str(data.get("node") or "") == "CMK Diagnostic Concat":
        groups = (data.get("metadata") or {}).get("diagnostic_groups") or []
        panels: list[np.ndarray] = []
        for group in groups:
            try:
                from .preview_payload import normalize_preview_payload

                group_data = normalize_preview_payload(group)
                panels.extend(_concat_detail_panels(group_data, depth=depth + 1))
            except Exception:
                continue
        if panels:
            return panels

    stage_panels = _concat_stage_panels(data)
    if stage_panels:
        return stage_panels
    return []


# -----------------------------------------------------------------------------
# Diagnostic card renderer
# -----------------------------------------------------------------------------

def _draw_title(draw: ImageDraw.ImageDraw, title: str, x: int, y: int, max_w: int, font, fill) -> int:
    lines = _wrap_text(draw, title or "CMK Preview", font, max_w, max_lines=int(_design()["max_title_lines"]))
    yy = y
    for line in lines:
        draw.text((x, yy), line, fill=fill, font=font)
        yy += _text_bbox(draw, line, font)[1] + 8
    return yy - y


def _fit_text_ellipsis(draw: ImageDraw.ImageDraw, value: str, font, max_w: int) -> str:
    value = str(value or "")
    if _text_w(draw, value, font) <= max_w:
        return value
    suffix = "…"
    if _text_w(draw, suffix, font) > max_w:
        return ""
    low, high = 0, len(value)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = value[:mid].rstrip() + suffix
        if _text_w(draw, candidate, font) <= max_w:
            low = mid
        else:
            high = mid - 1
    return value[:low].rstrip() + suffix


def _draw_caption(
    draw: ImageDraw.ImageDraw,
    items: list[tuple[str, str]],
    x: int,
    y: int,
    max_w: int,
    label_font,
    value_font,
    muted,
    text,
    *,
    wrap_values: bool = True,
) -> int:
    if not items:
        return 0
    col_gap = 24
    label_w = min(max(180, max((_text_w(draw, label, label_font) for label, _ in items), default=180)), max(180, int(max_w * 0.45)))
    yy = y
    row_gap = 12
    for label, value in items:
        label_text = _fit_text_ellipsis(draw, label, label_font, label_w)
        draw.text((x, yy), label_text, fill=muted, font=label_font)
        value_x = x + label_w + col_gap
        value_w = max(80, max_w - label_w - col_gap)
        if wrap_values:
            vlines = _wrap_text(draw, value, value_font, value_w, max_lines=2)
        else:
            vlines = [_fit_text_ellipsis(draw, value, value_font, value_w)]
        first_h = _text_bbox(draw, vlines[0] or "Ag", value_font)[1]
        draw.text((value_x, yy), vlines[0], fill=text, font=value_font)
        if len(vlines) > 1:
            draw.text((value_x, yy + first_h + 4), vlines[1], fill=text, font=value_font)
            yy += first_h * 2 + row_gap + 4
        else:
            yy += max(first_h, _text_bbox(draw, label or "Ag", label_font)[1]) + row_gap
    return yy - y


def _diagnostic_card(image: np.ndarray, *, title: str, summary: str, details: str, metadata: dict, caption: str) -> np.ndarray:
    spec = _design()
    pad = int(spec["pad"])
    gap = int(spec["gap"])
    bg = spec["card_bg"]
    text = spec["text"]
    muted = spec["muted"]
    line = spec["line"]

    title_font = _load_font(int(spec["title"]), bold=True)
    label_font = _load_font(int(spec["label"]), bold=False)
    value_font = _load_font(int(spec["value"]), bold=True)

    # Standard cards use one fixed square preview viewport. Different source
    # aspect ratios are fitted proportionally into it instead of changing the
    # card width or stretching the image.
    image = _as_rgb(image)
    image_box_h = int(spec["preview_h"])
    image_box_w = int(spec["preview_h"])
    ih0, iw0 = image.shape[:2]
    scale = min(
        image_box_w / float(max(1, iw0)),
        image_box_h / float(max(1, ih0)),
    )
    fitted_w = max(1, int(round(iw0 * scale)))
    fitted_h = max(1, int(round(ih0 * scale)))
    try:
        fitted = np.asarray(
            Image.fromarray(image).resize(
                (fitted_w, fitted_h),
                Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )
    except Exception:
        fitted = image
        fitted_h, fitted_w = fitted.shape[:2]

    image = np.zeros((image_box_h, image_box_w, 3), dtype=np.uint8)
    fit_x = max(0, (image_box_w - fitted_w) // 2)
    fit_y = max(0, (image_box_h - fitted_h) // 2)
    copy_h = min(fitted_h, image_box_h)
    copy_w = min(fitted_w, image_box_w)
    image[fit_y:fit_y + copy_h, fit_x:fit_x + copy_w] = fitted[:copy_h, :copy_w]
    ih, iw = image.shape[:2]
    card_w = image_box_w + pad * 2
    text_w = card_w - pad * 2

    probe = Image.new("RGB", (card_w, 1), bg)
    draw = ImageDraw.Draw(probe)
    title_line_h = _text_bbox(draw, "Ag", title_font)[1] + 8
    title_h = title_line_h * int(spec["max_title_lines"])
    items = _summary_items(title, summary, details, metadata, caption)
    caption_h = 0
    if caption != "Off":
        caption_h = _draw_caption(
            draw,
            items,
            pad,
            0,
            text_w,
            label_font,
            value_font,
            muted,
            text,
            wrap_values=(caption == "Details"),
        )
        if caption == "Standard":
            # Reserve five rows even when a diagnostic currently exposes fewer
            # values. This keeps every Standard card pixel-identical in height.
            standard_rows = [("Value", "Ag")] * int(spec["caption_rows_summary"])
            caption_h = max(
                caption_h,
                _draw_caption(
                    draw,
                    standard_rows,
                    pad,
                    0,
                    text_w,
                    label_font,
                    value_font,
                    muted,
                    text,
                    wrap_values=False,
                ),
            )

    divider_h = 4
    total_h = pad + title_h + gap + ih
    if caption != "Off":
        total_h += gap + divider_h + gap + caption_h
    total_h += pad

    panel = Image.new("RGB", (card_w, total_h), bg)
    d = ImageDraw.Draw(panel)
    y = pad
    _draw_title(d, title or "CMK Preview", pad, y, text_w, title_font, text)
    y += title_h + gap

    x_img = pad + max(0, (text_w - iw) // 2)
    panel.paste(Image.fromarray(image), (x_img, y))
    y += ih

    if caption != "Off":
        y += gap
        d.rectangle((pad, y, card_w - pad, y + divider_h), fill=line)
        y += divider_h + gap
        _draw_caption(
            d,
            items,
            pad,
            y,
            text_w,
            label_font,
            value_font,
            muted,
            text,
            wrap_values=(caption == "Details"),
        )

    return np.asarray(panel, dtype=np.uint8)


def render_preview_panel(preview_payload: dict, *, caption: str = "Standard") -> np.ndarray:
    from .preview_payload import normalize_preview_payload, select_preview_image

    caption = "Standard" if str(caption) == "Summary" else str(caption)
    caption = caption if caption in _CAPTION_MODES else "Standard"
    data = normalize_preview_payload(preview_payload)

    is_concat_timeline = str(data.get("node") or "") == "CMK Diagnostic Concat"
    if is_concat_timeline:
        # Concat joins diagnostics strictly in input order. Standard keeps one
        # card per input; Details expands a multi-stage input in place so every
        # process image becomes an equally readable full-size card.
        groups = (data.get("metadata") or {}).get("diagnostic_groups") or []
        group_panels = []
        for group in groups:
            try:
                group_data = normalize_preview_payload(group)
                if caption == "Details":
                    stage_panels = _concat_detail_panels(group_data)
                    if stage_panels:
                        group_panels.extend(stage_panels)
                        continue
                group_panels.append(
                    _diagnostic_card(
                        select_preview_image(group_data, image_index=-1),
                        title=str(
                            group_data.get("title")
                            or group_data.get("node")
                            or "CMK Preview"
                        ),
                        summary=str(group_data.get("summary") or ""),
                        details=str(
                            group_data.get("details")
                            or group_data.get("summary")
                            or ""
                        ),
                        metadata=dict(group_data.get("metadata") or {}),
                        caption=caption,
                    )
                )
            except Exception:
                continue
        if group_panels:
            return combine_preview_panels(group_panels)

    # Details is the explicit opt-in for a node's internal diagnostic timeline.
    # The same quartet-card contract applies everywhere: one full-size card per
    # stage, never a miniature stage strip inside a process wrapper.
    if caption == "Details":
        stage_panels = _concat_detail_panels(data)
        if stage_panels:
            return combine_preview_panels(stage_panels)

    image = select_preview_image(data, image_index=-1)
    return _diagnostic_card(
        image,
        title=str(data.get("title") or data.get("node") or "CMK Preview"),
        summary=str(data.get("summary") or ""),
        details=str(data.get("details") or data.get("summary") or ""),
        metadata=dict(data.get("metadata") or {}),
        caption=caption,
    )


def render_preview_board_panels(preview_payload: dict, *, caption: str = "Standard") -> list[np.ndarray]:
    """Render one board input as one or more flat chronological cards.

    Preview Render and Preview Board share one contract: every process stage is
    a full-size quartet card in chronological input order.
    """
    from .preview_payload import normalize_preview_payload

    caption = "Standard" if str(caption) == "Summary" else str(caption)
    data = normalize_preview_payload(preview_payload)
    if caption == "Details":
        panels = _concat_detail_panels(data)
        if panels:
            return panels
    return [render_preview_panel(data, caption=caption)]


# -----------------------------------------------------------------------------
# Text-only diagnostic renderer
# -----------------------------------------------------------------------------

def _load_mono_font(size: int, *, bold: bool = False):
    size = max(10, int(size))
    candidates = []
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/Library/Fonts/Menlo.ttc",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/Library/Fonts/Courier New.ttf",
    ])
    for path in candidates:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return _load_font(size, bold=bold)


def _wrap_mono_line(draw: ImageDraw.ImageDraw, line: str, font, max_width: int) -> list[str]:
    line = str(line or "")
    if not line:
        return [""]
    if _text_w(draw, line, font) <= max_width:
        return [line]

    indent_len = len(line) - len(line.lstrip(" "))
    indent = " " * min(indent_len, 8)
    words = line.split(" ")
    out: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if _text_w(draw, candidate, font) <= max_width or not current:
            current = candidate
        else:
            out.append(current)
            current = f"{indent}{word}" if indent else word
    if current:
        out.append(current)
    return out or [line]


def render_summary_panel(*, title: str = "CMK Summary", text: str = "") -> np.ndarray:
    """Render a text-only CMK diagnostic summary card as RGB uint8 image."""
    spec = _design()
    bg = spec["card_bg"]
    fg = spec["text"]
    muted = spec["muted"]
    line = spec["line"]
    pad = int(spec["pad"])
    gap = int(spec["gap"])

    title_font = _load_font(44, bold=True)
    body_font = _load_mono_font(28, bold=False)
    small_font = _load_font(22, bold=False)

    raw_text = str(text or "").strip() or "No summary available."
    card_w = 960
    text_w = card_w - pad * 2
    probe = Image.new("RGB", (card_w, 1), bg)
    draw = ImageDraw.Draw(probe)

    title_lines = _wrap_text(draw, str(title or "CMK Summary"), title_font, text_w, max_lines=2)
    body_lines: list[str] = []
    for raw in raw_text.splitlines():
        body_lines.extend(_wrap_mono_line(draw, raw, body_font, text_w))

    title_h = sum(_text_bbox(draw, line_text, title_font)[1] + 8 for line_text in title_lines)
    line_h = _text_bbox(draw, "Ag", body_font)[1]
    row_gap = 10
    body_h = len(body_lines) * (line_h + row_gap)
    footer_h = _text_bbox(draw, "CMK_DIAGNOSTIC", small_font)[1]

    total_h = pad + title_h + gap + 4 + gap + body_h + gap + footer_h + pad
    panel = Image.new("RGB", (card_w, total_h), bg)
    d = ImageDraw.Draw(panel)

    y = pad
    for line_text in title_lines:
        d.text((pad, y), line_text, fill=fg, font=title_font)
        y += _text_bbox(d, line_text, title_font)[1] + 8

    y += gap
    d.rectangle((pad, y, card_w - pad, y + 4), fill=line)
    y += 4 + gap

    for line_text in body_lines:
        d.text((pad, y), line_text, fill=fg, font=body_font)
        y += line_h + row_gap

    y += gap
    d.text((pad, y), "CMK_DIAGNOSTIC · Summary", fill=muted, font=small_font)
    return np.asarray(panel, dtype=np.uint8)


def combine_preview_panels(images: List[np.ndarray]) -> np.ndarray:
    images = [_as_rgb(img) for img in images if img is not None]
    if not images:
        return np.zeros((240, 720, 3), dtype=np.uint8)

    # Fixed left-to-right diagnostic flow. Align the final section divider of
    # every card before equalising the overall height. Merely padding shorter
    # cards at the bottom makes the preview/information boundary jump vertically
    # when a compact card sits next to a detailed timeline.
    spec = _design()
    gap_w = int(spec["card_gap"])
    line_rgb = np.asarray(spec["line"], dtype=np.uint8)

    def _last_section_divider_y(img: np.ndarray) -> int | None:
        matches = np.all(img == line_rgb.reshape(1, 1, 3), axis=2)
        # A real section divider spans most of its panel. Short rules inside
        # stage cards must not be mistaken for the outer card boundary.
        threshold = max(48, int(round(img.shape[1] * 0.58)))
        rows = np.flatnonzero(matches.sum(axis=1) >= threshold)
        if rows.size == 0:
            return None
        return int(rows[-1])

    divider_rows = [_last_section_divider_y(img) for img in images]
    if all(row is not None for row in divider_rows):
        target_divider = max(int(row) for row in divider_rows if row is not None)
        bottom_heights = [
            img.shape[0] - int(row)
            for img, row in zip(images, divider_rows)
            if row is not None
        ]
        target_bottom = max(bottom_heights)
        height = target_divider + target_bottom
        aligned: list[np.ndarray] = []
        for img, row in zip(images, divider_rows):
            canvas = np.zeros((height, img.shape[1], 3), dtype=np.uint8)
            # The alignment area belongs to the card, not to the board.  Using
            # board_bg here made compact detail cards appear visibly shorter
            # even though their image arrays already had the same height.
            canvas[:, :, :] = spec["card_bg"]
            row = int(row)
            # Keep a common top edge. Any extra space required to reach the
            # shared divider belongs immediately above that divider; moving
            # the complete card would make title and preview heights jump.
            canvas[:row, :img.shape[1]] = img[:row]
            bottom = img[row:]
            canvas[
                target_divider:target_divider + bottom.shape[0],
                :img.shape[1],
            ] = bottom
            aligned.append(canvas)
        images = aligned
    else:
        height = max(img.shape[0] for img in images)
        images = [_pad_to_height(img, height) for img in images]

    parts: list[np.ndarray] = []
    for idx, img in enumerate(images):
        if idx:
            parts.append(np.zeros((height, gap_w, 3), dtype=np.uint8))
        parts.append(img)
    return np.concatenate(parts, axis=1)
