"""Prepare text cards of the 2.8 diagnostic timelines.

A Prepare card is a text card next to image cards.  Its text is laid out for
the card renderer (``render_summary_panel``): at most ``CARD_COLUMNS``
characters per line, a short label column and aligned continuation lines, so
nothing wraps by accident.  ``SUMMARY_CARD_FOOTER`` makes the renderer draw the
section rule above the footer; ``combine_preview_panels`` aligns the cards at
that rule, so the text shares the image zone instead of enlarging the
information area of every image card (CMK 2.8 diagnostic rule).
"""

from __future__ import annotations

import re
import textwrap

from .comfy_preview_output import SUMMARY_CARD_FOOTER, SUMMARY_CARD_LAYOUT

CARD_COLUMNS = 52   # mono 28 px on the 960 px text card
CARD_ROWS = 18      # body lines that fit the 838 px image zone of an image card


def _wrap_path(text, room):
    """A path split after its ``/`` separators (a name breaks only when it is longer than a line)."""
    lines, current = [], ""
    for segment in re.findall(r"[^/]*/|[^/]+$", text):
        if current and len(current) + len(segment) > room:
            lines.append(current)
            current = ""
        current += segment
        while len(current) > room:
            lines.append(current[:room])
            current = current[room:]
    return lines + [current] if current else lines or [""]


def _wrap(value, room):
    """Lines of at most ``room`` characters, split at `` · `` first, then at words."""
    text = str(value)
    if len(text) <= room:
        return [text]
    lines, current = [], ""
    for part in text.split(" · "):
        candidate = f"{current} · {part}" if current else part
        if len(candidate) <= room:
            current = candidate
            continue
        if current:
            lines.append(current)
        pieces = _wrap_path(part, room) if "/" in part and " " not in part else (
            textwrap.wrap(part, room, break_on_hyphens=False) or [""])
        lines.extend(pieces[:-1])
        current = pieces[-1]
    lines.append(current)
    return lines


def card_text(rows) -> str:
    """``rows``: (label, value) with value a string or a list of lines."""
    width = max((len(label) for label, _value in rows), default=0)
    room = max(12, CARD_COLUMNS - width - 3)
    lines = []
    for label, value in rows:
        values = value if isinstance(value, (list, tuple)) else [value]
        wrapped = [part for item in values for part in _wrap(item, room)]
        lines.append(f"{label:<{width}} : {wrapped[0]}".rstrip())
        lines.extend(f"{'':<{width}}   {more}" for more in wrapped[1:])
    return "\n".join(lines)


_ROW = re.compile(r"^(\S(?:.*?\S)?)\s+:(?:\s(.*))?$")     # LOG row: padded label, " : value"
_SECTION = re.compile(r"^([A-Z0-9][A-Z0-9 /_.-]*):$")     # LOG section header: "POSITIVE PROMPT:"


def log_block_rows(log, title):
    """Rows of the last LOG block ``title``: ``LABEL   : value`` lines, and
    ``SECTION:`` headers with the content lines up to the next row or header
    (the LOG drops blank lines)."""
    blocks = [block for block in (log or {}).get("blocks", []) if isinstance(block, dict) and block.get("title") == title]
    rows, section = [], None
    for raw in (blocks[-1].get("lines", []) if blocks else []):
        line = str(raw).rstrip()
        if not line.strip():
            continue
        row, header = _ROW.match(line), _SECTION.match(line.strip())
        if row and not header:
            section = None
            rows.append((row.group(1), (row.group(2) or "").strip() or "—"))
        elif header:
            section = (header.group(1), [])
            rows.append(section)
        elif section is not None:
            section[1].append(line.strip())
        else:
            rows.append(("", line.strip()))
    return [(label, value if value else "—") for label, value in rows]


def merge_rows(rows, pairs):
    """Join related rows: ``pairs`` of (first, second, label, separator)."""
    values = dict(rows)
    merged = []
    for label, value in rows:
        pair = next((p for p in pairs if p[0] == label and p[1] in values), None)
        if pair:
            merged.append((pair[2], f"{value}{pair[3]}{values[pair[1]]}"))
        elif not any(p[1] == label and p[0] in values for p in pairs):
            merged.append((label, value))
    return merged


def footer_card(payload, rows, *, title=None):
    """The Prepare diagnostic as a footer-aligned card with ``rows`` as text
    (Details); the short summary of the producer stays for Standard."""
    if not isinstance(payload, dict) or not rows:
        return payload
    text = card_text(rows)
    card = dict(payload, details=text)
    if title:
        card["title"] = title
    card["metadata"] = {**(payload.get("metadata") or {}), SUMMARY_CARD_LAYOUT: SUMMARY_CARD_FOOTER}
    return card
