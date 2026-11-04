from __future__ import annotations

from ...utils.comfy_preview_output import render_summary_panel, save_preview_png, ui_images
from ...utils.preview_payload import normalize_diagnostic_payload


class CMKSummary:
    """Render the summary part of one CMK diagnostic payload."""

    CATEGORY = "CMK/Toolbox/Diagnostics"
    RETURN_TYPES = ("STRING", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("STRING", "diagnostic")
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"diagnostic": ("CMK_DIAGNOSTIC",)}}

    @staticmethod
    def _format_block(title, values):
        if not values:
            return []
        lines = [str(title)]
        for key, value in values.items():
            label = str(key).replace("_", " ").title()
            lines.append(f"{label:<18}: {value}")
        return lines

    @classmethod
    def _metadata_fallback(cls, data):
        metadata = dict(data.get("metadata") or {})
        metrics = dict(data.get("metrics") or {})
        warnings = [str(w) for w in (data.get("warnings") or []) if str(w).strip()]

        lines = []
        lines.extend(cls._format_block("Metadata", metadata))
        if metrics:
            if lines:
                lines.append("")
            lines.extend(cls._format_block("Metrics", metrics))
        if warnings:
            if lines:
                lines.append("")
            lines.append("Warnings")
            lines.extend(f"- {w}" for w in warnings)
        return "\n".join(lines)

    def run(self, diagnostic):
        try:
            data = normalize_diagnostic_payload(diagnostic)
            text = str(data.get("summary") or data.get("details") or "").strip()
            if not text:
                text = self._metadata_fallback(data).strip()
            if not text:
                text = "No summary available."
            title = str(data.get("title") or data.get("node") or "CMK Summary")
        except Exception as exc:
            text = f"Invalid diagnostic input: {exc}"
            title = "CMK Summary"

        panel = render_summary_panel(title=title, text=text)
        image_info = save_preview_png(panel, prefix="CMK_summary")

        return {
            "ui": ui_images([image_info], text)["ui"],
            "result": (text, diagnostic),
        }
