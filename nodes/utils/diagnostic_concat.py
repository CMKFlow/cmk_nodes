from __future__ import annotations

from typing import Any, Dict, List

from ...utils.preview_payload import normalize_diagnostic_payload


_MAX_DIAGNOSTICS = 32


class CMKDiagnosticConcat:
    """Combine several CMK diagnostics into one reusable timeline payload."""

    CATEGORY = "CMK/Toolbox/Diagnostics"
    RETURN_TYPES = ("CMK_DIAGNOSTIC",)
    RETURN_NAMES = ("diagnostic",)
    FUNCTION = "concat"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {"diagnostic_1": ("CMK_DIAGNOSTIC",)}
        for index in range(2, _MAX_DIAGNOSTICS + 1):
            optional[f"diagnostic_{index}"] = ("CMK_DIAGNOSTIC",)
        return {
            "required": {
                "title": (
                    "STRING",
                    {
                        "default": "CMK Flow · Diagnostic Timeline",
                        "multiline": False,
                    },
                ),
            },
            "optional": optional,
        }

    @staticmethod
    def _connected(diagnostic_1=None, **kwargs):
        diagnostics = []
        if diagnostic_1 is not None:
            diagnostics.append(diagnostic_1)
        for index in range(2, _MAX_DIAGNOSTICS + 1):
            value = kwargs.get(f"diagnostic_{index}")
            if value is not None:
                diagnostics.append(value)
        return diagnostics

    @staticmethod
    def _normalize_input(payload):
        """Accept direct diagnostics and ComfyUI's occasional single-item wrappers."""
        candidate = payload
        for _ in range(4):
            if isinstance(candidate, (tuple, list)) and len(candidate) == 1:
                candidate = candidate[0]
                continue
            if (
                isinstance(candidate, dict)
                and candidate.get("type") not in {"CMK_DIAGNOSTIC", "CMK_PREVIEW"}
                and isinstance(candidate.get("diagnostic"), dict)
            ):
                candidate = candidate["diagnostic"]
                continue
            break
        return normalize_diagnostic_payload(candidate)

    def concat(self, title: str, diagnostic_1=None, **kwargs):
        diagnostics = self._connected(diagnostic_1, **kwargs)
        stages: List[Dict[str, Any]] = []
        warnings: List[str] = []
        sources: List[Dict[str, Any]] = []
        diagnostic_groups: List[Dict[str, Any]] = []

        for diagnostic_index, payload in enumerate(diagnostics, start=1):
            try:
                data = self._normalize_input(payload)
            except Exception as exc:
                payload_type = type(payload).__name__
                warning = (
                    f"Diagnostic input {diagnostic_index} was skipped: "
                    f"expected CMK_DIAGNOSTIC, received {payload_type} ({exc})."
                )
                warnings.append(warning)
                sources.append(
                    {
                        "index": diagnostic_index,
                        "title": "Invalid diagnostic input",
                        "node": "",
                        "mode": "skipped",
                        "received_type": payload_type,
                    }
                )
                continue
            source_title = str(data.get("title") or f"Diagnostic {diagnostic_index}")
            source_node = str(data.get("node") or "")
            source_mode = str(data.get("mode") or "")
            source_summary = str(data.get("summary") or "")
            source_details = str(data.get("details") or source_summary)
            # Keep the complete normalized diagnostic as one ordered block.
            # Renderers can therefore preserve the process boundary instead of
            # flattening all internal cards into one ambiguous global grid.
            diagnostic_groups.append(data)

            sources.append(
                {
                    "index": diagnostic_index,
                    "title": source_title,
                    "node": source_node,
                    "mode": source_mode,
                    "summary": source_summary,
                    "details": source_details,
                    "warnings": list(data.get("warnings", []) or []),
                }
            )

            for warning in data.get("warnings", []) or []:
                warnings.append(f"{source_title}: {warning}")

            source_stages = data.get("stages", []) or []
            if source_stages:
                # Concat is intentionally non-semantic: every input contributes
                # all of its cards, unchanged and in connection order. Deciding
                # which process is active or visually important belongs to the
                # producer and renderer, never to this transport node.
                stages.extend(dict(stage) for stage in source_stages)
                continue

            # A nested concat with no valid stages only carries warnings. Do not
            # turn its normalizer fallback into a meaningless black timeline card.
            if source_node == "CMK Diagnostic Concat":
                continue

            images = data.get("images", []) or []
            if images:
                stages.append(
                    {
                        "title": source_title,
                        "subtitle": str(data.get("summary") or ""),
                        "image": images[-1],
                    }
                )

        source_names = [
            str(source.get("title") or f"Input {source.get('index', '?')}")
            for source in sources
        ]
        source_chain = " → ".join(source_names) or "None"
        summary = (
            f"Diagnostic Count: {len(diagnostics)}\n"
            f"Stage Count: {len(stages)}\n"
            f"Sources: {source_chain}"
        )
        details = (
            f"Diagnostics: {len(diagnostics)}\n"
            f"Stages: {len(stages)}\n"
            f"Order: {source_chain}"
        )

        return (
            {
                "type": "CMK_DIAGNOSTIC",
                "version": 2,
                "title": str(title or "CMK Flow · Diagnostic Timeline"),
                "node": "CMK Diagnostic Concat",
                "mode": "timeline",
                "summary": summary,
                "details": details,
                "metadata": {
                    "diagnostic_count": len(diagnostics),
                    "stage_count": len(stages),
                    "sources": source_chain,
                    "preview_layout": "grid",
                    "preview_columns": 4,
                    "diagnostic_groups": diagnostic_groups,
                    # Retain the structured provenance for programmatic consumers
                    # without exposing Python dictionary syntax in preview cards.
                    "source_records": sources,
                },
                "metrics": {
                    "diagnostics": len(diagnostics),
                    "stages": len(stages),
                },
                "warnings": warnings,
                "stages": stages,
                "preview": [stage["image"] for stage in stages],
                "images": [stage["image"] for stage in stages],
            },
        )
