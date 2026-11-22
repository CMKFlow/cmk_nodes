"""FaceRebuild v2 (2.8): MODE control and the 2.8 diagnostic contract.

The executors inherit the SDXL FaceRebuild processing of CMK FaceRebuild SDXL
(Standard / Advanced) unchanged; only their diagnostic differs:

* a Prepare text card (the FaceRebuild Prepare information of every slot),
* spatial stages on the target image from the rebuild geometry the run
  used: Rebuild ROI (target face + head/neck ROI), Inpaint Mask (sampling
  mask), Pasteback (pasted result x pasteback weight with PASTEBACK NECK);
  Advanced: Target Faces, Inpaint Masks, Pasteback of all slots in pasteback
  order and Overlap when slots overlap,
* Difference only when the image changed.

No Source, Detection or Rebuilt-ROI stage: the target is visible at the
loader, the detector result is the selected face drawn in Rebuild ROI, and the
rebuilt crop is a plain intermediate result.  A bypass is a text card.
"""

from __future__ import annotations

from .image.smart_detailer import CMK_SmartDetailer
from .instantid_face_rebuild_advanced import CMKInstantIDFaceRebuildAdvancedSDXL, CMKInstantIDFaceRebuildSDXL
from .utils.diagnostic_concat import CMKDiagnosticConcat
from ..utils.cmk_diagnostic import make_diagnostic_payload
from ..utils.comfy_preview_output import SUMMARY_CARD_FOOTER, SUMMARY_CARD_LAYOUT
from ..utils.prepare_card import card_text
from ..utils.facerebuild_diagnostic import advanced_stages, standard_stages

FACEREBUILD_MODE_WIDGET = "facerebuild_mode"
FACEREBUILD_MODES = ["ENABLE", "DISABLE"]
PREPARE_NODE = "CMK FaceRebuild v2 · Prepare"
STANDARD_NODE = "CMK FaceRebuild v2"
ADVANCED_NODE = "CMK FaceRebuild v2 · Advanced"


class CMKFaceRebuildModeEnable:
    """MODE of a FaceRebuild v2 subgraph: ENABLE runs FaceRebuild, DISABLE is the module bypass."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {FACEREBUILD_MODE_WIDGET: (FACEREBUILD_MODES, {"default": "ENABLE"})}}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("ENABLE",)
    FUNCTION = "read"
    CATEGORY = "CMK/Flow/Face"

    @staticmethod
    def read(facerebuild_mode="ENABLE"):
        return (str(facerebuild_mode) != "DISABLE",)


def _fields(payload):
    """``Label : value`` lines of a FaceRebuild Prepare / Detailer / Pasteback diagnostic."""
    fields = {}
    for line in str(payload.get("summary") or "").splitlines():
        label, sep, value = line.partition(":")
        if sep:
            fields[label.strip()] = value.strip()
    return fields


def _slot_rows(slot, prepare, detail, paste):
    context = slot["context"]
    return [
        ("TARGET", slot["target_face"]),
        ("FACE BOX", ", ".join(str(int(v)) for v in context["face_bbox"])),
        ("ROI", ", ".join(str(int(v)) for v in context["roi"])),
        ("WORKING", f"{context['working_size'][0]} × {context['working_size'][1]} · feather {int(context['feather'])} px"),
        ("NECK", f"{paste.get('Pasteback Neck', 'ON' if slot['pasteback_neck'] else 'OFF')} (pasteback)"),
        ("SCHEDULE", detail.get("Sampling schedule", "")),
        ("IDENTITY", f"{detail.get('Identity strength', '')} · pose {detail.get('Pose strength', '')}"
                     f" · noise {detail.get('Identity noise', '')}"),
        ("CONTENTGUARD", prepare.get("ContentGuard", "")),
        ("ROI BORDER", prepare.get("ROI Border Guard", "")),
    ]


def _module_rows(prepare, detail):
    return [
        ("MODE", [detail.get("Workflow", "").replace("INSTANTID FACE REBUILD · ", ""),
                  f"inpaint {detail.get('Inpaint', '')} · first pass {detail.get('First Pass', '')}"]),
        ("METHOD", f"{detail.get('Method', '')} · cfg {detail.get('CFG', '')}"),
        ("KPS", detail.get("KPS source", "")),
        ("LATENT", detail.get("Latent source", "")),
        ("CONDITIONING", [part.strip() for part in prepare.get("Conditioning", "").split("/")]),
        ("MASKS", prepare.get("Mask Separation", "")),
        ("SHAPE", prepare.get("Shape Transfer", "")),
    ]


def _prepare_card(title, rows, metadata, source):
    text = card_text(rows)
    return make_diagnostic_payload(
        title=title, node=PREPARE_NODE, previews=[source], summary=text, details=text, mode="Prepare",
        metadata={**metadata, SUMMARY_CARD_LAYOUT: SUMMARY_CARD_FOOTER},
    )


class _Diagnostic2:
    """2.8 diagnostic of the FaceRebuild v2 executors."""

    DIAGNOSTIC_NODE = STANDARD_NODE
    SLOT_CARDS = False

    def _bypass_diagnostic(self, image, summary):
        return make_diagnostic_payload(
            title=f"{self.DIAGNOSTIC_NAME} · Bypass", node=self.DIAGNOSTIC_NODE, previews=[image],
            summary=summary, details=summary, mode="Bypass", metadata={"bypassed": True},
        )

    def _spatial_stages(self, source, result, slots):
        raise NotImplementedError

    def _module_diagnostic(self, source, result, slots):
        several = self.SLOT_CARDS

        def section(slot, item):
            text = str(item.get("summary") or "")
            return f"SLOT {slot['index']} · {slot['target_face'].upper()}\n{text}" if several else text

        parsed = [(slot, *(_fields(item) for item in slot["diagnostics"])) for slot in slots]
        module = _module_rows(parsed[0][1], parsed[0][2])
        if several:
            # Advanced: one module card, then one card per active slot in
            # processing order (largest face first).
            order = " → ".join(f"F{slot['index']}" for slot in slots)
            cards = [_prepare_card("FaceRebuild · Prepare", [("SLOTS", f"{order} (processing order)")] + module,
                                   {"order": [slot["index"] for slot in slots]}, source)]
            cards += [_prepare_card(f"Slot {slot['index']} · Prepare", _slot_rows(slot, prepare, detail, paste),
                                    {"slot": slot["index"]}, source)
                      for slot, prepare, detail, paste in parsed]
        else:
            slot, prepare, detail, paste = parsed[0]
            cards = [_prepare_card("FaceRebuild · Prepare", _slot_rows(slot, prepare, detail, paste) + module,
                                   {"slot": slot["index"]}, source)]
        stages = self._spatial_stages(source, result, slots)
        detailer = CMK_SmartDetailer()
        changed = detailer._mean_abs_change(source, result)
        difference = detailer._make_difference_preview(source, result) if changed > 0.00001 else None
        if difference is not None:
            stages.append({"title": "Difference", "image": difference, "subtitle": f"source→result: {changed:.4f}"})
        stages = [dict(stage, title=f"{index:02d} {stage['title']}") for index, stage in enumerate(stages, start=1)]
        process_text = "\n\n".join(
            section(slot, slot["diagnostics"][1]) + "\n" + str(slot["diagnostics"][2].get("summary") or "")
            for slot in slots
        )
        process = make_diagnostic_payload(
            title="FaceRebuild", node=self.DIAGNOSTIC_NODE, stages=stages,
            summary=process_text, details=process_text, mode="Rebuild",
            metadata={"order": [slot["index"] for slot in slots], "change": changed,
                      "pasteback_neck": {str(slot["index"]): slot["pasteback_neck"] for slot in slots}},
        )
        items = [*cards, process]
        return CMKDiagnosticConcat().concat(self.DIAGNOSTIC_NAME, items[0], **{
            f"diagnostic_{index}": item for index, item in enumerate(items[1:], start=2)
        })[0]


class CMKInstantIDFaceRebuildSDXLV2(_Diagnostic2, CMKInstantIDFaceRebuildSDXL):
    """CMK FaceRebuild SDXL with the 2.8 diagnostic (FaceRebuild v2)."""

    DIAGNOSTIC_NODE = STANDARD_NODE

    def _spatial_stages(self, source, result, slots):
        return standard_stages(source, result, slots[0])


class CMKInstantIDFaceRebuildAdvancedSDXLV2(_Diagnostic2, CMKInstantIDFaceRebuildAdvancedSDXL):
    """CMK FaceRebuild SDXL · Advanced with the 2.8 diagnostic (FaceRebuild v2)."""

    DIAGNOSTIC_NODE = ADVANCED_NODE
    SLOT_CARDS = True

    def _spatial_stages(self, source, result, slots):
        return advanced_stages(source, result, slots)
