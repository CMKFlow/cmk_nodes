"""FaceSwap v2 (2.8): MODE control and the 2.8 diagnostic contract.

The executors inherit the FaceSwap processing of CMK FaceSwap Image -Pipe-
unchanged (detection, swap, enhancer, BLEND, POST RESTORE, IMAGE, SEGS, LOG,
VISUAL); only their diagnostic differs.  FaceSwap is model-independent: the
v2 subgraphs only forward MODEL ZIT and MODEL SDXL.

* Standard (``CMKFaceSwapImagePipeV2``): a Prepare text card, then Swap Face
  (selected target face + paste region), Paste Mask (the run's paste support),
  Pasteback and Difference only when the image changed.
* Advanced: every slot executor (``CMKFaceSwapSlotPipeV2``) passes its run on
  as a Prepare card with the run's data; ``CMKFaceSwapAdvancedDiagnosticV2``
  after the unchanged SEGS CONCAT builds module card, slot cards and the
  stages on the merged image: Target Faces, Paste Masks, Merge (SEGS CONCAT
  input order), Overlap only when paste masks overlap, Difference.

No Source, Target or Final stage: source and target are visible at their
loaders, the result at the module output.  A disabled run is a text card.
"""

from __future__ import annotations

from ..image.smart_detailer import CMK_SmartDetailer
from ..instantid_face_rebuild_v2 import CMKFaceRebuildModeEnable
from ..utils.diagnostic_concat import CMKDiagnosticConcat
from .face_swap import CMKFaceSwapImagePipe
from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...utils.comfy_preview_output import SUMMARY_CARD_FOOTER, SUMMARY_CARD_LAYOUT
from ...utils.faceswap_diagnostic import (
    advanced_stages,
    bypass_text,
    paste_support,
    run_record,
    run_rows,
    standard_stages,
)
from ...utils.prepare_card import card_text
from ...utils.stable_segs import CMKStableSEGS

FACESWAP_MODE_WIDGET = "faceswap_mode"
PREPARE_NODE = "CMK FaceSwap v2 · Prepare"
STANDARD_NODE = "CMK FaceSwap v2"
ADVANCED_NODE = "CMK FaceSwap v2 · Advanced"
STANDARD_TITLE = "CMK Flow · FaceSwap"
ADVANCED_TITLE = "CMK Flow · FaceSwap · Advanced"
SLOT_KEY = "faceswap_slot"
SLOT_ATTRIBUTE = "cmk_faceswap_slot"


class CMKFaceSwapModeEnable(CMKFaceRebuildModeEnable):
    """MODE of a FaceSwap v2 subgraph (the FaceRebuild v2 MODE control):
    ENABLE runs FaceSwap, DISABLE is the module bypass."""

    @classmethod
    def INPUT_TYPES(cls):
        modes, options = CMKFaceRebuildModeEnable.INPUT_TYPES()["required"]["facerebuild_mode"]
        return {"required": {FACESWAP_MODE_WIDGET: (modes, dict(options))}}

    @staticmethod
    def read(faceswap_mode="ENABLE"):
        return CMKFaceRebuildModeEnable.read(faceswap_mode)


def _card(title, rows, metadata, image, *, mode="Prepare"):
    text = card_text(rows)
    return make_diagnostic_payload(
        title=title, node=PREPARE_NODE, previews=[image], summary=text, details=text, mode=mode,
        metadata={**metadata, SUMMARY_CARD_LAYOUT: SUMMARY_CARD_FOOTER},
    )


def _bypass(title, node, record, image):
    text = bypass_text(record)
    return make_diagnostic_payload(
        title=title, node=node, previews=[image], summary=text, details=text, mode="Bypass",
        metadata={"bypassed": True, "slot": record["index"]},
    )


def _difference(stages, target, result):
    detailer = CMK_SmartDetailer()
    changed = detailer._mean_abs_change(target, result)
    if changed > 0.00001:
        stages.append({"title": "Difference", "image": detailer._make_difference_preview(target, result),
                       "subtitle": f"target→result: {changed:.4f}"})
    return [dict(stage, title=f"{index:02d} {stage['title']}") for index, stage in enumerate(stages, start=1)], changed


def _timeline(title, items):
    """One diagnostic stays itself (a text card stays a text card); several
    become the module timeline."""
    items = [item for item in items if item is not None]
    if len(items) == 1:
        return items[0]
    return CMKDiagnosticConcat().concat(title, items[0], **{
        f"diagnostic_{index}": item for index, item in enumerate(items[1:], start=2)
    })[0]


def slot_records(payload):
    """FaceSwap slot records of a (nested) diagnostic timeline, in chain order."""
    records = []
    if not isinstance(payload, dict):
        return records
    metadata = payload.get("metadata") or {}
    if isinstance(metadata.get(SLOT_KEY), dict):
        records.append(metadata[SLOT_KEY])
    for group in metadata.get("diagnostic_groups") or ():
        records.extend(slot_records(group))
    return records


class _InputsV2:
    """Keeps the run's inputs for the diagnostic: a disabled run's diagnostic
    carries no selection, the card still names the slot's target selector."""

    def run_pipe(self, **inputs):
        self._v2_inputs = inputs
        try:
            return super().run_pipe(**inputs)
        finally:
            self._v2_inputs = None

    def _record(self, diagnostic, segs, index=1):
        record = run_record(diagnostic, segs, index=index)
        inputs = getattr(self, "_v2_inputs", None) or {}
        record["target_face"] = record["target_face"] or str(inputs.get("TARGET FACE", "Largest") or "Largest")
        record["source_face"] = record["source_face"] or str(inputs.get("SOURCE FACE", "Largest") or "Largest")
        return record


class CMKFaceSwapImagePipeV2(_InputsV2, CMKFaceSwapImagePipe):
    """CMK FaceSwap Image -Pipe- with the 2.8 diagnostic (FaceSwap v2 Standard)."""

    def _merge_transport(self, image, segs, log_block, diagnostic, *, before, enabled, visual=None,
                         unique_id=None, opt_log=None, opt_diagnostic=None):
        image, segs, log, result_visual, _legacy = CMKFaceSwapImagePipe._merge_transport(
            image, segs, log_block, diagnostic, before=before, enabled=enabled, visual=visual,
            unique_id=unique_id, opt_log=opt_log,
        )
        record = self._record(diagnostic, segs)
        if not record["enabled"]:
            items = [opt_diagnostic, _bypass("FaceSwap · Bypass", STANDARD_NODE, record, before)]
            return image, segs, log, result_visual, _timeline(STANDARD_TITLE, items)
        card = _card("FaceSwap · Prepare", run_rows(record), {"target_face": record["target_face"]}, before)
        support = paste_support(segs)
        stages = standard_stages(before, image, record, support) if support is not None else []
        stages, changed = _difference(stages, before, image)
        summary = f"TARGET  : {record['target_face']}\nCHANGED : {changed:.4f}"
        process = make_diagnostic_payload(
            title="FaceSwap", node=STANDARD_NODE, stages=stages, summary=summary, details=summary, mode="Swap",
            metadata={"target_face": record["target_face"], "change": changed,
                      "paste_pixels": int((support > 0).sum()) if support is not None else 0},
        )
        return image, segs, log, result_visual, _timeline(STANDARD_TITLE, [opt_diagnostic, card, process])


class CMKFaceSwapSlotPipeV2(_InputsV2, CMKFaceSwapImagePipe):
    """One slot of FaceSwap v2 Advanced: the unchanged FaceSwap run; its
    diagnostic is the slot's Prepare card with the run's data, chained in
    slot order.  The slot number is its chain position; its SEGS carry it for
    the merge diagnostic."""

    def _merge_transport(self, image, segs, log_block, diagnostic, *, before, enabled, visual=None,
                         unique_id=None, opt_log=None, opt_diagnostic=None):
        index = len(slot_records(opt_diagnostic)) + 1
        record = self._record(diagnostic, segs, index=index)
        if isinstance(segs, CMKStableSEGS):
            setattr(segs, SLOT_ATTRIBUTE, index)
        if record["enabled"]:
            card = _card(f"Slot {index} · Prepare", run_rows(record), {SLOT_KEY: record, "slot": index}, before)
        else:
            card = _bypass(f"Slot {index} · Off", PREPARE_NODE, record, before)
            card["metadata"] = {**card.get("metadata", {}), SLOT_KEY: record}
        return CMKFaceSwapImagePipe._merge_transport(
            image, segs, log_block, card, before=before, enabled=enabled, visual=visual,
            unique_id=unique_id, opt_log=opt_log, opt_diagnostic=opt_diagnostic,
        )


class CMKFaceSwapAdvancedDiagnosticV2:
    """Diagnostic of FaceSwap v2 Advanced after the unchanged SEGS CONCAT.

    Inputs: the target image, the merged image of SEGS CONCAT, the slot
    chain's diagnostic and the SEGS in the SEGS CONCAT input order.  Runs only
    when the boundary needs a fresh diagnostic (cache miss)."""

    CATEGORY = "CMK/Flow/Face"
    RETURN_TYPES = ("CMK_DIAGNOSTIC",)
    RETURN_NAMES = ("diagnostic",)
    FUNCTION = "diagnose"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "IMAGE MERGED": ("IMAGE",),
                "diagnostic": ("CMK_DIAGNOSTIC",),
            },
            "optional": {"segs": ("SEGS",), "segs_2": ("SEGS",), "segs_3": ("SEGS",)},
        }

    def diagnose(self, IMAGE, diagnostic, segs=None, segs_2=None, segs_3=None, **kwargs):
        merged = kwargs.get("IMAGE MERGED", IMAGE)
        records = {}
        for record in slot_records(diagnostic):
            records[int(record["index"])] = record
        supports, merge_order = {}, []
        for value in (segs, segs_2, segs_3):
            index = getattr(value, SLOT_ATTRIBUTE, None)
            support = paste_support(value)
            if index is None or support is None or index not in records:
                continue
            supports[index] = support
            merge_order.append(index)
        active = [dict(records[index], support=supports[index]) for index in sorted(records)
                  if records[index]["enabled"] and index in supports]
        chain = " → ".join(f"F{index}" for index in merge_order) or "no active slot"
        module_rows = [
            ("SLOTS", [f"F{index} · {records[index]['target_face']} · {'ON' if records[index]['enabled'] else 'OFF'}"
                       for index in sorted(records)] or ["none"]),
            ("MERGE", f"{chain} (SEGS CONCAT order, later on top)" if merge_order else chain),
        ]
        cards = [_card("FaceSwap · Prepare", module_rows, {"order": merge_order}, IMAGE)]
        cards += [_card(f"Slot {slot['index']} · Prepare", run_rows(slot), {"slot": slot["index"]}, IMAGE)
                  for slot in active]
        if not active:
            return (_timeline(ADVANCED_TITLE, cards),)
        stages, changed = _difference(advanced_stages(IMAGE, merged, active, merge_order), IMAGE, merged)
        summary = f"ORDER   : {chain}\nCHANGED : {changed:.4f}"
        process = make_diagnostic_payload(
            title="FaceSwap", node=ADVANCED_NODE, stages=stages, summary=summary, details=summary, mode="Merge",
            metadata={"order": merge_order, "change": changed, "slots": [slot["index"] for slot in active]},
        )
        return (_timeline(ADVANCED_TITLE, [*cards, process]),)
