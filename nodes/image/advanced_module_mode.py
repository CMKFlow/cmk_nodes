"""MODE of the 2.8 Advanced Detailer and FaceProcess subgraphs.

The FaceRebuild v2 MODE control (as for MaskDetailer, FaceSwap and Upscale &
Save): ENABLE activates the module, DISABLE is its bypass.  It feeds the
ENABLE of the module's Bypass Gate, Diagnostic Gate and Preview & Compare, so
the gate decides without requesting the Prepare.  The widget keeps the name of
the former Prepare switch, so the outer MODE and its stored value stay as they
are.
"""

from __future__ import annotations

from ...pipe.cmk_faceprocess_multimodel import ADVANCED_GLOBAL_ENABLE
from ..instantid_face_rebuild_v2 import CMKFaceRebuildModeEnable

DETAILER_ADVANCED_MODE_WIDGET = "detailer_global_enable"
FACEPROCESS_ADVANCED_MODE_WIDGET = ADVANCED_GLOBAL_ENABLE


class CMKDetailerAdvancedModeEnable(CMKFaceRebuildModeEnable):
    """MODE of CMK Flow · Detailer · Advanced: ENABLE / DISABLE."""

    @classmethod
    def INPUT_TYPES(cls):
        modes, options = CMKFaceRebuildModeEnable.INPUT_TYPES()["required"]["facerebuild_mode"]
        return {"required": {DETAILER_ADVANCED_MODE_WIDGET: (modes, dict(options))}}

    @staticmethod
    def read(detailer_global_enable="ENABLE"):
        return CMKFaceRebuildModeEnable.read(detailer_global_enable)


class CMKFaceProcessAdvancedModeEnable(CMKFaceRebuildModeEnable):
    """MODE of CMK Flow · FaceProcess · Advanced: ENABLE / DISABLE."""

    @classmethod
    def INPUT_TYPES(cls):
        modes, options = CMKFaceRebuildModeEnable.INPUT_TYPES()["required"]["facerebuild_mode"]
        return {"required": {FACEPROCESS_ADVANCED_MODE_WIDGET: (modes, dict(options))}}

    @staticmethod
    def read(faceprocess_global_enable="ENABLE"):
        return CMKFaceRebuildModeEnable.read(faceprocess_global_enable)
