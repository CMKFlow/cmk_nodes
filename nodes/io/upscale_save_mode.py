"""MODE of CMK Flow · Upscale & Save v2 (2.8): the FaceRebuild v2 MODE control.

ENABLE saves (the former SAVE ENABLED = true), DISABLE is the bypass without
upscaler, boundary or file.  The Boolean output drives every former SAVE
ENABLED target inside the subgraph.
"""

from __future__ import annotations

from ..instantid_face_rebuild_v2 import CMKFaceRebuildModeEnable

UPSCALE_SAVE_MODE_WIDGET = "upscale_save_mode"


class CMKUpscaleSaveModeEnable(CMKFaceRebuildModeEnable):
    """MODE of Upscale & Save v2: ENABLE saves, DISABLE is the bypass."""

    @classmethod
    def INPUT_TYPES(cls):
        modes, options = CMKFaceRebuildModeEnable.INPUT_TYPES()["required"]["facerebuild_mode"]
        return {"required": {UPSCALE_SAVE_MODE_WIDGET: (modes, dict(options))}}

    @staticmethod
    def read(upscale_save_mode="ENABLE"):
        return CMKFaceRebuildModeEnable.read(upscale_save_mode)
