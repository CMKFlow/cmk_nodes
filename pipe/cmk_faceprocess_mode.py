"""FaceProcess 2.8 operating mode control (F1).

CMK FaceProcess Mode Enable is the FaceProcess counterpart of CMK Detailer
Family Enable: one control per FaceProcess branch.  Its own value set
(``None / Restore / SDXL / Z-Image Turbo``) keeps the Detailer control and
its prompt contract untouched.  RESTORE is a mode of its own, not a model
family: it never materialises a diffusion model.

ENABLE: the Boolean branch/module switch (None is off, every other mode on).
PROCESS MODE: the unchanged selection for the consumer's mode input.  The
prompt handler ``cmk_prompt_family_inline`` puts the literal selection in
place of that link, so gates, boundaries and fingerprints see the same prompt
as with a dropdown on the consumer.
"""

from __future__ import annotations

FACEPROCESS_MODE_WIDGET = "process_mode"
FACEPROCESS_MODES = ["None", "Restore", "SDXL", "Z-Image Turbo"]
_MODE_IDS = {"none": "none", "restore": "restore", "sdxl": "sdxl", "z-image turbo": "z_image_turbo"}


def faceprocess_mode(value):
    """Map a PROCESS MODE value to its id; anything unknown is ``none`` (bypass)."""
    return _MODE_IDS.get(str(value or "").strip().lower(), "none")


def faceprocess_model_family(value):
    """Diffusion family a mode needs: ``sdxl``, ``z_image_turbo`` or ``None``."""
    mode = faceprocess_mode(value)
    return mode if mode in ("sdxl", "z_image_turbo") else None


class CMKFaceProcessModeEnable:
    """Operating mode of one FaceProcess branch (2.8.0)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {FACEPROCESS_MODE_WIDGET: (FACEPROCESS_MODES, {"default": "None"})}}

    RETURN_TYPES = ("BOOLEAN", FACEPROCESS_MODES)
    RETURN_NAMES = ("ENABLE", "PROCESS MODE")
    FUNCTION = "read"
    CATEGORY = "CMK/Flow/PostProcess"

    @staticmethod
    def read(process_mode="None"):
        return (faceprocess_mode(process_mode) != "none", process_mode)
