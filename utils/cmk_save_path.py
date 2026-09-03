"""Pure helpers for deriving CMK output folders from PROCESS metadata."""

from __future__ import annotations


_IMAGE_SOURCE_FAMILIES = {"image", "loaded_image", "standalone_image"}

# Stable output order.  The *_applied keys are the preferred contract; the
# legacy aliases keep already-saved workflows useful while modules migrate.
_PROCESS_STAGE_MARKERS = (
    ("Detailer", ("detailer_applied",)),
    (
        "InstantID",
        (
            "instantid_applied",
            "identity_applied",
            "instantid_enabled",
        ),
    ),
    ("FaceSwap", ("faceswap_applied", "face_swap_applied")),
    ("FaceProcess", ("faceprocess_applied", "face_process_applied")),
    (
        "FaceRebuild",
        (
            "face_rebuild_applied",
            "face_rebuild_enabled",
            "instantid_face_rebuild",
        ),
    ),
)


def save_base_folder(process: dict) -> str:
    """Return the source operation folder for a CMK process pipe."""
    family = str(
        process.get("source_model_family", process.get("model_family", "")) or ""
    ).strip().casefold()
    origin = str(process.get("pipe_origin", "") or "").strip().casefold()

    if family in _IMAGE_SOURCE_FAMILIES or "image load" in origin:
        return "ImageProcessing"
    if bool(process.get("boolean_inpaint_mode", False)):
        return "Inpaint"
    return "Text2Image"


def save_stage_folders(process: dict) -> list[str]:
    """Return only stages recorded as active/applied in stable flow order."""
    return [
        folder
        for folder, markers in _PROCESS_STAGE_MARKERS
        if any(bool(process.get(marker, False)) for marker in markers)
    ]


def save_automatic_folders(process: dict) -> list[str]:
    return [save_base_folder(process), *save_stage_folders(process)]
