"""Pure helpers for deriving CMK output folders from PROCESS metadata."""

from __future__ import annotations


_IMAGE_SOURCE_FAMILIES = {"image", "loaded_image", "standalone_image"}
_GENERATED_FAMILIES = {"sdxl", "z_image_turbo"}

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


def save_base_folder(process: dict) -> str | None:
    """Return the source operation folder, or None when it is unknown.

    Module 01 always records the operation.  Missing flags mean a path lost it,
    which says nothing about the operation, so no category is claimed.
    """
    family = str(
        process.get("source_model_family", process.get("model_family", "")) or ""
    ).strip().casefold()
    origin = str(process.get("pipe_origin", "") or "").strip().casefold()

    if family in _IMAGE_SOURCE_FAMILIES or "image load" in origin:
        return "ImageProcessing"
    generation_mode = str(process.get("generation_mode", "") or "").strip().casefold()
    if (
        bool(process.get("boolean_inpaint_mode", False))
        or bool(process.get("hybrid_inpaint_mode", False))
        or generation_mode == "inpaint"
    ):
        return "Inpaint"
    if any(key in process for key in ("boolean_inpaint_mode", "hybrid_inpaint_mode", "generation_mode")):
        return "Text2Image"
    return None


def normalize_save_process(process: dict) -> dict:
    """Apply the active-family result semantics used by the Visualizer.

    Upscale & Save receives a family process directly and exposes the neutral
    result consumed by the Visualizer. A generated family can
    therefore still carry the original image source marker. Once sampling
    has produced an image, the active model family owns the save path.
    """
    if not isinstance(process, dict):
        return process
    normalized = dict(process)
    if normalized.get("type") == "CMK_MASK_DETAILER_PROCESS":
        return normalized
    family = str(normalized.get("model_family", "") or "").strip().casefold()
    sampled = any(
        normalized.get(key) is not None
        for key in ("image_1st_pass", "samples", "latent_1st_pass")
    )
    if (
        family in _GENERATED_FAMILIES
        and normalized.get("family_active", True)
        and sampled
    ):
        normalized["source_model_family"] = family
    return normalized


def save_stage_folders(process: dict) -> list[str]:
    """Return only stages recorded as active/applied in stable flow order."""
    return [
        folder
        for folder, markers in _PROCESS_STAGE_MARKERS
        if any(bool(process.get(marker, False)) for marker in markers)
    ]


def save_automatic_folders(process: dict) -> list[str]:
    process = normalize_save_process(process)
    base = save_base_folder(process)
    if base is None:
        print(
            "[CMK Save] Source operation unknown: PROCESS has no Inpaint flags, "
            "saving without a category folder",
            flush=True,
        )
        return save_stage_folders(process)
    return [base, *save_stage_folders(process)]
