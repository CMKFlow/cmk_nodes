from __future__ import annotations


_REGIONAL_PROMPT_KEYS = (
    "area",
    "strength",
    "set_area_to_bounds",
    "start_percent",
    "end_percent",
)

_EXTERNAL_CONTROLNET_KEYS = (
    "control",
    "control_apply_to_uncond",
)


def _hybrid_positive_conditioning(
    conditioning,
    *,
    instantid_mode,
    zero_pass_mode,
):
    if (
        zero_pass_mode
        or not str(instantid_mode or "").strip().upper().startswith("HYBRID")
    ):
        return conditioning

    global_conditioning = []
    for tensor, metadata in conditioning:
        global_metadata = metadata.copy()
        for key in (*_REGIONAL_PROMPT_KEYS, *_EXTERNAL_CONTROLNET_KEYS):
            global_metadata.pop(key, None)
        global_conditioning.append([tensor, global_metadata])
    return global_conditioning
