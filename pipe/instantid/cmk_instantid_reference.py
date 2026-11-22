"""Native reference conditioning of the InstantID identity pass.

``conditioning_mode = "InstantID reference"`` averages the CMK-prepared
conditioning with a native checkpoint-CLIP encoding (``ConditioningAverage``,
``reference_conditioning_weight`` on the native side).  The average is done per
entry, and every entry is paired with the native encoding of the prompt it was
prepared from:

* entry 1, the global prompt  ->  native(global prompt)
* entry 1 + i, regional prompt i  ->  native(regional prompt i)

Averaging every entry with the global prompt alone would replace most of a
regional prompt with the scene description.  Sampler Prepare builds the
positive conditioning as ``[global, region 1 .. region n]`` from the active
regions of the pipe that carries it; the assignment is accepted only while the
entries still show that structure.  Without active regions the single global
reference is averaged into the whole conditioning exactly as before.

Spatial keys are not touched here: the zero-pass keeps them, the hybrid
handoff has removed them before (cmk_instantid_conditioning).
"""

from __future__ import annotations

from ..cmk_regional_weighting import active_regions, regional_prompt_translation


class ReferenceAssignmentError(ValueError):
    """The positive conditioning cannot be matched to its prompts."""


def _same(value, expected) -> bool:
    try:
        return abs(float(value) - float(expected)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _fail(reason):
    raise ReferenceAssignmentError(
        f"CMK InstantID reference: {reason}. The per-entry native reference cannot be assigned; "
        "check the conditioning chain between Sampler Prepare and InstantID or use conditioning_mode "
        "'CMK prepared'."
    )


def pipe_regions(sampled_in, process):
    """Regions of the pipe that carries the conditioning (SAMPLED), else of PROCESS."""
    if isinstance(sampled_in, dict) and "regional_conditioning" in sampled_in:
        return sampled_in.get("regional_conditioning")
    return process.get("regional_conditioning") if isinstance(process, dict) else None


def regional_reference_prompts(source_positive, regions, translate):
    """Prompt text of every regional entry in order, or ``None`` without active regions.

    ``source_positive`` is the positive conditioning before the hybrid
    handoff removes its spatial keys.  Raises ReferenceAssignmentError when the
    entries do not match ``[global, region 1 .. region n]``.
    """
    active = active_regions(regions)
    if not active:
        return None
    entries = list(source_positive or [])
    if len(entries) != 1 + len(active):
        _fail(
            f"the positive conditioning has {len(entries)} entries, the pipe lists 1 global + "
            f"{len(active)} active regional prompts"
        )
    if "area" in entries[0][1] or "mask" in entries[0][1]:
        _fail("entry 1 carries spatial keys instead of the global prompt")
    texts = []
    for index, (region, (_, metadata)) in enumerate(zip(active, entries[1:]), 1):
        if not (
            "area" in metadata
            and _same(metadata.get("strength"), region.get("strength", 1.0))
            and _same(metadata.get("start_percent"), region.get("start", 0.0))
            and _same(metadata.get("end_percent"), region.get("end", 1.0))
        ):
            _fail(f"entry {index + 1} does not carry the area, strength and range of regional prompt {index}")
        texts.append(regional_prompt_translation(region, translate).text)
    return texts


def apply_reference_conditioning(
    *,
    source_positive,
    positive,
    negative,
    process,
    sampled_in,
    model_pipe,
    reference_conditioning_weight,
    call_node,
    outputs,
    translate,
    timed,
):
    """Average the prepared conditioning with its native reference.

    Returns ``(positive, negative, conditioning_source, translation_pos, translation_neg)``;
    the translations feed the node LOG.  ``positive`` must be
    ``source_positive`` after the hybrid handoff, entry for entry.
    """
    regional_texts = regional_reference_prompts(source_positive, pipe_regions(sampled_in, process), translate)
    if regional_texts is not None and len(positive) != len(source_positive):
        _fail(f"the handoff changed the entry count from {len(source_positive)} to {len(positive)}")

    clean_clip = model_pipe.get("clip")
    if clean_clip is None:
        from ..loaders.checkpoint_vae_loader import ensure_sdxl_text_encoder

        clean_clip, _ = ensure_sdxl_text_encoder(model_pipe)
    prompt_pos = str(process.get("prompt_pos", sampled_in.get("prompt_pos", "")) or "")
    prompt_neg = str(process.get("prompt_neg", sampled_in.get("prompt_neg", "")) or "")
    translation_pos = translate(prompt_pos)
    translation_neg = translate(prompt_neg)

    def encode(text):
        return outputs(call_node(("CLIPTextEncode",), clip=clean_clip, text=text))[0]

    detail = "native CLIP Text Encode"
    if regional_texts is not None:
        detail += f" · global + {len(regional_texts)} regional"
    with timed("15 INSTANTID REFERENCE CONDITIONING", detail):
        reference_positive = encode(translation_pos.text)
        reference_negative = encode(translation_neg.text)
        regional_references = []
        if regional_texts is not None:
            encoded = {}
            for text in regional_texts:
                if text not in encoded:
                    encoded[text] = encode(text)
                regional_references.append(encoded[text])
    if bool(process.get("unload_models_after_use", True)):
        from ..loaders.checkpoint_vae_loader import evict_sdxl_text_encoder

        evict_sdxl_text_encoder(model_pipe, sampled_in)
        clean_clip = None
        import gc

        gc.collect()
        try:
            import comfy.model_management

            comfy.model_management.cleanup_models_gc()
            comfy.model_management.soft_empty_cache(force=True)
        except Exception:
            pass
    # Same text, same tokenizer: a different token length means the native
    # reference was not encoded from the text of the prepared entry.
    for index, (reference, (tensor, _)) in enumerate(zip(regional_references, source_positive[1:]), 1):
        if tuple(reference[0][0].shape[1:2]) != tuple(tensor.shape[1:2]):
            _fail(
                f"the native encoding of regional prompt {index} has {reference[0][0].shape[1]} tokens, "
                f"its prepared entry {tensor.shape[1]}"
            )

    reference_weight = max(0.0, min(1.0, float(reference_conditioning_weight)))
    with timed(
        "15 INSTANTID HYBRID CONDITIONING",
        f"reference={reference_weight:.2f} | CMK prepared={1.0 - reference_weight:.2f}",
    ):
        if regional_texts is None:
            positive = outputs(call_node(
                ("ConditioningAverage",),
                conditioning_to=positive,
                conditioning_from=reference_positive,
                conditioning_to_strength=1.0 - reference_weight,
            ))[0]
        else:
            mixed = []
            for entry, reference in zip(positive, [reference_positive, *regional_references]):
                mixed.extend(outputs(call_node(
                    ("ConditioningAverage",),
                    conditioning_to=[entry],
                    conditioning_from=reference,
                    conditioning_to_strength=1.0 - reference_weight,
                ))[0])
            positive = mixed
        negative = outputs(call_node(
            ("ConditioningAverage",),
            conditioning_to=negative,
            conditioning_from=reference_negative,
            conditioning_to_strength=1.0 - reference_weight,
        ))[0]
    conditioning_source = (
        f"hybrid: {reference_weight:.0%} native reference | "
        f"{1.0 - reference_weight:.0%} CMK prepared"
    )
    if regional_texts is not None:
        conditioning_source += f" · per entry: global + {len(regional_texts)} regional"
    return positive, negative, conditioning_source, translation_pos, translation_neg
