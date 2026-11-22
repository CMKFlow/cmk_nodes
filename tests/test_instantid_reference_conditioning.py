"""InstantID reference conditioning: per-entry native reference.

Model-free.  The positive conditioning is built by the real Sampler Prepare
regional path; the hybrid handoff uses the real CMK strip; averaging uses
ComfyUI's real ConditioningAverage.  Text encoders are stubbed with
fingerprints: every text gets a distinct value, with separate values for the
CMK-prepared encode and the native reference encode, and the real SDXL token
length.  A mixed entry therefore shows exactly which prompts it was built from.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import uuid

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_instantid_reference_conditioning"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

prepare = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_sampler_prepare")
persistent = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_persistent_cache")
strip = importlib.import_module(f"{TEST_PACKAGE}.pipe.instantid.cmk_instantid_conditioning")
reference = importlib.import_module(f"{TEST_PACKAGE}.pipe.instantid.cmk_instantid_reference")

import comfy.samplers as samplers  # noqa: E402
import nodes  # noqa: E402
from comfy import sdxl_clip  # noqa: E402

W = H = 1024
WIDTH = 8
TOKENIZER = sdxl_clip.SDXLTokenizer()

STATION = ("Cinematic photorealistic photograph on the platform of a historic European railway station, elegant "
           "nineteenth-century iron and glass architecture, high vaulted roof, detailed stone platform, warm afternoon "
           "sunlight streaming through the windows, subtle atmospheric depth, authentic European cinematic realism, "
           "realistic skin texture, natural human proportions, eye-level camera, 35mm photography, realistic depth of "
           "field, carefully balanced composition, no background crowds, no text, no logos.")
WOMAN = ("An adult woman approximately 40 years old, wearing a tailored burgundy wool coat over a charcoal blouse, "
         "long wavy chestnut hair, composed thoughtful expression, facing the camera with her face fully visible.")
NEGATIVE = "blurry, deformed"
PERSONS = ["a tall man in a grey suit", "a young woman in a yellow raincoat", "an old woman with a red umbrella"]


def token_length(text):
    tokens = TOKENIZER.tokenize_with_weights(text)
    return 77 * max(len(tokens["g"]), len(tokens["l"]))


def fingerprint(text, kind):
    digest = hashlib.sha256(f"{kind}\0{text}".encode()).digest()
    return 1.0 + int.from_bytes(digest[:4], "big") / 2 ** 32


def encoded(text, kind, pooled=True):
    value = fingerprint(text, kind)
    metadata = {"pooled_output": torch.full((1, 4), value)} if pooled else {}
    return [[torch.full((1, token_length(text), WIDTH), value), metadata]]


class Translation:
    def __init__(self, text, status):
        self.text, self.status = text, status


class FakeTranslator:
    """translate_prompt with a cache: English stays, everything else gets a marker."""

    def __init__(self, prefix=""):
        self.prefix, self.calls = prefix, []

    def __call__(self, text):
        self.calls.append(text)
        if not str(text).strip():
            return Translation(str(text), "empty")
        return Translation(self.prefix + str(text), "cache_hit")


class Prepare(prepare.CMKSamplerPrepareSDXLPipe):
    def _encode_sdxl_plus(self, clip, width, height, size_cond_factor, text):
        return encoded(text, "prepared")


def region(prompt, x=0.067, y=0.135, width=0.55, height=0.865, blend=0, strength=1.75, start=0.0, end=1.0):
    return {"prompt": prompt, "x": x, "y": y, "width": width, "height": height, "strength": strength,
            "start": start, "end": end, "conditioning_blend": blend}


def prepared_positive(global_text, regions, translate):
    with patch.object(prepare, "translate_prompt", translate):
        base = encoded(translate(global_text).text, "prepared")
        combined, _, _ = Prepare()._apply_regional_conditioning(base, None, W, H, 4, regions)
    return combined


class Calls:
    def __init__(self):
        self.encoded = []
        self.averages = 0

    def __call__(self, names, **kwargs):
        name = names[0]
        if name == "CLIPTextEncode":
            self.encoded.append(kwargs["text"])
            return (encoded(kwargs["text"], "native"),)
        if name == "ConditioningAverage":
            self.averages += 1
            return nodes.ConditioningAverage().addWeighted(**kwargs)
        raise AssertionError(name)


def run(regions, *, zero_pass, global_text=STATION, translate=None, positive=None, weight=0.75, prepare_translate=None):
    translate = translate or FakeTranslator()
    source = positive if positive is not None else prepared_positive(global_text, regions, prepare_translate or translate)
    negative = encoded(NEGATIVE, "prepared")
    stripped_pos = strip._hybrid_positive_conditioning(source, instantid_mode="HYBRID", zero_pass_mode=zero_pass)
    stripped_neg = strip._hybrid_positive_conditioning(negative, instantid_mode="HYBRID", zero_pass_mode=zero_pass)
    calls = Calls()
    result = reference.apply_reference_conditioning(
        source_positive=source,
        positive=stripped_pos,
        negative=stripped_neg,
        process={"prompt_pos": global_text, "prompt_neg": NEGATIVE, "unload_models_after_use": False,
                 "regional_conditioning": regions},
        sampled_in={"regional_conditioning": regions},
        model_pipe={"clip": object()},
        reference_conditioning_weight=weight,
        call_node=calls,
        outputs=lambda value: value,
        translate=translate,
        timed=lambda *args: contextlib.nullcontext(),
    )
    return source, stripped_pos, result, calls


def legacy(stripped_pos, stripped_neg_source, global_text, weight=0.75):
    """The former single global reference averaged into the whole conditioning."""
    positive = nodes.ConditioningAverage().addWeighted(stripped_pos, encoded(global_text, "native"), 1.0 - weight)[0]
    negative = nodes.ConditioningAverage().addWeighted(stripped_neg_source, encoded(NEGATIVE, "native"), 1.0 - weight)[0]
    return positive, negative


def expected_entry(prepared_entry, text, weight=0.75):
    """0.25 x prepared entry + 0.75 x native(text), ComfyUI semantics."""
    return nodes.ConditioningAverage().addWeighted([prepared_entry], encoded(text, "native"), 1.0 - weight)[0][0]


def assert_same_entry(test, actual, expected):
    test.assertTrue(torch.equal(actual[0], expected[0]))
    test.assertEqual(set(actual[1]), set(expected[1]))
    for key, value in expected[1].items():
        if isinstance(value, torch.Tensor):
            test.assertTrue(torch.equal(actual[1][key], value), key)
        else:
            test.assertEqual(actual[1][key], value, key)


class WithoutRegions(unittest.TestCase):
    def test_global_reference_path_is_bit_identical_to_before(self):
        for zero_pass in (True, False):
            for regions in ([], [region("   ")], None):
                with self.subTest(zero_pass=zero_pass, regions=regions):
                    source, stripped, (positive, negative, label, *_), calls = run(regions, zero_pass=zero_pass)
                    old_pos, old_neg = legacy(stripped, encoded(NEGATIVE, "prepared"), STATION)
                    self.assertEqual(len(positive), 1)
                    assert_same_entry(self, positive[0], old_pos[0])
                    assert_same_entry(self, negative[0], old_neg[0])
                    self.assertEqual(calls.encoded, [STATION, NEGATIVE])
                    self.assertEqual(calls.averages, 2)
                    self.assertEqual(label, "hybrid: 75% native reference | 25% CMK prepared")


class OneRegion(unittest.TestCase):
    def check(self, blend, zero_pass):
        regions = [region(WOMAN, blend=blend)]
        source, stripped, (positive, negative, label, *_), calls = run(regions, zero_pass=zero_pass)
        self.assertEqual(len(positive), 2)
        assert_same_entry(self, positive[0], expected_entry(stripped[0], STATION))
        assert_same_entry(self, positive[1], expected_entry(stripped[1], WOMAN))
        # the global prompt is no longer part of the regional entry
        diluted = expected_entry(stripped[1], STATION)
        self.assertFalse(torch.equal(positive[1][0], diluted[0]))
        self.assertFalse(torch.equal(positive[1][1]["pooled_output"], diluted[1]["pooled_output"]))
        self.assertEqual(sorted(calls.encoded), sorted([STATION, NEGATIVE, WOMAN]))
        self.assertEqual(label, "hybrid: 75% native reference | 25% CMK prepared · per entry: global + 1 regional")
        return source, positive

    def test_zero_pass_keeps_every_spatial_key_of_the_blend_path(self):
        for blend in (0, 6):
            with self.subTest(blend=blend):
                source, positive = self.check(blend, True)
                kept = {key: value for key, value in source[1][1].items() if key != "pooled_output"}
                self.assertEqual(set(kept) | {"pooled_output"}, set(positive[1][1]))
                for key, value in kept.items():
                    same = torch.equal(value, positive[1][1][key]) if isinstance(value, torch.Tensor) else value == positive[1][1][key]
                    self.assertTrue(same, key)
                self.assertEqual(positive[1][1]["strength"], 1.75)
                self.assertEqual("mask" in positive[1][1], blend == 6)

    def test_hybrid_keeps_the_unchanged_spatial_strip(self):
        for blend in (0, 3, 6):
            with self.subTest(blend=blend):
                source, positive = self.check(blend, False)
                removed = set(strip._REGIONAL_PROMPT_KEYS) | set(strip._EXTERNAL_CONTROLNET_KEYS)
                self.assertTrue(removed.isdisjoint(positive[1][1]))
                self.assertEqual(set(positive[1][1]), set(source[1][1]) - removed)

    def test_reference_fix_does_not_depend_on_the_blend_step(self):
        tensors = []
        for blend in range(7):
            _, _, (positive, *_), _ = run([region(WOMAN, blend=blend)], zero_pass=True)
            tensors.append(positive[1][0])
        self.assertTrue(all(torch.equal(tensors[0], other) for other in tensors[1:]))

    def test_full_frame_area_and_mask_stay_structurally_equivalent(self):
        results = {}
        for blend in (0, 6):
            _, _, (positive, *_), _ = run([region(WOMAN, 0.0, 0.0, 1.0, 1.0, blend=blend)], zero_pass=True)
            conds = [dict(metadata, model_conds={}, uuid=uuid.uuid4()) for _, metadata in positive]
            samplers.resolve_areas_and_cond_masks_multidim(conds, (H // 8, W // 8), "cpu")
            mults = [samplers.get_area_and_mult(c, torch.zeros(1, 4, H // 8, W // 8), torch.tensor([500.0])) for c in conds]
            results[blend] = (positive, [(m.area, m.mult[0, 0]) for m in mults])
        (area_pos, area_mults), (mask_pos, mask_mults) = results[0], results[6]
        for a, b in zip(area_pos, mask_pos):
            self.assertTrue(torch.equal(a[0], b[0]))
            self.assertTrue(torch.equal(a[1]["pooled_output"], b[1]["pooled_output"]))
        for (area_a, mult_a), (area_b, mult_b) in zip(area_mults, mask_mults):
            self.assertEqual(area_a, area_b)
            self.assertTrue(torch.equal(mult_a.expand(H // 8, W // 8), mult_b.expand(H // 8, W // 8)))


class SeveralRegions(unittest.TestCase):
    def test_three_regions_get_their_own_reference_in_order(self):
        regions = [region(PERSONS[0], 0.0, 0.1, 0.3, 0.9, blend=6, strength=1.4),
                   region(""),                                   # empty: no entry
                   "not a region",                               # ignored like in Sampler Prepare
                   region(PERSONS[1], 0.35, 0.1, 0.3, 0.9, blend=3, strength=2.0),
                   region(PERSONS[2], 0.7, 0.1, 0.3, 0.9, blend=0, strength=1.4, start=0.0, end=0.8)]
        for zero_pass in (True, False):
            with self.subTest(zero_pass=zero_pass):
                _, stripped, (positive, *_), calls = run(regions, zero_pass=zero_pass)
                self.assertEqual(len(positive), 4)
                for entry, prepared_entry, text in zip(positive, stripped, [STATION, *PERSONS]):
                    assert_same_entry(self, entry, expected_entry(prepared_entry, text))
                # no entry is mixed with another prompt's reference
                for index, entry in enumerate(positive[1:]):
                    for other in [STATION, *PERSONS]:
                        if other != PERSONS[index]:
                            self.assertFalse(torch.equal(entry[0], expected_entry(stripped[index + 1], other)[0]))
                self.assertEqual(calls.encoded, [STATION, NEGATIVE, *PERSONS])

    def test_identical_regional_prompts_are_encoded_once(self):
        regions = [region(PERSONS[0], 0.0, 0.1, 0.3, 0.9), region(PERSONS[0], 0.6, 0.1, 0.3, 0.9)]
        _, stripped, (positive, *_), calls = run(regions, zero_pass=True)
        self.assertEqual(calls.encoded.count(PERSONS[0]), 1)
        for entry, prepared_entry in zip(positive[1:], stripped[1:]):
            assert_same_entry(self, entry, expected_entry(prepared_entry, PERSONS[0]))

    def test_repeated_runs_are_identical(self):
        regions = [region(text, 0.33 * i, 0.1, 0.3, 0.9) for i, text in enumerate(PERSONS)]
        first = run(regions, zero_pass=True)[2][0]
        second = run(regions, zero_pass=True)[2][0]
        for a, b in zip(first, second):
            assert_same_entry(self, a, b)


class Robustness(unittest.TestCase):
    def test_token_lengths_follow_each_entry_without_truncation(self):
        long_region = WOMAN + " " + STATION.replace("railway station", "market hall")
        regions = [region(PERSONS[0], 0.0, 0.1, 0.3, 0.9), region(long_region, 0.4, 0.1, 0.5, 0.9)]
        self.assertEqual((token_length(STATION), token_length(PERSONS[0]), token_length(long_region)), (154, 77, 154))
        _, stripped, (positive, *_), _ = run(regions, zero_pass=False)
        self.assertEqual([entry[0].shape[1] for entry in positive], [154, 77, 154])
        for entry, prepared_entry, text in zip(positive, stripped, [STATION, PERSONS[0], long_region]):
            assert_same_entry(self, entry, expected_entry(prepared_entry, text))
            # a pure own-prompt mix: no zero padding, no foreign tokens
            native = fingerprint(text, "native")
            prepared_value = float(prepared_entry[0].flatten()[0])
            self.assertTrue(torch.allclose(entry[0], torch.full_like(entry[0], 0.25 * prepared_value + 0.75 * native)))

    def test_missing_pooled_output_keeps_comfyui_average_semantics(self):
        regions = [region(WOMAN)]
        source = prepared_positive(STATION, regions, FakeTranslator())
        source[1][1].pop("pooled_output")
        _, stripped, (positive, *_), _ = run(regions, zero_pass=True, positive=source)
        assert_same_entry(self, positive[1], expected_entry(stripped[1], WOMAN))
        torch.testing.assert_close(positive[1][1]["pooled_output"], torch.full((1, 4), fingerprint(WOMAN, "native")))

    def test_reference_without_pooled_output_leaves_the_entry_pooled_unchanged(self):
        regions = [region(WOMAN)]

        class NoPooled(Calls):
            def __call__(self, names, **kwargs):
                if names[0] == "CLIPTextEncode":
                    self.encoded.append(kwargs["text"])
                    return (encoded(kwargs["text"], "native", pooled=False),)
                return super().__call__(names, **kwargs)

        source = prepared_positive(STATION, regions, FakeTranslator())
        calls = NoPooled()
        positive = reference.apply_reference_conditioning(
            source_positive=source, positive=source, negative=encoded(NEGATIVE, "prepared"),
            process={"prompt_pos": STATION, "prompt_neg": NEGATIVE, "unload_models_after_use": False},
            sampled_in={"regional_conditioning": regions}, model_pipe={"clip": object()},
            reference_conditioning_weight=0.75, call_node=calls, outputs=lambda v: v,
            translate=FakeTranslator(), timed=lambda *a: contextlib.nullcontext())[0]
        self.assertTrue(torch.equal(positive[1][1]["pooled_output"], source[1][1]["pooled_output"]))

    def test_translations_feed_prepare_and_reference_alike(self):
        translate = FakeTranslator(prefix="EN: ")
        regions = [region("eine Frau im roten Mantel"), region("ein Mann mit Hut", 0.6, 0.1, 0.3, 0.9)]
        _, stripped, (positive, _, _, translation_pos, _), calls = run(regions, zero_pass=True, global_text="Bahnhofshalle",
                                                                       translate=translate)
        self.assertEqual(calls.encoded, ["EN: Bahnhofshalle", "EN: blurry, deformed",
                                         "EN: eine Frau im roten Mantel", "EN: ein Mann mit Hut"])
        self.assertEqual(translation_pos.text, "EN: Bahnhofshalle")
        assert_same_entry(self, positive[1], expected_entry(stripped[1], "EN: eine Frau im roten Mantel"))

    def test_negative_conditioning_is_unchanged(self):
        for regions in ([], [region(WOMAN)]):
            for zero_pass in (True, False):
                _, _, (_, negative, *_), _ = run(regions, zero_pass=zero_pass)
                expected = legacy(encoded(STATION, "prepared"), encoded(NEGATIVE, "prepared"), STATION)[1]
                self.assertEqual(len(negative), 1)
                assert_same_entry(self, negative[0], expected[0])

    def test_regions_come_from_the_pipe_that_carries_the_conditioning(self):
        self.assertEqual(reference.pipe_regions({"regional_conditioning": [1]}, {"regional_conditioning": [2]}), [1])
        self.assertEqual(reference.pipe_regions({}, {"regional_conditioning": [2]}), [2])
        self.assertEqual(reference.pipe_regions({"regional_conditioning": []}, {"regional_conditioning": [2]}), [])


class UnassignableConditioning(unittest.TestCase):
    def assert_refused(self, positive, regions, message):
        with self.assertRaisesRegex(reference.ReferenceAssignmentError, message):
            run(regions, zero_pass=True, positive=positive)

    def test_extra_entry_is_refused_before_any_encode(self):
        regions = [region(WOMAN)]
        positive = prepared_positive(STATION, regions, FakeTranslator()) + encoded("foreign", "prepared")
        calls = Calls()
        with self.assertRaisesRegex(reference.ReferenceAssignmentError, "3 entries, the pipe lists 1 global \\+ 1"):
            reference.apply_reference_conditioning(
                source_positive=positive, positive=positive, negative=encoded(NEGATIVE, "prepared"),
                process={"prompt_pos": STATION, "prompt_neg": NEGATIVE}, sampled_in={"regional_conditioning": regions},
                model_pipe={"clip": object()}, reference_conditioning_weight=0.75, call_node=calls,
                outputs=lambda v: v, translate=FakeTranslator(), timed=lambda *a: contextlib.nullcontext())
        self.assertEqual(calls.encoded, [])

    def test_missing_regional_entry_is_refused(self):
        regions = [region(WOMAN), region(PERSONS[0], 0.6, 0.1, 0.3, 0.9)]
        positive = prepared_positive(STATION, regions[:1], FakeTranslator())
        self.assert_refused(positive, regions, "2 entries, the pipe lists 1 global \\+ 2")

    def test_entry_with_other_strength_or_range_is_refused(self):
        regions = [region(WOMAN)]
        for key, value in (("strength", 1.0), ("start_percent", 0.2), ("end_percent", 0.5)):
            positive = prepared_positive(STATION, regions, FakeTranslator())
            positive[1][1] = dict(positive[1][1], **{key: value})
            with self.subTest(key=key):
                self.assert_refused(positive, regions, "entry 2 does not carry")

    def test_swapped_regions_are_refused(self):
        regions = [region(PERSONS[0], 0.0, 0.1, 0.3, 0.9, strength=1.4), region(PERSONS[1], 0.6, 0.1, 0.3, 0.9, strength=2.0)]
        positive = prepared_positive(STATION, list(reversed(regions)), FakeTranslator())
        self.assert_refused(positive, regions, "entry 2 does not carry")

    def test_spatial_global_entry_is_refused(self):
        regions = [region(WOMAN)]
        positive = prepared_positive(STATION, regions, FakeTranslator())
        positive[0][1] = dict(positive[0][1], area=("percentage", 1.0, 1.0, 0.0, 0.0))
        self.assert_refused(positive, regions, "entry 1 carries spatial keys")

    def test_translation_that_differs_from_the_prepared_text_is_refused(self):
        regions = [region(WOMAN)]
        long_translation = FakeTranslator(prefix=STATION + " ")
        with self.assertRaisesRegex(reference.ReferenceAssignmentError, "regional prompt 1 has 154 tokens, its prepared entry 77"):
            run(regions, zero_pass=True, translate=long_translation, prepare_translate=FakeTranslator())


class StructuralShare(unittest.TestCase):
    """The probe of the root-cause analysis: concept channels instead of fingerprints."""

    def share(self, zero_pass):
        regions = [region(WOMAN)]

        def concept(text, kind):
            tensor = torch.zeros(1, token_length(text), WIDTH)
            tensor[..., 0] = float("railway station" in text)
            tensor[..., 1] = float("adult woman" in text)
            return [[tensor, {"pooled_output": tensor[:, 0, :4].clone()}]]

        class ConceptPrepare(prepare.CMKSamplerPrepareSDXLPipe):
            def _encode_sdxl_plus(self, clip, width, height, size_cond_factor, text):
                return concept(text, "prepared")

        class ConceptCalls(Calls):
            def __call__(self, names, **kwargs):
                if names[0] == "CLIPTextEncode":
                    return (concept(kwargs["text"], "native"),)
                return super().__call__(names, **kwargs)

        with patch.object(prepare, "translate_prompt", FakeTranslator()):
            source, _, _ = ConceptPrepare()._apply_regional_conditioning(concept(STATION, "prepared"), None, W, H, 4, regions)
        stripped = strip._hybrid_positive_conditioning(source, instantid_mode="HYBRID", zero_pass_mode=zero_pass)
        positive = reference.apply_reference_conditioning(
            source_positive=source, positive=stripped, negative=concept(NEGATIVE, "prepared"),
            process={"prompt_pos": STATION, "prompt_neg": NEGATIVE, "unload_models_after_use": False},
            sampled_in={"regional_conditioning": regions}, model_pipe={"clip": object()},
            reference_conditioning_weight=0.75, call_node=ConceptCalls(), outputs=lambda v: v,
            translate=FakeTranslator(), timed=lambda *a: contextlib.nullcontext())[0]
        dims = (H // 8, W // 8)
        conds = [dict(metadata, model_conds={}, uuid=uuid.uuid4()) for _, metadata in positive]
        samplers.resolve_areas_and_cond_masks_multidim(conds, dims, "cpu")
        numerator, denominator = torch.zeros(dims), torch.zeros(dims)
        for (tensor, _), cond in zip(positive, conds):
            result = samplers.get_area_and_mult(cond, torch.zeros(1, 4, *dims), torch.tensor([500.0]))
            h, w, y, x = result.area if result.area is not None else (*dims, 0, 0)
            numerator[y:y + h, x:x + w] += result.mult[0, 0] * float(tensor[..., 1].mean())
            denominator[y:y + h, x:x + w] += result.mult[0, 0]
        return float((numerator / denominator)[int(0.55 * dims[0]), int(0.34 * dims[1])])

    def test_regional_person_share_is_no_longer_replaced_by_the_scene(self):
        self.assertAlmostEqual(self.share(True), 1.75 / 2.75, places=6)   # 0.64, before 0.16
        self.assertAlmostEqual(self.share(False), 0.5, places=6)          # 0.50, before 0.13


class NodeWiring(unittest.TestCase):
    SOURCES = ("pipe/instantid/cmk_instantid_sampler.py", "pipe/instantid/cmk_instantid_regional_advanced.py")

    def test_both_samplers_use_the_one_shared_reference_mix(self):
        for relative in self.SOURCES:
            source = (PACKAGE_ROOT / relative).read_text(encoding="utf-8")
            with self.subTest(relative=relative):
                self.assertEqual(source.count("apply_reference_conditioning("), 1)
                self.assertNotIn('"ConditioningAverage"', source)
                self.assertNotIn("CLIPTextEncode", source)
                captured = source.index("source_positive = positive\n")
                stripped = source.index("positive = _hybrid_positive_conditioning(")
                selected = source.index('sampled_in.get("conditioning_identity_pos")')
                self.assertLess(selected, captured)
                self.assertLess(captured, stripped)
                self.assertIn("source_positive=source_positive", source)

    def test_sampler_prepare_and_reference_share_region_selection_and_translation(self):
        source = (PACKAGE_ROOT / "pipe/cmk_sampler_prepare.py").read_text(encoding="utf-8")
        self.assertIn("active = active_regions(regions)", source)
        self.assertIn("translation = regional_prompt_translation(region, translate_prompt)", source)
        tree = ast.parse((PACKAGE_ROOT / "pipe/instantid/cmk_instantid_reference.py").read_text(encoding="utf-8"))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
        self.assertTrue({"active_regions", "regional_prompt_translation"} <= imported)

    def test_stored_instantid_results_get_new_cache_keys(self):
        revision = persistent.INSTANTID_REFERENCE_REVISION
        for class_type in ("CMKInstantIDSamplerSDXLPipe", "CMKInstantIDRegionalAdvancedSDXLPipe"):
            # Advanced chains the adaptive-handoff revision onto the reference revision.
            self.assertTrue(persistent.NODE_BEHAVIOUR_REVISIONS[class_type].startswith(revision))
            prompt = {
                "1": {"class_type": "CMKPipeCreateImage", "inputs": {"prompt": "station"}},
                "2": {"class_type": class_type, "inputs": {"PROCESS": ["1", 0]}},
                # InstantID results are stored under the downstream gate (cmk_family_result)
                "3": {"class_type": "CMKSamplerBypassGate", "inputs": {"SAMPLED": ["2", 0]}},
            }
            persistent._FINGERPRINT_MEMO_PROMPT_ID = None
            current = persistent.build_node_fingerprint(prompt, "3", ("CMKSamplerBypassGate",), "test-schema")[0]
            with patch.dict(persistent.NODE_BEHAVIOUR_REVISIONS, clear=True):
                persistent._FINGERPRINT_MEMO_PROMPT_ID = None
                previous = persistent.build_node_fingerprint(prompt, "3", ("CMKSamplerBypassGate",), "test-schema")[0]
            self.assertNotEqual(current, previous)


if __name__ == "__main__":
    unittest.main()
