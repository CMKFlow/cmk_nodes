"""Mask Detailer 2.8, round 1: discrete two-model setup (SDXL / Z-Image Turbo).

Model loading, LoRA, text encoding and the legacy sampling are replaced by
recorders; the tests check which family is requested and materialised, the
prompt/LoRA/seed/sampling-entry contract and the unchanged public outputs.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_mask_detailer_multimodel_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import numpy as np  # noqa: E402
import torch  # noqa: E402

mm = importlib.import_module(f"{TEST_PACKAGE}.nodes.image.mask_detailer_multimodel")
legacy = importlib.import_module(f"{TEST_PACKAGE}.nodes.image.mask_detailer")
mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")
output = importlib.import_module(f"{TEST_PACKAGE}.utils.comfy_preview_output")
segs_diagnostic = importlib.import_module(f"{TEST_PACKAGE}.utils.segs_diagnostic")
translation = importlib.import_module(f"{TEST_PACKAGE}.utils.cmk_translation")

CMK_TEST_TIER = "T0"
PREPARE = mm.CMKMaskDetailerPrepareMultiModel
PROCESS_NODE = mm.CMKMaskDetailerProcessMultiModel
torch.manual_seed(5)
IMAGE = torch.rand(1, 64, 96, 3) * 0.5
MASK = torch.zeros(1, 64, 96)
MASK[:, 20:36, 40:56] = 1.0
SOURCE = {"result_contract": "family_neutral", "seed": 1234, "prompt_pos": "portrait", "prompt_neg": "blurry",
          "mask": MASK}
SDXL_SPEC = {"model_family": "sdxl", "ckpt_name": "ck.safetensors"}
ZIT_SPEC = {"model_family": "z_image_turbo", "diffusion_model_name": "z.safetensors"}


def settings(**overrides):
    values = {"fill_mask_holes": False, "solid_mask": False, "lora_name": "sdxl_lora.safetensors", "lora_strength": 0.8,
              "zit_lora_name": "zit_lora.safetensors", "zit_lora_strength": 0.6, "prompt_pos": "sharp eyes",
              "steps": 30, "cfg": 7.0, "sampling_entry": 45.0, "zit_steps": 8, "zit_cfg": 1.0, "zit_sampling_entry": 60.0,
              "sampler": "euler", "scheduler": "karras", "zit_sampler": "res_multistep", "zit_scheduler": "simple",
              "zit_model_shift": 3.0}
    values.update(overrides)
    return values


class Recorder(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls

        class LoRA:
            def load_loras(self, model, clip, opt_lora_syntax=""):
                calls.append(("lora", opt_lora_syntax))
                return model, clip, "", opt_lora_syntax

        def patch(owner, name, value):
            original = getattr(owner, name)
            self.addCleanup(setattr, owner, name, original)
            setattr(owner, name, value)

        patch(mm, "CMKLoRATextLoader", LoRA)
        # Never the real translation service (it reads the user's configuration).
        self.translate = lambda text: translation.TranslationResult(str(text or ""), "disabled")
        patch(mm, "translate_prompt", lambda text: self.translate(text))
        patch(mm, "release_zit_resources", lambda reason: calls.append(("release_zit", reason)))
        patch(mm, "resolve_postprocess_model", lambda spec: (calls.append(("load_sdxl", spec["ckpt_name"])),
                                                             {"model": "sdxl-model", "clip": "sdxl-clip", "vae": "sdxl-vae"})[1])
        patch(mm, "resolve_zit_postprocess_model", lambda spec: (calls.append(("load_zit", spec["diffusion_model_name"])),
                                                                 {"model": "zit-model", "clip": None, "vae": "zit-vae"})[1])
        patch(mm, "ensure_zit_text_encoder", lambda model: (calls.append(("text_encoder",)), ("zit-clip", None))[1])
        patch(mm, "evict_zit_text_encoder", lambda model: (calls.append(("evict",)), "RELEASED")[1])
        patch(mm, "_node", lambda name, method, clip=None, text=None: (calls.append(("encode", clip, text)), [f"cond:{text}"])[1])
        patch(mm.CMKDetailerPreparePipe, "_encode", staticmethod(lambda clip, text: (calls.append(("encode", clip, text)), f"cond:{text}")[1]))
        patch(mm, "_call_node", lambda names, *args: (calls.append((names[0],) + tuple(str(a) for a in args)), [f"{names[0]}"])[1])

        class InvertingSEGSDetailer:
            """Sampling replaced by inverting the crop; SEGSPaste stays real."""

            def doit(self, image, segs, **kwargs):
                calls.append(("sample", {key: kwargs[key] for key in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise")}))
                return (segs[0], [seg._replace(cropped_image=1.0 - seg.cropped_image) for seg in segs[1]])

        patch(legacy, "SEGSDetailer", InvertingSEGSDetailer)
        patch(legacy, "register_provider", lambda visual, **kwargs: visual)

    def prepare(self, family, **kwargs):
        values = {"MODEL SDXL": SDXL_SPEC, "MODEL ZIT": ZIT_SPEC, "PROCESS": dict(SOURCE), "IMAGE": IMAGE, **settings(**kwargs)}
        return PREPARE().prepare(model_family=family, **values)

    def names(self):
        return [call[0] for call in self.calls]


class Contract(unittest.TestCase):
    def test_registration_and_public_outputs(self):
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKMaskDetailerPrepareMultiModel"], PREPARE)
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKMaskDetailerProcessMultiModel"], PROCESS_NODE)
        self.assertEqual(PREPARE.RETURN_NAMES, legacy.CMKMaskDetailerPrepare.RETURN_NAMES)
        self.assertEqual(PREPARE.RETURN_TYPES, legacy.CMKMaskDetailerPrepare.RETURN_TYPES)
        self.assertEqual(PROCESS_NODE.RETURN_NAMES, legacy.CMKMaskDetailerProcess.RETURN_NAMES)
        self.assertEqual(PROCESS_NODE.RETURN_TYPES, legacy.CMKMaskDetailerProcess.RETURN_TYPES)

    def test_widgets_follow_the_detailer_template(self):
        required = list(PREPARE.INPUT_TYPES()["required"])
        self.assertEqual(required[:2], ["MODEL SDXL", "model_family"])
        self.assertEqual(PREPARE.INPUT_TYPES()["required"]["model_family"][0], ["None", "SDXL", "Z-Image Turbo"])
        self.assertIn("MODEL ZIT", PREPARE.INPUT_TYPES()["optional"])
        self.assertEqual(list(PROCESS_NODE.INPUT_TYPES()["required"]), ["MODEL", "MASK DETAILER PROCESS", "mask_expand",
                                                                         "mask_feather"], "seed and sampling come from PROCESS / the Prepare")

    def test_sampling_entry_range(self):
        spec = PREPARE.INPUT_TYPES()["required"]["sampling_entry"][1]
        self.assertEqual((spec["min"], spec["max"], spec["default"]), (0.0, 99.0, 45.0))
        self.assertAlmostEqual(mm.mask_entry_to_denoise(45), 0.55)
        self.assertAlmostEqual(mm.mask_entry_to_denoise(0), 1.0)
        self.assertAlmostEqual(mm.mask_entry_to_denoise(99), 0.01)

    def test_lazy_requests_only_the_family_model(self):
        node = PREPARE()
        self.assertEqual(node.check_lazy_status(model_family="None", **{"MODEL SDXL": None, "MODEL ZIT": None}), [])
        self.assertEqual(node.check_lazy_status(model_family="SDXL", **{"MODEL SDXL": None, "MODEL ZIT": None}), ["MODEL SDXL"])
        self.assertEqual(node.check_lazy_status(model_family="Z-Image Turbo", **{"MODEL SDXL": None, "MODEL ZIT": None}), ["MODEL ZIT"])
        self.assertEqual(node.check_lazy_status(model_family="Z-Image Turbo", **{"MODEL SDXL": None}), [],
                         "an unconnected MODEL ZIT is never requested")


class Families(Recorder):
    def test_sdxl(self):
        model, process, _log, _visual, diagnostic = self.prepare("SDXL")
        self.assertIs(model, SDXL_SPEC)
        self.assertEqual(self.names()[:3], ["release_zit", "load_sdxl", "lora"])
        self.assertNotIn("load_zit", self.names())
        self.assertIn(("lora", "<lora:sdxl_lora.safetensors:0.8>"), self.calls)
        self.assertIn(("encode", "sdxl-clip", "portrait\nsharp eyes"), self.calls, "source prompt + addition")
        self.assertIn(("encode", "sdxl-clip", "blurry"), self.calls, "source negative")
        self.assertEqual((process["model_family"], process["model"], process["result_contract"]), ("sdxl", "sdxl-model", "mask_detailer"))
        self.assertEqual(process["mask_detailer_sampling"], {"steps": 30, "cfg": 7.0, "entry": 45.0, "sampler": "euler",
                                                             "scheduler": "karras", "seed": 1234})
        self.assertEqual(diagnostic["metadata"]["model_family"], "sdxl")
        result = PROCESS_NODE.process(model, 8, 12, **{"MASK DETAILER PROCESS": process})
        sample = next(call[1] for call in self.calls if call[0] == "sample")
        self.assertEqual(sample, {"seed": 1234, "steps": 30, "cfg": 7.0, "sampler_name": "euler", "scheduler": "karras",
                                  "denoise": 0.55})
        self.assertEqual([group["node"] for group in result[5]["metadata"]["diagnostic_groups"]], ["CMKMaskDetailerProcessMultiModel"])
        self.assertIn("MODEL FAMILY     : SDXL", str(result[3]))

    def test_zit(self):
        model, process, *_ = self.prepare("Z-Image Turbo")
        self.assertIs(model, ZIT_SPEC)
        self.assertNotIn("load_sdxl", self.names())
        self.assertEqual(self.names()[:3], ["load_zit", "text_encoder", "lora"])
        self.assertIn(("lora", "<lora:zit_lora.safetensors:0.6>"), self.calls, "the ZIT LoRA, never the SDXL one")
        self.assertIn(("encode", "zit-clip", "portrait\nsharp eyes"), self.calls)
        self.assertIn("ConditioningZeroOut", self.names(), "negative = zeroed positive")
        self.assertIn(("ModelSamplingAuraFlow", "zit-model", "3.0"), self.calls, "shift on the LoRA-patched ZIT model")
        self.assertLess(self.names().index("encode"), self.names().index("evict"), "text encoder released after encoding")
        self.assertEqual(process["model_family"], "z_image_turbo")
        self.assertEqual(process["clip"], mm.ZIT_TEXT_ENCODER)
        self.assertEqual(process["mask_detailer_sampling"]["entry"], 60.0)
        PROCESS_NODE.process(model, 8, 12, **{"MASK DETAILER PROCESS": process})
        sample = next(call[1] for call in self.calls if call[0] == "sample")
        self.assertEqual((sample["steps"], sample["cfg"], sample["sampler_name"], sample["denoise"]), (8, 1.0, "res_multistep", 0.4))

    def test_none_is_the_bypass(self):
        model, process, log, _visual, _diagnostic = self.prepare("None")
        self.assertIsNone(model)
        self.assertEqual([name for name in self.names() if name != "encode"], [], "no model, LoRA or conditioning")
        self.assertEqual(process["model_family"], "none")
        result = PROCESS_NODE.process(model, 8, 12, **{"MASK DETAILER PROCESS": process})
        self.assertIs(result[2], IMAGE, "the image passes unchanged")
        self.assertNotIn("sample", self.names())
        self.assertIn("NONE · BYPASS", str(result[3]))

    def test_missing_seed_is_zero_and_mask_options_apply(self):
        _model, process, *_ = self.prepare("SDXL", solid_mask=True)
        self.assertEqual(process["mask_detailer_sampling"]["seed"], 1234)
        source = {key: value for key, value in SOURCE.items() if key != "seed"}
        values = {"MODEL SDXL": SDXL_SPEC, "PROCESS": source, "IMAGE": IMAGE, **settings()}
        _model, process, *_ = PREPARE().prepare(model_family="SDXL", **values)
        self.assertEqual(process["mask_detailer_sampling"]["seed"], 0)

    def test_prompt_without_addition(self):
        self.prepare("SDXL", prompt_pos="")
        self.assertIn(("encode", "sdxl-clip", "portrait"), self.calls)


def rgb(stage):
    """Stage image as float HWC in 0..1 (stages are stored as uint8)."""
    value = stage["image"]
    value = value[0] if getattr(value, "ndim", 0) == 4 else value
    value = torch.as_tensor(np.asarray(value)).float()
    return value / 255.0 if value.max() > 1.0 else value


class Diagnostic(Recorder):
    """Process: Sampling ROI, Mask, Pasteback, Difference on the source; the
    Prepare is a text card in the module timeline (Detailer contract)."""

    def run_sdxl(self, expand=8, feather=12):
        _model, process, _log, _visual, prepare = self.prepare("SDXL")
        result = PROCESS_NODE.process(SDXL_SPEC, expand, feather, opt_diagnostic=prepare, **{"MASK DETAILER PROCESS": process})
        groups = result[5]["metadata"]["diagnostic_groups"]
        return prepare, groups[1], result

    def test_stage_order_without_source_detection_or_result(self):
        _prepare, process, result = self.run_sdxl()
        self.assertEqual([stage["title"] for stage in process["stages"]],
                         ["01 Sampling ROI", "02 Mask", "03 Pasteback", "04 Difference"])
        self.assertEqual([step["title"] for step in output._preview_steps_from_payload(process)],
                         ["01 Sampling ROI", "02 Mask", "03 Pasteback", "04 Difference"], "all four are rendered")
        self.assertEqual(result[5]["title"], "CMK Flow · MaskDetailer")

    def test_prepare_is_a_text_card_in_the_timeline(self):
        prepare, _process, result = self.run_sdxl()
        self.assertEqual(prepare["stages"], [])
        self.assertEqual(output._preview_steps_from_payload(prepare), [], "no plain source image")
        self.assertIn("Active pixels", prepare["summary"])
        self.assertEqual([group["node"] for group in result[5]["metadata"]["diagnostic_groups"]],
                         ["CMKMaskDetailerPrepareMultiModel", "CMKMaskDetailerProcessMultiModel"])
        self.assertEqual(len(output._concat_detail_panels(result[5])), 4, "Prepare adds no image panel")

    def test_roi_geometry_is_the_pasteback_region(self):
        _prepare, process, _result = self.run_sdxl(expand=4)
        # mask 40..56 x 20..36 grown by 4 px -> 36..60 x 16..40; crop factor 2 around the centre
        self.assertEqual(process["metadata"]["mask_bounds"], [[36, 16, 60, 40]])
        self.assertEqual(process["metadata"]["sampling_roi"], [[24, 4, 72, 52]])
        self.assertEqual(process["metadata"]["image_size"], [96, 64])
        roi = rgb(process["stages"][0])
        self.assertEqual(tuple(roi.shape), (64, 96, 3), "drawn on the full source")
        orange = torch.tensor(segs_diagnostic.ROI_COLOR, dtype=torch.float32) / 255.0
        self.assertTrue(torch.allclose(roi[4, 24], orange, atol=0.01), "ROI frame at crop_region")
        self.assertTrue(torch.allclose(roi[30, 2], IMAGE[0, 30, 2] * 0.4, atol=0.01), "outside the ROI dimmed")

    def test_mask_overlay_on_the_source(self):
        _prepare, process, _result = self.run_sdxl(expand=0)
        mask = rgb(process["stages"][1])
        green = torch.tensor(segs_diagnostic.MASK_COLOR, dtype=torch.float32) / 255.0
        inside = mask[28, 48]
        self.assertTrue(torch.allclose(inside, IMAGE[0, 28, 48] * 0.45 * 0.15 + green * 0.85, atol=0.01), "tinted source")
        self.assertTrue(torch.allclose(mask[60, 90], IMAGE[0, 60, 90] * 0.45, atol=0.01), "source outside the mask")

    def test_pasteback_shows_the_result_with_its_weight(self):
        _prepare, process, result = self.run_sdxl(expand=0, feather=0)
        paste = rgb(process["stages"][2])
        after = result[2][0]
        self.assertTrue(torch.allclose(paste[28, 48], after[28, 48], atol=0.01), "weight 1: the pasted result")
        self.assertFalse(torch.equal(after[28, 48], IMAGE[0, 28, 48]), "the inverted crop was pasted")
        self.assertTrue(torch.equal(after[10, 10], IMAGE[0, 10, 10]), "outside the mask unchanged")
        self.assertLess(float(paste[10, 30].mean()), float(after[10, 30].mean()), "weight 0: dimmed")

    def test_difference_is_inside_the_mask(self):
        _prepare, process, result = self.run_sdxl(expand=0, feather=0)
        difference = rgb(process["stages"][3])
        self.assertGreater(float(difference[20:36, 40:56].mean()), 0.1)
        self.assertEqual(float(difference[:16].abs().sum()), 0.0)
        self.assertGreater(process["metadata"]["change"], 0.0)

    def test_without_change_there_is_no_difference_stage(self):
        class Identity:
            def doit(self, image, segs, **kwargs):
                return segs
        original = legacy.SEGSDetailer
        legacy.SEGSDetailer = Identity
        self.addCleanup(setattr, legacy, "SEGSDetailer", original)
        _prepare, process, _result = self.run_sdxl()
        self.assertEqual([stage["title"] for stage in process["stages"]], ["01 Sampling ROI", "02 Mask", "03 Pasteback"])

    def test_none_has_no_image_stage(self):
        _model, process, _log, _visual, prepare = self.prepare("None")
        result = PROCESS_NODE.process(None, 8, 12, opt_diagnostic=prepare, **{"MASK DETAILER PROCESS": process})
        self.assertEqual(result[5]["metadata"]["diagnostic_groups"][1]["stages"], [])
        for group in result[5]["metadata"]["diagnostic_groups"]:
            self.assertEqual(output._preview_steps_from_payload(group), [], f"{group['node']}: text only")

    def test_legacy_process_diagnostic_is_unchanged(self):
        _model, process, *_ = self.prepare("SDXL")
        legacy_result = legacy.CMKMaskDetailerProcess.process(SDXL_SPEC, 1, 20, 7.0, "euler", "karras", 0.5, 8, 12,
                                                             **{"MASK DETAILER PROCESS": process})
        self.assertEqual(legacy_result[5]["node"], "CMKMaskDetailerProcess")
        self.assertEqual([stage["title"] for stage in legacy_result[5]["stages"]], ["01 Source", "02 Sampling Mask", "03 Result"])
        prepare = legacy._prepare_diagnostic(IMAGE, MASK, False, False)
        self.assertEqual([stage["title"] for stage in prepare["stages"]], ["01 Source", "02 Effective Mask"])


def terminal(ready=True, mask=MASK):
    """CMK Mask Detailer Intake's MASK DETAILER SOURCE (snapshot, painted mask)."""
    return {"type": legacy.MASK_DETAILER_SOURCE, "version": 1, "image": IMAGE, "mask": mask if ready else torch.zeros_like(mask),
            "prompt_pos": "flow prompt", "prompt_neg": "flow negative", "terminal": True, "ready": ready,
            "snapshot": "cmk_mask_detailer/snapshot-7.png", "log": None}


class Translation(Recorder):
    """The central prompt translation, as in CMK Detailer Prepare (Multi-Model)."""

    def setUp(self):
        super().setUp()
        self.translated = []

        def translate(text):
            self.translated.append(text)
            return translation.TranslationResult(f"EN[{text}]", "translated") if text else translation.TranslationResult("", "empty")
        self.translate = translate

    def test_sdxl_translates_positive_and_negative(self):
        _model, process, log, *_ = self.prepare("SDXL")
        self.assertEqual(self.translated, ["portrait\nsharp eyes", "blurry"])
        self.assertIn(("encode", "sdxl-clip", "EN[portrait\nsharp eyes]"), self.calls)
        self.assertIn(("encode", "sdxl-clip", "EN[blurry]"), self.calls)
        self.assertEqual((process["prompt_pos"], process["prompt_neg"]), ("portrait\nsharp eyes", "blurry"), "the process keeps the original")
        self.assertIn("TRANSLATION", str(log).upper())

    def test_zit_translates_the_positive_only(self):
        _model, process, log, *_ = self.prepare("Z-Image Turbo")
        self.assertEqual(self.translated, ["portrait\nsharp eyes"])
        self.assertIn(("encode", "zit-clip", "EN[portrait\nsharp eyes]"), self.calls)
        self.assertIn("ConditioningZeroOut", self.names())
        self.assertEqual(process["prompt_pos"], "portrait\nsharp eyes")

    def test_disabled_translation_passes_the_text_and_logs_nothing(self):
        self.translate = lambda text: translation.TranslationResult(str(text or ""), "disabled")
        _model, _process, log, *_ = self.prepare("SDXL")
        self.assertIn(("encode", "sdxl-clip", "portrait\nsharp eyes"), self.calls)
        self.assertNotIn("TRANSLATION", str(log).upper())

    def test_none_and_pending_intake_translate_nothing(self):
        self.prepare("None")
        self.assertEqual(self.translated, [])


class Intake(Recorder):
    """Behind CMK Mask Detailer Intake: snapshot image + painted mask, seed of the flow PROCESS."""

    def intake_prepare(self, family, source, **kwargs):
        values = {"MODEL SDXL": SDXL_SPEC, "MODEL ZIT": ZIT_SPEC, "MASK DETAILER SOURCE": source,
                  "PROCESS": {"seed": 777, "prompt_pos": "ignored"}, **settings(**kwargs)}
        return PREPARE().prepare(model_family=family, **values)

    def test_pending_intake_requests_and_loads_nothing(self):
        node = PREPARE()
        for family in ("SDXL", "Z-Image Turbo"):
            with self.subTest(family=family):
                self.assertEqual(node.check_lazy_status(model_family=family, **{"MODEL SDXL": None, "MODEL ZIT": None,
                                                                              "MASK DETAILER SOURCE": terminal(False)}), [])
                model, process, log, _visual, prepare = self.intake_prepare(family, terminal(False))
                self.assertIsNone(model)
                self.assertEqual(process["model_family"], "none")
                self.assertEqual([name for name in self.names() if name != "encode"], [])
                self.assertIn("INTAKE PENDING", str(log))
                self.assertIn("Intake          : PENDING", prepare["summary"])
                result = PROCESS_NODE.process(model, 8, 12, opt_diagnostic=prepare, **{"MASK DETAILER PROCESS": process})
                self.assertIs(result[2], IMAGE)
                self.assertNotIn("sample", self.names())
                self.assertIn("NO PAINTED MASK", str(result[3]))

    def test_ready_intake_uses_snapshot_mask_prompt_and_flow_seed(self):
        node = PREPARE()
        self.assertEqual(node.check_lazy_status(model_family="SDXL", **{"MODEL SDXL": None, "MASK DETAILER SOURCE": terminal()}),
                         ["MODEL SDXL"])
        model, process, _log, _visual, prepare = self.intake_prepare("SDXL", terminal())
        self.assertIs(process["image"], IMAGE)
        self.assertTrue(torch.equal(process["mask"], MASK))
        self.assertIn(("encode", "sdxl-clip", "flow prompt\nsharp eyes"), self.calls)
        self.assertIn(("encode", "sdxl-clip", "flow negative"), self.calls)
        self.assertEqual(process["mask_detailer_sampling"]["seed"], 777, "the snapshot has no seed; the flow PROCESS has")
        self.assertIn("Snapshot        : cmk_mask_detailer/snapshot-7.png", prepare["summary"])
        result = PROCESS_NODE.process(model, 0, 0, opt_diagnostic=prepare, **{"MASK DETAILER PROCESS": process})
        sample = next(call[1] for call in self.calls if call[0] == "sample")
        self.assertEqual(sample["seed"], 777)
        titles = [stage["title"] for stage in result[5]["metadata"]["diagnostic_groups"][1]["stages"]]
        self.assertEqual(titles, ["01 Sampling ROI", "02 Mask", "03 Pasteback", "04 Difference"])

    def test_zit_behind_the_intake(self):
        model, process, *_ = self.intake_prepare("Z-Image Turbo", terminal())
        self.assertIs(model, ZIT_SPEC)
        self.assertNotIn("load_sdxl", self.names())
        self.assertEqual(process["mask_detailer_sampling"]["seed"], 777)


class LegacyUnchanged(unittest.TestCase):
    def test_legacy_prepare_shares_the_mask_and_source_helpers(self):
        source = {"mask": MASK, "prompt_pos": "a"}
        mask = legacy._prepared_mask(source, IMAGE, False, True)
        self.assertEqual(float(mask.sum()), 256.0)
        self.assertEqual(legacy._positive_text(source, "b"), "a\nb")
        with self.assertRaises(TypeError):
            legacy._prepare_source({}, None, "X")
        diagnostic = legacy._prepare_diagnostic(IMAGE, mask, False, True)
        self.assertEqual(diagnostic["node"], "CMKMaskDetailerPrepare")
        self.assertEqual([stage["title"] for stage in diagnostic["stages"]], ["01 Source", "02 Effective Mask"])


if __name__ == "__main__":
    unittest.main()
