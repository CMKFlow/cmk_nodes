"""FaceRebuild v2 (2.8): MODE control and the 2.8 diagnostic of the
Standard and Advanced executors, against the unchanged SDXL FaceRebuild.

The real executor, FaceRebuild Prepare (geometry, masks) and Pasteback run;
only the face detector, ContentGuard and the InstantID sampling are replaced
(the "rebuilt" ROI is the inverted target ROI).
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_facerebuild_v2_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import numpy as np  # noqa: E402
import torch  # noqa: E402

v2 = importlib.import_module(f"{TEST_PACKAGE}.nodes.instantid_face_rebuild_v2")
legacy = importlib.import_module(f"{TEST_PACKAGE}.nodes.instantid_face_rebuild_advanced")
rebuild = importlib.import_module(f"{TEST_PACKAGE}.nodes.instantid_face_rebuild")
detailer_module = importlib.import_module(f"{TEST_PACKAGE}.nodes.instantid_face_detailer")
detector = importlib.import_module(f"{TEST_PACKAGE}.engine.detector_engine")
engine = importlib.import_module(f"{TEST_PACKAGE}.engine.instantid_face_rebuild")
output = importlib.import_module(f"{TEST_PACKAGE}.utils.comfy_preview_output")
segs_diagnostic = importlib.import_module(f"{TEST_PACKAGE}.utils.segs_diagnostic")
mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")

CMK_TEST_TIER = "T0"
H, W = 320, 480
FACES = [{"bbox": [70, 90, 150, 190]}, {"bbox": [290, 80, 400, 210]}]   # left small, right large
torch.manual_seed(11)
IMAGE = torch.rand(1, H, W, 3) * 0.6 + 0.2
SOURCE = torch.rand(1, 128, 128, 3)
COMMON = {"TOTAL STEPS": 20, "HEAD AREA": 0.75, "NECK AREA": 0.4, "MASK FEATHER": 16, "WORKING RESOLUTION": "768",
          "DETECT MODEL": "fake", "DETECT SIZE": 640, "instantid_model": "ip", "controlnet_model": "cn",
          "provider": "CPU", "cfg": 4.0, "identity_noise": 0.0, "sampler": "euler", "scheduler": "karras"}


# The InstantID Face Detailer's summary lines (CMKInstantIDFaceDetailerSDXL.detail).
DETAIL_SUMMARY = "\n".join([
    "Workflow          : INSTANTID FACE REBUILD · IMG2IMG DETAILER", "Inpaint           : disabled",
    "First Pass        : none", "KPS source        : unchanged Target ROI", "Latent source     : VAE-encoded Target ROI",
    "Sampling schedule : 10-20/20 | add_noise=enable", "Identity strength : 1.00", "Pose strength     : 0.70",
    "Identity noise    : 0.75", "CFG               : 4.00", "Method            : euler | karras",
])


def card_fields(card):
    return {label.strip(): value.strip() for label, _sep, value in
            (line.partition(" : ") for line in card["summary"].splitlines()) if _sep}


def standard_values(**overrides):
    values = {"MODEL": {"ckpt": "x"}, "PROCESS": {"content_guard_enabled": False}, "IMAGE": IMAGE, "LOG": None,
              "FACEREBUILD ENABLE": True, "TARGET FACE": "Largest", "SOURCE FACE": SOURCE, "PASTEBACK NECK": False,
              "SAMPLING START": 0, "IDENTITY STRENGTH": 1.0, "POSE STRENGTH": 0.7, **COMMON}
    values.update(overrides)
    return values


def advanced_values(**overrides):
    values = {"MODEL": {"ckpt": "x"}, "PROCESS": {"content_guard_enabled": False}, "IMAGE": IMAGE, "LOG": None,
              "FACEREBUILD ENABLE": True, **COMMON}
    for index, target in ((1, "Leftmost"), (2, "Rightmost"), (3, "Center")):
        values.update({f"FACE {index} ENABLE": index < 3, f"TARGET FACE {index}": target, f"SOURCE FACE {index}": SOURCE,
                       f"PASTEBACK NECK {index}": False, f"SAMPLING START {index}": 0,
                       f"IDENTITY STRENGTH {index}": 1.0, f"POSE STRENGTH {index}": 0.7})
    values.update(overrides)
    return values


def rgb(stage):
    value = stage["image"]
    value = value[0] if getattr(value, "ndim", 0) == 4 else value
    value = torch.as_tensor(np.asarray(value)).float()
    return value / 255.0 if value.max() > 1.0 else value


def visible(payload):
    return [step["title"] for step in output._preview_steps_from_payload(payload)]


class Fakes(unittest.TestCase):
    def setUp(self):
        def patch(owner, name, value):
            original = getattr(owner, name)
            self.addCleanup(setattr, owner, name, original)
            setattr(owner, name, value)

        self.sampled = []
        patch(detector.CMKDetectorEngine, "detect_image", lambda self, rgb, settings: [dict(face) for face in FACES])
        patch(rebuild, "get_content_guard", lambda: types.SimpleNamespace(inspect_image=lambda *a, **k: None))
        sampled = self.sampled

        def detail(self, **inputs):
            roi = inputs["TARGET ROI"]
            sampled.append(tuple(roi.shape))
            diagnostic = rebuild.make_diagnostic_payload(
                title="InstantID Face Rebuild · Img2Img Detailer", node="CMK InstantID Face Detailer SDXL",
                stages=[{"title": "01 Target ROI", "image": roi}, {"title": "02 Rebuilt ROI", "image": 1.0 - roi}],
                summary=DETAIL_SUMMARY, details=DETAIL_SUMMARY)
            return inputs["MODEL"], inputs["PROCESS"], 1.0 - roi, inputs["LOG"], diagnostic

        patch(detailer_module.CMKInstantIDFaceDetailerSDXL, "detail", detail)


class ModeAndContract(unittest.TestCase):
    def test_registration(self):
        for name in ("CMKInstantIDFaceRebuildSDXLV2", "CMKInstantIDFaceRebuildAdvancedSDXLV2", "CMKFaceRebuildModeEnable"):
            self.assertIn(name, mappings.NODE_CLASS_MAPPINGS)
        self.assertEqual(v2.CMKInstantIDFaceRebuildSDXLV2.RETURN_NAMES, legacy.CMKInstantIDFaceRebuildSDXL.RETURN_NAMES)
        self.assertEqual(v2.CMKInstantIDFaceRebuildSDXLV2.INPUT_TYPES.__func__, legacy.CMKInstantIDFaceRebuildSDXL.INPUT_TYPES.__func__,
                         "same inputs: FACEREBUILD ENABLE is fed by CMK FaceRebuild · Mode")

    def test_mode_enable(self):
        node = v2.CMKFaceRebuildModeEnable
        self.assertEqual(node.INPUT_TYPES()["required"]["facerebuild_mode"][0], ["ENABLE", "DISABLE"])
        self.assertEqual(node.read("ENABLE"), (True,))
        self.assertEqual(node.read("DISABLE"), (False,))


class Standard(Fakes):
    def run_both(self, **overrides):
        values = standard_values(**overrides)
        return legacy.CMKInstantIDFaceRebuildSDXL().rebuild(**values), v2.CMKInstantIDFaceRebuildSDXLV2().rebuild(**values)

    def test_processing_is_unchanged(self):
        old, new = self.run_both()
        self.assertTrue(torch.equal(old[2], new[2]), "same image")
        self.assertEqual(old[1], new[1])
        self.assertEqual(old[3], new[3], "same LOG")

    def test_legacy_diagnostic_is_unchanged(self):
        old, _new = self.run_both()
        groups = old[5]["metadata"]["diagnostic_groups"]
        self.assertEqual([group["node"] for group in groups], ["CMK InstantID Face Rebuild Prepare",
                                                               "CMK InstantID Face Detailer SDXL",
                                                               "CMK InstantID Face Rebuild Pasteback"])
        self.assertEqual(old[5]["title"], "CMK Flow · FaceRebuild SDXL")

    def test_stages(self):
        _old, new = self.run_both()
        prepare, process = new[5]["metadata"]["diagnostic_groups"]
        self.assertEqual((prepare["node"], process["node"]), ("CMK FaceRebuild v2 · Prepare", "CMK FaceRebuild v2"))
        self.assertEqual(prepare["title"], "FaceRebuild · Prepare")
        self.assertEqual(visible(prepare), [], "Prepare is a text card")
        fields = card_fields(prepare)
        self.assertEqual(fields["TARGET"], "Largest")
        self.assertEqual(fields["SCHEDULE"], "10-20/20 | add_noise=enable")
        self.assertEqual(fields["METHOD"], "euler | karras · cfg 4.00")
        self.assertEqual(fields["CONDITIONING"], "75% native reference")
        self.assertIn("25% CMK prepared", prepare["summary"], "continuation line, nothing dropped")
        titles = [stage["title"] for stage in process["stages"]]
        self.assertEqual(titles, ["01 Rebuild ROI", "02 Inpaint Mask", "03 Pasteback", "04 Difference"])
        self.assertEqual(visible(process), titles, "all spatial stages are rendered")
        self.assertIn("Sampling schedule : 10-20/20", process["summary"])
        self.assertIn("Outside Unchanged   : PASS", process["summary"])

    def test_geometry_is_the_processing_geometry(self):
        _old, new = self.run_both()
        prepare, process = new[5]["metadata"]["diagnostic_groups"]
        fields = card_fields(prepare)
        slot = {"face_bbox": [int(v) for v in fields["FACE BOX"].split(", ")], "roi": [int(v) for v in fields["ROI"].split(", ")]}
        self.assertEqual(slot["face_bbox"], [290, 80, 400, 210], "Largest")
        left, top, right, bottom = slot["roi"]
        roi = rgb(process["stages"][0])
        orange = torch.tensor(segs_diagnostic.ROI_COLOR, dtype=torch.float32) / 255.0
        self.assertTrue(torch.allclose(roi[top + 1, (left + right) // 2], orange, atol=0.01), "ROI frame at the ROI")
        self.assertTrue(torch.allclose(roi[5, 5], IMAGE[0, 5, 5] * 0.4, atol=0.01) or roi[5, 5].max() < 0.05,
                        "outside the ROI dimmed (or under the legend)")
        geometry = engine.FaceRebuildGeometry(face_bbox=tuple(slot["face_bbox"]), roi=tuple(slot["roi"]),
                                              working_size=(0, 0), feather=16)
        paste = engine.make_pasteback_mask(geometry, include_neck=False)
        y, x = np.unravel_index(np.argmax(paste), paste.shape)
        pasted = rgb(process["stages"][2])
        self.assertTrue(torch.allclose(pasted[top + y, left + x], new[2][0, top + y, left + x], atol=0.01),
                        "weight 1: the pasted result")
        inpaint = engine.make_inpaint_mask(geometry)
        y, x = np.unravel_index(np.argmax(inpaint), inpaint.shape)
        green = torch.tensor(segs_diagnostic.MASK_COLOR, dtype=torch.float32) / 255.0
        self.assertTrue(torch.allclose(rgb(process["stages"][1])[top + y, left + x],
                                       IMAGE[0, top + y, left + x] * 0.45 * 0.15 + green * 0.85, atol=0.01),
                        "inpaint mask tinted on the target")

    def test_pasteback_neck_widens_the_pasteback(self):
        _old, off = self.run_both()
        _old, on = self.run_both(**{"PASTEBACK NECK": True})
        pixels = lambda result: int(result[5]["metadata"]["diagnostic_groups"][-1]["stages"][2]["subtitle"].split(" · ")[1].split()[0])
        self.assertGreater(pixels(on), pixels(off))

    def test_bypass_is_a_text_card(self):
        new = v2.CMKInstantIDFaceRebuildSDXLV2().rebuild(**standard_values(**{"FACEREBUILD ENABLE": False}))
        self.assertIs(new[2], IMAGE)
        self.assertEqual(visible(new[5]), [])
        self.assertEqual(self.sampled, [])


class Advanced(Fakes):
    def run_both(self, **overrides):
        values = advanced_values(**overrides)
        return (legacy.CMKInstantIDFaceRebuildAdvancedSDXL().rebuild(**values),
                v2.CMKInstantIDFaceRebuildAdvancedSDXLV2().rebuild(**values))

    def test_processing_is_unchanged(self):
        old, new = self.run_both()
        self.assertTrue(torch.equal(old[2], new[2]))
        self.assertEqual(old[3], new[3])
        self.assertEqual(len(self.sampled), 4, "two slots, each sampled once per executor")

    def test_stages_follow_the_pasteback_order(self):
        old, new = self.run_both()
        self.assertEqual(len(old[5]["metadata"]["diagnostic_groups"]), 6, "legacy: three diagnostics per slot")
        *cards, process = new[5]["metadata"]["diagnostic_groups"]
        self.assertEqual(process["node"], "CMK FaceRebuild v2 · Advanced")
        self.assertEqual(process["metadata"]["order"], [2, 1], "the larger right face is sampled and pasted first")
        self.assertEqual([card["title"] for card in cards], ["FaceRebuild · Prepare", "Slot 2 · Prepare", "Slot 1 · Prepare"],
                         "module card, then the slots in processing order")
        self.assertEqual(card_fields(cards[0])["SLOTS"], "F2 → F1 (processing order)")
        self.assertEqual(card_fields(cards[1])["TARGET"], "Rightmost")
        self.assertEqual(card_fields(cards[2])["TARGET"], "Leftmost")
        self.assertNotIn("TARGET", card_fields(cards[0]), "the module card holds no slot settings")
        self.assertEqual([visible(card) for card in cards], [[], [], []])
        titles = [stage["title"] for stage in process["stages"]]
        self.assertEqual(titles, ["01 Target Faces", "02 Inpaint Masks", "03 Pasteback", "04 Difference"],
                         "no Overlap stage: the two faces do not overlap")
        self.assertEqual(visible(process), titles)
        self.assertEqual(process["stages"][0]["subtitle"], "F2 #1 · F1 #2")

    def test_overlap_only_when_slots_overlap(self):
        global FACES
        saved = FACES
        FACES = [{"bbox": [150, 90, 240, 200]}, {"bbox": [215, 95, 310, 205]}]
        self.addCleanup(lambda: globals().__setitem__("FACES", saved))
        _old, new = self.run_both()
        titles = [stage["title"] for stage in new[5]["metadata"]["diagnostic_groups"][-1]["stages"]]
        self.assertIn("04 Overlap", titles)

    def test_one_active_slot_and_no_active_slot(self):
        _old, new = self.run_both(**{"FACE 2 ENABLE": False})
        *cards, process = new[5]["metadata"]["diagnostic_groups"]
        self.assertEqual(process["metadata"]["order"], [1])
        self.assertEqual([card["title"] for card in cards], ["FaceRebuild · Prepare", "Slot 1 · Prepare"],
                         "no card for the disabled slot")
        _old, new = self.run_both(**{"FACE 1 ENABLE": False, "FACE 2 ENABLE": False})
        self.assertIs(new[2], IMAGE)
        self.assertEqual(visible(new[5]), [], "no active slot: text card")


class Rendering(Fakes):
    """The rendered Preview Render timeline: Prepare cards do not resize the image cards."""

    @staticmethod
    def divider(img):
        line = np.asarray(output._design()["line"], dtype=np.uint8)
        rows = np.flatnonzero(np.all(img == line.reshape(1, 1, 3), axis=2).sum(axis=1) >= max(48, int(img.shape[1] * 0.58)))
        return int(rows[-1])

    def check(self, payload):
        groups = payload["metadata"]["diagnostic_groups"]
        *cards, process = groups
        from PIL import Image, ImageDraw
        draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        font = output._load_mono_font(28)
        width = 960 - 2 * int(output._design()["pad"])
        for card in cards:
            for line in card["summary"].splitlines():
                self.assertEqual(len(output._wrap_mono_line(draw, line, font, width)), 1, f"wraps: {line!r}")
        alone = output.render_preview_panel(process, caption="Details")
        timeline = output.render_preview_panel(payload, caption="Details")
        self.assertEqual(timeline.shape[0], alone.shape[0], "the image cards keep their height")
        card = output.render_summary_panel(title=cards[0]["title"], text=cards[0]["summary"], footer_rule=True)
        self.assertLessEqual(card.shape[0] - self.divider(card), alone.shape[0] - self.divider(alone),
                             "the text card's information area fits the image cards'")
        return timeline

    def test_standard(self):
        self.check(v2.CMKInstantIDFaceRebuildSDXLV2().rebuild(**standard_values())[5])

    def test_advanced_two_slots(self):
        self.check(v2.CMKInstantIDFaceRebuildAdvancedSDXLV2().rebuild(**advanced_values())[5])

    def test_other_text_cards_are_unchanged(self):
        plain = output.render_summary_panel(title="Detailer Prepare", text="a : b\nc : d")
        self.assertEqual(self.divider(plain), self.divider(output.render_summary_panel(title="Detailer Prepare", text="a : b\nc : d")))
        self.assertLess(self.divider(plain), 200, "without the option the rule stays under the title")


if __name__ == "__main__":
    unittest.main()
