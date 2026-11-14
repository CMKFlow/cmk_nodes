"""CMK diagnostic display rule (2.8): Preview / Compare show what came out, a
diagnostic shows where, how and with which effect.

The renderers (CMK Preview Render, CMK Preview Board) leave out plain full
images (source, primary and Refine results, final result, ordinary node
preview); detection, ROI, masks, alpha, pasteback, overlaps, shares and
differences stay.  The payloads themselves are not changed.
"""

from __future__ import annotations

import copy
import importlib
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import test_faceprocess_advanced_multimodel as advanced_tests  # noqa: E402
from test_faceprocess_advanced_multimodel import TEST_PACKAGE, image, merge_branches  # noqa: E402

CMK_TEST_TIER = "T0"
output = importlib.import_module(f"{TEST_PACKAGE}.utils.comfy_preview_output")
rule = importlib.import_module(f"{TEST_PACKAGE}.utils.cmk_diagnostic")
concat = importlib.import_module(f"{TEST_PACKAGE}.nodes.utils.diagnostic_concat").CMKDiagnosticConcat
controlnet = importlib.import_module(f"{TEST_PACKAGE}.nodes.controlnet.controlnet")
IMG = torch.rand(1, 32, 48, 3)


def payload(node, titles, **kwargs):
    return rule.make_diagnostic_payload(title=node, node=node, stages=[{"title": t, "image": IMG} for t in titles],
                                        summary="summary text", **kwargs)


def visible(data):
    return [step["title"] for step in output._preview_steps_from_payload(data)]


class Rule(unittest.TestCase):
    def test_every_rule_entry_names_a_real_producer_stage(self):
        package = Path(rule.__file__).resolve().parents[1]
        sources = "\n".join(path.read_text(encoding="utf-8") for path in package.rglob("*.py")
                            if "tests" not in path.parts and path.name != "cmk_diagnostic.py")
        for node, titles in rule.PLAIN_IMAGE_STAGES.items():
            with self.subTest(node=node):
                self.assertIn(node.split(" / ")[0], sources)
                for title in titles:
                    # FaceProcess builds "03 Primary {mode}" / "05 Refine {mode}" from templates.
                    head = title.split(" ")[0]
                    if node == "CMK FaceProcess" and head in ("Primary", "Refine"):
                        found = f"{head} {{" in sources
                    else:
                        found = re.search(rf"[\"' ]{re.escape(title)}[\"']", sources) is not None
                    self.assertTrue(found, f"{node}: stage {title!r} is produced nowhere")
        for node in rule.PLAIN_PREVIEW_NODES:
            with self.subTest(node=node):
                self.assertIn(node.split(" / ")[0], sources)

    def test_bare_titles(self):
        self.assertEqual(rule.bare_stage_title("05 Final"), "Final")
        self.assertEqual(rule.bare_stage_title("04 · FINAL"), "FINAL")
        self.assertEqual(rule.bare_stage_title("09 Refine Sharpen"), "Refine Sharpen")
        self.assertEqual(rule.bare_stage_title("Source"), "Source")

    def test_smart_detailer_keeps_its_spatial_stages(self):
        data = payload("CMK Smart Detailer", ["01 Source", "02 Detection", "03 Sampling ROI", "04 Mask", "05 Pasteback",
                                              "06 Detailer", "07 Difference", "08 Final"])
        self.assertEqual(visible(data), ["02 Detection", "03 Sampling ROI", "04 Mask", "05 Pasteback", "07 Difference"])

    def test_same_title_is_judged_per_node(self):
        crop = payload("CMK Face Crop", ["01 Source", "02 Cropped Image"])
        self.assertEqual(visible(crop), ["01 Source"], "Face Crop's Source shows the selection with boxes and landmarks")

    def test_plain_preview_nodes_show_no_preview(self):
        prepare = rule.make_diagnostic_payload(title="FaceProcess Prepare", node="CMK FaceProcess Prepare Multi-Model -Pipe-",
                                               previews=[IMG], summary="MODE : RESTORE")
        self.assertEqual(visible(prepare), [])
        mask = rule.make_diagnostic_payload(title="Face Mask", node="CMK Face Mask", previews=[IMG])
        self.assertEqual(visible(mask), ["Preview 1"], "technical previews of other nodes stay")

    def test_all_plain_stages_do_not_fall_back_to_previews(self):
        upscale = payload("CMK Smart Upscaler", ["01 Source", "02 Final"], previews=[IMG, IMG])
        self.assertEqual(visible(upscale), [])

    def test_a_diagnostic_without_visible_step_renders_its_text(self):
        upscale = payload("CMK Smart Upscaler", ["01 Source", "02 Final"])
        text_card = output.render_summary_panel(title="CMK Smart Upscaler", text="summary text")
        for caption in ("Standard", "Details"):
            with self.subTest(caption=caption):
                panel = output.render_preview_panel(upscale, caption=caption)
                self.assertEqual(panel.shape, text_card.shape)

    def test_payload_data_is_kept(self):
        data = payload("CMK Smart Detailer", ["01 Source", "02 Detection", "08 Final"])
        before = copy.deepcopy([stage["title"] for stage in data["stages"]])
        output.render_preview_panel(data, caption="Details")
        self.assertEqual([stage["title"] for stage in data["stages"]], before, "caches and other consumers keep everything")

    def test_controlnet_names_its_images(self):
        diagnostic = controlnet._build_controlnet_diagnostic(
            title="ControlNet Prepare", node="CMK ControlNet Prepare", use_controlnet=True, controlnet_model="cn",
            image_source="Base Image", reference_image="", apply_mask=False, preprocessor="canny", resolution=512,
            base_image=IMG, mask=torch.ones(1, 32, 48), controlnet_image=IMG, enabled=True, log="")
        self.assertEqual([stage["title"] for stage in diagnostic["stages"]], ["01 Source", "02 Mask", "03 Control Image"])
        self.assertEqual(visible(diagnostic), ["02 Mask", "03 Control Image"])
        self.assertEqual(len(diagnostic["preview"]), 3, "the payload's previews are unchanged")

    def test_concat_timeline_details(self):
        timeline = concat().concat(
            "CMK Flow · FaceProcess",
            rule.make_diagnostic_payload(title="Prepare", node="CMK Detailer Prepare Multi-Model -Pipe-", previews=[IMG]),
            diagnostic_2=payload("CMK Smart Detailer", ["01 Source", "02 Detection", "07 Difference", "08 Final"]),
        )[0]
        groups = timeline["metadata"]["diagnostic_groups"]
        panels = output._concat_detail_panels(timeline)
        self.assertEqual(len(panels), 2, "Detection and Difference; no Prepare preview, no Source, no Final")
        self.assertEqual(len(groups), 2)
        board = output.render_preview_board_panels(timeline, caption="Details")
        self.assertEqual(len(board), 2)


class FaceProcess(unittest.TestCase):
    def setUp(self):
        self.runner = advanced_tests._Runner()
        self.runner.setUp()
        self.addCleanup(self.runner.doCleanups)

    def test_branch_tiles(self):
        source = image()
        restore = self.runner.branch("restore", source, restore_visibility=0.6)[6]
        self.assertEqual(visible(restore), ["02 Detection · Assignment", "03 Restore Region", "04 Restore Alpha",
                                            "05 Visibility", "07 Difference Source→Primary"])
        refined = self.runner.branch("sdxl", source, refine_mode="Detail", select_face="Center")[6]
        self.assertEqual(visible(refined), ["02 Detection", "03 Sampling ROI", "04 Mask", "05 Pasteback",
                                            "07 Difference Source→Primary", "08 Refine Mask",
                                            "10 Difference Primary→Refine", "11 Difference Source→Final"],
                         "the three differences stay with Refine")
        self.assertEqual(len(refined["stages"]), 12, "the payload keeps all stages")

    def test_merge_tiles(self):
        source = image()
        d1 = self.runner.branch("restore", source, restore_visibility=0.7)
        d2 = self.runner.branch("z_image_turbo", source)
        diagnostic = merge_branches([d1, d2, None], source, ("Restore", "Z-Image Turbo", "None"))[3]
        self.assertEqual(visible(diagnostic), ["01 · INPUT LAYERS", "02 · OVERLAP", "03 · COMPOSITE"])
        self.assertEqual(len(diagnostic["stages"]), 4)

    def test_rendering_does_not_touch_the_result(self):
        source = image()
        branch = self.runner.branch("restore", source, refine_mode="Sharpen")
        result = branch[0].clone()
        output.render_preview_panel(branch[6], caption="Details")
        output.render_preview_panel(branch[6], caption="Standard")
        self.assertTrue(torch.equal(branch[0], result))
        self.assertTrue(np.isfinite(output.render_preview_panel(branch[6], caption="Details")).all())


if __name__ == "__main__":
    unittest.main()
