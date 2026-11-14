"""CMK 2.8 Prepare cards: every Prepare text card of Detailer (Standard /
Advanced), FaceProcess (Standard / Advanced) and MaskDetailer is laid out for
the card renderer and never enlarges the image cards of its timeline.

The Prepare diagnostics come from the existing test set-ups of each module;
nothing loads a model.  ``CMK_PRINT_CARDS=1`` prints the cards.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import test_detailer_advanced_multimodel as detailer_advanced  # noqa: E402
import test_detailer_multimodel as detailer  # noqa: E402
import test_faceprocess_advanced_multimodel as faceprocess_advanced  # noqa: E402
import test_faceprocess_multimodel as faceprocess  # noqa: E402
import test_mask_detailer_multimodel as mask_detailer  # noqa: E402
from test_mask_detailer_multimodel import TEST_PACKAGE  # noqa: E402

import importlib  # noqa: E402

output = importlib.import_module(f"{TEST_PACKAGE}.utils.comfy_preview_output")
cards = importlib.import_module(f"{TEST_PACKAGE}.utils.prepare_card")

CMK_TEST_TIER = "T0"


class CardContract(unittest.TestCase):
    def check(self, name, payload):
        if os.environ.get("CMK_PRINT_CARDS"):
            print(f"\n--- {name}: {payload.get('title')}\n{payload['details']}")
        metadata = payload.get("metadata") or {}
        self.assertEqual(metadata.get(output.SUMMARY_CARD_LAYOUT), output.SUMMARY_CARD_FOOTER, name)
        lines = payload["details"].splitlines()
        self.assertLessEqual(len(lines), cards.CARD_ROWS, f"{name}: too many lines")
        for line in lines:
            self.assertLessEqual(len(line), cards.CARD_COLUMNS, f"{name}: {line!r}")
        card = output.render_summary_panel(title=payload.get("title") or name, text=payload["details"], footer_rule=True)
        stage = output._diagnostic_card(np.zeros((640, 480, 3), np.uint8), title="02 Detection", summary="Info: x",
                                        details="", metadata={}, caption="Details")
        timeline = output.combine_preview_panels([card, stage])
        self.assertEqual(timeline.shape[0], stage.shape[0], f"{name}: the image card keeps its height")


class Detailer(CardContract, unittest.TestCase):
    def setUp(self):
        detailer._Recorder(self)

    def test_families(self):
        for family in ("None", "SDXL", "Z-Image Turbo"):
            with self.subTest(family=family):
                args = detailer._mm_args(model_family=family)
                diagnostic = detailer.mm_mod.CMKDetailerPrepareMultiModelPipe().prepare(**args)[2]
                self.check(f"Detailer {family}", diagnostic)

    def test_the_card_holds_the_prepare_log(self):
        result = detailer.mm_mod.CMKDetailerPrepareMultiModelPipe().prepare(**detailer._mm_args(model_family="SDXL"))
        text = result[2]["details"]
        for row in ("STATUS", "STEPS", "SAMPLER", "PROMPT SOURCE", "POSITIVE PROMPT", "detailed hands"):
            self.assertIn(row, text)


class DetailerAdvanced(CardContract, unittest.TestCase):
    def test_slots(self):
        for families in (("SDXL", "Z-Image Turbo", "SDXL"), ("SDXL", "None", "Z-Image Turbo"), ("None", "None", "None")):
            with self.subTest(families=families):
                diagnostic = detailer_advanced.AdvancedPrepare().run_prepare(families)[-1]
                self.check(f"Detailer Advanced {families}", diagnostic)
                for slot in (1, 2, 3):
                    self.assertIn(f"DETAILER {slot}", diagnostic["details"], "every slot in slot (merge) order")


class FaceProcess(CardContract, unittest.TestCase):
    def test_modes(self):
        runner = faceprocess.PrepareSpec()
        runner.setUp()
        self.addCleanup(runner.doCleanups)
        for mode in ("None", "Restore", "SDXL", "Z-Image Turbo"):
            with self.subTest(mode=mode):
                diagnostic = runner.face(mode, **{"MODEL SDXL": faceprocess.SDXL_SPEC, "MODEL ZIT": faceprocess.ZIT_SPEC})[2]
                self.check(f"FaceProcess {mode}", diagnostic)


class FaceProcessAdvanced(CardContract, unittest.TestCase):
    def test_slots(self):
        image = torch.zeros(1, 32, 32, 3)
        prepare = faceprocess_advanced.PREPARE()
        for modes in (("Restore", "SDXL", "Z-Image Turbo"), ("None", "Restore", "None")):
            with self.subTest(modes=modes):
                diagnostic = prepare.prepare(image, PROCESS=faceprocess_advanced.PROCESS,
                                             **{**faceprocess_advanced.inputs(modes), "MODEL SDXL": faceprocess_advanced.SDXL_SPEC,
                                                "MODEL ZIT": faceprocess_advanced.ZIT_SPEC})[-1]
                self.check(f"FaceProcess Advanced {modes}", diagnostic)
        disabled = prepare.prepare(image, **faceprocess_advanced.inputs(enable="DISABLE"))[-1]
        self.check("FaceProcess Advanced DISABLE", disabled)
        self.assertIn("GLOBAL", disabled["details"])


class MaskDetailer(CardContract, mask_detailer.Recorder):
    def test_families_and_intake(self):
        for family in ("None", "SDXL", "Z-Image Turbo"):
            with self.subTest(family=family):
                self.check(f"MaskDetailer {family}", self.prepare(family)[4])
        intake = mask_detailer.Intake.intake_prepare(self, "SDXL", mask_detailer.terminal())[4]
        self.check("MaskDetailer Intake", intake)
        self.assertIn("cmk_mask_detailer/snapshot-7.png", intake["details"].replace("\n", "").replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
