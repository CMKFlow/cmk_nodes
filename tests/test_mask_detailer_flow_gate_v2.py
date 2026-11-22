"""CMK Mask Detailer Flow Gate v2: the module diagnostic behind the gate.

DISABLE never requests DIAGNOSTIC and blocks the diagnostic output, so the
Preview Render and the public diagnostic no longer pull Process, Prepare and
Intake.  ENABLE requests it independent of READY (S0-S2 pending, S3 spatial).
IMAGE, PROCESS, MODEL, LOG and VISUAL are those of the unchanged Flow Gate.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from test_mask_detailer_multimodel import TEST_PACKAGE  # noqa: E402

legacy = importlib.import_module(f"{TEST_PACKAGE}.nodes.image.mask_detailer")
v2 = importlib.import_module(f"{TEST_PACKAGE}.nodes.image.mask_detailer_multimodel")
mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")

CMK_TEST_TIER = "T0"
GATE = "8447"
CONNECTED = ("MODEL BYPASS", "PROCESS BYPASS", "IMAGE BYPASS", "LOG BYPASS", "VISUAL BYPASS",
             "MODEL ACTIVE", "PROCESS ACTIVE", "IMAGE ACTIVE", "LOG ACTIVE", "VISUAL ACTIVE", "READY", "DIAGNOSTIC")
PROMPT = {GATE: {"class_type": "CMKMaskDetailerFlowGateV2",
                 "inputs": {name: ["1", 0] for name in CONNECTED} | {"enabled": ["2", 0]}}}
IMAGE = torch.zeros(1, 8, 8, 3)
BYPASS = {"MODEL BYPASS": {"model_family": "sdxl"}, "PROCESS BYPASS": {"model_family": "sdxl"},
          "IMAGE BYPASS": IMAGE, "LOG BYPASS": {"type": "CMK_LOG_PIPE", "blocks": []},
          "VISUAL BYPASS": {"type": "CMK_VISUAL_PIPE", "version": 1, "providers": []}}
ACTIVE = {"MODEL ACTIVE": {"model_family": "sdxl"}, "PROCESS ACTIVE": {"type": "MASK"}, "IMAGE ACTIVE": IMAGE + 1,
          "LOG ACTIVE": {"type": "CMK_LOG_PIPE", "blocks": []},
          "VISUAL ACTIVE": {"type": "CMK_VISUAL_PIPE", "version": 1, "providers": []}}
DIAGNOSTIC = {"type": "CMK_DIAGNOSTIC", "title": "CMK Flow · MaskDetailer"}


def lazy(**inputs):
    return v2.CMKMaskDetailerFlowGateV2().check_lazy_status(prompt=PROMPT, unique_id=GATE, **inputs)


class Contract(unittest.TestCase):
    def test_registration_and_ports(self):
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKMaskDetailerFlowGateV2"], v2.CMKMaskDetailerFlowGateV2)
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKMaskDetailerFlowGate"], legacy.CMKMaskDetailerFlowGate)
        gate = v2.CMKMaskDetailerFlowGateV2
        self.assertEqual(gate.RETURN_NAMES, (*legacy.CMKMaskDetailerFlowGate.RETURN_NAMES, "diagnostic"))
        types, old = gate.INPUT_TYPES(), legacy.CMKMaskDetailerFlowGate.INPUT_TYPES()
        self.assertEqual(types["optional"]["DIAGNOSTIC"], ("CMK_DIAGNOSTIC", {"lazy": True}))
        self.assertEqual({k: v for k, v in types["optional"].items() if k != "DIAGNOSTIC"}, old["optional"])
        self.assertNotIn("DIAGNOSTIC", legacy.CMKMaskDetailerFlowGate.INPUT_TYPES()["optional"])


class Mode(unittest.TestCase):
    """Outer MODE of the 2.8 Mask Detailer: ENABLE / DISABLE only (the model
    family is chosen inside by CMK Detailer Family Enable)."""

    def test_mode_node(self):
        node = mappings.NODE_CLASS_MAPPINGS["CMKMaskDetailerModeEnable"]
        self.assertIs(node, v2.CMKMaskDetailerModeEnable)
        self.assertEqual(node.INPUT_TYPES()["required"]["maskdetailer_mode"], (["ENABLE", "DISABLE"], {"default": "ENABLE"}))
        self.assertEqual((node.read("ENABLE"), node.read("DISABLE")), ((True,), (False,)))
        self.assertEqual((node.RETURN_TYPES, node.RETURN_NAMES), (("BOOLEAN",), ("ENABLE",)))

    def test_family_enable_is_unchanged(self):
        family = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_detailer_prepare_multimodel").CMKDetailerFamilyEnable
        self.assertEqual(family.INPUT_TYPES()["required"]["model_family"][0], ["None", "SDXL", "Z-Image Turbo"])
        self.assertEqual(family.read("None"), (False, "None"))
        self.assertEqual(family.read("Z-Image Turbo"), (True, "Z-Image Turbo"))

    def test_inner_family_reaches_the_prepare_as_a_literal(self):
        inline = importlib.import_module(f"{TEST_PACKAGE}.cmk_prompt_family_inline")
        prompt = {"8452": {"class_type": "CMKDetailerFamilyEnable", "inputs": {"model_family": "Z-Image Turbo"}},
                  "8450": {"class_type": "CMKMaskDetailerPrepareMultiModel", "inputs": {"model_family": ["8452", 1]}}}
        result, count = inline.inline_family_links(prompt)
        self.assertEqual((count, result["8450"]["inputs"]["model_family"]), (1, "Z-Image Turbo"))


class LazyRequests(unittest.TestCase):
    def test_disable_never_requests_the_diagnostic(self):
        self.assertEqual(lazy(enabled=False), ["MODEL BYPASS"])
        self.assertEqual(lazy(enabled=False, **{k: v for k, v in BYPASS.items() if k != "VISUAL BYPASS"}),
                         ["VISUAL BYPASS"])
        self.assertEqual(lazy(enabled=False, **BYPASS), [], "READY, ACTIVE and DIAGNOSTIC stay unrequested")

    def test_enable_requests_the_diagnostic_independent_of_ready(self):
        self.assertEqual(lazy(enabled=True), ["READY", "DIAGNOSTIC"])
        self.assertEqual(lazy(enabled=True, READY=False, **BYPASS), ["DIAGNOSTIC"], "S0-S2: pending diagnostic")
        self.assertEqual(lazy(enabled=True, READY=True, **ACTIVE), ["DIAGNOSTIC"], "S3: spatial diagnostic")
        self.assertEqual(lazy(enabled=True, READY=True, DIAGNOSTIC=DIAGNOSTIC, **ACTIVE), [])

    def test_unconnected_diagnostic_is_not_requested(self):
        prompt = {GATE: {"class_type": "CMKMaskDetailerFlowGateV2", "inputs": {"READY": ["1", 0]}}}
        self.assertEqual(v2.CMKMaskDetailerFlowGateV2().check_lazy_status(
            enabled=True, READY=True, prompt=prompt, unique_id=GATE, **ACTIVE), [])


class Outputs(unittest.TestCase):
    def check(self, enabled, ready, inputs):
        old = legacy.CMKMaskDetailerFlowGate().gate(enabled=enabled, READY=ready, **inputs)
        new = v2.CMKMaskDetailerFlowGateV2().gate(enabled=enabled, READY=ready, DIAGNOSTIC=DIAGNOSTIC, **inputs)
        self.assertEqual(len(new), 6)
        for before, after in zip(old, new[:5]):
            if isinstance(before, torch.Tensor):
                self.assertTrue(torch.equal(before, after))
            elif isinstance(before, dict):
                self.assertEqual({k: v for k, v in before.items() if k != "image"},
                                 {k: v for k, v in after.items() if k != "image"})
            else:
                self.assertEqual(before, after)
        return new[5]

    def test_disable_blocks_the_diagnostic(self):
        self.assertIsInstance(self.check(False, None, BYPASS), v2.ExecutionBlocker)

    def test_enable_passes_the_diagnostic_in_every_intake_state(self):
        self.assertIs(self.check(True, False, BYPASS), DIAGNOSTIC, "S0-S2")
        self.assertIs(self.check(True, True, ACTIVE), DIAGNOSTIC, "S3")


if __name__ == "__main__":
    unittest.main()
