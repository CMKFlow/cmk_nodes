"""S1-T6 §3.C — MaskDetailer READY state machine with the real Intake node.

S0 live → S1 capture → S2 locked/empty mask → S3 locked/painted mask →
input change → S0.  The mask editor is simulated by writing the alpha channel
the Intake reads (mask = 1 - alpha).  Processing after S3 needs real SDXL
sampling and is documented as open (T2).
"""

from __future__ import annotations

import unittest

from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
MD_CLASSES = ("CMKMaskDetailerPrepare", "CMKMaskDetailerProcess")


class MaskDetailerStates(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        result = cls.run_case("mask_detailer_sequence")
        cls.steps = {} if result is None else {step["label"]: step for step in result["steps"]}
        cls.intake = None if result is None else result["intake"]

    def _ui(self, label):
        return self.steps[label]["mask_ui"][self.intake]

    def test_state_sequence(self):
        expected = [
            ("S0", ["LIVE · CAPTURE CURRENT"], False),
            ("S1", ["CAPTURED · MASK NOW"], False),
            ("S2", ["LOCKED · PAINT MASK"], False),
            ("S3", ["LOCKED"], True),
            ("S0_after_input_change", ["LIVE · CAPTURE CURRENT"], False),
        ]
        for label, ui, ready in expected:
            with self.subTest(state=label):
                self.assertEqual(self._ui(label), ui)
                self.assertEqual(self.steps[label]["ready"], ready)

    def test_pending_states_pass_image_unchanged_and_run_no_processing(self):
        for label in ("S0", "S1", "S2", "S0_after_input_change"):
            with self.subTest(state=label):
                step = self.steps[label]
                self.assertTrue(step["succeeded"], step["errors"])
                if label in ("S0", "S0_after_input_change"):
                    self.assertEqual(step["outputs"]["gate_image"]["hash"], step["outputs"]["loader_image"]["hash"])
                for class_type in MD_CLASSES:
                    self.assertNotIn(class_type, step["function_calls"])

    def test_k6_7_no_model_materialisation_while_pending(self):
        for label in ("S0", "S1", "S2", "S0_after_input_change"):
            with self.subTest(state=label):
                seams = self.steps[label]["seams"]
                self.assertEqual(seams["stopped_calls"], {})
                self.assertEqual(seams["original_calls"], {})

    def test_s3_ready_hands_over_to_prepare(self):
        step = self.steps["S3"]
        # Prepare starts materialising the model, where the T1 stop seam ends the run.
        self.assertEqual(step["seams"]["stopped_calls"], {"prepare.resolve_postprocess_model": 1})
        self.assertEqual([e["node_type"] for e in step["errors"]], ["CMKMaskDetailerPrepare"])

    def test_s3_processing_result(self):
        self.skipTest("OPEN (T2): MaskDetailer sampling after S3 needs a real SDXL model")

    def test_k6_1_model_out_is_unmaterialised_spec(self):
        self.skipTest("OPEN (T2): needs MaskDetailer Prepare/Process to complete (K6-1..K6-6)")

    def test_baseline_z7_pending_public_process_carries_pixels(self):
        for label in ("S0", "S1", "S2"):
            with self.subTest(state=label):
                step = self.steps[label]
                self.assertTrue(step["public_process_has_tensor"])
                self.assertIn("image", step["public_process_keys"])
                self.assertIn("mask_detailer_bypassed", step["public_process_keys"])

    @unittest.expectedFailure  # Z-7
    def test_public_process_carries_no_pixels(self):
        for label in ("S0", "S1", "S2"):
            self.assertFalse(self.steps[label]["public_process_has_tensor"], label)

    def test_baseline_pending_input_has_no_status_in_visual(self):
        for label in ("S0", "S1", "S2"):
            with self.subTest(state=label):
                self.assertEqual(self.steps[label]["visual"].get("8418"), [])

    @unittest.expectedFailure  # OD-10 target: explicit pending_input status
    def test_pending_input_is_published_as_status(self):
        for label in ("S0", "S1", "S2"):
            statuses = [p["status"] for p in self.steps[label]["visual"].get("8418", [])]
            self.assertIn("pending_input", statuses, label)


if __name__ == "__main__":
    unittest.main()
