"""S1 §4 V-C4 — port type compatibility judged by ComfyUI's own rule.

Uses the real node classes (loaded in an isolated harness process, no models)
and ``comfy_execution.validation.validate_node_input``; there is no CMK-side
type logic.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from harness.restart import run_scenario

CMK_TEST_TIER = "T1"

POSTPROCESS_TARGETS = (
    "CMK Flow · Detailer SDXL", "CMK Flow · Detailer SDXL · Advanced",
    "CMK Flow · FaceProcess SDXL", "CMK Flow · FaceProcess SDXL · Advanced",
    "CMK Flow · FaceRebuild SDXL", "CMK Flow · FaceRebuild SDXL · Advanced",
)


class PortTypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(tempfile.mkdtemp(prefix="cmk-s1-types-"))
        try:
            cls.result = run_scenario("graph_type_check", root)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_all_existing_links_are_type_compatible(self):
        self.assertEqual(self.result["link_findings"], [])

    def test_baseline_od3_faceswap_outputs_not_composable(self):
        expected = sorted(
            [source, target, role, f"CMK_RESULT_{role}", {"MODEL": "CMK_MODEL_PIPE", "IMAGE": "IMAGE", "LOG": "CMK_LOG_PIPE"}[role]]
            for source in ("CMK Flow · FaceSwap", "CMK Flow · FaceSwap · Advanced")
            for target in POSTPROCESS_TARGETS
            for role in ("MODEL", "IMAGE", "LOG")
        )
        self.assertEqual(sorted(self.result["incompatible_compositions"]), expected)

    @unittest.expectedFailure  # OD-3 / Phase 4: one public PostProcess transport dialect
    def test_regular_postprocess_modules_are_mutually_composable(self):
        self.assertEqual(self.result["incompatible_compositions"], [])


if __name__ == "__main__":
    unittest.main()
