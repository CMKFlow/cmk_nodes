"""S1 §5 — every API-prompt fixture passes ComfyUI's own validate_prompt.

Combo values (checkpoint, VAE, SAM, YOLO, upscale models) are validated
against the model lists of the target instance, so the instance's
``--extra-model-paths-config`` YAML is required.  Set CMK_EXTRA_MODEL_PATHS to
its path; without it the test is reported as skipped (S1 acceptance requires
it to run).  Nothing is executed and no model is loaded.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from harness.restart import run_scenario

CMK_TEST_TIER = "T1"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class FixtureValidation(unittest.TestCase):
    def test_all_api_prompts_pass_validate_prompt(self):
        model_paths = os.environ.get("CMK_EXTRA_MODEL_PATHS", "")
        if not model_paths or not Path(model_paths).is_file():
            self.skipTest("CMK_EXTRA_MODEL_PATHS not set to the instance's extra model paths YAML")
        manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        root = Path(tempfile.mkdtemp(prefix="cmk-s1-fixval-"))
        try:
            results = {
                entry["name"]: run_scenario(
                    "validate_fixture", root / entry["name"],
                    fixture=entry["name"], extra_model_paths=model_paths,
                )
                for entry in manifest["fixtures"]
            }
        finally:
            shutil.rmtree(root, ignore_errors=True)
        # A fixture of a shipped workflow with third-party nodes (not loaded by
        # the harness) may fail only because such a declared node is missing.
        third_party = {entry["name"]: set(entry.get("third_party_nodes", ())) for entry in manifest["fixtures"]}
        invalid = {}
        for name, r in results.items():
            if r["valid"]:
                continue
            error = r["error"] or {}
            missing = (error.get("extra_info") or {}).get("class_type")
            if error.get("type") == "missing_node_type" and missing in third_party[name]:
                continue
            invalid[name] = (r["error"], r["node_errors"])
        self.assertEqual(invalid, {})


if __name__ == "__main__":
    unittest.main()
