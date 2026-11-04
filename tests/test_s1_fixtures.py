"""S1 §3.0 / §5 — API-prompt fixtures exported by the real ComfyUI frontend.

Until the fixtures are exported (tests/FIXTURES_EXPORT.md) the tests report a
skip with the pending step; S1 acceptance requires zero skips here.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_ROOT))

from validator.graph_model import load_definitions  # noqa: E402
from validator.report import canonical_definitions  # noqa: E402

CMK_TEST_TIER = "T0"
FIXTURES = TESTS_ROOT / "fixtures"
MANIFEST = FIXTURES / "manifest.json"
REQUIRED = {
    "detailer_std", "detailer_adv", "faceprocess_std", "faceprocess_adv",
    "facerebuild_std", "facerebuild_adv", "faceswap_std", "faceswap_adv",
    "maskdetailer", "upscale_save", "upscale_save_legacy_fullflow", "chain_postprocess",
}


class ApiPromptFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else None

    def setUp(self):
        if self.manifest is None:
            self.skipTest("S1-T4 pending: API prompt fixtures not exported (see tests/FIXTURES_EXPORT.md)")

    def test_required_fixtures_present(self):
        self.assertEqual(REQUIRED - {entry["name"] for entry in self.manifest["fixtures"]}, set())

    def test_fixture_files_unchanged_since_recording(self):
        for entry in self.manifest["fixtures"]:
            for key in ("workflow", "api_prompt"):
                path = FIXTURES / entry[f"{key}_file"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry[f"{key}_sha256"], path.name)

    def test_api_prompt_freshness(self):
        """A fixture recorded from a canonical definition becomes stale when the
        canonical subgraph changes; it must then be re-exported."""
        canon = canonical_definitions()
        stale = []
        for entry in self.manifest["fixtures"]:
            for definition in entry["definitions"]:
                if not definition["canonical_at_export"]:
                    continue  # deliberately legacy fixtures keep their embedded state
                current = canon.get(definition["name"])
                if current is None or current.signature() != definition["signature"]:
                    stale.append((entry["name"], definition["name"]))
        self.assertEqual(stale, [], "re-export the listed fixtures")

    def test_workflow_and_prompt_agree(self):
        for entry in self.manifest["fixtures"]:
            workflow = FIXTURES / entry["workflow_file"]
            recorded = {d["name"]: d["signature"] for d in entry["definitions"]}
            current = {g.name: g.signature() for g in load_definitions(workflow)}
            self.assertEqual(current, recorded, entry["name"])


if __name__ == "__main__":
    unittest.main()
