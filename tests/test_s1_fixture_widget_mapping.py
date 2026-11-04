"""S1-T6 — the scenario overrides address subgraph widgets through the
definition links (harness.scenarios.widget_targets).  This checks that mapping
against what the real frontend wrote into the exported API prompts.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_ROOT))

from harness.scenarios import (  # noqa: E402
    instance_widget_values,
    load_api_prompt,
    load_fixture_workflow,
    widget_targets,
)

CMK_TEST_TIER = "T0"
MANIFEST = TESTS_ROOT / "fixtures" / "manifest.json"


class WidgetMapping(unittest.TestCase):
    def test_instance_widgets_reach_their_inner_targets(self):
        checked = 0
        for entry in json.loads(MANIFEST.read_text(encoding="utf-8"))["fixtures"]:
            fixture = entry["name"]
            workflow = load_fixture_workflow(fixture)
            prompt = load_api_prompt(fixture)
            definitions = {item["id"] for item in (workflow.get("definitions") or {}).get("subgraphs", [])}
            for node in workflow["nodes"]:
                if node["type"] not in definitions:
                    continue
                for name, value in instance_widget_values(fixture, node["id"]).items():
                    targets = widget_targets(fixture, node["id"], name)
                    with self.subTest(fixture=fixture, widget=name):
                        self.assertTrue(targets, "promoted widget without inner target")
                        for inner, input_name in targets:
                            self.assertEqual(prompt[inner]["inputs"][input_name], value)
                            checked += 1
        # Every fixture carries at least its module's global ENABLE.
        self.assertGreaterEqual(checked, 12)


if __name__ == "__main__":
    unittest.main()
