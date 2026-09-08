import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RegionalConditioningUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            ROOT / "web" / "js" / "cmk_regional_conditioning_ui.js"
        ).read_text(encoding="utf-8")

    def test_default_size_is_only_applied_to_new_nodes(self):
        self.assertIn(
            "if (applyDefaultSize && !node._cmkRegionalLoadedFromWorkflow)",
            self.source,
        )
        self.assertIn(
            'if (hook === "onConfigure") this._cmkRegionalLoadedFromWorkflow = true;',
            self.source,
        )

    def test_loaded_workflow_marks_node_before_rebuilding_ui(self):
        self.assertIn("loadedGraphNode(node)", self.source)
        self.assertIn("node._cmkRegionalLoadedFromWorkflow = true;", self.source)
        self.assertIn("schedule(node, false);", self.source)


if __name__ == "__main__":
    unittest.main()
