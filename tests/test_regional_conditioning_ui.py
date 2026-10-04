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

    def test_node_uses_800_px_minimum_and_migrates_legacy_default(self):
        self.assertIn("const MIN_NODE_HEIGHT = 800;", self.source)
        self.assertIn("const LEGACY_DEFAULT_NODE_HEIGHT = 1225;", self.source)
        self.assertIn("height < MIN_NODE_HEIGHT || isLegacyDefault", self.source)

    def test_prompt_rows_remain_flexible_without_spacer_rows(self):
        self.assertIn(
            'widget.computeSize = typeof widget.computeLayoutSize === "function"',
            self.source,
        )
        self.assertIn("widget.computeLayoutSize = undefined;", self.source)
        self.assertNotIn("const DEFAULT_NODE_HEIGHT = 1225;", self.source)

    def test_hidden_widgets_do_not_leave_vue_socket_rows(self):
        self.assertIn("widget.type = original.type;", self.source)
        self.assertIn("widget.options.hidden = true;", self.source)
        self.assertNotIn('widget.type = "converted-widget";', self.source)

if __name__ == "__main__":
    unittest.main()
