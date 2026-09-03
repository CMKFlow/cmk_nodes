import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RefinerWidgetVisibilityTests(unittest.TestCase):
    def test_steps_visibility_follows_sampling_inheritance(self):
        source = (ROOT / "web/js/cmk_refiner_prepare_labels.js").read_text(encoding="utf-8")
        self.assertIn('widget?.name === "sampling_source"', source)
        self.assertIn('source?.value === "Local settings"', source)
        self.assertIn('localLabels', source)
        self.assertIn('label.parentElement.style.display = showLocal ? "" : "none"', source)
        self.assertNotIn('widget.type = "converted-widget"', source)


if __name__ == "__main__":
    unittest.main()
