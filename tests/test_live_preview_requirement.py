import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LivePreviewRequirementTests(unittest.TestCase):
    def test_notice_checks_setting_without_changing_it(self):
        source = (
            ROOT / "web" / "js" / "cmk_live_preview_requirement.js"
        ).read_text(encoding="utf-8")
        self.assertIn('const SETTING_ID = "Comfy.Execution.PreviewMethod"', source)
        self.assertIn("api.fetchApi", source)
        self.assertNotIn("storeSetting", source)
        self.assertNotIn("setSetting", source)

    def test_notice_accepts_supported_preview_methods(self):
        source = (
            ROOT / "web" / "js" / "cmk_live_preview_requirement.js"
        ).read_text(encoding="utf-8")
        for method in ("auto", "latent2rgb", "taesd"):
            self.assertIn(f'method === "{method}"', source)


if __name__ == "__main__":
    unittest.main()
