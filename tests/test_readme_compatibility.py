from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "update_readme_compatibility.py"
SPEC = importlib.util.spec_from_file_location("cmk_readme_compatibility", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


class ReadmeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "pyproject.toml").write_text(
            '[project]\nname = "cmk-flow"\nversion = "9.8.7"\n', encoding="utf-8"
        )
        self.reference = {
            "schema_version": 1,
            "comfyui": "1.2.3",
            "python": "3.13.12",
            "pytorch": "2.12.1",
            "compute_backend": "Test Backend",
        }
        self._write_reference(self.reference)
        for filename in ("README.md", "README.en.md"):
            (self.root / filename).write_text(
                "prefix\n"
                + generator.START_MARKER
                + "\nstale content\n"
                + generator.END_MARKER
                + "\nsuffix\n",
                encoding="utf-8",
            )

    def tearDown(self):
        self.temporary.cleanup()

    def _write_reference(self, payload):
        (self.root / generator.REFERENCE_FILENAME).write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_updates_both_languages_from_shared_data_and_preserves_outer_content(self):
        generator.run(self.root)
        german = (self.root / "README.md").read_text(encoding="utf-8")
        english = (self.root / "README.en.md").read_text(encoding="utf-8")
        for text in (german, english):
            self.assertTrue(text.startswith("prefix\n" + generator.START_MARKER))
            self.assertTrue(text.endswith(generator.END_MARKER + "\nsuffix\n"))
            for value in ("9.8.7", "1.2.3", "3.13.12", "2.12.1", "Test Backend"):
                self.assertIn(f"`{value}`", text)
        self.assertIn("Kompatibilitätsgarantie", german)
        self.assertIn("compatibility guarantee", english)

    def test_second_run_is_idempotent(self):
        generator.run(self.root)
        before = {
            name: (self.root / name).read_bytes()
            for name in ("README.md", "README.en.md")
        }
        generator.run(self.root)
        after = {
            name: (self.root / name).read_bytes()
            for name in ("README.md", "README.en.md")
        }
        self.assertEqual(before, after)

    def test_check_reports_stale_documentation_without_writing(self):
        before = (self.root / "README.md").read_bytes()
        with self.assertRaisesRegex(
            generator.CompatibilityUpdateError, "documentation is out of date"
        ):
            generator.run(self.root, check=True)
        self.assertEqual(before, (self.root / "README.md").read_bytes())

    def test_rejects_missing_or_conflicting_reference_data(self):
        missing = dict(self.reference)
        missing.pop("pytorch")
        self._write_reference(missing)
        with self.assertRaisesRegex(generator.CompatibilityUpdateError, "missing keys: pytorch"):
            generator.load_reference(self.root / generator.REFERENCE_FILENAME)

        (self.root / generator.REFERENCE_FILENAME).write_text(
            '{"schema_version":1,"comfyui":"1.2.3","comfyui":"9.9.9",'
            '"python":"3.13.12","pytorch":"2.12.1",'
            '"compute_backend":"Test Backend"}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(generator.CompatibilityUpdateError, "duplicate key"):
            generator.load_reference(self.root / generator.REFERENCE_FILENAME)

    def test_rejects_missing_markers(self):
        (self.root / "README.md").write_text("no markers\n", encoding="utf-8")
        with self.assertRaisesRegex(generator.CompatibilityUpdateError, "marker pair"):
            generator.generated_readmes(self.root)


class CurrentReadmeCompatibilityTests(unittest.TestCase):
    def test_current_readmes_match_confirmed_reference(self):
        self.assertEqual(generator.run(PACKAGE_ROOT, check=True), 0)


if __name__ == "__main__":
    unittest.main()
