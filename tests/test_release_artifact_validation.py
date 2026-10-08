import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_release_artifact",
        ROOT / "scripts" / "validate_release_artifact.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseArtifactValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = _load_validator()
        cls.manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))

    def _write_zip(self, path, omitted=()):
        omitted = set(omitted)
        required = self.validator.expected_source_files(ROOT)
        with zipfile.ZipFile(path, "w") as archive:
            for relative in sorted(required - omitted):
                archive.write(ROOT / relative, relative)

    def test_current_source_contains_all_manifest_components(self):
        reader = self.validator.ContentReader(ROOT)
        try:
            version, _files, required = self.validator.validate(reader, self.manifest)
        finally:
            reader.close()
        self.assertEqual("2.5.7", version)
        self.assertGreaterEqual(required, 9)

    def test_packed_artifact_passes_and_matches_source(self):
        with tempfile.TemporaryDirectory() as folder:
            archive_path = Path(folder) / "cmk-flow.zip"
            self._write_zip(archive_path)
            reader = self.validator.ContentReader(archive_path)
            try:
                version, _files, _required = self.validator.validate(
                    reader, self.manifest, ROOT,
                )
            finally:
                reader.close()
        self.assertEqual("2.5.7", version)

    def test_missing_release_component_fails_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            archive_path = Path(folder) / "cmk-flow.zip"
            self._write_zip(archive_path, {"nodes/image/image_resize.py"})
            reader = self.validator.ContentReader(archive_path)
            try:
                with self.assertRaisesRegex(
                    self.validator.ValidationError,
                    "nodes/image/image_resize.py",
                ):
                    self.validator.validate(reader, self.manifest, ROOT)
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
