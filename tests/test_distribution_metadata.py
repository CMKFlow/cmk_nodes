from pathlib import Path
import unittest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


class DistributionMetadataTests(unittest.TestCase):
    def test_registry_metadata_matches_cmk_release(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual("cmk-flow", metadata["project"]["name"])
        self.assertEqual("2.5.6", metadata["project"]["version"])

    def test_registry_package_forces_unicode_workflow_directories(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(
            ["subgraphs", "workflows", "web/assets"],
            metadata["tool"]["comfy"]["includes"],
        )
        self.assertEqual(
            "Modular workflow system for SDXL, Z-Image Turbo and HYBRID "
            "generation with integrated identity, ControlNet and post-processing tools.",
            metadata["project"]["description"],
        )
        self.assertEqual(
            "https://github.com/CMKFlow/cmk_nodes",
            metadata["project"]["urls"]["Repository"],
        )
        self.assertEqual("cmkflow", metadata["tool"]["comfy"]["PublisherId"])
        self.assertEqual("CMK Flow", metadata["tool"]["comfy"]["DisplayName"])

    def test_registry_dependencies_are_sourced_from_requirements(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(["dependencies"], metadata["project"]["dynamic"])
        self.assertEqual([], metadata["tool"]["setuptools"]["packages"])
        self.assertEqual(
            ["requirements.txt"],
            metadata["tool"]["setuptools"]["dynamic"]["dependencies"]["file"],
        )


if __name__ == "__main__":
    unittest.main()
