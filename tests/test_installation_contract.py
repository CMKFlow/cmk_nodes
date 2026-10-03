import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_cmk_requirements.py"


def load_installer():
    spec = importlib.util.spec_from_file_location("cmk_install_requirements", INSTALLER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InstallationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.installer = load_installer()

    def test_desktop_venv_is_preferred_over_caller_python(self):
        with tempfile.TemporaryDirectory() as directory:
            comfy = Path(directory) / "ComfyUI"
            repo = comfy / "custom_nodes" / "cmk_nodes"
            python = comfy / ".venv" / "bin" / "python3"
            repo.mkdir(parents=True)
            (comfy / "main.py").touch()
            python.parent.mkdir(parents=True)
            python.touch(mode=0o755)

            self.assertEqual(comfy.resolve(), self.installer.find_comfy_root(repo))
            self.assertEqual(
                python.absolute(),
                self.installer.find_comfy_python(comfy, Path("/usr/bin/python3")),
            )

    def test_manual_install_requires_the_canonical_repository_location(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "cmk_nodes"
            repo.mkdir()
            with self.assertRaisesRegex(RuntimeError, "ComfyUI/custom_nodes/cmk_nodes"):
                self.installer.find_comfy_root(repo)

    def test_virtual_environment_launcher_symlink_is_not_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comfy = root / "ComfyUI"
            base_python = root / "standalone-env" / "bin" / "python3"
            venv_python = comfy / ".venv" / "bin" / "python3"
            base_python.parent.mkdir(parents=True)
            base_python.touch(mode=0o755)
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(base_python)

            self.assertEqual(
                venv_python.absolute(),
                self.installer.find_comfy_python(comfy, Path("/usr/bin/python3")),
            )

    def test_requirements_cover_all_runtime_imports(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        package_for_import = {
            "cv2": "opencv-python",
            "imageio_ffmpeg": "imageio-ffmpeg",
            "insightface": "insightface",
            "nudenet": "nudenet",
            "onnxruntime": "onnxruntime",
            "segment_anything": "segment-anything",
            "ultralytics": "ultralytics",
        }
        self.assertEqual(set(package_for_import), set(self.installer.REQUIRED_IMPORTS))
        for package in package_for_import.values():
            self.assertIn(package, requirements)

    def test_bilingual_readmes_use_the_environment_aware_installer(self):
        command = "scripts/install_cmk_requirements.py"
        german = (ROOT / "README.md").read_text(encoding="utf-8")
        english = (ROOT / "README.en.md").read_text(encoding="utf-8")
        self.assertIn(command, german)
        self.assertIn(command, english)
        self.assertIn("CMK dependency check: OK", german)
        self.assertIn("CMK dependency check: OK", english)
        self.assertNotIn("cd /Pfad/zu/ComfyUI", german)
        self.assertNotIn("cd /path/to/ComfyUI", english)
        self.assertNotIn("ComfyUI Manager", german)
        self.assertNotIn("ComfyUI Manager", english)


if __name__ == "__main__":
    unittest.main()
