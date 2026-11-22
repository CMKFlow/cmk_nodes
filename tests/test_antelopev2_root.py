"""InstantID FaceAnalysis root: registered and local InsightFace folders.

CMK 2.8.0 r3 passed ComfyUI/models/insightface to FaceAnalysis regardless of
the registered (shared) InsightFace folders.  InsightFace then downloaded
antelopev2.zip into that folder, the archive extracted one level too deep and
FaceAnalysis failed with ``assert "detection" in self.models``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FILES = ("1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "glintr100.onnx", "scrfd_10g_bnkps.onnx")


def _load(path: Path, name: str, folder_paths=None):
    previous = sys.modules.get("folder_paths")
    if folder_paths is not None:
        sys.modules["folder_paths"] = folder_paths
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses resolve their module while it loads
        spec.loader.exec_module(module)
    finally:
        if folder_paths is not None:
            if previous is None:
                sys.modules.pop("folder_paths", None)
            else:
                sys.modules["folder_paths"] = previous
    return module


def _pack(root: Path, files=FILES, nested=False) -> Path:
    pack = root / "models" / "antelopev2"
    if nested:
        pack = pack / "antelopev2"
    pack.mkdir(parents=True, exist_ok=True)
    for name in files:
        (pack / name).write_bytes(b"onnx")
    return root


class Antelopev2Root(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.models = self.tmp / "ComfyUI" / "models"
        self.local = self.models / "insightface"
        self.shared = self.tmp / "Shared" / "models" / "insightface"
        self.other = self.tmp / "Other" / "insightface"

    def root(self, *registered):
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.models_dir = str(self.models)
        folder_paths.supported_pt_extensions = {".pt", ".pth"}
        folder_paths.folder_names_and_paths = (
            {"insightface": ([str(path) for path in registered], set())} if registered else {}
        )
        manager = _load(PACKAGE_ROOT / "models" / "model_manager.py", "_cmk_antelopev2_under_test", folder_paths)
        return manager.antelopev2_root()

    def test_only_in_the_registered_shared_folder(self):
        _pack(self.shared)
        self.assertEqual(self.root(self.shared), self.shared)

    def test_only_in_the_local_folder(self):
        _pack(self.local)
        self.assertEqual(self.root(), self.local)
        self.assertEqual(self.root(self.shared), self.local)

    def test_incomplete_local_folder_does_not_displace_a_valid_shared_one(self):
        _pack(self.shared)
        _pack(self.local, files=FILES[:2])
        self.assertEqual(self.root(self.local, self.shared), self.shared)
        self.assertEqual(self.root(self.shared), self.shared)

    def test_several_valid_folders_follow_the_registration_order(self):
        _pack(self.shared)
        _pack(self.other)
        _pack(self.local)
        self.assertEqual(self.root(self.shared, self.other), self.shared)
        self.assertEqual(self.root(self.other, self.shared), self.other)

    def test_no_valid_folder_names_the_expected_layout(self):
        with self.assertRaises(FileNotFoundError) as raised:
            self.root(self.shared)
        message = str(raised.exception)
        self.assertIn("scrfd_10g_bnkps.onnx", message)
        self.assertIn(str(self.shared), message)
        self.assertIn(str(self.local), message)

    def test_nested_download_alone_is_not_used(self):
        _pack(self.local, nested=True)
        with self.assertRaises(FileNotFoundError) as raised:
            self.root(self.shared)
        self.assertIn("nested", str(raised.exception))

    def test_nested_download_next_to_a_valid_shared_pack(self):
        _pack(self.local, nested=True)
        _pack(self.shared)
        self.assertEqual(self.root(self.local, self.shared), self.shared)


class InstantIDFaceAnalysisCall(unittest.TestCase):
    def test_face_analysis_uses_the_resolved_root(self):
        source = (PACKAGE_ROOT / "pipe" / "instantid" / "upstream" / "InstantID.py").read_text(encoding="utf-8")
        self.assertNotIn("INSIGHTFACE_DIR", source)
        self.assertIn("from ....models.model_manager import antelopev2_root", source)
        calls = re.findall(r"FaceAnalysis\((.*)\)", source)
        self.assertEqual(
            calls,
            ["name=\"antelopev2\", root=str(antelopev2_root()), providers=[provider + 'ExecutionProvider',]"],
        )


class ResourceCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.check = _load(PACKAGE_ROOT / "scripts" / "check_cmk_resources.py", "_cmk_resource_check_under_test")
        cls.resource = next(r for r in cls.check.RESOURCES if r.resource_id == "insightface-antelopev2")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.local = self.tmp / "ComfyUI" / "models"
        self.shared = self.tmp / "Shared" / "models"

    def test_flat_pack_is_found_in_any_root(self):
        _pack(self.shared / "insightface")
        found = self.check.locate(self.resource, [self.local, self.shared])
        self.assertEqual(found.parent, self.shared / "insightface" / "models" / "antelopev2")

    def test_nested_pack_is_missing_and_reported(self):
        _pack(self.local / "insightface", nested=True)
        self.assertIsNone(self.check.locate(self.resource, [self.local, self.shared]))
        self.assertEqual(
            self.check.nested_install(self.resource, [self.local, self.shared]),
            self.local / "insightface" / "models" / "antelopev2" / "antelopev2",
        )

    def test_incomplete_pack_is_missing(self):
        _pack(self.local / "insightface", files=FILES[:4])
        self.assertIsNone(self.check.locate(self.resource, [self.local]))
        self.assertIsNone(self.check.nested_install(self.resource, [self.local]))

    def test_archive_members_extract_flat(self):
        self.assertEqual(
            sorted(Path(name).name for name in self.resource.archive_members), sorted(FILES)
        )
        self.assertEqual(self.resource.target_path, "insightface/models/antelopev2")


if __name__ == "__main__":
    unittest.main()
