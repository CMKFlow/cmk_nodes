"""FaceSwap model lookup: local and ComfyUI-registered Hyperswap/Reswapper folders.

Listing (SWAP MODEL choices) and loading (resolve_swap_model) share one search
order, so every offered name loads from the first directory that holds it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _load_model_manager(models_dir: Path, registered: dict):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(models_dir)
    folder_paths.supported_pt_extensions = {".pt", ".pth"}
    folder_paths.folder_names_and_paths = {
        name: ([str(path) for path in paths], set()) for name, paths in registered.items()
    }
    previous = sys.modules.get("folder_paths")
    sys.modules["folder_paths"] = folder_paths
    try:
        spec = importlib.util.spec_from_file_location(
            "_cmk_swap_model_paths_under_test", PACKAGE_ROOT / "models" / "model_manager.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("folder_paths", None)
        else:
            sys.modules["folder_paths"] = previous
    return module


def _model(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"onnx")
    return path


class SwapModelPaths(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.models = self.root / "ComfyUI" / "models"
        self.shared = self.root / "Shared" / "models"
        self.insightface = self.shared / "insightface"
        _model(self.insightface, "inswapper_128.onnx")

    def manager(self, **registered):
        registered.setdefault("insightface", [self.insightface])
        return _load_model_manager(self.models, registered)

    def assert_consistent(self, manager):
        """Every offered name resolves to the first search directory that holds it."""
        dirs = manager.swap_model_dirs()
        for name in manager.list_swap_models():
            expected = next(root / name for root in dirs if (root / name).exists())
            self.assertEqual(manager.resolve_swap_model(name), str(expected), name)

    def test_model_only_in_the_local_folder(self):
        local = _model(self.models / "hyperswap", "hyperswap_1b_256.onnx")
        manager = self.manager()
        self.assertIn("hyperswap_1b_256.onnx", manager.list_swap_models())
        self.assertEqual(manager.resolve_swap_model("hyperswap_1b_256.onnx"), str(local))
        self.assert_consistent(manager)

    def test_model_only_in_a_registered_folder(self):
        hyperswap = _model(self.shared / "hyperswap", "hyperswap_1b_256.onnx")
        reswapper = _model(self.shared / "reswapper", "reswapper_256.onnx")
        manager = self.manager(hyperswap=[self.shared / "hyperswap"], reswapper=[self.shared / "reswapper"])
        self.assertFalse((self.models / "hyperswap").exists())
        self.assertEqual(
            manager.list_swap_models(),
            ["hyperswap_1b_256.onnx", "inswapper_128.onnx", "reswapper_256.onnx"],
        )
        self.assertEqual(manager.resolve_swap_model("hyperswap_1b_256.onnx"), str(hyperswap))
        self.assertEqual(manager.resolve_swap_model("reswapper_256.onnx"), str(reswapper))
        self.assert_consistent(manager)

    def test_local_folder_still_searched_next_to_registered_folders(self):
        _model(self.shared / "hyperswap", "hyperswap_1b_256.onnx")
        local = _model(self.models / "hyperswap", "hyperswap_1a_256.onnx")
        manager = self.manager(hyperswap=[self.shared / "hyperswap"])
        self.assertIn("hyperswap_1a_256.onnx", manager.list_swap_models())
        self.assertEqual(manager.resolve_swap_model("hyperswap_1a_256.onnx"), str(local))
        self.assert_consistent(manager)

    def test_same_name_in_several_folders_follows_the_registration_order(self):
        first = _model(self.shared / "hyperswap", "hyperswap_1b_256.onnx")
        _model(self.root / "Other" / "hyperswap", "hyperswap_1b_256.onnx")
        _model(self.models / "hyperswap", "hyperswap_1b_256.onnx")
        manager = self.manager(hyperswap=[self.shared / "hyperswap", self.root / "Other" / "hyperswap"])
        self.assertEqual(manager.list_swap_models().count("hyperswap_1b_256.onnx"), 1)
        self.assertEqual(manager.resolve_swap_model("hyperswap_1b_256.onnx"), str(first))
        self.assert_consistent(manager)

    def test_missing_model_is_not_substituted(self):
        _model(self.shared / "hyperswap", "hyperswap_1b_256.onnx")
        manager = self.manager(hyperswap=[self.shared / "hyperswap"])
        with self.assertRaises(FileNotFoundError) as raised:
            manager.resolve_swap_model("hyperswap_1c_256.onnx")
        message = str(raised.exception)
        self.assertIn("hyperswap_1c_256.onnx", message)
        self.assertIn(str(self.shared / "hyperswap"), message)
        self.assertIn(str(self.models / "hyperswap"), message)

    def test_insightface_root_is_unchanged(self):
        manager = self.manager(insightface=[self.insightface, self.models / "insightface"])
        self.assertEqual(manager.swap_model_dirs()[0], self.insightface)
        self.assertEqual(manager.list_swap_models(), ["inswapper_128.onnx"])
        self.assertEqual(
            manager.resolve_swap_model("inswapper_128.onnx"), str(self.insightface / "inswapper_128.onnx")
        )


if __name__ == "__main__":
    unittest.main()
