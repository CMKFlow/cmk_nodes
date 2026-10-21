from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_persistent_cache_clear"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

persistent = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_persistent_cache")
completion = importlib.import_module(
    f"{TEST_PACKAGE}.pipe.cmk_generation_completion_cache"
)
boundaries = importlib.import_module(
    f"{TEST_PACKAGE}.pipe.cmk_module_boundary_cache"
)
refiner = importlib.import_module(
    f"{TEST_PACKAGE}.pipe.cmk_refiner_boundary_cache"
)
control = importlib.import_module(
    f"{TEST_PACKAGE}.utils.cmk_persistent_cache_control"
)


class FakePromptQueue:
    def __init__(self, *, running=False, queued=False):
        self.mutex = threading.RLock()
        self.currently_running = {1: object()} if running else {}
        self.queue = [object()] if queued else []


class PersistentCacheClearTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.package_root = self.root / "package"
        self.temp_root = self.root / "comfy-temp"
        self.package_root.mkdir()
        self.completion_patch = patch.object(
            completion, "_PACKAGE_ROOT", self.package_root
        )
        self.completion_patch.start()

    def tearDown(self):
        self.completion_patch.stop()
        boundaries.reset_module_boundary_cache_state()
        refiner.reset_refiner_boundary_cache_state()
        persistent.reset_persistent_fingerprint_memo()
        self.temporary.cleanup()

    def _completion_entry(self, key="a" * 64):
        directory = self.package_root / "cache" / "generation_completion"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{key}.safetensors").write_bytes(b"tensor")
        (directory / f"{key}.json").write_text("{}", encoding="utf-8")
        return key

    def _scope_file(self, scope, name="b" * 64 + ".pkl", data=b"cache"):
        path = self.temp_root / "cmk" / scope / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_clears_completion_and_every_registered_scope_idempotently(self):
        key = self._completion_entry()
        for scope in persistent.PERSISTENT_DISK_SCOPES:
            self._scope_file(scope)

        boundaries._DETAILER_SESSION["detailer"] = {"model": object()}
        boundaries._Z_IMAGE_SESSION["z"] = ({}, object())
        boundaries._FACE_REBUILD_SESSION["face"] = ({}, object())
        refiner._SESSION_STATE["refiner"] = {"model": object()}
        persistent._FINGERPRINT_MEMO["generic"] = ("key", "detail")
        refiner._FINGERPRINT_MEMO["refiner"] = ("key", "detail")

        result = control.clear_persistent_cmk_cache(
            FakePromptQueue(), temp_root=self.temp_root
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["areas"]["generation_completion"]["entries"], 1)
        self.assertIsNone(completion.lookup_completion(key))
        for scope in persistent.PERSISTENT_DISK_SCOPES:
            path = self.temp_root / "cmk" / scope
            self.assertTrue(path.is_dir())
            self.assertEqual(list(path.iterdir()), [])
        self.assertFalse(boundaries._DETAILER_SESSION)
        self.assertFalse(boundaries._Z_IMAGE_SESSION)
        self.assertFalse(boundaries._FACE_REBUILD_SESSION)
        self.assertFalse(refiner._SESSION_STATE)
        self.assertFalse(persistent._FINGERPRINT_MEMO)
        self.assertFalse(refiner._FINGERPRINT_MEMO)

        again = control.clear_persistent_cmk_cache(
            FakePromptQueue(), temp_root=self.temp_root
        )
        self.assertEqual(again["totals"], {"entries": 0, "files": 0, "bytes": 0})

    def test_preserves_unknown_scopes_and_everything_outside_cmk_root(self):
        known = self._scope_file("detailer_branch")
        unknown = self._scope_file("foreign_custom_node", "keep.bin")
        outside = self.temp_root / "preview.png"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_bytes(b"user data")

        control.clear_persistent_cmk_cache(
            FakePromptQueue(), temp_root=self.temp_root
        )

        self.assertFalse(known.exists())
        self.assertEqual(unknown.read_bytes(), b"cache")
        self.assertEqual(outside.read_bytes(), b"user data")

    def test_rejects_scope_symlink_before_any_deletion(self):
        self._completion_entry()
        cmk_root = self.temp_root / "cmk"
        cmk_root.mkdir(parents=True)
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "keep.bin"
        sentinel.write_bytes(b"keep")
        (cmk_root / "detailer_branch").symlink_to(outside, target_is_directory=True)

        with self.assertRaises(control.PersistentCacheClearError) as raised:
            control.clear_persistent_cmk_cache(
                FakePromptQueue(), temp_root=self.temp_root
            )

        self.assertEqual(raised.exception.phase, "path_validation")
        self.assertEqual(completion.completion_cache_inventory()["entries"], 1)
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_unlinks_nested_symlink_without_following_target(self):
        scope = self.temp_root / "cmk" / "faceprocess_branch"
        scope.mkdir(parents=True)
        outside = self.root / "outside.bin"
        outside.write_bytes(b"keep")
        (scope / "linked.bin").symlink_to(outside)

        control.clear_persistent_cmk_cache(
            FakePromptQueue(), temp_root=self.temp_root
        )

        self.assertEqual(outside.read_bytes(), b"keep")
        self.assertEqual(list(scope.iterdir()), [])

    def test_rejects_completion_scope_symlink(self):
        outside = self.root / "outside-completion"
        outside.mkdir()
        sentinel = outside / ("a" * 64 + ".json")
        sentinel.write_text("{}", encoding="utf-8")
        cache_root = self.package_root / "cache"
        cache_root.mkdir()
        (cache_root / "generation_completion").symlink_to(
            outside, target_is_directory=True
        )

        with self.assertRaises(control.PersistentCacheClearError) as raised:
            control.clear_persistent_cmk_cache(
                FakePromptQueue(), temp_root=self.temp_root
            )

        self.assertEqual(raised.exception.phase, "generation_completion")
        self.assertTrue(sentinel.exists())

    def test_unlinks_completion_file_symlink_without_following_target(self):
        outside = self.root / "outside-completion.bin"
        outside.write_bytes(b"keep")
        directory = self.package_root / "cache" / "generation_completion"
        directory.mkdir(parents=True)
        link = directory / ("a" * 64 + ".safetensors")
        link.symlink_to(outside)

        result = control.clear_persistent_cmk_cache(
            FakePromptQueue(), temp_root=self.temp_root
        )

        self.assertFalse(link.exists())
        self.assertEqual(outside.read_bytes(), b"keep")
        self.assertEqual(result["areas"]["generation_completion"]["files"], 1)

    def test_active_generation_returns_busy_without_deleting_anything(self):
        self._completion_entry()
        scope_file = self._scope_file("detailer_boundary")

        with self.assertRaises(control.PersistentCacheBusyError):
            control.clear_persistent_cmk_cache(
                FakePromptQueue(running=True), temp_root=self.temp_root
            )

        self.assertTrue(scope_file.exists())
        self.assertEqual(completion.completion_cache_inventory()["entries"], 1)

    def test_waiting_prompt_does_not_block_atomic_clear(self):
        self._completion_entry()
        result = control.clear_persistent_cmk_cache(
            FakePromptQueue(queued=True), temp_root=self.temp_root
        )
        self.assertTrue(result["cleared"])

    def test_incomplete_and_corrupt_scope_entries_are_removed(self):
        orphan = self._scope_file("faceprocess_branch", "orphan.rev", b"revision")
        corrupt = self._scope_file("refiner_boundary", "broken.json", b"{")
        control.clear_persistent_cmk_cache(
            FakePromptQueue(), temp_root=self.temp_root
        )
        self.assertFalse(orphan.exists())
        self.assertFalse(corrupt.exists())


class PersistentCacheUIContractTests(unittest.TestCase):
    def test_ui_and_backend_contracts_are_wired(self):
        javascript = (PACKAGE_ROOT / "web/js/cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        backend = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('data-cache-clear', javascript)
        self.assertIn('confirmStorageAction(', javascript)
        self.assertIn('clearButton.disabled = true', javascript)
        self.assertIn('error.status === 409', javascript)
        self.assertIn('/cmk/persistent-cache/clear', javascript)
        self.assertIn('routes.post("/cmk/persistent-cache/clear")', backend)
        self.assertIn('status=409', backend)


if __name__ == "__main__":
    unittest.main()
