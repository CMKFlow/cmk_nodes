from __future__ import annotations

from contextlib import ExitStack
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_faceswap_image_content_guard"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)
distribution = importlib.import_module(f"{TEST_PACKAGE}.cmk_distribution")
loader = importlib.import_module(
    f"{TEST_PACKAGE}.pipe.loaders.cmk_swap_image_loader"
)
swap_engine = importlib.import_module(
    f"{TEST_PACKAGE}.engine.swap_selected_engine"
)
content_guard = importlib.import_module(f"{TEST_PACKAGE}.engine.content_guard")


class FaceSwapImageContentGuardTests(unittest.TestCase):
    def _required(self, mode):
        with patch.object(distribution, "DISTRIBUTION_MODE", mode):
            return loader.CMKSwapImageLoaderPipe.INPUT_TYPES()["required"]

    def test_baseline_control_is_visible_default_on_and_appended(self):
        required = self._required(distribution.BASELINE)
        self.assertEqual(list(required)[-1], distribution.CONTENT_GUARD_PROCESS_KEY)
        self.assertIs(
            required[distribution.CONTENT_GUARD_PROCESS_KEY][1]["default"],
            True,
        )

    def test_release_omits_control_and_forces_on(self):
        self.assertNotIn(
            distribution.CONTENT_GUARD_PROCESS_KEY,
            self._required(distribution.RELEASE),
        )
        with patch.object(distribution, "DISTRIBUTION_MODE", distribution.RELEASE):
            self.assertTrue(distribution.effective_content_guard_enabled({
                distribution.CONTENT_GUARD_PROCESS_KEY: False,
            }))
            self.assertEqual(
                distribution.content_guard_state_token({
                    distribution.CONTENT_GUARD_PROCESS_KEY: False,
                }),
                "release:on",
            )

    def test_baseline_state_switches_and_missing_value_defaults_on(self):
        with patch.object(distribution, "DISTRIBUTION_MODE", distribution.BASELINE):
            states = [
                distribution.effective_content_guard_enabled({
                    distribution.CONTENT_GUARD_PROCESS_KEY: value,
                })
                for value in (True, False, True)
            ]
            self.assertEqual(states, [True, False, True])
            self.assertTrue(distribution.effective_content_guard_enabled({}))

    def _load_process(self, mode, requested=None):
        inputs = {
            "IMAGE PAIR": '{"target":"target.png","source":"source.png"}',
        }
        if requested is not None:
            inputs[distribution.CONTENT_GUARD_PROCESS_KEY] = requested
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        with ExitStack() as stack:
            stack.enter_context(patch.object(distribution, "DISTRIBUTION_MODE", mode))
            stack.enter_context(
                patch.object(loader, "_probe_image", return_value=(8, 8, "PNG"))
            )
            stack.enter_context(
                patch.object(
                    loader,
                    "_load_rgb_frames",
                    return_value=(image, 8, 8, (0, 0, 8, 8)),
                )
            )
            stack.enter_context(
                patch.object(loader, "parse_resolution", return_value=(8, 8))
            )
            stack.enter_context(
                patch.object(loader, "resize_image_tensor", return_value=image)
            )
            return loader.CMKSwapImageLoaderPipe().load_swap_images(**inputs)[0]

    def test_loader_process_carries_baseline_off_and_legacy_default_on(self):
        self.assertFalse(
            self._load_process(distribution.BASELINE, False)[
                distribution.CONTENT_GUARD_PROCESS_KEY
            ]
        )
        self.assertTrue(
            self._load_process(distribution.BASELINE)[
                distribution.CONTENT_GUARD_PROCESS_KEY
            ]
        )

    def test_loader_process_forces_saved_off_on_in_release(self):
        self.assertTrue(
            self._load_process(distribution.RELEASE, False)[
                distribution.CONTENT_GUARD_PROCESS_KEY
            ]
        )

    def test_cache_identity_tracks_guard_state(self):
        inputs = {"IMAGE PAIR": '{"target":"target.png","source":"source.png"}'}
        with (
            patch.object(distribution, "DISTRIBUTION_MODE", distribution.BASELINE),
            patch.object(loader, "_file_digest", side_effect=["target", "source"] * 3),
        ):
            on_first = loader.CMKSwapImageLoaderPipe.IS_CHANGED(
                **inputs, content_guard_enabled=True,
            )
            off = loader.CMKSwapImageLoaderPipe.IS_CHANGED(
                **inputs, content_guard_enabled=False,
            )
            on_again = loader.CMKSwapImageLoaderPipe.IS_CHANGED(
                **inputs, content_guard_enabled=True,
            )
        self.assertNotEqual(on_first, off)
        self.assertEqual(on_first, on_again)

    def test_frontend_routes_loader_state_to_faceswap_execution_boundary(self):
        source = (
            PACKAGE_ROOT / "web" / "js" / "cmk_swap_image_loader_pair_v1.js"
        ).read_text(encoding="utf-8")
        self.assertIn('const CONTENT_GUARD_CONSUMER = "CMKFaceSwapImagePipe";', source)
        self.assertIn("function routeContentGuardAtExecution(output)", source)
        self.assertIn("node.inputs[CONTENT_GUARD_EXECUTION_INPUT] = requested;", source)
        self.assertNotIn("delete startNode.inputs[CONTENT_GUARD_WIDGET]", source)

    def test_guard_block_aborts_before_swap_backend_can_return_an_image(self):
        guard = MagicMock(unsafe=True)
        guard.assert_swap_allowed.side_effect = content_guard.ContentGuardBlocked(
            "CG_EXPLICIT_CONTENT", "target"
        )
        backend = MagicMock()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        settings = swap_engine.SelectedSwapSettings(swap_model="test.onnx")

        with (
            patch.object(swap_engine, "get_content_guard", return_value=guard),
            patch.object(swap_engine, "get_default_swap_backend", return_value=backend),
            self.assertRaises(content_guard.ContentGuardBlocked),
        ):
            swap_engine.CMKSelectedSwapEngine().swap_faces_with_mask(
                image,
                image,
                object(),
                object(),
                settings,
                content_guard_enabled=True,
            )

        backend.swap_with_mask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
