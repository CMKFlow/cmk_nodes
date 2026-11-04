from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE_ROOT / "pipe" / "instantid" / "upstream" / "CrossAttentionPatch.py"
PACKAGE_NAME = "_cmk_instantid_area_test"

package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(MODULE_PATH.parent)]
utils = types.ModuleType(f"{PACKAGE_NAME}.utils")


def tensor_to_size(tensor, size):
    if tensor.shape[0] == size:
        return tensor
    repeats = (size + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))[:size]


utils.tensor_to_size = tensor_to_size
comfy = types.ModuleType("comfy")
comfy.__path__ = []
comfy_ldm = types.ModuleType("comfy.ldm")
comfy_ldm.__path__ = []
comfy_modules = types.ModuleType("comfy.ldm.modules")
comfy_modules.__path__ = []
comfy_attention = types.ModuleType("comfy.ldm.modules.attention")
comfy_attention.optimized_attention = lambda q, _k, _v, _heads: torch.zeros_like(q)

stub_modules = (package, utils, comfy, comfy_ldm, comfy_modules, comfy_attention)
previous_modules = {module.__name__: sys.modules.get(module.__name__) for module in stub_modules}
for module in stub_modules:
    sys.modules[module.__name__] = module

spec = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.CrossAttentionPatch", MODULE_PATH)
area_mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(area_mapping)

for name, previous in previous_modules.items():
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


def spatial_grid():
    return torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)


class FakeControl:
    def __init__(self, hint):
        self._cmk_instantid_full_hint = hint
        self.cond_hint_original = hint
        self.cond_hint = torch.ones(1)


class IdentityProjection:
    def __call__(self, tensor):
        return tensor


class InstantIDAreaMappingTests(unittest.TestCase):
    def test_global_tensor_is_unchanged(self):
        tensor = spatial_grid()
        self.assertIs(area_mapping.crop_spatial_tensor(tensor, None, (4, 6)), tensor)

    def test_area_crops_mask_or_keypoints_before_resize(self):
        tensor = spatial_grid()
        result = area_mapping.crop_spatial_tensor(tensor, (2, 3, 1, 2), (4, 6))
        torch.testing.assert_close(result, tensor[..., 1:3, 2:5])

    def test_equal_regions_at_different_positions_use_different_crops(self):
        tensor = spatial_grid()
        left = area_mapping.crop_spatial_tensor(tensor, (2, 2, 0, 0), (4, 6))
        right = area_mapping.crop_spatial_tensor(tensor, (2, 2, 2, 4), (4, 6))
        torch.testing.assert_close(left, tensor[..., 0:2, 0:2])
        torch.testing.assert_close(right, tensor[..., 2:4, 4:6])
        self.assertFalse(torch.equal(left, right))

    def test_wrapper_maps_multiple_controls_by_uuid_and_position(self):
        hint = spatial_grid()
        global_control = FakeControl(hint)
        left_control = FakeControl(hint)
        right_control = FakeControl(hint)
        conds = [[
            {"uuid": "global", "control": global_control},
            {"uuid": "left", "area": (2, 2, 0, 0), "control": left_control},
            {"uuid": "right", "area": (2, 2, 2, 4), "control": right_control},
        ]]
        original_options = {"transformer_options": {"existing": True}}
        captured = {}

        def executor(model, received_conds, x_in, timestep, model_options):
            captured.update(model_options)
            return "result"

        result = area_mapping.instantid_calc_cond_batch(
            executor,
            "model",
            conds,
            torch.zeros((1, 4, 4, 6)),
            "timestep",
            original_options,
        )

        self.assertEqual(result, "result")
        self.assertEqual(original_options, {"transformer_options": {"existing": True}})
        self.assertIs(global_control.cond_hint_original, hint)
        torch.testing.assert_close(left_control.cond_hint_original, hint[..., 0:2, 0:2])
        torch.testing.assert_close(right_control.cond_hint_original, hint[..., 2:4, 4:6])
        self.assertIsNone(left_control.cond_hint)
        self.assertEqual(
            captured["transformer_options"][area_mapping.CMK_INSTANTID_AREAS],
            {"global": None, "left": (2, 2, 0, 0), "right": (2, 2, 2, 4)},
        )

    def test_control_cache_is_keyed_by_full_area_not_only_dimensions(self):
        control = FakeControl(spatial_grid())
        conds = [[
            {"uuid": "left", "area": (2, 2, 0, 0), "control": control},
            {"uuid": "right", "area": (2, 2, 2, 4), "control": control},
        ]]
        with self.assertRaisesRegex(RuntimeError, "shared across different conditioning areas"):
            area_mapping.instantid_calc_cond_batch(
                lambda *_args: None,
                "model",
                conds,
                torch.zeros((1, 4, 4, 6)),
                "timestep",
                {},
            )

    def test_attention_mask_uses_uuid_specific_area_crop(self):
        original_attention = area_mapping.optimized_attention
        area_mapping.optimized_attention = lambda q, _k, _v, _heads: torch.ones_like(q)
        try:
            adapter = types.SimpleNamespace(
                ip_layers=types.SimpleNamespace(to_kvs={
                    "1_to_k_ip": IdentityProjection(),
                    "1_to_v_ip": IdentityProjection(),
                })
            )
            full_mask = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
            output = area_mapping.instantid_attention(
                torch.zeros((2, 4, 1)),
                torch.zeros((2, 4, 1)),
                torch.zeros((2, 4, 1)),
                torch.zeros((2, 4, 1)),
                {
                    "cond_or_uncond": [0, 0],
                    "block": ("input", 4),
                    "transformer_index": 0,
                    "original_shape": [2, 4, 2, 2],
                    "n_heads": 1,
                    "uuids": ["left", "right"],
                    area_mapping.CMK_INSTANTID_AREAS: {
                        "left": (2, 2, 0, 0),
                        "right": (2, 2, 2, 4),
                    },
                    area_mapping.CMK_INSTANTID_FULL_DIMS: (4, 6),
                },
                module_key="1",
                ipadapter=adapter,
                cond=torch.ones((1, 4, 1)),
                uncond=torch.zeros((1, 4, 1)),
                mask=full_mask,
            )
            expected = torch.cat((
                full_mask[..., 0:2, 0:2].reshape(1, 4, 1),
                full_mask[..., 2:4, 4:6].reshape(1, 4, 1),
            ))
            torch.testing.assert_close(output, expected)
        finally:
            area_mapping.optimized_attention = original_attention

    def test_handshake_fallback_is_black_at_target_resolution(self):
        source = torch.ones((1, 64, 64, 3))
        target = torch.ones((1, 96, 160, 3))
        fallback = area_mapping.neutral_keypoint_hint(source, target)
        self.assertEqual(fallback.shape, target.shape)
        self.assertEqual(torch.count_nonzero(fallback).item(), 0)

    def test_zero_pass_black_hint_stays_black(self):
        source = torch.ones((1, 64, 64, 3))
        black = torch.zeros((1, 96, 160, 3))
        fallback = area_mapping.neutral_keypoint_hint(source, black)
        torch.testing.assert_close(fallback, black)


if __name__ == "__main__":
    unittest.main()
