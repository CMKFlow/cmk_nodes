from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE_ROOT / "engine" / "fooocus_inpaint.py"

comfy = types.ModuleType("comfy")
comfy.__path__ = []
comfy_lora = types.ModuleType("comfy.lora")
comfy_lora.calculate_weight = lambda *args, **kwargs: args[1]
comfy_lora.model_lora_keys_unet = lambda *_args: {}
comfy_patcher_extension = types.ModuleType("comfy.patcher_extension")
comfy_patcher_extension.WrappersMP = types.SimpleNamespace(CALC_COND_BATCH="calc_cond_batch")
comfy_utils = types.ModuleType("comfy.utils")
comfy_model_base = types.ModuleType("comfy.model_base")
comfy_model_base.BaseModel = object
comfy_model_management = types.ModuleType("comfy.model_management")
comfy_model_management.cast_to_device = lambda value, device, dtype: value.to(device=device, dtype=dtype)
comfy_model_patcher = types.ModuleType("comfy.model_patcher")
comfy_model_patcher.ModelPatcher = type("ModelPatcher", (), {})
comfy.lora = comfy_lora
comfy.patcher_extension = comfy_patcher_extension
comfy.utils = comfy_utils

folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = ""
node_helpers = types.ModuleType("node_helpers")

stub_modules = (
    comfy,
    comfy_lora,
    comfy_patcher_extension,
    comfy_utils,
    comfy_model_base,
    comfy_model_management,
    comfy_model_patcher,
    folder_paths,
    node_helpers,
)
previous_modules = {module.__name__: sys.modules.get(module.__name__) for module in stub_modules}
for module in stub_modules:
    sys.modules[module.__name__] = module

spec = importlib.util.spec_from_file_location("_cmk_fooocus_inpaint_test", MODULE_PATH)
fooocus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fooocus)
for name, previous in previous_modules.items():
    if previous is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


def feature_grid(batch=1):
    base = torch.arange(batch * 24, dtype=torch.float32).reshape(batch, 1, 4, 6)
    return base


def options(uuids, areas):
    return {
        "block": ("input", 0),
        "uuids": uuids,
        fooocus._FOOOCUS_AREAS: areas,
    }


class FooocusRegionalInpaintTests(unittest.TestCase):
    def patch(self, feature=None):
        patch = fooocus.InpaintBlockPatch()
        patch.inpaint_head_feature = feature_grid() if feature is None else feature
        return patch

    def test_global_conditioning_uses_full_feature(self):
        feature = feature_grid()
        result = self.patch(feature)(torch.zeros_like(feature), options(["global"], {"global": None}))
        torch.testing.assert_close(result, feature)

    def test_single_regional_area_uses_spatial_crop(self):
        feature = feature_grid()
        result = self.patch(feature)(torch.zeros((1, 1, 2, 3)), options(["region"], {"region": (2, 3, 1, 2)}))
        torch.testing.assert_close(result, feature[:, :, 1:3, 2:5])

    def test_equal_regions_at_different_positions_follow_uuid_order(self):
        feature = feature_grid()
        areas = {"left": (2, 2, 0, 0), "right": (2, 2, 2, 4)}
        result = self.patch(feature)(torch.zeros((2, 1, 2, 2)), options(["right", "left"], areas))
        expected = torch.cat((feature[:, :, 2:4, 4:6], feature[:, :, 0:2, 0:2]))
        torch.testing.assert_close(result, expected)

    def test_different_region_sizes_do_not_share_cached_feature(self):
        feature = feature_grid()
        patch = self.patch(feature)
        first = patch(torch.zeros((1, 1, 2, 2)), options(["first"], {"first": (2, 2, 0, 0)}))
        second = patch(torch.zeros((1, 1, 3, 4)), options(["second"], {"second": (3, 4, 1, 2)}))
        torch.testing.assert_close(first, feature[:, :, 0:2, 0:2])
        torch.testing.assert_close(second, feature[:, :, 1:4, 2:6])

    def test_overlapping_regions_keep_their_own_crops(self):
        feature = feature_grid()
        areas = {"a": (3, 3, 0, 1), "b": (3, 3, 1, 2)}
        result = self.patch(feature)(torch.zeros((2, 1, 3, 3)), options(["a", "b"], areas))
        expected = torch.cat((feature[:, :, 0:3, 1:4], feature[:, :, 1:4, 2:5]))
        torch.testing.assert_close(result, expected)

    def test_image_batch_is_preserved_inside_conditioning_order(self):
        feature = feature_grid(batch=2)
        areas = {"bottom": (2, 3, 2, 3), "top": (2, 3, 0, 0)}
        result = self.patch(feature)(torch.zeros((4, 1, 2, 3)), options(["bottom", "top"], areas))
        expected = torch.cat((feature[:, :, 2:4, 3:6], feature[:, :, 0:2, 0:3]))
        torch.testing.assert_close(result, expected)

    def test_legacy_full_feature_call_without_area_metadata_is_unchanged(self):
        feature = feature_grid(batch=2)
        result = self.patch(feature)(torch.zeros((4, 1, 4, 6)), {"block": ("input", 0)})
        torch.testing.assert_close(result, feature.repeat(2, 1, 1, 1))

    def test_missing_area_metadata_for_regional_input_has_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "regional model input without conditioning area information"):
            self.patch()(torch.zeros((1, 1, 2, 3)), {"block": ("input", 0), "uuids": ["region"]})

    def test_calc_cond_batch_passes_clamped_areas_without_mutating_options(self):
        original_options = {"transformer_options": {"existing": True}}
        conds = [[
            {"uuid": "global"},
            {"uuid": "edge", "area": (3, 4, 2, 4)},
        ]]
        captured = {}

        def executor(model, received_conds, x_in, timestep, model_options):
            captured.update(model_options)
            return "result"

        result = fooocus._fooocus_calc_cond_batch(
            executor, "model", conds, torch.zeros((1, 4, 4, 6)), "timestep", original_options
        )
        self.assertEqual(result, "result")
        self.assertEqual(original_options, {"transformer_options": {"existing": True}})
        self.assertEqual(captured["transformer_options"][fooocus._FOOOCUS_AREAS], {
            "global": None,
            "edge": (2, 2, 2, 4),
        })

    def test_apply_registers_calc_cond_batch_wrapper(self):
        class Base:
            def process_latent_in(self, samples):
                return samples

            def state_dict(self):
                return {}

        class Model:
            def __init__(self):
                self.model = Base()
                self.block_patch = None
                self.wrapper = None

            def clone(self):
                clone = Model()
                clone.model = self.model
                return clone

            def set_model_input_block_patch(self, patch):
                self.block_patch = patch

            def add_wrapper_with_key(self, wrapper_type, key, wrapper):
                self.wrapper = (wrapper_type, key, wrapper)

            def add_patches(self, patches, strength):
                return set(patches)

        class Head:
            def to(self, **_kwargs):
                return self

            def __call__(self, feed):
                return torch.ones((feed.shape[0], 320, feed.shape[-2], feed.shape[-1]))

        patched = fooocus.apply_fooocus_inpaint(
            Model(),
            (Head(), {}),
            {"samples": torch.zeros((1, 4, 2, 3)), "noise_mask": torch.zeros((1, 1, 16, 24))},
        )
        self.assertIsInstance(patched.block_patch, fooocus.InpaintBlockPatch)
        self.assertEqual(patched.wrapper, ("calc_cond_batch", "cmk_fooocus_areas", fooocus._fooocus_calc_cond_batch))

    def test_unresolved_area_is_rejected(self):
        conds = [[{"uuid": "region", "area": ("percentage", 0.5, 0.5, 0.0, 0.0)}]]
        with self.assertRaisesRegex(RuntimeError, "resolved integer conditioning area"):
            fooocus._fooocus_areas_by_uuid(conds, torch.zeros((1, 4, 4, 6)))


if __name__ == "__main__":
    unittest.main()
