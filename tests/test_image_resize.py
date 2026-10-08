import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FakeTensor:
    def __init__(self, shape, fill=None):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.fill = fill

    def movedim(self, source, destination):
        if source == -1 and destination == 1:
            batch, height, width, channels = self.shape
            return FakeTensor((batch, channels, height, width), self.fill)
        if source == 1 and destination == -1:
            batch, channels, height, width = self.shape
            return FakeTensor((batch, height, width, channels), self.fill)
        raise AssertionError((source, destination))

    def new_full(self, shape, fill):
        return FakeTensor(shape, fill)

    def __setitem__(self, _key, _value):
        pass

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.shape, self.fill)


def _load_module():
    for name in (
        "cmk_nodes", "cmk_nodes.nodes", "cmk_nodes.nodes.image",
        "cmk_nodes.pipe", "cmk_nodes.utils",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")

    def common_upscale(image, width, height, _method, _crop):
        batch, channels, _old_height, _old_width = image.shape
        return FakeTensor((batch, channels, height, width))

    comfy_utils.common_upscale = common_upscale
    comfy.utils = comfy_utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = comfy_utils

    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    sys.modules["torch"] = torch

    log_pipe = types.ModuleType("cmk_nodes.pipe.cmk_log_pipe")
    log_pipe.cmk_add_block = lambda log, *args: dict(log, resized=True)
    sys.modules[log_pipe.__name__] = log_pipe

    cache = types.ModuleType("cmk_nodes.pipe.cmk_module_boundary_cache")
    cache._faceswap_disk_available = lambda *_args: False
    cache._load_diagnostic = lambda *_args: {}
    cache._load_image_log = lambda *_args: (None, None)
    cache._retain_latest_boundary_entry = lambda *_args: None
    cache._save_diagnostic = lambda *_args: None
    cache._save_image_log = lambda *_args: None
    sys.modules[cache.__name__] = cache

    persistent = types.ModuleType("cmk_nodes.pipe.cmk_persistent_cache")
    persistent.build_node_fingerprint = lambda *_args, **_kwargs: (None, "disabled")
    persistent.write_status = lambda *_args, **_kwargs: None
    sys.modules[persistent.__name__] = persistent

    visual = types.ModuleType("cmk_nodes.pipe.cmk_visual")
    visual.empty_visual = lambda: {"providers": []}

    def register_provider(value, **provider):
        result = dict(value)
        result["providers"] = [*value.get("providers", []), provider]
        return result

    visual.register_provider = register_provider
    sys.modules[visual.__name__] = visual

    diagnostic = types.ModuleType("cmk_nodes.utils.cmk_diagnostic")
    diagnostic.make_diagnostic_payload = lambda **kwargs: kwargs
    sys.modules[diagnostic.__name__] = diagnostic

    timing = types.ModuleType("cmk_nodes.utils.cmk_timing")
    timing.cmk_timed_call = lambda _label: (lambda function: function)
    sys.modules[timing.__name__] = timing

    spec = importlib.util.spec_from_file_location(
        "cmk_nodes.nodes.image.image_resize",
        ROOT / "nodes" / "image" / "image_resize.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ImageResizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()
        cls.node = cls.module.CMKImageResize()

    def test_public_contract_and_all_three_methods_are_exposed(self):
        node_type = self.module.CMKImageResize
        self.assertEqual(
            ("MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"),
            node_type.RETURN_NAMES,
        )
        methods = node_type.INPUT_TYPES()["required"]["scale_method"][0]
        self.assertEqual(tuple(methods), self.module.METHODS)

    def test_max_dimension_preserves_aspect_ratio(self):
        result = self.node._scale_to_max_dimension(FakeTensor((1, 400, 800, 3)), 1600, "lanczos")
        self.assertEqual((1, 800, 1600, 3), result.shape)

    def test_megapixel_target_preserves_aspect_ratio(self):
        result = self.node._scale_to_megapixels(FakeTensor((1, 400, 800, 3)), 1.0, "area")
        self.assertAlmostEqual(2.0, result.shape[2] / result.shape[1], places=2)
        self.assertAlmostEqual(1024 * 1024, result.shape[1] * result.shape[2], delta=1800)

    def test_resize_and_pad_uses_exact_target_and_requested_color(self):
        result = self.node._resize_and_pad(FakeTensor((1, 400, 800, 3)), 512, 512, "bilinear", "white")
        self.assertEqual((1, 512, 512, 3), result.shape)
        self.assertEqual(1.0, result.fill)

    def test_disabled_node_is_true_transport_bypass(self):
        model = {"model": object()}
        process = {"family": "sdxl"}
        image = FakeTensor((1, 64, 96, 3))
        log = {"blocks": []}
        visual = {"providers": []}
        result = self.node.resize(
            process, image, log,
            **{"MODEL (opt)": model, "VISUAL": visual, "SCALE ENABLE": False},
        )
        self.assertIs(model, result[0])
        self.assertIs(process, result[1])
        self.assertIs(image, result[2])
        self.assertIs(log, result[3])
        self.assertIs(visual, result[4])
        self.assertTrue(result[5]["metadata"]["bypassed"])

    def test_execution_updates_log_visual_and_diagnostic(self):
        image = FakeTensor((1, 400, 800, 3))
        result = self.node.resize(
            {"family": "sdxl"}, image, {"blocks": []},
            **{
                "MODEL (opt)": {"model": object()},
                "VISUAL": {"providers": []},
                "SCALE ENABLE": True,
                "scale_method": self.module.METHOD_MAX_DIMENSION,
                "interpolation": "lanczos",
                "largest_size": 1600,
                "unique_id": "resize-test",
            },
        )
        self.assertEqual((1, 800, 1600, 3), result[2].shape)
        self.assertTrue(result[3]["resized"])
        self.assertEqual("CMKImageResize", result[4]["providers"][0]["module_type"])
        self.assertEqual("Scale to Max Dimension", result[5]["metadata"]["method"])

    def test_boundary_cache_hit_restores_image_log_and_diagnostic(self):
        module = self.module
        original = {
            "build": module.build_node_fingerprint,
            "available": module._faceswap_disk_available,
            "load_image_log": module._load_image_log,
            "load_diagnostic": module._load_diagnostic,
        }
        cached = FakeTensor((1, 300, 600, 3))
        try:
            module.build_node_fingerprint = lambda *_args, **_kwargs: ("cache-key", "unit")
            module._faceswap_disk_available = lambda *_args: True
            module._load_image_log = lambda *_args: (cached, {"cached": True})
            module._load_diagnostic = lambda *_args: {"metadata": {"method": "cached"}}
            result = self.node.resize(
                {"family": "sdxl"}, FakeTensor((1, 400, 800, 3)), {"blocks": []},
                **{
                    "SCALE ENABLE": True,
                    "scale_method": module.METHOD_MAX_DIMENSION,
                    "interpolation": "lanczos",
                    "largest_size": 1600,
                    "unique_id": "resize-cache-hit",
                },
            )
        finally:
            module.build_node_fingerprint = original["build"]
            module._faceswap_disk_available = original["available"]
            module._load_image_log = original["load_image_log"]
            module._load_diagnostic = original["load_diagnostic"]
        self.assertIs(cached, result[2])
        self.assertEqual({"cached": True}, result[3])
        self.assertTrue(result[5]["metadata"]["cache_hit"])

    def test_boundary_cache_miss_stores_materialized_result(self):
        module = self.module
        original = {
            "build": module.build_node_fingerprint,
            "available": module._faceswap_disk_available,
            "save_image_log": module._save_image_log,
            "save_diagnostic": module._save_diagnostic,
        }
        stored = []
        try:
            module.build_node_fingerprint = lambda *_args, **_kwargs: ("cache-key", "unit")
            module._faceswap_disk_available = lambda *_args: False
            module._save_image_log = lambda *args: stored.append(("image", args))
            module._save_diagnostic = lambda *args: stored.append(("diagnostic", args))
            result = self.node.resize(
                {"family": "sdxl"}, FakeTensor((1, 400, 800, 3)), {"blocks": []},
                **{
                    "SCALE ENABLE": True,
                    "scale_method": module.METHOD_MEGAPIXELS,
                    "interpolation": "area",
                    "megapixels": 1.0,
                    "unique_id": "resize-cache-miss",
                },
            )
        finally:
            module.build_node_fingerprint = original["build"]
            module._faceswap_disk_available = original["available"]
            module._save_image_log = original["save_image_log"]
            module._save_diagnostic = original["save_diagnostic"]
        self.assertEqual(["image", "diagnostic"], [kind for kind, _args in stored])
        self.assertFalse(result[5]["metadata"]["cache_hit"])

    def test_ui_keeps_stable_serialization_and_conditional_widgets(self):
        source = (ROOT / "web/js/cmk_image_resize_ui.js").read_text(encoding="utf-8")
        self.assertIn("installStableSerialization", source)
        self.assertIn("data.widgets_values = state.order", source)
        for widget in ("largest_size", "megapixels", "target_width", "target_height", "padding_color"):
            self.assertIn(widget, source)


if __name__ == "__main__":
    unittest.main()
