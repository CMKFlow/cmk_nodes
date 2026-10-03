import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_sampler_module():
    for name in (
        "cmk_nodes", "cmk_nodes.pipe", "cmk_nodes.utils",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    common = types.ModuleType("cmk_nodes.cmk_common")
    common.SAMPLERS = []
    common.SCHEDULERS = []
    sys.modules[common.__name__] = common

    final_preview = types.ModuleType("cmk_nodes.pipe.cmk_final_preview")
    final_preview.send_final_preview = lambda *args, **kwargs: None
    sys.modules[final_preview.__name__] = final_preview

    visual = types.ModuleType("cmk_nodes.pipe.cmk_visual")
    visual.empty_visual = lambda: {}
    visual.register_provider = lambda value, **kwargs: value
    sys.modules[visual.__name__] = visual

    timing = types.ModuleType("cmk_nodes.utils.cmk_timing")
    timing.cmk_timed = lambda *args, **kwargs: None
    sys.modules[timing.__name__] = timing

    warnings = types.ModuleType("cmk_nodes.utils.cmk_sampling_warnings")
    warnings.ignore_torchsde_boundary_rounding = lambda: None
    sys.modules[warnings.__name__] = warnings

    unloaded = []
    management = types.ModuleType("comfy.model_management")
    management.unload_model_and_clones = (
        lambda model, unload_additional_models=False: unloaded.append(model)
    )
    management.soft_empty_cache = lambda force=False: None
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = management
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = management

    spec = importlib.util.spec_from_file_location(
        "cmk_nodes.pipe.cmk_pipe_sampler", ROOT / "pipe" / "cmk_pipe_sampler.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, unloaded


class FakeControl:
    def __init__(self, model):
        self.model = model
        self.cleanup_calls = 0

    def get_models(self):
        return [self.model]

    def cleanup(self):
        self.cleanup_calls += 1


class SDXLControlNetLifecycleTests(unittest.TestCase):
    def test_terminal_sampler_releases_pipe_and_conditioning_references(self):
        module, unloaded = _load_sampler_module()
        model = object()
        control = FakeControl(model)
        metadata = {"control": control, "control_apply_to_uncond": True}
        pipe = {
            "boolean_controlnet_enable": True,
            "control_net": control,
            "conditioning_pos": [[object(), metadata]],
        }

        status = module._unload_completed_controlnet(pipe)

        self.assertNotIn("control_net", pipe)
        self.assertNotIn("control", metadata)
        self.assertNotIn("control_apply_to_uncond", metadata)
        self.assertEqual([model], unloaded)
        self.assertGreaterEqual(control.cleanup_calls, 1)
        self.assertIn("1 conditioning refs released", status)

    def test_module_05_defers_sdxl_model_load_to_module_10(self):
        prepare = (ROOT / "pipe" / "controlnet" / "cmk_controlnet_prepare.py").read_text()
        sampler = (ROOT / "pipe" / "cmk_sampler_prepare.py").read_text()
        self.assertIn('"DEFER MODEL LOAD": True', prepare)
        self.assertIn('new_pipe["controlnet_model_name"]', prepare)
        self.assertIn("def _ensure_sdxl_controlnet_loaded", sampler)
        self.assertIn('(\"ControlNetLoader\",)', sampler)


if __name__ == "__main__":
    unittest.main()
