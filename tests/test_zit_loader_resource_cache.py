import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    calls = []
    for name in ("cmk_nodes", "cmk_nodes.pipe", "cmk_nodes.pipe.loaders"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda _kind: []
    sys.modules["folder_paths"] = folder_paths

    graph_utils = types.ModuleType("comfy_execution.graph_utils")
    graph_utils.ExecutionBlocker = lambda message: ("blocked", message)
    comfy_execution = types.ModuleType("comfy_execution")
    comfy_execution.graph_utils = graph_utils
    sys.modules["comfy_execution"] = comfy_execution
    sys.modules["comfy_execution.graph_utils"] = graph_utils

    model_management = types.ModuleType("comfy.model_management")
    model_management.unload_all_models = lambda: None
    model_management.cleanup_models_gc = lambda: None
    model_management.soft_empty_cache = lambda force=False: None
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = model_management
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management

    log_pipe = types.ModuleType("cmk_nodes.pipe.cmk_log_pipe")
    log_pipe.cmk_add_block = lambda incoming, _title, _order, lines, _active: {
        "incoming": incoming,
        "lines": lines,
    }
    sys.modules[log_pipe.__name__] = log_pipe

    sampler_prepare = types.ModuleType("cmk_nodes.pipe.cmk_sampler_prepare")

    def call_node(names, **_kwargs):
        calls.append(names[0])
        return {"resource": names[0], "instance": len(calls)}

    sampler_prepare._call_node_kwargs = call_node
    sampler_prepare._unwrap_node_output = lambda value: value
    sys.modules[sampler_prepare.__name__] = sampler_prepare

    spec = importlib.util.spec_from_file_location(
        "cmk_nodes.pipe.loaders.z_image_turbo_loader",
        ROOT / "pipe" / "loaders" / "z_image_turbo_loader.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, calls


class ZITLoaderResourceCacheTests(unittest.TestCase):
    def test_prompt_or_seed_change_reuses_identical_heavy_resources(self):
        module, calls = _load_module()
        loader = module.CMKZImageTurboLoaderPipe()
        common = {
            "diffusion_model": "unet.safetensors",
            "text_encoder": "text.safetensors",
            "vae": "vae.safetensors",
        }
        first_model, _ = loader.load(
            {"model_family": "z_image_turbo", "seed": 1, "prompt_pos": "one"},
            **common,
        )
        second_model, _ = loader.load(
            {"model_family": "z_image_turbo", "seed": 2, "prompt_pos": "two"},
            **common,
        )

        self.assertEqual(["UNETLoader", "CLIPLoader", "VAELoader"], calls)
        self.assertIs(first_model["model"], second_model["model"])
        self.assertIs(first_model["clip"], second_model["clip"])
        self.assertIs(first_model["vae"], second_model["vae"])
        self.assertEqual("MISS · LOADED", first_model["resource_cache_status"])
        self.assertEqual("HIT · REUSED", second_model["resource_cache_status"])

    def test_explicit_lifecycle_drops_old_resource_set_before_switch(self):
        module, calls = _load_module()
        loader = module.CMKZImageTurboLoaderPipe()
        process = {
            "model_family": "z_image_turbo",
            "unload_models_after_use": True,
        }

        loader.load(
            process,
            diffusion_model="old-unet.safetensors",
            text_encoder="old-text.safetensors",
            vae="old-vae.safetensors",
        )
        loader.load(
            process,
            diffusion_model="new-unet.safetensors",
            text_encoder="new-text.safetensors",
            vae="new-vae.safetensors",
        )

        self.assertEqual(6, len(calls))
        self.assertEqual(
            [(
                "new-unet.safetensors",
                "new-text.safetensors",
                "new-vae.safetensors",
                "default",
                "default",
            )],
            list(module._ZIT_RESOURCE_CACHE),
        )

    def test_evicted_text_encoder_is_rematerialized_without_reloading_unet_or_vae(self):
        module, calls = _load_module()
        loader = module.CMKZImageTurboLoaderPipe()
        common = {
            "diffusion_model": "unet.safetensors",
            "text_encoder": "text.safetensors",
            "vae": "vae.safetensors",
        }
        first_model, _ = loader.load(
            {"model_family": "z_image_turbo", "prompt_pos": "one"},
            **common,
        )
        unet = first_model["model"]
        vae = first_model["vae"]

        self.assertEqual(
            "EVICTED AFTER ENCODE",
            module.evict_zit_text_encoder(first_model),
        )
        self.assertIsNone(first_model["clip"])

        clip, status = module.ensure_zit_text_encoder(first_model)
        self.assertEqual("REMATERIALIZED", status)
        self.assertIsNotNone(clip)
        self.assertIs(first_model["model"], unet)
        self.assertIs(first_model["vae"], vae)
        self.assertEqual(
            ["UNETLoader", "CLIPLoader", "VAELoader", "CLIPLoader"],
            calls,
        )

    def test_loader_repairs_an_evicted_cached_encoder(self):
        module, calls = _load_module()
        loader = module.CMKZImageTurboLoaderPipe()
        common = {
            "diffusion_model": "unet.safetensors",
            "text_encoder": "text.safetensors",
            "vae": "vae.safetensors",
        }
        first_model, _ = loader.load(
            {"model_family": "z_image_turbo", "prompt_pos": "one"},
            **common,
        )
        module.evict_zit_text_encoder(first_model)
        second_model, _ = loader.load(
            {"model_family": "z_image_turbo", "prompt_pos": "two"},
            **common,
        )

        self.assertIsNotNone(second_model["clip"])
        self.assertEqual(
            "HIT · TEXT ENCODER REMATERIALIZED",
            second_model["resource_cache_status"],
        )
        self.assertEqual(
            ["UNETLoader", "CLIPLoader", "VAELoader", "CLIPLoader"],
            calls,
        )

    def test_interrupted_lifecycle_invalidates_resources_and_forces_full_reload(self):
        module, calls = _load_module()
        loader = module.CMKZImageTurboLoaderPipe()
        common = {
            "diffusion_model": "unet.safetensors",
            "text_encoder": "text.safetensors",
            "vae": "vae.safetensors",
        }
        first_model, _ = loader.load(
            {
                "model_family": "z_image_turbo",
                "unload_models_after_use": True,
            },
            **common,
        )

        invalidated = module.invalidate_zit_resource_cache("unit-test interrupt")

        self.assertEqual(1, invalidated)
        self.assertEqual({}, module._ZIT_RESOURCE_CACHE)
        second_model, _ = loader.load(
            {
                "model_family": "z_image_turbo",
                "unload_models_after_use": True,
            },
            **common,
        )
        self.assertIsNot(first_model["model"], second_model["model"])
        self.assertIsNot(first_model["vae"], second_model["vae"])
        self.assertEqual(
            [
                "UNETLoader", "CLIPLoader", "VAELoader",
                "UNETLoader", "CLIPLoader", "VAELoader",
            ],
            calls,
        )


if __name__ == "__main__":
    unittest.main()
