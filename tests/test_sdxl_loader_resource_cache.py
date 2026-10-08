import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    calls = []
    for name in (
        "cmk_nodes", "cmk_nodes.pipe", "cmk_nodes.pipe.loaders",
        "cmk_nodes.loader",
    ):
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

    resource_loader = types.ModuleType("cmk_nodes.loader.checkpoint_vae_loader")

    def load_resources(ckpt_name, vae_name, checkpoint_vae):
        calls.append((ckpt_name, vae_name, checkpoint_vae))
        serial = len(calls)
        return (
            {"model": serial}, {"clip": serial}, {"vae": serial},
            {"ckpt_name": ckpt_name, "vae_name": vae_name},
        )

    resource_loader.load_checkpoint_vae_resources = load_resources
    sys.modules[resource_loader.__name__] = resource_loader

    spec = importlib.util.spec_from_file_location(
        "cmk_nodes.pipe.loaders.checkpoint_vae_loader",
        ROOT / "pipe" / "loaders" / "checkpoint_vae_loader.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, calls


class SDXLLoaderResourceCacheTests(unittest.TestCase):
    def test_postprocess_spec_materializes_only_at_first_consumer(self):
        module, calls = _load_module()
        spec = module.make_postprocess_model_spec(
            "postprocess.safetensors", "vae.safetensors", False
        )
        self.assertEqual([], calls)
        self.assertFalse(spec["materialized"])
        self.assertTrue(module.is_postprocess_model_spec(spec))

        first = module.resolve_postprocess_model(spec)
        second = module.resolve_postprocess_model(spec)
        self.assertEqual(1, len(calls))
        self.assertTrue(first["materialized"])
        self.assertEqual("MISS · LOADED", first["resource_cache_status"])
        self.assertEqual("HIT · REUSED", second["resource_cache_status"])
        self.assertFalse(module.is_postprocess_model_spec(first))

    def test_disabled_postprocess_spec_never_attempts_a_checkpoint_load(self):
        module, calls = _load_module()
        spec = module.make_postprocess_model_spec(
            "None", "vae.safetensors", False
        )

        self.assertTrue(spec["postprocess_model_disabled"])
        self.assertTrue(module.is_postprocess_model_spec(spec))
        self.assertEqual("NOT CONFIGURED", module.unload_model_pipe(spec))
        with self.assertRaisesRegex(ValueError, "set to None in the PostProcess Boundary"):
            module.resolve_postprocess_model(spec)
        self.assertEqual([], calls)

    def test_process_change_reuses_base_and_refiner_resources(self):
        module, calls = _load_module()
        loader = module.CMKCheckpointVAELoaderPipe()

        first_base = loader.load_checkpoint_vae_pipe(
            "base.safetensors", "vae.safetensors", True,
            {"model_family": "sdxl", "seed": 1},
        )[0]
        refiner = loader.load_checkpoint_vae_pipe(
            "refiner.safetensors", "vae.safetensors", True,
            {"model_family": "sdxl", "seed": 1},
        )[0]
        second_base = loader.load_checkpoint_vae_pipe(
            "base.safetensors", "vae.safetensors", True,
            {"model_family": "sdxl", "seed": 2},
        )[0]

        self.assertEqual(2, len(calls))
        self.assertIs(first_base["model"], second_base["model"])
        self.assertEqual("MISS · LOADED", first_base["resource_cache_status"])
        self.assertEqual("MISS · LOADED", refiner["resource_cache_status"])
        self.assertEqual("HIT · REUSED", second_base["resource_cache_status"])

    def test_explicit_lifecycle_drops_old_checkpoint_before_switch(self):
        module, calls = _load_module()
        loader = module.CMKCheckpointVAELoaderPipe()
        process = {
            "model_family": "sdxl",
            "unload_models_after_use": True,
        }

        loader.load_checkpoint_vae_pipe(
            "old.safetensors", "vae.safetensors", True, process,
        )
        loader.load_checkpoint_vae_pipe(
            "new.safetensors", "vae.safetensors", True, process,
        )

        self.assertEqual(2, len(calls))
        self.assertEqual(
            [("new.safetensors", "vae.safetensors", True)],
            list(module._SDXL_RESOURCE_CACHE),
        )

    def test_evicted_sdxl_encoder_is_rematerialized_without_reloading_model_or_vae(self):
        module, calls = _load_module()
        loader = module.CMKCheckpointVAELoaderPipe()
        model_pipe = loader.load_checkpoint_vae_pipe(
            "base.safetensors", "vae.safetensors", True,
            {"model_family": "sdxl"},
        )[0]
        model = model_pipe["model"]
        vae = model_pipe["vae"]
        sampler_pipe = {
            "clip": model_pipe["clip"],
            "clip_base": model_pipe["clip"],
            "clip_patched": {"patched": True},
        }
        replacement = {"clip": "replacement"}
        module._load_checkpoint_clip = lambda _name: replacement

        self.assertEqual(
            "EVICTED AFTER ENCODE",
            module.evict_sdxl_text_encoder(model_pipe, sampler_pipe),
        )
        self.assertIsNone(model_pipe["clip"])
        self.assertIsNone(sampler_pipe["clip"])
        self.assertIsNone(sampler_pipe["clip_base"])
        self.assertIsNone(sampler_pipe["clip_patched"])

        clip, status = module.ensure_sdxl_text_encoder(model_pipe)
        self.assertIs(clip, replacement)
        self.assertEqual("REMATERIALIZED", status)
        self.assertIs(model_pipe["model"], model)
        self.assertIs(model_pipe["vae"], vae)
        self.assertEqual(1, len(calls))

    def test_loader_repairs_evicted_cached_sdxl_encoder(self):
        module, calls = _load_module()
        loader = module.CMKCheckpointVAELoaderPipe()
        first = loader.load_checkpoint_vae_pipe(
            "base.safetensors", "vae.safetensors", True,
            {"model_family": "sdxl", "seed": 1},
        )[0]
        replacement = {"clip": "replacement"}
        module._load_checkpoint_clip = lambda _name: replacement
        module.evict_sdxl_text_encoder(first)

        second = loader.load_checkpoint_vae_pipe(
            "base.safetensors", "vae.safetensors", True,
            {"model_family": "sdxl", "seed": 2},
        )[0]
        self.assertIs(second["clip"], replacement)
        self.assertEqual(
            "HIT · TEXT ENCODER REMATERIALIZED",
            second["resource_cache_status"],
        )
        self.assertEqual(1, len(calls))

    def test_repeated_consumers_rematerialize_after_each_explicit_eviction(self):
        module, calls = _load_module()
        loader = module.CMKCheckpointVAELoaderPipe()
        model_pipe = loader.load_checkpoint_vae_pipe(
            "base.safetensors", "vae.safetensors", True,
            {"model_family": "sdxl", "unload_models_after_use": True},
        )[0]
        model = model_pipe["model"]
        vae = model_pipe["vae"]
        rematerialized = []

        def load_clip(_name):
            clip = {"clip": len(rematerialized) + 1}
            rematerialized.append(clip)
            return clip

        module._load_checkpoint_clip = load_clip
        for _index in range(2):
            module.evict_sdxl_text_encoder(model_pipe)
            clip, status = module.ensure_sdxl_text_encoder(model_pipe)
            self.assertIs(clip, rematerialized[-1])
            self.assertEqual("REMATERIALIZED", status)
            self.assertIs(model_pipe["model"], model)
            self.assertIs(model_pipe["vae"], vae)

        self.assertEqual(2, len(rematerialized))
        self.assertEqual(1, len(calls), "MODEL and VAE must not be reloaded")

    def test_sampler_prepare_requests_an_evicted_encoder_before_rejecting_it(self):
        source = (ROOT / "pipe" / "cmk_sampler_prepare.py").read_text(encoding="utf-8")
        lookup = "clip = MODEL.get(\"clip\")"
        rematerialize = "clip, _ = ensure_sdxl_text_encoder(MODEL)"
        rejection = "MODEL['clip'] is missing"
        self.assertLess(source.index(lookup), source.index(rematerialize))
        self.assertLess(source.index(rematerialize), source.index(rejection))


if __name__ == "__main__":
    unittest.main()
