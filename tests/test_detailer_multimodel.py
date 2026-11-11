"""Detailer generations — the SDXL Detailer nodes of 2.7.0.1 and the new
Multi-Model nodes (SDXL / Z-Image Turbo) side by side.

Legacy:      CMKDetailerPreparePipe, CMK_SmartDetailerPipe (unchanged)
Multi-Model: CMKDetailerPrepareMultiModelPipe, CMK_SmartDetailerMultiModelPipe

Model-free: ComfyUI nodes, SAM, LoRA loading and the ZIT text encoder
lifecycle are replaced by recorders.  The legacy identities are pinned to the
values recorded on HEAD b57c230 (= DEV 2b.7.0.1) before any 2.7.1 change.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_detailer_multimodel_tests"
TESTS_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import torch  # noqa: E402

legacy_mod = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_detailer_prepare")
mm_mod = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_detailer_prepare_multimodel")
zit_loader = importlib.import_module(f"{TEST_PACKAGE}.pipe.loaders.z_image_turbo_loader")
smart = importlib.import_module(f"{TEST_PACKAGE}.nodes.image.smart_detailer")
boundary = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_module_boundary_cache")
contract = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_module_cache_contract")
mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")

CMK_TEST_TIER = "T0"

LEGACY_ARTIFACT_KEY = "34a25260396dbec903943cf69c98fb78896095606480d68289f38893bcf80c18"
LEGACY_FINGERPRINTS = {
    "detailer_std": {
        "8417:7706": "d825292fe5f676d35f0453eb56af070cd8933bfd43c881aa02d489795ea0c57f",
        "8417:7719": "c750cc8e46b5e95d61f537805585d75be14915ccc2ae22c42e401c689bc91975",
    },
    "detailer_adv": {
        "8417:7880": "534e4e97f5ac8af49d26ecac525ea883a1c7de5325d47e7785d8628d3832dc49",
        "8417:7881": "f4bdc41b94af34700e6a8c3ab169b9ff19a0364bd49607c06aed48569e0dfe22",
        "8417:7883": "74759646a20ae677bbc24cf3e669564c729ff107413dac499bfe482616f393f9",
        "8417:7884": "d2562a74d092e4d7c1b5b6d8294cff7eb73159f75ea9608bc4708ba96332a9ef",
    },
}
LEGACY_PREPARE_WIDGETS = [
    "sam_model_name", "sam_device_mode", "detailer_global_enable",
    "use_prompt_lora_from_sampler", "use_lora_from_1st_pass", "use_1st_pass_sampling",
    "lora_name", "strength_model", "strength_clip", "prompt_pos", "prompt_neg",
    "steps", "cfg", "sampler", "scheduler", "stop_at_clip_layer", "pag_scale",
    "sampling", "zsnr", "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2",
]
LEGACY_DETAILER_WIDGETS = [
    "enable", "output_image_proceed", "model_name", "bbox_threshold", "bbox_dilation",
    "crop_factor", "drop_size", "guide_size", "guide_size_for", "max_size", "denoise",
    "noise_mask", "force_inpaint",
]
MM_PREPARE_WIDGETS = [
    "model_family", "sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler",
    "use_lora_from_1st_pass", "use_1st_pass_sampling", "lora_name", "strength_model",
    "strength_clip", "prompt_pos", "prompt_neg", "steps", "cfg", "sampling_entry",
    "zit_steps", "zit_cfg", "zit_sampling_entry", "sampler", "scheduler", "zit_sampler",
    "zit_scheduler", "stop_at_clip_layer", "pag_scale", "sampling", "zsnr", "zit_model_shift",
    "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2",
]
LINK_TYPES = {"CMK_MODEL_PIPE", "CMK_RESULT_PROCESS", "IMAGE", "CMK_LOG_PIPE", "CMK_LORA_ZIT_PIPE",
              "CMK_DETAILER_PIPE", "CMK_DETAILER_MULTIMODEL_PIPE", "CMK_VISUAL_PIPE", "CMK_DIAGNOSTIC"}
_LIVE_PROMPTS = []


def _api_prompt(name):
    data = json.loads((TESTS_ROOT / "fixtures" / "api_prompts" / f"{name}.json").read_text(encoding="utf-8"))
    return data.get("prompt", data)


def _fingerprints(prompt):
    # The fingerprint memo is keyed by id(prompt); keep every prompt alive so a
    # recycled id cannot return the memo of an earlier test prompt.
    _LIVE_PROMPTS.append(prompt)
    rows = {}
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type")
        if cls in ("CMK_SmartDetailerPipe", "CMK_SmartDetailerMultiModelPipe"):
            rows[node_id] = mappings.NODE_CLASS_MAPPINGS[cls].dependency_cache_key(prompt, node_id)[0]
        elif cls == "CMKDetailerBoundaryCache":
            rows[node_id] = boundary.CMKDetailerBoundaryCache()._cache_key(prompt, node_id)[0]
    return rows


def _multimodel_prompt(name="detailer_std", **family_inputs):
    """The Detailer SDXL fixture rewired to the Multi-Model node pair."""
    prompt = _api_prompt(name)
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if node.get("class_type") == "CMKDetailerPreparePipe":
            node["class_type"] = "CMKDetailerPrepareMultiModelPipe"
            enabled = inputs.pop("detailer_global_enable", True)
            if "MODEL" in inputs:
                inputs["MODEL SDXL"] = inputs.pop("MODEL")
            inputs.update({
                "model_family": "SDXL" if enabled else "None", "sampling_entry": 95.0,
                "zit_steps": 8, "zit_cfg": 1.0, "zit_sampling_entry": 85.0, "zit_sampler": "res_multistep",
                "zit_scheduler": "simple", "zit_model_shift": 3.0,
            })
            inputs.update(family_inputs)
        elif node.get("class_type") == "CMK_SmartDetailerPipe":
            node["class_type"] = "CMK_SmartDetailerMultiModelPipe"
            inputs.pop("denoise", None)
    return prompt


def _widget_names(spec):
    return [
        name for section in ("required", "optional") for name, value in spec.get(section, {}).items()
        if not (isinstance(value[0], str) and value[0] in LINK_TYPES)
    ]


class _Recorder:
    """Replaces every model-touching dependency of both Prepare generations."""

    def __init__(self, testcase):
        self.calls = []
        self.lora_args = None
        test = self

        class FakeSAM:
            def load(self, name, mode):
                test.calls.append(("sam", name))
                return ("SAM",)

        class FakeLoRA:
            def load_loras(self, model, clip, opt_lora_syntax="", opt_lora_stack=None):
                test.calls.append(("lora", opt_lora_syntax, opt_lora_stack))
                test.lora_args = (model, clip, opt_lora_syntax, opt_lora_stack)
                return (f"{model}+lora", f"{clip}+lora", "", f"loaded[{opt_lora_syntax}|{opt_lora_stack}]")

        def call_node(names, *args):
            test.calls.append(("node", names[0]) + tuple(args[1:]))
            return (f"{names[0]}({args[0]})",)

        def ensure(model_pipe):
            test.calls.append(("ensure_te",))
            model_pipe["clip"] = "QWEN"
            return "QWEN", "REMATERIALIZED"

        def evict(model_pipe):
            test.calls.append(("evict_te",))
            model_pipe["clip"] = None
            return "EVICTED AFTER ENCODE"

        def resolve(value):
            test.calls.append(("resolve_sdxl",))
            return value

        def sdxl_step(name):
            def fn(self, *args):
                test.calls.append(("sdxl", name))
                return args[0] if name != "_apply_single_lora" else (args[0], args[1], "")
            return fn

        class Translation:
            def __init__(self, text):
                self.text = text
                self.status = "disabled"

        patches = {}
        for module in (legacy_mod, mm_mod):
            patches[(module, "CMKSAMLoader")] = FakeSAM
            patches[(module, "translate_prompt")] = Translation
            patches[(module, "make_diagnostic_payload")] = lambda **kw: {"summary": kw.get("summary")}
        patches[(legacy_mod, "resolve_postprocess_model")] = resolve
        patches[(mm_mod, "CMKLoRATextLoader")] = FakeLoRA
        patches[(mm_mod, "_call_node")] = call_node
        patches[(mm_mod, "ensure_zit_text_encoder")] = ensure
        patches[(mm_mod, "evict_zit_text_encoder")] = evict
        patches[(mm_mod, "_validate_conditioning")] = lambda value, name: value
        patches[(legacy_mod.CMKDetailerPreparePipe, "_encode")] = staticmethod(
            lambda clip, text: test.calls.append(("encode", clip, text)) or [[f"COND:{text}", {}]]
        )
        for name in ("_clip_set_last_layer", "_apply_pag", "_apply_sampling", "_apply_freeu", "_apply_single_lora"):
            patches[(legacy_mod.CMKSamplerPrepareSDXLPipe, name)] = sdxl_step(name)
        for (owner, name), value in patches.items():
            original = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
            testcase.addCleanup(setattr, owner, name, original)
            setattr(owner, name, value)

    def kinds(self):
        return [call[0] for call in self.calls]


def _zit_model():
    return {
        "model": "UNET", "clip": None, "vae": "AE", "model_family": "z_image_turbo",
        "diffusion_model_name": "z_image_turbo_bf16.safetensors", "text_encoder_name": "qwen_3_4b.safetensors",
        "vae_name": "ae.safetensors", "weight_dtype": "default", "clip_device": "default",
    }


def _sdxl_args():
    return dict(
        sam_model_name="sam_vit_b_01ec64.pth", sam_device_mode="CPU",
        use_prompt_lora_from_sampler=False, use_lora_from_1st_pass=False, use_1st_pass_sampling=False,
        lora_name="None", strength_model=1.0, strength_clip=1.0, prompt_pos="detailed hands",
        prompt_neg="blurry", steps=30, cfg=7.0, sampler="euler", scheduler="karras",
        stop_at_clip_layer=-2, pag_scale=2.5, sampling="v_prediction", zsnr=True, freeu_enabled=True,
        freeu_b1=1.3, freeu_b2=1.4, freeu_s1=0.9, freeu_s2=0.2,
    )


def _mm_args(**overrides):
    args = dict(
        PROCESS={"result_contract": "family_neutral", "source_model_family": "image", "seed": 5},
        IMAGE=torch.zeros(1, 32, 32, 3), LOG=None,
        model_family="Z-Image Turbo", sampling_entry=95.0, zit_steps=8, zit_cfg=1.0,
        zit_sampling_entry=85.0, zit_sampler="res_multistep", zit_scheduler="simple", zit_model_shift=3.0,
        **_sdxl_args(),
    )
    args["MODEL ZIT"] = _zit_model()
    args["MODEL SDXL"] = {"model": "M", "clip": "C", "vae": "V", "model_family": "sdxl", "ckpt_name": "ck"}
    args.update(overrides)
    return args


def _lazy_kwargs(args):
    return {k: v for k, v in args.items() if k not in ("MODEL SDXL", "PROCESS", "IMAGE", "LOG")}


class Registration(unittest.TestCase):
    def test_both_generations_are_registered(self):
        expected = {
            "CMKDetailerPreparePipe": "CMK Detailer Prepare -Pipe-",
            "CMK_SmartDetailerPipe": "CMK Smart Detailer -Pipe-",
            "CMKDetailerPrepareMultiModelPipe": "CMK Detailer Prepare Multi-Model -Pipe-",
            "CMK_SmartDetailerMultiModelPipe": "CMK Smart Detailer Multi-Model -Pipe-",
        }
        for name, label in expected.items():
            self.assertIn(name, mappings.NODE_CLASS_MAPPINGS)
            self.assertEqual(mappings.NODE_DISPLAY_NAME_MAPPINGS[name], label)

    def test_generations_cannot_be_cross_wired(self):
        legacy_out = mappings.NODE_CLASS_MAPPINGS["CMKDetailerPreparePipe"].RETURN_TYPES[0]
        mm_out = mappings.NODE_CLASS_MAPPINGS["CMKDetailerPrepareMultiModelPipe"].RETURN_TYPES[0]
        legacy_in = mappings.NODE_CLASS_MAPPINGS["CMK_SmartDetailerPipe"].INPUT_TYPES()["required"]["DETAILER"][0]
        mm_in = mappings.NODE_CLASS_MAPPINGS["CMK_SmartDetailerMultiModelPipe"].INPUT_TYPES()["required"]["DETAILER"][0]
        self.assertEqual((legacy_out, legacy_in), ("CMK_DETAILER_PIPE", "CMK_DETAILER_PIPE"))
        self.assertEqual((mm_out, mm_in), ("CMK_DETAILER_MULTIMODEL_PIPE", "CMK_DETAILER_MULTIMODEL_PIPE"))


class LegacyContract(unittest.TestCase):
    def test_widget_order_is_the_2701_order(self):
        self.assertEqual(_widget_names(legacy_mod.CMKDetailerPreparePipe.INPUT_TYPES()), LEGACY_PREPARE_WIDGETS)
        self.assertEqual(_widget_names(smart.CMK_SmartDetailerPipe.INPUT_TYPES()), LEGACY_DETAILER_WIDGETS)

    def test_saved_workflows_carry_legacy_widget_values(self):
        roots = [PACKAGE_ROOT / "subgraphs", PACKAGE_ROOT / "workflows"]
        counts = {"CMKDetailerPreparePipe": 0, "CMK_SmartDetailerPipe": 0}

        def walk(value):
            if isinstance(value, dict):
                kind = value.get("type")
                if kind in counts and isinstance(value.get("widgets_values"), list):
                    expected = LEGACY_PREPARE_WIDGETS if kind == "CMKDetailerPreparePipe" else LEGACY_DETAILER_WIDGETS
                    self.assertEqual(len(value["widgets_values"]), len(expected), kind)
                    counts[kind] += 1
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        for root in roots:
            for path in root.rglob("*.json"):
                walk(json.loads(path.read_text(encoding="utf-8")))
        self.assertTrue(all(counts.values()), counts)

    def test_fingerprints_and_artifact_key_unchanged(self):
        for fixture, pinned in LEGACY_FINGERPRINTS.items():
            with self.subTest(fixture=fixture):
                self.assertEqual(_fingerprints(_api_prompt(fixture)), pinned)
        self.assertEqual(smart.CMK_SmartDetailerPipe()._cache_key(None, None, CacheIdentity.sdxl_pipe(),
                                                                   CacheIdentity.RUN)[0], LEGACY_ARTIFACT_KEY)

    def test_sampling_entry_stays_a_smart_detailer_widget(self):
        spec = smart.CMK_SmartDetailerPipe.INPUT_TYPES()["required"]["denoise"]
        self.assertEqual((spec[1]["default"], spec[1]["min"], spec[1]["max"]), (80.0, 50.0, 100.0))


class LoaderProcessOptional(unittest.TestCase):
    def setUp(self):
        self.loaded = []

        def call(names, **kwargs):
            self.loaded.append(names[0])
            return (f"{names[0]}:{sorted(kwargs.items())}",)

        original = zit_loader._call_node_kwargs
        self.addCleanup(setattr, zit_loader, "_call_node_kwargs", original)
        zit_loader._call_node_kwargs = call
        self.addCleanup(zit_loader._ZIT_RESOURCE_CACHE.clear)
        zit_loader._ZIT_RESOURCE_CACHE.clear()

    def _load(self, **kwargs):
        return zit_loader.CMKZImageTurboLoaderPipe().load(
            "z_image_turbo_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors", **kwargs
        )

    def test_loads_without_process(self):
        self.assertNotIn("PROCESS", zit_loader.CMKZImageTurboLoaderPipe.INPUT_TYPES()["required"])
        model, _log = self._load()
        self.assertEqual(self.loaded, ["UNETLoader", "CLIPLoader", "VAELoader"])
        self.assertEqual(model["model_family"], "z_image_turbo")

    def test_connected_process_keeps_its_contract(self):
        blocked, _ = self._load(PROCESS={"model_family": "z_image_turbo", "family_active": False})
        self.assertEqual(type(blocked).__name__, "ExecutionBlocker")
        with self.assertRaises(ValueError):
            self._load(PROCESS={"model_family": "sdxl"})
        self.assertEqual(self.loaded, [])
        model, _ = self._load(PROCESS={"model_family": "z_image_turbo"})
        self.assertEqual(model["resource_cache_status"], "MISS · LOADED")


class ZitPostProcessSpec(unittest.TestCase):
    """The ZIT spec loads nothing; resolution shares the ZIT loader cache."""

    FILES = ("z_image_turbo_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors", "default", "default")

    def setUp(self):
        self.loaded = []

        def call(names, **kwargs):
            self.loaded.append(names[0])
            return (f"{names[0]}#{len(self.loaded)}",)

        original = zit_loader._call_node_kwargs
        self.addCleanup(setattr, zit_loader, "_call_node_kwargs", original)
        zit_loader._call_node_kwargs = call
        self.addCleanup(zit_loader._ZIT_RESOURCE_CACHE.clear)
        zit_loader._ZIT_RESOURCE_CACHE.clear()
        self.spec = zit_loader.CMKZImageTurboPostProcessModelPipe().load(*self.FILES)[0]

    def test_spec_loads_nothing_and_is_a_primitive_contract(self):
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.spec["model_family"], "z_image_turbo")
        self.assertEqual(self.spec["model_role"], "postprocess")
        self.assertFalse(self.spec["materialized"])
        self.assertTrue(all(type(value) in (str, bool) for value in self.spec.values()))
        contract._canonical(self.spec)
        node = zit_loader.CMKZImageTurboPostProcessModelPipe
        self.assertEqual(node.RETURN_TYPES, ("CMK_MODEL_PIPE",))
        self.assertEqual(node.INPUT_TYPES()["required"], zit_loader.CMKZImageTurboLoaderPipe.INPUT_TYPES()["required"])
        self.assertEqual(mappings.NODE_DISPLAY_NAME_MAPPINGS["CMKZImageTurboPostProcessModelPipe"],
                         "CMK Z-Image Turbo PostProcess Model")

    def test_resolve_returns_a_copy_and_loads_unet_and_vae_once(self):
        before = dict(self.spec)
        first = zit_loader.resolve_zit_postprocess_model(self.spec)
        self.assertEqual(self.spec, before, "the spec itself must stay unchanged")
        self.assertIsNot(first, self.spec)
        self.assertEqual(self.loaded, ["UNETLoader", "VAELoader"])
        self.assertEqual((first["model"], first["clip"], first["vae"]), ("UNETLoader#1", None, "VAELoader#2"))
        self.assertEqual(first["resource_cache_status"], "MISS · LOADED")
        second = zit_loader.resolve_zit_postprocess_model(self.spec)
        self.assertEqual(self.loaded, ["UNETLoader", "VAELoader"], "a second consumer shares the resources")
        self.assertIs(second["model"], first["model"])
        self.assertEqual(second["resource_cache_status"], "HIT · REUSED")

    def test_text_encoder_lifecycle(self):
        pipe = zit_loader.resolve_zit_postprocess_model(self.spec)
        clip, status = zit_loader.ensure_zit_text_encoder(pipe)
        self.assertEqual((clip, status, self.loaded[-1]), ("CLIPLoader#3", "REMATERIALIZED", "CLIPLoader"))
        self.assertEqual(zit_loader.evict_zit_text_encoder(pipe), "EVICTED AFTER ENCODE")
        self.assertIsNone(zit_loader._ZIT_RESOURCE_CACHE[self.FILES][1])
        self.assertIsNone(self.spec.get("clip"))
        again = zit_loader.resolve_zit_postprocess_model(self.spec)
        self.assertEqual((again["model"], again["clip"]), ("UNETLoader#1", None))
        self.assertEqual(self.loaded, ["UNETLoader", "VAELoader", "CLIPLoader"])

    def test_loader_and_spec_share_one_cache(self):
        zit_loader.resolve_zit_postprocess_model(self.spec)
        loader_pipe, _log = zit_loader.CMKZImageTurboLoaderPipe().load(*self.FILES)
        self.assertEqual(loader_pipe["resource_cache_status"], "HIT · TEXT ENCODER REMATERIALIZED")
        self.assertEqual(self.loaded, ["UNETLoader", "VAELoader", "CLIPLoader"])
        resolved = zit_loader.resolve_zit_postprocess_model(self.spec)
        self.assertEqual(resolved["clip"], loader_pipe["clip"])
        self.assertEqual(len(self.loaded), 3)
        identity = ("diffusion_model_name", "text_encoder_name", "vae_name", "weight_dtype", "clip_device")
        self.assertEqual({k: loader_pipe[k] for k in identity}, {k: self.spec[k] for k in identity})

    def test_loader_behaviour_is_unchanged(self):
        first, _ = zit_loader.CMKZImageTurboLoaderPipe().load(*self.FILES)
        self.assertEqual((first["resource_cache_status"], self.loaded), ("MISS · LOADED", ["UNETLoader", "CLIPLoader", "VAELoader"]))
        second, _ = zit_loader.CMKZImageTurboLoaderPipe().load(*self.FILES)
        self.assertEqual(second["resource_cache_status"], "HIT · REUSED")
        zit_loader.evict_zit_text_encoder(second)
        third, _ = zit_loader.CMKZImageTurboLoaderPipe().load(*self.FILES)
        self.assertEqual(third["resource_cache_status"], "HIT · TEXT ENCODER REMATERIALIZED")
        self.assertEqual(self.loaded, ["UNETLoader", "CLIPLoader", "VAELoader", "CLIPLoader"])

    def test_model_switch_replaces_the_cache_entry(self):
        zit_loader.resolve_zit_postprocess_model(self.spec)
        other = zit_loader.make_zit_postprocess_model_spec("other.safetensors", *self.FILES[1:])
        resolved = zit_loader.resolve_zit_postprocess_model(other)
        self.assertEqual(resolved["resource_cache_status"], "MISS · LOADED")
        self.assertEqual(list(zit_loader._ZIT_RESOURCE_CACHE), [("other.safetensors", *self.FILES[1:])])

    def test_sdxl_and_zit_specs_are_separate(self):
        sdxl_loader = importlib.import_module(f"{TEST_PACKAGE}.pipe.loaders.checkpoint_vae_loader")
        self.assertFalse(sdxl_loader.is_postprocess_model_spec(self.spec))
        self.assertFalse(zit_loader.is_zit_postprocess_model_spec(
            sdxl_loader.make_postprocess_model_spec("ck.safetensors", "v.safetensors", True)))
        with self.assertRaisesRegex(ValueError, "needs MODEL SDXL"):
            sdxl_loader.resolve_postprocess_model(self.spec)
        key = sdxl_loader._resource_key(sdxl_loader.DEFAULT_CHECKPOINT, "", True)
        self.addCleanup(sdxl_loader._SDXL_RESOURCE_CACHE.pop, key, None)
        sdxl_loader._SDXL_RESOURCE_CACHE[key] = ("SDXL", "C", "V", {})
        self.assertEqual(sdxl_loader.unload_model_pipe(self.spec), "NOT MATERIALIZED")
        self.assertEqual(sdxl_loader._SDXL_RESOURCE_CACHE[key][0], "SDXL")
        self.assertEqual(self.loaded, [])


class MultiModelWidgets(unittest.TestCase):
    def test_order_and_defaults(self):
        spec = mm_mod.CMKDetailerPrepareMultiModelPipe.INPUT_TYPES()
        self.assertEqual(_widget_names(spec), MM_PREPARE_WIDGETS)
        required = spec["required"]
        self.assertEqual(required["model_family"], (["None", "SDXL", "Z-Image Turbo"], {"default": "None"}))
        self.assertEqual(required["sampling_entry"][1]["default"], 95.0)
        self.assertEqual(required["zit_sampling_entry"][1]["default"], 85.0)
        self.assertEqual(required["zit_sampler"][1]["default"], "res_multistep")
        self.assertEqual(required["zit_scheduler"][1]["default"], "simple")
        self.assertEqual(required["zit_steps"][1]["default"], 8)
        self.assertEqual(required["zit_cfg"][1]["default"], 1.0)
        self.assertEqual(required["zit_model_shift"][1]["default"], 3.0)
        for name in ("MODEL ZIT", "LoRA ZIT", "LOG"):
            self.assertTrue(spec["optional"][name][1]["lazy"], name)
        self.assertNotIn("detailer_global_enable", required)

    def test_model_sockets(self):
        spec = mm_mod.CMKDetailerPrepareMultiModelPipe.INPUT_TYPES()
        self.assertEqual(list(spec["required"])[:3], ["MODEL SDXL", "PROCESS", "IMAGE"])
        self.assertNotIn("MODEL", spec["required"])
        self.assertEqual(spec["required"]["MODEL SDXL"], ("CMK_MODEL_PIPE", {"lazy": True}))
        self.assertEqual(spec["optional"]["MODEL ZIT"], ("CMK_MODEL_PIPE", {"lazy": True}))
        legacy = legacy_mod.CMKDetailerPreparePipe.INPUT_TYPES()["required"]
        self.assertIn("MODEL", legacy)

    def test_family_enable(self):
        node = mm_mod.CMKDetailerFamilyEnable
        self.assertEqual(node.INPUT_TYPES()["required"]["model_family"][0], ["None", "SDXL", "Z-Image Turbo"])
        self.assertEqual((node.RETURN_TYPES, node.CATEGORY), (("BOOLEAN",), "CMK/Internal/Gates"))
        self.assertEqual([node.read(value)[0] for value in ("None", "SDXL", "Z-Image Turbo")], [False, True, True])
        self.assertEqual(mappings.NODE_CLASS_MAPPINGS["CMKDetailerFamilyEnable"], node)

    def test_smart_detailer_has_no_sampling_entry_widget(self):
        spec = smart.CMK_SmartDetailerMultiModelPipe.INPUT_TYPES()
        self.assertEqual(_widget_names(spec), [name for name in LEGACY_DETAILER_WIDGETS if name != "denoise"])


class MultiModelLazy(unittest.TestCase):
    def setUp(self):
        self.node = mm_mod.CMKDetailerPrepareMultiModelPipe()
        self.image = torch.zeros(1, 8, 8, 3)

    def lazy(self, **kwargs):
        args = _lazy_kwargs(_mm_args())
        args.pop("MODEL ZIT")
        args.update(kwargs)
        return self.node.check_lazy_status(PROCESS=None, IMAGE=self.image, LOG="L", **{"MODEL SDXL": None}, **args)

    def test_upstream_result_first(self):
        args = _lazy_kwargs(_mm_args())
        self.assertEqual(self.node.check_lazy_status(PROCESS=None, IMAGE=None, LOG=None, **{"MODEL SDXL": None}, **args),
                         ["IMAGE", "LOG"])

    def test_family_dependent_requests(self):
        linked = {"MODEL ZIT": None, "LoRA ZIT": None}
        self.assertEqual(self.lazy(model_family="None", **linked), [])
        self.assertEqual(self.lazy(model_family="SDXL", **linked), ["PROCESS", "MODEL SDXL"])
        self.assertEqual(self.lazy(model_family="Z-Image Turbo", **linked), ["PROCESS", "MODEL ZIT", "LoRA ZIT"])
        self.assertEqual(self.lazy(model_family="Z-Image Turbo", **{"MODEL ZIT": None}), ["PROCESS", "MODEL ZIT"])
        self.assertEqual(self.lazy(model_family="Z-Image Turbo"), ["PROCESS"])


class MultiModelPrepare(unittest.TestCase):
    def setUp(self):
        self.rec = _Recorder(self)
        self.node = mm_mod.CMKDetailerPrepareMultiModelPipe()

    def test_none_is_the_complete_bypass(self):
        args = _mm_args(model_family="None", PROCESS=None, **{"MODEL ZIT": None, "MODEL SDXL": None})
        pipe, _log, _diag = self.node.prepare(**args)
        self.assertTrue(pipe["detailer_prepare_bypassed"])
        self.assertFalse(pipe["detailer_global_enable"])
        self.assertEqual(pipe["detailer_model_family"], "none")
        self.assertIs(pipe["detailer_image"], args["IMAGE"])
        self.assertEqual(self.rec.calls, [])

    def test_sdxl_is_the_legacy_preparation_plus_its_sampling_entry(self):
        args = _mm_args(model_family="SDXL", sampling_entry=93.0, zit_sampling_entry=70.0)
        zit_model = args["MODEL ZIT"]
        pipe, _log, _diag = self.node.prepare(**args)
        self.assertEqual([c[1] for c in self.rec.calls if c[0] == "sdxl"],
                         ["_clip_set_last_layer", "_apply_single_lora", "_apply_pag", "_apply_sampling", "_apply_freeu"])
        self.assertIn("resolve_sdxl", self.rec.kinds())
        self.assertNotIn("ensure_te", self.rec.kinds())
        self.assertEqual(pipe["detailer_model_family"], "sdxl")
        self.assertEqual(pipe["detailer_sampling_entry"], 93.0)
        self.assertEqual((pipe["detailer_steps"], pipe["detailer_cfg"]), (30, 7.0))
        self.assertEqual((pipe["detailer_sampler"], pipe["detailer_scheduler"]), ("euler", "karras"))
        self.assertEqual(zit_model, _zit_model())

        legacy, _, _ = legacy_mod.CMKDetailerPreparePipe().prepare(
            MODEL=args["MODEL SDXL"], PROCESS=args["PROCESS"], IMAGE=args["IMAGE"], detailer_global_enable=True,
            LOG=None, **_sdxl_args())
        self.assertEqual({k: v for k, v in pipe.items() if k not in ("detailer_model_family", "detailer_sampling_entry")},
                         legacy)

    def test_zit_reference_contract(self):
        bundle = {"family": "z_image_turbo", "lora_stack": [("ZIT/a.safetensors", 0.8, 0.8)],
                  "active_loras": "<lora:ZIT/b:1>", "trigger_words": "nina"}
        args = _mm_args(sampling_entry=99.0, zit_sampling_entry=82.0, **{"LoRA ZIT": bundle})
        zit_model = args["MODEL ZIT"]
        pipe, log, _diag = self.node.prepare(**args)
        self.assertEqual(self.rec.kinds(), ["ensure_te", "sam", "lora", "encode", "node", "evict_te", "node"])
        self.assertNotIn("sdxl", self.rec.kinds())
        self.assertNotIn("resolve_sdxl", self.rec.kinds())
        self.assertEqual(self.rec.lora_args, ("UNET", "QWEN", "<lora:ZIT/b:1>", bundle["lora_stack"]))
        self.assertEqual(self.rec.calls[3], ("encode", "QWEN+lora", "detailed hands\nnina"))
        self.assertEqual(self.rec.calls[4][1], "ConditioningZeroOut")
        self.assertEqual(self.rec.calls[6], ("node", "ModelSamplingAuraFlow", 3.0))
        self.assertIsNone(zit_model["clip"])
        self.assertEqual(pipe["detailer_model_family"], "z_image_turbo")
        self.assertEqual(pipe["detailer_sampling_entry"], 82.0)
        self.assertEqual((pipe["detailer_steps"], pipe["detailer_cfg"]), (8, 1.0))
        self.assertEqual((pipe["detailer_sampler"], pipe["detailer_scheduler"]), ("res_multistep", "simple"))
        self.assertIsNone(pipe["detailer_clip"])
        self.assertIn("SAMPLING ENTRY  : 82 %", str(log))

    def test_zit_spec_is_resolved_in_prepare_only(self):
        spec = zit_loader.make_zit_postprocess_model_spec(
            "z_image_turbo_bf16.safetensors", "qwen_3_4b.safetensors", "ae.safetensors")
        resolved = []

        def resolve(value):
            resolved.append(value)
            return {**value, "model": "UNET", "vae": "AE", "clip": None, "materialized": True}

        self.addCleanup(setattr, mm_mod, "resolve_zit_postprocess_model", mm_mod.resolve_zit_postprocess_model)
        mm_mod.resolve_zit_postprocess_model = resolve
        pipe, _log, _diag = self.node.prepare(**_mm_args(**{"MODEL ZIT": spec}))
        self.assertEqual(resolved, [spec])
        self.assertFalse(spec["materialized"])
        self.assertEqual(pipe["detailer_model_identity"], {
            "diffusion_model_name": "z_image_turbo_bf16.safetensors", "text_encoder_name": "qwen_3_4b.safetensors",
            "vae_name": "ae.safetensors", "weight_dtype": "default", "clip_device": "default"})
        self.rec.calls.clear()
        resolved.clear()
        self.node.prepare(**_mm_args(model_family="None", **{"MODEL ZIT": spec}))
        self.node.prepare(**_mm_args(model_family="SDXL", **{"MODEL ZIT": spec}))
        self.assertEqual(resolved, [], "None and SDXL never resolve the ZIT spec")

    def test_sdxl_lora_bundle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "LoRA ZIT does not match"):
            self.node.prepare(**_mm_args(**{"LoRA ZIT": {"family": "sdxl", "active_loras": "<lora:x:1>"}}))

    def test_first_pass_lora_only_from_zit_source(self):
        process = {"result_contract": "family_neutral", "source_model_family": "z_image_turbo", "seed": 1,
                   "lora_syntax": "<lora:ZIT/pass:1>", "lora_stack": None, "prompt_pos": "pass prompt"}
        self.node.prepare(**_mm_args(PROCESS=process, use_lora_from_1st_pass=True))
        self.assertEqual(self.rec.lora_args[2], "<lora:ZIT/pass:1>")
        process = dict(process, source_model_family="sdxl")
        pipe, _, _ = self.node.prepare(**_mm_args(PROCESS=process, use_lora_from_1st_pass=True))
        self.assertEqual(self.rec.lora_args[2], "")
        self.assertFalse(pipe["detailer_use_lora_from_1st_pass"])

    def test_missing_zit_loader_fails_clearly(self):
        args = _mm_args()
        args.pop("MODEL ZIT")
        with self.assertRaisesRegex(ValueError, "MODEL ZIT"):
            self.node.prepare(**args)
        self.assertNotIn("sam", self.rec.kinds())


class MultiModelSmartDetailer(unittest.TestCase):
    def setUp(self):
        self.seen = {}
        test = self

        def fake_run(self, **kwargs):
            test.seen.update(kwargs)
            image = kwargs["image"]
            empty = ((image.shape[2], image.shape[1]), [])
            return image, empty, empty, "log", True, {"summary": "x"}

        for owner, name, value in (
            (smart.CMK_SmartDetailer, "run", fake_run),
            (smart, "make_stable_segs", lambda segs, *a, **k: segs),
        ):
            self.addCleanup(setattr, owner, name, getattr(owner, name))
            setattr(owner, name, value)
        self.image = torch.zeros(1, 16, 16, 3)

    def pipe(self, **changes):
        pipe = {"detailer_image": self.image, "detailer_global_enable": True, "detailer_model": "M",
                "detailer_clip": "C", "detailer_vae": "V", "detailer_conditioning_pos": "P",
                "detailer_conditioning_neg": "N", "detailer_model_family": "sdxl", "detailer_sampling_entry": 95.0,
                "detailer_steps": 30, "detailer_cfg": 7.0, "detailer_sampler": "euler",
                "detailer_scheduler": "karras", "detailer_seed": 3}
        pipe.update(changes)
        return pipe

    def run_pipe(self, pipe):
        return smart.CMK_SmartDetailerMultiModelPipe().run_pipe(
            pipe, True, True, "bbox/hand_yolov8s.pt", 0.5, 0, 2.0, 10, 512, True, 768, True, True,
        )

    def log_text(self, result):
        return json.dumps(result[3], ensure_ascii=False)

    def test_sdxl_uses_the_pipe_sampling_entry(self):
        result = self.run_pipe(self.pipe())
        self.assertAlmostEqual(self.seen["denoise"], 0.05)
        self.assertEqual((self.seen["steps"], self.seen["cfg"], self.seen["sampler_name"], self.seen["scheduler"]),
                         (30, 7.0, "euler", "karras"))
        text = self.log_text(result)
        for line in ("MODEL FAMILY    : SDXL", "SAMPLING ENTRY  : 95 %", "DENOISE         : 0.05",
                     "STEPS           : 30", "CFG             : 7.0", "SAMPLER         : euler",
                     "SCHEDULER       : karras"):
            self.assertIn(line, text)

    def test_zit_uses_its_own_entry_and_released_text_encoder(self):
        result = self.run_pipe(self.pipe(detailer_model_family="z_image_turbo", detailer_clip=None,
                                         detailer_sampling_entry=85.0, detailer_steps=8, detailer_cfg=1.0,
                                         detailer_sampler="res_multistep", detailer_scheduler="simple"))
        self.assertAlmostEqual(self.seen["denoise"], 0.15)
        self.assertEqual(self.seen["basic_pipe"][1], smart.CMK_SmartDetailerMultiModelPipe._RELEASED_TEXT_ENCODER)
        self.assertIn("MODEL FAMILY    : Z-IMAGE TURBO", self.log_text(result))
        self.assertIn("SAMPLING ENTRY  : 85 %", self.log_text(result))

    def test_none_is_passthrough(self):
        bypass = {"source_pipe": {}, "_source_pipe": {}, "detailer_image": self.image,
                  "detailer_global_enable": False, "boolean_detailer_enable": False,
                  "detailer_prepare_bypassed": True, "detailer_model_family": "none"}
        result = self.run_pipe(bypass)
        self.assertEqual(self.seen, {})
        self.assertIs(result[2], self.image)
        self.assertIn("MODEL FAMILY    : NONE", self.log_text(result))
        self.assertIn("SAMPLING        : SKIPPED", self.log_text(result))


class CacheIdentity(unittest.TestCase):
    RUN = {"enable": True, "model_name": "bbox/hand_yolov8s.pt", "bbox_threshold": 0.5, "bbox_dilation": 0,
           "crop_factor": 2.0, "drop_size": 10, "guide_size": 512.0, "guide_size_for": True, "max_size": 768.0,
           "denoise": 85.0, "noise_mask": True, "force_inpaint": True}

    @staticmethod
    def sdxl_pipe():
        source = contract.stamp_artifact({"cmk_artifacts": {"sdxl.refiner": "a" * 64}}, "sdxl.refiner", "a" * 64)
        return {
            "source_pipe": source, "detailer_seed": 7, "detailer_global_enable": True, "detailer_steps": 30,
            "detailer_cfg": 7.0, "detailer_sampler": "euler", "detailer_scheduler": "simple",
            "detailer_sam_model_name": "sam_vit_b_01ec64.pth", "detailer_use_prompt_lora_from_sampler": False,
            "detailer_use_prompt_from_1st_pass": False, "detailer_use_lora_from_1st_pass": False,
            "detailer_use_1st_pass_sampling": False, "detailer_prompt_pos": "hands", "detailer_prompt_neg": "",
            "detailer_active_loras": "", "detailer_stop_at_clip_layer": -2, "detailer_pag_scale": 2.5,
            "detailer_sampling": "v_prediction", "detailer_zsnr": True, "detailer_freeu_enabled": True,
            "detailer_freeu_b1": 1.3, "detailer_freeu_b2": 1.4, "detailer_freeu_s1": 0.9, "detailer_freeu_s2": 0.2,
        }

    def mm_key(self, pipe):
        node = smart.CMK_SmartDetailerMultiModelPipe()
        run = dict(self.RUN, denoise=pipe.get("detailer_sampling_entry"))
        return node._cache_key(None, None, pipe, run)[0]

    def mm_sdxl(self, **changes):
        return {**self.sdxl_pipe(), "detailer_model_family": "sdxl", "detailer_sampling_entry": 95.0, **changes}

    def mm_zit(self, **changes):
        pipe = self.mm_sdxl(
            detailer_model_family="z_image_turbo", detailer_sampling_entry=85.0, detailer_steps=8, detailer_cfg=1.0,
            detailer_sampler="res_multistep", detailer_scheduler="simple", detailer_loaded_loras="",
            detailer_zit_trigger_words="", detailer_zit_model_shift=3.0,
            detailer_model_identity={"diffusion_model_name": "z_image_turbo_bf16.safetensors",
                                     "text_encoder_name": "qwen_3_4b.safetensors", "vae_name": "ae.safetensors",
                                     "weight_dtype": "default", "clip_device": "default"},
        )
        pipe.update(changes)
        return pipe

    def test_generations_never_share_artifacts(self):
        legacy = smart.CMK_SmartDetailerPipe()._cache_key(None, None, self.sdxl_pipe(), dict(self.RUN, denoise=95.0))[0]
        self.assertNotEqual(self.mm_key(self.mm_sdxl()), legacy)
        self.assertEqual(self.mm_key(self.mm_sdxl()), self.mm_key(self.mm_sdxl()))

    def test_multimodel_identity_follows_every_effective_setting(self):
        identity = dict(self.mm_zit()["detailer_model_identity"], diffusion_model_name="other.safetensors")
        keys = {self.mm_key(self.mm_sdxl()), self.mm_key(self.mm_zit())}
        variants = [
            self.mm_sdxl(detailer_sampling_entry=90.0),
            self.mm_sdxl(detailer_scheduler="karras"),
            self.mm_zit(detailer_sampling_entry=80.0),
            self.mm_zit(detailer_model_identity=identity),
            self.mm_zit(detailer_loaded_loras="ZIT/a 0.8"),
            self.mm_zit(detailer_zit_trigger_words="nina"),
            self.mm_zit(detailer_zit_model_shift=2.5),
            self.mm_zit(detailer_sampler="euler"),
            self.mm_zit(detailer_steps=6),
        ]
        for pipe in variants:
            keys.add(self.mm_key(pipe))
        self.assertEqual(len(keys), len(variants) + 2)

    def test_multimodel_fingerprints(self):
        legacy = _fingerprints(_api_prompt("detailer_std"))
        sdxl = _fingerprints(_multimodel_prompt())
        self.assertEqual(set(sdxl), set(legacy))
        for node_id in sdxl:
            self.assertNotEqual(sdxl[node_id], legacy[node_id], "generations must not share fingerprints")
        self.assertEqual(sdxl, _fingerprints(_multimodel_prompt(zit_steps=4, zit_sampling_entry=70.0,
                                                                zit_sampler="euler")))
        self.assertNotEqual(sdxl, _fingerprints(_multimodel_prompt(sampling_entry=90.0)))
        zit = _fingerprints(_multimodel_prompt(model_family="Z-Image Turbo"))
        self.assertNotEqual(zit, sdxl)
        for change in ({"zit_sampling_entry": 80.0}, {"zit_sampler": "euler"}, {"zit_model_shift": 2.0}):
            changed = _fingerprints(_multimodel_prompt(model_family="Z-Image Turbo", **change))
            for node_id in zit:
                self.assertNotEqual(zit[node_id], changed[node_id], change)


class MultiModelBoundary(unittest.TestCase):
    def boundary_id(self, prompt):
        return next(node_id for node_id, node in prompt.items()
                    if isinstance(node, dict) and node.get("class_type") == "CMKDetailerBoundaryCache")

    def test_disabled_state(self):
        for family, disabled in (("None", True), ("SDXL", False), ("Z-Image Turbo", False)):
            prompt = _multimodel_prompt(model_family=family)
            with self.subTest(family=family):
                self.assertEqual(boundary._detailer_disabled_state(prompt, self.boundary_id(prompt))[0], disabled)
        legacy = _api_prompt("detailer_std")
        self.assertFalse(boundary._detailer_disabled_state(legacy, self.boundary_id(legacy))[0])

    def test_manifest_uses_the_multimodel_branch_key(self):
        prompt = _multimodel_prompt()
        _LIVE_PROMPTS.append(prompt)
        node_id = self.boundary_id(prompt)
        manifest = boundary.CMKDetailerBoundaryCache()._dependencies(prompt, node_id)[0]
        branches = manifest["branches"]
        self.assertEqual([entry["class_type"] for entry in branches], ["CMK_SmartDetailerMultiModelPipe"])
        entry = branches[0]
        self.assertEqual(entry["cache_key"],
                         smart.CMK_SmartDetailerMultiModelPipe.dependency_cache_key(prompt, entry["node_id"])[0])


class FamilyRetentionSlot(unittest.TestCase):
    """2.7.1: one latest Detailer boundary entry per node and model family."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._original = boundary.cache_directory

        def cache_directory(scope):
            path = root / scope
            path.mkdir(parents=True, exist_ok=True)
            return path

        boundary.cache_directory = cache_directory
        self.scope = "detailer_boundary"

    def tearDown(self):
        boundary.cache_directory = self._original
        self._tmp.cleanup()

    def entry(self, key):
        for path in boundary._entry_paths(self.scope, key):
            path.write_bytes(b"x")
        return key

    def exists(self, key):
        return boundary._disk_available(self.scope, key)

    def index(self):
        return json.loads((boundary.cache_directory(self.scope) / "latest_by_node.json").read_text())

    def retain(self, key, node="7822:7706", family=None):
        boundary._retain_latest_boundary_entry(self.scope, self.entry(key), node, slot_family=family)

    def boundary_id(self, prompt):
        return next(node_id for node_id, node in prompt.items()
                    if isinstance(node, dict) and node.get("class_type") == "CMKDetailerBoundaryCache")

    def test_family_from_the_multimodel_prepare(self):
        for family, expected in (("SDXL", "sdxl"), ("Z-Image Turbo", "z_image_turbo"), ("None", None)):
            prompt = _multimodel_prompt(model_family=family)
            with self.subTest(family=family):
                self.assertEqual(boundary._detailer_retention_family(prompt, self.boundary_id(prompt)), expected)

    def test_legacy_and_unresolvable_graphs_keep_the_plain_slot(self):
        legacy = _api_prompt("detailer_std")
        self.assertIsNone(boundary._detailer_retention_family(legacy, self.boundary_id(legacy)))
        linked = _multimodel_prompt(model_family=["999", 0])
        self.assertIsNone(boundary._detailer_retention_family(linked, self.boundary_id(linked)))
        self.assertIsNone(boundary._detailer_retention_family(None, "1"))

    def test_instances_resolve_by_their_own_prefix(self):
        prompt = {}
        for prefix, family in (("10", "SDXL"), ("20", "Z-Image Turbo")):
            for node_id, node in _multimodel_prompt(model_family=family).items():
                prompt[f"{prefix}:{node_id}"] = json.loads(json.dumps(node).replace('["', f'["{prefix}:'))
        ids = sorted(node_id for node_id, node in prompt.items() if node["class_type"] == "CMKDetailerBoundaryCache")
        self.assertEqual([boundary._detailer_retention_family(prompt, node_id) for node_id in ids],
                         ["sdxl", "z_image_turbo"])

    def test_alternating_families_keep_both_entries(self):
        self.retain("sdxl-a", family="sdxl")
        self.retain("zit-a", family="z_image_turbo")
        self.assertTrue(self.exists("sdxl-a") and self.exists("zit-a"))
        self.retain("sdxl-a", family="sdxl")
        self.retain("zit-a", family="z_image_turbo")
        self.assertTrue(self.exists("sdxl-a") and self.exists("zit-a"))
        self.assertEqual(self.index(), {"7822:7706#sdxl": "sdxl-a", "7822:7706#z_image_turbo": "zit-a"})

    def test_changed_parameters_replace_only_their_family(self):
        self.retain("sdxl-a", family="sdxl")
        self.retain("zit-a", family="z_image_turbo")
        self.retain("sdxl-b", family="sdxl")
        self.assertEqual([self.exists(k) for k in ("sdxl-a", "sdxl-b", "zit-a")], [False, True, True])
        self.retain("zit-b", family="z_image_turbo")
        self.assertEqual([self.exists(k) for k in ("sdxl-b", "zit-a", "zit-b")], [True, False, True])
        self.assertEqual(len(self.index()), 2)

    def test_plain_slot_rule_is_unchanged(self):
        self.retain("a")
        self.retain("b")
        self.retain("other", node="99")
        self.assertEqual([self.exists(k) for k in ("a", "b", "other")], [False, True, True])
        self.assertEqual(self.index(), {"7822:7706": "b", "99": "other"})

    def test_pre_family_entry_is_adopted_without_migration(self):
        self.retain("old")
        self.retain("old", family="sdxl")
        self.assertTrue(self.exists("old"))
        self.assertEqual(self.index(), {"7822:7706#sdxl": "old"})

    def test_pre_family_entry_of_the_other_family_stays_readable(self):
        self.retain("old")
        self.retain("zit-a", family="z_image_turbo")
        self.assertTrue(self.exists("old") and self.exists("zit-a"))
        self.assertEqual(self.index(), {"7822:7706": "old", "7822:7706#z_image_turbo": "zit-a"})

    def test_boundary_passes_the_family_slot(self):
        import inspect
        source = inspect.getsource(boundary._DetailerBoundaryCore.boundary)
        self.assertEqual(source.count("slot_family=_detailer_retention_family(prompt, unique_id)"), 2)
        for core in ("_FaceRebuildBoundaryCore", "_ZImageBoundaryCore", "_FaceBoundaryCore", "_FaceSwapBoundaryCore"):
            self.assertNotIn("slot_family", inspect.getsource(getattr(boundary, core)))

if __name__ == "__main__":
    unittest.main()
