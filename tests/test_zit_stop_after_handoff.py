from __future__ import annotations

import copy
import importlib
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_zit_stop_after_handoff"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

guard_module = types.ModuleType(f"{TEST_PACKAGE}.engine.content_guard")
guard_module.GUARD_VERSION = "test-guard-v1"
guard_module.get_content_guard = lambda: None


class _ContentGuardBlocked(RuntimeError):
    pass


guard_module.ContentGuardBlocked = _ContentGuardBlocked
sys.modules[guard_module.__name__] = guard_module

tensor_utils = types.ModuleType(f"{TEST_PACKAGE}.utils.tensor_utils")
tensor_utils.tensor_to_uint8_rgb = lambda value: value
sys.modules[tensor_utils.__name__] = tensor_utils

visual_module = types.ModuleType(f"{TEST_PACKAGE}.pipe.cmk_visual")
visual_module.empty_visual = lambda: {"version": 1, "providers": []}
sys.modules[visual_module.__name__] = visual_module

result_state = types.ModuleType(f"{TEST_PACKAGE}.pipe.cmk_prompt_result_state")
result_state.IncompleteResultError = RuntimeError
result_state.UnsafeSnapshotTypeError = RuntimeError
result_state.PromptStateUnavailableError = RuntimeError
result_state.MISSING = object()
result_state._results = {}
result_state.get_result = lambda key: result_state._results.get(
    key, result_state.MISSING
)
result_state.make_key = lambda **values: values["fingerprint"]
result_state.publish_result = lambda key, value: result_state._results.__setitem__(
    key, value
)
sys.modules[result_state.__name__] = result_state


def _lookup(prompt, node_id):
    return prompt.get(node_id) or prompt.get(str(node_id))


def _is_link(prompt, value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
        and isinstance(_lookup(prompt, value[0]), dict)
    )


refiner = types.ModuleType(f"{TEST_PACKAGE}.pipe.cmk_refiner_boundary_cache")
refiner._is_link = _is_link
refiner._resolve_current_node = lambda prompt, unique_id: (
    (unique_id, _lookup(prompt, unique_id))
    if isinstance(_lookup(prompt, unique_id), dict)
    else (None, None)
)
refiner._canonical_node = lambda _prompt, node_id, _memo, _stack: {"node": str(node_id)}
sys.modules[refiner.__name__] = refiner

distribution = importlib.import_module(f"{TEST_PACKAGE}.cmk_distribution")
persistent = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_persistent_cache")
bridge = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_hybrid_bridge")


def _prompt(
    *,
    stop=True,
    seed=7,
    zit_lora="portrait-zit.safetensors",
    sdxl_lora="layout-sdxl.safetensors",
    checkpoint="base-sdxl.safetensors",
    balance=2,
):
    return {
        "1": {
            "class_type": "CMKPipeCreateImage",
            "inputs": {
                "model_family": "Hybrid",
                "stop_after_handoff": stop,
                "content_guard_enabled": False,
                "GLOBAL PROMPT POS": "a portrait in a garden",
                "GLOBAL PROMPT NEG": "blur",
                "seed": seed,
                "CHECKPOINT SDXL": checkpoint,
                "VAE SDXL": "vae-sdxl.safetensors",
                "LoRA SDXL": sdxl_lora,
                "HYBRID BALANCE": balance,
                "LoRA ZIT": zit_lora,
                "live_preview_zit": True,
                "unload_models_after_use": True,
            },
        },
        "2": {
            "class_type": "CMKKSamplerPipe",
            "inputs": {"SAMPLER": ["1", 0]},
        },
        "3": {
            "class_type": "CMKFamilyBranchGateSDXLSampled",
            "inputs": {
                "MODEL": ["1", 0],
                "PROCESS": ["1", 0],
                "SAMPLED": ["2", 0],
                "LOG": ["1", 2],
                "VISUAL": ["2", 1],
            },
        },
        "4": {
            "class_type": "CMKHybridZITInputPipe",
            "inputs": {
                "PROCESS ZIT": ["1", 1],
                "SAMPLED SDXL": ["3", 1],
                "LOG SDXL": ["3", 2],
                "VISUAL SDXL": ["3", 5],
            },
        },
        "5": {
            "class_type": "CMKSamplerPrepareZImageTurboPipe",
            "inputs": {"IMAGE": ["4", 0], "steps": 9},
        },
    }


def _process(*, stop=True):
    return {
        "model_family": "sdxl",
        "hybrid_mode": True,
        "family_active": True,
        "stop_after_handoff": stop,
        "content_guard_enabled": False,
        "hybrid_sdxl_handoff": 90,
        "hybrid_zit_denoise": 0.20,
        "hybrid_inpaint_mode": False,
        "sdxl_width": 1152,
        "sdxl_height": 832,
    }


MODEL = {"ckpt_name": "base-sdxl.safetensors", "vae_selection_name": "vae-sdxl.safetensors"}
DIAGNOSTIC = {"type": "CMK_DIAGNOSTIC", "title": "Sampler Prepare"}


def _persist(process, image, log, visual, prompt, *, inspect=True, diagnostic=DIAGNOSTIC, model=MODEL):
    """The SDXL gate's call for a computed Hybrid first pass (v2 contract)."""
    return bridge.persist_hybrid_sdxl_boundary(
        process, image, log, visual, diagnostic, model, prompt, "3", inspect=inspect,
    )


class _Guard:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def inspect_content(self, _image, _context):
        self.calls += 1
        if self.error:
            raise self.error


class ZITStopAfterHandoffTests(unittest.TestCase):
    def setUp(self):
        result_state._results.clear()
        self.temporary = tempfile.TemporaryDirectory()
        self.temp_root = Path(self.temporary.name)
        self.cache_patch = patch.object(
            persistent,
            "cache_directory",
            lambda scope: self._cache_directory(scope),
        )
        self.cache_patch.start()

    def tearDown(self):
        self.cache_patch.stop()
        persistent.reset_persistent_fingerprint_memo()
        self.temporary.cleanup()

    def _cache_directory(self, scope):
        path = self.temp_root / "cmk" / scope
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_hybrid_stop_is_active_without_instantid(self):
        self.assertTrue(distribution.stop_after_handoff_active(_process()))
        self.assertFalse(
            distribution.stop_after_handoff_active(_process(stop=False))
        )
        self.assertTrue(distribution.stop_after_handoff_prompt_state(_prompt(), "4"))

    def test_hybrid_ui_exposes_stop_after_hybrid_balance(self):
        source = (
            PACKAGE_ROOT / "web/js/cmk_flow_start_guidance_v51.js"
        ).read_text(encoding="utf-8")
        self.assertIn('mode.startsWith("hybrid")', source)
        self.assertIn('visibleNames.indexOf("HYBRID BALANCE")', source)

    def test_zit_only_changes_do_not_invalidate_sdxl_boundary(self):
        original = _prompt()
        original_key = bridge._hybrid_boundary_cache_key(original, "4")
        variants = []
        stop_off = copy.deepcopy(original)
        stop_off["1"]["inputs"]["stop_after_handoff"] = False
        variants.append(stop_off)
        zit_lora = copy.deepcopy(original)
        zit_lora["1"]["inputs"]["LoRA ZIT"] = "changed-zit.safetensors"
        variants.append(zit_lora)
        zit_steps = copy.deepcopy(original)
        zit_steps["5"]["inputs"]["steps"] = 30
        variants.append(zit_steps)
        for value in variants:
            self.assertEqual(
                original_key,
                bridge._hybrid_boundary_cache_key(value, "4"),
            )

    def test_sdxl_changes_invalidate_boundary(self):
        original_key = bridge._hybrid_boundary_cache_key(_prompt(), "4")
        for changed in (
            _prompt(seed=8),
            _prompt(balance=4),
            _prompt(sdxl_lora="changed-sdxl.safetensors"),
            _prompt(checkpoint="changed-sdxl.safetensors"),
        ):
            self.assertNotEqual(
                original_key,
                bridge._hybrid_boundary_cache_key(changed, "4"),
            )

    def test_stop_on_to_off_restores_exact_image_without_sdxl_inputs(self):
        first_prompt = _prompt(stop=True)
        second_prompt = _prompt(stop=False)
        image = torch.rand((1, 8, 10, 3), dtype=torch.float32)
        log = {"blocks": ["sdxl"]}
        visual = {"version": 1, "providers": []}
        key = _persist(_process(stop=True), image, log, visual, first_prompt)
        self.assertIsNotNone(key)

        persistent.reset_persistent_fingerprint_memo()
        restored = bridge.load_hybrid_sdxl_boundary(
            _process(stop=False), second_prompt, "4"
        )
        self.assertIsNotNone(restored)
        torch.testing.assert_close(restored[0], image)
        self.assertEqual(restored[2], visual)

        lazy = bridge.CMKHybridZITInputPipe().check_lazy_status(
            **{
                "PROCESS ZIT": dict(_process(stop=False), model_family="z_image_turbo"),
                "prompt": second_prompt,
                "unique_id": "4",
            }
        )
        self.assertEqual(lazy, [])
        selected = bridge.CMKHybridZITInputPipe.select(
            **{
                "PROCESS ZIT": dict(_process(stop=False), model_family="z_image_turbo"),
                "prompt": second_prompt,
                "unique_id": "4",
            }
        )
        torch.testing.assert_close(selected[0], image)
        state_key = bridge._hybrid_handoff_state_key(second_prompt, "4")
        published = result_state.get_result(state_key)
        self.assertIsInstance(published, dict)
        torch.testing.assert_close(published["image"], image)
        self.assertEqual(published["visual"], visual)
        self.assertEqual(
            published["metadata"],
            {
                "handoff_percent": 90,
                "zit_denoise": 0.20,
                "hybrid_inpaint_mode": False,
            },
        )

    def test_live_and_persistent_paths_publish_equivalent_handoff_contracts(self):
        image = torch.rand((1, 4, 6, 3), dtype=torch.float32)
        visual = {"version": 1, "providers": []}
        process = dict(
            _process(stop=False),
            model_family="z_image_turbo",
            unload_models_after_use=False,
        )

        live_prompt = _prompt(stop=False, seed=19)
        live = bridge.CMKHybridZITInputPipe.select(
            **{
                "PROCESS ZIT": process,
                "SAMPLED SDXL": {"image_1st_pass": image},
                "LOG SDXL": {"blocks": ["live"]},
                "VISUAL SDXL": visual,
                "prompt": live_prompt,
                "unique_id": "4",
            }
        )
        live_key = bridge._hybrid_handoff_state_key(live_prompt, "4")
        live_state = result_state.get_result(live_key)
        self.assertIsInstance(live_state, dict)

        result_state._results.clear()
        stop_prompt = _prompt(stop=True, seed=23)
        resume_prompt = _prompt(stop=False, seed=23)
        _persist(_process(stop=True), image, {"blocks": ["cached"]}, visual, stop_prompt)
        persistent.reset_persistent_fingerprint_memo()
        resumed = bridge.CMKHybridZITInputPipe.select(
            **{
                "PROCESS ZIT": process,
                "prompt": resume_prompt,
                "unique_id": "4",
            }
        )
        resumed_key = bridge._hybrid_handoff_state_key(resume_prompt, "4")
        resumed_state = result_state.get_result(resumed_key)
        self.assertIsInstance(resumed_state, dict)
        self.assertEqual(resumed_state["metadata"], live_state["metadata"])
        torch.testing.assert_close(resumed_state["image"], live_state["image"])
        self.assertEqual(resumed_state["visual"], live_state["visual"])
        torch.testing.assert_close(resumed[0], image)
        torch.testing.assert_close(live[0], image)

    def test_cache_miss_requests_sdxl_once_without_a_lazy_loop(self):
        prompt = _prompt(stop=False)
        process = dict(_process(stop=False), model_family="z_image_turbo")
        node = bridge.CMKHybridZITInputPipe()
        self.assertEqual(
            node.check_lazy_status(
                **{"PROCESS ZIT": process, "prompt": prompt, "unique_id": "4"}
            ),
            ["SAMPLED SDXL"],
        )
        ready = {
            "PROCESS ZIT": process,
            "SAMPLED SDXL": {"image_1st_pass": torch.zeros((1, 2, 2, 3))},
            "LOG SDXL": {},
            "VISUAL SDXL": {"providers": []},
            "prompt": prompt,
            "unique_id": "4",
        }
        self.assertEqual(node.check_lazy_status(**ready), [])

    def test_content_guard_must_approve_before_publication(self):
        prompt = _prompt()
        prompt["1"]["inputs"]["content_guard_enabled"] = True
        process = dict(_process(), content_guard_enabled=True)

        passing = _Guard()
        with patch.object(bridge, "get_content_guard", return_value=passing):
            key = _persist(process, torch.zeros((2, 2, 2, 3)), {}, {}, prompt)
        self.assertEqual(passing.calls, 2)
        self.assertTrue(persistent.pickle_available(bridge._HYBRID_BOUNDARY_SCOPE, key))

        persistent.clear_persistent_disk_scopes(self.temp_root)
        blocking = _Guard(RuntimeError("blocked"))
        with patch.object(bridge, "get_content_guard", return_value=blocking):
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                _persist(process, torch.zeros((1, 2, 2, 3)), {}, {}, prompt)
        self.assertFalse(persistent.pickle_available(bridge._HYBRID_BOUNDARY_SCOPE, key))

    def test_registered_cache_cleanup_removes_boundary(self):
        key = _persist(_process(), torch.zeros((1, 2, 2, 3)), {}, {}, _prompt())
        self.assertTrue(persistent.pickle_available(bridge._HYBRID_BOUNDARY_SCOPE, key))
        inventory = persistent.clear_persistent_disk_scopes(self.temp_root)
        self.assertGreaterEqual(inventory[bridge._HYBRID_BOUNDARY_SCOPE]["entries"], 1)
        self.assertFalse(persistent.pickle_available(bridge._HYBRID_BOUNDARY_SCOPE, key))

    def test_payload_contains_no_model_or_latent(self):
        prompt = _prompt()
        key = _persist(_process(), torch.zeros((1, 2, 2, 3)), {}, {"providers": []}, prompt)
        envelope = persistent.load_pickle(bridge._HYBRID_BOUNDARY_SCOPE, key)
        self.assertEqual(envelope["schema"], "cmk_hybrid_sdxl_image_boundary_v2")
        payload = pickle.loads(envelope["payload"])
        self.assertEqual(
            set(payload),
            {"contract", "image", "log", "visual", "diagnostic", "model_identity", "metadata"},
        )
        # Only the checkpoint/VAE identity strings, never model objects or latents.
        self.assertEqual(payload["model_identity"],
                         {"checkpoint": "base-sdxl.safetensors", "vae": "vae-sdxl.safetensors"})
        self.assertEqual(payload["diagnostic"], DIAGNOSTIC)
        self.assertNotIn("latent", repr(payload).lower())
        tensors = [name for name, value in payload.items() if isinstance(value, torch.Tensor)]
        self.assertEqual(tensors, ["image"])

    def test_corrupt_boundary_is_rejected_and_removed(self):
        prompt = _prompt()
        key = _persist(_process(), torch.zeros((1, 2, 2, 3)), {}, {"providers": []}, prompt)
        persistent.cache_path(bridge._HYBRID_BOUNDARY_SCOPE, key, ".pkl").write_bytes(
            b"corrupt"
        )
        persistent.reset_persistent_fingerprint_memo()
        self.assertIsNone(
            bridge.load_hybrid_sdxl_boundary(_process(stop=False), _prompt(stop=False), "4")
        )
        self.assertFalse(
            persistent.pickle_available(bridge._HYBRID_BOUNDARY_SCOPE, key)
        )

    def test_hybrid_boundary_has_its_own_scope(self):
        self.assertEqual(bridge._HYBRID_BOUNDARY_SCOPE, "sdxl_hybrid_boundary")
        self.assertIn("sdxl_hybrid_boundary", persistent.PERSISTENT_DISK_SCOPES)
        self.assertEqual(bridge._HYBRID_BOUNDARY_MAX_ENTRIES, 8)

    def _guarded(self, stop):
        prompt = _prompt(stop=stop)
        prompt["1"]["inputs"]["content_guard_enabled"] = True
        return prompt, dict(_process(stop=stop), content_guard_enabled=True)

    def test_not_inspected_entry_is_inspected_once_before_use(self):
        bridge._INSPECTED_PAYLOADS.clear()
        prompt, process = self._guarded(stop=False)
        guard = _Guard()
        with patch.object(bridge, "get_content_guard", return_value=guard):
            key = _persist(process, torch.zeros((1, 2, 2, 3)), {}, {}, prompt, inspect=False)
            self.assertEqual(guard.calls, 0)
            envelope = persistent.load_pickle(bridge._HYBRID_BOUNDARY_SCOPE, key)
            self.assertEqual(pickle.loads(envelope["payload"])["metadata"]["content_guard_handoff"], "NOT INSPECTED")
            persistent.reset_persistent_fingerprint_memo()
            self.assertIsNotNone(bridge.load_hybrid_sdxl_boundary(process, prompt, "4"))
            self.assertIsNotNone(bridge.load_hybrid_sdxl_boundary(process, prompt, "4"))
        self.assertEqual(guard.calls, 1)

    def test_rejected_entry_is_removed_and_not_returned(self):
        bridge._INSPECTED_PAYLOADS.clear()
        prompt, process = self._guarded(stop=False)
        key = _persist(process, torch.zeros((1, 2, 2, 3)), {}, {}, prompt, inspect=False)
        blocking = _Guard(guard_module.ContentGuardBlocked("CG_EXPLICIT_CONTENT"))
        with patch.object(bridge, "get_content_guard", return_value=blocking):
            persistent.reset_persistent_fingerprint_memo()
            self.assertIsNone(bridge.load_hybrid_sdxl_boundary(process, prompt, "4"))
            self.assertIsNone(bridge.restore_hybrid_sdxl_first_pass(process, prompt, "3"))
        self.assertEqual(blocking.calls, 1)
        self.assertFalse(persistent.pickle_available(bridge._HYBRID_BOUNDARY_SCOPE, key))

    def test_v1_entry_is_rejected_and_removed(self):
        prompt = _prompt()
        key = _persist(_process(), torch.zeros((1, 2, 2, 3)), {}, {}, prompt)
        payload = pickle.dumps({"image": None}, protocol=pickle.HIGHEST_PROTOCOL)
        import hashlib
        persistent.save_pickle(bridge._HYBRID_BOUNDARY_SCOPE, key, {
            "schema": "cmk_hybrid_sdxl_image_boundary_v1",
            "sha256": hashlib.sha256(payload).hexdigest(), "payload": payload,
        })
        persistent.reset_persistent_fingerprint_memo()
        self.assertIsNone(bridge.load_hybrid_sdxl_boundary(_process(stop=False), _prompt(stop=False), "4"))
        self.assertFalse(persistent.pickle_available(bridge._HYBRID_BOUNDARY_SCOPE, key))

    def test_zit_side_never_reads_the_entry_of_its_own_prompt(self):
        prompt = _prompt(stop=False)
        with patch.object(bridge, "_active_prompt_id", return_value="prompt-a"):
            _persist(_process(stop=False), torch.zeros((1, 2, 2, 3)), {}, {}, prompt, inspect=False)
            persistent.reset_persistent_fingerprint_memo()
            self.assertIsNone(bridge.load_hybrid_sdxl_boundary(_process(stop=False), prompt, "4"))
        with patch.object(bridge, "_active_prompt_id", return_value="prompt-b"):
            persistent.reset_persistent_fingerprint_memo()
            self.assertIsNotNone(bridge.load_hybrid_sdxl_boundary(_process(stop=False), prompt, "4"))

    def test_sdxl_side_restore_returns_the_complete_first_pass(self):
        image = torch.rand((1, 4, 6, 3), dtype=torch.float32)
        log = {"blocks": [{"title": "Sampler Prepare", "lines": ["seed=7"], "order": 10, "enabled": True}]}
        visual = {"version": 1, "providers": [{"provider_id": "p"}]}
        _persist(_process(stop=False), image, log, visual, _prompt(stop=False), inspect=False)
        persistent.reset_persistent_fingerprint_memo()
        restored = bridge.restore_hybrid_sdxl_first_pass(_process(stop=True), _prompt(stop=True), "3")
        torch.testing.assert_close(restored["image"], image)
        self.assertEqual(restored["visual"], visual)
        self.assertEqual(restored["diagnostic"], DIAGNOSTIC)
        self.assertEqual((restored["checkpoint"], restored["vae"]), ("base-sdxl.safetensors", "vae-sdxl.safetensors"))
        self.assertEqual(restored["log"]["blocks"][0], log["blocks"][0])
        self.assertEqual(restored["log"]["blocks"][-1]["title"], "SDXL First Pass · Hybrid Boundary")

    def test_sdxl_side_restore_needs_the_model_identity(self):
        _persist(_process(), torch.zeros((1, 2, 2, 3)), {}, {}, _prompt(), model={})
        persistent.reset_persistent_fingerprint_memo()
        self.assertIsNone(bridge.restore_hybrid_sdxl_first_pass(_process(), _prompt(), "3"))


if __name__ == "__main__":
    unittest.main()
