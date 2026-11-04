from __future__ import annotations

import copy
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_stop_after_handoff"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

common = types.ModuleType(f"{TEST_PACKAGE}.cmk_common")
common.SAMPLERS = ["euler"]
common.SCHEDULERS = ["simple"]
sys.modules[common.__name__] = common

guard = types.ModuleType(f"{TEST_PACKAGE}.engine.content_guard")
guard.ContentGuardBlocked = type("ContentGuardBlocked", (RuntimeError,), {})
guard.GUARD_VERSION = "test-guard-v1"
guard.cleanup_content_guard_abort = lambda *args, **kwargs: None
guard.get_content_guard = lambda: None
sys.modules[guard.__name__] = guard

final_preview = types.ModuleType(f"{TEST_PACKAGE}.pipe.cmk_final_preview")
final_preview.send_final_preview = lambda *args, **kwargs: None
sys.modules[final_preview.__name__] = final_preview

visual = types.ModuleType(f"{TEST_PACKAGE}.pipe.cmk_visual")
visual.empty_visual = lambda: {"version": 1, "providers": []}
visual.register_provider = lambda value, **kwargs: value
sys.modules[visual.__name__] = visual

tensor_utils = types.ModuleType(f"{TEST_PACKAGE}.utils.tensor_utils")
tensor_utils.tensor_to_uint8_rgb = lambda value: value
sys.modules[tensor_utils.__name__] = tensor_utils

timing = types.ModuleType(f"{TEST_PACKAGE}.utils.cmk_timing")
class _NoopTimer:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
timing.cmk_timed = lambda *args, **kwargs: _NoopTimer()
timing.cmk_timed_call = lambda *_args, **_kwargs: (lambda function: function)
sys.modules[timing.__name__] = timing

warnings_module = types.ModuleType(f"{TEST_PACKAGE}.utils.cmk_sampling_warnings")
warnings_module.ignore_torchsde_boundary_rounding = lambda: _NoopTimer()
sys.modules[warnings_module.__name__] = warnings_module

distribution = importlib.import_module(f"{TEST_PACKAGE}.cmk_distribution")
persistent = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_persistent_cache")
sampler = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_pipe_sampler")
boundaries = importlib.import_module(
    f"{TEST_PACKAGE}.pipe.cmk_module_boundary_cache"
)


def _prompt(stop=False, seed=7, identity_strength=1.0):
    return {
        "1": {
            "class_type": "CMKPipeCreateImage",
            "inputs": {
                "model_family": "SDXL",
                "instantid_enabled": True,
                "instantid_mode": "HYBRID",
                "instantid_handoff_percent": 70,
                "stop_after_handoff": stop,
                "content_guard_enabled": True,
                "GLOBAL PROMPT POS": "two people in a field",
                "seed": seed,
            },
        },
        "10": {
            "class_type": "CMKSamplerPrepareSDXLPipe",
            "inputs": {"PROCESS": ["1", 0], "steps_1st_pass": 25},
        },
        "11": {
            "class_type": "CMKKSamplerPipe",
            "inputs": {"SAMPLER": ["10", 0]},
        },
        "15": {
            "class_type": "CMKInstantIDSamplerSDXLPipe",
            "inputs": {"SAMPLED": ["11", 0], "identity_strength": identity_strength},
        },
    }


def _prompt_variant(*, inpaint=False, controlnet=False):
    prompt = _prompt(stop=True)
    prompt["1"]["inputs"].update({
        "INPAINT_MODE": "Inpaint" if inpaint else "Text2Image",
        "process_mode": "Replace Object" if inpaint else "Custom",
        "source_image": "input-a.png" if inpaint else "",
    })
    if controlnet:
        prompt["5"] = {
            "class_type": "CMKCombinedControlNetPreparePipe",
            "inputs": {
                "PROCESS SDXL": ["1", 0],
                "REFERENCE IMAGE": "pose-a.png",
                "STRENGTH": 0.75,
            },
        }
        prompt["10"]["inputs"]["PROCESS"] = ["5", 0]
    return prompt


def _pipe():
    return {
        "model_family": "sdxl",
        "instantid_enabled": True,
        "instantid_mode": "HYBRID",
        "instantid_end_at_step": 18,
        "instantid_handoff_percent": 70,
        "steps_1st_pass": 25,
        "checkpoint_name": "base.safetensors",
        "vae_name": "base.vae",
        "seed": 7,
        "cfg": 5.0,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1.0,
        "sampling": "lcm",
        "zsnr": False,
        "sdxl_width": 1152,
        "sdxl_height": 832,
        "boolean_inpaint_mode": False,
        "controlnet_applied": False,
        "content_guard_enabled": True,
        "model": object(),
        "model_identity": object(),
        "vae": object(),
        "conditioning_pos": object(),
        "conditioning_neg": object(),
        "samples": {"samples": [1, 2, 3]},
        "latent": {"samples": [1, 2, 3]},
        "latent_image": {"samples": [1, 2, 3]},
        "latent_1st_pass": {"samples": [1, 2, 3]},
        "instantid_keypoints_latent": {"samples": [4, 5, 6]},
        "instantid_inpaint_bridge": False,
        "content_guard_handoff": "PASS · SDXL HANDOFF",
        "image_1st_pass": [[[0.5]]],
        "controlnet_model_status": "NOT ACTIVE",
        "ksampler_log": "computed",
    }


def _postprocess_chain_prompt(stop):
    return {
        "1": {
            "class_type": "CMKPipeCreateImage",
            "inputs": {
                "model_family": "SDXL",
                "instantid_enabled": True,
                "instantid_mode": "HYBRID",
                "stop_after_handoff": stop,
            },
        },
        "30": {
            "class_type": "CMKPostProcessSetupPipe",
            "inputs": {"PROCESS SDXL": ["1", 0]},
        },
        "42": {
            "class_type": "CMKInstantIDFaceRebuildAdvancedSDXL",
            "inputs": {
                "MODEL": ["30", 0],
                "PROCESS": ["30", 1],
                "IMAGE": ["30", 2],
                "LOG": ["30", 3],
                "FACEREBUILD ENABLE": True,
            },
        },
        "42-boundary": {
            "class_type": "CMKFaceRebuildBoundaryCache",
            "inputs": {
                "MODEL": ["42", 0],
                "IMAGE": ["42", 2],
                "LOG": ["42", 3],
                "diagnostic": ["42", 5],
            },
        },
        "40": {
            "class_type": "CMKFaceSwapImagePipe",
            "inputs": {
                "PROCESS": ["30", 1],
                "IMAGE_TARGET": ["42-boundary", 1],
                "GLOBAL ENABLE": True,
                "ENABLE": True,
            },
        },
        "40-boundary": {
            "class_type": "CMKFaceSwapBoundaryCache",
            "inputs": {
                "MODEL (opt)": ["40", 0],
                "IMAGE": ["40", 1],
                "LOG": ["40", 2],
                "diagnostic": ["40", 3],
            },
        },
        "visualizer": {
            "class_type": "CMKVisualizer",
            "inputs": {
                "MODEL (opt)": ["40-boundary", 0],
                "PROCESS": ["30", 1],
                "IMAGE": ["40-boundary", 1],
                "LOG": ["40-boundary", 2],
            },
        },
    }


class StopAfterHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.temp_root = Path(self.temporary.name)
        self.cache_patch = patch.object(
            persistent,
            "cache_directory",
            lambda scope: self._cache_directory(scope),
        )
        self.cache_patch.start()
        self.boundary_cache_patch = patch.object(
            boundaries,
            "cache_directory",
            lambda scope: self._cache_directory(scope),
        )
        self.boundary_cache_patch.start()

    def tearDown(self):
        self.boundary_cache_patch.stop()
        self.cache_patch.stop()
        persistent.reset_persistent_fingerprint_memo()
        self.temporary.cleanup()

    def _cache_directory(self, scope):
        path = self.temp_root / "cmk" / scope
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_default_and_supported_modes(self):
        self.assertFalse(distribution.stop_after_handoff_active({}))
        base = {
            "stop_after_handoff": True,
            "instantid_enabled": True,
            "instantid_mode": "HYBRID",
            "model_family": "sdxl",
        }
        self.assertTrue(distribution.stop_after_handoff_active(base))
        for mode in ("ZERO-PASS", "REFERENCE-LATENT"):
            value = dict(base, instantid_mode=mode)
            self.assertFalse(distribution.stop_after_handoff_active(value))

    def test_stop_toggle_and_downstream_identity_do_not_change_fingerprint(self):
        before = _prompt(stop=False, identity_strength=0.5)
        after = _prompt(stop=True, identity_strength=1.5)
        key_before = sampler._sdxl_handoff_cache_key(before, "11")
        persistent.reset_persistent_fingerprint_memo()
        key_after = sampler._sdxl_handoff_cache_key(after, "11")
        self.assertEqual(key_before, key_after)

    def test_relevant_upstream_change_invalidates_fingerprint(self):
        before = _prompt(stop=True, seed=7)
        after = _prompt(stop=True, seed=8)
        key_before = sampler._sdxl_handoff_cache_key(before, "11")
        persistent.reset_persistent_fingerprint_memo()
        key_after = sampler._sdxl_handoff_cache_key(after, "11")
        self.assertNotEqual(key_before, key_after)

    def test_content_guard_configuration_invalidates_approval(self):
        guarded = _prompt(stop=True)
        bypassed = _prompt(stop=True)
        bypassed["1"]["inputs"]["content_guard_enabled"] = False
        key_guarded = sampler._sdxl_handoff_cache_key(guarded, "11")
        persistent.reset_persistent_fingerprint_memo()
        key_bypassed = sampler._sdxl_handoff_cache_key(bypassed, "11")
        self.assertNotEqual(key_guarded, key_bypassed)

    def test_text2image_inpaint_and_controlnet_have_distinct_contracts(self):
        keys = []
        for prompt in (
            _prompt_variant(),
            _prompt_variant(inpaint=True),
            _prompt_variant(controlnet=True),
        ):
            persistent.reset_persistent_fingerprint_memo()
            keys.append(sampler._sdxl_handoff_cache_key(prompt, "11"))
        self.assertEqual(len(set(keys)), 3)

    def test_two_queue_runs_restore_without_sampling_resources_in_payload(self):
        first_prompt = _prompt(stop=True)
        second_prompt = copy.deepcopy(first_prompt)
        second_prompt["1"]["inputs"]["stop_after_handoff"] = False
        first_pipe = _pipe()
        current_model = object()
        second_pipe = _pipe()
        second_pipe["model"] = current_model

        key = sampler._persist_sdxl_handoff(
            first_pipe, {"providers": [{"channels": {"result": [[[0.5]]]}}]},
            first_prompt, "11",
        )
        self.assertIsNotNone(key)
        persistent.reset_persistent_fingerprint_memo()
        restored, restored_visual = sampler._load_sdxl_handoff(
            second_pipe, {"providers": []}, second_prompt, "11",
        )

        self.assertIs(restored["model"], current_model)
        self.assertEqual(restored["samples"], first_pipe["samples"])
        self.assertEqual(restored["instantid_end_at_step"], 18)
        self.assertTrue(restored["sdxl_handoff_cache_hit"])
        self.assertTrue(restored_visual["providers"])

    def test_sampler_returns_persistent_hit_before_any_sampling_import(self):
        first_prompt = _prompt(stop=True)
        second_prompt = _prompt(stop=False)
        sampler._persist_sdxl_handoff(
            _pipe(), {"providers": []}, first_prompt, "11"
        )
        persistent.reset_persistent_fingerprint_memo()

        restored, _visual = sampler.CMKKSamplerPipe()._run(
            _pipe(), {"providers": []}, second_prompt, "11"
        )
        self.assertTrue(restored["sdxl_handoff_cache_hit"])
        self.assertEqual(restored["instantid_end_at_step"], 18)

    def test_stop_blocks_final_postprocess_boundary_before_upstream_requests(self):
        prompt = _postprocess_chain_prompt(True)
        node = boundaries.CMKFaceSwapBoundaryCache()

        with patch.object(boundaries, "_faceswap_disabled_state") as disabled_state:
            self.assertEqual(
                node.check_lazy_status(
                    prompt=prompt,
                    unique_id="40-boundary",
                ),
                [],
            )
            disabled_state.assert_not_called()

        blocked = node.boundary(
            prompt=prompt,
            unique_id="40-boundary",
        )
        # Four result outputs plus the VISUAL output added by H-3 / F-03.
        self.assertEqual(len(blocked), 4 + 1)
        self.assertEqual(node.RETURN_NAMES[-1], "VISUAL")
        self.assertTrue(
            all(value.__class__.__name__ == "ExecutionBlocker" for value in blocked)
        )

    def test_stop_blocks_each_postprocess_module_boundary_without_lazy_loop(self):
        prompt = _postprocess_chain_prompt(True)
        # (node, unique_id, result outputs, publishes VISUAL since H-3 / F-03)
        cases = (
            # Detailer: MODEL/IMAGE/LOG plus the boundary-owned diagnostic (K-5).
            (boundaries.CMKDetailerBoundaryCache(), "40-boundary", 4, True),
            (boundaries.CMKFaceRebuildBoundaryCache(), "42-boundary", 4, True),
            # FaceProcess: MODEL/IMAGE/LOG plus the boundary-owned diagnostic (H-5).
            (boundaries.CMKFaceBoundaryCache(), "40-boundary", 4, True),
            (boundaries.CMKFaceSwapBoundaryCache(), "40-boundary", 4, True),
            (boundaries.CMKUpscaleSaveBoundaryCache(), "40-boundary", 4, False),
        )
        for node, unique_id, output_count, publishes_visual in cases:
            with self.subTest(node=node.__class__.__name__):
                first = node.check_lazy_status(prompt=prompt, unique_id=unique_id)
                second = node.check_lazy_status(prompt=prompt, unique_id=unique_id)
                self.assertEqual(first, [])
                self.assertEqual(second, [])
                blocked = node.boundary(prompt=prompt, unique_id=unique_id)
                self.assertEqual(len(blocked), output_count + int(publishes_visual))
                self.assertEqual(len(node.RETURN_TYPES), len(blocked))
                self.assertEqual("VISUAL" in node.RETURN_NAMES, publishes_visual)
                self.assertTrue(
                    all(
                        value.__class__.__name__ == "ExecutionBlocker"
                        for value in blocked
                    )
                )

    def test_stop_off_preserves_faceswap_boundary_upstream_requests(self):
        prompt = _postprocess_chain_prompt(False)
        node = boundaries.CMKFaceSwapBoundaryCache()
        requested = node.check_lazy_status(
            prompt=prompt,
            unique_id="40-boundary",
        )
        self.assertEqual(
            requested,
            ["IMAGE", "LOG", "diagnostic"],
        )
        self.assertEqual(
            node.check_lazy_status(
                IMAGE=object(),
                LOG={},
                diagnostic={},
                prompt=prompt,
                unique_id="40-boundary",
            ),
            ["MODEL (opt)"],
        )

    def test_guard_unapproved_handoff_is_not_persisted(self):
        pipe = _pipe()
        pipe.pop("content_guard_handoff")
        self.assertIsNone(
            sampler._persist_sdxl_handoff(
                pipe, {"providers": []}, _prompt(stop=True), "11"
            )
        )

    def test_contract_mismatch_and_corruption_are_misses(self):
        prompt = _prompt(stop=True)
        key = sampler._persist_sdxl_handoff(
            _pipe(), {"providers": []}, prompt, "11"
        )
        mismatch = _pipe()
        mismatch["instantid_end_at_step"] = 15
        self.assertIsNone(
            sampler._load_sdxl_handoff(
                mismatch, {"providers": []}, prompt, "11"
            )
        )

        key = sampler._persist_sdxl_handoff(
            _pipe(), {"providers": []}, prompt, "11"
        )
        persistent.cache_path(
            sampler._SDXL_HANDOFF_CACHE_SCOPE, key, ".pkl"
        ).write_bytes(b"corrupt")
        self.assertIsNone(
            sampler._load_sdxl_handoff(
                _pipe(), {"providers": []}, prompt, "11"
            )
        )
        self.assertFalse(
            persistent.pickle_available(sampler._SDXL_HANDOFF_CACHE_SCOPE, key)
        )

    def test_source_contracts_keep_preview_path_lazy(self):
        start_source = (PACKAGE_ROOT / "pipe/cmk_pipe_image.py").read_text()
        family_source = (PACKAGE_ROOT / "pipe/cmk_family_result.py").read_text()
        refiner_source = (PACKAGE_ROOT / "pipe/cmk_refiner_boundary_cache.py").read_text()
        visual_source = (PACKAGE_ROOT / "pipe/cmk_visual.py").read_text()
        frontend_source = (
            PACKAGE_ROOT / "web/js/cmk_flow_start_guidance_v51.js"
        ).read_text()
        sampler_s = (
            PACKAGE_ROOT / "pipe/instantid/cmk_instantid_sampler.py"
        ).read_text()
        sampler_a = (
            PACKAGE_ROOT / "pipe/instantid/cmk_instantid_regional_advanced.py"
        ).read_text()
        face_detailer_source = (
            PACKAGE_ROOT / "nodes/instantid_face_detailer.py"
        ).read_text()
        self.assertIn('"stop_after_handoff": (', start_source)
        self.assertIn('"default": False', start_source)
        self.assertIn('"stop_after_handoff": stop_after_handoff', start_source)
        self.assertIn('"stop_after_handoff": "STOP AFTER HANDOFF"', frontend_source)
        self.assertIn("stop_after_handoff_prompt_state(\n            prompt, unique_id", family_source)
        self.assertIn('str(flag) == "instantid_enabled" and stop_after_handoff_active(PROCESS)', family_source)
        self.assertIn("_upstream_sdxl_first_pass_hit(prompt, unique_id)", refiner_source)
        self.assertNotIn("CMKRefinerPrepareSDXLPipe", family_source[family_source.index("class CMKFamilyBranchGateSDXL"):family_source.index("class CMKFamilyBranchGateZImage")])
        self.assertIn("save_enabled = bool(kwargs.get(\"SAVE ENABLED\", True)) and not preview_only", visual_source)
        self.assertIn("INSTANTID · RESUMED FROM HANDOFF", sampler_s)
        self.assertIn("INSTANTID · RESUMED FROM HANDOFF", sampler_a)
        self.assertIn(
            'raise ValueError("CMK InstantID Face Detailer requires the PostProcess Boundary")',
            face_detailer_source,
        )


if __name__ == "__main__":
    unittest.main()
