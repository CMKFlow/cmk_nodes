from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_instantid_completion_contract"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)
family_result = importlib.import_module(
    f"{TEST_PACKAGE}.pipe.cmk_family_result"
)
hybrid_bridge = importlib.import_module(
    f"{TEST_PACKAGE}.pipe.cmk_hybrid_bridge"
)


def _node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


def _instantid_prompt(sampler_class, instantid_mode):
    prompt = {
        "source": _node(
            "CMKPipeCreateImage",
            model_family="SDXL",
            instantid_enabled=True,
            instantid_mode=instantid_mode,
        ),
        "sampled_gate": _node(
            "CMKFamilyBranchGateSDXLSampled",
            PROCESS=["source", 0],
        ),
        "enable_flag": _node(
            "CMKProcessEnableFlag",
            PROCESS=["source", 0],
            flag="instantid_enabled",
        ),
        "sampler": _node(
            sampler_class,
            PROCESS=["source", 0],
            SAMPLED=["sampled_gate", 1],
        ),
        "instantid_boundary": _node(
            "CMKInstantIDBoundary",
            MODEL=["sampler", 0],
            SAMPLED=["sampler", 2],
        ),
        "instantid_gate": _node(
            "CMKSamplerBypassGate",
            **{
                "ENABLE": ["enable_flag", 0],
                "MODEL BYPASS": ["sampled_gate", 0],
                "SAMPLED BYPASS": ["sampled_gate", 1],
                "LOG BYPASS": ["sampled_gate", 2],
                "VISUAL BYPASS": ["sampled_gate", 5],
                "MODEL ACTIVE": ["instantid_boundary", 0],
                "SAMPLED ACTIVE": ["instantid_boundary", 1],
                "LOG ACTIVE": ["instantid_boundary", 2],
                "DIAGNOSTIC ACTIVE": ["instantid_boundary", 3],
                "VISUAL ACTIVE": ["sampler", 4],
            },
        ),
        "refiner": _node(
            "CMKRefinerPipe",
            SAMPLED=["instantid_gate", 1],
        ),
        "refiner_gate": _node(
            "CMKFamilyBranchGateSDXL",
            MODEL=["instantid_gate", 0],
            IMAGE=["refiner", 0],
        ),
    }
    prompt["postprocess"] = _node(
        "CMKPostProcessSetupPipe",
        **{
            role: ["refiner_gate", index]
            for index, role in enumerate(
                (
                    "MODEL SDXL",
                    "PROCESS SDXL",
                    "IMAGE SDXL",
                    "LOG SDXL",
                    "VISUAL SDXL",
                )
            )
        },
    )
    return prompt


def _set_instantid_off(prompt):
    prompt["source"]["inputs"]["instantid_enabled"] = False
    return prompt


def _direct_visualizer_prompt(prompt):
    prompt = dict(prompt)
    prompt.pop("postprocess", None)
    prompt["visualizer"] = _node(
        "CMKVisualizer",
        **{
            role: ["refiner_gate", index]
            for index, role in enumerate(
                (
                    "MODEL SDXL",
                    "PROCESS SDXL",
                    "IMAGE SDXL",
                    "LOG SDXL",
                    "VISUAL SDXL",
                )
            )
        },
    )
    return prompt


class InstantIDCompletionContractTests(unittest.TestCase):
    def _lazy_status(self, sampler_class, instantid_mode, **inputs):
        prompt = _instantid_prompt(sampler_class, instantid_mode)
        with patch.object(family_result, "lookup_completion", return_value=None):
            return family_result.CMKPostProcessSetupPipe().check_lazy_status(
                prompt=prompt,
                unique_id="postprocess",
                **inputs,
            )

    def _direct_preflight(self, prompt):
        with patch.object(family_result, "lookup_completion", return_value=None):
            return family_result.visualizer_completion_preflight(
                prompt,
                "visualizer",
            )

    def _sdxl_graph(self, prompt):
        return family_result.CMKPostProcessSetupPipe._sdxl_completion_graph(
            prompt,
            "postprocess",
        )

    def test_zit_hybrid_completion_requires_handoff_and_final_zit_evidence(self):
        process = {
            "hybrid_mode": True,
            "hybrid_sdxl_handoff": 90,
            "hybrid_zit_denoise": 0.20,
            "hybrid_inpaint_mode": False,
        }
        handoff = {
            "image": object(),
            "log": {},
            "visual": {},
            "metadata": {
                "handoff_percent": 90,
                "zit_denoise": 0.20,
                "hybrid_inpaint_mode": False,
            },
        }
        descriptor = family_result.make_released_generation_model_descriptor(
            "z_image_turbo", "zit-model", "zit-vae", "UNLOADED"
        )
        zit_final = {
            "image": object(),
            "log": {},
            "visual": {},
            "diagnostic": {
                "metadata": {
                    "model_family": "z_image_turbo",
                    "zit_model_status": "UNLOADED",
                }
            },
            "released_model_descriptor": descriptor,
        }
        graph = {"handoff_id": "handoff", "zit_gate_id": "zit_gate"}

        with (
            patch.object(
                family_result.CMKPostProcessSetupPipe,
                "_hybrid_graph",
                return_value=(graph, "complete"),
            ),
            patch.object(
                hybrid_bridge,
                "_hybrid_handoff_state_key",
                return_value="handoff-key",
            ),
            patch.object(
                family_result,
                "_zit_final_state_key",
                return_value="zit-key",
            ),
            patch.object(
                family_result,
                "get_result",
                side_effect=lambda key: {
                    "handoff-key": handoff,
                    "zit-key": zit_final,
                }.get(key, family_result.MISSING),
            ),
        ):
            evidence, detail = (
                family_result.CMKPostProcessSetupPipe._hybrid_finalization_evidence(
                    {}, "postprocess", process
                )
            )
        self.assertEqual(evidence["handoff_percent"], 90)
        self.assertEqual(evidence["zit_model_status"], "UNLOADED")
        self.assertEqual(evidence["model_descriptor"], descriptor)
        self.assertIn("complete", detail)

        with (
            patch.object(
                family_result.CMKPostProcessSetupPipe,
                "_hybrid_graph",
                return_value=(graph, "complete"),
            ),
            patch.object(
                hybrid_bridge,
                "_hybrid_handoff_state_key",
                return_value="handoff-key",
            ),
            patch.object(
                family_result,
                "_zit_final_state_key",
                return_value="zit-key",
            ),
            patch.object(
                family_result,
                "get_result",
                side_effect=lambda key: (
                    handoff if key == "handoff-key" else family_result.MISSING
                ),
            ),
        ):
            evidence, detail = (
                family_result.CMKPostProcessSetupPipe._hybrid_finalization_evidence(
                    {}, "postprocess", process
                )
            )
        self.assertIsNone(evidence)
        self.assertIn("ZIT Final Gate has no completed result", detail)

    def test_zit_hybrid_completion_is_stored_after_final_evidence(self):
        descriptor = family_result.make_released_generation_model_descriptor(
            "z_image_turbo", "zit-model", "zit-vae", "UNLOADED"
        )
        process = {
            "model_family": "z_image_turbo",
            "family_active": True,
            "hybrid_mode": True,
        }
        image = object()
        log = {"blocks": []}
        visual = {"providers": []}
        result = ({"model": "zit"}, process, image, log, visual)
        preflight = {
            "contract": family_result._SDXL_ZIT_HYBRID_COMPLETION,
            "key": "hybrid-completion-key",
            "cached": None,
            "detail": "expanded SDXL/ZIT Hybrid graph complete",
        }

        def finish(*args):
            args[-1]["generation_unload"] = "UNLOADED"
            return result

        with (
            patch.object(family_result, "_resolve_completion_preflight", return_value=preflight),
            patch.object(
                family_result.CMKPostProcessSetupPipe,
                "_finish",
                side_effect=finish,
            ),
            patch.object(
                family_result,
                "_completion_finalization_evidence",
                return_value=(
                    {"zit_model_status": "UNLOADED", "model_descriptor": descriptor},
                    "complete",
                ),
            ),
            patch.object(
                family_result,
                "_completion_released_descriptor",
                return_value=descriptor,
            ),
            patch.object(family_result, "_store_completion_result") as store_result,
        ):
            merged = family_result.CMKPostProcessSetupPipe().merge(
                "unused.safetensors",
                "unused.vae",
                True,
                **{
                    "PROCESS SDXL": {"family_active": False},
                    "PROCESS ZIT": process,
                    "MODEL ZIT": result[0],
                    "IMAGE ZIT": image,
                    "LOG ZIT": log,
                    "VISUAL ZIT": visual,
                    "prompt": {},
                    "unique_id": "postprocess",
                },
            )
        self.assertEqual(merged, result)
        store_result.assert_called_once()

    def test_15s_and_15a_off_resolve_as_sdxl_via_node30_and_visualizer(self):
        cases = (
            ("CMKInstantIDSamplerSDXLPipe", "HYBRID"),
            ("CMKInstantIDSamplerSDXLPipe", "ZERO-PASS"),
            ("CMKInstantIDRegionalAdvancedSDXLPipe", "HYBRID"),
            ("CMKInstantIDRegionalAdvancedSDXLPipe", "ZERO-PASS"),
        )
        for sampler, configured_mode in cases:
            with self.subTest(
                sampler=sampler,
                configured_mode=configured_mode,
                endpoint="Node 30",
            ):
                prompt = _set_instantid_off(
                    _instantid_prompt(sampler, configured_mode)
                )
                with patch.object(
                    family_result,
                    "lookup_completion",
                    return_value=None,
                ):
                    status = family_result.CMKPostProcessSetupPipe().check_lazy_status(
                        prompt=prompt,
                        unique_id="postprocess",
                    )
                self.assertEqual(status, ["PROCESS SDXL"])

            with self.subTest(
                sampler=sampler,
                configured_mode=configured_mode,
                endpoint="Visualizer",
            ):
                prompt = _direct_visualizer_prompt(
                    _set_instantid_off(
                        _instantid_prompt(sampler, configured_mode)
                    )
                )
                preflight = self._direct_preflight(prompt)
                self.assertIs(preflight["contract"], family_result._SDXL_COMPLETION)
                self.assertEqual(preflight["path_kind"], "direct")

    def test_existing_on_contracts_resolve_via_node30_and_visualizer(self):
        cases = (
            (
                "CMKInstantIDSamplerSDXLPipe",
                "HYBRID",
                family_result._INSTANTID_HYBRID_COMPLETION,
            ),
            (
                "CMKInstantIDSamplerSDXLPipe",
                "ZERO-PASS",
                family_result._INSTANTID_ZERO_PASS_COMPLETION,
            ),
            (
                "CMKInstantIDRegionalAdvancedSDXLPipe",
                "HYBRID",
                family_result._INSTANTID_HYBRID_COMPLETION,
            ),
        )
        for sampler, mode, expected_contract in cases:
            with self.subTest(sampler=sampler, mode=mode, endpoint="Node 30"):
                self.assertEqual(self._lazy_status(sampler, mode), ["PROCESS SDXL"])
            with self.subTest(sampler=sampler, mode=mode, endpoint="Visualizer"):
                preflight = self._direct_preflight(
                    _direct_visualizer_prompt(_instantid_prompt(sampler, mode))
                )
                self.assertIs(preflight["contract"], expected_contract)
                self.assertEqual(preflight["path_kind"], "direct")

    def test_15a_hybrid_is_accepted_and_preserves_lazy_input_order(self):
        sampler = "CMKInstantIDRegionalAdvancedSDXLPipe"

        self.assertEqual(self._lazy_status(sampler, "HYBRID"), ["PROCESS SDXL"])
        self.assertEqual(
            self._lazy_status(
                sampler,
                "HYBRID",
                **{"PROCESS SDXL": {"family_active": True}},
            ),
            ["MODEL SDXL"],
        )
        self.assertEqual(
            self._lazy_status(
                sampler,
                "HYBRID",
                **{
                    "PROCESS SDXL": {"family_active": True},
                    "MODEL SDXL": {"model": "test"},
                    "IMAGE SDXL": object(),
                    "LOG SDXL": {},
                },
            ),
            ["VISUAL SDXL"],
        )

    def test_15a_zero_pass_remains_outside_the_advanced_contract(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "requires InstantID-Hybrid preliminary-layout sampling",
        ):
            self._lazy_status(
                "CMKInstantIDRegionalAdvancedSDXLPipe",
                "ZERO-PASS",
            )
        preflight = self._direct_preflight(
            _direct_visualizer_prompt(
                _instantid_prompt(
                    "CMKInstantIDRegionalAdvancedSDXLPipe",
                    "ZERO-PASS",
                )
            )
        )
        self.assertIsNone(preflight["contract"])
        self.assertIn(
            "requires InstantID-Hybrid preliminary-layout sampling",
            preflight["detail"],
        )

    def test_15s_existing_hybrid_and_zero_pass_contracts_still_resolve(self):
        for mode in ("HYBRID", "ZERO-PASS"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    self._lazy_status("CMKInstantIDSamplerSDXLPipe", mode),
                    ["PROCESS SDXL"],
                )

    def test_unknown_instantid_sampler_is_not_accepted(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "expanded graph does not prove the current InstantID-Hybrid contract",
        ):
            self._lazy_status("CMKInstantIDUnknownSDXLPipe", "HYBRID")

    def test_off_gate_rejects_manipulated_enable_origin_and_flag(self):
        prompt = _set_instantid_off(
            _instantid_prompt("CMKInstantIDSamplerSDXLPipe", "HYBRID")
        )
        prompt["instantid_gate"]["inputs"]["ENABLE"] = ["source", 0]
        graph, detail = self._sdxl_graph(prompt)
        self.assertIsNone(graph)
        self.assertIn("does not originate from CMKProcessEnableFlag", detail)

        prompt = _set_instantid_off(
            _instantid_prompt("CMKInstantIDSamplerSDXLPipe", "HYBRID")
        )
        prompt["enable_flag"]["inputs"]["flag"] = "controlnet_enabled"
        graph, detail = self._sdxl_graph(prompt)
        self.assertIsNone(graph)
        self.assertIn("does not read instantid_enabled", detail)

    def test_off_gate_rejects_missing_or_mismatched_process_provenance(self):
        prompt = _set_instantid_off(
            _instantid_prompt("CMKInstantIDSamplerSDXLPipe", "HYBRID")
        )
        prompt["orphan_process"] = _node("CMKProcessForwardPipe")
        prompt["enable_flag"]["inputs"]["PROCESS"] = ["orphan_process", 0]
        graph, detail = self._sdxl_graph(prompt)
        self.assertIsNone(graph)
        self.assertIn("does not identify exactly one authoritative source", detail)

        prompt = _set_instantid_off(
            _instantid_prompt("CMKInstantIDSamplerSDXLPipe", "HYBRID")
        )
        prompt["other_source"] = _node(
            "CMKPipeCreateImage",
            model_family="SDXL",
            instantid_enabled=False,
            instantid_mode="HYBRID",
        )
        prompt["enable_flag"]["inputs"]["PROCESS"] = ["other_source", 0]
        graph, detail = self._sdxl_graph(prompt)
        self.assertIsNone(graph)
        self.assertIn("does not originate from the active SDXL source", detail)

    def test_off_gate_requires_every_bypass_connection(self):
        for input_name in (
            "MODEL BYPASS",
            "SAMPLED BYPASS",
            "LOG BYPASS",
            "VISUAL BYPASS",
        ):
            with self.subTest(input_name=input_name):
                prompt = _set_instantid_off(
                    _instantid_prompt("CMKInstantIDSamplerSDXLPipe", "HYBRID")
                )
                prompt["instantid_gate"]["inputs"].pop(input_name)
                graph, detail = self._sdxl_graph(prompt)
                self.assertIsNone(graph)
                self.assertIn(input_name, detail)

    def test_enabled_and_ambiguous_gates_remain_outside_sdxl_contract(self):
        prompt = _instantid_prompt("CMKInstantIDSamplerSDXLPipe", "HYBRID")
        graph, detail = self._sdxl_graph(prompt)
        self.assertIsNone(graph)
        self.assertIn("does not explicitly disable InstantID", detail)

        prompt = _set_instantid_off(
            _instantid_prompt("CMKInstantIDSamplerSDXLPipe", "HYBRID")
        )
        prompt["instantid_gate_2"] = _node(
            "CMKSamplerBypassGate",
            **{
                "ENABLE": ["enable_flag", 0],
                "MODEL BYPASS": ["instantid_gate", 0],
                "SAMPLED BYPASS": ["instantid_gate", 1],
                "LOG BYPASS": ["instantid_gate", 2],
                "VISUAL BYPASS": ["instantid_gate", 4],
                "MODEL ACTIVE": ["instantid_boundary", 0],
                "SAMPLED ACTIVE": ["instantid_boundary", 1],
                "LOG ACTIVE": ["instantid_boundary", 2],
                "DIAGNOSTIC ACTIVE": ["instantid_boundary", 3],
                "VISUAL ACTIVE": ["sampler", 4],
            },
        )
        prompt["refiner_gate"]["inputs"]["MODEL"] = ["instantid_gate_2", 0]
        graph, detail = self._sdxl_graph(prompt)
        self.assertIsNone(graph)
        self.assertIn("multiple InstantID bypass gates are ambiguous", detail)

    def test_stop_after_handoff_uses_terminal_preview_contract(self):
        prompt = _instantid_prompt("CMKInstantIDSamplerSDXLPipe", "HYBRID")
        prompt["source"]["inputs"]["stop_after_handoff"] = True
        process = {
            "model_family": "sdxl",
            "family_active": True,
            "instantid_enabled": True,
            "instantid_mode": "HYBRID",
            "stop_after_handoff": True,
        }

        self.assertFalse(
            family_result.CMKProcessEnableFlag.read(
                process, flag="instantid_enabled"
            )[0]
        )

        gate = family_result.CMKFamilyBranchGateSDXL()
        with patch.object(
            family_result, "_upstream_sdxl_first_pass_hit", return_value=None
        ):
            self.assertEqual(
                gate.check_lazy_status(
                    PROCESS=process,
                    prompt=prompt,
                    unique_id="refiner_gate",
                ),
                ["MODEL"],
            )

        model = {"model": "layout"}
        preview = object()
        log = {"blocks": []}
        visual = {"providers": []}
        diagnostic = {"status": "preview"}
        hit = (
            model,
            {"instantid_end_at_step": 18},
            {
                "image": preview,
                "log": log,
                "visual": visual,
                "diagnostic": diagnostic,
            },
        )
        with patch.object(
            family_result, "_upstream_sdxl_first_pass_hit", return_value=hit
        ):
            self.assertEqual(
                gate.check_lazy_status(
                    PROCESS=process,
                    prompt=prompt,
                    unique_id="refiner_gate",
                ),
                [],
            )
            self.assertEqual(
                gate.gate(
                    PROCESS=process,
                    prompt=prompt,
                    unique_id="refiner_gate",
                ),
                (model, preview, log, visual, diagnostic),
            )

        with patch.object(family_result, "_store_completion_result") as store_result:
            merged = family_result.CMKPostProcessSetupPipe().merge(
                "unused.safetensors",
                "unused.vae",
                True,
                **{
                    "PROCESS SDXL": process,
                    "MODEL SDXL": model,
                    "IMAGE SDXL": preview,
                    "LOG SDXL": log,
                    "VISUAL SDXL": visual,
                },
            )
        store_result.assert_not_called()
        self.assertIs(merged[2], preview)
        self.assertEqual(merged[1]["result_contract"], "preview_handoff")
        self.assertTrue(merged[1]["stop_after_handoff_preview"])
        self.assertTrue(merged[1]["postprocess_model_disabled"])

    def test_zit_hybrid_stop_requests_only_the_sdxl_preview_branch(self):
        prompt = {
            "source": _node(
                "CMKPipeCreateImage",
                model_family="Hybrid",
                hybrid_mode=True,
                instantid_enabled=False,
                stop_after_handoff=True,
            ),
            "postprocess": _node(
                "CMKPostProcessSetupPipe",
                **{
                    "PROCESS SDXL": ["source", 0],
                    "PROCESS ZIT": ["source", 1],
                    "MODEL SDXL": ["sdxl", 0],
                    "IMAGE SDXL": ["sdxl", 1],
                    "LOG SDXL": ["sdxl", 2],
                    "VISUAL SDXL": ["sdxl", 3],
                    "MODEL ZIT": ["zit", 0],
                    "IMAGE ZIT": ["zit", 1],
                    "LOG ZIT": ["zit", 2],
                    "VISUAL ZIT": ["zit", 3],
                },
            ),
        }
        process = {
            "model_family": "sdxl",
            "family_active": True,
            "hybrid_mode": True,
            "instantid_enabled": False,
            "stop_after_handoff": True,
            "unload_models_after_use": False,
        }
        self.assertEqual(
            family_result.CMKPostProcessSetupPipe().check_lazy_status(
                **{
                    "PROCESS SDXL": process,
                    "MODEL SDXL": object(),
                    "IMAGE SDXL": object(),
                    "LOG SDXL": {},
                    "VISUAL SDXL": {"providers": []},
                    "prompt": prompt,
                    "unique_id": "postprocess",
                }
            ),
            [],
        )

        loader_name = f"{TEST_PACKAGE}.pipe.loaders.checkpoint_vae_loader"
        loader = types.ModuleType(loader_name)
        loader.unload_all_phase_models = lambda _label: "UNLOADED"
        process["unload_models_after_use"] = True
        with (
            patch.dict(sys.modules, {loader_name: loader}),
            patch.object(
                family_result,
                "clear_sdxl_sampled_boundary_cache",
                return_value=2,
            ) as clear_boundary,
            patch.object(
                loader,
                "unload_all_phase_models",
                return_value="UNLOADED",
            ) as unload_models,
        ):
            merged = family_result.CMKPostProcessSetupPipe().merge(
                "unused.safetensors",
                "unused.vae",
                True,
                **{
                    "PROCESS SDXL": process,
                    "MODEL SDXL": object(),
                    "IMAGE SDXL": object(),
                    "LOG SDXL": {},
                    "VISUAL SDXL": {"providers": []},
                },
            )
        clear_boundary.assert_called_once_with()
        unload_models.assert_called_once_with(
            "Hybrid SDXL -> ZIT STOP AFTER HANDOFF"
        )
        self.assertEqual(merged[1]["result_contract"], "preview_handoff")
        self.assertTrue(merged[1]["stop_after_handoff_preview"])
        self.assertEqual(merged[1]["generation_model_status"], "UNLOADED")
        self.assertIn("ZIT             : NOT REQUESTED", repr(merged[3]))


if __name__ == "__main__":
    unittest.main()
