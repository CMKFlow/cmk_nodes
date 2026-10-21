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


if __name__ == "__main__":
    unittest.main()
