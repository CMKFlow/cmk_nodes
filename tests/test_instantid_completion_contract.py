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
            MODEL_BYPASS=["sampled_gate", 0],
            MODEL_ACTIVE=["instantid_boundary", 0],
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


class InstantIDCompletionContractTests(unittest.TestCase):
    def _lazy_status(self, sampler_class, instantid_mode, **inputs):
        prompt = _instantid_prompt(sampler_class, instantid_mode)
        with patch.object(family_result, "lookup_completion", return_value=None):
            return family_result.CMKPostProcessSetupPipe().check_lazy_status(
                prompt=prompt,
                unique_id="postprocess",
                **inputs,
            )

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


if __name__ == "__main__":
    unittest.main()
