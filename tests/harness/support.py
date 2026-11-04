"""Shared helpers for S1-T6 scenario tests."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from .restart import run_scenario
from .scenarios import inner_ids

ENV_MODEL_PATHS = "CMK_EXTRA_MODEL_PATHS"

# Global module switch per fixture (instance 8417 in every module fixture).
GLOBAL_ENABLE = {
    "detailer_std": "detailer_global_enable",
    "detailer_adv": "detailer_global_enable",
    "faceprocess_std": "face_global_enable",
    "faceprocess_adv": "face_global_enable",
    "facerebuild_std": "FACEREBUILD ENABLE",
    "facerebuild_adv": "facerebuild_global_enable",
    "faceswap_std": "FACESWAP ENABLE",
    "faceswap_adv": "FACESWAP ENABLE",
    "maskdetailer": "enabled",
    "upscale_save": "SAVE ENABLED",
}
CHECKPOINT_FIXTURES = {
    "detailer_std", "detailer_adv", "faceprocess_std", "faceprocess_adv",
    "facerebuild_std", "facerebuild_adv", "maskdetailer", "chain_postprocess",
}
# Module nodes that must stay idle when the module is bypassed.
MODULE_WORK_CLASSES = {
    "CMKDetailerPreparePipe", "CMK_SmartDetailerPipe", "CMKDetailerBoundaryCache",
    "CMKFaceProcessPreparePipe", "CMKFaceProcessPipe", "CMKFaceBoundaryCache",
    "CMKInstantIDFaceRebuildSDXL", "CMKInstantIDFaceRebuildAdvancedSDXL", "CMKFaceRebuildBoundaryCache",
    "CMKFaceSwapImagePipe", "CMKFaceSwapBoundaryCache", "CMK_SEGSConcate",
    "CMKMaskDetailerIntake", "CMKMaskDetailerPrepare", "CMKMaskDetailerProcess",
    "CMK_SmartUpscalerPipe", "CMK_SaveProjectImage", "CMKUpscaleSaveBoundaryCache",
}


def model_paths() -> str | None:
    value = os.environ.get(ENV_MODEL_PATHS, "")
    return value if value and Path(value).is_file() else None


class ScenarioCase(unittest.TestCase):
    """Base class: skips when the instance model-path YAML is not configured
    (combo values would not validate) and provides a private work directory."""

    @classmethod
    def setUpClass(cls):
        cls.paths = model_paths()
        cls._root = Path(tempfile.mkdtemp(prefix=f"cmk-s1-{cls.__name__}-"))
        cls._counter = 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._root, ignore_errors=True)

    def setUp(self):
        if self.paths is None:
            self.skipTest(f"{ENV_MODEL_PATHS} not set to the instance's extra model paths YAML")

    @classmethod
    def run_case(cls, name, workdir=None, **kwargs):
        if cls.paths is None:
            return None
        if workdir is None:
            cls._counter += 1
            workdir = cls._root / f"w{cls._counter}"
        return run_scenario(name, Path(workdir), extra_model_paths=cls.paths, **kwargs)

    @classmethod
    def workdir(cls, label):
        return cls._root / label


def gate_of(fixture: str) -> str:
    ids = inner_ids(fixture, "CMKModuleBypassGate") or inner_ids(fixture, "CMKMaskDetailerFlowGate")
    return ids[0]


def work_calls(run: dict) -> dict:
    return {name: count for name, count in run["function_calls"].items() if name in MODULE_WORK_CLASSES}
