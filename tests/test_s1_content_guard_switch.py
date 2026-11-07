"""The ContentGuard switch reaches the node that enforces it.

Standalone FaceSwap: the shipped flow leaves the PROCESS input of
CMKFaceSwapImagePipe open, so the loader's switch never arrived and the node
fell back to ON.  The node now reads the switch from the graph when PROCESS is
not connected; a connected PROCESS still decides.

Generation completion: the key left the guard out, so a completion stored with
one guard state was restored after the switch changed, together with a PROCESS
that carried the old state into every later module.

Model-free: the FaceSwap engine is a seam that records the guard state it gets.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"


class StandaloneFaceSwap(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.runs = cls.run_case("standalone_faceswap_guard")

    def guard(self, label):
        run = self.runs[label]
        self.assertTrue(run["succeeded"], run["errors"])
        return run["engine_guard"]

    def test_switch_off_reaches_the_engine(self):
        self.assertEqual(self.guard("off"), [False])

    def test_switch_on_reaches_the_engine(self):
        self.assertEqual(self.guard("on"), [True])

    def test_off_through_the_process_chain(self):
        # Through CMKResultUnpackPipe the loader's PROCESS reaches the CMK_PIPE input.
        self.assertEqual(self.guard("off_unpack"), [False])
        self.assertEqual(self.guard("on_unpack"), [True])


class CompletionKey(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.keys = cls.run_case("completion_guard_keys")

    def test_guard_state_is_part_of_the_key(self):
        self.assertNotEqual(self.keys["on"], self.keys["off"])
        self.assertEqual(self.keys["on"], self.keys["on_again"])


class PromptState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from importlib import util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "cmk_distribution.py"
        spec = util.spec_from_file_location("cmk_distribution_under_test", path)
        cls.module = util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def state(self, prompt, root=None):
        with patch.object(self.module, "DISTRIBUTION_MODE", self.module.BASELINE):
            return self.module.content_guard_prompt_state(prompt, root)

    def test_start_here_wins_over_a_loader(self):
        prompt = {
            "1": {"class_type": "CMKSwapImageLoaderPipe", "inputs": {"content_guard_enabled": True}},
            "2": {"class_type": "CMKPipeCreateImage", "inputs": {"content_guard_enabled": False}},
        }
        self.assertIs(self.state(prompt), False)

    def test_loader_switch(self):
        for value in (True, False):
            prompt = {"1": {"class_type": "CMKSwapImageLoaderPipe", "inputs": {"content_guard_enabled": value}},
                      "2": {"class_type": "CMKFaceSwapImagePipe", "inputs": {"IMAGE_TARGET": ["1", 1]}}}
            self.assertIs(self.state(prompt, "2"), value)

    def test_missing_switch_stays_on(self):
        prompt = {"1": {"class_type": "CMKSwapImageLoaderPipe", "inputs": {}},
                  "2": {"class_type": "CMKFaceSwapImagePipe", "inputs": {"IMAGE_TARGET": ["1", 1]}}}
        self.assertIs(self.state(prompt, "2"), True)
        self.assertIs(self.state(None, "2"), True)

    def test_unrelated_loader_is_ignored(self):
        prompt = {"1": {"class_type": "CMKSwapImageLoaderPipe", "inputs": {"content_guard_enabled": False}},
                  "2": {"class_type": "CMKFaceSwapImagePipe", "inputs": {}}}
        self.assertIs(self.state(prompt, "2"), True)

    def test_release_ignores_every_switch(self):
        prompt = {"1": {"class_type": "CMKSwapImageLoaderPipe", "inputs": {"content_guard_enabled": False}}}
        with patch.object(self.module, "DISTRIBUTION_MODE", self.module.RELEASE):
            self.assertIs(self.module.content_guard_prompt_state(prompt), True)


if __name__ == "__main__":
    unittest.main()
