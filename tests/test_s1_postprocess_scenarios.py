"""S1-T6 §3.A — PostProcess module behaviour on the exported fixtures.

Real PromptExecutor, real CMK nodes, deterministic engine seams (T1).  Known
contract violations are expectedFailure with their finding ID; each has a
test_baseline_* partner pinning the exact current behaviour.
"""

from __future__ import annotations

import unittest

from harness.scenarios import inner_ids
from harness.support import CHECKPOINT_FIXTURES, GLOBAL_ENABLE, ScenarioCase, gate_of, work_calls

CMK_TEST_TIER = "T1"
BYPASS_FIXTURES = sorted(GLOBAL_ENABLE)


class Bypass(ScenarioCase):
    """V3: global OFF → only BYPASS inputs; image unchanged; no module work."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.results = {}
        for fixture in BYPASS_FIXTURES:
            result = cls.run_case(
                "run_fixture", fixture=fixture,
                widget_overrides={f"8417.{GLOBAL_ENABLE[fixture]}": False},
                inspect={"gate": f"{gate_of(fixture)}:2" if fixture == "maskdetailer" else f"{gate_of(fixture)}:1",
                         "loader": "1:2"},
            )
            cls.results[fixture] = None if result is None else result["runs"][0]

    def test_bypass_succeeds_and_passes_image_unchanged(self):
        for fixture, run in self.results.items():
            with self.subTest(fixture=fixture):
                self.assertTrue(run["succeeded"], run["errors"])
                self.assertEqual(run["outputs"]["gate"]["hash"], run["outputs"]["loader"]["hash"])
                self.assertEqual(run["seams"]["tripped"], [])

    def test_bypass_runs_no_module_work(self):
        # Includes Upscale & Save since H-4: its save endpoint is a lazy leaf.
        for fixture, run in self.results.items():
            with self.subTest(fixture=fixture):
                self.assertEqual(work_calls(run), {})
                self.assertEqual(run["new_files"], {})

    def test_bypassed_module_triggers_no_checkpoint_load(self):
        # H-2 regression: the standalone model source (CMK PostProcess Model)
        # loads nothing; the old eager wiring is pinned in test_s1_postprocess_model.
        loads = {
            fixture: self.results[fixture]["seams"]["substituted_calls"].get("checkpoint.load_resources", 0)
            for fixture in CHECKPOINT_FIXTURES & set(self.results)
        }
        self.assertEqual(set(loads.values()), {0})


class ActivePassthrough(ScenarioCase):
    """V4: global ON, all local branches OFF → unchanged image, no Prepare work."""

    LOCAL = {
        "detailer_std": ("CMK_SmartDetailerPipe", "enable"),
        "detailer_adv": ("CMK_SmartDetailerPipe", "enable"),
        "faceprocess_std": ("CMKFaceProcessPipe", "enable"),
        "faceprocess_adv": ("CMKFaceProcessPipe", "enable"),
        "faceswap_adv": ("CMKFaceSwapImagePipe", "ENABLE"),
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.results = {}
        for fixture, (class_type, name) in cls.LOCAL.items():
            overrides = {f"{node}.{name}": False for node in inner_ids(fixture, class_type)}
            result = cls.run_case(
                "run_fixture", fixture=fixture, overrides=overrides,
                inspect={"gate": f"{gate_of(fixture)}:1", "loader": "1:2"},
            )
            cls.results[fixture] = None if result is None else result["runs"][0]

    def test_faceswap_adv_all_local_off_is_clean_passthrough(self):
        run = self.results["faceswap_adv"]
        self.assertTrue(run["succeeded"], run["errors"])
        self.assertEqual(run["outputs"]["gate"]["hash"], run["outputs"]["loader"]["hash"])
        self.assertEqual(run["seams"]["substituted_calls"], {})
        self.assertIn("DISABLED_PASSTHROUGH", [status for _scope, status, _key in run["statuses"]])
        self.assertIn("8418", run["visual"])

    def test_baseline_k5_detailer_passthrough_runs_expensive_prepare(self):
        for fixture in ("detailer_std", "detailer_adv"):
            with self.subTest(fixture=fixture):
                run = self.results[fixture]
                self.assertFalse(run["succeeded"])
                self.assertEqual(run["seams"]["stopped_calls"], {"prepare.resolve_postprocess_model": 1})
                self.assertEqual([e["node_type"] for e in run["errors"]], ["CMKDetailerPreparePipe"])

    @unittest.expectedFailure  # K-5
    def test_detailer_passthrough_skips_prepare(self):
        for fixture in ("detailer_std", "detailer_adv"):
            run = self.results[fixture]
            self.assertTrue(run["succeeded"])
            self.assertEqual(run["seams"]["stopped_calls"], {})

    def test_baseline_h5_faceprocess_passthrough_blocks_all_downstream_output(self):
        for fixture in ("faceprocess_std", "faceprocess_adv"):
            with self.subTest(fixture=fixture):
                run = self.results[fixture]
                self.assertTrue(run["succeeded"])
                self.assertIn("DISABLED_ACTIVE_BRANCH_BLOCKED", [s for _scope, s, _k in run["statuses"]])
                self.assertNotIn("CMKVisualizer", run["function_calls"])
                self.assertEqual(run["visual"], {})
                # The gate output is a stored ExecutionBlocker: nothing reaches
                # Visual Compare, Visualizer or any save endpoint.
                self.assertEqual(run["outputs"]["gate"], {"type": "ExecutionBlocker"})

    @unittest.expectedFailure  # H-5
    def test_faceprocess_passthrough_delivers_unchanged_image(self):
        for fixture in ("faceprocess_std", "faceprocess_adv"):
            run = self.results[fixture]
            self.assertEqual(run["outputs"]["gate"].get("hash"), run["outputs"]["loader"]["hash"])
            self.assertIn("CMKVisualizer", run["function_calls"])


class FaceSwapActive(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.results = {}
        cls.empty = {}
        for fixture in ("faceswap_std", "faceswap_adv"):
            result = cls.run_case("run_fixture", fixture=fixture,
                                  inspect={"gate": f"{gate_of(fixture)}:1", "loader": "1:2"})
            cls.results[fixture] = None if result is None else result["runs"][0]
            empty = cls.run_case("run_fixture", fixture=fixture, fake_faces=0)
            cls.empty[fixture] = None if empty is None else empty["runs"][0]

    def test_active_faceswap_processes_image(self):
        for fixture, run in self.results.items():
            with self.subTest(fixture=fixture):
                self.assertTrue(run["succeeded"], run["errors"])
                self.assertNotEqual(run["outputs"]["gate"]["hash"], run["outputs"]["loader"]["hash"])
                self.assertEqual(run["seams"]["substituted_calls"].get("faceswap.swap"), 1)
                self.assertEqual(run["seams"]["original_calls"], {})

    def test_active_faceswap_publishes_visual_provider(self):
        # H-3 regression (the FaceSwap reference observation K3): MODEL is None
        # without a checkpoint, yet the module provider reaches Visualizer and
        # Visual Compare.
        for fixture, run in self.results.items():
            with self.subTest(fixture=fixture):
                compare = inner_ids(fixture, "CMKVisualCompare")[0]
                self.assertEqual(len(run["visual"]["8418"]), 1)
                self.assertEqual(run["visual"][compare], run["visual"]["8418"])

    def test_baseline_m3_no_detection_raises(self):
        for fixture, run in self.empty.items():
            with self.subTest(fixture=fixture):
                self.assertFalse(run["succeeded"])
                self.assertEqual(
                    [(e["node_type"], (e["exception_message"] or "").strip()) for e in run["errors"]],
                    [("CMKFaceSwapImagePipe", "No faces found for image_index 0")],
                )

    @unittest.expectedFailure  # M3 / OD-5: completed_empty
    def test_no_detection_is_completed_empty(self):
        for fixture, run in self.empty.items():
            self.assertTrue(run["succeeded"], fixture)


class ForwardWithoutCompute(ScenarioCase):
    """A5: requesting only PROCESS or MODEL must not run Prepare or Compute."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.process = cls.run_case("forward_only", fixture="detailer_std", source="8417:7707:0")
        cls.model = cls.run_case("forward_only", fixture="detailer_std", source=f"{gate_of('detailer_std')}:0")

    def test_process_forward_runs_no_module_work(self):
        self.assertTrue(self.process["succeeded"], self.process["errors"])
        self.assertEqual(work_calls(self.process), {})
        self.assertEqual(self.process["seams"]["stopped_calls"], {})

    def test_baseline_od2_model_request_runs_prepare(self):
        self.assertFalse(self.model["succeeded"])
        self.assertEqual(self.model["seams"]["stopped_calls"], {"prepare.resolve_postprocess_model": 1})

    @unittest.expectedFailure  # OD-2: MODEL currently routed through gate and boundary
    def test_model_forward_runs_no_module_work(self):
        self.assertTrue(self.model["succeeded"])
        self.assertEqual(self.model["seams"]["stopped_calls"], {})


if __name__ == "__main__":
    unittest.main()
