"""STOP AFTER HANDOFF in Hybrid (SDXL -> ZIT): no SDXL re-sampling caused by STOP.

Cause (before): STOP is a START HERE widget, so it changes the ComfyUI cache
signature of every descendant including the SDXL first pass.  The first pass
had no cross-run CMK cache; the STOP-independent Hybrid boundary was written
only on STOP runs and read only by the ZIT side.  Sequence STOP false, true,
false, true sampled SDXL 1 / 1 / 0 / 1.

Now: every computed Hybrid first pass is stored in its own scope
``sdxl_hybrid_boundary`` (schema v2: IMAGE, LOG, VISUAL, diagnostic, model
identity, ContentGuard status PASS / BYPASS / NOT INSPECTED).  The SDXL gate
restores it on STOP before requesting MODEL or SAMPLED; the ZIT side reads the
same entry.  MODEL becomes the existing released-generation descriptor.

Restart: the harness restarts the executor in a new process with the same
work directory.  A real ComfyUI start clears ``temp/`` (cleanup_temp_filesystem),
so the entries do not survive a real ComfyUI restart.

Model-free: install_generation_seams fakes CLIP, latents, VAE, sampling and the
ZIT loaders below the real node orchestration; sampling is counted per family.
"""

from __future__ import annotations

import pickle
import unittest
from pathlib import Path

from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
SCOPE, INSTANTID_SCOPE = "sdxl_hybrid_boundary", "sdxl_handoff_boundary"
SEQUENCE = [False, True, False, True]
VISUALIZER_ON_FIRST_PASS = {  # topology of "CMK 2.6 · Flow · InOutpaint HYBRID"
    "8334.MODEL SDXL": ["8430:7417", 0], "8334.PROCESS SDXL": ["8430:7417", 3],
    "8334.IMAGE SDXL": ["8430:7417", 4], "8334.LOG SDXL": ["8430:7417", 2],
    "8334.VISUAL SDXL": ["8430:7417", 5],
}
RESTORE_BLOCK = "SDXL First Pass · Hybrid Boundary"
FIRST_PASS_LIVE, HYBRID_SDXL_GATE = "8430:7415", "8430:7417"
ZIT_CHAIN = {"CMKHybridZITInputPipe", "CMKZImageTurboLoaderPipe", "CMKSamplerPrepareZImageTurboPipe",
             "CMKZImageTurboFinalizePipe", "CMKZImageBoundaryCache", "CMKFamilyBranchGateZImage"}


def sdxl(runs):
    return [run["sampling"]["sdxl"] for run in runs]


def checkpoint_loads(runs):
    return [run["seams"]["substituted_calls"].get("checkpoint.load_resources", 0)
            + run["seams"]["substituted_calls"].get("checkpoint.load_clip", 0) for run in runs]


def handoff_inspections(run):
    return run["guard_calls"].count("target_intermediate")


class HybridStopSequence(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.classic = cls.run_case("hybrid_stop_sequence", stops=SEQUENCE, inspect={"zit_handoff_log": "8431:7657:1"})
        cls.pressure = cls.run_case("hybrid_stop_sequence", stops=SEQUENCE, mode="ram_pressure",
                                    evict_between_runs=True)
        cls.fresh_stop = cls.run_case("hybrid_stop_sequence", stops=[True])
        workdir = cls.workdir("restart")
        first = cls.run_case("hybrid_stop_sequence", workdir=workdir, stops=[False])
        second = cls.run_case("hybrid_stop_sequence", workdir=workdir, stops=[True, False, True])
        cls.restart = None if first is None else first["runs"] + second["runs"]

    def test_sdxl_sampled_once_classic(self):
        runs = self.classic["runs"]
        self.assertTrue(all(run["succeeded"] for run in runs), [run["errors"] for run in runs])
        self.assertEqual(sdxl(runs), [1, 0, 0, 0])
        self.assertEqual(checkpoint_loads(runs), [1, 0, 0, 0])
        self.assertEqual([run["function_calls"].get("CMKCheckpointVAELoaderPipe", 0) for run in runs], [1, 0, 0, 0])
        self.assertEqual([run["function_calls"].get("CMKSamplerPrepareSDXLPipe", 0) for run in runs], [1, 0, 0, 0])

    def test_sdxl_sampled_once_ram_pressure(self):
        runs = self.pressure["runs"]
        self.assertTrue(all(run["succeeded"] for run in runs), [run["errors"] for run in runs])
        self.assertEqual(sdxl(runs), [1, 0, 0, 0])
        self.assertEqual(checkpoint_loads(runs), [1, 0, 0, 0])

    def test_sdxl_sampled_once_across_executor_restart(self):
        self.assertTrue(all(run["succeeded"] for run in self.restart))
        self.assertEqual(sdxl(self.restart), [1, 0, 0, 0])
        self.assertEqual(checkpoint_loads(self.restart), [1, 0, 0, 0])

    def test_zit_side_unchanged(self):
        # ZIT runs only without STOP; the run-3 ZIT sampling is the known F-03
        # path (ZIT gate VISUAL ACTIVE before the boundary), identical before.
        self.assertEqual([run["sampling"]["zit"] for run in self.classic["runs"]], [1, 0, 1, 0])

    def test_fresh_run_keeps_the_fresh_zit_handoff(self):
        # The entry stored in run 1 is not read back by the ZIT side of the same
        # prompt: the fresh handoff runs (and releases the SDXL models first).
        first, third = (self.classic["runs"][index]["outputs"]["zit_handoff_log"]["blocks"][-1] for index in (0, 2))
        self.assertEqual(first[0], "Hybrid Handoff")
        self.assertTrue(any(line.startswith("SDXL MODEL") for line in first[1]), first[1])
        self.assertNotIn("SDXL BOUNDARY   : PERSISTENT CACHE HIT", first[1])
        self.assertIn("SDXL BOUNDARY   : PERSISTENT CACHE HIT", third[1])

    def test_one_entry_in_own_scope_and_stop_independent_key(self):
        listings = [run["scopes"] for run in self.classic["runs"]]
        self.assertEqual(len(listings[0][SCOPE]), 1)
        self.assertTrue(all(item[SCOPE] == listings[0][SCOPE] for item in listings))
        self.assertTrue(all(item[INSTANTID_SCOPE] == [] for item in listings))

    def test_restored_stop_output_equals_fresh_stop(self):
        fresh = self.fresh_stop["runs"][0]["sdxl_gate"]
        for restored in (self.classic["runs"][1]["sdxl_gate"], self.classic["runs"][3]["sdxl_gate"]):
            for name in ("image", "visual", "diagnostic"):
                self.assertEqual(restored[name], fresh[name], name)
            self.assertIsNotNone(restored["diagnostic"])

    def test_restored_log_keeps_every_block_and_adds_the_hit(self):
        fresh = self.fresh_stop["runs"][0]["sdxl_gate"]["log"]["blocks"]
        restored = self.classic["runs"][1]["sdxl_gate"]["log"]["blocks"]
        self.assertEqual(restored[:len(fresh)], fresh)
        self.assertEqual([block["title"] for block in restored[len(fresh):]], [RESTORE_BLOCK])
        self.assertIn("SDXL SAMPLING   : NOT REQUESTED", restored[-1]["lines"])

    def test_restored_model_is_the_released_generation_descriptor(self):
        model = self.classic["runs"][1]["sdxl_gate"]["model"]
        self.assertEqual(model["descriptor_type"], "cmk_generation_model_descriptor")
        self.assertEqual(model["model_role"], "generation_result")
        self.assertEqual(model["materialized"], False)
        self.assertEqual(model["resource_release_status"], "UNLOADED")
        self.assertEqual(model["generation_model_identity"], "juggernautXL_ragnarok.safetensors")
        self.assertEqual(self.classic["runs"][1]["sdxl_gate"]["sampled_blocked"], "ExecutionBlocker")

    def test_restored_stop_keeps_the_final_preview(self):
        fresh = self.fresh_stop["runs"][0]["final_previews"]
        restored = self.classic["runs"][1]["final_previews"]
        # Pixel-identical image, sent for the first-pass stage: by the KSampler
        # when computed, by the SDXL sampled gate on a boundary HIT.  The
        # Visualizer maps both node ids to the first-pass provider
        # (tests/test_visualizer_live_relay.mjs).
        self.assertEqual([item[1] for item in restored], [item[1] for item in fresh])
        self.assertEqual([item[0] for item in fresh], [FIRST_PASS_LIVE])
        self.assertEqual([item[0] for item in restored], [HYBRID_SDXL_GATE])

    def test_stop_requests_no_zit_chain(self):
        for run in (self.classic["runs"][1], self.classic["runs"][3], self.fresh_stop["runs"][0]):
            executed = set(run["function_calls"])
            self.assertEqual(executed & ZIT_CHAIN, set(), run["function_calls"])
            self.assertEqual(run["sampling"]["zit"], 0)


class HybridStopVisualizer(ScenarioCase):
    """Visualizer fed directly by the first pass: the restored VISUAL renders."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sequence = cls.run_case("hybrid_stop_sequence", stops=[False, True], overrides=VISUALIZER_ON_FIRST_PASS)
        cls.fresh = cls.run_case("hybrid_stop_sequence", stops=[True], overrides=VISUALIZER_ON_FIRST_PASS)

    def test_restored_provider_renders_like_fresh(self):
        restored, fresh = self.sequence["runs"][1], self.fresh["runs"][0]
        self.assertEqual(sdxl(self.sequence["runs"]), [1, 0])
        self.assertTrue(restored["succeeded"], restored["errors"])
        self.assertEqual(restored["function_calls"].get("CMKVisualizer"), 1)
        self.assertEqual(restored["visual"], fresh["visual"])
        providers = restored["visual"]["8334"]
        self.assertEqual([(p["provider_id"], p["status"], p["channels"]) for p in providers],
                         [("cmk-CMKKSamplerPipe-8430-7415", "completed", ["result"])])


class HybridStopContentGuard(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.not_inspected = cls.run_case("hybrid_stop_sequence", stops=[False, True, True], guard="pass")
        cls.fresh_pass = cls.run_case("hybrid_stop_sequence", stops=[True, True], guard="pass")
        cls.reject_stop = cls.run_case("hybrid_stop_sequence", stops=[False, True], guard="reject")
        # Run 2 re-executes the ZIT side (a START input that the Hybrid key
        # ignores changes) while the SDXL key stays the same.
        cls.reject_zit = cls.run_case("hybrid_stop_sequence", stops=[False, False], guard="reject",
                                      start_inputs=[{}, {"instantid_handoff_percent": 80}])

    def test_not_inspected_entry_is_inspected_once_before_use(self):
        runs = self.not_inspected["runs"]
        self.assertTrue(all(run["succeeded"] for run in runs), [run["errors"] for run in runs])
        self.assertEqual([handoff_inspections(run) for run in runs], [0, 1, 0])
        self.assertEqual(sdxl(runs), [1, 0, 0])
        self.assertIn("CONTENT GUARD   : PASS · SDXL/ZIT HANDOFF", runs[1]["sdxl_gate"]["log"]["blocks"][-1]["lines"])

    def test_stop_entry_is_inspected_when_stored(self):
        runs = self.fresh_pass["runs"]
        self.assertTrue(all(run["succeeded"] for run in runs))
        self.assertEqual([handoff_inspections(run) for run in runs], [1, 0])
        self.assertEqual(sdxl(runs), [1, 0])

    def test_rejected_entry_is_never_published_and_removed(self):
        stored, refused = self.reject_stop["runs"]
        self.assertTrue(stored["succeeded"])
        self.assertEqual(len(stored["scopes"][SCOPE]), 1)
        # The cached handoff is refused, removed, recomputed and refused again
        # by the unchanged fresh STOP guard: the run aborts as before.
        self.assertFalse(refused["succeeded"])
        self.assertIn("ContentGuard", " ".join(str(e["exception_type"]) for e in refused["errors"]))
        self.assertEqual(sdxl(self.reject_stop["runs"]), [1, 1])
        self.assertEqual(refused["scopes"][SCOPE], [])
        self.assertFalse(any(node.endswith(":7417") for node, _hash in refused["final_previews"]))

    def test_rejected_entry_on_zit_side_falls_back_to_fresh_handoff(self):
        stored, second = self.reject_zit["runs"]
        self.assertTrue(stored["succeeded"] and second["succeeded"], second["errors"])
        self.assertEqual(handoff_inspections(second), 1)
        self.assertEqual(sdxl(self.reject_zit["runs"]), [1, 1])
        self.assertEqual(len(second["scopes"][SCOPE]), 1)  # recomputed and stored again (NOT INSPECTED)


class HybridStopCacheIdentity(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.v1_workdir = cls.workdir("v1")
        cls.v1_first = cls.run_case("hybrid_stop_sequence", workdir=cls.v1_workdir, stops=[False])
        if cls.v1_first is not None:
            entry = cls.v1_first["runs"][0]["scopes"][SCOPE][0]
            path = cls.v1_workdir / "temp" / "cmk" / SCOPE / entry
            v1 = {"schema": "cmk_hybrid_sdxl_image_boundary_v1", "sha256": "", "payload": b""}
            payload = pickle.dumps({"image": None}, protocol=pickle.HIGHEST_PROTOCOL)
            import hashlib
            v1.update(payload=payload, sha256=hashlib.sha256(payload).hexdigest())
            path.write_bytes(pickle.dumps(v1, protocol=pickle.HIGHEST_PROTOCOL))
            legacy = cls.v1_workdir / "temp" / "cmk" / INSTANTID_SCOPE / "legacy_v1_entry.pkl"
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_bytes(b"v1")
        cls.v1_second = cls.run_case("hybrid_stop_sequence", workdir=cls.v1_workdir, stops=[True])
        instantid = {f"{INSTANTID_SCOPE}/instantid_{index}.pkl": b"instantid".hex() for index in range(3)}
        cls.capacity = cls.run_case("hybrid_stop_sequence", stops=[True] * 10, seeds=list(range(1, 11)),
                                    preseed=instantid)

    def test_v1_entry_is_ignored_and_replaced(self):
        run = self.v1_second["runs"][0]
        self.assertTrue(run["succeeded"], run["errors"])
        self.assertEqual(sdxl([run]), [1])  # not misread: recomputed
        self.assertEqual(len(run["scopes"][SCOPE]), 1)
        self.assertEqual(run["scopes"][INSTANTID_SCOPE], ["legacy_v1_entry.pkl"])  # untouched

    def test_hybrid_runs_never_evict_instantid_entries(self):
        runs = self.capacity["runs"]
        last = runs[-1]
        self.assertTrue(all(run["succeeded"] for run in runs), [run["errors"] for run in runs])
        self.assertEqual(len(last["scopes"][SCOPE]), 8)
        self.assertEqual(last["scopes"][INSTANTID_SCOPE], [f"instantid_{index}.pkl" for index in range(3)])

    def test_seed_changes_the_key(self):
        runs = self.capacity["runs"]
        self.assertEqual([len(run["scopes"][SCOPE]) for run in runs], [1, 2, 3, 4, 5, 6, 7, 8, 8, 8])
        self.assertEqual(sdxl(runs), [1] * 10)


if __name__ == "__main__":
    unittest.main()
