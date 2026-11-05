"""STOP AFTER HANDOFF for SDXL -> InstantID, Text2Image and Inpaint.

Contract: module 01 resolves one effective InstantID mode (Inpaint always
means HYBRID INPAINT, whatever the hidden widget holds).  PROCESS and prompt
evaluation use that one rule, so InstantID, Refiner, Visualizer and PostProcess
decide alike.

The SDXL handoff cache also serves HYBRID INPAINT.  Its entry is the x0 estimate
after the Inpaint Bridge.  Inpaint depends on file pixels and mask, which the
prompt only names, so in this configuration the fingerprint of an image-file
loader carries its IS_CHANGED content digest.

Model-free: install_generation_seams fakes the Fooocus patch, VAE, sampling,
InstantID and the Refiner below the real node orchestration.
"""

from __future__ import annotations

import unittest

from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
FIXTURE = "instantid_inpaint"
SCOPE = "sdxl_handoff_boundary"
TEXT2IMAGE = {"INPAINT_MODE": "Text2Image"}
SEQUENCE = [False, True, False, True]
REFINER_GATE_IMAGE = "8433:7812:1"
FIRST_PASS = "cmk-CMKKSamplerPipe-8430-7415"
IDENTITY_MODULES = {"CMKInstantIDBoundary", "CMKRefinerPipe", "CMKRefinerPrepareSDXLPipe"}
DARK, BRIGHT = {"name": "same_name.png", "shade": 40}, {"name": "same_name.png", "shade": 200}
OTHER_MASK = {"name": "same_name.png", "shade": 40, "mask_box": (8, 8, 32, 32)}


def counts(runs, name):
    return [run["sampling"][name] for run in runs]


def entries(run):
    return len(run["scopes"][SCOPE])


def guard_handoffs(run):
    return run["guard_calls"].count("target_intermediate")


class Sequence(ScenarioCase):
    """STOP OFF, ON, OFF, ON with a real executor (cases 1, 2, 4, 5, 6)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        inspect = {"gate_image": REFINER_GATE_IMAGE}
        cls.inpaint = cls.run_case("hybrid_stop_sequence", fixture=FIXTURE, stops=SEQUENCE, inspect=inspect)
        cls.text2image = cls.run_case("hybrid_stop_sequence", fixture=FIXTURE, stops=SEQUENCE, inspect=inspect,
                                      start_inputs=[TEXT2IMAGE] * 4)
        cls.first_stop = cls.run_case("hybrid_stop_sequence", fixture=FIXTURE, stops=[True, False, True])

    def test_every_run_succeeds(self):
        for sequence in (self.inpaint, self.text2image, self.first_stop):
            for run in sequence["runs"]:
                self.assertTrue(run["succeeded"], run["errors"])

    def test_inpaint_samples_sdxl_once(self):
        self.assertEqual(counts(self.inpaint["runs"], "sdxl"), [1, 0, 0, 0])

    def test_text2image_samples_sdxl_once(self):
        self.assertEqual(counts(self.text2image["runs"], "sdxl"), [1, 0, 0, 0])

    def test_stop_on_keeps_instantid_and_refiner_out(self):
        for sequence in (self.inpaint, self.text2image):
            self.assertEqual(counts(sequence["runs"], "instantid"), [1, 0, 1, 0])
            self.assertEqual(counts(sequence["runs"], "refiner"), [1, 0, 1, 0])
            for run in sequence["runs"][1::2]:
                self.assertFalse(IDENTITY_MODULES & set(run["function_calls"]))

    def test_stop_as_first_run_computes_only_the_handoff(self):
        runs = self.first_stop["runs"]
        self.assertEqual(counts(runs, "sdxl"), [1, 0, 0])
        self.assertEqual(counts(runs, "instantid"), [0, 1, 0])

    def test_stop_off_after_stop_on_continues_from_the_stored_handoff(self):
        computed, restored = self.inpaint["runs"][0], self.inpaint["runs"][2]
        self.assertEqual(restored["instantid_latents"], computed["instantid_latents"])
        self.assertEqual(restored["outputs"], computed["outputs"])
        self.assertEqual(restored["visual"], computed["visual"])

    def test_stop_outputs_the_handoff_image_and_shows_it(self):
        for sequence in (self.inpaint, self.text2image):
            run = sequence["runs"][1]
            providers = run["visual"]["8334"]
            self.assertEqual(providers[-1]["provider_id"], FIRST_PASS)
            self.assertEqual(providers[-1]["status"], "completed")
            self.assertEqual(providers[-1]["channel_hashes"]["result"], run["public_image_png"])
            self.assertEqual(run["function_calls"].get("CMKVisualizer"), 1)

    def test_stop_preview_is_the_same_for_computed_and_restored_handoff(self):
        restored, fresh = self.inpaint["runs"][1], self.first_stop["runs"][0]
        self.assertEqual(restored["public_image_png"], fresh["public_image_png"])
        self.assertEqual(restored["visual"]["8334"], fresh["visual"]["8334"])

    def test_one_entry_serves_all_runs(self):
        self.assertEqual([entries(run) for run in self.inpaint["runs"]], [1, 1, 1, 1])


class HiddenWidget(ScenarioCase):
    """A hidden ZERO-PASS must not change STOP in Inpaint."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.zero_pass = cls.run_case("hybrid_stop_sequence", fixture=FIXTURE, stops=[False, True],
                                     start_inputs=[{"instantid_mode": "ZERO-PASS"}] * 2)

    def test_stop_applies(self):
        runs = self.zero_pass["runs"]
        self.assertTrue(all(run["succeeded"] for run in runs), [run["errors"] for run in runs])
        self.assertEqual(counts(runs, "sdxl"), [1, 0])
        self.assertEqual(counts(runs, "instantid"), [1, 0])
        self.assertEqual(runs[1]["visual"]["8334"][-1]["provider_id"], FIRST_PASS)


class HandoffPercent(ScenarioCase):
    """The slider value passes ComfyUI's validation and reaches the sampler."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        percents = [{"instantid_handoff_percent": value} for value in (40, 50, 60, 70, 80)]
        stops = [False] * 5
        cls.inpaint = cls.run_case("hybrid_stop_sequence", fixture=FIXTURE, stops=stops, start_inputs=percents)
        cls.text2image = cls.run_case("hybrid_stop_sequence", fixture=FIXTURE, stops=stops,
                                      start_inputs=[{**TEXT2IMAGE, **item} for item in percents])

    def test_sampler_and_instantid_use_the_step(self):
        for sequence in (self.inpaint, self.text2image):
            runs = sequence["runs"]
            self.assertTrue(all(run["succeeded"] for run in runs), [run["errors"] for run in runs])
            self.assertEqual([run["sampler_last_steps"] for run in runs], [[10], [13], [15], [18], [20]])
            self.assertEqual([run["instantid_latents"][0]["start_step"] for run in runs], [10, 13, 15, 18, 20])


class EvictedExecutorCache(ScenarioCase):
    """The handoff survives without ComfyUI's own cache."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ram = cls.run_case("hybrid_stop_sequence", fixture=FIXTURE, stops=SEQUENCE, mode="ram_pressure",
                               evict_between_runs=True)

    def test_sdxl_samples_once(self):
        self.assertEqual(counts(self.ram["runs"], "sdxl"), [1, 0, 0, 0])
        self.assertEqual(counts(self.ram["runs"], "instantid"), [1, 0, 1, 0])


class StopContract(ScenarioCase):
    """Case 3: the real START HERE PROCESS and the prompt agree everywhere."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.matrix = cls.run_case("stop_contract_matrix")

    def test_process_and_prompt_agree(self):
        rows = self.matrix["rows"]
        self.assertEqual([row for row in rows if row["process"] != row["prompt"]], [])
        self.assertEqual(len(rows), 3 * 4 * 2 * 4 * 2)

    def test_valid_combinations(self):
        for row in self.matrix["rows"]:
            widgets = row["widgets"]
            inpaint = widgets["INPAINT_MODE"] in {"Inpaint", True}
            expected = bool(widgets["stop_after_handoff"]) and (
                widgets["model_family"] == "Hybrid"
                or (widgets["model_family"] == "SDXL" and widgets["instantid_enabled"]
                    and (inpaint or widgets["instantid_mode"] != "ZERO-PASS"))
            )
            self.assertEqual(row["process"], expected, widgets)

    def test_inpaint_always_resolves_to_hybrid_inpaint(self):
        for row in self.matrix["rows"]:
            if row["widgets"]["INPAINT_MODE"] in {"Inpaint", True}:
                self.assertEqual(row["process_mode"], "HYBRID INPAINT")


class ContentIdentity(ScenarioCase):
    """Case 7: same file name, other pixels or mask: no stale handoff."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.keys = cls.run_case("input_content_keys")
        cls.stop = cls.run_case("hybrid_stop_sequence", fixture=FIXTURE, stops=[True] * 4,
                                input_images=[DARK, BRIGHT, OTHER_MASK, DARK],
                                inspect={"gate_image": REFINER_GATE_IMAGE})
        cls.final = cls.run_case("hybrid_stop_sequence", fixture=FIXTURE, stops=[False, True, False],
                                 input_images=[DARK, BRIGHT, BRIGHT])

    def test_pixels_or_mask_change_the_keys(self):
        first, pixels, mask = (self.keys[label]["inpaint"] for label in ("a", "pixels", "mask"))
        for index in (0, 1):
            self.assertEqual(len({first[index], pixels[index], mask[index]}), 3)

    def test_other_configurations_keep_their_keys(self):
        self.assertEqual(self.keys["a"]["text2image"], self.keys["pixels"]["text2image"])
        self.assertEqual(self.keys["a"]["text2image"], self.keys["mask"]["text2image"])

    def test_changed_file_recomputes_the_handoff(self):
        self.assertEqual(counts(self.stop["runs"], "sdxl"), [1, 1, 1, 0])
        self.assertEqual([entries(run) for run in self.stop["runs"]], [1, 2, 3, 3])

    def test_changed_content_reaches_the_sampler(self):
        prepared = [run["inpaint_prepares"][0] for run in self.stop["runs"]]
        self.assertEqual(len({item["image"] for item in prepared}), 3)
        self.assertEqual(len({item["mask"] for item in prepared}), 2)
        self.assertEqual(prepared[0], prepared[3])

    def test_stop_preview_follows_the_file(self):
        views = [run["outputs"]["gate_image"]["hash"] for run in self.stop["runs"]]
        self.assertEqual(len(set(views)), 3)
        self.assertEqual(views[0], views[3])

    def test_stop_off_does_not_serve_the_old_result(self):
        old, stopped, final = self.final["runs"]
        self.assertEqual(counts(self.final["runs"], "sdxl"), [1, 1, 0])
        self.assertEqual(counts(self.final["runs"], "refiner"), [1, 0, 1])
        self.assertNotEqual(final["instantid_latents"], old["instantid_latents"])
        self.assertNotEqual(final["visual"], old["visual"])


class InpaintGuard(ScenarioCase):
    """Case 8: ContentGuard on the x0 before the bridge, and the Inpaint mask."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.plain = cls.run_case("hybrid_stop_sequence", fixture=FIXTURE, stops=[False, True, False])
        cls.passed = cls.run_case("hybrid_stop_sequence", fixture=FIXTURE, stops=[True, True, False], guard="pass")
        cls.rejected = cls.run_case("hybrid_stop_sequence", fixture=FIXTURE, stops=[True, True], guard="reject")

    def test_guard_inspects_once_and_hit_keeps_the_approval(self):
        runs = self.passed["runs"]
        self.assertTrue(all(run["succeeded"] for run in runs), [run["errors"] for run in runs])
        self.assertEqual([guard_handoffs(run) for run in runs], [1, 0, 0])
        self.assertEqual(counts(runs, "sdxl"), [1, 0, 0])
        self.assertEqual(counts(runs, "instantid"), [0, 0, 1])

    def test_rejected_handoff_is_neither_stored_nor_shown(self):
        runs = self.rejected["runs"]
        self.assertEqual([run["succeeded"] for run in runs], [False, False])
        self.assertEqual(counts(runs, "sdxl"), [1, 1])
        self.assertEqual([entries(run) for run in runs], [0, 0])
        for run in runs:
            self.assertEqual(run["visual"].get("8334"), None)
            self.assertEqual(run["function_calls"].get("CMKVisualizer", 0), 0)

    def test_bridge_removes_the_inpaint_latent_contract(self):
        computed, restored = self.plain["runs"][0], self.plain["runs"][2]
        for run in (computed, restored):
            self.assertEqual(run["instantid_latents"][0]["keys"], ["samples"])
            self.assertTrue(run["instantid_latents"][0]["bridge"])

    def test_mask_reaches_fooocus_and_the_sampler(self):
        run = self.plain["runs"][0]
        prepared = run["inpaint_prepares"][0]
        self.assertEqual(run["seams"]["substituted_calls"]["generation.fooocus_prepare"], 1)
        self.assertEqual(prepared["grow_mask_by"], 16)
        self.assertTrue(prepared["noise_mask"])
        self.assertEqual(run["sampler_noise_masks"], [prepared["mask"]])


if __name__ == "__main__":
    unittest.main()
