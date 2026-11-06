"""SDXL -> ZIT with an empty global prompt and active regional prompts.

Regional nodes sit on the PROCESS SDXL branch, so PROCESS ZIT from START HERE
never carries their text.  CMKSamplerPrepareZImageTurboPipe reads the active
region texts from the expanded prompt, upstream of the SDXL handoff that its
IMAGE comes from.  No port, no pipe field and no sampling value changes.

Model-free: install_generation_seams fakes CLIP, VAE and sampling; the text that
reaches the ZIT text encoder is recorded.
"""

from __future__ import annotations

import unittest

from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
GLOBAL = "a global scene"
ONE, TWO, THREE = "woman in a green coat", "old brewery", "evening light"
EMPTY = {"GLOBAL PROMPT POS": ""}
INPAINT_IMAGE = "cmk_s1_input.png"


def sequence(case, *, global_prompt=True, regional=None, stops=(False,), **kwargs):
    count = len(stops)
    return case.run_case(
        "hybrid_stop_sequence", stops=list(stops),
        start_inputs=[{} if global_prompt else EMPTY] * count,
        regional=[regional] * count if not isinstance(regional, list) else regional, **kwargs,
    )


class ZitPrompt(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.plain = sequence(cls)
        cls.both = sequence(cls, regional={"layout": [ONE, TWO]})
        cls.one = sequence(cls, global_prompt=False, regional={"layout": [ONE]})
        cls.many = sequence(cls, global_prompt=False, regional={"layout": [ONE, TWO, THREE]})
        cls.conditioning = sequence(cls, global_prompt=False, regional={"conditioning": [TWO, ONE]})
        cls.blank_regions = sequence(cls, global_prompt=False, regional={"layout": [" ", ONE, ""]})
        cls.empty = sequence(cls, global_prompt=False, regional={"layout": ["  "]})
        cls.none = sequence(cls, global_prompt=False)
        cls.inpaint = sequence(cls, global_prompt=False, regional={"layout": [ONE, TWO]},
                               inpaint_source=INPAINT_IMAGE)
        cls.sdxl = sequence(cls, global_prompt=False, regional={"layout": [ONE, TWO]},
                            stops=(False,), fixture="instantid_inpaint",)

    def zit_prompt(self, result):
        run = result["runs"][0]
        self.assertTrue(run["succeeded"], run["errors"])
        return run["zit_prompts"]

    def test_global_prompt_only(self):
        prompts = self.zit_prompt(self.plain)
        self.assertEqual(len(prompts), 1)
        self.assertTrue(prompts[0].startswith("Cinematic editorial portrait"))

    def test_global_prompt_keeps_priority_over_regions(self):
        self.assertEqual(self.zit_prompt(self.both), self.zit_prompt(self.plain))

    def test_single_region(self):
        self.assertEqual(self.zit_prompt(self.one), [ONE])

    def test_several_regions_in_order(self):
        self.assertEqual(self.zit_prompt(self.many), [f"{ONE}\n{TWO}\n{THREE}"])

    def test_regional_conditioning_node(self):
        self.assertEqual(self.zit_prompt(self.conditioning), [f"{TWO}\n{ONE}"])

    def test_inactive_regions_are_skipped(self):
        self.assertEqual(self.zit_prompt(self.blank_regions), [ONE])

    def test_no_prompt_at_all_is_a_clear_error(self):
        for result in (self.empty, self.none):
            run = result["runs"][0]
            self.assertFalse(run["succeeded"])
            self.assertEqual(run["errors"][0]["node_type"], "CMKSamplerPrepareZImageTurboPipe")
            self.assertIn("GLOBAL PROMPT POS or an active regional prompt", run["errors"][0]["exception_message"])

    def test_inoutpaint_with_zit_finish(self):
        self.assertEqual(self.zit_prompt(self.inpaint), [f"{ONE}\n{TWO}"])
        run = self.inpaint["runs"][0]
        self.assertEqual(run["sampling"]["sdxl"], 1)
        self.assertEqual(run["sampling"]["zit"], 1)
        self.assertEqual(len(run["inpaint_prepares"]), 1)

    def test_zit_gets_only_semantic_text(self):
        text = self.zit_prompt(self.many)[0]
        self.assertEqual(text.split("\n"), [ONE, TWO, THREE])
        for control in ("region-id", "REGION NAME", "0.1", "0.2", "0.3", "0.4", "0.65", "1.5", "box", "strength"):
            self.assertNotIn(control, text)

    def test_spatial_control_stays_on_the_sdxl_path(self):
        run = self.many["runs"][0]
        self.assertEqual(len(self.zit_prompt(self.many)), 1)
        self.assertEqual(len(run["regional_areas"]), 3)  # the three SDXL regions, none for ZIT

    def test_sdxl_regions_stay_spatial(self):
        run = self.many["runs"][0]
        self.assertEqual(len(run["regional_areas"]), 3)
        self.assertEqual({(item["x"], item["strength"]) for item in run["regional_areas"]}, {(0.1, 1.5)})

    def test_sdxl_without_zit_is_unchanged(self):
        run = self.sdxl["runs"][0]
        self.assertTrue(run["succeeded"], run["errors"])
        self.assertEqual(run["zit_prompts"], [])
        self.assertEqual(run["sampling"]["zit"], 0)
        self.assertEqual(len(run["regional_areas"]), 2)


class HandoffReuse(ScenarioCase):
    """The reported sequence: STOP stores the handoff, the next run finishes it."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.reuse = sequence(cls, global_prompt=False, regional={"layout": [ONE, TWO]},
                             stops=(True, False, False))
        cls.changed = cls.run_case(
            "hybrid_stop_sequence", stops=[False, False], start_inputs=[EMPTY, EMPTY],
            regional=[{"layout": [ONE]}, {"layout": [TWO]}],
        )
        cls.finish_only = cls.run_case(
            "hybrid_stop_sequence", stops=[False, False],
            start_inputs=[{**EMPTY, "hybrid_zit_denoise": "0.20"}, {**EMPTY, "HYBRID BALANCE": 3}],
            regional=[{"layout": [ONE]}, {"layout": [ONE]}],
        )

    def test_zit_runs_from_the_stored_handoff(self):
        stop, finish, repeat = self.reuse["runs"]
        self.assertTrue(stop["succeeded"] and finish["succeeded"] and repeat["succeeded"])
        self.assertEqual([run["sampling"]["sdxl"] for run in self.reuse["runs"]], [1, 0, 0])
        self.assertEqual([run["sampling"]["zit"] for run in self.reuse["runs"]], [0, 1, 0])
        self.assertEqual(finish["zit_prompts"], [f"{ONE}\n{TWO}"])

    def test_changed_region_text_reaches_zit_and_recomputes(self):
        first, second = self.changed["runs"]
        self.assertEqual((first["zit_prompts"], second["zit_prompts"]), ([ONE], [TWO]))
        self.assertEqual(len(second["scopes"]["sdxl_hybrid_boundary"]), 2)
        self.assertEqual(second["sampling"]["sdxl"], 1)
        self.assertEqual(second["sampling"]["zit"], 1)


if __name__ == "__main__":
    unittest.main()
