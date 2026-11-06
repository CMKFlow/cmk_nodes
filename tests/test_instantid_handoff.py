from __future__ import annotations

import ast
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE_ROOT / "pipe" / "cmk_sampler_prepare.py"
START_PATH = PACKAGE_ROOT / "pipe" / "cmk_pipe_image.py"
START_UI_PATH = PACKAGE_ROOT / "web" / "js" / "cmk_flow_start_guidance_v51.js"
SAMPLER_PATH = PACKAGE_ROOT / "pipe" / "cmk_pipe_sampler.py"
INSTANTID_PATHS = (
    PACKAGE_ROOT / "pipe" / "instantid" / "cmk_instantid_sampler.py",
    PACKAGE_ROOT / "pipe" / "instantid" / "cmk_instantid_regional_advanced.py",
)


def _load_handoff_resolver():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_instantid_handoff"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(MODULE_PATH), "exec"),
        namespace,
    )
    return namespace["_resolve_instantid_handoff"]


class InstantIDHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolve = staticmethod(_load_handoff_resolver())

    def handoff(self, *, percent=70, mode="HYBRID", steps=25):
        return self.resolve(
            active=True,
            instantid_mode=mode,
            effective_steps=steps,
            handoff_percent=percent,
        )

    def test_only_20_to_80_in_tens_are_accepted_and_70_is_default(self):
        self.assertEqual(self.handoff(percent=20), (5, "20 %", 20))
        self.assertEqual(self.handoff(percent=30), (8, "30 %", 30))
        self.assertEqual(self.handoff(percent=40), (10, "40 %", 40))
        self.assertEqual(self.handoff(percent=50), (13, "50 %", 50))
        self.assertEqual(self.handoff(percent=60), (15, "60 %", 60))
        self.assertEqual(self.handoff(percent=70), (18, "70 %", 70))
        self.assertEqual(self.handoff(percent=80), (20, "80 %", 80))
        self.assertEqual(self.handoff(percent=65), (18, "70 %", 70))
        self.assertEqual(self.handoff(percent=45), (18, "70 %", 70))
        self.assertEqual(self.handoff(percent=10), (18, "70 %", 70))
        self.assertEqual(self.handoff(percent="AUTO"), (18, "70 %", 70))

    def test_percentage_handoffs_scale_and_round_half_up(self):
        cases = (
            (10, 6, 7, 8),
            (15, 9, 11, 12),
            (30, 18, 21, 24),
            (35, 21, 25, 28),
        )
        for steps, at_60, at_70, at_80 in cases:
            for percent, expected in ((60, at_60), (70, at_70), (80, at_80)):
                with self.subTest(steps=steps, percent=percent):
                    self.assertEqual(self.handoff(percent=percent, steps=steps)[0], expected)

    def test_text2image_and_inpaint_share_the_same_percent_contract(self):
        prepare_source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("inpaint_handoff", prepare_source)
        self.assertNotIn("inpaint_mode=inpaint_mode", prepare_source)
        self.assertIn('handoff_percent=pipe.get("instantid_handoff_percent", 70)', prepare_source)

    def test_controlnet_cannot_override_the_percent_handoff(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_resolve_instantid_handoff"
        )
        function_source = ast.get_source_segment(source, function)
        self.assertNotIn("controlnet", function_source.lower())
        for percent, expected in ((60, 15), (70, 18), (80, 20)):
            with self.subTest(percent=percent):
                self.assertEqual(self.handoff(percent=percent)[0], expected)

    def test_zero_pass_remains_step_0(self):
        self.assertEqual(
            self.handoff(percent=80, mode="ZERO-PASS"),
            (0, "ZERO-PASS", 80),
        )

    def test_handoff_is_still_clamped_below_total_steps(self):
        result = self.resolve(
            active=True,
            instantid_mode="HYBRID",
            effective_steps=2,
            handoff_percent=80,
        )
        self.assertEqual(result, (1, "80 %", 80))

    def test_every_limit_uses_the_same_values(self):
        for path, expected in (
            (MODULE_PATH, "{20, 30, 40, 50, 60, 70, 80}"),
            (START_PATH, "{20, 30, 40, 50, 60, 70, 80}"),
            (START_UI_PATH, "[20, 30, 40, 50, 60, 70, 80]"),
        ):
            source = path.read_text(encoding="utf-8")
            self.assertIn(expected, source)
            self.assertNotIn(expected.replace("20, 30, ", ""), source)

    def test_ui_contract_is_slider_20_to_80_with_default_70(self):
        source = START_PATH.read_text(encoding="utf-8")
        ui_source = START_UI_PATH.read_text(encoding="utf-8")
        self.assertIn('"instantid_handoff_percent": (', source)
        self.assertIn('"default": 70', source)
        self.assertIn('"min": 20', source)
        self.assertIn('"max": 80', source)
        self.assertIn('"step": 10', source)
        self.assertIn('"display": "slider"', source)
        self.assertIn('"label": "HYBRID HANDOFF"', source)
        self.assertIn('"instantid_handoff_percent": "HYBRID HANDOFF"', ui_source)

    def test_legacy_strategy_logic_is_removed_from_active_handoff_path(self):
        active_sources = (
            START_PATH.read_text(encoding="utf-8"),
            MODULE_PATH.read_text(encoding="utf-8"),
            START_UI_PATH.read_text(encoding="utf-8"),
        )
        for source in active_sources:
            self.assertNotIn("instantid_handshake_strategy", source)
            self.assertNotIn('"EARLY"', source)
            self.assertNotIn('"SAFE"', source)

    def test_handoff_value_is_shared_by_module_10_15s_and_15a(self):
        prepare_source = MODULE_PATH.read_text(encoding="utf-8")
        sampler_source = SAMPLER_PATH.read_text(encoding="utf-8")

        self.assertIn('new_pipe["instantid_end_at_step"] = instantid_handoff', prepare_source)
        self.assertIn('new_pipe["instantid_handoff_percent"] = instantid_handoff_percent', prepare_source)
        self.assertIn('end_at_step = int(pipe.get("instantid_end_at_step", 2))', sampler_source)
        self.assertIn('last_step=end_at_step', sampler_source)
        for path in INSTANTID_PATHS:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIn(
                    'start_at_step = 0 if zero_pass_mode else int(sampled_in.get("instantid_end_at_step", 2))',
                    source,
                )


if __name__ == "__main__":
    unittest.main()
