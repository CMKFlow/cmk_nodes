from __future__ import annotations

import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SAMPLER_PATH = PACKAGE_ROOT / "pipe" / "cmk_pipe_sampler.py"
BOUNDARY_PATH = PACKAGE_ROOT / "pipe" / "cmk_family_result.py"
INSTANTID_PATHS = (
    PACKAGE_ROOT / "pipe" / "instantid" / "cmk_instantid_sampler.py",
    PACKAGE_ROOT / "pipe" / "instantid" / "cmk_instantid_regional_advanced.py",
)


def _load_handoff_guard(namespace):
    tree = ast.parse(SAMPLER_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_guard_instantid_handoff_x0"
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SAMPLER_PATH), "exec"),
        namespace,
    )
    return namespace["_guard_instantid_handoff_x0"]


class InstantIDHandoffContentGuardTests(unittest.TestCase):
    def test_handoff_guard_decodes_and_inspects_the_supplied_x0(self):
        calls = []
        latent = object()
        image = object()

        class Batch:
            shape = (1, 1, 1, 3)

            def __iter__(self):
                return iter((image,))

        class VAEDecode:
            def decode(self, vae, samples):
                calls.append(("decode", vae, samples))
                return (Batch(),)

        class Guard:
            def inspect_content(self, rgb, role):
                calls.append(("inspect", rgb, role))

        class Blocked(RuntimeError):
            pass

        function = _load_handoff_guard({
            "ContentGuardBlocked": Blocked,
            "get_content_guard": lambda: Guard(),
            "tensor_to_uint8_rgb": lambda value: ("rgb", value),
        })
        with patch.dict(sys.modules, {"nodes": types.SimpleNamespace(VAEDecode=VAEDecode)}):
            function(latent, "vae")

        self.assertEqual(calls[0], ("decode", "vae", latent))
        self.assertEqual(calls[1], ("inspect", ("rgb", image), "target_intermediate"))

    def test_handoff_guard_block_propagates(self):
        class Batch:
            shape = (1, 1, 1, 3)

            def __iter__(self):
                return iter((object(),))

        class VAEDecode:
            def decode(self, vae, samples):
                return (Batch(),)

        class Blocked(RuntimeError):
            def __init__(self, code="blocked", role="target_intermediate"):
                self.code = code
                self.role = role
                super().__init__(code, role)

        class Guard:
            def inspect_content(self, rgb, role):
                raise Blocked()

        function = _load_handoff_guard({
            "ContentGuardBlocked": Blocked,
            "get_content_guard": lambda: Guard(),
            "tensor_to_uint8_rgb": lambda value: value,
        })
        with patch.dict(sys.modules, {"nodes": types.SimpleNamespace(VAEDecode=VAEDecode)}):
            with self.assertRaises(Blocked):
                function(object(), "vae")

    def test_existing_sdxl_x0_is_guarded_before_inpaint_bridge(self):
        source = SAMPLER_PATH.read_text(encoding="utf-8")
        x0 = source.index('keypoints_latent["samples"] = model.model.process_latent_out(')
        guard = source.index("_guard_instantid_handoff_x0(keypoints_latent, vae)", x0)
        bridge = source.index('if pipe.get("boolean_inpaint_mode", False):', guard)
        self.assertLess(x0, guard)
        self.assertLess(guard, bridge)
        self.assertIn('samples["samples"] = sampled_tensor', source)

    def test_handoff_block_uses_existing_abort_cleanup_and_cannot_return_sampled(self):
        source = SAMPLER_PATH.read_text(encoding="utf-8")
        self.assertIn("except ContentGuardBlocked as exc:", source)
        self.assertIn('label="10 SDXL InstantID Handoff"', source)
        self.assertIn("cleanup_content_guard_abort(", source)
        self.assertIn("raise exc.with_traceback(None) from None", source)
        self.assertIn('new_pipe["content_guard_handoff"] = content_guard_handoff_status', source)

    def test_module_15_paths_require_the_handoff_guard_and_have_no_midpoint(self):
        for path in INSTANTID_PATHS:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIn('sampled_in.get("content_guard_handoff") != "PASS · SDXL HANDOFF"', source)
                self.assertIn('raise ContentGuardBlocked("CG_HANDOFF_UNVERIFIED"', source)
                self.assertIn("result = _sample_continuation(**sampling_args)", source)
                self.assertNotIn("_sample_with_midpoint_guard", source)
                self.assertNotIn("_guard_midpoint_x0", source)
                self.assertNotIn("Early Guard", source)
                self.assertNotIn("50% x0", source)

    def test_source_and_final_target_guards_remain_active(self):
        for path in INSTANTID_PATHS:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIn("_guard_source_images", source)
                self.assertIn("_guard_generated_latent", source)
                self.assertIn("if content_guard_enabled:", source)
                self.assertIn("SOURCE + GENERATED TARGET", source)

    def test_zero_pass_skips_only_the_handoff_requirement(self):
        for path in INSTANTID_PATHS:
            source = path.read_text(encoding="utf-8")
            self.assertIn("and not zero_pass_mode", source)
            self.assertIn(
                'add_noise = "enable" if (zero_pass_mode or inpaint_bridge_mode) else "disable"',
                source,
            )
            self.assertNotIn("reference_latent_mode", source)

    def test_first_pass_boundary_cache_includes_content_guard_state(self):
        source = BOUNDARY_PATH.read_text(encoding="utf-8")
        # Lazy check, execution, and the STOP-preview lookup must all address
        # the same guard-qualified first-pass boundary.
        self.assertEqual(source.count("include_content_guard=True"), 3)
        self.assertIn('payload["content_guard"] = content_guard_prompt_token(prompt)', source)


if __name__ == "__main__":
    unittest.main()
