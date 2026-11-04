from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SAMPLER_PATHS = (
    PACKAGE_ROOT / "pipe" / "instantid" / "cmk_instantid_sampler.py",
    PACKAGE_ROOT / "pipe" / "instantid" / "cmk_instantid_regional_advanced.py",
)
CONTROLNET_PATH = PACKAGE_ROOT / "pipe" / "controlnet" / "cmk_controlnet_prepare.py"


def _sample_method(path: Path) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "_sample":
                    return member
    raise AssertionError(f"No _sample method in {path}")


class InstantIDReferenceLatentRemovalTests(unittest.TestCase):
    def test_active_reference_latent_contract_is_removed(self):
        forbidden = (
            "instantid_reference_latent_mode",
            "instantid_reference_image",
            "reference_latent_mode",
            "REFERENCE-LATENT",
        )
        for path in (*SAMPLER_PATHS, CONTROLNET_PATH):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                for token in forbidden:
                    self.assertNotIn(token, source)

    def test_legacy_value_is_accepted_but_never_read(self):
        for path in SAMPLER_PATHS:
            source = path.read_text(encoding="utf-8")
            sample = _sample_method(path)
            arguments = {argument.arg for argument in sample.args.args}
            loaded_names = {
                node.id
                for node in ast.walk(sample)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            with self.subTest(path=path.name):
                self.assertIn('"reference_start_at_step": ("INT", {', source)
                self.assertIn('"default": 8, "min": 0, "max": 200, "step": 1', source)
                self.assertIn("reference_start_at_step", arguments)
                self.assertNotIn("reference_start_at_step", loaded_names)

    def test_active_start_step_is_zero_pass_or_central_handoff_only(self):
        expected = (
            'start_at_step = 0 if zero_pass_mode else '
            'int(sampled_in.get("instantid_end_at_step", 2))'
        )
        for path in SAMPLER_PATHS:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn(expected, source)
                self.assertIn(
                    'add_noise = "enable" if (zero_pass_mode or inpaint_bridge_mode) else "disable"',
                    source,
                )

    def test_bundled_workflow_json_remains_parseable(self):
        workflow_paths = sorted(
            path
            for folder in (PACKAGE_ROOT / "workflows", PACKAGE_ROOT / "subgraphs")
            if folder.exists()
            for path in folder.rglob("*.json")
        )
        self.assertTrue(workflow_paths)
        for path in workflow_paths:
            with self.subTest(path=str(path.relative_to(PACKAGE_ROOT))):
                with path.open("r", encoding="utf-8") as handle:
                    json.load(handle)


if __name__ == "__main__":
    unittest.main()
