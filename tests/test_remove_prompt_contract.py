import ast
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "pipe" / "cmk_sampler_prepare.py"


def load_effective_prompt_function():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_effective_inpaint_prompts"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PATH), "exec"), namespace)
    return namespace["_effective_inpaint_prompts"]


class RemovePromptContractTests(unittest.TestCase):
    def test_remove_uses_user_prompt_and_adds_internal_negative_guard(self):
        function = load_effective_prompt_function()
        positive, negative, source = function(
            "futuristic botanical observatory", "low quality", True, "remove"
        )

        self.assertEqual(positive, "futuristic botanical observatory")
        self.assertIn("low quality", negative)
        self.assertIn("foreground subject", negative)
        self.assertEqual(source, "SOURCE + INTERNAL REMOVE GUARD")

    def test_remove_without_positive_prompt_uses_generic_fallback(self):
        function = load_effective_prompt_function()
        positive, negative, source = function("   ", "", True, "remove")

        self.assertIn("surrounding scene", positive)
        self.assertIn("foreground subject", negative)
        self.assertNotIn("sofa", positive.lower())
        self.assertNotIn("wall", positive.lower())
        self.assertEqual(source, "INTERNAL REMOVE GUIDANCE")

    def test_non_remove_modes_keep_source_prompts(self):
        function = load_effective_prompt_function()
        self.assertEqual(
            function("user positive", "user negative", True, "custom"),
            ("user positive", "user negative", "SOURCE"),
        )


if __name__ == "__main__":
    unittest.main()
