import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LogExportContractTests(unittest.TestCase):
    def test_export_accepts_pipeline_and_family_result_logs(self):
        source = (ROOT / "pipe" / "cmk_log_pipe.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        export_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CMKLogExportText"
        )
        input_types = next(
            node
            for node in export_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "INPUT_TYPES"
        )
        returned = next(
            node.value for node in ast.walk(input_types) if isinstance(node, ast.Return)
        )
        contract = ast.literal_eval(returned)["required"]["log_pipe"][0]

        self.assertEqual("CMK_LOG_PIPE,CMK_RESULT_LOG", contract)


if __name__ == "__main__":
    unittest.main()
