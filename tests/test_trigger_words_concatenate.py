from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE_ROOT / "nodes" / "utils" / "trigger_words_concatenate.py"
SPEC = spec_from_file_location("cmk_trigger_words_concatenate", MODULE_PATH)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TriggerWordsConcatenateTests(unittest.TestCase):
    def setUp(self):
        self.node = MODULE.CMKTriggerWordsConcatenate()

    def concatenate(self, first, second):
        return self.node.concatenate(**{
            "trigger_words_1": first,
            "trigger_words_2": second,
        })[0]

    def test_two_filled_inputs_preserve_order_format_and_duplicates(self):
        self.assertEqual(self.concatenate("alpha, beta", "alpha, gamma"), "alpha, beta,, alpha, gamma")

    def test_empty_first_input_is_identity(self):
        self.assertEqual(self.concatenate("", "alpha, beta"), "alpha, beta")

    def test_empty_second_input_is_identity(self):
        self.assertEqual(self.concatenate("alpha, beta", ""), "alpha, beta")

    def test_unconnected_second_input_is_identity(self):
        self.assertEqual(
            self.node.concatenate(trigger_words_1="alpha, beta")[0],
            "alpha, beta",
        )

    def test_both_empty_inputs_return_empty_string(self):
        result = self.concatenate("", "")
        self.assertEqual(result, "")
        self.assertIs(type(result), str)

    def test_three_stacks_can_be_chained(self):
        first_pair = self.concatenate("one", "two, three")
        self.assertEqual(self.concatenate(first_pair, "four"), "one,, two, three,, four")

    def test_cumulative_lora_stacker_output_is_not_duplicated(self):
        cumulative_ls2_output = "ls1 words,, ls2 words"
        self.assertEqual(
            self.node.concatenate(trigger_words_1=cumulative_ls2_output)[0],
            cumulative_ls2_output,
        )

    def test_port_contract_matches_lora_manager_string_contract(self):
        schema = MODULE.CMKTriggerWordsConcatenate.INPUT_TYPES()
        self.assertEqual(schema["required"]["trigger_words_1"][0], "STRING")
        self.assertTrue(schema["required"]["trigger_words_1"][1]["forceInput"])
        self.assertNotIn("trigger_words_2", schema["required"])
        self.assertEqual(schema["optional"]["trigger_words_2"][0], "STRING")
        self.assertTrue(schema["optional"]["trigger_words_2"][1]["forceInput"])
        self.assertEqual(MODULE.CMKTriggerWordsConcatenate.RETURN_TYPES, ("STRING",))
        self.assertEqual(MODULE.CMKTriggerWordsConcatenate.RETURN_NAMES, ("trigger_words",))


if __name__ == "__main__":
    unittest.main()
