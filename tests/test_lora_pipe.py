import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "cmk_lora_pipe", ROOT / "pipe" / "cmk_lora_pipe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoRAPipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_family_packs_are_strictly_typed_and_complete(self):
        stack = [("example.safetensors", 0.8, 0.7)]
        sdxl = self.module.CMKLoRASDXLPackPipe().pack(
            stack, "example.safetensors", "example trigger"
        )[0]
        zit = self.module.CMKLoRAZITPackPipe().pack(stack, "zit", "zit trigger")[0]

        self.assertEqual("CMK_LORA_SDXL_PIPE", self.module.CMKLoRASDXLPackPipe.RETURN_TYPES[0])
        self.assertEqual("CMK_LORA_ZIT_PIPE", self.module.CMKLoRAZITPackPipe.RETURN_TYPES[0])
        self.assertEqual("sdxl", sdxl["family"])
        self.assertEqual(stack, sdxl["lora_stack"])
        self.assertEqual("example.safetensors", sdxl["active_loras"])
        self.assertEqual("example trigger", sdxl["trigger_words"])
        self.assertEqual("z_image_turbo", zit["family"])


if __name__ == "__main__":
    unittest.main()
