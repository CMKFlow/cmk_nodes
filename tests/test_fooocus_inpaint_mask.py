import unittest
from pathlib import Path


class FooocusInpaintMaskTests(unittest.TestCase):
    def test_sampler_uses_expanded_mask(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "engine"
            / "fooocus_inpaint.py"
        ).read_text(encoding="utf-8")

        self.assertIn("grow_mask_by: int = 16", source)
        self.assertIn("def grow_mask_for_sampling", source)
        self.assertIn("grown = F.conv2d(", source)
        self.assertIn("sampler_mask = grow_mask_for_sampling", source)
        self.assertIn('sampler_latent["noise_mask"] = sampler_mask', source)

    def test_create_image_exposes_bounded_advanced_mask_expand(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "pipe"
            / "cmk_pipe_image.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"inpaint_mask_expand": (', source)
        self.assertIn('"default": 16', source)
        self.assertIn('"min": 4', source)
        self.assertIn('"max": 32', source)
        self.assertIn('"inpaint_mask_expand": inpaint_mask_expand', source)


if __name__ == "__main__":
    unittest.main()
