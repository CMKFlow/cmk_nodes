import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def _load_module():
    node_helpers = types.ModuleType("node_helpers")
    node_helpers.conditioning_set_values = lambda conditioning, *_args, **_kwargs: conditioning
    sys.modules["node_helpers"] = node_helpers
    path = Path(__file__).resolve().parents[1] / "engine" / "context_reference.py"
    spec = importlib.util.spec_from_file_location("cmk_context_reference_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContextReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_reference_latent_excludes_masked_generation_area(self):
        samples = torch.ones((1, 4, 2, 2))
        mask = torch.zeros((1, 16, 16))
        mask[:, :, 8:] = 1

        reference = self.module.mask_reference_latent(samples, mask)

        self.assertTrue(torch.allclose(reference[:, :, :, 0], torch.ones((1, 4, 2))))
        self.assertTrue(torch.allclose(reference[:, :, :, 1], torch.zeros((1, 4, 2))))

    def test_positive_expand_and_blur_feather_only_outside_source_mask(self):
        source = torch.zeros((1, 32, 32))
        source[:, 12:20, 12:20] = 1
        conditioning = [[torch.zeros((1, 1)), {}]]
        latent = {"samples": torch.zeros((1, 4, 4, 4))}

        _, _, processed = self.module.CMKContextReferenceLatentMask().prepare(
            conditioning,
            latent,
            source,
            expand=3,
            blur=5.0,
            mask_only=True,
        )

        self.assertTrue(torch.all(processed[:, 12:20, 12:20] == 1))
        self.assertGreater(float(processed[:, 8:12, 12:20].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
