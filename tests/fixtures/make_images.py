"""Create the deterministic S1 input images (no randomness across runs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent / "images"


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(7).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(HERE / "cmk_s1_input.png")
    # Same file name, different content: used by the file-replacement cache test.
    replaced = np.random.default_rng(8).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    Image.fromarray(replaced).save(HERE / "cmk_s1_input_replacement.png")


if __name__ == "__main__":
    main()
