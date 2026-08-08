# Next larger session

## CMK Flow · 01 START HERE · Create Image

### Aspect-ratio-safe preparation for `Extend Image`

Current issue:

- Resizing stretches the input image to the selected target dimensions.
- A crop option is missing.
- The input image also needs an option to be fitted into the target canvas without distortion.
- Proper fitting and the resulting uncovered canvas area are prerequisites for Outpaint / Process Mode `Extend Image`.

Expected outcome:

- Preserve the source aspect ratio.
- Provide suitable crop and fit behavior instead of forced stretching.
- Generate a correct image and mask for the uncovered target area.
- Make `Extend Image` usable as an actual outpainting workflow.
