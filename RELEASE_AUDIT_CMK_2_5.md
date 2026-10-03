# CMK 2.5 Release Audit

Date: 2026-10-03

Scope: classification of the 52 deviations from the first complete test run against the final CMK 2.5 contracts, published subgraphs, and SHOWCASE workflows.

## Result

- `STALE TEST`: 49
- `TEST / FIXTURE ISSUE`: 1
- `REAL RELEASE DEFECT`: 2
- `UNCLEAR`: 0

The audit also found two release defects that were hidden by earlier test aborts or outside the original 52 deviations: one invalid FaceProcess Advanced target slot and one orphaned link in the historical `CMK FaceRestore` example. Both were corrected.

## Classification of the original 52 deviations

| # | Deviation | Classification | Resolution |
|---:|---|---|---|
| 1 | FaceProcess Advanced transport test used removed internal node IDs | STALE TEST | Resolve nodes and chains by type and named ports. |
| 2 | FaceSwap Advanced transport test used removed internal node IDs | STALE TEST | Resolve execute nodes and log/diagnostic chain semantically. |
| 3 | FaceProcess Advanced expected the removed native `ImageCompare` topology | STALE TEST | Validate the current internal `CMKVisualCompare` side path. |
| 4 | Upscale & Save expected obsolete result unpack/pack nodes | STALE TEST | Validate the direct neutral result contract. |
| 5 | Full Flow expected old hard-coded node/link IDs and a separate boundary subgraph | STALE TEST | Validate family merge, neutral result handoff, postprocess chain, and terminal Visualizer semantically. |
| 6 | ZIT finalize expected the former boundary slot layout | STALE TEST | Validate current named finalize/boundary/gate routes. |
| 7 | ControlNet Combined expected old 450×200 layout and title-based `variantOf` | STALE TEST | Use 300×190 clean-view contract and technical UUID identity. |
| 8 | ControlNet ZIT expected old 450×200 layout | STALE TEST | Use current 300×190/300×100 clean-view sizes. |
| 9 | Detailer Advanced expected 450×230 | STALE TEST | Use current 300×190 layout. |
| 10 | Detailer Advanced link `20022` targeted slot `-51` instead of named `VISUAL` slot `33` | REAL RELEASE DEFECT | Corrected serialized target slot and revalidated both endpoint backrefs. |
| 11 | Detailer Standard expected old size and disabled default | STALE TEST | Use current size and enabled module default. |
| 12 | FaceProcess Standard expected old size metadata | STALE TEST | Use current expanded and clean-view sizes. |
| 13 | FaceRebuild Advanced expected 450×230 | STALE TEST | Use current 300×190 layout. |
| 14 | FaceRebuild Standard expected old size and absence of an empty preview declaration | STALE TEST | Validate current explicit empty declaration. |
| 15 | FaceRebuild Standard expected obsolete public `image` combo | STALE TEST | Validate internal packaged reference plus `opt_image_file`. |
| 16 | FaceSwap Standard expected 450×230 | STALE TEST | Use current 300×190 layout. |
| 17 | FaceSwap Advanced expected 450×230 | STALE TEST | Use current 300×190 layout. |
| 18 | FaceSwap Advanced link `16040` targeted slot `-23` instead of named `VISUAL` slot `33` | REAL RELEASE DEFECT | Corrected serialized target slot and revalidated both endpoint backrefs. |
| 19 | FaceSwap Standard expected disabled default and obsolete source picker contract | STALE TEST | Validate enabled default and `opt_image_file`. |
| 20 | Unpublished legacy Toolbox FaceSwap fixture still serializes the pre-2.5 boundary PROCESS port | TEST / FIXTURE ISSUE | Excluded explicitly from tests for published 2.5 boundary contracts; retained as unpublished historical material. |
| 21 | FaceProcess prepare expected SDXL-only PROCESS | STALE TEST | Validate the final family-neutral `CMK_RESULT_PROCESS`. |
| 22 | FaceProcess Standard expected old process-forward class | STALE TEST | Validate `CMKResultProcessForwardPipe`. |
| 23 | FaceProcess Advanced expected old process-forward class | STALE TEST | Validate `CMKResultProcessForwardPipe`. |
| 24 | FaceProcess restore test expected cache schema v7 | STALE TEST | Validate current v8 cache schema. |
| 25 | Upscale & Save expected removed `PROJECT FOLDER` runtime input | STALE TEST | Match current saved runtime input order. |
| 26 | Detailer Standard expected SDXL-only PROCESS | STALE TEST | Validate family-neutral postprocess contract. |
| 27 | Detailer Advanced expected SDXL-only PROCESS | STALE TEST | Validate family-neutral postprocess contract. |
| 28 | FaceProcess Standard expected SDXL-only PROCESS | STALE TEST | Validate family-neutral postprocess contract. |
| 29 | FaceProcess Advanced expected SDXL-only PROCESS | STALE TEST | Validate family-neutral postprocess contract. |
| 30 | FaceSwap Standard expected obsolete public source picker | STALE TEST | Validate final public inputs. |
| 31 | FaceSwap Advanced omitted final `opt_image_file` input | STALE TEST | Validate final public inputs. |
| 32 | Upscale & Save expected obsolete model/folder controls and omitted VISUAL | STALE TEST | Validate compact final public contract. |
| 33 | Stand-alone image loader test omitted final `image_file` return | STALE TEST | Validate complete neutral result return contract. |
| 34 | Brownian warning guard expected three scopes | STALE TEST | Validate four current sampling scopes. |
| 35 | InstantID English content lookup used visible title as identity | STALE TEST | Use technical subgraph UUID. |
| 36 | InstantID subgraph expected old dimensions, definition UUID, and widget serialization | STALE TEST | Validate current technical identity and compact contract. |
| 37 | ControlNet SDXL compact-size assertion expected 450×230 | STALE TEST | Use 300×190 expanded / 300×100 clean view. |
| 38 | KSampler SDXL compact-size assertion expected 450×230 | STALE TEST | Use current 300×190. |
| 39 | InstantID compact-size assertion expected 450×230 | STALE TEST | Use current 300×190. |
| 40 | Refiner compact-size assertion expected 450×230 | STALE TEST | Use current 300×190. |
| 41 | Detailer Standard compact-size assertion expected 450×230 | STALE TEST | Use current dimensions. |
| 42 | Detailer Advanced compact-size assertion expected 450×230 | STALE TEST | Use current dimensions. |
| 43 | FaceRebuild Standard compact-size assertion expected 450×230 | STALE TEST | Use current dimensions. |
| 44 | FaceRebuild Advanced compact-size assertion expected 450×230 | STALE TEST | Use current dimensions. |
| 45 | FaceProcess Standard compact-size assertion expected 450×230 | STALE TEST | Use current dimensions. |
| 46 | Sampling VISUAL test assumed VISUAL was the last widget input and fixed live node IDs | STALE TEST | Validate named VISUAL port and runtime-remappable provider identity. |
| 47 | SDXL ControlNet expected obsolete public reference-image combo | STALE TEST | Validate packaged internal reference and current public contract. |
| 48 | SDXL ControlNet expected old 450×230 sizes | STALE TEST | Use current expanded and clean-view sizes. |
| 49 | ZIT catalog test expected superseded wording and recommendation representation | STALE TEST | Match final editorial metadata and technical targets. |
| 50 | ZIT native chain expected public IMAGE to bypass the HYBRID input bridge | STALE TEST | Validate `CMKHybridZITInputPipe` handoff. |
| 51 | ZIT public contract omitted VISUAL and SDXL handoff inputs | STALE TEST | Validate the final direct/HYBRID-compatible contract. |
| 52 | ZIT ControlNet recommendation expected the previous KSampler UUID | STALE TEST | Use the final KSampler ZIT UUID. |

## Additional defects found during the audit

1. `CMK Flow · FaceProcess SDXL · Advanced.json`: link `16020` used target slot `-23`; corrected to the named `VISUAL` input at slot `33`.
2. `workflows/examples/CMK FaceRestore.json`: link `12087` targeted removed node `4938`; removed the orphaned link and its output back-reference.

## Verification

- Complete unit suite: `359` tests, all passing.
- JSON parse: all release JSON files passed.
- Python AST parse: all release Python files passed.
- JavaScript syntax: all browser JavaScript files passed `node --check`.
- Structural graph validation: `183` embedded subgraphs and `3962` links; no invalid slot, missing node, or broken endpoint back-reference.
- `git diff --check`: clean.

No commit, tag, or push was performed as part of this audit.
