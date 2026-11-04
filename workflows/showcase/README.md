# CMK 2.5 Flow Browser workflows

This directory is the publication source for the **Referenzen / References** tab in the CMK Flow Browser.

The workflow JSON files and their same-named sidecars in `metadata/` reproduce
the prepared SHOWCASE structure. The published categories are `TASK WORKFLOWS`,
`MODULE WORKFLOWS`, `COMPARISONS`, `REAL-WORLD WORKFLOWS`, `SYSTEM WORKFLOW`,
and `LEGACY`.

Technical references that are not Flow Browser entries remain in
`workflows/reference` and are deliberately not shown in this tab.

Each published workflow needs a same-named sidecar file in `metadata/`. The sidecar controls its browser name, order, bilingual descriptions, CMK-specific highlight, and preview images. A workflow appears only when its sidecar contains `"published": true`.

Reference inputs that are required to reproduce a workflow belong in the CMK
package under `assets/references/`. Workflows select these assets through their
`CMK Package · …` entry; CMK reads and previews them in place and never copies
them into the user's ComfyUI input directory.

Persistent FaceSwap video project templates are maintained separately because
their project contract requires dedicated migration and compatibility checks.
