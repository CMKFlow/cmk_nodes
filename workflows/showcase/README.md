# CMK Flow example workflows

This directory is the publication source for the **Referenzen / References** tab in the CMK Flow Browser.

Only complete, curated example workflows belong here. They are copied from the
confirmed workflows in the CMK user directory and should teach the structure,
signal flow, and intended use of CMK Flow while remaining suitable for opening
as an unsaved starting copy.

Technical module references and standalone test workflows remain in `workflows/reference` and are deliberately not shown in this tab.

Each published workflow needs a same-named sidecar file in `metadata/`. The sidecar controls its browser name, order, bilingual descriptions, CMK-specific highlight, and preview images. A workflow appears only when its sidecar contains `"published": true`.

Reference inputs that are required to reproduce a workflow belong in the CMK
package under `assets/references/`. Workflows select these assets through their
`CMK Package · …` entry; CMK reads and previews them in place and never copies
them into the user's ComfyUI input directory.

`CMK FaceSwap Video.json` is maintained separately because its persistent video
project contract requires dedicated migration and compatibility checks.
