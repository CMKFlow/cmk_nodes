# CMK Flow Referenzworkflows / Reference Workflows

Dieses Verzeichnis ist die Veröffentlichungsquelle für den Reiter **Referenzen / References** im CMK Flow Browser.

This directory is the publication source for the **Referenzen / References** tab in the CMK Flow Browser.

Only complete, curated example workflows belong here. They are copied from the
confirmed workflows in the CMK user directory and should teach the structure,
signal flow, and intended use of CMK Flow while remaining suitable for opening
as an unsaved starting copy.

Technical module references and standalone test workflows remain in `workflows/reference` and are deliberately not shown in this tab.

Jeder veröffentlichte Workflow benötigt eine gleichnamige Sidecar-Datei in `metadata/`. Sie steuert Browsernamen, Reihenfolge, zweisprachige Beschreibungen, CMK-Hervorhebung und Vorschaubilder. Ein Workflow erscheint nur mit `"published": true`.

Die Reihenfolge lautet: zuerst `Flow #01` bis `Flow #10`, danach `Ref #01` bis `Ref #04`, anschließend die übrigen vollständigen Workflows.

Each published workflow needs a same-named sidecar file in `metadata/`. The sidecar controls its browser name, order, bilingual descriptions, CMK-specific highlight, and preview images. A workflow appears only when its sidecar contains `"published": true`.

The order is: `Flow #01` through `Flow #10`, followed by `Ref #01` through `Ref #04`, then the remaining complete workflows.

Benötigte Bildquellen liegen im CMK-Paket unter `assets/references/`. Workflows wählen sie über ihren Eintrag `CMK Package · …`; CMK liest und zeigt sie direkt aus dem Paket an, ohne sie in das ComfyUI-Eingabeverzeichnis des Benutzers zu kopieren.

Reference inputs required to reproduce a workflow belong in the CMK package under `assets/references/`. Workflows select these assets through their `CMK Package · …` entry; CMK reads and previews them in place and never copies them into the user's ComfyUI input directory.

`CMK FaceSwap Video.json` is maintained separately because its persistent video
project contract requires dedicated migration and compatibility checks.
