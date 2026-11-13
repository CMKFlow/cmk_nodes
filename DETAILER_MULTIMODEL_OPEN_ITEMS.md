# CMK Flow 2.7.1 – Detailer Multi-Model: offene Punkte vor der Modulabnahme

> Stand 2026-11-13: Der Detailer-Stand ist für 2.8.0 abgenommen und paketiert
> (siehe `CHANGELOG.md`, `ARCHITECTURE.md` 5.6a/5.6b). Dieses Dokument bleibt
> als Entscheidungsverlauf erhalten.

Stand: 2026-11-11. Dieses Dokument sammelt offene Layout- und Vertragsfragen
des Subgraphen `CMK Flow · Detailer`. Der gültige Vertrag steht in
`ARCHITECTURE.md` (5.2, 5.6a). Bis zur Klärung bleiben Legacy-Nodes,
MODEL-Verträge und Cache-Verhalten unverändert.

## 1. Layout der Innenansicht – erledigt

- Prepare ist in Vue im SDXL-Zustand (Advanced eingeklappt) etwa 940 Einheiten
  hoch, gespeichert sind 590. Die Gate-Zeile begann bei y = 2555 und wurde
  dadurch überdeckt.
- **Korrektur:** Die untere Zeile (Module Bypass Gate, Boundary, Result Process
  Forward, beide Model Forwards, Family Enable) und der Subgraph-Ausgang sind um
  375 Einheiten nach unten gerückt. Der Gruppenrahmen umschließt die obere
  Zeile. Die waagrechte Anordnung der Vorlage ist unverändert.
- **Nachweis:** DOM-Messung im isolierten Vue-Frontend, keine Überdeckung im
  SDXL-Zustand. Ein aufgeklappter Advanced-Bereich vergrößert Prepare weiterhin
  über die Gate-Zeile hinaus.

## 2. Visualizer: `MODEL ZIT` ist kein Visualizer-Eingang

- **Befund:** `MODEL ZIT` ist ein Forward-Ausgang für nachgelagerte
  Multi-Model-Module. Am gemeinsamen Eingang `MODEL (opt)` des Visualizers
  lehnt `CMK Result Unpack` ein ZIT-Modell bei aktivem Detailer ab („received
  MODEL and PROCESS from different families“). Zulässig ist dort neben dem
  neutralen Bildpfad nur ein SDXL-PostProcess-Modell
  (`pipe/cmk_family_result.py`, `valid_postprocess_model`).
- **Nachweis:** PromptExecutor mit den vom Frontend expandierten Prompts: Bei
  SDXL und ZIT mit `MODEL ZIT` am Visualizer gibt es den Fehler; bei `None`
  läuft derselbe Graph durch, weil der Pfad die Prüfung nicht erreicht. In
  keinem Fall wurde ZIT materialisiert.
- **Gelöst (2.7.1):** Der Visualizer hat in der Gruppe `POSTPROCESS` eigene
  Eingänge `MODEL ZIT` und `MODEL SDXL`. `MODEL ZIT` (intern
  `POSTPROCESS MODEL ZIT`) nimmt das ZIT-Modell auf, ohne es anzufordern;
  `MODEL SDXL` ist der bisherige `MODEL (opt)`. Der Result-Unpack-Vertrag ist
  unverändert (`ARCHITECTURE.md` 3, Visualizer).
- **Upscale & Save:** hat weiterhin nur einen MODEL-Eingang (SDXL).

## 3. Visualizer-Provider des Multi-Model Smart Detailers

- `CMK_SmartDetailerMultiModelPipe` registriert seinen VISUAL-Provider mit den
  SDXL-Kennungen (`branch = sdxl`, `stage_key = sdxl.detailer.standard`), auch bei
  ZIT. Diese Werte verbinden den Laufzeit-Provider mit der Provider-Deklaration
  des Subgraphen. `branch` wird nur zum Abgleich verwendet und nicht angezeigt.
- **Kleinster sicherer Weg (umgesetzt):** Die Kennungen bleiben unverändert;
  der VISUAL-Vertrag ändert sich nicht. `cmk_visualizer.js` ordnet die
  Live-Vorschau zusätzlich dem Typ `CMK_SmartDetailerMultiModelPipe` zu; vorher
  blieb sie ohne Zuordnung.
- **Offen:** Eigene Multi-Model-Kennungen würden Deklaration, Laufzeit-Provider
  und Visualizer-Abgleich gleichzeitig betreffen und brauchen eine Freigabe.

## 4. Modellanschlüsse des Subgraphen

- **`MODEL SDXL` ist Pflicht**, auch bei reiner ZIT-Nutzung: Prepare und
  Detailer-Boundary benötigen ihn. Erwartet wird die SDXL-PostProcess-
  Spezifikation (PostProcess Boundary, `CMK PostProcess Model` oder `CMK PostProcess Model SDXL`, auch mit
  Checkpoint `None`). Ein sofort ladender SDXL-Loader an dieser Stelle würde
  über Boundary und Forward ausgeführt.
- **`MODEL ZIT` kommt verbindlich von einer ZIT-PostProcess-Spec:**
  `CMK PostProcess Model` (gemeinsame Quelle, Ausgang `MODEL ZIT`) oder
  `CMK Z-Image Turbo PostProcess Model`.
  Ein direkt angeschlossener `CMK Z-Image Turbo Loader -Pipe-` lädt sofort, und
  ComfyUI hält seine Ausgabe im Ausgabecache; die ZIT-Freigabe vor SDXL bleibt
  dann wirkungslos (`KEPT · IN USE`). Realvalidierung siehe `ARCHITECTURE.md` 5.2.

## 5. Mehrere Instanzen

- MODEL FAMILY ist pro Instanz eigenständig: Ausführung, gespeicherte Werte und
  Innenansicht folgen der jeweiligen Instanz. Die Innenansicht wird beim Betreten
  (`subgraph-opened`) an die betretene Instanz gekoppelt.
- **Gemeinsame interne Parameter (Befund):** Alle übrigen Einstellungen des
  Subgraphen (Prepare-, Smart-Detailer- und Preview-Widgets) liegen in der
  gemeinsamen Subgraph-Definition und gelten für alle Instanzen. Geprüft im
  Frontend: Eine Änderung der SDXL-Steps erscheint im Prompt beider Instanzen
  (41 / 41), während MODEL FAMILY getrennt bleibt (SDXL / None). Das ist das
  Verhalten von ComfyUI-Subgraphen; eigene Werte pro Instanz wären nur über
  weitere Außen-Widgets möglich.
- **Festlegung:** Die Parameterarchitektur bleibt in diesem Schritt unverändert;
  es gibt keine instanzspezifische Parameterverwaltung. Wie mehrere Detailer im
  Advanced-Subgraphen individuell konfiguriert werden, wird bei dessen eigener
  Architekturplanung entschieden.

## 6. Boundary-Aufbewahrung je Modellfamilie – erledigt

- **Befund:** Die Detailer-Boundary behielt pro Node nur einen Eintrag; ein
  Wechsel SDXL ↔ ZIT löschte jeweils den Eintrag der anderen Familie. Folge:
  echter Boundary-MISS, Prepare lief erneut, bei ZIT wurde der Text-Encoder
  neu geladen.
- **Korrektur:** Slot pro Node und Familie nur für den Multi-Model-Detailer
  (`ARCHITECTURE.md` 5.6a). Legacy und andere Boundaries unverändert.
- **Offen (nicht Teil dieser Korrektur):** Ein beschädigtes Boundary-Bild
  (HIT_FAILED) bricht den Lauf mit „cache miss requires IMAGE, LOG“ ab, statt
  neu zu berechnen; der nächste Lauf rechnet korrekt. Betrifft beide
  Generationen (`_DetailerBoundaryCore.boundary`).
- **Hinweis:** Ein vor der Korrektur unter dem einfachen Slot gespeicherter
  Eintrag der jeweils anderen Familie bleibt bis zu seiner Wiederverwendung
  liegen (höchstens ein Eintrag pro Node); es gibt keine Bereinigung.

## 7. Untersuchungshinweis

Weichen Harness und Realbetrieb ab, zuerst die tatsächliche Workflow-Verdrahtung
einschließlich der Modellquellen mit der Referenzarchitektur vergleichen. Der
fehlgeschlagene Realtest vom 2026-11-12 ging auf einen ZIT-Loader statt der
ZIT-PostProcess-Spec zurück.

## 8. Weitere Beobachtungen

- **Advanced-Variante:** `CMK SEGS CONCAT` kennt den Multi-Model Smart Detailer
  noch nicht (`nodes/image/segs_concate.py`).
- **Speicher ZIT → SDXL:** abgesicherte ZIT-Freigabe im SDXL-Zweig des
  Multi-Model Prepare, real validiert am 2026-11-12 (`ARCHITECTURE.md` 5.2).
  Offen: ZIT ist auf 16 GB auch
  allein langsam (etwa 90 s/Step, bf16 11,7 GB); bei ZIT → SDXL → ZIT in einem
  Prompt bleibt ZIT geladen (Familien im Advanced-Subgraphen gruppieren); der
  SDXL-Detailer-Prepare gibt seinen CLIP nicht frei (Legacy, unverändert).
