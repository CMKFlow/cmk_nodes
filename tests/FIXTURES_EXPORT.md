# S1-T4 — API-Prompt-Fixtures exportieren

Die Integrationstests brauchen API-Prompts mit expandierten Subgraphen
(`"outer:inner"`-IDs). Diese Expansion leistet **ausschließlich** das
ComfyUI-Frontend (`app.graphToPrompt()`). Sie wird bewusst **nicht** in Python
nachgebaut, weil ein eigener Konverter von der echten Expansion abweichen und
falsche Testerfolge erzeugen könnte.

Für den Export werden keine Workflows ausgeführt und keine Modelle geladen.
Nicht auf „Queue“ klicken.

## Verwendete Nodes (aus der tatsächlichen Registrierung, `cmk_mappings.py`)

| Rolle | Python-Klasse | Anzeigename in der Node-Library | Kategorie |
|---|---|---|---|
| Bildeingang | `CMKImageLoadAndResizePipe` | **`CMK Load Image`** | `CMK/Flow/Input` |
| SDXL-Modell (nur wo nötig, s. u.) | `CMKCheckpointVAELoaderPipe` | **`CMK Flow · Checkpoint & VAE`** | `CMK/Flow/Input` |
| Endpunkt | `CMKVisualizer` | in der Library unter `CMK/Flow/Core` | `CMK/Flow/Core` |

**Verwechslungsgefahr:** Die Suche nach „CMK Load Image“ findet auch
`CMK Flow · Load Image` (`CMKLoadImage`). Das ist eine **andere** Node.
Richtig ist die Node mit den Widgets `RESOLUTION`, `SWAP DIMENSIONS`,
`RESIZE METHOD`, `CROP`, `CROP POSITION` und dem optionalen Eingang
`MODEL SDXL (opt)`.

Der in ARCHITECTURE.md und CHANGELOG verwendete Name
„CMK Image Load and Resize -Pipe-“ ist **historisch**. Er erscheint nur noch
in LOG- und Diagnose-Texten der Node, nicht in der Library.

## Ausgänge des Bildeingangs ohne Checkpoint (`cmk_image_load_resize.py:≈420`)

`CMK Load Image` gibt `MODEL / PROCESS / IMAGE / LOG / diagnostic / MASK / image_file` aus.
- **PROCESS:** `CMK_RESULT_PROCESS`, familienneutral (`pipe_origin = "CMK Image Load and Resize -Pipe-"`).
- **MODEL** ist **exakt der Wert am optionalen Eingang `MODEL SDXL (opt)`**. Ohne
  angeschlossenen Checkpoint ist MODEL zur Laufzeit `None`.
- **VISUAL** gibt der Loader nicht aus. Der VISUAL-Eingang des ersten Moduls bleibt unverbunden; das ist zulässig (leere Pipe).

Gültigkeit ohne Checkpoint:

| Modul | Export gültig | Bypass ausführbar | aktive Verarbeitung ausführbar |
|---|---|---|---|
| FaceSwap, FaceSwap · Advanced | ja | ja | ja (`MODEL (opt)` ist optional) |
| Upscale & Save | ja | ja | ja (kein MODEL-Eingang) |
| Detailer, FaceProcess, FaceRebuild (je Std./Adv.), MaskDetailer | ja | ja | **nein**: Prepare bzw. Compute verlangt eine CMK-Model-Pipe (`resolve_postprocess_model` bzw. `isinstance(MODEL, dict)`) und scheitert mit `None` |

Deshalb bekommen die SDXL-abhängigen Fixtures zusätzlich
`CMK Flow · Checkpoint & VAE` am Eingang `MODEL SDXL (opt)`. Beim Export wird
nichts geladen. Im späteren Harness-Lauf (S1-T6) wird das Laden des Checkpoints
über eine Nahtstelle ersetzt; T1 lädt also kein Modell.

## Vorbereitung

1. ComfyUI dieser Installation starten (Comfy Desktop, Instanz „DEV 2b.7.0.1 Claude“).
   Frontend laut *Settings → About*: **1.57.4**. Diesen Wert später an
   `update_manifest.py` übergeben.

   Hinweis: Die Desktop-App liefert ihr Frontend selbst aus. Das in
   `standalone-env` installierte `comfyui-frontend-package` 1.49.6 ist nicht
   das tatsächlich verwendete Frontend.
2. Testbild in das Eingangsverzeichnis **der laufenden Instanz** kopieren. Die
   Instanz startet mit `--input-directory ~/ComfyUI-Shared/input`:
   - `tests/fixtures/images/cmk_s1_input.png` → `~/ComfyUI-Shared/input/cmk_s1_input.png`

   Der Harness verwendet ein eigenes, isoliertes Eingangsverzeichnis; nur der
   Dateiname muss übereinstimmen.
3. Für die SDXL-abhängigen Fixtures **einen** SDXL-Checkpoint wählen und
   denselben Namen in allen Fixtures verwenden. Er wird nur referenziert, nicht
   geladen.

## Pro Fixture

Jede Fixture besteht aus einem gespeicherten Workflow und dessen API-Export mit
**identischem Dateinamen**:

| Datei | Ablage |
|---|---|
| Workflow (Menü *Workflow → Export*) | `tests/fixtures/workflows/<name>.json` |
| API-Prompt (Menü *Workflow → Export (API)*) | `tests/fixtures/api_prompts/<name>.json` |

*Export (API)* speichert genau die Ausgabe von `app.graphToPrompt().output`.

### Grundaufbau A (ohne Checkpoint): FaceSwap, FaceSwap · Advanced, Upscale & Save

```text
CMK Load Image  (image = cmk_s1_input.png, RESOLUTION = 1024x1024, übrige Widgets Standard,
                 MODEL SDXL (opt) unverbunden)
   │ MODEL / PROCESS / IMAGE / LOG
   ▼
<kanonisches Modul aus der Node-Library „CMK Flow · …“>
   │ MODEL / PROCESS / IMAGE / LOG / VISUAL   (Upscale & Save: IMAGE / LOG / VISUAL; PROCESS zum Visualizer vom Loader)
   ▼
CMK Visualizer  (SAVE ENABLED = OFF)
```

### Grundaufbau B (mit Checkpoint): Detailer, FaceProcess, FaceRebuild, MaskDetailer

Wie A, zusätzlich:
```text
CMK Flow · Checkpoint & VAE  (ckpt_name = gewählter SDXL-Checkpoint, vae_name = Standard,
                              checkpoint_vae = true; PROCESS-Eingang unverbunden)
   │ MODEL SDXL
   ▼
CMK Load Image . MODEL SDXL (opt)
```

Das Modul wird aus der Node-Library eingefügt, damit die **kanonische**
Definition eingebettet wird. Es darf nicht aus einem bestehenden Showcase
kopiert werden.

| `<name>` | Aufbau | Modul | Abweichende Einstellung |
|---|---|---|---|
| `detailer_std` | B | CMK Flow · Detailer SDXL | — |
| `detailer_adv` | B | CMK Flow · Detailer SDXL · Advanced | — |
| `faceprocess_std` | B | CMK Flow · FaceProcess SDXL | — |
| `faceprocess_adv` | B | CMK Flow · FaceProcess SDXL · Advanced | — |
| `facerebuild_std` | B | CMK Flow · FaceRebuild SDXL | Quellbild-Auswahl = `cmk_s1_input.png` |
| `facerebuild_adv` | B | CMK Flow · FaceRebuild SDXL · Advanced | alle Quellbild-Auswahlen = `cmk_s1_input.png` |
| `faceswap_std` | A | CMK Flow · FaceSwap | Quellbild-Auswahl = `cmk_s1_input.png` |
| `faceswap_adv` | A | CMK Flow · FaceSwap · Advanced | alle Quellbild-Auswahlen = `cmk_s1_input.png` |
| `maskdetailer` | B | CMK Flow · MaskDetailer SDXL | — |
| `upscale_save` | A | CMK Flow · Upscale & Save | im Modul `SAVE ENABLED` = ON |
| `chain_postprocess` | B | Detailer SDXL → FaceProcess SDXL → Upscale & Save → Visualizer | Upscale & Save `SAVE ENABLED` = ON. Die MODEL- und PROCESS-Eingänge **hinter** Upscale & Save kommen, wie in *Full Flow*, von FaceProcess SDXL, weil Upscale & Save diese Ausgänge nicht hat |

**Legacy-Fixture `upscale_save_legacy_fullflow` (Aufbau A):**
1. Showcase `CMK 2.6 · Full Flow` öffnen und **sofort** unter dem neuen Namen speichern.
2. Alle Nodes außer der vorhandenen `CMK Flow · Upscale & Save`-Instanz löschen.
3. Davor `CMK Load Image` und dahinter `CMK Visualizer` setzen.
4. Im Modul `SAVE ENABLED` = ON.

So bleibt die **eingebettete Altdefinition** erhalten (K-3, Option A). Den
Showcase selbst nicht überschreiben.

## Manifest aufzeichnen

```bash
"ComfyUI/.venv/bin/python3" -I "ComfyUI/custom_nodes/cmk-flow/tests/fixtures/update_manifest.py" --frontend-version 1.57.4
```

Das Skript liest nur die exportierten Dateien und die kanonischen Subgraphen.
Es schreibt ausschließlich `tests/fixtures/manifest.json`.

## Prüfen

```bash
"ComfyUI/.venv/bin/python3" -I "ComfyUI/custom_nodes/cmk-flow/tests/run_tests.py" --only test_s1_fixtures --no-node
```

Erwartet: 4 Tests bestanden, 0 übersprungen. Ändert sich später eine kanonische
Definition, meldet `test_api_prompt_freshness` die betroffenen Fixtures zum
erneuten Export.

## Automatisierter Export (verifiziert, 2026-11-04) — Standardweg

Die manuelle Montage ist nur noch für die Referenz `detailer_std` belegt. Die
übrigen elf Fixtures entstehen automatisiert, wobei die Expansion **ausschließlich
das originale Frontend** übernimmt.

**Machbarkeitsnachweis:** Das manuell exportierte `workflows/detailer_std.json`
wurde im Browser-Pane mit `app.loadGraphData()` geladen und mit
`app.graphToPrompt()` expandiert. Das Ergebnis war **byte-identisch** zu
`api_prompts/detailer_std.json`.

**Ablauf:**
1. `fixtures/build_workflows.py <ComfyUI>/temp/cmk_s1_export` erzeugt
   Staging-Workflows (UI-Format). Die Module stammen aus `subgraphs/` (kanonische
   Instanz-Node plus Definition). Loader, Checkpoint und Visualizer samt
   Widget-Werten kommen aus `workflows/detailer_std.json`. Die Legacy-Fixture
   übernimmt Instanz und eingebettete Definition aus `CMK 2.6 · Full Flow`.
   Das Skript expandiert nichts.
2. Im Frontend der laufenden Instanz (`http://127.0.0.1:8188`, Frontend 1.57.4)
   `fixtures/export_in_frontend.js` ausführen. Es lädt jeden Staging-Workflow,
   ruft `app.graphToPrompt()` auf und speichert `prompt.workflow` (die
   Frontend-Serialisierung) sowie `prompt.output` über `/api/userdata` nach
   `user/default/cmk_s1_export/`. Es wird nichts in die Queue gestellt.
3. Die Dateien nach `fixtures/workflows/<name>.json` und
   `fixtures/api_prompts/<name>.json` kopieren. Danach
   `temp/cmk_s1_export` und `user/default/cmk_s1_export` löschen.
4. `fixtures/update_manifest.py --frontend-version 1.57.4` ausführen.
5. Prüfen: `test_s1_fixtures.py` (Struktur, Frische) und
   `test_s1_fixture_validation.py` (`validate_prompt`, benötigt
   `CMK_EXTRA_MODEL_PATHS=<inst-*.yaml der Instanz>`).

**Abweichungen gegenüber der früheren manuellen Anleitung:**
- **FaceSwap/FaceRebuild:** Die Quellbild-Auswahl bleibt auf dem kanonischen
  Standardwert (paketierte Referenz). Sie wird nicht auf `cmk_s1_input.png`
  umgestellt, damit die eingebettete Definition kanonisch bleibt.
- **Bildeingang:** `RESOLUTION` wie in der Referenz `detailer_std`
  (`SDXL 1152x832`).
- **Legacy-Fixture:** ID, Nodes, Links und Widget-Werte der Full-Flow-Definition
  sind unverändert. Das Frontend lässt beim Serialisieren unverbundene
  Widget-Eingangseinträge innerer Nodes weg und setzt `state.lastLinkId` neu.
  Das ist identisch mit einem manuellen Speichern in der UI.

## Einschränkung H-2: kein modellfreier Lauf ohne Substitution (bestätigt an `detailer_std`)

**Befund (statisch belegt, an `detailer_std` geprüft):** `CMK Load Image`
deklariert `MODEL SDXL (opt)` als **nicht-lazy** Eingang
(`pipe/loaders/cmk_image_load_resize.py:174`) und hat kein `check_lazy_status`.
ComfyUI wertet verbundene, nicht-lazy Eingänge immer vor der Node aus.

Jede Ausführung des Loaders führt deshalb `CMK Flow · Checkpoint & VAE` aus.
Diese Node lädt den Checkpoint echt über `_load_cached_resources`
(`pipe/loaders/checkpoint_vae_loader.py`), auch wenn das nachfolgende Modul
gebypasst ist.

Folgen:
- Alle Fixtures nach **Aufbau B** sind ohne Substitution **nicht** modellfrei
  ausführbar, auch nicht in Bypass-Szenarien.
- Fixtures nach **Aufbau A** (ohne Checkpoint) sind davon nicht betroffen.
- Wie teuer das Verhalten im Produktivbetrieb ist (Laden trotz Bypass), bleibt
  eine Laufzeitfrage. Sie wird in S1-T7 bzw. per Zählung im Harness geklärt und
  ist kein Export-Problem.

### Checkpoint-Loader-Substitution — ausschließlich im S1-Harness

- **Ort:** `tests/harness/seams.py` (S1-T6). Kein Produktcode wird verändert,
  keine Datei außerhalb von `tests/`.
- **Mechanismus:** Im Harness-Prozess wird für die Dauer eines T1-Laufs die
  Modulfunktion `checkpoint_vae_loader._load_cached_resources` durch eine
  deterministische Ersatzfunktion ersetzt. Wo der Name in andere Module
  importiert wurde, wird er dort ebenfalls ersetzt. Nach dem Lauf wird das
  Original wiederhergestellt.
- **Ersatzwert:** gleiche Struktur wie das Original, also ein Tupel
  `(model, clip, vae, metadata)` und ein `status`-String. Die drei Objekte sind
  billige Platzhalter. Die Metadaten-Schlüssel entsprechen denen des Originals,
  damit nachgelagerte Verträge (`model_role`, `materialized`,
  `resource_cache_status`) unverändert bedient werden.
- **Nicht ersetzt:** `CMKCheckpointVAELoaderPipe` selbst und alle Node-Klassen
  (`INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `check_lazy_status`, `IS_CHANGED`,
  `NOT_IDEMPOTENT`). Ersetzt wird nur die Engine-Ebene unterhalb der Node.
- **Weitere Nahtstellen bei aktiver Verarbeitung**, nach demselben Prinzip:
  SAM-Lader (`CMKSAMLoader`), YOLO-Detektor (Detector-Engine), Sampler-Aufruf
  (`SEGSDetailer`). Jede Nahtstelle wird einzeln benannt und einzeln gezählt.
- **Geltungsbereich:** nur Tier T1. In T2 (S1-T7, echte Modelle) ist keine
  Substitution aktiv.

### Pflicht für die Probe-Zähler

Die Zähler müssen strikt trennen zwischen

| Zähler | Bedeutung |
|---|---|
| `original_calls[<seam>]` | Das **Original** wurde tatsächlich aufgerufen |
| `substituted_calls[<seam>]` | Die **Ersatzfunktion** wurde aufgerufen |

- Ein T1-Lauf ist nur gültig, wenn für jede substituierte Nahtstelle
  `original_calls == 0` gilt. Das belegt, dass kein Modell geladen wurde.
- Aussagen wie „Compute wurde (nicht) ausgeführt“ stützen sich auf
  `substituted_calls` und die Node-`FUNCTION`-Zähler, **nie** auf eine Summe aus
  beiden.
- Ein Szenario, das ein bestimmtes Laden erwartet (z. B. „Bypass darf keinen
  Checkpoint laden“), prüft `substituted_calls` gegen den Soll-Wert. Der Ist-Wert
  aus H-2 wird als Baseline festgehalten und nicht abgeschwächt.

### Selbsttest der Substitution (S1-T6, `test_s1_harness_selfcheck.py`)

`test_seam_substitution_neutrality` muss bestehen, bevor ein Szenario mit
Substitution als Evidenz gilt. Er prüft in je einem frischen Prozess mit und
ohne installierte Substitution:

1. **ComfyUI-Cache-Schlüssel unverändert:** Für den Fixture-Prompt
   `detailer_std` werden die Cache-Signaturen aller Nodes mit ComfyUIs eigenem
   `CacheKeySetInputSignature` berechnet. Das geschieht vor jeder Ausführung,
   also ohne Laden. Beide Varianten müssen identische Schlüssel liefern.
2. **Klassen-Oberfläche unverändert:** Für alle registrierten Node-Klassen sind
   `INPUT_TYPES()`, `RETURN_TYPES`, `FUNCTION`, die Identität von
   `check_lazy_status` sowie Vorhandensein und Ergebnis von `IS_CHANGED`
   bzw. `NOT_IDEMPOTENT` vor und nach der Installation gleich.
3. **Lazy-Eingangsverträge unverändert:** Auf der modellfreien Selbsttest-Kette
   (`selfcheck_chain`, berührt keine Nahtstelle) sind die Folgen der
   `check_lazy_status`-Anforderungen, die `execution_cached`-Listen und die
   Ausgabe-Hashes mit und ohne installierte Substitution identisch.
4. **Zähler-Trennung wirksam:** In einem Bypass-Lauf von `detailer_std` mit
   Substitution gilt `original_calls["_load_cached_resources"] == 0` und
   `substituted_calls["_load_cached_resources"] >= 1`. Letzteres ist das
   H-2-Verhalten und wird als Baseline-Wert festgehalten.

**Grenze des Selbsttests:** Er beweist, dass die Substitution weder
Cache-Schlüssel noch Klassenverträge noch das Lazy-Verhalten unbeteiligter Nodes
verändert. Ob der Ersatzwert strukturell genau dem Original entspricht, kann T1
nicht beweisen, weil das Original nicht aufgerufen werden darf. Das bestätigt
einmalig der T2-Gegenlauf (S1-T7) durch Vergleich der Rückgabestruktur.

## Folgepunkt für S1-T6 (kein Export-Schritt)

Die Fixtures mit Aufbau B enthalten den Checkpoint-Namen der Desktop-Instanz.
ComfyUI prüft Combo-Werte bei der Validierung gegen die vorhandene Modellliste.
Die Desktop-Instanz nutzt dafür `--extra-model-paths-config …/instance-model-paths/inst-*.yaml`.
Der Harness muss in S1-T6 dieselbe Modellpfad-Konfiguration **lesend** laden.
Ohne sie schlägt `validate_prompt` fehl, obwohl der Prompt korrekt ist.

## Re-Export nach H-4 (2026-11-04): `upscale_save`, `chain_postprocess`

Gleiches automatisiertes Verfahren, beschränkt auf die beiden Fixtures mit der
kanonischen Upscale-&-Save-Definition 1.4.0.

- Ein neuer Node-Typ (`CMKSaveEndpoint`) erfordert einen Neustart der Instanz
  **und ein Neuladen der Frontend-Seite**. Ohne Neuladen serialisiert das
  Frontend den Node mit `UNKNOWN`-Widgets und ohne `class_type`. Das Skript
  prüft deshalb vorab `LiteGraph.registered_node_types`.
- Das Frontend vergibt innere Node-IDs neu, wenn sie mit Nodes der äußeren
  Ebene kollidieren (kanonisch 8417/8418, in der Fixture 8419/8420). Die
  Topologie-Signatur bleibt davon unberührt.
- Kontrolle: Die mitexportierte, nicht übernommene Legacy-Fixture
  `upscale_save_legacy_fullflow` ergab einen byte-identischen API-Prompt. Im
  UI-Workflow wich nur die von der DOM-Darstellung abhängige Höhe der äußeren
  Instanz ab (145 statt 130).
- Manifest neu aufgezeichnet. Die Legacy-Fixture steht jetzt mit
  `canonical_at_export: false`.

## Re-Export nach H-3 / F-03 (2026-11-04): acht Modul-Fixtures und `chain_postprocess`

- `detailer_std` war bisher die manuell exportierte Referenz. Jetzt wird sie wie
  die übrigen Fixtures aus der kanonischen Definition erzeugt
  (`build_workflows.py`, Eintrag `detailer_std`). Die bisherige Datei war in
  Instanz- und inneren Widget-Werten identisch mit der kanonischen Definition.
  Der neue API-Prompt unterscheidet sich ausschließlich in den beiden neuen
  VISUAL-Links.
- Neustart und Seiten-Neuladen wie bei H-4, weil sich die Boundary-Schnittstellen
  geändert haben.
- **Befund zur Expansion:** Ist der VISUAL-Eingang einer Modulinstanz außen nicht
  verbunden (Einzelmodul, erstes Modul einer Kette), entfernt das Frontend auch
  den inneren Link `SG_IN.VISUAL → Boundary.VISUAL IN`. Die Boundary erkennt die
  neue Verdrahtung deshalb am internen Link `VISUAL` vom Compute-Node.
- Kontrolle: `upscale_save` mitexportiert, API und Workflow byte-identisch.
  `upscale_save_legacy_fullflow`: API identisch, im Workflow nur die bekannte
  Instanzhöhe abweichend.
