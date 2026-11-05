# CMK Flow 2.7 · Showcase-Migration

Stand 2026-11-05. Alle 37 ausgelieferten Showcase-Workflows (`workflows/showcase/*.json`)
verwenden ausschließlich die aktuellen kanonischen Subgraph-Definitionen
(`subgraphs/*.json`). Der Validator klassifiziert **alle 135 eingebetteten Definitionen als
`IDENTICAL`**; Legacy-, Defekt- und Review-Register für Definitionen sind leer.

## Werkzeug und Verfahren

* `tests/migration/migrate_showcase.py` ersetzt veraltete eingebettete Definitionen durch die
  kanonische Fassung (Regeln im Modul-Docstring). Konflikte blockieren das Schreiben.
* Eingebettete Subgraph-IDs bleiben erhalten; Instanzen verweisen weiter auf dieselbe UUID.
* Innere Nodes werden über ID und Typ, sonst über den einzigen Kandidaten gleichen Typs bzw.
  über bereits zugeordnete Nachbarn zugeordnet. Widget-Werte werden namensbasiert übertragen,
  wenn beide Seiten vollständige Namenslisten führen, sonst positionsweise mit
  Typprüfung gegen die kanonische Serialisierung.
* Instanz-Ports werden nach Namen in kanonischer Reihenfolge neu aufgebaut; bestehende
  Links behalten ihre ID. Ein neuer Eingang erbt die externe Verbindung des alten Eingangs,
  der exakt dieselben inneren Ziele speiste.
* Danach wurde jede Datei im echten Frontend (ComfyUI-Frontend 1.57.4) geladen und über
  `app.graphToPrompt()` serialisiert; die ausgelieferte Datei ist diese Frontend-Serialisierung.
* `workflows/showcase/metadata/` (Kategorien, Veröffentlichungsstatus, Flow-Browser-Einträge)
  ist unverändert; die zwei unveröffentlichten Workflows bleiben unveröffentlicht.

## Entscheidungen

| Gegenstand | Entscheidung |
|---|---|
| Umfang | Vollständige Migration aller Showcases; frühere Freezes (K-3 Option A, E1 = a, F-04/F-11, Abweichungen LoRA Stack · ZIT, 15A, FaceRebuild Adv, FaceSwap Adv) aufgehoben |
| Identity Processing Matrix | Basis ist die korrigierte Benutzerkopie (`SHOWCASE/COMPARISONS`), kein `CMKPostProcessBoundarySDXLPipe` |
| Full Flow · FaceSwap (7849, 8461, 8465, 8410) | **Variante A**: `POST RESTORE = True`, `POST RESTORE MODEL = GFPGANv1.4.pth`, `RESTORE VISIBILITY = 1` |

### Unsicherheit FaceSwap-Widgets (Full Flow)

Die 2.6-Serialisierung von `CMKFaceSwapImagePipe` hatte 19 Werte, aber nur 16 Widget-Namen
(`BLEND`, `crop_factor`, `feather` fehlen in der Liste). Die Werte 0–15 stimmen nach Typ und
Bedeutung mit der kanonischen Reihenfolge überein; das alte Ende lautet
`[POST RESTORE MODEL, RESTORE VISIBILITY, POST RESTORE]` = `['GFPGANv1.4.pth', 1, True]`,
kanonisch steht `POST RESTORE` davor. Die Zuordnung des abschließenden `True` zu
`POST RESTORE` ist aus Reihenfolge und Typen abgeleitet, nicht aus einem Namen belegbar.
Sie ist als eng gefasste Regel `WIDGET_REORDERS` im Werkzeug festgehalten (greift nur bei
exakt dieser Typfolge) und wurde fachlich freigegeben (Variante A).

## Übersicht aller 37 Workflows

`–` = unverändert (alle Definitionen waren bereits kanonisch). `M` = migriert.

| # | Workflow | Status | Ersetzte Definitionen und Anmerkungen |
|---|---|---|---|
| 1 | CMK 2.6 · FaceProcess · Modul | M | FaceProcess SDXL (+ `CMKModelForwardPipe`; VISUAL-Gate-Verdrahtung jetzt kanonisch, F-04/F-11 behoben) |
| 2 | CMK 2.6 · FaceProcess | – | ohne eingebettete Definitionen |
| 3 | CMK 2.6 · FaceRebuild · Modul | M | FaceRebuild SDXL (+ `CMKModelForwardPipe`; F-04/F-11 behoben) |
| 4 | CMK 2.6 · FaceRebuild | – | ohne eingebettete Definitionen |
| 5 | CMK 2.6 · FaceSwap · Modul | M | FaceSwap (namensbasierte Widget-Übernahme; F-04/F-11 behoben) |
| 6 | CMK 2.6 · FaceSwap | – | ohne eingebettete Definitionen |
| 7 | CMK 2.6 · Flow · InOutpaint HYBRID | – | |
| 8 | CMK 2.6 · Flow · InOutpaint SDXL · HYBRID · ZIT | – | |
| 9 | CMK 2.6 · Flow · InOutpaint SDXL | – | |
| 10 | CMK 2.6 · Flow · Inpaint Z-Image Turbo | M | LoRA Stack · ZIT (+ `CMKTriggerWordsConcatenate`; LoRA-Stacker-Werte übernommen) |
| 11 | CMK 2.6 · Flow · InstantID · Hybrid Inpaint SDXL | – | |
| 12 | CMK 2.6 · Flow · Text2Image - InstantID SDXL and FaceRebuild HYBRID · ZIT | M | FaceRebuild SDXL |
| 13 | CMK 2.6 · Flow · Text2Image - Regional Conditioning HYBRID | – | |
| 14 | CMK 2.6 · Flow · Text2Image - Regional Conditioning SDXL | – | |
| 15 | CMK 2.6 · Flow · Text2Image ControlNet HYBRID | – | |
| 16 | CMK 2.6 · Flow · Text2Image ControlNet SDXL · HYBRID · ZIT | – | |
| 17 | CMK 2.6 · Flow · Text2Image ControlNet Z-Image Turbo | – | |
| 18 | CMK 2.6 · Flow · Text2Image HYBRID | – | |
| 19 | CMK 2.6 · Flow · Text2Image SDXL · HYBRID · ZIT | – | |
| 20 | CMK 2.6 · Flow · Text2Image SDXL | – | |
| 21 | CMK 2.6 · Flow · Text2Image Z-Image Turbo | – | |
| 22 | CMK 2.6 · Flow · Text2Image-ControlNet SDXL | – | |
| 23 | CMK 2.6 · Flow · Text2Image-FaceRebuild HYBRID | M | FaceRebuild SDXL |
| 24 | CMK 2.6 · Flow · Text2Image-FaceRebuild Z-Image Turbo | M | FaceRebuild SDXL |
| 25 | CMK 2.6 · Flow · Text2Image-InstantID SDXL | – | |
| 26 | CMK 2.6 · Full Flow | M | 10 Definitionen: FaceRebuild SDXL, FaceSwap (Variante A), FaceRebuild SDXL · Advanced, FaceSwap · Advanced (Variante A), FaceProcess SDXL, FaceProcess SDXL · Advanced, **Upscale & Save 1.4.0**, Detailer SDXL, Detailer SDXL · Advanced, 15A InstantID Regional SDXL |
| 27 | CMK 2.6 · InOutPaint | M | FaceProcess SDXL |
| 28 | CMK 2.6 · MaskDetailer | – | ohne eingebettete Definitionen |
| 29 | CMK 2.6 · MaskDetailer · Modul | – | |
| 30 | CMK 2.6 · SmartDetailer · Modul | M | Detailer SDXL (F-04/F-11 behoben) |
| 31 | CMK 2.6 · SmartDetailer | – | ohne eingebettete Definitionen |
| 32 | CMK 2.6 · Text2Image Identity 1 | M | FaceRebuild SDXL, FaceSwap |
| 33 | CMK 2.6 · Text2Image Identity 3 | M | 15A InstantID Regional SDXL, FaceRebuild SDXL · Advanced, FaceSwap · Advanced |
| 34 | CMK 2.6 · Text2Image | M | Detailer SDXL · Advanced |
| 35 | CMK FaceSwap Video | – | ohne eingebettete Definitionen |
| 36 | FaceSwap vs FaceRebuild | – | ohne eingebettete Definitionen |
| 37 | Identity Processing Matrix | M | 14 Definitionen aus der korrigierten Benutzerkopie: 3× FaceSwap, 3× FaceRebuild SDXL, 3× `10 1st Pass` → `CMK Flow · 10 KSampler SDXL 1st Pass`, 3× `20 Refiner` → `CMK Flow · 20 Refiner SDXL` (Ausgang `IMAGE REFINED` → `IMAGE`, gleicher Slot und Typ), 2× `15 InstantID` → `CMK Flow · 15S InstantID Regional SDXL` |

### Full Flow im Detail

* **Upscale & Save 1.4.0:** `CMK_SaveProjectImage` (8371) entfällt, `CMKSaveEndpoint` und
  `CMKVisualProvider` kommen hinzu. Die alten Save-Werte `[True, 'image', '', True]` entsprechen
  den kanonischen Endpoint-Werten. Vertrag geprüft: `IMAGE ACTIVE` des Gates und `SG_OUT.IMAGE`
  kommen aus `SG_IN.IMAGE` (IMAGE OUT = IMAGE IN); die hochskalierte Kopie geht ausschließlich
  über die Boundary an `CMKSaveEndpoint`. Instanz 8444 behält `SAVE ENABLED = False`,
  `FILENAME PREFIX = image`.
* **`opt_image_file`-Aufteilung:** FaceRebuild · Advanced (8440) und FaceSwap · Advanced (8441)
  hatten einen gemeinsamen Eingang `opt_image_file`; die kanonischen Definitionen führen
  `opt_image_file`/`_1`/`_2`. `_1` und `_2` erben dieselbe Quelle (neue Links 22644–22647).
  Bei 15A speiste der alte Eingang nur den ersten Loader; `_1` bleibt unverbunden.
* **FaceSwap:** Variante A, siehe oben.

## Abweichungen (dokumentiert, nicht semantisch)

* **Positionen:** in keinem der 37 Workflows hat sich eine Top-Level-Position geändert
  (Identity Processing Matrix gemessen gegen die Benutzerkopie).
* **Instanzgrößen:** Das Frontend berechnet die Mindesthöhe von Subgraph-Instanzen mit neuen
  Ports neu (z. B. 450×90 → 450×160, 300×100 → 300×290); einzelne Breiten 300 → 305/310/320.
  Betroffen sind nur migrierte Workflows.
* **Innere Node-IDs:** Das Frontend vergibt Node-IDs workflow-weit eindeutig. Kollidierende
  innere IDs wurden umnummeriert (Full Flow: FaceSwap · Advanced 8465 → 8494, Upscale & Save
  `CMKSaveEndpoint` 8417 → 8497; Text2Image Identity 3: FaceSwap · Advanced 8462/8464/8465 →
  8482/8483/8484; Identity Processing Matrix: innere Nodes der 1st-Pass-Instanzen). Links und
  `proxyWidgets` aller 37 Dateien verweisen nach Prüfung ausschließlich auf existierende Nodes.
* **Port-Serialisierung:** Das Frontend führt Widget-Eingänge von `CMKVisualizer` und
  `CMKImageLoadAndResizePipe` (und, wie bereits zuvor, `CMKPipeCreateImage`) jetzt in der
  `inputs`-Liste; Link-Slotnummern verschieben sich entsprechend. Nach Portnamen sind alle
  Verbindungen identisch.
* **START HERE (`CMKPipeCreateImage`) unverändert gelassen:** Der Entwicklungsserver
  exponiert das Widget `content_guard_enabled`. Die Frontend-Serialisierung schrieb deshalb in
  den 10 migrierten Workflows mit START-HERE-Node (8332 bzw. IPM 7525/7704/7730) ein
  positionales Array mit Guard-Slot, ohne die Widget-Eingänge, über die
  `withoutLegacyReleaseGuardValue` den Guard beim Release-Laden entfernt; dazu
  `widgets_values_named` mit Guard-Schlüssel. Im Release-Schema hätte das die Werte ab
  `controlnet_enabled` verschoben (z. B. `instantid_handoff_percent = 'AUTO'`) und verletzt den
  Content-Guard-Vertrag (`test_packaged_workflows_contain_no_guard_widget_serialization`).
  Korrektur: Die START-HERE-Nodes wurden vollständig auf ihren Stand vor der Migration
  (git HEAD) zurückgesetzt; ihre Links haben dieselben Slots. Für die Identity Processing
  Matrix ist das bedeutungsgleich mit der Benutzerkopie: Deren Werte ohne den Guard-Slot sind
  die 22 HEAD-Werte plus die vier deklarierten Defaults (`live_preview_zit = True`,
  `controlnet_enabled = False`, `instantid_handoff_percent = 70`, `stop_after_handoff = False`);
  Position und Größe stammen aus der Benutzerkopie. Kein Paketfile enthält mehr eine
  Guard-Serialisierung.
* **Frontend-Normalform innerer Widgets:** `CMKLoadImage` wird vom Frontend mit doppeltem
  Upload-Pseudo-Widget geschrieben (`[file, 'image', 'image']`, kanonisch `[file, 'image']`),
  `TriggerWord Toggle (LoraManager)` ohne leeren Zusatzzustand (`[true, true, true]`, kanonisch
  `[true, true, true, [], '']`). Beides ist die Round-Trip-Form des Frontends ohne
  Bedeutungsänderung; die Topologie-Signatur bleibt `IDENTICAL`.
* **Normalisierung:** Doppelt serialisierte Upload-Pseudo-Widgets von `CMKLoadImage`
  (`[file, 'image', 'image']`) wurden auf die kanonische Form gekürzt; `opt_in = False` der
  Advanced-Loader bleibt erhalten.

## Prüfungen

| Prüfung | Ergebnis |
|---|---|
| Laden im Frontend, alle 37 | ohne Fehler, keine unbekannten Node-Typen, keine Konsolenfehler, API-Prompt erzeugt |
| Registrierte Node-Typen (Server `object_info`) | alle registriert; einzige Ausnahme `Note` (reiner Frontend-Node) |
| Validator (`classify_embedded_definitions`) | 135/135 `IDENTICAL` |
| Gate-Pfade eingebetteter Definitionen | vollständig (F-04-Befund der vier Modul-Showcases behoben) |
| Links (Top-Level und Definitionen) | keine verlorenen, verschobenen, verwaisten oder doppelten Links gegenüber der Quelle |
| Widget-Werte Top-Level | unverändert (START HERE auf Vor-Migrationsstand, IPM gegen Benutzerkopie bedeutungsgleich) |
| Content-Guard-Vertrag | keine Guard-Serialisierung in `subgraphs/` und `workflows/` |
| Python- und Frontend-Suite | S1 195 Tests (185 PASS, 8 erwartete Fehlschläge, 2 SKIP, 0 unerwartet); Bestand 159 (158 PASS, 1 Altfehler Sourcemap); `.mjs` 7/7; geschützte Caches unverändert (Details `tests/S1_STATUS.md`) |

## Angepasste Tests und Verträge

| Datei | Änderung |
|---|---|
| `tests/contracts/legacy_registry.json` | Einträge geleert (keine Legacy-Definitionen mehr) |
| `tests/contracts/known_defects.json` | `definition_drift` geleert; M8-Liste ohne IPM 7900 (durch Migration behoben) |
| `tests/contracts/drift_review.json` | Einträge geleert (S1-REVIEW erledigt) |
| `tests/test_s1_graph_contracts.py` | Zielzustand: keine Review-, Defekt- oder Legacy-Definitionen, nur `IDENTICAL`, vollständige Gate-Pfade |
| `tests/test_s1_postprocess_model.py` | Migrierte Modul-Showcases: Links nach Portnamen, Widgets, Positionen, Typen, Modi |
| `tests/test_stop_after_handoff_output_roots.py` | IPM-Ausnahme für Gates ohne VISUAL-Port entfernt |
| `tests/test_image_resize_port_contract.py` | Full-Flow-SHA-256 aktualisiert; fünf Vertragslinks unverändert |

Fixtures (`tests/fixtures/`) sind eingefrorene Kopien und blieben unverändert.

## Benutzer-Workflows

* Archiv: `ComfyUI/_archive/user_workflows_pre_2.7_2026-11-05/` mit relativer Struktur,
  `MANIFEST.sha256` (40 Einträge) und `directories.json`. Prüfung mit Skript und
  `shasum -a 256 -c`: 40/40 OK gegen Archiv und gegen die Originale vor dem Löschen.
* Danach alle 40 `*.json` aus `ComfyUI/user/default/workflows/` entfernt; Verzeichnisse
  (`FLOW`, `SHOWCASE` mit sechs Unterordnern) und `.DS_Store` unverändert; keine neuen
  Benutzerkopien.
* Staging bereinigt: `ComfyUI/temp/cmk_migration/`, `ComfyUI/temp/cmk_migration_all/`
  (Ladeprüfung) und `ComfyUI/user/default/cmk_migration_export/`.
