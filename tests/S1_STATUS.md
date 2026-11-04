# S1 — Stand des Testfundaments

Stand: 2026-11-03. Ausgeführt mit `ComfyUI/.venv/bin/python3` (Python 3.13.12,
torch 2.12.1, CPU-Modus im Harness). Keine GPU-Läufe, keine Modellladung.

## Ausführen

```bash
"ComfyUI/.venv/bin/python3" -I "ComfyUI/custom_nodes/cmk-flow/tests/run_tests.py" --no-node
```

| Option | Wirkung |
|---|---|
| `--tier T0,T1` | Standard |
| `--only <Muster>` | nur passende Dateien |
| `--json <Pfad>` | Bericht als JSON |

Der Runner startet jede Testdatei in einem eigenen Interpreter. Er prüft vor und
nach dem Lauf, ob sich die echten Caches (`cmk-flow/cache`, `ComfyUI/temp/cmk`)
verändert haben. Ist das der Fall, schlägt der Lauf fehl. Gelöscht wird nichts.

**Interpreter:** `standalone-env/bin/python3.13` enthält kein torch. Die richtige
Umgebung ist `ComfyUI/.venv`, wie im ComfyUI-Log.

## Teilschritte und Ergebnisse

| Schritt | Ergebnis |
|---|---|
| S1-T1 Runner und Baseline der Bestandstests | 20 Python-Dateien: 19 grün, 1 rot (bestehender Fehler, s. u.); 6 `.mjs`-Dateien nicht ausgeführt (kein `node`) |
| S1-T2 In-Prozess-Laufzeit (`harness/comfy_runtime.py`, `fake_server.py`, `restart.py`) | echter `execution.PromptExecutor`, Cache-Modi `classic` und `ram_pressure` wie in `main.py`; Boot ca. 5 s |
| S1-T3 Probes und Selbsttests (`harness/probes.py`, `test_s1_harness_selfcheck.py`) | 4/4 grün: Neutralität, Warmlauf, Neustart-Isolation, RAM-Pressure |
| S1-T5 Validator (`validator/`, `contracts/`, `test_s1_graph_contracts.py`, `test_s1_graph_types.py`) | 17 Tests grün, 5 strikt erwartete Fehlschläge |
| Cache-Baseline (Teil von S1-T6, `test_s1_cache_contract.py`) | 4 grün, 3 strikt erwartete Fehlschläge |
| S1-T4 Fixtures | Anleitung `FIXTURES_EXPORT.md`, `fixtures/update_manifest.py`, `test_s1_fixtures.py` (4 Tests **übersprungen**, bis exportiert) |

Drei aufeinanderfolgende Läufe der S1-Dateien lieferten identische Ergebnisse;
die geschützten Caches blieben jeweils unverändert.

## Bestandstests mit Fehlern (unverändert gelassen)

`test_image_resize_port_contract.py`:
- `test_full_flow_links_match_registered_port_contract`: Der festgeschriebene
  SHA-256 von `CMK 2.6 · Full Flow .json` entspricht nicht mehr der Datei.
- `test_installed_frontend_hook_precedes_input_creation`: Der erwartete
  Quellpfad `../../src/services/litegraphService.ts` fehlt in den Sourcemaps
  des installierten Frontends 1.49.6.

Beide Fehler bestanden schon vor S1.

## Strikt erwartete Abweichungen (Ist-Fehler)

Jede Abweichung hat einen `test_baseline_*`-Partner, der das genaue Fehlverhalten
festhält. Wird der Fehler behoben, meldet `unittest` einen unerwarteten Erfolg
(= Fehlschlag), und Soll- und Baseline-Test müssen gemeinsam angepasst werden.

| ID | Soll-Test | Ist |
|---|---|---|
| **H-1 (neu, durch den Harness gefunden)** | `test_boundary_result_persists_across_restart` | Die Upscale-&-Save-Boundary meldet `STORE_FAILED`: safetensors lehnt Diagnose-Tensoren mit geteiltem Speicher ab (`diagnostic_0000`/`diagnostic_0001`). Diese Eingaben ergeben nach einem Neustart nie einen HIT |
| B4 | `test_every_declared_scope_is_registered` | `image_resize_boundary` fehlt in `PERSISTENT_DISK_SCOPES` |
| A3.2 | `test_no_scope_is_shared_between_caches` | `sdxl_handoff_boundary` wird von `cmk_pipe_sampler` und `cmk_hybrid_bridge` gemeinsam genutzt |
| F-04/F-11 | `test_shipped_workflows_have_no_known_definition_defects`, `test_embedded_gate_paths_complete` | vier „· Modul“-Showcases: Gate `VISUAL ACTIVE/BYPASS` unverbunden |
| M8 (breiter als bisher dokumentiert) | `test_provider_metadata_references_existing_nodes` | 77 gespeicherte `cmkVisualProviders`-Einträge verweisen auf nicht existierende Nodes, davon 12 von 16 kanonischen Subgraphen |
| OD-3 (jetzt mit ComfyUIs eigener Typregel belegt) | `test_regular_postprocess_modules_are_mutually_composable` | FaceSwap und FaceSwap · Advanced geben `CMK_RESULT_*` aus und sind mit Detailer, FaceProcess und FaceRebuild (je Standard und Advanced) nicht verbindbar (36 Port-Kombinationen) |
| S1-REVIEW | `test_shipped_workflows_have_no_unreviewed_drift` | 7 eingebettete Definitionen weichen ohne dokumentierten Grund von der kanonischen ab (`contracts/drift_review.json`); **Entscheidung durch den Maintainer nötig** |

## Harness-Erkenntnisse zur ComfyUI-Semantik (statisch abgesichert, im Harness bestätigt)

- `executing`-Ereignisse wiederholen sich bei Lazy-Nodes. „Betreten“ ist nicht
  „ausgeführt“; tatsächliche Ausführungen zählen die Probes.
- ComfyUI schreibt `is_changed` in das Prompt-Dict zurück. Der Harness kopiert
  jeden Prompt, wie es das Frontend mit einem neuen Prompt pro Queue tut.
- `CMKVisualizer` (`IS_CHANGED = nan`) läuft bei jedem Prompt erneut; Loader und
  Boundary kommen im Warmlauf aus dem ComfyUI-Cache.

## Offen in S1 (keine weitere Umsetzung ohne Freigabe)

| Punkt | Stand |
|---|---|
| S1-O1 Node.js | `node` ist nicht im PATH. Gefunden wurde nur eine Laufzeit im App-Verzeichnis einer anderen Anwendung (`~/Library/Application Support/Codex/.../runtime/bin/node`); nicht verwendet. Der Runner akzeptiert `CMK_NODE=<Pfad>`, sobald eine Node.js-Laufzeit freigegeben ist |
| S1-T4 | Fixtures gemäß `FIXTURES_EXPORT.md` exportieren und das Manifest aufzeichnen |
| S1-T6 | Szenarien §3.A, §3.C, §3.D und der Rest von §3.B. Sie brauchen die Fixtures (Subgraph-Prompts) und die Engine-Nahtstellen (`harness/seams` folgt in S1-T3b/S1-T6) |
| S1-T7 | T2-Läufe mit echten Modellen: freigabepflichtig (GPU) |
| S1-REVIEW | Owner-Entscheidung zu den 7 Einträgen in `contracts/drift_review.json` |

## Neue Dateien (ausschließlich unter `tests/`)

```text
run_tests.py
S1_STATUS.md
FIXTURES_EXPORT.md
harness/      __init__.py · comfy_runtime.py · fake_server.py · probes.py · restart.py · scenarios.py
validator/    __init__.py · graph_model.py · checks.py · report.py
contracts/    legacy_registry.json · known_defects.json · drift_review.json
fixtures/     make_images.py · update_manifest.py · images/cmk_s1_input.png · images/cmk_s1_input_replacement.png
              workflows/ · api_prompts/   (leer bis zum Export)
_reports/     JSON-Berichte der Läufe
test_s1_harness_selfcheck.py · test_s1_cache_contract.py · test_s1_graph_contracts.py
test_s1_graph_types.py · test_s1_fixtures.py
```

## Fixture-Prüfung `detailer_std` (2026-11-03)

| Prüfpunkt | Ergebnis |
|---|---|
| Expansion | vollständig: Instanz `8417` mit allen 6 inneren Nodes (`8417:7706…7719`); Node-IDs im API-Prompt = Definitions-IDs |
| Struktur | eingebettete Definition topologisch identisch mit der kanonischen `CMK Flow · Detailer SDXL` (andere UUID `7eceb300-…`, normal beim Einfügen aus der Library; der Validator vergleicht nach Name und Struktur) |
| Port-Typen | alle 10 äußeren Links typgleich; Gate-Pfade vollständig; VISUAL-Endpunkte verbunden |
| Checkpoint | `8419 CMKCheckpointVAELoaderPipe`: `juggernautXL_ragnarok.safetensors`, `ClearVAE_V2.2.safetensors`, `checkpoint_vae = true` → `1.MODEL SDXL (opt)` |
| `validate_prompt` | **ohne** Desktop-Modellpfade ungültig: Checkpoint, VAE, SAM, YOLO und RealESRGAN sind nicht in den Listen. **Mit** `--extra-model-paths-config` der Desktop-Instanz (lesend geladen, neuer Parameter `boot(..., extra_model_paths=...)`) **gültig**, Outputs `8417:7708`, `8418` |
| Abweichung | `RESOLUTION = SDXL 1152x832` statt laut Anleitung `1024x1024`; gültiger Wert, für S1-T6 unkritisch |
| Bekannt | Instanz-Metadaten `cmkVisualProviders` veraltet (M8, aus der kanonischen Definition geerbt) |

**Folgerung für S1-T6 (H-2, statisch belegt):** `CMK Load Image` deklariert
`MODEL SDXL (opt)` als **nicht-lazy** Eingang und hat kein `check_lazy_status`.
Jede Ausführung des Loaders erzwingt deshalb `CMKCheckpointVAELoaderPipe`
(`_load_cached_resources`, echtes Laden), auch wenn das Modul gebypasst ist.
Ohne Nahtstelle für `_load_cached_resources` ist **kein** Lauf dieser Fixture
modellfrei. Bei aktivem Detailer kommen SAM (`CMKSAMLoader`) und der YOLO-Detektor
hinzu.

## S1-T4 abgeschlossen (2026-11-04): alle 12 Fixtures

| Fixture | Entstehung | Struktur | `validate_prompt` |
|---|---|---|---|
| `detailer_std` | manuell (Referenz) | kanonisch | gültig |
| `detailer_adv`, `faceprocess_std`, `faceprocess_adv`, `facerebuild_std`, `facerebuild_adv`, `faceswap_std`, `faceswap_adv`, `maskdetailer`, `upscale_save`, `chain_postprocess` | automatisiert über das echte Frontend 1.57.4 | kanonisch | gültig |
| `upscale_save_legacy_fullflow` | automatisiert; eingebettete Full-Flow-Definition (`bae09c41-…`) | identisch mit Full Flow | gültig |

- Die Zahl expandierter innerer Nodes entspricht bei jeder Fixture exakt der
  Node-Zahl der jeweiligen Definition.
- Manifest aufgezeichnet (`fixtures/manifest.json`, Frontend 1.57.4, ComfyUI 0.43.2).
- `test_s1_fixtures.py` 4/4 grün (0 Skips).
- `test_s1_fixture_validation.py` grün mit `CMK_EXTRA_MODEL_PATHS`.
- Staging-Ordner der Instanz (`temp/cmk_s1_export`, `user/default/cmk_s1_export`)
  wurden entfernt. Settings und Workflows der Instanz blieben unverändert.

## S1-T6 abgeschlossen (2026-11-04): Szenarien auf den 12 Fixtures

Alle Szenarien laufen auf dem echten `PromptExecutor` mit exportierten
API-Prompts und produktiven CMK-Nodes. Ersetzt werden ausschließlich die
Engine-Aufrufe (`harness/seams.py`): Checkpoint-Ressourcen (H-2),
FaceSwap-Detektor/-Swap und Upscale-Modell/-Lauf. `resolve_postprocess_model`
und `CMKSAMLoader.load` werden **gestoppt** (`SeamStop`), sodass jede
Modell-Materialisierung als Abbruch sichtbar wird statt still zu laufen.
Tripwires auf `load_torch_file`, `load_checkpoint_guess_config`,
`load_diffusion_model`, `onnxruntime.InferenceSession` und `FaceAnalysis`
haben in keinem Szenario ausgelöst.

| Datei | Tests | Ergebnis |
|---|---|---|
| `test_s1_seams.py` | 5 | 5 PASS (Nahtstellen-Neutralität: Klassenoberfläche, Cache-Signaturen, Lazy-Reihenfolge, Cache-Reuse; Zählertrennung) |
| `test_s1_postprocess_scenarios.py` | 19 | 12 PASS, 7 erwartete Fehlschläge |
| `test_s1_cache_scenarios.py` | 7 | 5 PASS, 2 erwartete Fehlschläge |
| `test_s1_maskdetailer_scenarios.py` | 10 | 6 PASS, 2 erwartete Fehlschläge, 2 SKIP (T2) |
| `test_s1_upscale_save_scenarios.py` | 9 | 7 PASS, 2 erwartete Fehlschläge |
| `test_s1_fixture_widget_mapping.py` | 1 | PASS (Widget-Adressierung der Overrides = Frontend-Expansion) |

### Neu durch Ausführung belegte Defekte

| ID | Befund | Beleg (Test) |
|---|---|---|
| H-1 (Ursache präzisiert) | `utils/cmk_diagnostic.py:127–128` legt `preview` und `images` als dieselbe Liste ab; safetensors lehnt geteilte Tensoren ab ⇒ `STORE_FAILED` bei **jeder** Boundary mit Bild-Diagnose (FaceSwap, Upscale & Save). Nach Eviction oder Neustart nie HIT | `test_baseline_h1_eviction_and_restart_recompute_because_store_fails` |
| H-2 | `CMK Load Image.MODEL SDXL (opt)` ist nicht lazy ⇒ der Checkpoint-Loader läuft in **jedem** Lauf, auch bei gebypasstem Modul | `test_baseline_h2_*` (Bypass, Seams) |
| H-3 | Das Gate fordert wiederholt `MODEL ACTIVE` an (ohne Checkpoint legitim `None`); ComfyUI beendet daraufhin die Lazy-Schleife, `VISUAL ACTIVE` wird nie angefordert ⇒ der FaceSwap-Provider geht verloren (erklärt K3) | `test_baseline_h3_visual_provider_lost_at_gate` |
| H-4 | `CMK_SaveProjectImage` ist OUTPUT_NODE im Subgraph ⇒ Upscaler und Save laufen auch im Bypass (ohne Datei) | `test_baseline_h4_*` |
| H-5 | FaceProcess (std/adv): global an, alle lokalen Zweige aus ⇒ Boundary liefert `ExecutionBlocker`; Visual Compare, Visualizer und Save werden still blockiert | `test_baseline_h5_*` |
| K-5 (behoben, s. u.) | Detailer: global an, lokal aus ⇒ Boundary meldet `DISABLED_PASSTHROUGH_READY`, trotzdem materialisiert Prepare das Modell | Legacy: `LegacyPreK5` in `test_s1_k5_od2.py` |
| OD-2 (behoben, s. u.) | Anforderung nur von MODEL startet Prepare (MODEL läuft über Gate/Boundary) | Legacy: `LegacyPreK5` in `test_s1_k5_od2.py` |
| M3 | „Kein Gesicht gefunden“ ist ein Fehler statt `completed_empty` | `test_baseline_m3_*` |
| M7 | gleicher Dateiname, neuer Inhalt ⇒ identischer Boundary-Schlüssel | `test_baseline_m7_*` |
| T-5 | warmer Lauf (classic/ram_pressure): Save läuft nicht erneut; nach Eviction/Neustart wird neu gespeichert (`image_00002`) | `test_baseline_t5_*`, `test_save_after_restart_writes_next_counter` |
| OD-8 | Upscale & Save gibt das hochskalierte Bild weiter (2×) | `test_baseline_od8_*` |
| Z-7 | MaskDetailer S0–S2: öffentlicher PROCESS enthält Pixel | `test_baseline_z7_*` |
| OD-10 | `pending_input` wird nicht als Status publiziert | `test_baseline_pending_input_*` |

### Offen (nicht modellfrei testbar)

- MaskDetailer-Verarbeitung nach S3 und K6-1…K6-6: brauchen echtes SDXL-Sampling (T2, S1-T7 gesperrt).
- Aktive Verarbeitung von Detailer, FaceProcess, FaceRebuild: Prepare materialisiert das Modell (gestoppt); kein T1-Ersatz ohne Eingriff in die Sampler-Nahtstelle.
- M1 (Artefakt-Token), M2 (Presentation-Label im Fingerprint), A15/A16 (`CMKBranchActivity` existiert noch nicht).
- `.mjs`-Tests: kein `node` (S1-O1).

### Gesamtlauf 2026-11-04 (`run_tests.py --no-node`, mit `CMK_EXTRA_MODEL_PATHS`)

| Gruppe | Dateien | Tests | PASS | erwarteter Fehlschlag | SKIP | unerwarteter Fehler |
|---|---|---|---|---|---|---|
| S1 (neu) | 13 | 84 | 61 | 21 | 2 | 0 |
| Bestand | 20 | 159 | 157 | – | – | 2 in `test_image_resize_port_contract.py` (bestand schon vor S1) |
| `.mjs` | 6 | – | – | – | – | nicht ausgeführt (kein `node`) |

Geschützte Caches unverändert. Bericht: `_reports/full_suite.json`.

## Erste produktive Korrektur: H-1 (2026-11-04)

**Freigabe:** ausschließlich H-1 im Boundary-Serialisierungspfad (Option 1).

### Belegte Ursache

safetensors lehnt Tensoren mit überlappendem gemeinsamen Speicher ab
(„Some tensors share memory“). `_encode_diagnostic_value` vergab für jedes
Vorkommen einer Referenz einen eigenen Schlüssel. Geteilte Referenzen sind in
DIAGNOSTIC aber zulässig und kommen an mehreren Stellen vor (Pfade aus dem
Testlauf):

| Fixture | gemeinsam gespeichert | Erzeuger |
|---|---|---|
| Upscale & Save | `preview[i]` = `images[i]` | `make_diagnostic_payload` (gleiche Liste) |
| FaceSwap | `stages[i].image` = `preview[i]` = `images[i]` = `metadata.diagnostic_groups[0].stages[i].image`, dazu `diagnostic_groups[0].preview/images` | `CMKDiagnosticConcat` und Stage-Rückfall in `make_diagnostic_payload` |

### Änderung

`pipe/cmk_module_boundary_cache.py` (einzige Produktdatei):
- `_encode_diagnostic_value(..., views)`: Eine identische Tensor-Ansicht
  (Gerät, Storage, Offset, Shape, Stride, dtype) verwendet den bereits
  vergebenen Schlüssel wieder.
- `_overlaps_stored_tensor`: Eine abweichende Ansicht wird nur kopiert, wenn
  sie sich mit bereits gespeichertem Speicher überschneidet. Das ist dieselbe
  Regel, nach der safetensors ablehnt. Disjunkte Ansichten eines Speichers
  bleiben ohne Kopie.
- `_save_diagnostic` übergibt das Register (`{}`). Decoder, JSON-Format
  (`{"__cmk_diagnostic_tensor__": key}`) und DIAGNOSTIC-Struktur sind
  unverändert. `image_resize.py` nutzt `_save_diagnostic` und profitiert ohne
  Änderung.

Nicht geändert: `make_diagnostic_payload`, `CMKDiagnosticConcat`, Lazy-Verträge,
Gate-Routing, Fingerprints, Modell-Lifecycle, Subgraphen, Workflows.

### Teständerungen (`tests/`)

| Datei | Änderung |
|---|---|
| `test_s1_diagnostic_serialization.py` (neu) | 7 Roundtrip-Tests über den echten Save/Load-Pfad: identische Referenzen (numpy, torch), disjunkte Views ohne Kopie, überlappende Views mit Kopie, nicht zusammenhängende identische Views, mehrfach verschachtelte Diagnosegruppen, reine Text-Diagnose |
| `test_s1_cache_scenarios.py` | H-1-Paar ersetzt durch reguläre Tests: Kaltlauf `MISS_STORED` (Regression), persistenter `HIT` nach Eviction, Neustart `classic` und Neustart `ram_pressure`, IMAGE- und VISUAL-Pixelhashes gleich dem Kaltlauf. H-3 und H-4 als eigene Paare (erwarteter Fehlschlag plus Baseline) |
| `test_s1_cache_contract.py` | H-1-Soll-Test regulär; Baseline jetzt Regression (`MISS_STORED`) |
| `harness/scenarios.py` | VISUAL-Kanäle und gespeicherte Dateien werden zusätzlich mit Pixelhash erfasst |

### Gemessene Werte nach der Korrektur

| Fixture | Variante | Lauf 1 | Lauf 2 |
|---|---|---|---|
| FaceSwap std | Eviction (`ram_pressure`), Neustart `classic`, Neustart `ram_pressure` | `MISS_STORED`, Swap 1×, Detektor 2× | `HIT_READY`, `HIT_READY`, `HIT`; **0 Engine-Aufrufe**, Compute-Node nicht ausgeführt, 0 Nodes aus dem ComfyUI-Cache, IMAGE identisch |
| Upscale & Save | dieselben drei | `MISS_STORED`, Upscale 1×, `image_00001` | `HIT_READY`, `HIT`; IMAGE und VISUAL identisch; **Upscale weiterhin 1×** und Save 1× (`image_00002`, pixelgleich) wegen H-4 |
| beide | warm `classic` / `ram_pressure` | – | 0 Engine, Boundary nicht betreten (ComfyUI-Cache) |

### Weiterhin offen (nicht Teil von H-1)

- **H-3:** FaceSwap publiziert auch nach HIT keinen VISUAL-Provider.
  `test_faceswap_hit_runs_no_compute` besteht derzeit **nur wegen H-3**. Sobald
  H-3 behoben ist, zieht F-03 die Compute-Node wieder.
- **H-4:** Upscaler und Save laufen trotz Boundary-HIT.

### Gesamtlauf nach H-1 (`_reports/full_suite_h1.json`)

| Gruppe | Dateien | Tests | PASS | erwarteter Fehlschlag | SKIP | unerwarteter Fehler |
|---|---|---|---|---|---|---|
| S1 | 14 | 95 | 72 | 21 | 2 | 0 |
| Bestand | 20 | 159 | 157 | – | – | 2 in `test_image_resize_port_contract.py` (unverändert, schon vor S1) |
| `.mjs` | 6 | – | – | – | – | nicht ausgeführt (kein `node`) |

Geschützte Caches unverändert. Keine Modellladung, Tripwires nicht ausgelöst.

## Zweite produktive Korrektur: H-4, Upscale & Save 1.4.0 (2026-11-04)

**Freigabe:** Variante B und K-3 Option A.

### Ursache (belegt durch Experiment)

Beim Boundary-HIT zogen **zwei unabhängige Kanten** den Upscaler:
1. `CMK_SaveProjectImage` (OUTPUT_NODE mit nicht-lazy Eingängen) lag vor der Boundary.
2. Das Gate bekam `VISUAL ACTIVE` direkt vom Upscaler (Instanz von F-03).

Erst ohne beide Kanten war die Zahl der Upscaler-Aufrufe 0. Der Upscaler ruft
vor dem Hochskalieren `unload_all_models()` auf. Jeder HIT entlud deshalb alle
Diffusionsmodelle.

### Änderungen

| Datei | Änderung |
|---|---|
| `nodes/io/save_project_image.py` | neuer `CMKSaveEndpoint`: OUTPUT_NODE ohne Ausgänge, Lazy-Eingänge, `IS_CHANGED` liefert pro Ausführung einen neuen Wert (nur konstant `False` bleibt gecacht), schreibt über `CMK_SaveProjectImage.run`. `CMK_SaveProjectImage` ist unverändert |
| `cmk_mappings.py` | Registrierung „CMK Save Endpoint“ |
| `subgraphs/CMK Flow · Upscale & Save.json` | Modulversion 1.4.0. Der Save-Node vor der Boundary entfällt; die Boundary bekommt ihr Bild vom Upscaler; `IMAGE ACTIVE` kommt aus `SG_IN.IMAGE`; `VISUAL ACTIVE` liefert `CMKVisualProvider`, der das Boundary-Bild als `result` und das Eingangsbild als `before` publiziert; der Endpunkt liest das Boundary-Ergebnis |
| `ARCHITECTURE.md` | Abschnitt 3.0 Upscale & Save; veralteter Rückfall-Satz entfernt |
| `tests/contracts/legacy_registry.json` | Full-Flow-Definition vor H-4 als Legacy |

Unverändert: Upscaler, Boundary, Gate, `CMKVisualProvider`, Visualizer, alle
Workflows, H-2, H-3, H-5.

### Tests

- `test_s1_upscale_save_legacy_reference.py` (neu, unverändert seit der
  Aufzeichnung vor der Korrektur): feste Hashes, Shape, Pixel, Zähler, Status,
  Aufrufzahlen und VISUAL der Full-Flow-Definition. Vor und nach der Korrektur
  5/5 grün.
- `test_s1_upscale_save_h4.py` (neu): vor der Korrektur 15 Tests mit 5 strikt
  erwarteten Fehlschlägen und 5 Baselines. Gegen die echten Exporte wurden alle
  5 Soll-Tests zum unerwarteten Erfolg und alle Baselines rot. Danach wurden sie
  auf 9 reguläre Regressionstests umgestellt.
- Umgestellt (erwarteter Fehlschlag → PASS, Baselines entfernt):
  `test_s1_upscale_save_scenarios.py` (OD-8, T-5),
  `test_s1_cache_scenarios.py` (H-4), `test_s1_postprocess_scenarios.py`
  (H-4 im Bypass). Der Direktvergleich Legacy = kanonisch ist ersetzt durch
  „beide speichern dieselbe hochskalierte Kopie“.
- Harness: eigener Zähler `watched_calls` für `unload_all_models`.

### Gemessene Werte (echter Export, T1-Seams)

| Szenario | Ergebnis |
|---|---|
| Kaltlauf | `MISS_STORED`, Upscale 1×, `unload_all_models` 1×, `image_00001`, IMAGE OUT = IMAGE IN |
| Warmlauf (`classic`) | 0 Upscale, 0 Entladen; Endpunkt läuft erneut und schreibt `image_00002`; Loader, Upscaler, Boundary, Gate und Provider kommen aus dem ComfyUI-Cache |
| Eviction / Neustart | `HIT_READY`, `HIT`; kein Upscaler, 0 Entladen; `image_00002` pixelgleich mit dem Boundary-Ergebnis und dem `result`-Kanal |
| Bypass | kein Upscaler, keine Boundary, kein Provider, kein `CMK_SaveProjectImage`, keine Datei, kein Cache-Status. Der Endpunkt wird einmal als No-op betreten (Lazy-Anforderung `[]`) und kommt im zweiten Lauf aus dem Cache |
| VISUAL | ein Provider `CMKVisualProvider`, Kanäle `result` und `before` |

### Gesamtlauf nach H-4 (`_reports/full_suite_h4.json`)

| Gruppe | Dateien | Tests | PASS | erwarteter Fehlschlag | SKIP | unerwarteter Fehler |
|---|---|---|---|---|---|---|
| S1 | 16 | 104 | 85 | 17 | 2 | 0 |
| Bestand | 20 | 159 | 157 | – | – | 2 in `test_image_resize_port_contract.py` (unverändert, schon vor S1) |
| `.mjs` | 6 | – | – | – | – | nicht ausgeführt (kein `node`) |

Validator: 0 × `DRIFT`. Geschützte Caches unverändert, keine Modellladung.

## Dritte produktive Korrektur: H-3 / F-03, VISUAL-Publikation an der Boundary (2026-11-04)

**Freigabe:** Variante B, E1 = a, E2 = a, G1.

### Befund (vor der Änderung als Test belegt)

- **H-3:** Das Gate fragte pro Aufruf genau einen fehlenden Eingang an. Ohne
  Checkpoint ist MODEL legitim `None`; ComfyUI verwirft die erneute Anfrage
  bereits ausgewerteter Eingänge, und das Gate lief, bevor VISUAL je angefordert
  war. Die gesamte VISUAL-Pipe ging dadurch verloren, einschließlich
  vorgeschalteter Provider, im aktiven Zweig wie im Bypass. Nachgewiesen für
  FaceSwap Standard und Advanced sowie für alle sechs Checkpoint-Module im
  Bypass ohne Modell.
- **F-03:** `VISUAL ACTIVE` kam vom Compute-Node. Mit G1 allein rechnete FaceSwap
  beim HIT wieder (Swap 1×).

### Änderungen

| Datei | Änderung |
|---|---|
| `pipe/cmk_module_boundary_cache.py` | Mixin `_VisualPublication` für die vier Boundaries; bisherige Klassenkörper als Kerne. Neue Eingänge `VISUAL IN` und `VISUAL` (lazy), neuer letzter Ausgang `VISUAL`. Provider-Ablage in `.visual` und `.visual-tensors`, außerhalb der bisherigen Prune-Muster. Alte Einträge ohne Provider-Datei werden einmal neu berechnet. `_cache_key` schließt beide VISUAL-Eingänge aus. Schreib- und Lese-Code des Diagnose-Speichers gemeinsam genutzt (`_write_encoded`, `_read_encoded`). `CMKUpscaleSaveBoundaryCache` erbt vom unveränderten FaceSwap-Kern; seine Schnittstelle ist per Test fixiert |
| `pipe/cmk_visual.py` | `publish_providers` (Regel „ersetzen oder anhängen“); `register_provider` nutzt sie |
| `pipe/cmk_family_result.py` | G1 in `CMKModuleBypassGate.check_lazy_status`: zuerst die Ergebnis-Eingänge gemeinsam, danach die Hilfs-Eingänge gemeinsam |
| acht kanonische Subgraphen | `VISUAL ACTIVE` ← `Boundary.VISUAL`; Boundary `VISUAL` ← Compute bzw. SEGS CONCAT; `VISUAL IN` ← `SG_IN.VISUAL`; Modulversion jeweils +0.1 |
| `ARCHITECTURE.md` | Abschnitt 3.2 ergänzt |
| `tests/contracts/legacy_registry.json` | 13 eingebettete Modul-Definitionen vor H-3 als Legacy (E1 = a) |

Unverändert: ZIT-Familiengate, Generations-Gates (`CMKSamplerBypassGate`),
MaskDetailer, H-2, H-5, die H-4-Architektur (Save-Endpunkt, Legacy-Referenz,
Schnittstelle der Upscale-&-Save-Boundary).

### Abweichungen während der Umsetzung (dokumentiert, innerhalb der Freigabe gelöst)

1. **Routing-Erkennung:** Bei außen unverbundenem Modul-VISUAL entfernt das
   Frontend den Link zu `VISUAL IN`. Die Erkennung prüft deshalb den internen
   Link `VISUAL`.
2. **Kanal-Reihenfolge:** Der JSON-Index wird mit sortierten Schlüsseln
   geschrieben (`after, before`). Die Kanäle werden deshalb als geordnete Paare
   gespeichert.
3. **G1 und H-5:** Fordert man alle Eingänge auf einmal an, wird
   `DIAGNOSTIC ACTIVE` auch dann vom Compute-Node geholt, wenn die Boundary
   blockiert. Bei FaceProcess mit allen lokalen Zweigen aus lief dadurch
   Prepare. Das ist nicht die dokumentierte Blockade durch konvergierende
   Subgraph-Ausgänge, sondern ein Verlust des bisherigen Kurzschlusses. Lösung
   innerhalb von G1: zwei Anforderungsphasen. Der H-5-Baseline-Test ist damit
   wieder unverändert grün.

Die dokumentierte Blockade durch konvergierende globale Subgraph-Ausgänge trat
in keinem Lauf auf.

### Tests

- `test_s1_visual_publication.py` (neu): vor der Änderung 14 Tests, davon 6
  strikt erwartete Fehlschläge mit Baselines. Gegen die echten Exporte meldeten
  alle 6 einen unerwarteten Erfolg; danach umgestellt auf 12 reguläre Tests.
  Abgedeckt: unveränderte Cache-Schlüssel und Manifeste aller acht Module,
  Upstream- und Modul-Provider bei MISS, identischer Provider (Kanäle und
  Metadaten) beim HIT nach Neustart und Eviction ohne Compute, Bypass mit und
  ohne Modell, lokal deaktiviert ohne Provider, Gate-Protokoll (Matrix und
  Anforderungsfolge), fixierte Schnittstelle der Upscale-&-Save-Boundary.
- Umgestellt: H-3-Paare in `test_s1_postprocess_scenarios.py` und
  `test_s1_cache_scenarios.py`.
- Harness: `upstream_visual`, `drop_model`, `visual_meta`; `boundary_key`
  liefert zusätzlich das Abhängigkeits-Manifest.

### Weiterhin offen

- Detailer, FaceProcess und FaceRebuild sind im aktiven Zweig nicht modellfrei
  prüfbar (Prepare). Ihre Publikation ist nur über Bypass, Passthrough und den
  gemeinsamen Mixin-Code abgesichert.
- „Kein Provider ohne Inhalt“ (SEGS CONCAT ohne Treffer) ist modellfrei nicht
  auslösbar; FaceSwap bricht ohne Gesicht ab (M3).
- Live-Vorschau im Frontend ungeprüft (kein `node`).
- `ARCHITECTURE.md` §5.8a beschreibt Upscale & Save noch im Stand vor H-4.

### Nachträge aus dem Gesamtlauf nach H-3 / F-03

- **Import:** `test_stop_after_handoff.py` ersetzt `cmk_visual` durch einen Stub
  ohne `normalize_visual`. Die Boundary importiert deshalb das Modul selbst
  (`from . import cmk_visual`) statt einzelner Namen.
- **Blockierte Ergebnisse:** Bei nicht verdrahteter Boundary (Alt-Definition)
  gab der Mixin für VISUAL `None` statt `ExecutionBlocker` aus. Jetzt ist bei
  blockiertem Ergebnis immer auch VISUAL blockiert.
- **Offen, zur Entscheidung:** `test_stop_after_handoff.py` prüft die genaue Zahl
  der Boundary-Ausgänge (3 bzw. 4). Durch den freigegebenen `VISUAL`-Ausgang sind
  es 4 bzw. 5. Der inhaltliche Vertrag hält nachweislich: keine
  Lazy-Anforderung, alle Ausgänge einschließlich VISUAL sind `ExecutionBlocker`;
  Upscale & Save unverändert mit 4. Bestandstest nicht geändert.

### Gesamtlauf (`_reports/full_suite_h3b.json`)

| Gruppe | Dateien | Tests | PASS | erwarteter Fehlschlag | SKIP | Fehler |
|---|---|---|---|---|---|---|
| S1 | 17 | 114 | 97 | 15 | 2 | 0 |
| Bestand | 20 | 159 | 152 | – | – | 2 alt (`test_image_resize_port_contract.py`), **5 neu** (`test_stop_after_handoff.py`, nur die Ausgangszahl) |
| `.mjs` | 6 | – | – | – | – | nicht ausgeführt (kein `node`) |

Geschützte Caches unverändert, keine Modellladung.

### H-3 / F-03 abgeschlossen: Endstand (`_reports/full_suite_final.json`)

- `test_stop_after_handoff.py` angepasst (freigegeben): Die bisherigen
  Ausgangszahlen bleiben als Ergebnis-Ausgänge erhalten. Der VISUAL-Ausgang wird
  zusätzlich gezählt und ebenfalls auf `ExecutionBlocker` geprüft.
  `CMKUpscaleSaveBoundaryCache` ist ausdrücklich ohne VISUAL und mit 4 Ausgängen
  abgesichert. Kein Produktcode geändert.
- `ARCHITECTURE.md` §5.8a an den H-4-Vertrag aus §3.0 angeglichen.

| Gruppe | Dateien | Tests | PASS | erwarteter Fehlschlag | SKIP | Fehler |
|---|---|---|---|---|---|---|
| S1 | 17 | 114 | 97 | 15 | 2 | 0 |
| Bestand | 20 | 159 | 157 | – | – | 2 alt (`test_image_resize_port_contract.py`, schon vor S1) |
| `.mjs` | 6 | – | – | – | – | nicht ausgeführt (kein `node`) |

Keine neuen Regressionen. Geschützte Caches unverändert, keine Modellladung.

## Vierte produktive Korrektur: H-2, verzögerte Modell-Materialisierung (2026-11-04)

**Freigabe:** Variante D, E1 bis E3.

- **Ursache:** `CMK Load Image` reicht `MODEL SDXL (opt)` nur durch, der Eingang
  ist aber nicht lazy. `CMKCheckpointVAELoaderPipe` lädt sofort. Eigenständige
  PostProcess-Workflows luden den Checkpoint deshalb in jedem Lauf, auch im
  Bypass und bei modellfreien Modulen.
- **Neu:** `CMKPostProcessModelPipe` („CMK PostProcess Model“, `CMK/Flow/Input`)
  liefert den `make_postprocess_model_spec`-Vertrag der PostProcess-Boundary.
  Gleiche Widgets wie der Checkpoint-Loader, Auswahl `None` am Ende. Die
  Materialisierung erfolgt beim ersten echten Verbraucher
  (`resolve_postprocess_model`). Die `None`-Meldung nennt jetzt beide Quellen.
  Registriert in `cmk_mappings.py`, Metadaten in `flow_node_metadata.json`,
  `browser_content_en.json` und `cmk_node_dimensions.js`.
- **Unverändert:** `CMKCheckpointVAELoaderPipe`, `CMKImageLoadAndResizePipe`,
  Generations-Nodes, Sampling, Cache-Architektur, H-1, H-3/F-03, H-4,
  01 START HERE v2.
- **Migriert:** genau die neun eigenständigen Showcase-Workflows; nur der
  Modellquellen-Node wird an Ort und Stelle getauscht. Originale eingefroren
  unter `fixtures/legacy_workflows/showcase_pre_h2` (SHA-256 in
  `contracts/legacy_postprocess_model_workflows.json`).
- **Fixtures:** acht Checkpoint-Fixtures über das echte Frontend neu exportiert;
  `upscale_save` als Kontrolle im API-Prompt identisch. Das alte, sofortige
  Laden ist über `detailer_std_legacy_checkpoint` und
  `maskdetailer_legacy_checkpoint` festgehalten (Exporte vor H-2).
- **Befund Cache-Identität:** Die Modellquelle ist Teil des Upstream-Fingerprints.
  Migrierte Workflows berechnen ihre Boundaries daher einmalig neu (MISS):
  Schlüssel bzw. bei FaceProcess das Branch-Manifest. Der H-3-Test für fixierte
  Schlüssel belegt, dass mit zurückgesetzter Quellklasse exakt die alten Werte
  entstehen.
- **Tests:** `test_s1_postprocess_model.py` (13 Tests: Legacy-Referenz, Migration,
  Bypass, aktiver Pfad, wartender MaskDetailer, Spezifikation, Lebenszyklus,
  `None`). Umgestellt: H-2-Paare in `test_s1_postprocess_scenarios.py` und
  `test_s1_seams.py`. Offen bleiben K-5 und OD-2 (erwartete Fehlschläge
  unverändert).

## Fünfte produktive Korrektur: K-5 / OD-2 (2026-11-04)

**Freigabe:** E1 = K5-A, E2 schlanke Diagnose, E3 Eingangs-LOG + Statusblock,
E4 `CMKModelForwardPipe` optional, E5 OD2-B für sechs Subgraphen, E6 Neustart-HIT
zurückgestellt, E7 Tests und Schemawechsel, E8 FaceProcess erst mit H-5,
E9 modellfreier CLIP/LoRA-Seam. Referenz: Git `cb9d24b`.

- **Ursache K-5 (Experiment belegt):** zwei unabhängige Compute-Abhängigkeiten im
  Passthrough: Boundary fordert IMAGE/LOG vom Compute-Node, Gate holt
  `DIAGNOSTIC ACTIVE` direkt vom Compute-Node. Erst beide entfernt ⇒ kein Prepare.
- **Ursache OD-2:** SG_OUT.MODEL kam vom Gate; das Gate wird als Ganzes
  ausgewertet und zieht bei aktivem Modul Boundary, Compute und Prepare nach.
- **Produktcode:**
  - `pipe/cmk_model_forward.py`: MODEL optional, `None` wird durchgereicht.
  - `pipe/cmk_module_boundary_cache.py`: Mixin `_PassthroughSources` (interne
    Eingänge `IMAGE IN`, `LOG IN`, `diagnostic`; Ausgang `diagnostic` nach VISUAL;
    Passthrough ohne Core; Diagnose bei MISS gespeichert, bei HIT geladen).
    `CMKDetailerBoundaryCache` nutzt ihn. Schema `cmk_detailer_boundary_v4`;
    die neuen Eingänge sind vom Fingerprint ausgeschlossen.
- **Subgraphen:** Detailer (std/adv) 1.2.0, FaceProcess (std/adv) 1.3.0,
  FaceRebuild (std/adv) 1.2.0: SG_OUT.MODEL ← `CMKModelForwardPipe` ← SG_IN.MODEL,
  Gate-Eingänge `MODEL ACTIVE/BYPASS` entfernt. Nur Detailer zusätzlich:
  `IMAGE IN`/`LOG IN` ← SG_IN, `diagnostic` ← Compute, Gate `DIAGNOSTIC ACTIVE`
  ← Boundary. FaceSwap und FaceProcess-Passthrough (H-5) unverändert.
- **Fixtures:** sieben Fixtures über das Frontend 1.57.4 neu exportiert
  (Detailer, FaceProcess, FaceRebuild je std/adv, `chain_postprocess`). Exporte
  vor K-5 eingefroren als `detailer_std_legacy_pre_k5` / `detailer_adv_legacy_pre_k5`.
  Schlüssel vor der Änderung: `contracts/boundary_keys_pre_k5.json`.
- **Harness:** `fake_detailer` (E9: SAM, CLIP-Layer, LoRA, Modell-Patches,
  CLIP-Encode und Smart-Detailer-Lauf ersetzt; `resolve_postprocess_model` echt
  und gezählt), `forward_only(drop_model)`, `boundary_key(schema)`, LOG- und
  Diagnose-Zusammenfassung in `outputs`.
- **Tests:** neu `test_s1_k5_od2.py` (27 Tests). Umgestellt: K-5/OD-2 in
  `test_s1_postprocess_scenarios.py` (erwartete Fehlschläge und Baselines
  aufgelöst). Angepasst als direkte Folge der freigegebenen Änderungen:
  `test_synced_user_subgraphs.py` (Forward im Layout-Pin FaceRebuild Advanced),
  `test_s1_seams.py` (8 statt 7 verglichene Nodes), `test_s1_visual_publication.py`
  (Detailer-Schlüssel mit Schema v3 verglichen), Bestandstest
  `test_stop_after_handoff.py` (Detailer-Boundary blockiert 4 + VISUAL Ausgänge).

### Neuer, vorbestehender Befund K5-B (behoben, s. „Sechste produktive Korrektur“)

Das Abhängigkeitsmanifest der Detailer-Boundary fingerprintet den Zweig mit
`cmk_detailer_branch_v6` und ohne Eingangsausschlüsse; `CMK_SmartDetailerPipe`
veröffentlicht seine Revision unter `cmk_detailer_branch_v7` mit Ausschlüssen.
Das Manifest wird nie vollständig ⇒ die Detailer-Boundary speichert nie; nach
Eviction oder Neustart läuft Prepare erneut (nur der Zweig-Pickle trifft).
FaceProcess analog (`v4` gegenüber `v8`). Bereits in `cb9d24b` vorhanden.
Belegt durch `test_baseline_k5b_detailer_boundary_never_stores` und den strikt
erwarteten Fehlschlag `test_boundary_hit_after_eviction_runs_no_prepare`. Der
Diagnose-HIT-Pfad ist isoliert in `PassthroughSourcesMixin` geprüft.

### Gesamtlauf nach K-5 / OD-2 (`_reports/full_suite_k5.json` + Nachläufe `full_suite_k5_rerun_*`)

| Gruppe | Tests | PASS | erwarteter Fehlschlag | SKIP | unerwartet |
|---|---|---|---|---|---|
| S1 | 152 | 137 | 13 | 2 | 0 |
| Bestand | 159 | 157 | – | – | 2 in `test_image_resize_port_contract.py` (Altfehler) |
| `.mjs` | 7 Dateien | 7 | – | – | 0 |

Geschützte Caches unverändert.

## Sechste produktive Korrektur: K5-B, persistente Detailer-/FaceProcess-Boundaries (2026-11-04)

**Freigabe:** D1 = A2, D2 ja, D3 = Zusatz C, D4 kein Schemawechsel, D5 FaceProcess-Seams,
D6 Manifest-Referenzen aktualisieren. Basis: Git `92b7a67`.

- **Ursache (Experiment belegt):** Das Abhängigkeitsmanifest berechnete den
  Zweig-Schlüssel mit eigenen Konstanten (`cmk_detailer_branch_v6`,
  `cmk_faceprocess_branch_v4`) und ohne die Eingangsausschlüsse des Zweigs. Jede
  der beiden Abweichungen verhinderte allein ein vollständiges Manifest; erst
  beide korrigiert ⇒ `MISS_STORED` und HIT. Zusätzlich verlangte der Detailer-HIT
  eine Session-Kopie von MODEL ⇒ nach Neustart kein HIT.
- **Produktcode:**
  - `nodes/image/smart_detailer.py`, `pipe/cmk_faceprocess.py`: Klassenmethode
    `dependency_cache_key` als einzige Schlüsselberechnung des Zweigs (Branch-Cache,
    Revision und Manifest). Schlüssel unverändert.
  - `pipe/cmk_persistent_cache.py`: `build_upstream_cache_manifest` nimmt Scope und
    Schlüssel aus der Zweig-Klasse.
  - `pipe/cmk_module_boundary_cache.py`: `_DETAILER_BRANCH_SPECS` /
    `_FACEPROCESS_BRANCH_SPECS` verweisen nur noch auf die Zweig-Klassen (lazy
    importiert); Detailer-HIT ohne Session (MODEL vom aktuellen Eingang, beim HIT
    angefordert). `_DETAILER_SESSION` bleibt für den Reset-/Statistikvertrag
    bestehen, wird vom Detailer aber nicht mehr befüllt.
- **Tests:** neu `test_s1_k5b.py` (14 Tests: statische Schlüsselgleichheit über 14
  Zweige in 7 Fixtures, Zweig-Schlüssel gegen `contracts/branch_keys_pre_k5b.json`,
  Revisions-Invalidierung, FaceProcess-Laufzeit, K5-C). `test_s1_k5_od2.py`:
  K5-B-Baseline und erwarteter Fehlschlag aufgelöst; Detailer-HIT nach Eviction
  und Neustart. Harness: `manifest_keys`, Seam-Flag `fake_postprocess`
  (Detailer und FaceProcess, D5). D6: Manifeste in `boundary_keys_pre_h3.json`
  und `boundary_keys_pre_k5.json` aktualisiert, Schlüssel geprüft unverändert.
- **Gemessen (T1, modellfrei):** Detailer std: Kaltlauf `MISS_STORED`; nach
  Eviction und nach Neustart `HIT` mit 0 Prepare, 0 Compute, 0
  `resolve_postprocess_model`; IMAGE, LOG, Diagnose, VISUAL identisch.
  FaceProcess std: `MISS_STORED`, `HIT` nach Eviction und Neustart; Prepare und
  Compute laufen auf dem HIT weiter über `DIAGNOSTIC ACTIVE` (H-5, unverändert,
  als Baseline + erwarteter Fehlschlag festgehalten).
- **Revisionen:** FaceProcess-Zweig neu berechnet ⇒ neue Revision ⇒
  `STALE_DEPENDENCIES` ⇒ `MISS_STORED`. Detailer-Revision ist deterministisch
  (Artefakt-Schlüssel): manipulierte Revision wird erkannt, der Zweig-HIT stellt
  sie wieder her, die Boundary trifft unverändert.

### Neuer, vorbestehender Befund K5-C (nicht korrigiert)

Lokal deaktivierte Zweige speichern keinen Pickle und veröffentlichen keine
Revision. Das Manifest eines Advanced-Moduls mit mindestens einem deaktivierten
Zweig (Standard: 2 von 3) ist deshalb nie vollständig; Detailer und FaceProcess
Advanced speichern weiterhin nicht. Belegt durch
`test_baseline_k5c_advanced_with_disabled_branch_never_stores` und den strikt
erwarteten Fehlschlag `test_advanced_boundary_hits_after_eviction`.
