## 2026-07-20 — ControlNet-Parameter und konsistente Flow-Darstellung

- `CMK Flow · 05 ControlNet (optional)` zeigt die technischen Advanced-Parameter wieder klein als `start_percent` und `end_percent` im verständlichen Bereich `0–100`; intern werden sie weiterhin auf ComfyUIs Bereich `0.0–1.0` normiert.
- Bestehende gespeicherte Fraction-Werte werden beim Laden einmalig in Prozentwerte migriert. Neue Defaults sind `STRENGTH = 1.50`, `start_percent = 0` und `end_percent = 30`.
- Der kanonische Subgraph `30 Detailer · Advanced` und sein Referenzworkflow verwenden nun die einheitliche persistente Außenfläche `600 × 1225`. Dieser Standard ist in den Design Guidelines dokumentiert.
- `CMK Flow · Checkpoint & VAE` verwendet für neue Nodes bevorzugt `juggernautXL_ragnarok.safetensors`, sofern das Modell installiert ist. Der Flow Browser enthält den aktualisierten Aufbau-Screenshot unter einem neuen cache-sicheren Asset-Namen.

## 2026-07-19 — Verpflichtender lokaler FaceSwap ContentGuard

- Alle öffentlichen CMK-FaceSwap-Pfade prüfen Quell- und Zielbilder vor der Verarbeitung; der Videopfad prüft jedes Ziel-Frame.
- Explizite Inhalte, geschätzte Minderjährigkeit, unsichere Altersschätzungen unter 25 sowie fehlende oder fehlerhafte Schutzmodelle brechen fail-closed ab.
- Der Guard besitzt keine öffentliche Umgehungsoption. Deaktivierte FaceSwap-Nodes bleiben unveränderte Pass-through-Pfade.
- Video-Caches tragen die ContentGuard-Version in ihrer Signatur. Bei einem Guard-Abbruch werden alle im aktuellen Lauf neu erzeugten Swap-Segmente verworfen.
- NudeNet läuft lokal für die Erkennung expliziter Inhalte; InsightFace liefert die lokale Altersschätzung.
- Guard-Abbrüche erscheinen für Anwender knapp als `ContentGuard activated — FaceSwap aborted`; der neutrale Diagnosecode bleibt für die Fehlersuche erhalten.

## 2026-07-19 — Image Input ohne verzerrtes Standard-Resize

- `CMK Flow · Image Input` zeigt `CROP` und `CROP POSITION` wieder direkt in der normalen Node-Oberfläche.
- Neue Nodes starten mit aktivem, seitenverhältnistreuem Crop; `center`, `top`, `bottom`, `left` und `right` steuern die Verankerung des erhaltenen Bildausschnitts.
- Bereits gespeicherte Nodes behalten ihren gespeicherten Crop-Wert und können einmalig auf `ON` gestellt werden.

## 2026-07-19 — Native Installation ohne fremde Custom Nodes

- CMK Flow registriert sich in einer leeren ComfyUI-Installation ohne Impact Pack, Impact Subpack, ReActor oder AIO Aux Preprocessors.
- Eine CMK-eigene Laufzeit übernimmt Ultralytics-Erkennung, `SEGS`, Detailer-Sampling, SAM-Maskierung und Pasteback.
- FaceProcess-Restore verwendet die vorhandenen CMK-Engines für InsightFace-Erkennung, GPEN/CodeFormer und maskiertes Pasteback statt ReActor-Code zu laden.
- ControlNet-Preprocessing und SDXL-Conditioning verwenden CMK- beziehungsweise ComfyUI-native Pfade.
- `requirements.txt` und die Installationsanleitung beschreiben nun den vollständigen GitHub-Clean-Room-Weg.
- CMK warnt verständlich, wenn im aktiven ComfyUI-Profil Vue Nodes / Nodes 2.0 deaktiviert ist, und entfernt den Hinweis unmittelbar nach dem Aktivieren.
- Die Installationshinweise nennen `Live preview method: auto` als Voraussetzung für laufende Sampler- und Refiner-Vorschauen.
- README und „About CMK Flow“ würdigen den ComfyUI LoRA Manager als gestalterische Inspiration für die integrierte Browsererfahrung.

## 2026-07-19 — Videoverwaltung und Projektauftritt im Flow Browser

- Der Flow Browser besitzt eine integrierte Videoverwaltung für dauerhaft gespeicherte CMK-Videoprojekte. Sie zeigt Speicherbelegung, Segmente und zusammengeführte Arbeitsvideos, öffnet Projekte erneut im Flow und kann deren Speicherordner anzeigen.
- Arbeitsdateien lassen sich projektweise oder gesammelt nach ausdrücklicher Bestätigung löschen. Laufende Prozesse und Fehlschläge werden verständlich angezeigt; bestehende Segmente bleiben bis zur gezielten Löschung für die spätere Weiterarbeit erhalten.
- Der About-Dialog trägt nun das CMK-Logo dezent in der Titelzeile und enthält einen zurückhaltenden Link zur freiwilligen Unterstützung über PayPal. Der Hinweis stellt ausdrücklich klar, dass eine Unterstützung keinen Einfluss auf den Funktionsumfang von CMK Flow hat.
- Der About-Dialog und das README verlinken den öffentlichen Quellcode und die Dokumentation direkt unter `https://github.com/CMKFlow/cmk_nodes`.

## 2026-07-17 — FaceSwap-Referenzen

- Zwei klar getrennte FaceSwap-Referenzen ergänzt: `CMK Flow · FaceSwap Image` als empfohlener Einstieg mit einer Instanz und `CMK Flow · FaceSwap Image · Advanced` mit drei parallelen Zielpositionen. Advanced verwendet eine eigene Subgraph-ID und enthält bewusst weder Detailer noch FaceProcess. Im Flow Browser erscheint Advanced als untergeordnete, bewusst wählbare Variante des voreingestellten FaceSwap-Eintrags.
- Die verrutschten Widget-Werte der Standardreferenz korrigiert: `GLOBAL ENABLE`, `ENABLE`, Swap-Modell, Detektor, Ziel- und Quellgesicht sowie Enhancer und Blend werden wieder den richtigen UI-Feldern zugeordnet. Verbliebene Pins wurden entfernt. Die Advanced-Platzierung spricht nun korrekt von bis zu drei Zielpersonen.
- `CMK FaceSwap Image -Pipe-` besitzt den Advanced-Parameter `IDENTITY STRENGTH` (`0.50–1.50`, Standard `1.00`). Er gewichtet den normalisierten INSwapper-Identity-Latent vor der Modellinferenz; `BLEND` bleibt unabhängig davon die abschließende Bildmischung. Der Pfad bei `1.00` verwendet weiterhin unverändert die native INSwapper-Methode.
- Die in ComfyUI getesteten User-Subgraphen für FaceSwap Standard und Advanced als kanonische Fassungen übernommen. Standard nutzt `IMAGE PROCEED` direkt und `feather = 0`; Advanced führt drei parallele `SEGS PROCESSED`-Ausgänge zusammen und startet mit deaktiviertem mittleren Zweig. Öffentliche UUIDs und Browser-Metadaten bleiben stabil, interne Pins wurden entfernt. Die realen Aufbau-Screenshots wurden aktualisiert.
- FaceSwap Standard und Advanced nach erfolgreichem Praxistest von `BETA` auf `STABLE` gesetzt. Browser-Texte beschreiben die zentrale Modulaktivierung und die getrennten FaceSwap-Zweige für bis zu drei Zielpersonen nun anwenderorientierter.
- Detailer nach derselben Produktlogik getrennt: `CMK Flow · 30 Detailer` enthält einen Smart-Detailer mit direktem `IMAGE PROCEED`-Weg; `CMK Flow · 30 Detailer · Advanced` übernimmt den funktionalen Doppelzweig aus Flow v3 und führt beide `SEGS PROCEED` gemeinsam zusammen. Advanced besitzt eine eigene UUID und erscheint nur als Variante des Haupteintrags. Beide neu geschnittenen Fassungen bleiben bis zum Praxistest `BETA`.
- Detailer Standard und Advanced haben den Standalone-Praxistest bestanden und wurden als getestete User-Subgraphen übernommen. Standard verwendet einen Segment-Detailer; Advanced kombiniert Hand- und Personen-Detailer. Beide stehen nun auf `STABLE`, behalten ihre kanonischen UUIDs, enthalten keine internen Pins und besitzen reale, getrennte Aufbau-Screenshots.
- FaceProcess analog getrennt: `50 FaceProcess` besitzt einen Execute-Zweig mit direktem `IMAGE PROCEED`; `50 FaceProcess · Advanced` besitzt zwei getrennt ausgewählte Gesichts-Zweige mit gemeinsamer `SEGS PROCESSED`-Zusammenführung und eigener UUID. Beide beginnen bis zum Standalone-Test als `BETA`. Die Modulnummern wurden als verbindlicher Bestandteil von Namen und Suche dokumentiert.
- Die getesteten User-Fassungen von `50 FaceProcess` und `50 FaceProcess · Advanced` übernommen. Advanced wurde kongruent zu `40 FaceSwap · Advanced` auf drei Zweige für `Leftmost`, `Center` und `Rightmost` erweitert; der mittlere Zweig startet deaktiviert. Beide Varianten haben ihre Standalone-Tests bestanden, stehen auf `STABLE`, behalten ihre kanonischen UUIDs und besitzen reale Aufbau-Screenshots.
- Der neue Standard-Aufbau-Screenshot von `50 FaceProcess` verwendet einen eindeutigen Asset-Namen, damit ComfyUI nicht die frühere Grafik aus dem Browsercache anzeigt.
- Auch der neue Standard-Aufbau-Screenshot von `30 Detailer` verwendet einen eindeutigen Asset-Namen und kann dadurch nicht mehr mit der früheren Grafik aus dem Browsercache verwechselt werden.

## 2026-07-17 — ControlNet-Bildauswahl

- `CMK Flow · 05 ControlNet (optional)` verwendet für `REFERENCE IMAGE` jetzt eine Dateiliste plus Upload-Schaltfläche statt eines freien Feldes für absolute Pfade. Die native Rohbild-Vorschau bleibt bewusst deaktiviert, damit die Vorschaufläche ausschließlich das aufbereitete ControlNet-Bild zeigt.
- Die Upload-Schaltfläche wird vor dem Wiederherstellen gespeicherter Werte angelegt. Ihr leerer Speicherplatz wird beim Speichern ans Ende der Werteliste verschoben und beim Laden kontrolliert an seine sichtbare Position zurückgesetzt. Bereits durch mehrfache Verschiebungen beschädigte ControlNet-Wertelisten werden anhand ihrer Werttypen und des Preprocessor-Namens wiederhergestellt.
- Neue Instanzen von `CMK Flow · 05 ControlNet (optional)` verwenden `Reference Image` als Standard für `IMAGE SOURCE`.

## 2026-07-17 — Create Image ohne Inpainting

- `CMK Flow · 01 START HERE · Create Image` führt `IMAGE`, `MASK` und `FILENAME STRING` als optionale Anschlüsse. Bei deaktiviertem `INPAINT_MODE` kann der Flow damit ohne vorgeschalteten Bild-Loader beginnen; bei aktiviertem Inpainting prüft die Node weiterhin, dass alle drei Werte vorhanden sind.
- `CMK Flow · Load Image` stellt seine Ausgänge in der Reihenfolge `PROCESS`, `IMAGE`, `MASK`, `FILENAME_STRING`, `LOG` bereit. Dateiname, Bild und Maske bleiben zugleich konsistent in `PROCESS` beziehungsweise `LOG` hinterlegt.

## 2026-07-17 — Stabiles Einfügen aus dem Flow Browser

- Der Flow Browser übergibt ComfyUI bei jeder Moduleinfügung eine neue reine JSON-Kopie der Blueprint-Daten. Interne `Map`- und Proxy-Objekte einer vorherigen Einfügung können dadurch keine späteren Einfügungen mehr blockieren.
- `CMK FaceSwap Image -Pipe-` reicht weder `PROCESS` noch das eingehende `LOG` durch. Die Execute-Node liefert nur noch `IMAGE PROCEED`, `SEGS PROCESSED`, `LOG BLOCK` und `diagnostic`; Flow-Weitergabe und Log-Zusammenführung liegen außerhalb der Node. Aktive Subgraphs, Flow-v3-Beispiele und die Referenzdefinition wurden auf den neuen Vertrag migriert.

## 2026-07-14 — Video FaceSwap v1.2

- `CMK Merge Video Segments` und `CMK Video Preview` durch `CMK Merge and Save Video` ersetzt.
- Player bleibt unterhalb der Parameter.
- Optionales Save/Publish kopiert das persistente Merge-Ergebnis ohne erneutes Encoding.
- `Randomize` und `Last Segment` erzeugen lokale, zusammenhängende Segmentindizes ab `0`; der ursprüngliche Index bleibt als `source_index` erhalten.

## Video inline player rebuild

- `CMK Split Video into Segments`, `CMK Merge Video Segments` und `CMK Video Preview` verwenden einen einzelnen schlanken HTML5-Player unterhalb ihrer nativen Parameter.
- Keine Parameterprojektion, keine Widget-Konvertierung und keine versteckten Pflicht-Widgets.
- Die erfolglose legacy-`gifs`/native-Preview-Ausgabe wurde durch einen expliziten `cmk_video_player`-UI-Kanal ersetzt.
- `CMK Video Compare` bleibt unverändert.

- Video-Frontend vollständig neu auf sauberem Stand aufgebaut: keine Split-/Merge-Parameterprojektion, keine Hidden-/Resize-/Configure-Hooks.
- `CMK Split Video into Segments` und `CMK Merge Video Segments` geben ihre Videos über ComfyUIs native Video-Preview aus.
- Split-Quelle: einzelnes serialisierbares `VIDEO`-Widget mit Auswahl und Upload nach `input/video/`; alle übrigen Parameter bleiben native Widgets.
- `CMK Video Compare`: FPS-relative Toleranz PASS bis 2 Frames, WARNING bis 6 Frames; synchrone Doppelplayer bleiben erhalten.
- `CMK Video Preview` als optionale Publish-/Copy-Node ohne Re-Encoding ergänzt.

- Neu: `CMK Video Compare` validiert den Roundtrip Original → Split → Merge objektiv anhand von Auflösung, FPS, Dauer, Framezahl und Audiozeit. Ausgabe: `CMK_VIDEO_METRICS`, `LOG`, `diagnostic` mit `PASS / WARNING / FAIL` und vollständiger Segment-/Trim-Timeline.

- Neu: `CMK Merge Video Segments` mit overlap-sicherem Roundtrip, persistentem `CMK_VIDEO`, Ausgabe nach `output/video/merged/<video_name>/` und Manifest-Reuse.
- `CMK Split Video into Segments`: Manifest-Reuse gehärtet. Vorhandene Segmente werden nur noch bei identischer Quelle, identischen Einstellungen, passendem Schema, konsistenter Segmentliste, gültigen Zeitbereichen und vorhandenen nichtleeren Dateien wiederverwendet.
- Cache-Status ist jetzt eindeutig `REUSED`, `SPLIT` oder `INVALIDATED` und wird mit Begründung in `SEGMENTS`, `LOG` und `diagnostic` ausgegeben.

## 2026-07-14 — `CMK Split Video into Segments`

- Neue eigenständige Video-Segmentierungsnode mit Videoauswahl aus `input/video/` statt manueller absoluter Pfadeingabe.
- Ausgabe wird dauerhaft unter `output/video/segments/<video_name>/` gespeichert; Segmente verschiedener Quellen werden nicht vermischt.
- Öffentliche Ausgänge: `SEGMENTS` (`CMK_VIDEO_SEGMENTS`), `LOG`, `diagnostic`; der frühere reine Verzeichnisausgang entfällt.
- `CMK_VIDEO_SEGMENTS` enthält Quelle, Ausgabeordner, Manifest, sortierte Segmentpfade, Zeitbereiche, Videoeigenschaften sowie effektive Segmentierungs- und Encodingparameter.
- Segmentlänge wird weiterhin aus `MAX FRAMES 720P` beziehungsweise `MAX FRAMES 1080P` und der Quell-FPS berechnet; mindestens 180 Frames und mindestens 3 Sekunden werden eingehalten.
- Direkte FFmpeg-Verarbeitung vermeidet das Laden des vollständigen Videos in den Arbeitsspeicher. `ffprobe` wird verwendet, wenn vorhanden; ComfyUI-Desktop-Installationen ohne separates `ffprobe` nutzen automatisch den Metadaten-Fallback von `imageio_ffmpeg`.
- Exaktes Manifest-Reuse verhindert unnötiges erneutes Splitten nach einem ComfyUI-Neustart.
- Audio bleibt intern fest auf AAC/192k; Standard-UI: Video, Framegrenzen, Overlap, Videocodec, Bitrate und Preset.

## 2026-07-14 — FaceSwap als einschleifbares CMK-Modul

- Neuer persistenter `CMKFaceSwapBoundaryCache` mit sichtbarem Namen `CMK Boundary Cache`.
- Der Boundary ist die einzige Quelle für die öffentlichen Modulrollen `MODEL`, `PROCESS`, `IMAGE` und `LOG`; die interne Preview liegt ebenfalls dahinter.
- `PROCESS`, `IMAGE` und `LOG` werden gemeinsam persistent materialisiert; `MODEL` bleibt read-only und wird nicht serialisiert.
- `CMK FaceSwap Image -Pipe-` behandelt `IMAGE_SOURCE` als Lazy-Eingang. Bei `ENABLE = OFF` werden interner Source-Loader, Face Detection und Swap vollständig übersprungen; `IMAGE_TARGET` läuft unverändert durch.
- Referenz-Subgraph `SwapFace`: eingehendes Workflowbild als Target, interner `CMK Load Image -Pipe-` ausschließlich als Source-Loader.

- `crop_factor` default für `CMK FaceSwap Image -Pipe-` auf `1.5` gesetzt.
- `bbox_dilation`, `crop_factor`, `drop_size` und `feather` auf die offene `CMK FaceSwap Image` übertragen; bestehende Bild-, Auswahl-, `opt_selected_face`- und Ausgangsschnittstellen bleiben erhalten.

- Neu: `CMK FaceSwap Image -Pipe-` als geschlossenes FaceSwap-Modul mit Eingängen `PROCESS` / `IMAGE_TARGET` / `IMAGE_SOURCE` / `LOG` und Ausgängen `PROCESS` / `IMAGE` / `LOG` / `diagnostic`.
- Advanced für `CMK FaceSwap Image -Pipe-`: `BBOX DILATION`, `CROP FACTOR`, `DROP SIZE`, `feather`.
- FaceSwap-Pasteback erweitert: `bbox_dilation` und `crop_factor` beeinflussen die finale Swap-Maske, um abgeschnittene Stirn-/Kinnbereiche besser abzufangen.

## 2026-07-13 — SideKick `CMK Swap Image Loader -Pipe-` ergänzt

- Neue geführte Dual-Loader-Node als Bildeinstieg für das kommende FaceSwap-Image-Modul.
- Target und Source werden in einer gemeinsamen zweispaltigen Oberfläche nebeneinander ausgewählt, hochgeladen und vorangezeigt.
- Das Target übernimmt vollständig die Funktion von `CMK Image Load and Resize -Pipe-`: Auflösungspreset, Dimensionswechsel, Resize-Methode sowie optionaler Advanced-Crop mit `center/top/bottom/left/right`.
- Die Source wird EXIF-korrigiert geladen, aber weder beschnitten noch skaliert.
- Öffentliche Ausgänge: `PROCESS`, `IMAGE TARGET`, `IMAGE SOURCE`, `LOG`, `diagnostic`.
- `PROCESS` enthält ausschließlich Metadaten; beide Pixelinformationen bleiben in getrennten authoritative IMAGE-Ausgängen.
- `LOG` und Diagnostic dokumentieren beziehungsweise zeigen Target und Source getrennt.
- Dateihash-Invalidierung berücksichtigt beide Bilddateien.

## 2026-07-13 — Advanced-Crop für `CMK Image Load and Resize -Pipe-`

- Das bisher sichtbare UI bleibt vollständig STANDARD: `IMAGE`, `RESOLUTION`, `SWAP DIMENSIONS`, `RESIZE METHOD`.
- Advanced ergänzt `CROP` und `CROP POSITION` mit `center`, `top`, `bottom`, `left`, `right`.
- Bei aktivem Crop wird das Quellbild vor dem Resize ohne Seitenverzerrung auf das Seitenverhältnis der gewählten Zielauflösung beschnitten.
- Die Position verankert den Ausschnitt auf der tatsächlich beschnittenen Achse; nicht anwendbare Richtungen werden auf dieser Achse zentriert.
- `PROCESS`, `LOG` und `diagnostic` dokumentieren Crop-Status, Position, Größe und Crop-Box; `IMAGE` bleibt alleiniger Pixeltransport.
- Bei deaktiviertem Crop bleibt das bisherige Lade- und Resize-Verhalten unverändert.

## FaceProcess Dynamic Mode UI v6

- Restore and Detailer now project physically separate visible widget lists.
- The complete canonical widget order remains internal for workflow serialization and configuration.
- First mode switch after loading can no longer shift values between modes.
- Works identically for standalone nodes and nodes inside ComfyUI subgraphs.

## 2026-07-13 — FaceProcess-Moduswechsel über onWidgetChanged und Subgraph-Watcher korrigiert

- `CMK FaceProcess -Pipe-` reagiert jetzt über ComfyUIs kanonisches `onWidgetChanged` unmittelbar auf Änderungen von `PROCESS MODE`.
- Ein leichter instanzlokaler Wächter synchronisiert die Modusdarstellung zusätzlich unabhängig von Draw-Hooks und Widget-Callback-Ersetzungen innerhalb von Subgraphs.
- Im Modus `restore` sind ausschließlich gemeinsame und Restore-Parameter sichtbar; im Modus `detailer` ausschließlich gemeinsame und Detailer-Parameter.
- Alle Widgets verbleiben in kanonischer Reihenfolge, sodass beide Parametersätze positionsstabil gespeichert werden.
- Die UI-Erweiterung besitzt einen neuen Modulpfad `cmk_faceprocess_pipe_dynamic_v4.js`, damit eine zuvor gecachte JavaScript-Datei den Fix nicht verdeckt.

# CMK Changelog

## 2026-07-13 — FaceProcess-Modus-UI reaktiv und positionsstabil korrigiert

- `CMK FaceProcess -Pipe-` zeigt weiterhin ausschließlich die gemeinsamen Parameter und den Parametersatz der aktiven Betriebsart `restore` beziehungsweise `detailer`.
- Der Moduswechsel wird nicht mehr nur über einen beim Laden ersetzbaren Widget-Callback erkannt, sondern zusätzlich über eine leichte Zustandsprüfung der Node während der Darstellung.
- Nach Node-Erzeugung, Workflow-Laden und ComfyUI-Neustart reagiert die sichtbare Parametergruppe deshalb unmittelbar auf jeden Wechsel von `PROCESS MODE`.
- Widget-Referenzen werden nach Frontend-Konfiguration stets auf die tatsächlich aktiven Widget-Objekte aktualisiert; veraltete Vor-Konfigurationsreferenzen können die Umschaltung nicht mehr blockieren.
- Inaktive Modusparameter bleiben unsichtbar, ihre Werte verbleiben jedoch in kanonischer Serialisierungsreihenfolge und werden beim späteren Zurückschalten wiederhergestellt.
- Backend, Ports, Kabel, Caches und Referenzworkflow bleiben unverändert.

## 2026-07-13 — Räumlicher Branch-Merge und FaceProcess-UI korrigiert

- Die zuvor verwendete Pixel-Differenzmaske wurde vollständig aus dem Branch-Merge entfernt. Sie konnte Quell- und Ergebnisbild pixelweise ineinander verschachteln und dadurch das sichtbare Salz-und-Pfeffer-Muster erzeugen.
- Cache-stabile Detailer-/FaceProcess-Artefakte besitzen jetzt eine räumliche Support-Maske aus den tatsächlich zugeordneten SEG-Crop-Regionen.
- `CMK SEGS CONCAT` übernimmt das vollständig komponierte Branch-Bild ausschließlich innerhalb dieses räumlichen Supports; bei echten Überschneidungen gewinnt weiterhin der spätere Eingang.
- `CMK FaceProcess -Pipe-` gibt als `SEGS PROCESSED` nur noch die ausgewählten und tatsächlich dem Branch zugeordneten Gesichter aus. Unveränderte Geschwistergesichter sind kein Bestandteil des Branch-Ergebnisses mehr.
- Detailer- und FaceProcess-Branch-Caches wurden auf Schema v4 erhöht; ältere Artefakte werden einmalig neu berechnet.
- Die dynamische FaceProcess-UI entfernt oder sortiert Widgets nicht mehr. Inaktive Betriebsart-Widgets bleiben in kanonischer Serialisierungsreihenfolge erhalten und werden ausschließlich optisch ausgeblendet.
- Eindeutig verschobene Altwerte wie `512` als Restore-Modell, `true` als Face-Detector oder `768` als Visibility werden beim Laden validiert und zurückgesetzt.
- Öffentliche Ports, Kabel, Node-Instanzen und Referenzworkflow bleiben unverändert.

## 2026-07-13 — Cache-stabile SEGS-Branch-Artefakte ergänzt

- Smart-Detailer- und FaceProcess-Branch-Caches verlassen sich beim späteren Merge nicht mehr auf erneut deserialisierte `cropped_image`-Felder einzelner Impact-SEGS.
- Jeder Branch speichert zusätzlich das vollständig zusammengesetzte Branch-Bild und seine effektive Pixelmaske als CPU-normalisiertes internes Artefakt.
- `CMK SEGS CONCAT` übernimmt dieses Artefakt bei frischer Ausführung und Cache-Hit über denselben Codepfad; beide Ergebnisse sind dadurch pixelidentisch.
- Der SEGS-Wert bleibt Impact-kompatibel und behält dieselben öffentlichen Ports.
- Branch-Cache-Schemata wurden auf v3 erhöht; vorhandene v2-Einträge werden einmalig neu berechnet.
- Kabel, Node-Instanzen, Moduloberflächen und Referenzworkflow bleiben unverändert.

Historische Einträge beschreiben den Entwicklungsweg. Sie definieren **nicht** die aktuelle API. Der verbindliche aktuelle Stand steht ausschließlich in `ARCHITECTURE.md`.

## 2026-07-13 — SideKick `CMK Image Load and Resize -Pipe-` ergänzt

- Neue kompakte Loader-Node für standalone betriebene, direkt pixelbasierte CMK-Module.
- Kombiniert Bildauswahl und Resize ohne die Generierungsfunktionen von `CMK Pipe Create Image -Pipe-`.
- Öffentliche Eingänge: `IMAGE`, `RESOLUTION`, `SWAP DIMENSIONS`, `RESIZE METHOD`.
- Öffentliche Ausgänge: `PROCESS`, `IMAGE`, `LOG`, `diagnostic`.
- `IMAGE` bleibt die einzige authoritative Pixelinformation; `PROCESS` enthält ausschließlich Quell-/Zielauflösung und Dateimetadaten.
- Keine Maske, keine Prompts, keine LoRAs, kein Inpaint, kein Outpaint und keine Latent-Erzeugung.

## 2026-07-13 — Branch-sichere SEGS-Zusammenführung korrigiert

- `CMK SEGS CONCAT` setzt parallele SEGS-Sammlungen nicht mehr seriell auf das bereits veränderte Geschwisterergebnis.
- Jede SEGS-Sammlung wird unabhängig auf dasselbe authoritative Eingangsbild angewendet.
- Nur das tatsächliche Pixel-Delta eines Branches wird in das kumulierte Ergebnis übernommen.
- Unveränderte Segmente späterer Detailer-/FaceProcess-Branches können frühere Bearbeitungen dadurch nicht mehr auf das Ausgangsbild zurücksetzen.
- Bei echten räumlichen Überschneidungen gilt weiterhin eine eindeutige Priorität: der spätere SEGS-Eingang gewinnt.
- Öffentliche Ports, bestehende Kabel, Node-Instanzen und Referenzworkflow bleiben unverändert.

## 2026-07-13 — Boundary-/Concat-Synchronisation korrigiert

- Persistente Smart-Detailer- und FaceProcess-Branch-Caches erhalten atomare Revisionsmarker.
- Detailer- und FaceProcess-Boundaries speichern ein internes Dependency-Manifest der tatsächlich zusammengeführten Branch-Revisionen.
- Ein neu materialisierter Einzelzweig invalidiert jetzt zuverlässig einen älteren Module-Boundary-Hit, auch wenn sein statischer Prompt-Fingerprint unverändert ist.
- `SEGS CONCAT` und `LOG CONCAT` werden vor der erneuten Boundary-Materialisierung zwingend ausgewertet; öffentliche Bilder können dadurch kein veraltetes Einzelzweigergebnis mehr liefern.
- Branch- und Module-Boundary-Fingerprints sind instanzgebunden, damit parallele formal identische Nodes keine persistenten Cache-Einträge teilen.
- Öffentliche Ports, bestehende Kabel und Moduloberflächen bleiben unverändert.

## 2026-07-13 — Interface Contract 1.0 und vollständige Kaskadenentkopplung

- `ARCHITECTURE.md` als einzige verbindliche Quelle für den aktuellen Schnittstellenvertrag neu konsolidiert.
- Vier öffentliche Transportrollen endgültig festgelegt: `MODEL`, `PROCESS`, `IMAGE`, `LOG`.
- `SAMPLED`, `SAMPLER`, `REFINER`, `DETAILER` und `FACE` als proprietäre, geschützte Modulübergaben dokumentiert.
- `CMK Smart Detailer -Pipe-` gibt weder `DETAILER`, vollständiges `LOG` noch unverändertes `IMAGE` weiter.
- `CMK FaceProcess -Pipe-` gibt weder `FACE`, vollständiges `LOG` noch unverändertes `IMAGE` weiter.
- `CMK_LOG_BLOCK` als dedizierter Typ eingeführt; Zusammenführung ausschließlich über `CMK LOG CONCAT`.
- Smart-Detailer- und FaceProcess-Instanzen parallelisiert.
- Persistente Branch-Caches für jede teure Detailer-/FaceProcess-Instanz ergänzt.
- Verbindliche Modul-Boundaries für Refiner, Detailer und FaceProcess ergänzt.
- Interne Comparer und öffentliche Modul-Ausgänge ausschließlich hinter den Boundaries angeschlossen.
- Refiner-Boundary speichert First-Pass- und Refiner-Bild.
- Dynamische Eingänge für `SEGS`, `LOG BLOCK` und Preview-Board-Diagnostics ergänzt.
- Sichtbare Bezeichnungen vereinheitlicht: `CMK SEGS CONCAT`, `CMK LOG CONCAT`, `CMK Boundary Cache`.
- Falsche Refiner-Modulbezeichnung mit `SmartUpscale` korrigiert.
- Überflüssige MODEL-/PROCESS-Forward-Nodes aus dem Detailer-Referenzmodul entfernt.
- Dokumentationsaudit durchgeführt:
  - widersprüchlicher Drei-Pipe-/Vier-Leitungs-Stand bereinigt;
  - überholte serielle Detailer-/FaceProcess-Schnittstellen entfernt;
  - alte IMAGE-/LOG-/Arbeits-Pipe-Passthrough-Verträge entfernt;
  - mehrfach wiederholte ControlNet-Zwischenstände aus README und Design Guidelines entfernt;
  - doppelte Changelog-Überschriften und konkurrierende Zwischenverträge zusammengeführt.

## 2026-07-12 — Refiner-Boundary, Inpaint und ControlNet stabilisiert

- Refiner-Ergebnisse im ComfyUI-Tempbereich materialisiert.
- Direkten Refiner-zu-Comparer-Bypass entfernt.
- Refiner-Boundary auf First-Pass- und Refiner-Bild erweitert.
- Fooocus-Inpaint korrigiert: neutraler Concat-Latent für den Patch, Original-Latent plus Noise-Mask für den Sampler.
- Doppelte LoRA-Anwendung zentral im LoRA-Text-Loader dedupliziert.
- ControlNet-Stärke, optionale Hint-Invertierung und stabile Vorschau-/UI-Regeln umgesetzt.
- Regressionstests für freie Generierung, ControlNet und Inpaint bestanden.

## 2026-07-11 — Getrennte Transportrollen und modulare Prepare-/Execute-Grenzen

- `MODEL` als read-only Modellressourcenleitung eingeführt.
- `PROCESS`, `IMAGE` und `LOG` als getrennte öffentliche Rollen migriert.
- Sampler-, Refiner-, Detailer- und FaceProcess-Prepare-Nodes auf proprietäre Arbeits-Pipes umgestellt.
- `CMK KSampler -Pipe-` auf `SAMPLER → SAMPLED` reduziert.
- Refiner auf Latent-Übergabe über `SAMPLED` migriert.
- Detailer und FaceProcess mit globalem und lokalem Enable ausgestattet.
- Standard-/Advanced-UI und konsistente große Standardlabels eingeführt.
- Save Project Image auf direkten `LOG`-Eingang und Fullpath-Ausgabe migriert.

## 2026-07-10 — SDXL Prepare-/Execute-Migration

- SDXL-Sampler-Prepare mit Conditioning, LoRA, PAG, Sampling, ZSNR, FreeU und Inpaint-Pfaden konsolidiert.
- Refiner-Prepare und Refiner-Execute als getrennte Verantwortlichkeiten eingeführt.
- First-Pass- und Refiner-Bild als getrennte Refiner-Ergebnisse verfügbar gemacht.
- Feste Seed-Policy für reproduzierbare Tests etabliert.

## 2026-07-09 und früher — Aufbauphase

Die Aufbauphase umfasste unter anderem:

- modulare Pipe-Create/Peek/Set/Get-Nodes;
- Face- und Detailer-Grundfunktionen;
- ControlNet-Prepare-Iterationen;
- Loader-, LoRA- und Pipe-Inspect-Werkzeuge;
- Diagnostic-, Preview- und Summary-Infrastruktur;
- Verzeichnis- und Mapping-Refactoring;
- Entfernung öffentlicher Legacy-Aliase.

Diese Zwischenstände wurden bewusst nicht als parallele aktuelle Verträge beibehalten. Einzelne damalige Socketfolgen, Passthrough-Regeln und Node-Versionen gelten als historisch und wurden durch den Vertrag 1.0 ersetzt.

## 2026-07-13 — FaceProcess dynamic UI v5

- Corrected mode-dependent visibility for ComfyUI Vue Nodes 2.0.
- Visibility is now written to both legacy `widget.hidden` and reactive `widget.options.hidden` / WidgetValueStore metadata.
- Full canonical widget order remains intact for workflow serialization and backend execution.
- Added `CMK FaceSwap Video Loader` with a bottom-aligned dual media panel; restored the universal Split node to its compact interface.
- Centralized FaceSwap enhancer validation and replaced the seam-prone changed-pixel pasteback with the exact INSwapper affine/native-style soft-mask path for Image, Image -Pipe-, and Video.
- Fixed GPEN/CodeFormer aligned-crop geometry: enhancer outputs now return to the original INSwapper crop size before shared pasteback.
- Added a top-level `FACE SWAP` OFF/ON toggle to `CMK FaceSwap Video`; OFF cleanly bypasses processing and forwards the original `CMK_VIDEO_SEGMENTS` context.
- Excluded affine out-of-frame padding from the shared FaceSwap pasteback mask, removing rotated black crop borders on tilted or edge-near faces.
- Limited FaceSwap `crop_factor` to 1.0–3.0 in Image, Image -Pipe-, and Video, with central runtime clamping for older workflows.
- Limited all CMK Detailer denoise controls and runtime paths to a maximum of 0.5, including Smart Detailer and FaceProcess Detailer.
- Documented as binding architecture and UI policy why CMK exposes task-specific practical parameter ranges instead of raw backend maxima; explicitly covers FaceSwap `crop_factor` 1.0–3.0 and Detailer `denoise` max 0.5, including central runtime clamping and backward handling of older workflow values.
- Extended `CMK FaceSwap Image -Pipe-` with `GLOBAL ENABLE` and a cache-stable `SEGS PROCESSED` output based on the exact shared pasteback mask for parallel `CMK SEGS CONCAT` composition.
- Changed `CMK FaceSwap Image -Pipe-` from a pass-through `LOG` output to an independent `LOG BLOCK` (`CMK_LOG_BLOCK`) for explicit parallel collection through `CMK LOG CONCAT`.
- Added a true global early-bypass to `CMK FaceProcess Prepare -Pipe-`: global OFF no longer requests MODEL/PROCESS or initializes SAM, LoRA, CLIP and conditioning.
- Fixed disabled Detailer/FaceProcess branches: true early bypass before SAM/detection/cache, no full-image disabled branch payloads, and no boundary persistence for incomplete disabled dependencies.
- Removed the global `sys.modules` scan from the ControlNet AIO resolver; it now uses ComfyUI's node registry and known ControlNet Aux modules only, with a cached result.

- Added temporary timestamped execution tracing for `INPUT_TYPES`, `IS_CHANGED`, `check_lazy_status`, and each registered CMK node's execution method to locate the post-prompt stall.
- Fixed the FaceProcess module boundary stall: fully disabled modules now bypass fingerprinting and persistent cache I/O, while the boundary key no longer recursively canonicalizes IMAGE/LOG upstream graphs.
- Added an explicit `None` option to the local LoRA dropdowns in `CMK Sampler Prepare SDXL -Pipe-` and `CMK Refiner Prepare SDXL -Pipe-`; the Refiner now defaults to `None` to prevent forced incompatible LoRA application and shape-mismatch errors.
- Added `None` as the default local LoRA selection in `CMK Detailer Prepare -Pipe-` and added an explicit console marker confirming that global OFF skips MODEL, PROCESS, SAM, LoRA, CLIP and conditioning preparation.

## v3.7 Diagnostic Documentation

- Documented the confirmed module-boundary architecture.
- Defined global OFF as a hard functional bypass.
- Documented the FaceProcess boundary cache bypass and exclusion of `IMAGE`
  and `LOG` from recursive boundary identity fingerprinting.
- Documented that disabled modules must skip MODEL, PROCESS, LoRA, CLIP,
  Conditioning, SAM and persistent cache operations.
- Defined module ordering as arbitrary and optional.
- Documented `None` as the neutral local LoRA selection for Refiner and
  Detailer.
- Documented model-specific LoRA compatibility and the distinction between
  SDXL Base and SDXL Refiner architectures.
- Recorded the confirmed clean reference configurations for central
  `lora_syntax`, Refiner and Detailer.
- Kept timestamped diagnostic tracing enabled for continued boundary
  verification.

## v3.8 Diagnostic

- Memoized node fingerprints for the lifetime of one immutable expanded prompt.
- Reused the active FaceProcess boundary analysis between lazy evaluation and
  execution.
- Removed the redundant third FaceProcess branch-manifest calculation before
  storing a boundary miss.
- Bumped the FaceProcess boundary schema to v4.
- Preserved all existing cache keys, branch revision checks and functional
  FaceProcess outputs.

## v3.8.1 Diagnostic Hotfix

- Fixed an accidental recursive call inside
  `CMKFaceBoundaryCache._analysis()`.
- `_analysis()` now correctly computes `_cache_key()` and `_dependencies()`
  once, then reuses that result between lazy evaluation and execution.
- No functional FaceProcess, cache-key or invalidation behavior was changed.

## v3.9 Diagnostic – FaceSwap Compositing Phase 1

- Split FaceSwap paste-back into an eroded identity `SWAP MASK` and a separately
  dilated, Gaussian-blurred `BLEND MASK`.
- Added adaptive blend width based on detected face extent when `feather = 0`.
- Preserved hair, ears, jewellery and background as target content.
- Kept all existing UI controls and defaults unchanged.
- Applied the shared improvement to Image and Video FaceSwap.
- Bumped FaceSwap boundary schema to v2 and video engine version to 5.

## v3.9.1 Diagnostic – Conservative GPEN

- Reduced GPEN influence from a fixed 82% to an edge-adaptive 12–42%.
- Added protection for existing high-contrast structures such as glasses,
  eyebrows, eye contours and teeth.
- Added a high-frequency detail limiter relative to the original swap patch.
- Changed GPEN downsampling from Lanczos to `INTER_AREA`.
- Bumped FaceSwap boundary schema to v3 and video engine version to 6.
## 2026-07-18 — Kuratierte Showcase-Workflows

- Sechs aktualisierte Flow-Subgraphen für KSampler, Refiner, Detailer, FaceSwap, FaceProcess sowie Upscale & Save wurden übernommen; ihre bestehenden Browser-Metadaten bleiben erhalten.
- Acht vollständige Showcase-Workflows besitzen nun eigenständige, zweisprachige Metadaten mit einer kurzen Funktionsbeschreibung und einem separaten Hinweis auf die jeweils gezeigte CMK-Philosophie.
- Neun aktuelle Screenshots wurden zugeordnet. `CMK Full Flow` zeigt Text2Image und Inpainting als getrennte Vorschauansichten.
- Die Registerkarte `Referenzen` veröffentlicht nur Workflows mit freigegebener Metadatendatei und zeigt Beschreibung, CMK-Besonderheit und Vorschau direkt im Browser.
## 2026-07-18 — Diagnosewerkzeuge im Baukasten

- Die temporäre globale Ausführungszeitmessung wurde vollständig aus der Node-Registrierung entfernt; CMK-Ausführungen schreiben keine `[CMK TRACE]`-Laufzeitzeilen mehr in die Konsole.
- `CMK Preview Render`, `CMK Preview Board` und `CMK Summary` erhalten im Baukasten eine hervorgehobene Beschreibung ihrer Diagnoseleistung sowie jeweils einen realen Screenshot.

## 2026-07-18 — Veröffentlichung bereinigt

- Die überholten Workflow-Exporte `CMK Flow`, `CMK Flow v3.0` und die beiden Auslieferungsfassungen von `CMK Flow v4.5` wurden aus dem Veröffentlichungspaket entfernt.
- Der dokumentierte Einstieg erfolgt nun über die aktuellen Subgraphen und die kuratierten Showcase-Workflows im Flow Browser.
- Persönliche Testpfade wurden aus dem Video-Segmentierungs-Subgraphen entfernt.
- Das Paket wird als freie Software unter `GPL-3.0-or-later` veröffentlicht; Lizenztext, Rechteinhaber und die wesentlichen Bedingungen sind im Paket dokumentiert.
- Das About-Fenster des Flow Browsers zeigt Copyright und Lizenz nun unmittelbar in deutscher und englischer Sprache an.
- Das About-Fenster enthält eine zweisprachige Verantwortungserklärung zu privater Selbstbestimmung, Einwilligung und der unzulässigen Veröffentlichung manipulierter Darstellungen realer Personen.
- Vier macOS-Metadatendateien (`.DS_Store`) wurden entfernt und für künftige Veröffentlichungen ausgeschlossen.
- Der sichtbare CMK-Flow-Button verwendet ComfyUIs aktuelle `comfyAPI` und importiert nicht länger die veraltete Legacy-Komponente `scripts/ui/components/button.js`.
