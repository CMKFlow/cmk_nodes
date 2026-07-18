# CMK Subgraph Audit

**Stand:** 2026-07-16  
**Historischer Bezugsstand:** inzwischen aus der Veröffentlichung entfernter Export `CMK Flow v4.5`

Dieses Inventar trennt nachgewiesene Nutzung von vermuteter historischer Bedeutung. Ein Status `prüfen` bedeutet deshalb nicht `löschen`, sondern: Der Subgraph ist durch keinen aktuellen Referenz- oder Beispielworkflow als erforderlich belegt und benötigt vor einer Migration eine fachliche Entscheidung oder einen Funktionstest.

## Ergebnisübersicht

| Klasse | Anzahl | Bedeutung |
|---|---:|---|
| Flow-Kern | 8 | sieben öffentliche Referenzmodule plus eigenständige Flow-Oberfläche für FaceSwap Image |
| Prüfen/Migration | 18 | nicht durch aktuelle Referenz- oder Beispielworkflows direkt verwendet |

Es wurden keine byte-identischen Subgraph-Dubletten gefunden. Ähnliche Namen stehen für unterschiedliche Graphdefinitionen.

## Kern des aktuellen Referenzworkflows

| Subgraph | UUID | Rolle | Empfehlung |
|---|---|---|---|
| `CMK Flow · 02 LoRA Stack` | `b9fa321e-d7fb-48ee-8f06-0a74b492ce96` | Prompt- und LoRA-Stack des Flow-Frontends | eigenständig versioniert behalten |
| `CMK Flow · 10 KSampler 1st Pass` | `0cfdb7a5-24a1-47a1-a070-b1e778474284` | erster Sampling-Pass | unverändert behalten |
| `CMK Flow · 20 Refiner` | `2565ad34-d9fd-4be0-94e0-5f24b17879ea` | Refiner-Pass | unverändert behalten |
| `CMK Flow · 30 Detailer` | `14a5af49-a274-4a82-8c0b-fe4f49721eed` | paralleles Detailer-Modul | unverändert behalten |
| `CMK Flow · 40 FaceSwap` | `c33298e6-bd6f-4e82-abcd-a73d51178b90` | eingeschleiftes FaceSwap-Modul | unverändert behalten |
| `CMK Toolbox · FaceSwap Image` | `de0cadaa-2b43-4c6a-91dc-95e793d104b5` | eigenständiger FaceSwap-Baustein; nicht Teil der geführten Flow-Reihenfolge | als vollständig funktionsfähigen Toolbox-Baustein behalten; spätere Anpassungen separat planen |
| `CMK Flow · 50 FaceProcess` | `3638560f-0975-4df3-8a3b-1d22af32b336` | paralleles FaceProcess-Modul | unverändert behalten |
| `CMK Flow · 90 Upscale & Save` | `ee5e7e66-78b4-47ff-9e6f-b0b1c55fb039` | verbindlicher Abschluss des Flow-Hauptwegs (`Flow/Finish`) | unverändert behalten |

Der aktuelle Referenzworkflow verwendet für die Bild-/Prozessquelle direkt die Python-Node `CMKPipeCreateImage`. Der frühere Entwicklungs-Subgraph `CMK Pipe Create Image v2` wurde deshalb entfernt.

## Baukasten

`CMK Toolbox · FaceSwap Image` und die offene Node `CMK FaceSwap Image` ohne `-Pipe-` gehören zum Baukasten. Für die weitere Bereinigung benötigt der Baukasten ein eigenes Funktionsinventar, das Python-Nodes, Subgraphs und Beispielworkflows gemeinsam betrachtet.

## Prüf- und Migrationskandidaten

| Gruppe | Subgraph | Befund | Nächster sicherer Schritt |
|---|---|---|---|
| alte Pipe-Grenzen | `CMK - Pipe In` | nur indirekte Abhängigkeit des Inpaint-Pipe-In | zusammen mit Inpaint-Kette testen |
| alte Pipe-Grenzen | `CMK - Pipe In (Inpaint_1st-Pass)` | aktuell nicht referenziert | Inpaint-Funktionstest oder archivieren |
| alte Pipe-Grenzen | `CMK - Pipe In v2` | aktuell nicht referenziert | gegen heutige Prepare-Nodes abgleichen |
| alte Pipe-Grenzen | `CMK - Pipe In v2 - 2nd Pass` | aktuell nicht referenziert | gegen Refiner-Prepare abgleichen |
| alte Pipe-Grenzen | `CMK - Pipe Out` | aktuell nicht referenziert | Socket-Vertrag prüfen, dann archivieren oder als Legacy markieren |
| alte Pipe-Grenzen | `CMK - Pipe Out v2` | aktuell nicht referenziert | Socket-Vertrag prüfen, dann archivieren oder als Legacy markieren |
| alte Pipe-Grenzen | `CMK - Pipe Out - 2nd Pass` | aktuell nicht referenziert | gegen Refiner-Boundary abgleichen |
| Conditioning | `CMK - Conditioning Builder - SDXL -` | aktuell nicht referenziert | prüfen, ob vollständig von Prepare-Nodes ersetzt |
| Conditioning | `CMK - Conditioning Builder - SDXL-Refiner -` | aktuell nicht referenziert | prüfen, ob vollständig von Refiner Prepare ersetzt |
| Conditioning | `Model and Conditioning Switch` | aktuell nicht referenziert | als allgemeines Hilfsmodul oder Legacy klassifizieren |
| Inpaint | `CMK - => Kontext Reference Latent Mask (only Inpaint-Mode)` | aktuell nicht referenziert; externe Kontext-Node | separaten Inpaint-Workflow als Beleg erstellen oder archivieren |
| Inpaint | `CMK - Fooocus Inpaint Apply` | aktuell nicht referenziert; externe Fooocus-Nodes | separaten Inpaint-Workflow als Beleg erstellen oder archivieren |
| Inpaint | `CMK - Mask Finalize` | aktuell nicht referenziert | mit Inpaint-Kette gemeinsam testen |
| Inpaint | `CMK - Smart Image Prepare` | aktuell nicht referenziert | mit Inpaint-Kette gemeinsam testen |
| Inpaint | `CMK - VAE Encode Dummy-Switch` | aktuell nicht referenziert | prüfen, ob heutige Lazy-/Bypass-Logik Ersatz ist |
| Dateiausgabe | `CMK - Create Caption and Foldername` | aktuell nicht referenziert | als optionales Utility mit eigenem Beispiel belegen oder archivieren |
| Video | `CMK - Create Video, Caption and Foldername` | nur in einem archivierten Video-Workflow verwendet | archivieren, sofern aktueller Video-Save es ersetzt |
| Video | `CMK - Split Video into Segments` | aktueller Video-Referenzworkflow verwendet native CMK-Video-Nodes | gegen aktuelle Split-/Loader-Nodes abgleichen und wahrscheinlich archivieren |

## Abgeschlossene Entscheidungen

- `CMK Pipe Create Image v2` war ein Entwicklungsartefakt und wurde entfernt.
- Der alte `SwapFace` war die frühere Bezeichnung von `CMK Face Swap Image -Pipe-` und wurde entfernt. Damit ist die UUID-Kollision aufgelöst.
- Der bisher nur eingebettete LoRA-Stack wurde mit unveränderter UUID als `subgraphs/CMK Flow · 02 LoRA Stack.json` exportiert.

## Empfohlene Reihenfolge

1. Die acht Flow-Kernartefakte unverändert einfrieren; spätere Anpassungen am FaceSwap-Image-Subgraph als eigene Aufgabe behandeln.
2. Den Baukasten unabhängig vom Flow inventarisieren und für jede weiterhin gewünschte Funktion mindestens einen aktuellen Beispielworkflow pflegen.
3. Für Inpaint entscheiden, ob es weiterhin ein unterstützter Baukasten-Funktionsbereich ist. Falls ja, einen aktuellen Beispielworkflow erstellen und die fünf zusammengehörigen Hilfsgraphen gemeinsam testen.
4. Alte Pipe-In/Pipe-Out- und Conditioning-Graphen gegen die heutigen Prepare-/Boundary-Nodes vergleichen.
5. Video- und Dateinamengraphen gegen die heutigen nativen CMK-Video-/Save-Nodes vergleichen.
6. Erst danach Dateien physisch nach `flow`, `toolbox`, `internal` und `archive` verschieben oder UUID-Migrationen durchführen.

## Validierungsregeln für die Bereinigung

Nach jedem Migrationsschritt müssen mindestens geprüft werden:

- JSON-Syntax aller veränderten Dateien;
- eindeutige öffentliche Subgraph-UUIDs;
- vollständige eingebettete Definitionen in Referenzworkflows;
- unveränderte öffentliche Socket-Typen der Kernmodule;
- Laden des Workflows nach vollständigem ComfyUI-Neustart;
- Lazy-Bypass sowie Branch- und Boundary-Cache-Verhalten.
