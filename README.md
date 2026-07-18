<p align="center">
  <img src="web/assets/brand/cmk-logo.png" width="140" alt="CMK Flow logo">
</p>

# CMK Flow

Modulares Custom-Node-Paket für ComfyUI.

[Quellcode](https://github.com/CMKFlow/cmk_nodes) · [Dokumentation](https://github.com/CMKFlow/cmk_nodes#dokumente) · [Freiwillig unterstützen](https://paypal.me/CMKFlow)

Copyright (C) 2026 Carsten Kirschner

CMK Flow verfolgt zwei klar getrennte Konzepte:

```text
CMK **** -Pipe- → geschlossenes, geführtes Ökosystem
übrige Nodes    → offener Experimentierkasten
```

Der verbindliche Architektur- und Schnittstellenvertrag steht in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Einstieg

Der zentrale Einstieg ist der `CMK Flow Browser`. Er stellt die aktuellen Module,
den Baukasten und die kuratierten Showcase-Workflows direkt in ComfyUI bereit.

Äußerer Standardweg:

```text
Bild/Prozessquelle
    ↓
ControlNet optional
    ↓
KSampler 1st-Pass
    ↓
Refiner
    ↓
Detailer
    ↓
FaceProcess
    ↓
Upscale / Save
```

Die sichtbaren Hauptrollen sind:

```text
MODEL | PROCESS | IMAGE | LOG
```

Zwischen Sampler und Refiner wird statt `IMAGE` der proprietäre Latent-Übergabetyp `SAMPLED` verwendet.

## Wesentliche Eigenschaften

- klare Trennung von Modellressourcen, Prozesszustand, Bild und Dokumentation;
- proprietäre Prepare-/Execute-Schnittstellen gegen Fehlverkabelung;
- parallele Smart-Detailer- und FaceProcess-Instanzen;
- dynamische `SEGS`, `LOG BLOCK` und `DIAGNOSTIC`-Eingänge;
- persistente Branch-Caches für unveränderte parallele Instanzen;
- verpflichtende Modul-Boundaries vor Comparer, nachfolgenden Modulen und öffentlichen Ausgängen;
- eigenständige Nutzung von Detailer und FaceProcess über die CMK-Loader bleibt möglich;
- `CMK Image Load and Resize -Pipe-` lädt und skaliert ein Bild als kompakte SideKick-Quelle für direkt pixelbasierte Module; optionaler Advanced-Crop erhält das Zielseitenverhältnis mit `center/top/bottom/left/right`.
- `CMK Swap Image Loader -Pipe-` lädt Target und Source in einer zweispaltigen Oberfläche; nur das Target nutzt Resize und optionalen Advanced-Crop, die Source bleibt pixelmäßig unverändert.

## Installation

1. Bestehenden Ordner `custom_nodes/cmk_nodes` vollständig sichern.
2. Den vorhandenen Ordner vollständig ersetzen; keine alten Einzeldateien daneben liegen lassen.
3. Die JSON-Dateien unter `subgraphs/` im Node-Pack belassen. ComfyUI lädt `custom_nodes/cmk_nodes/subgraphs/*.json` automatisch; zusätzliche Kopien unter `user/default/subgraphs/` erzeugen doppelte Blueprint-Einträge.
4. Kuratierte Beispielworkflows im Flow Browser unter `Referenzen` als neue Kopie öffnen. `workflows/reference/` enthält ausschließlich technische Modulreferenzen und ist nicht für die parallele Standardinstallation vorgesehen.
5. ComfyUI vollständig neu starten.
6. Den zum Paketstand passenden Referenzworkflow laden.
7. Bei geänderten Node-Sockets oder Subgraph-Schnittstellen alte Instanzen im Workflow ersetzen, falls ComfyUI gespeicherte Portlayouts beibehält.

Vor dem Kopieren sollten vorhandene gleichnamige Workflows gesichert werden. Dateien unter `workflows/archive/` werden nicht für die normale Installation benötigt.

## Laufzeitabhängigkeiten

Je nach genutztem Modul werden insbesondere benötigt:

- ComfyUI Impact Pack;
- ComfyUI Impact Subpack;
- Ultralytics-Detector-Modelle;
- SAM-Modelle;
- ReActor beziehungsweise dessen Restore-Backend für FaceProcess-Restore;
- AIO Aux Preprocessors für entsprechende ControlNet-Preprocessor.

Fehlende optionale Module dürfen keine unbeteiligten CMK-Nodes aus der Registrierung entfernen. Der jeweilige Funktionspfad muss stattdessen eine klare Laufzeitmeldung liefern.

## Cache-Verhalten

Interne CMK-Caches liegen unter:

```text
ComfyUI/temp/cmk/
```

Ein erster Lauf nach Neustart, Codeänderung oder Cache-Bereinigung erzeugt erwartbar `MISS → STORED`. Unveränderte parallele Zweige und abgeschlossene Modulgrenzen sollen anschließend als `HIT` aufgelöst werden, ohne die teure Verarbeitung erneut auszuführen.

Branch-Caches besitzen Revisionsmarker. Detailer- und FaceProcess-Boundaries akzeptieren einen persistenten HIT nur dann, wenn ihr Dependency-Manifest exakt zu den aktuell materialisierten Branch-Revisionen passt. Nach dem Neustart eines einzelnen Zweigs werden deshalb Merge und Boundary aktualisiert, während unveränderte Geschwister aus ihrem Branch-Cache kommen.

`CMK SEGS CONCAT` übernimmt jedes vollständig komponierte Branch-Bild ausschließlich innerhalb des räumlichen Supports seiner tatsächlich zugeordneten SEG-Crop-Regionen. Eine Pixel-Differenzmaske wird nicht verwendet; dadurch können Quell- und Ergebnisbild nicht mehr als Salz-und-Pfeffer-Muster ineinander verschachtelt werden.
Persistente Detailer-/FaceProcess-Branches liefern dafür ein cache-stabiles internes Vollbild-/Support-Artefakt. FaceProcess-Branches enthalten ausschließlich ihre ausgewählten Gesichter. Frische Ausführung und Cache-Hit verwenden denselben Merge-Pfad.

Die UI von `CMK FaceProcess -Pipe-` zeigt abhängig von `PROCESS MODE` ausschließlich die gemeinsamen Parameter sowie den aktiven Restore- oder Detailer-Parametersatz. Die Werte des inaktiven Satzes bleiben intern erhalten und werden beim Zurückschalten wiederhergestellt.

Caches sind temporäre Beschleuniger und kein portables Projektformat.

## Dokumente

| Dokument | Aufgabe |
|---|---|
| `ARCHITECTURE.md` | verbindlicher aktueller Schnittstellen- und Architekturvertrag |
| `CMK_Design_Guidelines.md` | UI-, Benennungs- und Darstellungsregeln |
| `README.md` | Installation und Projektüberblick |
| `CHANGELOG.md` | historische Änderungen; keine aktuelle API-Definition |
| `WORKFLOWS.md` | Subgraph-, Workflow- und Installationsübersicht |
| `SUBGRAPH_AUDIT.md` | Nutzungsinventar und sicherer Bereinigungsplan der Subgraphs |
| `TOOLBOX.md` | Produktgrenze, Funktionsinventar und Pflegeplan des offenen Baukastens |
| `CMK_FLOW_COMPATIBILITY.md` | Entwurf des Integrationsvertrags für externe Baukasten-Nodes und Flow-Module |

## Lizenz

CMK Flow ist freie Software unter der **GNU General Public License,
Version 3 oder – nach eigener Wahl – jeder späteren Version**
(`GPL-3.0-or-later`). Das bedeutet insbesondere:

- CMK Flow darf kostenlos privat und kommerziell verwendet werden;
- der Quellcode darf untersucht, verändert und weitergegeben werden;
- weitergegebene Versionen und Änderungen müssen unter derselben Lizenz
  verfügbar bleiben und ihren Quellcode offenlegen;
- Copyright- und Lizenzhinweise müssen erhalten bleiben;
- die Software wird ohne Gewährleistung bereitgestellt.

Der vollständige Lizenztext steht in [`LICENSE`](LICENSE). Lizenzen externer
Custom Nodes, Modelle und anderer Abhängigkeiten gelten unabhängig davon weiter.

## Entwicklungsregel

Eine funktionierende technische Lösung ist noch keine CMK-Lösung, wenn sie:

- unnötige Anwenderentscheidungen erzeugt;
- Fehlverkabelungen leicht ermöglicht;
- interne Komplexität in den Hauptworkflow verlagert;
- oder unveränderte teure Module erneut ausführt.

In solchen Fällen gewinnt die Architektur.
## Video segmentieren

`CMK Split Video into Segments` wählt Videos aus `input/video/`, schreibt quellengetrennte Segmente nach `output/video/segments/<video_name>/` und liefert den strukturierten Arbeitskontext `CMK_VIDEO_SEGMENTS` für nachfolgende Video-Workflows.


`CMK Merge and Save Video` setzt `CMK_VIDEO_SEGMENTS` overlap-bereinigt nach `output/video/merged/<video_name>/` zusammen, zeigt das Ergebnis im Player unten an und veröffentlicht es optional ohne erneutes Encoding. Ausgänge: `CMK_VIDEO`, `FULLPATH`, `LOG`, `diagnostic`.

- `CMK FaceSwap Video Loader`: optional closed video/source-image loader over the persistent Split backend.
