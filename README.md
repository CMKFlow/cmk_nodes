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

## Inspiration und Anerkennung

Der [ComfyUI LoRA Manager von willmiao](https://github.com/willmiao/ComfyUI-Lora-Manager)
hat die Erwartungen an den CMK Flow Browser wesentlich mitgeprägt. Seine direkt
aus ComfyUI geöffnete Browseroberfläche für Checkpoints, LoRAs, Metadaten und
Civitai-Inhalte zeigte eindrucksvoll, wie komfortabel eine integrierte
Verwaltungsoberfläche sein kann.

CMK Flow ist ein eigenständiges Projekt ohne offizielle Verbindung zum LoRA
Manager. Diese Nennung würdigt die gestalterische Inspiration und die umfangreiche
Arbeit hinter dem Open-Source-Projekt; sie behauptet keine Übernahme von Code.

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

In einem Terminal der gewünschten ComfyUI-Installation:

```bash
cd /Pfad/zu/ComfyUI
git clone https://github.com/CMKFlow/cmk_nodes.git custom_nodes/cmk_nodes
python -m pip install -r custom_nodes/cmk_nodes/requirements.txt
```

Anschließend ComfyUI vollständig beenden und neu starten. Für ein Update:

```bash
git -C custom_nodes/cmk_nodes pull --ff-only
python -m pip install -r custom_nodes/cmk_nodes/requirements.txt
```

### Erforderliche ComfyUI-Oberfläche

CMK Flow benötigt die ComfyUI-Einstellung **Vue Nodes / Nodes 2.0**. Ohne sie
fallen dynamische CMK-Nodes auf die alte LiteGraph-Darstellung zurück;
Advanced-Umschaltung, Dropdowns, Shapes und automatische Größenanpassung stehen
dann nicht wie vorgesehen zur Verfügung. CMK zeigt beim Start einen Hinweis,
wenn die Einstellung im aktuellen Benutzerprofil nicht aktiviert ist. Die
Oberfläche wechselt beim Aktivieren unmittelbar; ein Neuladen ist nicht nötig.

Für Vorschauen während Sampler- und Refiner-Läufen sollte unter
**Comfy → Execution → Live preview method** der Wert **auto** gewählt sein.

Bei einer bestehenden manuellen CMK-Installation den bisherigen Ordner zuerst sichern und vollständig ersetzen; keine alten Einzeldateien daneben liegen lassen. Die JSON-Dateien unter `subgraphs/` bleiben im Node-Pack. Zusätzliche Kopien unter `user/default/subgraphs/` erzeugen doppelte Blueprint-Einträge.

Vor dem Kopieren sollten vorhandene gleichnamige Workflows außerhalb des Node-Packs gesichert werden. Historische Entwicklungsstände sind nicht Bestandteil der öffentlichen CMK-Veröffentlichung.

## Laufzeitabhängigkeiten

CMK Flow benötigt keine fremden Custom-Node-Pakete. Detektion, `SEGS`, Detailer,
Pasteback, FaceProcess-Restore, SAM-Laden und die angebotenen
ControlNet-Preprozessoren werden innerhalb von CMK bereitgestellt.

Die Python-Bibliotheken stehen in `requirements.txt`. Modellgestützte Funktionen
benötigen weiterhin die jeweils ausgewählten Modelle, insbesondere
Ultralytics-Detektormodelle, SAM-Modelle, InsightFace-Modelle sowie optionale
Face-Restore-Modelle. Fehlende Modelle blockieren nicht die Registrierung
unbeteiligter CMK-Nodes, sondern erzeugen im gewählten Funktionspfad eine klare
Laufzeitmeldung.

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
