<p align="center">
  <img src="web/assets/brand/cmk-logo.png" width="140" alt="CMK Flow logo">
</p>

# CMK Flow

Modulares Custom-Node-Paket für ComfyUI.

**Aktueller Release: CMK 2.5.0**

**Deutsch** · [English](README.en.md)

[Quellcode](https://github.com/CMKFlow/cmk_nodes) · [Dokumentation](#dokumente) · [Freiwillig unterstützen](https://paypal.me/CMKFlow)

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

Der `CMK Flow Browser` ist die zentrale Schnittstelle zwischen Anwender und CMK.
Er bietet direkt in ComfyUI einen kompakten Überblick, schnellen Zugriff auf
Nodes und Subgraphen, echte Vorschauen sowie kuratierte Referenzworkflows. Er
ist ein Arbeitswerkzeug und bewusst weder Handbuch noch technische
Dokumentation.

CMK 2.5 trennt die familientypische Generationszone von der
familienunabhängigen PostProcess-Zone:

```text
Vorbereitung: Loader / LoRA → 01 START HERE
Generierung: 02 Regional Conditioning (optional, SDXL)
              → 05 ControlNet (optional)
              → 10 KSampler
              → 15 InstantID (optional, SDXL)
              → 20 Refiner (SDXL)
Übergabe:     PostProcess Boundary (SDXL / Z-Image Turbo / Combined)
PostProcess:  FaceRebuild · FaceSwap · FaceProcess · Detailer · MaskDetailer
Ausgabe:      Visualizer und/oder Upscale & Save
```

SDXL, Z-Image Turbo und HYBRID bleiben während der Generierung technisch
getrennte Pfade. HYBRID baut das Bild zunächst mit SDXL auf und übergibt es für
den abschließenden Finish an Z-Image Turbo. Die passende `PostProcess Boundary`
beendet den Generationsabschnitt und stellt anschließend einen unabhängigen,
familienneutralen Arbeitskontext bereit. Bei parallelen Familien übernimmt die
Variante `Combined` ausschließlich den aktiven SDXL-, ZIT- oder HYBRID-Zweig.

Die sichtbaren Hauptrollen sind:

```text
MODEL | PROCESS | IMAGE | LOG | VISUAL
```

Zwischen SDXL-Sampler, InstantID und Refiner wird statt `IMAGE` der proprietäre
Latent-Übergabetyp `SAMPLED` verwendet. Sichtbare Titel dienen ausschließlich
der Darstellung; technische Identität und Navigation beruhen auf Metadaten,
Node-Klassen, Provider-Keys und UUIDs.

## Wesentliche Eigenschaften

- klare Trennung von Modellressourcen, Prozesszustand, Bild, Log und Visualisierung;
- proprietäre Prepare-/Execute-Schnittstellen gegen Fehlverkabelung;
- getrennte SDXL-, Z-Image-Turbo- und HYBRID-Generationspfade mit gemeinsamer
  familienneutraler PostProcess-Zone;
- parallele Smart-Detailer- und FaceProcess-Instanzen;
- dynamische `SEGS`, `LOG BLOCK` und `DIAGNOSTIC`-Eingänge;
- persistente Branch-Caches für unveränderte parallele Instanzen;
- verpflichtende Modul-Boundaries vor Comparer, nachfolgenden Modulen und öffentlichen Ausgängen;
- zentrale `VISUAL`-Kette für registrierte Bearbeitungsstufen im Visualizer;
- eigenständige Nutzung der PostProcess-Module in diskreten oder gekapselten
  Modul-Workflows;
- `CMK Flow · Image Input` verwendet für neue Nodes standardmäßig den sichtbaren seitenverhältnistreuen Crop. `center/top/bottom/left/right` bestimmen, welcher Bildbereich beim Resize erhalten bleibt; dadurch wird das Bild nicht auf das Zielseitenverhältnis verzerrt.
- `CMK Swap Image Loader -Pipe-` lädt Target und Source in einer zweispaltigen Oberfläche; nur das Target nutzt Resize und optionalen Advanced-Crop, die Source bleibt pixelmäßig unverändert.
- Z-Image Turbo besitzt eigene Loader-, Sampler- und optionale ControlNet-Module. ZIT-Inpaint bleibt aufgrund der sehr hohen Speicher- und Laufzeitanforderungen ausdrücklich `EXPERIMENTAL` und vorläufig eingefroren.
- Der Referenzkatalog gliedert sich in Task Workflows, Module Workflows,
  Comparisons, Real-World Workflows, System Workflow und Legacy. Er enthält 20
  aufsteigend komplexe Task-Workflows, diskrete und gekapselte Modulbeispiele,
  direkte Vergleiche, vollständige Praxisabläufe und den Full Flow als
  Systemreferenz.

## Installation

CMK wird derzeit manuell von GitHub installiert. ComfyUI vor der Installation
vollständig beenden.

### 1. Den richtigen ComfyUI-Ordner öffnen

Öffne den Hauptordner deiner ComfyUI-Installation. Es ist der Ordner, der die
Datei `main.py` und den Unterordner `custom_nodes` enthält. Öffne **genau in
diesem Ordner** ein Terminal.

Falls dort bereits `custom_nodes/cmk_nodes` existiert, nicht darüberinstallieren:
Den vorhandenen Ordner zuerst sichern oder vollständig entfernen.

### 2. CMK herunterladen und Abhängigkeiten installieren

Diese beiden Befehle nacheinander in das geöffnete Terminal kopieren:

```bash
git clone https://github.com/CMKFlow/cmk_nodes.git custom_nodes/cmk_nodes
python3 custom_nodes/cmk_nodes/scripts/install_cmk_requirements.py
```

Der Installationshelfer ermittelt aus dem Zielpfad automatisch die tatsächlich
von ComfyUI verwendete Python-Umgebung, installiert dort `requirements.txt` und
prüft anschließend alle verpflichtenden CMK-Importe. Das ist insbesondere bei
ComfyUI Desktop wichtig, weil System-Python, `standalone-env` und
`ComfyUI/.venv` nebeneinander vorhanden sein können.

Die Installation war erfolgreich, wenn am Ende diese Meldung erscheint:

```text
CMK dependency check: OK
```

### 3. Modellressourcen prüfen

Die Python-Prüfung lädt keine großen oder lizenzpflichtigen Modelle. Prüfe die
Ressourcen deshalb ausdrücklich gegen die Installation und optional gegen den
gemeinsamen Modellordner:

```bash
python3 custom_nodes/cmk_nodes/scripts/check_cmk_resources.py \\
  --models-root "/Pfad/zum/ComfyUI-Shared"
```

Jede Ressource wird einzeln als `FOUND` oder `MISSING` ausgegeben. Direkt nach
einem fehlenden, öffentlich verfügbaren Download fragt der Helfer, ob die Datei
jetzt in den angegebenen Modellordner installiert werden soll. Mit `n` läuft
die Prüfung weiter; mit `y` wird zuerst die aktuelle Datei installiert. Die
beiden öffentlichen InstantID-Dateien können so einzeln installiert werden;
alle anderen Ressourcen mit Lizenz- oder Modellwahl bleiben bewusst manuell.

```bash
python3 custom_nodes/cmk_nodes/scripts/check_cmk_resources.py \\
  --models-root "/Pfad/zum/ComfyUI-Shared" \\
  --install instantid-adapter \\
  --target-root "/Pfad/zum/ComfyUI-Shared"

python3 custom_nodes/cmk_nodes/scripts/check_cmk_resources.py \\
  --models-root "/Pfad/zum/ComfyUI-Shared" \\
  --install instantid-controlnet \\
  --target-root "/Pfad/zum/ComfyUI-Shared"
```

Der ControlNet-Download ist etwa 2,5 GB groß. Workflows und technische
Identitäten werden durch den Ressourcencheck nicht verändert.

Für automatisierte Prüfungen ohne Nachfrage steht `--non-interactive` zur
Verfügung.

### 4. ComfyUI neu starten

ComfyUI erst nach der Erfolgsmeldung wieder starten. Der Flow Browser wird mit
dem geladenen CMK-Paket automatisch registriert.

### Update

Für ein Update erneut ein Terminal im ComfyUI-Hauptordner öffnen und ausführen:

```bash
git -C custom_nodes/cmk_nodes pull --ff-only
python3 custom_nodes/cmk_nodes/scripts/install_cmk_requirements.py
```

Ein manueller Git-Clone führt niemals automatisch `pip` aus. Deshalb gehört der
zweite Befehl verbindlich zur Installation und zu jedem Update.

### Erforderliche ComfyUI-Oberfläche

> **Bestätigte Zielversion**
>
> CMK 2.5.0 wurde mit **ComfyUI 0.37.0** und
> **comfyui-frontend-package 1.52.7** vollständig geprüft. Dazu gehören die
> Flow-Browser-Navigation, paketierte Subgraphen und Referenzworkflows,
> eingebettete Vorschauen, Cache-Pfade sowie die serialisierten Link-, Socket-,
> UUID- und Topologieverträge.

CMK Flow benötigt die ComfyUI-Einstellung **Vue Nodes / Nodes 2.0**. Ohne sie
fallen dynamische CMK-Nodes auf die alte LiteGraph-Darstellung zurück;
Advanced-Umschaltung, Dropdowns, Shapes und automatische Größenanpassung stehen
dann nicht wie vorgesehen zur Verfügung. CMK zeigt beim Start einen Hinweis,
wenn die Einstellung im aktuellen Benutzerprofil nicht aktiviert ist. Die
Oberfläche wechselt beim Aktivieren unmittelbar; ein Neuladen ist nicht nötig.

Für Vorschauen während Sampler- und Refiner-Läufen sollte unter
**Comfy → Execution → Live preview method** der Wert **auto** gewählt sein.

Keine alten Einzeldateien neben einer aktuellen Installation liegen lassen. Die
JSON-Dateien unter `subgraphs/` bleiben im Node-Pack. Zusätzliche Kopien unter
`user/default/subgraphs/` erzeugen doppelte Blueprint-Einträge.

Vor dem Kopieren sollten vorhandene gleichnamige Workflows außerhalb des Node-Packs gesichert werden. Historische Entwicklungsstände sind nicht Bestandteil der öffentlichen CMK-Veröffentlichung.

## Laufzeitabhängigkeiten

Die CMK-Kernnodes für Detektion, `SEGS`, Detailer, Pasteback,
FaceProcess-Restore, SAM-Laden und die angebotenen ControlNet-Preprozessoren
benötigen keine fremden Custom-Node-Pakete. Die optionalen Subgraphen
`LoRA Stack · SDXL`, `LoRA Stack · ZIT` und `LoRA Stack · Combined` sowie
Referenzworkflows, die sie verwenden, setzen den
[ComfyUI LoRA Manager](https://github.com/willmiao/ComfyUI-Lora-Manager)
voraus. Ohne ihn bleiben die übrigen CMK-Module verwendbar.

Die Python-Bibliotheken stehen in `requirements.txt`. Modellgestützte Funktionen
benötigen weiterhin die jeweils ausgewählten Modelle, insbesondere
Ultralytics-Detektormodelle, SAM-Modelle, InsightFace-Modelle sowie optionale
Face-Restore-Modelle. Fehlende Modelle blockieren nicht die Registrierung
unbeteiligter CMK-Nodes, sondern erzeugen im gewählten Funktionspfad eine klare
Laufzeitmeldung.

## FaceSwap ContentGuard

Die öffentlichen CMK-FaceSwap-Pfade besitzen einen verpflichtenden lokalen
ContentGuard. Er prüft Quell- und Zielbilder vor dem Swap; im Videopfad wird
jedes Ziel-Frame geprüft. Explizite Inhalte, ein geschätztes Alter unter 18,
ein Source-Alter unter der konservativen 25er-Grenze sowie fehlende oder
fehlerhafte Schutzmodelle führen zu einem harten Abbruch. Es gibt in der
öffentlichen Oberfläche keine Umgehungsoption. Deaktivierte FaceSwap-Nodes
bleiben echte Pass-through-Pfade und laden den Guard nicht.

Der Guard arbeitet vollständig lokal mit NudeNet zur Erkennung expliziter
Inhalte und InsightFace zur Altersschätzung. Für Target-Gesichter gilt die
Erwachsenen-Grenze 18, um instabile Schätzungen
bei generierten oder stilisierten Gesichtern nicht fälschlich zu sperren.
Die Bewertung hängt von den Bildinhalten und dem Altersmodell des installierten
InsightFace-Pakets ab. Sie ist eine technische Risikobegrenzung, keine
Einwilligungsprüfung und keine Garantie gegen Fehlklassifikationen. Details zur
Policy und zu den neutralen Diagnosecodes stehen in
[`CONTENT_GUARD.md`](CONTENT_GUARD.md).

## Cache-Verhalten

Interne CMK-Caches liegen unter:

```text
ComfyUI/temp/cmk/
```

Ein erster Lauf nach Neustart, Codeänderung oder Cache-Bereinigung erzeugt
erwartbar `MISS → STORED`. Unveränderte parallele Zweige und abgeschlossene
Modulgrenzen sollen innerhalb derselben ComfyUI-Sitzung anschließend als `HIT`
aufgelöst werden, ohne die teure Verarbeitung erneut auszuführen. Ein
ComfyUI-Neustart leert diese Bild-Boundary-Caches; nur die ausdrücklich
persistente Videoverarbeitung besitzt einen sitzungsübergreifenden
Fortsetzungsvertrag.

Branch-Caches besitzen Revisionsmarker. Detailer- und FaceProcess-Boundaries
akzeptieren einen HIT nur dann, wenn ihr Dependency-Manifest exakt zu den
aktuell materialisierten Branch-Revisionen passt. Nach einer Änderung eines
einzelnen Zweigs werden deshalb Merge und Boundary aktualisiert, während
unveränderte Geschwister innerhalb derselben Sitzung aus ihrem Branch-Cache
kommen.

`CMK SEGS CONCAT` übernimmt jedes vollständig komponierte Branch-Bild ausschließlich innerhalb des räumlichen Supports seiner tatsächlich zugeordneten SEG-Crop-Regionen. Eine Pixel-Differenzmaske wird nicht verwendet; dadurch können Quell- und Ergebnisbild nicht mehr als Salz-und-Pfeffer-Muster ineinander verschachtelt werden.
Persistente Detailer-/FaceProcess-Branches liefern dafür ein cache-stabiles internes Vollbild-/Support-Artefakt. FaceProcess-Branches enthalten ausschließlich ihre ausgewählten Gesichter. Frische Ausführung und Cache-Hit verwenden denselben Merge-Pfad.

Die UI von `CMK FaceProcess -Pipe-` zeigt abhängig von `PROCESS MODE` ausschließlich die gemeinsamen Parameter sowie den aktiven Restore- oder Detailer-Parametersatz. Die Werte des inaktiven Satzes bleiben intern erhalten und werden beim Zurückschalten wiederhergestellt.

Caches sind temporäre Beschleuniger und kein portables Projektformat.

## Videoverarbeitung

`CMK Split Video into Segments` wählt Videos aus `input/video/`, schreibt
quellengetrennte Segmente nach `output/video/segments/<video_name>/` und liefert
den persistenten Arbeitskontext `CMK_VIDEO_SEGMENTS` für nachfolgende
Video-Workflows.

`CMK Merge and Save Video` setzt die Segmente overlap-bereinigt nach
`output/video/merged/<video_name>/` zusammen, zeigt das Ergebnis im Player und
veröffentlicht es optional ohne erneutes Encoding. `CMK FaceSwap Video Loader`
ergänzt den persistenten Split-Pfad um Video- und Source-Auswahl.

`CMK FaceSwap Video` ist die historische Legacy-Referenz, mit der die
Entwicklung von CMK begann. Der Workflow stammt aus CMK 1.0, wurde bewusst
nicht auf die CMK-2.5-Architektur modernisiert und bleibt unverändert im
aktuellen Umfeld lauffähig.

## Dokumente

| Dokument | Aufgabe |
|---|---|
| `ARCHITECTURE.md` | verbindlicher aktueller Schnittstellen- und Architekturvertrag |
| `CMK_Design_Guidelines.md` | UI-, Benennungs- und Darstellungsregeln |
| `README.md` | Installation und Projektüberblick |
| `README.en.md` | englische Installation und Projektübersicht |
| `CHANGELOG.md` | historische Änderungen; keine aktuelle API-Definition |
| `WORKFLOWS.md` | Subgraph-, Workflow- und Installationsübersicht |
| `SUBGRAPH_AUDIT.md` | Nutzungsinventar und sicherer Bereinigungsplan der Subgraphs |
| `TOOLBOX.md` | Produktgrenze, Funktionsinventar und Pflegeplan des offenen Baukastens |
| `CMK_FLOW_COMPATIBILITY.md` | Entwurf des Integrationsvertrags für externe Baukasten-Nodes und Flow-Module |
| `CONTENT_GUARD.md` | verbindliche lokale FaceSwap-Schutzpolicy, Abbruchregeln und Grenzen |
| `RELEASE_AUDIT_CMK_2_5.md` | Abschlussaudit der CMK-2.5-Verträge, Tests und Release-Abweichungen |

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
