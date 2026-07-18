# CMK Architecture

**Vertragsstand:** 1.0  
**Status:** verbindlicher Architektur- und Schnittstellenvertrag  
**Referenzbasis:** veröffentlichte Module unter `subgraphs/` und kuratierte Workflows unter `workflows/showcase/`

Dieses Dokument ist die einzige verbindliche Quelle für die aktuelle Architektur des CMK-`-Pipe-`-Ökosystems. README, Design Guidelines und Changelog dürfen diesen Vertrag erläutern, aber nicht abweichend definieren.

Bei einem Widerspruch gilt folgende Reihenfolge:

```text
ARCHITECTURE.md
    ↓
implementierte Socket-Typen und Modulgrenzen
    ↓
README / Design Guidelines
    ↓
historische Changelog-Einträge
```

## 1. Grundprinzip

**CMK bietet keine Möglichkeiten, sondern Lösungen.**

> Jede Entscheidung, welche der Anwender treffen muss, ist eine Niederlage.

Technische Komplexität darf intern beliebig hoch sein, wenn sie nach außen eine einfache, vorhersehbare und fehlertolerante Bedienung ermöglicht.

### 1.1 Drei Auslieferungs- und Ausführungsebenen

CMK besteht aus drei aufeinander aufbauenden Artefaktebenen:

```text
Referenz- und Beispielworkflows
    ↓ verwenden
ComfyUI-Subgraphs
    ↓ komponieren
registrierte CMK-Python-Nodes und Engines
```

- Die Python-Nodes in diesem Paket implementieren Typverträge, Verarbeitung, Caches und Diagnostik.
- Die JSON-Dateien unter `subgraphs/` bilden daraus die geführten sichtbaren CMK-Module. Ihre UUIDs sind Teil des Workflow-Vertrags und dürfen nicht ohne Migration geändert werden.
- Die JSON-Dateien unter `workflows/reference/` definieren den verbindlichen Gesamtaufbau. `workflows/examples/` enthält eigenständige Anwendungsbeispiele; `workflows/archive/` dient ausschließlich der historischen Nachvollziehbarkeit.

Die versionierten Subgraphdateien werden direkt aus `custom_nodes/cmk_nodes/subgraphs/` geladen. Sie dürfen nicht zusätzlich nach `ComfyUI/user/default/subgraphs/` kopiert werden, weil ComfyUI Node-Pack-Blueprints und benutzereigene Blueprints als getrennte Quellen registriert. Nur installierte Workflows werden nach `ComfyUI/user/default/workflows/` kopiert.

## 2. Zwei klar getrennte Produktwelten

### 2.1 Geschlossenes `-Pipe-`-Ökosystem

Der Namenszusatz `-Pipe-` ist ein Architekturvertrag.

> `CMK **** -Pipe-` bildet ein in sich geschlossenes Ökosystem – vergleichbar mit Apple.

Für `-Pipe-`-Nodes und die daraus gebildeten Module gelten verbindlich:

- Der vorgesehene Standardweg ist aus Namen, Socket-Reihenfolge und Kabelverlauf unmittelbar verständlich.
- Interne Hilfsnodes, proprietäre Datentypen, parallele Zweige, Merge-Nodes und Caches bleiben innerhalb des Moduls.
- Fehlverkabelungen werden durch dedizierte Socket-Typen verhindert oder zumindest deutlich erschwert.
- Ein verpflichtender interner Boundary darf nicht über einen alternativen Bild-, Pipe- oder Log-Ausgang umgangen werden.
- Die äußere Workflow-Topologie bleibt kompakt und möglichst linear.
- Die normale Verwendung erfordert keine Kenntnis der internen Implementierung.
- Module bleiben dennoch über die kanonischen CMK-Einstiegspunkte eigenständig nutzbar.

Eine Node ist erst dann als `-Pipe-`-Node fertig, wenn dieser Vertrag erfüllt ist.

### 2.2 Offener Experimentierkasten

Alle übrigen CMK-Nodes bilden den offenen Experimentierkasten.

Sie dürfen:

- native ComfyUI-Typen direkt verwenden;
- einzelne Verarbeitungsschritte offenlegen;
- manuelle Verkabelung und alternative Reihenfolgen erlauben;
- Flexibilität über Schutzmechanismen stellen.

Die offene Toolbox darf das geschlossene `-Pipe-`-Ökosystem ergänzen, aber dessen Schutz- und Bedienkonzept nicht aufweichen.

Der Baukasten ist ein eigenständiger Produktbereich und kein Ablageort für veraltete Flow-Bausteine. Er benötigt eigene, aktuelle Beispiele, dokumentierte Abhängigkeiten und regelmäßige Funktionstests. Eine Funktion kann im Baukasten bewusst mehr Eingriffe erlauben als im Apple-like Flow-Frontend, muss aber weiterhin einen klar benannten und gepflegten Einsatzzweck besitzen.

`CMK FaceSwap Image -Pipe-` gehört zum geschlossenen Flow. Im Referenzworkflow wird seine Execute-Node innerhalb von `CMK SwapFace -Pipe-` gemeinsam mit Source-Loader und verpflichtender Boundary komponiert. Der eigenständig gespeicherte Subgraph bleibt eine alternative Flow-Oberfläche desselben vollständig funktionsfähigen Moduls. Geplante Anpassungen an dieser Oberfläche ändern seine Produktzuordnung nicht.

Zum offenen Baukasten gehört dagegen `CMK FaceSwap Image` **ohne** `-Pipe-`. Diese Variante stellt zusätzliche manuelle Auswahl- und Experimentierschnittstellen bereit.

## 3. Die vier sichtbaren Transportrollen

Der normale pixelbasierte CMK-Workflow verwendet vier semantisch getrennte Leitungen:

| Sichtbarer Name | Typ | Verantwortung | Darf nicht enthalten oder bewirken |
|---|---|---|---|
| `MODEL` | `CMK_MODEL_PIPE` | gemeinsame, unveränderte Modellressourcen und kleine Loader-Metadaten | Modulzustand, Bilder, Latents, Logs, lokale Patches |
| `PROCESS` | `CMK_PIPE` | nicht-pixelbasierter Prozesskontext, Masken, Dimensionen, Prompts, LoRA-Angebote und Workflow-Zustand | das authoritative Ergebnisbild, Modellressourcen, Dokumentationslogik |
| `IMAGE` | `IMAGE` | authoritative Pixelinformation zwischen Bildmodulen | versteckte Zustands- oder Logfunktion |
| `LOG` | `CMK_LOG_PIPE` | strukturierte Dokumentation bereits getroffener Entscheidungen und ausgeführter Schritte | Steuerung, Bildveränderung, Modellpatching, Modulaktivierung |

Die Rollen beantworten vier unterschiedliche Fragen:

```text
MODEL   → Welche unveränderten Modellressourcen stehen zur Verfügung?
PROCESS → Welcher Prozesskontext wird angeboten?
IMAGE   → Welches Bild wird tatsächlich weiterverarbeitet?
LOG     → Was wurde dokumentiert?
```

### 3.1 Warum IMAGE eine eigene Leitung ist

`IMAGE` liegt bewusst nicht im öffentlichen `PROCESS`-Transport:

- Bildänderungen bleiben im Workflow sichtbar.
- Ein Modul kann kein veraltetes, intern gespiegeltes Bild versehentlich bevorzugen.
- Pixelverarbeitung und Prozessmetadaten werden unabhängig cachebar.
- Standalone-Bildmodule können mit nativen `IMAGE`-Sockets arbeiten.
- Ein Bild muss nicht durch eine Node geleitet werden, die es fachlich gar nicht verarbeitet.

Interne Arbeits-Pipes dürfen zur Ausführung ein Bild referenzieren. Das ändert nicht den öffentlichen Vertrag: Zwischen Modulen bleibt `IMAGE` authoritative.

## 4. Proprietäre Arbeits- und Übergabetypen

Die folgenden Typen sind keine zusätzlichen allgemeinen Workflow-Leitungen. Sie schützen konkrete Modulgrenzen:

| Typ | Erzeuger | Zulässiger Hauptverbraucher | Zweck |
|---|---|---|---|
| `CMK_SAMPLER_PIPE` / `SAMPLER` | `CMK Sampler Prepare SDXL -Pipe-` | `CMK KSampler -Pipe-` | vollständig vorbereiteter erster Sampling-Kontext |
| `CMK_SAMPLED_PIPE` / `SAMPLED` | `CMK KSampler -Pipe-` | `CMK Refiner Prepare SDXL -Pipe-` | abgeschlossener First-Pass-Latent samt benötigtem Kontext |
| `CMK_REFINER_PIPE` / `REFINER` | `CMK Refiner Prepare SDXL -Pipe-` | `CMK Refiner -Pipe-` | vollständig vorbereiteter Refiner-Kontext |
| `CMK_DETAILER_PIPE` / `DETAILER` | `CMK Detailer Prepare -Pipe-` | parallele `CMK Smart Detailer -Pipe-` | unveränderlicher gemeinsamer Detailer-Kontext |
| `CMK_FACE_PIPE` / `FACE` | `CMK FaceProcess Prepare -Pipe-` | parallele `CMK FaceProcess -Pipe-` | unveränderlicher gemeinsamer FaceProcess-Kontext |
| `CMK_LOG_BLOCK` / `LOG BLOCK` | einzelne Execute-Instanz | ausschließlich `CMK LOG CONCAT` | lokaler Logbeitrag ohne Verkettung vollständiger LOG-Pipes |

Diese Typen verhindern, dass ein allgemeiner `PROCESS`-, `IMAGE`- oder `LOG`-Socket eine geschützte Modulschnittstelle ersetzt.

## 5. Verbindliche Moduloberflächen des Referenzworkflows

### 5.1 Bild- und Prozessquelle

`CMK Pipe Create Image -Pipe-`:

```text
IMAGE + MASK + FILENAME STRING
+ LORA STACK + optional LORA SYNTAX
+ PROMPT POS + PROMPT NEG
+ Bild-/Maskenparameter
    ↓
PROCESS + IMAGE + LOG + diagnostic
```

Verantwortung:

- erzeugt den initialen nicht-pixelbasierten `PROCESS`-Kontext;
- gibt das skalierte authoritative `IMAGE` separat aus;
- erzeugt den ersten strukturierten `LOG`;
- erstellt noch kein Sampler-Latent.

`CMK Load Image -Pipe-` ist der kompakte Standalone-Einstieg ohne Größenänderung:

```text
Bilddatei → PROCESS + IMAGE + LOG
```

`CMK Image Load and Resize -Pipe-` ist der SideKick-Einstieg für direkt pixelbasierte Module:

```text
Bilddatei + RESOLUTION + SWAP DIMENSIONS + RESIZE METHOD
+ optional Advanced: CROP + CROP POSITION
    ↓
PROCESS + IMAGE + LOG + diagnostic
```

Verantwortung:

- lädt und skaliert das Bild in einer einzigen geführten Node;
- kann das Quellbild optional vor dem Resize auf das Seitenverhältnis der Zielauflösung beschneiden;
- die Advanced-Crop-Positionen sind `center`, `top`, `bottom`, `left` und `right`;
- transportiert Pixel ausschließlich über das authoritative `IMAGE`;
- legt in `PROCESS` nur Quell-/Zielauflösung, Crop- und Dateimetadaten ab;
- erzeugt weder Maske noch Prompt-, LoRA-, Inpaint-, Outpaint- oder Latent-Zustand;
- speist insbesondere Detailer, FaceProcess und vergleichbare direkt pixelbasierte CMK-Module standalone.

`CMK Swap Image Loader -Pipe-` ist der geführte Dual-Loader für das FaceSwap-Image-Modul:

```text
TARGET IMAGE + RESOLUTION + SWAP DIMENSIONS + RESIZE METHOD
+ optional Advanced: CROP + CROP POSITION
SOURCE IMAGE ohne Resize/Crop
    ↓
PROCESS + IMAGE TARGET + IMAGE SOURCE + LOG + diagnostic
```

Verantwortung:

- zeigt Target und Source in einer gemeinsamen, zweispaltigen Loader-Oberfläche;
- behandelt das Target exakt wie `CMK Image Load and Resize -Pipe-`;
- lädt das Source-/Referenzbild EXIF-korrigiert, aber unverändert in seiner nativen Pixelgröße;
- transportiert beide Bilder ausschließlich über getrennte authoritative IMAGE-Ausgänge;
- hält `PROCESS` frei von Pixeln, Masken, Prompt-, LoRA-, Inpaint-, Outpaint- und Latent-Zustand;
- dokumentiert beide Dateien und ihre unabhängigen Größen im gemeinsamen strukturierten `LOG`;
- ist der kanonische standalone Bildeinstieg für `CMK FaceSwap Image -Pipe-`.

`CMK FaceSwap Image -Pipe-` verwendet:

```text
PROCESS + IMAGE_TARGET + IMAGE_SOURCE + LOG
    ↓
PROCESS + IMAGE + LOG + diagnostic
```

Regeln:

- `IMAGE_TARGET` ist das authoritative Zielbild; `IMAGE_SOURCE` ist ausschließlich Referenzbild.
- `PROCESS` bleibt frei von Pixeln und wird nur um FaceSwap-Metadaten ergänzt.
- `LOG` dokumentiert den Swap, steuert ihn aber nicht.
- Die offene `CMK FaceSwap Image` behält zusätzlich ihre optionalen `CMK_SELECTED_FACE`-Eingänge für den Experimentierkasten.
- Gemeinsame technische Advanced-Parameter beider Nodes sind `bbox_dilation`, `crop_factor`, `drop_size` und `feather`.
- `crop_factor` besitzt den Default `1.5`.


Das eingeschleifte FaceSwap-Modul verwendet dieselbe Execute-Node mit einem internen Source-Loader:

```text
MODEL + PROCESS + IMAGE_TARGET + LOG + ENABLE
    ↓
[CMK Load Image -Pipe- nur für IMAGE_SOURCE]
    ↓
CMK FaceSwap Image -Pipe-
    ↓
CMK Boundary Cache
    ↓
MODEL + PROCESS + IMAGE + LOG
```

Modulregeln:

- Das eingehende `IMAGE_TARGET` bleibt das authoritative Workflowbild.
- `CMK Load Image -Pipe-` liefert intern ausschließlich das Referenzbild an `IMAGE_SOURCE`; sein `PROCESS` und `LOG` ersetzen niemals den eingehenden Modulzustand.
- `MODEL` bleibt read-only und wird durch den FaceSwap-Boundary geführt, ohne von FaceSwap verwendet oder verändert zu werden.
- Öffentliche Ausgänge und interne Ergebnis-Previews beziehen `MODEL`, `PROCESS`, `IMAGE` beziehungsweise `LOG` ausschließlich vom verpflichtenden `CMK Boundary Cache`.
- `ENABLE = OFF` ist ein echter Lazy-Passthrough: `IMAGE_TARGET` wird unverändert ausgegeben; interner Source-Loader, Face Detection und Swap werden nicht angefordert.
- Der persistente FaceSwap-Boundary materialisiert `PROCESS`, `IMAGE` und `LOG` gemeinsam. `MODEL` wird aufgrund seiner read-only Rolle nur aktuell durchgereicht und nicht serialisiert.
- Der standalone Einstieg über `CMK Swap Image Loader -Pipe-` und das eingeschleifte Subgraph-Modul bleiben zwei gleichwertige Einsatzformen derselben FaceSwap-Engine.

### 5.2 Modellquelle

`CMK Checkpoint VAE Loader -Pipe-`:

```text
CHECKPOINT + VAE-Auswahl → MODEL
```

`MODEL` enthält ausschließlich:

```python
{
    "model": ...,
    "clip": ...,
    "vae": ...,
    "ckpt_name": str,
    "vae_name": str,
    "checkpoint_vae": bool,
    "vae_source": str,
    "model_pipe_source": str,
}
```

Lokale LoRAs, PAG, Sampling-Patches, FreeU und modulbezogener Zustand werden niemals in `MODEL` zurückgeschrieben.

### 5.3 ControlNet

`CMK ControlNet Prepare -Pipe-`:

```text
PROCESS + IMAGE + LOG
    ↓
PROCESS + IMAGE + LOG + diagnostic
```

- `IMAGE` bleibt authoritative und wird nicht durch eine versteckte Kopie ersetzt.
- ControlNet-Zustand wird ausschließlich dem kopierten `PROCESS` hinzugefügt.
- `LOG` erhält ausschließlich Dokumentation.
- Bei deaktiviertem ControlNet werden Modell, Referenzbild und Preprocessor nicht unnötig geladen.

### 5.4 Sampler-Modul

Öffentliche Moduloberfläche im Referenzworkflow:

```text
PROCESS + IMAGE + LOG
    ↓
MODEL + PROCESS + SAMPLED + LOG
```

Interne Aufteilung:

```text
CMK Checkpoint VAE Loader -Pipe- → MODEL

MODEL + PROCESS + IMAGE + optional LOG
    ↓
CMK Sampler Prepare SDXL -Pipe-
    ↓
SAMPLER + LOG + diagnostic

SAMPLER
    ↓
CMK KSampler -Pipe-
    ↓
SAMPLED
```

Verbindliche Gründe:

- `CMK KSampler -Pipe-` akzeptiert ausschließlich `SAMPLER`; ein unvorbereiteter `PROCESS` kann nicht angeschlossen werden.
- Der Execute-Knoten gibt ausschließlich `SAMPLED` aus. `PROCESS`, `IMAGE` und `LOG` werden nicht durch den teuren Compute-Knoten transportiert.
- `SAMPLED` ist der fachlich richtige Übergabetyp, weil der Refiner den First-Pass-Latent benötigt.
- Der Sampler gibt kein Bild an den Refiner weiter; der Refiner dekodiert das First-Pass-Bild aus dem unveränderten Latent.

### 5.5 Refiner-Modul

Öffentliche Moduloberfläche:

```text
MODEL + PROCESS + SAMPLED + LOG
    ↓
MODEL + PROCESS + IMAGE REFINED + LOG
```

Wichtige Semantik:

- Das eingehende öffentliche `MODEL` ist die gemeinsame Modellleitung für nachfolgende Module.
- Der Refiner darf intern einen eigenen Checkpoint über einen eigenen `CMK Checkpoint VAE Loader -Pipe-` verwenden.
- `CMK Refiner Prepare SDXL -Pipe-` benötigt deshalb `SAMPLED`, aber kein externes `IMAGE`.

Interne Ausführung:

```text
Refiner-MODEL + PROCESS + SAMPLED + LOG
    ↓
CMK Refiner Prepare SDXL -Pipe-
    ↓
REFINER + LOG + diagnostic

REFINER
    ↓
CMK Refiner -Pipe-
    ↓
IMAGE 1ST PASS + IMAGE REFINED

MODEL + PROCESS + beide Bilder + LOG
    ↓
CMK Boundary Cache
    ↓
MODEL + PROCESS + beide Bilder + LOG
```

Der Boundary ist verpflichtend:

- Er speichert `IMAGE 1ST PASS` und `IMAGE REFINED`.
- Der interne Image Comparer erhält beide Bilder ausschließlich vom Boundary.
- Der öffentliche Refiner-Ausgang erhält `IMAGE REFINED` ausschließlich vom Boundary.
- Kein Bildverbraucher darf direkt an `CMK Refiner -Pipe-` angeschlossen werden.

Damit kann weder ein Comparer noch ein nachfolgendes Modul den Sampler/Refiner erneut anfordern, solange der Refiner-Fingerprint unverändert ist.

### 5.6 Detailer-Modul

Öffentliche Moduloberfläche:

```text
MODEL + PROCESS + IMAGE + LOG + GLOBAL ENABLE
    ↓
MODEL + PROCESS + IMAGE + LOG
```

Prepare:

```text
MODEL + PROCESS + IMAGE + optional LOG
    ↓
CMK Detailer Prepare -Pipe-
    ↓
DETAILER + LOG + diagnostic
```

`DETAILER` ist ein unveränderlicher gemeinsamer Arbeitskontext. Mehrere Smart Detailer erhalten ihn parallel.

Einzelne Execute-Instanz:

```text
DETAILER + LOCAL ENABLE + lokale Detailer-Parameter
    ↓
SEGS DETECTED
SEGS PROCEED
IMAGE PROCEED
diagnostic
LOG BLOCK
```

Bewusst nicht vorhanden:

```text
kein DETAILER-Ausgang
kein vollständiger LOG-Ausgang
kein unverändertes IMAGE-Ausgangskabel
```

Begründung:

- Ein weitergereichter `DETAILER`-Ausgang würde Detailer 2 formal von Detailer 1 abhängig machen.
- Ein vollständiger `LOG → LOG`-Durchlauf würde dieselbe künstliche Kette erzeugen.
- Ein unverändertes `IMAGE` am Execute-Knoten wäre fachfremder Transport und eine Umgehungsmöglichkeit des Merge-/Boundary-Pfads.

Zusammenführung:

```text
authoritative IMAGE + SEGS 1 ... N
    ↓
CMK SEGS CONCAT
    ↓
zusammengesetztes IMAGE

Basis-LOG + LOG BLOCK 1 ... N
    ↓
CMK LOG CONCAT
    ↓
zusammengeführtes LOG
```

`CMK SEGS CONCAT` behandelt jeden parallelen SEGS-Eingang als unabhängigen Branch. CMK-Branches liefern ein vollständig komponiertes Branch-Bild und einen räumlichen Support aus den tatsächlich zugeordneten SEG-Crop-Regionen. Der Merge übernimmt das Branch-Bild ausschließlich innerhalb dieses Supports. Eine Pixel-Differenzmaske ist verboten, weil sie Quell- und Ergebnisbild pixelweise verschachteln kann. Unveränderte Geschwistersegmente besitzen keine überschreibende Wirkung. Bei tatsächlicher räumlicher Überlappung zweier bearbeiteter Branches definiert die Eingangsreihenfolge die Priorität; der spätere Eingang gewinnt.

Modulabschluss:

```text
MODEL + PROCESS + zusammengesetztes IMAGE + zusammengeführtes LOG
    ↓
CMK Boundary Cache
    ↓
öffentliche Detailer-Ausgänge und interner Comparer
```

Alle öffentlichen Detailer-Ausgänge und der Comparer liegen hinter diesem Boundary.

### 5.7 FaceProcess-Modul

Öffentliche Moduloberfläche:

```text
MODEL + PROCESS + IMAGE + LOG + GLOBAL ENABLE
    ↓
IMAGE + LOG
```

Prepare:

```text
MODEL + PROCESS + IMAGE + optional LOG
    ↓
CMK FaceProcess Prepare -Pipe-
    ↓
FACE + LOG + diagnostic
```

Mehrere FaceProcess-Instanzen erhalten denselben unveränderlichen `FACE`-Kontext parallel.

Einzelne Execute-Instanz:

```text
FACE
+ LOCAL ENABLE
+ PROCESS MODE: restore | detailer
+ FACE SELECTION
+ optional REFINE MODE
+ lokale Parameter
    ↓
IMAGE PROCEED
SELECTED FACE
SEGS PROCESSED
ENABLED
diagnostic
LOG BLOCK
```

Bewusst nicht vorhanden:

```text
kein FACE-Ausgang
kein vollständiger LOG-Ausgang
kein unverändertes IMAGE-Ausgangskabel
```

`SEGS PROCESSED` mehrerer Instanzen werden über `CMK SEGS CONCAT` auf das gemeinsame authoritative Eingangsbild angewendet. Jede FaceProcess-Instanz darf darin ausschließlich die ausgewählten und tatsächlich von ihr verantworteten Gesichter ausgeben. Das cache-stabile Branch-Bild wird über deren räumlichen SEG-Support übernommen; unveränderte Geschwistergesichter sind kein Branch-Beitrag. Die Logbeiträge werden über `CMK LOG CONCAT` zusammengeführt.

Der Modulabschluss ist ein verpflichtender `CMK Boundary Cache`:

```text
zusammengesetztes IMAGE + zusammengeführtes LOG
    ↓
CMK Boundary Cache
    ↓
öffentliche FaceProcess-Ausgänge und interner Comparer
```

### 5.8 Upscale und Save

`CMK Upscale and Save -Pipe-` ist der verbindliche Abschluss des geschlossenen CMK-Flow-Hauptwegs. Der öffentliche Subgraph bündelt finales Upscaling, Ergebnisvorschau und Projektspeicherung. Er gehört deshalb zur Produktrolle `Flow/Finish` und nicht zu einer allgemeinen Save- oder Baukastenkategorie.

Nach diesem Modul existiert im normalen Flow kein weiterer bildverarbeitender Schritt. Einzelne offene Save- und Upscale-Nodes bleiben unabhängig davon im Baukasten verfügbar.

`CMK Smart Upscaler -Pipe-`:

```text
IMAGE + LOG → IMAGE + LOG + diagnostic
```

`CMK Save Project Image -Pipe-`:

```text
IMAGE + LOG + Speicherparameter → FULLPATH
```

Speichern ist ein Endpunkt. Es darf keine Bildverarbeitung rückwärts erzwingen, wenn die vorherige Modulgrenze unverändert ist.

### 5.9 Video-Segmentierung

`CMK Split Video into Segments` ist der kanonische Einstieg für lange Videoquellen:

```text
VIDEO aus input/video/
+ MAX FRAMES 720P / MAX FRAMES 1080P
+ OVERLAP / VIDEO CODEC / VIDEO BITRATE / PRESET
    ↓
SEGMENTS + LOG + diagnostic
```

Verantwortung:

- wählt die Quelle ausschließlich aus dem ComfyUI-Verzeichnis `input/video/`; absolute freie Pfade gehören nicht zur öffentlichen Oberfläche;
- schreibt dauerhaft und quellengetrennt nach `output/video/segments/<video_name>/`;
- lädt das Video nicht als vollständigen Frame-Batch in den Arbeitsspeicher, sondern segmentiert direkt über ffmpeg;
- darf kein separates `ffprobe` voraussetzen: falls es nicht installiert ist, werden Metadaten über das von ComfyUI beziehungsweise `imageio_ffmpeg` bereitgestellte FFmpeg-Binary gelesen;
- bestimmt anhand der Quellhöhe, ob `MAX FRAMES 720P` oder `MAX FRAMES 1080P` gilt;
- berechnet daraus mit der Quell-FPS die Segmentdauer und hält mindestens 180 Frames sowie mindestens drei Sekunden ein;
- führt Audio intern mit AAC/192k und dokumentiert alle effektiven Parameter;
- erzeugt ein persistentes `segments.json`-Manifest und darf dieses nur wiederverwenden, wenn Quelle und sämtliche Segmentierungs-/Encodingparameter identisch sind und alle Segmentdateien existieren;
- gibt keinen separaten Verzeichnispfad aus. Der Speicherort ist Bestandteil von `SEGMENTS`, `LOG` und `diagnostic`.

`SEGMENTS` besitzt den proprietären Typ `CMK_VIDEO_SEGMENTS` und ist ein als immutable behandelter Arbeitskontext. Er enthält mindestens:

```python
{
    "type": "CMK_VIDEO_SEGMENTS",
    "source_path": str,
    "output_directory": str,
    "manifest_path": str,
    "segment_paths": tuple[str, ...],
    "segments": tuple[dict, ...],
    "width": int,
    "height": int,
    "fps": float,
    "frame_count": int,
    "duration": float,
    "target_frames": int,
    "real_frames": int,
    "segment_length": float,
    "overlap": float,
    "video_codec": str,
    "video_bitrate": str,
    "preset": str,
}
```

Nachfolgende Video-Nodes dürfen diesen Arbeitskontext lesen, aber nicht mutieren. `LOG` bleibt rein dokumentarisch und darf die Segmentverarbeitung nicht steuern.

## 6. Parallele Kaskaden: Problem und endgültige Lösung

### 6.1 Das beobachtete Problem

Eine rein serielle Kaskade erzeugt künstliche Abhängigkeiten:

```text
Detailer 1 → DETAILER/LOG → Detailer 2
```

Das Aktivieren von Detailer 2 fordert dadurch Detailer 1 erneut an.

Eine reine Parallelisierung beseitigt diese direkte Abhängigkeit, löst aber das vollständige Problem noch nicht:

```text
Detailer 1 ─┐
            ├→ CONCAT
Detailer 2 ─┘
```

Ein Merge-Knoten benötigt alle Eingänge. Wenn ComfyUI ein unverändertes Geschwisterergebnis nicht mehr im normalen Cache hält, wird dieser Zweig erneut ausgeführt.

### 6.2 Verbindliche Zweistufenlösung

CMK verwendet deshalb zwei unterschiedliche Cache-Ebenen:

```text
teurer Einzelzweig
    ↓
persistenter Branch Cache

parallele Branch-Ergebnisse
    ↓
SEGS/LOG Merge
    ↓
persistenter Module Boundary Cache
```

#### Branch Cache

Jede teure Smart-Detailer- und FaceProcess-Instanz besitzt einen eigenen persistenten Cache.

Der Fingerprint enthält:

- den eigenen unveränderlichen Arbeitskontext;
- das eigene authoritative Quellbild;
- die eigenen lokalen Parameter;
- die eigene Aktivierung.

Er enthält keine Geschwisterinstanz. Deshalb verändert das Aktivieren von Detailer 2 nicht den Fingerprint von Detailer 1.

Bei `CMK Smart Detailer -Pipe-` ist `output_image_proceed` ausdrücklich vom Rechen-Fingerprint ausgeschlossen, weil dieser Schalter nur die optionale Ausgabe sichtbar macht und nicht das Detailer-Ergebnis berechnet.

Persistente Branch-Caches speichern zusätzlich ein cache-stabiles internes Branch-Artefakt aus vollständig zusammengesetztem Branch-Bild und räumlichem Support der tatsächlich zugeordneten SEG-Crop-Regionen. Der Support darf niemals aus einzelnen Pixel-Differenzen zwischen Quelle und Ergebnis abgeleitet werden. `CMK SEGS CONCAT` verwendet dieses Artefakt bei Cache-Hits direkt. Der öffentliche Wert bleibt strukturell ein Impact-kompatibles `SEGS`; Ports und Kabel ändern sich nicht. Ein Cache-Hit muss pixelidentisch zu einer frischen Branch-Ausführung sein.

#### Module Boundary Cache

Nach `SEGS CONCAT` und `LOG CONCAT` materialisiert ein Boundary das vollständige Modulergebnis.

Ein Boundary-Hit ist nur gültig, wenn zusätzlich zum eigenen Prompt-Fingerprint die gespeicherte Dependency-Manifest-Version exakt mit den aktuell materialisierten Branch-Caches übereinstimmt. Jede Branch-Cache-Datei besitzt dafür einen kleinen Revisionsmarker. Wird ein einzelner Detailer-/FaceProcess-Zweig unter demselben Rechen-Fingerprint neu geschrieben, ändert sich dessen Revision; der Boundary fordert daraufhin `SEGS CONCAT` und `LOG CONCAT` erneut an und materialisiert erst danach ein neues vollständiges Modulergebnis. Ein veralteter Boundary darf niemals ein neu zusammengeführtes Ergebnis übergehen.

Branch- und Module-Boundary-Fingerprints sind instanzgebunden. Zwei formal identische parallele Node-Instanzen besitzen deshalb getrennte persistente Cache-Einträge.

Alle folgenden Verbraucher müssen diesen Boundary verwenden:

- öffentliche Modul-Ausgänge;
- interne Image Comparer;
- nachfolgende Module;
- Save-/Exportpfade.

Damit kann eine Änderung in FaceProcess nicht den gesamten Detailer erneut anfordern, und ein Save-/Comparer-Zweig kann keinen internen Compute-Knoten umgehen.

### 6.3 Cache-Speicherung

Cache-Dateien liegen ausschließlich unter:

```text
ComfyUI/temp/cmk/
```

Aktuelle Implementierung:

| Ebene | gespeicherte Nutzdaten | Format | zusätzliche Laufzeitdaten |
|---|---|---|---|
| Refiner Boundary | First-Pass-Bild, Refiner-Bild, LOG | `safetensors` + JSON | `MODEL` und `PROCESS` im aktuellen ComfyUI-Prozess |
| Smart-Detailer-Branch | SEGS, optionales Bild, Diagnostic, LOG BLOCK | lokales Pickle-Bundle + Revisionsmarker | keine öffentliche API |
| FaceProcess-Branch | Bild/SEGS, Status, Diagnostic, LOG BLOCK | lokales Pickle-Bundle + Revisionsmarker | keine öffentliche API |
| Detailer Boundary | Modulbild, LOG, Branch-Dependency-Manifest | `safetensors` + JSON + internes Manifest | `MODEL` und `PROCESS` im aktuellen ComfyUI-Prozess |
| FaceProcess Boundary | Modulbild, LOG, Branch-Dependency-Manifest | `safetensors` + JSON + internes Manifest | keine weitere Leitung |
| FaceSwap Boundary | PROCESS, Modulbild, LOG | `safetensors` + JSON | `MODEL` wird read-only aktuell durchgereicht |

Diese Caches sind Beschleuniger, kein portables Projektformat. Nach einem Neustart, einer Codeänderung, einer geänderten Upstream-Konfiguration oder einer Cache-Bereinigung ist ein erster `MISS → STORED` erwartbar.

### 6.4 Warum nicht anders

| Verworfene Lösung | Grund der Ablehnung |
|---|---|
| serielle DETAILER-/FACE-/LOG-Durchleitung | macht frühere Instanzen zu formalen Vorgängern späterer Instanzen |
| nur sternförmige Parallelverkabelung | Merge-Nodes können unveränderte Zweige trotzdem erneut anfordern |
| vollständiges `LOG` an jeder Execute-Instanz | erzeugt Verarbeitungsketten und erlaubt Fehlverkabelungen |
| `STRING` statt `CMK_LOG_BLOCK` | ein beliebiger Text könnte an die Log-Zusammenführung angeschlossen werden; Logblöcke wären nicht typgeschützt |
| unverändertes `IMAGE` durch Execute-Nodes | fachfremder Transport, zusätzliche Invalidierung und Umgehung des Merge-Pfads |
| allein auf den normalen ComfyUI-Cache vertrauen | nicht hinreichend stabil bei Merge-Pfaden, Cache-Eviction und begrenztem Speicher |
| Comparer direkt an Compute-Ausgang | erzeugt einen zweiten, ungeschützten Ausführungspfad |
| sichtbare Sternverkabelung im Hauptworkflow | widerspricht der `-Pipe-`-Produktphilosophie; interne Parallelität darf nicht zur Anwenderaufgabe werden |
| Bilder im öffentlichen PROCESS verstecken | unklare Bildautorität, schwer nachvollziehbare Änderungen und unnötige Kopplung |
| Refiner mit externem First-Pass-IMAGE speisen | redundant; das korrekte First-Pass-Bild wird aus dem unverarbeiteten Sampler-Latent dekodiert |

## 7. LOG-Vertrag

`LOG` dokumentiert. Es verarbeitet nicht.

Ein vollständiger `CMK_LOG_PIPE` wird nur an Modulgrenzen weitergegeben. Parallele Execute-Instanzen geben ausschließlich `CMK_LOG_BLOCK` aus.

Verbindlicher Ablauf:

```text
Basis-LOG
    ├─ LOG BLOCK 1
    ├─ LOG BLOCK 2
    └─ LOG BLOCK N
          ↓
      CMK LOG CONCAT
          ↓
      neues vollständiges LOG
```

`CMK_LOG_BLOCK` kann technisch nicht an einen normalen `CMK_LOG_PIPE`-Eingang angeschlossen werden. Dadurch ist die unerwünschte `LOG → Execute → LOG → Execute`-Kette ausgeschlossen.

Logs enthalten keine Bildtensoren, Masken, Modelle oder Cache-Objekte. Sie speichern nur strukturierte, menschenlesbare Metadaten.

## 8. Aktivierung

Optionale Module unterscheiden zwei Ebenen:

```text
GLOBAL ENABLE → Teilnahme des gesamten Moduls
LOCAL ENABLE  → Ausführung einer einzelnen parallelen Instanz
```

Die effektive Aktivierung ist:

```text
GLOBAL ENABLE AND LOCAL ENABLE
```

Ein Funktions- oder Moduswähler beschreibt ausschließlich eine reale Verarbeitung. Er ersetzt keinen Enable-Schalter.

Für dynamische Betriebsarten-UI gilt zusätzlich: Backend-Widgets bleiben dauerhaft in kanonischer Reihenfolge in `node.widgets`. Inaktive Widgets dürfen nur optisch ausgeblendet, niemals entfernt oder umsortiert werden, weil ComfyUI gespeicherte `widgets_values` positionsgebunden wiederherstellt.

Für `CMK FaceProcess -Pipe-` gilt:

```text
PROCESS MODE: restore | detailer
REFINE MODE: Off | Detail | Sharpen | Smooth
```

`Off` im Refine-Modus ist zulässig, weil Refine eine eigenständige optionale zweite Stufe ist. Die gesamte FaceProcess-Instanz wird über `LOCAL ENABLE` deaktiviert.

## 9. Diagnostik und Preview

`diagnostic` ist ein lokaler, passiver Beobachtungsausgang. Er darf weder Verarbeitung steuern noch einen Modulpfad ersetzen.

`CMK Preview Board` besitzt dynamische Eingänge:

```text
DIAGNOSTIC 1 ... N
```

Es bleibt genau ein leerer Folgeeingang sichtbar. Dieselbe Regel gilt für:

```text
CMK SEGS CONCAT → SEGS 1 ... N
CMK LOG CONCAT  → LOG BLOCK 1 ... N
```

Preview- und Comparer-Nodes innerhalb eines geschlossenen Moduls müssen hinter dem jeweiligen Boundary angeschlossen sein, sofern sie ein verarbeitetes Modulergebnis anzeigen.

## 10. UI- und Benennungsvertrag

- Sichtbare Standard-Schnittstellen verwenden kurze Großbuchstaben: `MODEL`, `PROCESS`, `IMAGE`, `LOG`, `SAMPLED`, `DETAILER`, `FACE`.
- Dynamische Eingänge werden fortlaufend und lesbar bezeichnet: `SEGS 1`, `SEGS 2`, `LOG BLOCK 1`, `DIAGNOSTIC 1`.
- Technische Advanced-Parameter dürfen ihre präzisen internen Namen behalten.
- Nodes heißen nach ihrer Funktion, nicht nach ihrer Datei.
- `CONCAT` ist die einheitliche sichtbare Schreibweise.
- Der neutrale sichtbare Name lautet `CMK Boundary Cache`; seine konkrete Modulrolle ergibt sich aus dem umgebenden Subgraphen.
- Ein Refiner-Modul wird nicht als Upscaler bezeichnet, wenn kein Upscale stattfindet.

## 11. Standalone-Kompatibilität

Das geschlossene Ökosystem darf keine unnötige Abhängigkeit vom vollständigen Referenzworkflow erzeugen.

Detailer und FaceProcess müssen weiterhin aus den kanonischen Quellen direkt aufgebaut werden können:

```text
CMK Checkpoint VAE Loader -Pipe- → MODEL
CMK Load Image -Pipe-            → PROCESS + IMAGE + LOG
CMK Image Load and Resize -Pipe- → PROCESS + IMAGE + LOG + diagnostic
```

Daraus können `CMK Detailer Prepare -Pipe-` beziehungsweise `CMK FaceProcess Prepare -Pipe-` gespeist werden.

Die proprietären Arbeits-Pipes schützen die Execute-Schnittstellen, ohne Standalone-Verwendung zu verhindern.

## 12. Nicht verhandelbare Invarianten

Eine Änderung ist architektonisch unzulässig, wenn sie eine der folgenden Regeln verletzt:

1. `MODEL` wird von einem Modul mutiert.
2. Das authoritative Bild wird nur versteckt in `PROCESS` transportiert.
3. `LOG` oder `LOG BLOCK` steuert eine Bildberechnung.
4. Eine Execute-Instanz schleift `DETAILER`, `FACE` oder ein vollständiges `LOG` zur nächsten Instanz durch.
5. Ein interner Comparer oder öffentlicher Ausgang umgeht den zugehörigen Boundary.
6. Eine Änderung an einem späteren parallelen Zweig führt zu einer tatsächlichen Neuberechnung eines unveränderten früheren Zweigs.
7. Ein nachfolgendes Modul zieht ein unverändertes vorheriges Modul erneut durch dessen Compute-Knoten.
8. Interne Parallelität wird als vermeidbare Sternverkabelung in den Hauptworkflow verlagert.
9. Ein `-Pipe-`-Knoten verlangt vom Anwender Wissen über seine interne Implementierungsreihenfolge.
10. Dokumentation beschreibt einen historischen Zwischenstand als aktuellen Vertrag.

## 13. Verbindliche Regressionstests

Nach Änderungen an Modulgrenzen oder Cache-Logik sind mindestens diese Tests durchzuführen:

```text
Detailer 1 aktivieren
→ kein Sampler-/Refiner-Neustart

Detailer 2 aktivieren
→ kein Sampler-/Refiner-Neustart
→ Detailer 1 höchstens CACHE HIT, keine Verarbeitung

FaceProcess 1 aktivieren
→ Detailer-Modul höchstens Boundary/Branch CACHE HIT
→ keine Detailer-Verarbeitung

FaceProcess 2 aktivieren
→ FaceProcess 1 höchstens CACHE HIT
→ keine Verarbeitung von FaceProcess 1

Detector oder lokale Parameter einer Instanz ändern
→ nur diese Instanz berechnet neu
→ Boundary erkennt die neue Branch-Revision
→ SEGS/LOG CONCAT wird erneut materialisiert
→ unveränderte Geschwister bleiben Branch-CACHE-HIT

Eine Instanz unter identischem Fingerprint neu materialisieren
→ ihr Revisionsmarker ändert sich
→ ein bestehender Boundary-Hit wird verworfen
→ das öffentliche Bild enthält erneut alle aktiven Zweige

Save oder internen Comparer anfordern
→ kein direkter Rückgriff auf Compute-Nodes hinter einer gültigen Boundary
```

## 14. Dokumentationspflege

- `ARCHITECTURE.md` definiert ausschließlich den aktuellen Vertrag.
- `CMK_Design_Guidelines.md` definiert ausschließlich Darstellung, Benennung und UI-Verhalten.
- `README.md` erklärt Installation und Nutzung, ohne Architektur zu duplizieren.
- `CHANGELOG.md` dokumentiert zeitliche Änderungen. Historische Einträge sind niemals eine aktuelle API-Definition.
- Überholte Zwischenstände werden nicht parallel als vermeintlich gültige Alternativen dokumentiert.
### CMK Video Segments Manifest

`CMK Split Video into Segments` schreibt `segments.json` als persistenten Vertrag. Ein vorhandener Split wird ausschließlich wiederverwendet, wenn Quelle und Split-/Encoding-Einstellungen identisch sind und Manifest, Segmentanzahl, Segmentreihenfolge, Zeitbereiche, Pfade sowie nichtleere Dateien vollständig konsistent sind. Öffentliche Statuswerte: `REUSED`, `SPLIT`, `INVALIDATED`.



## CMK Merge and Save Video

`CMK Merge and Save Video` nimmt ausschließlich den proprietären Transport `CMK_VIDEO_SEGMENTS` entgegen und erzeugt daraus einen persistenten `CMK_VIDEO`-Kontext. Der Merge schreibt nach `output/video/merged/<video_name>/<video_name>_merged.mp4`. Überlappungen werden anhand der im Segmentkontext gespeicherten Start-/Endzeiten am Beginn jedes Folgefragments entfernt; Segmentreihenfolge und Zeitachse bleiben damit authoritative.

Öffentliche Schnittstelle:

- Eingänge: `SEGMENTS`, `LOG`, `VIDEO CODEC`, `VIDEO BITRATE`, `PRESET`
- Ausgänge: `VIDEO`, `LOG`, `diagnostic`

Ein persistentes `merged.json` validiert Segmentpfade, Dateigröße, Änderungszeit, Zeitbereiche und Encodingparameter. Statuswerte sind `MERGED`, `REUSED` und `INVALIDATED`. Der Merger darf später auch von Processing-Nodes erzeugte `CMK_VIDEO_SEGMENTS` verwenden; er ist nicht an den ursprünglichen Splitterordner gebunden.

## CMK Video Compare

`CMK Video Compare` ist die objektive Roundtrip-Validierung für `VIDEO → SEGMENTS → MERGE → VIDEO`.

Eingänge:

- `SEGMENTS` (`CMK_VIDEO_SEGMENTS`) liefert Quelle, Segmentgrenzen und Overlap-Metadaten.
- `VIDEO` (`CMK_VIDEO`) liefert das zusammengeführte Ergebnis.
- `LOG` bleibt rein dokumentarisch.

Ausgänge:

- `METRICS` (`CMK_VIDEO_METRICS`) enthält immutable Vergleichswerte und den Gesamtstatus `PASS`, `WARNING` oder `FAIL`.
- `LOG` dokumentiert Auflösung, FPS, Dauer, Framezahl und Audiovergleich.
- `diagnostic` enthält zusätzlich die vollständige Segment-/Trim-Timeline.

Die Node steuert keine Verarbeitung und verändert weder Segmente noch Video.
## Native Video-Previews

`CMK Split Video into Segments` und `CMK Merge and Save Video` verwenden den eingebetteten CMK-HTML5-Player. Der Player steht unterhalb aller nativen Parameter. Seine Position darf nicht nach oben verschoben werden.

- Split zeigt das ausgewählte Quellvideo.
- Merge zeigt das persistente zusammengeführte Ergebnis.
- `CMK Merge and Save Video` übernimmt Merge, Player und optionales Publish/Save in einer Node. `SAVE ENABLED = ON` kopiert das bereits gemergte Cache-Ergebnis ohne erneutes Encoding. Separate Nodes `CMK Merge Video Segments` und `CMK Video Preview` existieren nicht.
- `CMK Video Compare` behält seine spezialisierte synchrone Doppelplayer-Oberfläche.
- Die Split-Videoauswahl ersetzt ausschließlich das einzelne `VIDEO`-Widget durch eine gleich hohe Auswahl-/Upload-Zeile; alle übrigen Widgets bleiben native ComfyUI-Widgets.


### 5.10 CMK FaceSwap Video

`CMK FaceSwap Video` ist die Processing-Node zwischen `CMK Split Video into Segments` und `CMK Merge and Save Video`.

Öffentliche Schnittstelle:

```text
CMK_VIDEO_SEGMENTS + IMAGE SOURCE + CMK_LOG_PIPE
    -> CMK FaceSwap Video
    -> CMK_VIDEO_SEGMENTS + CMK_LOG_PIPE + CMK_DIAGNOSTIC
```

Verbindliche Regeln:

- Die persistenten Segmentdateien des Split-Manifests sind autoritatives und unveränderliches Ausgangsmaterial.
- Die Node segmentiert niemals erneut und überschreibt niemals Quellsegmente.
- Verarbeitung erfolgt segmentweise und innerhalb eines Segments frameweise; das vollständige Video wird nicht in den Arbeitsspeicher geladen.
- Swap-Ausgaben und Fortschrittsmanifest werden persistent außerhalb von `temp/` gespeichert.
- Der Cache ist segmentweise. Identische Quelle, Source-Face, Modelle, Swap-/Trackingparameter und Engine-/Schema-Version dürfen valide Segmentausgaben wiederverwenden.
- Fehler dürfen nicht durch ein unbemerktes Originalsegment ersetzt werden. Erfolgreich abgeschlossene Segmente bleiben nach Fehlern und Neustarts erhalten.
- Frames ohne erkennbares oder plausibel zugeordnetes Zielgesicht bleiben unverändert, werden jedoch ausdrücklich gezählt und diagnostiziert.
- Die Segmentstruktur und Timeline-Metadaten bleiben erhalten; verarbeitete Segmentpfade zeigen auf neue persistente Dateien.
- Audio wird nicht in Python verarbeitet. Es wird aus dem Quellsegment unverändert gemuxt, sofern der Container dies erlaubt.
- `LOG` dokumentiert ausschließlich und beeinflusst weder Verarbeitung noch Cacheentscheidungen.

`SEGMENT SELECTION` ist ein Standardparameter mit den Modi:

- `Full Path (Batch)`: alle Segmente in Originalreihenfolge.
- `Randomize`: genau ein zufällig ausgewähltes Segment als Kontrollclip.
- `Last Used Segment`: ausschließlich das letzte Segment als Kontrollclip.

Bei `Randomize` und `Last Used Segment` enthält der ausgegebene `CMK_VIDEO_SEGMENTS`-Kontext ausschließlich das ausgewählte und erfolgreich verarbeitete beziehungsweise valide wiederverwendete Segment. Unbearbeitete Originalsegmente werden nicht ergänzt.

Die minimale zeitliche Zielgesichtszuordnung verwendet Bounding-Box-Kontinuität, Position und – sofern verfügbar – InsightFace-Embedding-Ähnlichkeit. Ein kurzzeitig verlorenes Ziel wird nicht durch ein beliebiges anderes Gesicht ersetzt. Nach Überschreiten des Missing-Fensters erfolgt eine deterministische Neuinitialisierung anhand von `TARGET FACE`.

### Video-Compare-Referenz nach Processing-Nodes

`CMK_VIDEO_SEGMENTS` darf zusätzlich `compare_source_path` enthalten.

- `source_path` bleibt die unveränderte Referenz auf das vollständige Ursprungsvideo.
- `compare_source_path` bezeichnet das originale Material, das zeitlich exakt zur aktuell aktiven Segmentausgabe gehört.
- `CMK Video Compare` verwendet `compare_source_path`, wenn vorhanden, andernfalls `source_path`.
- `Full Path (Batch)` verweist auf das vollständige Ursprungsvideo.
- `Randomize` und `Last Used Segment` verweisen auf das originale, unbearbeitete ausgewählte Split-Segment.
- Die Information wird im bestehenden `CMK_VIDEO_SEGMENTS`-Transport weitergegeben; ein zusätzliches öffentliches Kabel ist nicht erforderlich.

#### Preview-Auflösung für Video Compare

Player-Deskriptoren müssen aus dem tatsächlichen ComfyUI-Speicherort der jeweiligen Datei abgeleitet werden. Ein ausgewähltes originales Split-Segment liegt persistent unter `output/` und darf nicht als `input` adressiert werden. `CMK Video Compare` berücksichtigt außerdem die Dateizustände von Vergleichsquelle und Ergebnis in seinem Cache-Schlüssel, damit die Player-Informationen nach Neustart oder Workflow-Neuladen erneut an das Frontend übertragen werden.

#### Ausführung von CMK Video Compare

`CMK Video Compare` ist ein sichtbarer Workflow-Endpunkt und muss als `OUTPUT_NODE = True` registriert sein. Die Node wird dadurch auch ausgeführt, wenn ihre Ausgänge nicht weiterverkabelt sind. Nur bei tatsächlicher Ausführung werden Metriken, Diagnostic und die beiden Player-Deskriptoren an das Frontend übertragen.

### Segmentauswahl im Video-FaceSwap

`SEGMENT SELECTION` besitzt folgende verbindliche Semantik:

- `Full Path (Batch)`: verarbeitet alle Segmente.
- `Randomize`: wählt bei jedem gestarteten Workflow neu ein Segment aus. Bei mehr als einem Segment darf nicht unmittelbar erneut derselbe Segmentindex gewählt werden.
- `Last Used Segment`: verwendet den zuletzt durch `Randomize` ausgewählten Segmentindex erneut. Dadurch können Parameteränderungen direkt am identischen Testsegment verglichen werden.
- Der zuletzt verwendete Segmentindex wird persistent pro Quellvideo beziehungsweise Split-Manifest gespeichert.
- `Randomize` darf nicht vom normalen ComfyUI-Ausführungscache festgehalten werden. Der persistente Swap-Dateicache bleibt davon unabhängig und darf bereits berechnete Segmentausgaben wiederverwenden.

#### Cache-Signatur und Segmentanzeige

`SEGMENT SELECTION` bestimmt ausschließlich, welche vorhandenen Quellsegmente ausgegeben beziehungsweise geprüft werden. Der Auswahlmodus ist kein swap-relevanter Bildparameter und darf deshalb nicht Bestandteil der persistenten Swap-Cache-Signatur sein. Ein durch `Randomize` erzeugtes Segment muss unter `Last Used Segment` mit identischen Swap-Einstellungen direkt wiederverwendet werden.

Segmentindizes bleiben intern nullbasiert und unverändert. Anwenderseitige Ausgaben in Konsole, LOG und Diagnostic sind dagegen einsbasiert und werden als `aktuelles Segment / Gesamtzahl` dargestellt.

### `CMK FaceSwap Video Loader`

`CMK FaceSwap Video Loader` ist eine ergänzende geschlossene Einstiegs-Node für den Video-FaceSwap. Die universelle Node `CMK Split Video into Segments` bleibt unverändert und ausschließlich für Videoauswahl und Segmentierung verantwortlich.

Öffentliche Ausgänge:

```text
SEGMENTS      CMK_VIDEO_SEGMENTS
IMAGE SOURCE  IMAGE
LOG           CMK_LOG_PIPE
diagnostic    CMK_DIAGNOSTIC
```

Die Node verwendet intern unverändert dieselbe Split-Engine, dasselbe Split-Manifest und denselben persistenten Segmentcache wie `CMK Split Video into Segments`. Es existiert keine zweite Segmentierungslogik.

`IMAGE SOURCE` bleibt ein eigenständiger nativer Bildtransport und wird nicht Bestandteil von `CMK_VIDEO_SEGMENTS`. Eine Änderung des Source-Bildes darf keine erneute Segmentierung auslösen, invalidiert aber korrekt den nachgelagerten FaceSwap.

UI-Regel:

- Segmentierungs- und Encodingparameter stehen oben.
- Das gemeinsame Medien-Panel steht als letzter Block unten.
- `SOURCE VIDEO` und `SOURCE IMAGE` werden in zwei gleich hohen Karten nebeneinander dargestellt.
- Die Node darf keinen zusätzlichen separaten Video-Player außerhalb dieses Panels erzeugen.

### Gemeinsame FaceSwap-Engine: Enhancer und Pasteback

Alle öffentlichen Swap-Nodes verwenden denselben zentralen Swap-Pfad:

- `CMK FaceSwap Image`
- `CMK FaceSwap Image -Pipe-`
- `CMK FaceSwap Video`

Ein Fehler des ausgewählten Enhancers darf den Swap- oder Pasteback-Pfad nicht wechseln. Enhancer, aligned Swap und Pasteback sind getrennte Fehlergrenzen. Fehler werden klar ausgelöst; ein als `CodeFormer` gekennzeichnetes Ergebnis darf niemals ohne funktionierenden CodeFormer erzeugt werden.

`CodeFormer` wird nur in den Enhancer-Dropdowns angeboten, wenn sowohl die Python-Architektur als auch das Modell verfügbar sind. `Off` und `GPEN` bleiben unabhängig davon verfügbar.

Der reguläre Pasteback verwendet die exakte von INSwapper zurückgegebene affine Matrix und einen erodierten, weichgezeichneten Crop-Maskenpfad nach dem Verhalten des nativen InsightFace-Pastebacks. Der frühere Pfad, der zunächst das vollständige quadratische Alignment einfügte und anschließend aus Pixeländerungen eine Gesichtsmaske rekonstruierte, ist für Swap-Ausgaben nicht mehr autoritativ.

Da sich die erzeugten Pixel aller Video-Swap-Modi ändern können, wird die Video-Swap-Engine-Version erhöht und der bisherige Swap-Cache gezielt invalidiert. Die ursprüngliche Video-Segmentierung bleibt davon unberührt.

#### Enhancer-Dimensionsvertrag

Jeder Face-Enhancer darf intern mit einer anderen Modellauflösung arbeiten, muss
aber vor der Rückgabe exakt auf Breite und Höhe des von INSwapper gelieferten
aligned Face-Crops zurückkehren. Die affine Rücktransformationsmatrix ist an
diese ursprüngliche Crop-Größe gebunden.

Die gemeinsame Swap-Engine prüft diesen Vertrag zusätzlich defensiv. Dadurch
können GPEN- oder CodeFormer-Ausgaben mit Modellauflösung 512×512 nicht mit
einer für 128×128 erzeugten INSwapper-Matrix zurückprojiziert werden.

Da diese Korrektur die Video-Swap-Pixel verändert, wird ausschließlich der
persistente Video-Swap-Cache invalidiert. Die ursprüngliche Segmentierung und
ihre Manifeste bleiben erhalten.


#### FACE SWAP OFF/ON

`CMK FaceSwap Video` besitzt einen öffentlichen Boolean-Schalter `FACE SWAP` als obersten Standardparameter.

- `ON`: normale segmentweise FaceSwap-Verarbeitung.
- `OFF`: vollständiger Bypass. Die Node gibt den eingehenden `CMK_VIDEO_SEGMENTS`-Kontext unverändert weiter und dokumentiert den Bypass nur in LOG und Diagnostic.

Der Bypass erzeugt keine neuen Swap-Segmente, verändert keinen bestehenden Swap-Cache und lässt die übrige Video-Kette funktionsfähig.

#### Affine Padding-Maske

Der gemeinsame FaceSwap-Pasteback darf die gültige Einfügefläche nicht aus dem vollständigen aligned Crop ableiten. Bei gedrehten oder randnahen Gesichtern enthält dieser Crop geometrisches schwarzes Padding aus Bereichen außerhalb des Zielbildes.

Die gültige Maske wird deshalb aus einer vollständig weißen Maske des ursprünglichen Zielbildes mit derselben INSwapper-Affine in den aligned Raum transformiert. Nur tatsächlich aus dem Zielbild stammende Pixel bleiben gültig; ein zusätzlicher innerer Sicherheitsrand wird vor der Rückprojektion entfernt. Die Prüfung ist geometrisch und unabhängig von der Bildhelligkeit, sodass dunkle Haare, Kleidung und Hintergründe nicht fälschlich ausgeschlossen werden.

Die Änderung gilt zentral für `CMK FaceSwap Image`, `CMK FaceSwap Image -Pipe-` und `CMK FaceSwap Video`. Der Video-Swap-Cache wird durch Erhöhung der Engine-Version invalidiert; Split-Segmente und Split-Manifeste bleiben unverändert.

#### Begrenzung von `crop_factor`

Der FaceSwap-Parameter `crop_factor` besitzt für alle öffentlichen Swap-Nodes den verbindlichen Bereich `1.0–3.0` bei Default `1.5`.

Die Begrenzung gilt für:

- `CMK FaceSwap Image`
- `CMK FaceSwap Image -Pipe-`
- `CMK FaceSwap Video`

Das UI verhindert höhere Neueingaben. Werte aus älteren Workflows werden bei der Ausführung defensiv auf `3.0` begrenzt. Die gemeinsame Swap-Engine und der Pasteback validieren denselben Bereich zusätzlich zentral.

Die Begrenzung betrifft ausschließlich den FaceSwap-Pasteback-Parameter. Gleichnamige Detection- oder Detailer-Parameter besitzen eine andere Verantwortlichkeit und werden dadurch nicht verändert.

#### Zentrale Begrenzung des Detailer-Denoise

Alle CMK-Detailerpfade verwenden für `denoise` beziehungsweise `detail_denoise` den verbindlichen Bereich `0.0001–0.5`. Der Default bleibt `0.5`.

Die Begrenzung gilt für:

- `CMK Smart Detailer`
- `CMK Smart Detailer -Pipe-`
- den Detailer-Modus von `CMK FaceProcess`
- Face-/Detailer-Pipe-Transportwerte

Das UI verhindert höhere Neueingaben. Werte aus älteren Workflows oder bestehenden Pipes werden zentral vor der Verarbeitung auf `0.5` begrenzt. Sampler-Denoise außerhalb von Detailer-Verantwortlichkeiten wird nicht verändert.

## Verbindliche Parametergrenzen: fachlich sinnvoller Arbeitsbereich vor theoretischem Wertebereich

CMK begrenzt öffentliche Parameter nicht nach dem maximal technisch akzeptierten Wertebereich einer zugrunde liegenden ComfyUI- oder Backend-Funktion, sondern nach dem für die jeweilige Node-Verantwortlichkeit fachlich sinnvollen und kontrollierbaren Arbeitsbereich.

Die Gründe sind:

- Extremwerte können formal gültig sein, aber die eigentliche Funktion der Node faktisch verlassen.
- Überbreite Wertebereiche erschweren die präzise Einstellung des praktisch relevanten Bereichs im UI.
- In gespeicherten oder kopierten Workflows können versehentlich extreme Werte unbemerkt erhalten bleiben.
- Ein öffentlicher CMK-Parameter soll einen erwartbaren, fehlertoleranten Arbeitsbereich anbieten und keine technisch möglichen, aber für die konkrete Aufgabe destruktiven Werte propagieren.
- Die UI-Begrenzung allein reicht nicht aus. Verbindliche Grenzen müssen zusätzlich an der zentralen Engine- beziehungsweise Ausführungsgrenze validiert werden, damit alte Workflows und programmatisch übergebene Werte denselben Regeln folgen.

### FaceSwap: `crop_factor`

Für den FaceSwap gilt verbindlich:

```text
crop_factor: 1.0–3.0
Default:     1.5
```

Diese Grenze gilt für alle öffentlichen Swap-Pfade, insbesondere:

- `CMK FaceSwap Image`
- `CMK FaceSwap Image -Pipe-`
- `CMK FaceSwap Video`

Hintergrund: Ein extrem hoher `crop_factor` kann den wirksamen Pasteback-Bereich so weit vergrößern, dass ungültige Rand- oder Padding-Bereiche des aligned Face-Crops in die Rückprojektion gelangen. Im praktischen Fehlerfall war in einer Workflow-Kopie versehentlich `crop_factor = 10` gespeichert; dies führte zu einem deutlich sichtbaren, mit dem Gesicht rotierten schwarzen Crop-Rahmen.

Daraus folgt verbindlich:

- Das UI bietet maximal `3.0` an.
- Die gemeinsame Swap-Engine begrenzt eingehende Werte defensiv auf `1.0–3.0`.
- Der Pasteback validiert denselben Bereich nochmals an seiner Ausführungsgrenze.
- Alte Workflows mit höheren gespeicherten Werten bleiben ladbar; bei der Ausführung wird der Wert auf `3.0` begrenzt.
- Die Begrenzung ist zentral und darf nicht in einzelnen Swap-Nodes unterschiedlich implementiert werden.

### Detailer: `denoise`

Für Detailer-Verarbeitung gilt verbindlich:

```text
denoise: 0.0001–0.5
```

Die Begrenzung gilt zentral für alle CMK-Pfade, die semantisch als Detailer arbeiten, insbesondere Smart-Detailer-, Face-Detailer- und entsprechende Pipe-Transportwerte.

Hintergrund: Der theoretische ComfyUI-Wertebereich bis `1.0` ist für die CMK-Detailer-Verantwortlichkeit nicht sinnvoll. Bereits Werte um `0.5` können eine weitgehende Neuinterpretation des erkannten Bereichs bewirken; der typische reale Arbeitsbereich liegt deutlich darunter und reicht häufig nur bis ungefähr `0.3`. Ein UI-Bereich bis `1.0` verschlechtert daher die Feinjustierung des tatsächlich relevanten Bereichs und suggeriert einen für Detailer fachlich nicht vorgesehenen Arbeitsbereich.

Daraus folgt verbindlich:

- Öffentliche Detailer-UIs bieten maximal `0.5` an.
- Die zentrale Detailer-Engine beziehungsweise gemeinsame Detailer-Grenzlogik begrenzt eingehende Werte defensiv auf maximal `0.5`.
- Alte Workflows und Pipe-Werte über `0.5` bleiben technisch verarbeitbar, werden bei der Ausführung jedoch auf `0.5` begrenzt.
- Die Grenze gilt nur für Detailer-Verantwortlichkeiten. Sie darf nicht pauschal auf normale Sampler- oder andere Denoise-Parameter übertragen werden.
- Neue Detailer-Nodes müssen dieselbe zentrale Grenzdefinition verwenden und dürfen keinen abweichenden lokalen Maximalwert einführen.

### Architekturregel für zukünftige Parameterbegrenzungen

Wenn ein technisch möglicher Wertebereich deutlich größer ist als der kontrollierbare Arbeitsbereich einer CMK-Node, ist der öffentliche Wertebereich bewusst auf den fachlich sinnvollen Bereich zu begrenzen.

Eine solche Begrenzung muss:

1. in der verbindlichen Architektur dokumentiert sein,
2. im UI die präzise Bedienung des realen Arbeitsbereichs unterstützen,
3. zentral an der Engine- oder Ausführungsgrenze abgesichert sein,
4. alte Workflow-Werte defensiv behandeln,
5. auf die konkrete Node-Verantwortlichkeit beschränkt bleiben.

Theoretische Backend-Maximalwerte sind kein ausreichender Grund, sie unverändert als öffentliche CMK-Parametergrenzen zu übernehmen.

### Parallelbetrieb von `CMK FaceSwap Image -Pipe-`

`CMK FaceSwap Image -Pipe-` unterstützt den direkten Einzelbetrieb und den parallelen Modulbetrieb über `CMK SEGS CONCAT`.

Öffentliche Standard-Schalter:

```text
GLOBAL ENABLE
ENABLE
```

Die wirksame Freigabe lautet:

```text
effective_enable = GLOBAL ENABLE AND ENABLE
```

`GLOBAL ENABLE` ist die modulweite Freigabe und darf aus der Subgraph-/Moduloberfläche herausgeführt werden. `ENABLE` bleibt die lokale Freigabe der einzelnen FaceSwap-Instanz. Bei deaktivierter wirksamer Freigabe werden Source-Laden, Gesichtserkennung, INSwapper, Enhancer und Pasteback vollständig übersprungen.

Öffentliche Ausgänge:

```text
PROCESS
IMAGE
SEGS PROCESSED
LOG
diagnostic
```

`IMAGE` bleibt das vollständige Einzelergebnis. `SEGS PROCESSED` ist ein cache-stabiler CMK-SEGS-Branch aus:

- dem vollständigen Branch-Ergebnisbild,
- der tatsächlich vom gemeinsamen FaceSwap-Pasteback verwendeten räumlichen Alpha-Unterstützung,
- der Signatur des unveränderten autoritativen Zielbildes.

Mehrere parallele FaceSwap-Instanzen müssen dasselbe unveränderte `IMAGE TARGET` erhalten. Ihre `SEGS PROCESSED`-Ausgänge werden über `CMK SEGS CONCAT` auf dieses autoritative Zielbild zusammengeführt. Die Reihenfolge der SEGS-Eingänge definiert bei echten Überlappungen die Priorität.

Bei deaktivierter Instanz wird ein gültiges leeres SEGS ausgegeben. Die Parallelverkabelung bleibt dadurch unverändert.

Die Pasteback-Maske wird zentral von der gemeinsamen Swap-Engine bereitgestellt. Eine Rekonstruktion über Pixel-Differenzen ist unzulässig.

#### LOG-Ausgang im parallelen FaceSwap-Modul

`CMK FaceSwap Image -Pipe-` behält den Eingang `LOG` als Dokumentationskontext, gibt jedoch bewusst **keine durchgeschleifte `CMK_LOG_PIPE`** aus.

Der öffentliche Ausgang lautet:

```text
LOG BLOCK    CMK_LOG_BLOCK
```

Jede parallele FaceSwap-Instanz erzeugt ausschließlich ihren eigenen serialisierten Dokumentationsblock. Mehrere `LOG BLOCK`-Ausgänge werden über `CMK LOG CONCAT` gesammelt und dort gemeinsam mit der autoritativen Haupt-LOG-Pipe wieder zusammengeführt.

Verbindlicher Parallelaufbau:

```text
FaceSwap A.LOG BLOCK ─┐
FaceSwap B.LOG BLOCK ─┼─> CMK LOG CONCAT
FaceSwap C.LOG BLOCK ─┘
Haupt-LOG ───────────────> CMK LOG CONCAT.LOG
```

Damit wird verhindert, dass mehrere parallele Branches jeweils die vollständige eingehende LOG-Kette duplizieren oder unabhängig weiterführen. `LOG` dokumentiert weiterhin ausschließlich und beeinflusst weder Swap-Ausführung noch SEGS-Zusammenführung.

### Globaler Early-Bypass von `CMK FaceProcess Prepare -Pipe-`

`face_global_enable = OFF` ist eine harte Modulgrenze und wird bereits in
`CMK FaceProcess Prepare -Pipe-` ausgewertet.

Bei globalem OFF erzeugt der Prepare-Pfad ausschließlich eine minimale FACE-Pipe
mit dem autoritativen Eingangsbild und dem deaktivierten Status. Vollständig
übersprungen werden:

- MODEL-Anforderung beziehungsweise Modellinitialisierung,
- PROCESS-Anforderung,
- SAM-Modell,
- LoRA- und CLIP-Verarbeitung,
- PAG, Sampling und FreeU,
- Prompt-Encoding und Conditioning.

MODEL und PROCESS sind lazy Eingänge. Bei globalem OFF fordert der Prepare-Node
nur IMAGE und LOG an. Die nachgelagerten FaceProcess-Branches erhalten eine
gültige leichte FACE-Pipe und liefern sofort Passthrough-Bild, leere SEGS und
ihren LOG BLOCK.

```text
effective_enable = face_global_enable AND enable
```

Ein global deaktiviertes Face-Modul darf keine rechenintensiven Ressourcen
initialisieren, auch wenn ein lokaler Branch-Schalter weiterhin ON steht.

### Disabled branch execution and cache rule

For Detailer and FaceProcess modules, GLOBAL OFF or LOCAL OFF is a hard execution boundary. Detector, SAM, model preparation, selection, processing, preview generation and persistent branch caching must be skipped. Disabled branches return the authoritative source image, ordinary empty SEGS and a LOG BLOCK only. Boundary caches must not persist results when their branch dependency manifest is incomplete; this is the expected state for disabled branches.

### ControlNet-AIO-Auflösung ohne globale Modulinspektion

`CMK ControlNet Prepare` darf zur Ermittlung des AIO-Preprocessors nicht über
`sys.modules` iterieren und bei beliebigen Python-Modulen
`NODE_CLASS_MAPPINGS` abfragen.

Lazy Module wie `transformers` führen bei diesem Attributzugriff umfangreiche
Alias-Auflösungen aus. Dadurch kann bereits die Prompt-Validierung minutenlang
blockieren, obwohl `USE CONTROLNET = OFF` gesetzt ist.

Die Auflösung erfolgt verbindlich in dieser Reihenfolge:

1. über die bereits aufgebaute öffentliche ComfyUI-Registry
   `nodes.NODE_CLASS_MAPPINGS`,
2. ersatzweise ausschließlich über bekannte ControlNet-Aux-Modulpfade,
3. einmalig gecacht für die laufende ComfyUI-Instanz.

Eine globale Inspektion aller geladenen Python-Module ist unzulässig.

### FaceProcess-Boundary: deaktivierter Modul-Passthrough ohne Cache-Fingerprinting

`CMK Face Boundary Cache` muss vor jeder persistenten Cache-Prüfung statisch
ermitteln, ob das gesamte FaceProcess-Modul deaktiviert ist.

Das Modul gilt als vollständig deaktiviert, wenn mindestens eine der folgenden
Bedingungen erfüllt ist:

```text
face_global_enable = OFF
alle CMK FaceProcess -Pipe--Instanzen: enable = OFF
```

In diesem Zustand gilt verbindlich:

- keine rekursive Fingerprint-Berechnung des expandierten Upstream-Graphs,
- keine Branch-Manifest-Berechnung,
- kein Lesen oder Schreiben persistenter Boundary-Dateien,
- unverzüglicher Passthrough von IMAGE und LOG,
- keine Änderung des autoritativen Bildes.

Die Prüfung liest ausschließlich die Boolean-Werte der relevanten Prepare- und
Execute-Nodes aus dem Promptgraphen. Sie darf keine Bild-, Pipe-, LOG- oder
SEGS-Payloads kanonisieren.

Der FaceProcess-Boundary-Key schließt die transportierenden Eingänge `IMAGE`
und `LOG` von der eigenen Node-Signatur aus. Die inhaltliche Cache-Validierung
aktiver Branches bleibt Aufgabe des separaten Branch-Manifests. Dadurch darf
die Boundary-Identität nicht erneut den vollständigen vorgelagerten Workflow
rekursiv serialisieren.

### Neutrale lokale LoRA-Auswahl in Prepare-Nodes

Die lokalen LoRA-Dropdowns von

```text
CMK Sampler Prepare SDXL -Pipe-
CMK Refiner Prepare SDXL -Pipe-
```

müssen eine explizite neutrale Auswahl `None` anbieten.

`None` bedeutet:

- keine lokale Einzel-LoRA laden,
- MODEL und CLIP durch diesen lokalen Auswahlpfad nicht patchen,
- keine LoRA-Fehlermeldung oder stiller Ersatz,
- geerbte beziehungsweise separat aktivierte LoRA-Stacks bleiben davon
  unabhängig.

Für `CMK Refiner Prepare SDXL -Pipe-` ist `None` der Standard. Eine Refiner-LoRA
muss bewusst gewählt werden, da eine zum SDXL-Basismodell passende LoRA nicht
zwangsläufig mit der Refiner-Architektur kompatibel ist. Shape-Mismatch-Fehler
dürfen nicht durch eine erzwungene Dropdown-Auswahl provoziert werden.

`CMK Sampler Prepare SDXL -Pipe-` behält den bestehenden bevorzugten
1st-Pass-Standard, sofern diese Datei installiert ist; `None` steht dort
zusätzlich als bewusste Abschaltung zur Verfügung.

### Detailer Prepare: globaler Bypass und lokale LoRA

Für `CMK Detailer Prepare -Pipe-` gilt:

```text
DETAILER GLOBAL ENABLE = OFF
```

führt unmittelbar zu einem leichten deaktivierten `CMK_DETAILER_PIPE`-Passthrough.
Vor diesem Rückgabepunkt dürfen keine der folgenden Operationen stattfinden:

- MODEL oder PROCESS materialisieren,
- SAM laden,
- lokale LoRA anwenden,
- vom Sampler angebotene LoRA-Syntax oder einen LoRA-Stack übernehmen,
- CLIP verändern,
- Conditioning erzeugen.

Der Bypass muss in der Konsole eindeutig als übersprungener Prepare-Pfad
diagnostizierbar sein.

Das lokale LoRA-Dropdown enthält die neutrale Auswahl `None` und verwendet
`None` als Standard. Eine lokale Detailer-LoRA ist eine bewusste, optionale
Modulressource.

## Modul-Boundaries, Disabled-Passthrough und persistenter Cache

### Grundprinzip

Eine CMK-Modul-Boundary ist kein beliebiger Cache-Endpunkt des gesamten
Workflows. Sie bildet ausschließlich die technische Grenze des unmittelbar
zugehörigen Moduls ab.

Die Reihenfolge der Module ist nicht festgelegt. Insbesondere darf keine
Boundary davon ausgehen, dass ihr Modul:

- das letzte Glied der Kette ist,
- zwingend vor oder nach einem bestimmten anderen Modul steht,
- überhaupt im Workflow vorhanden ist,
- stets aktive Branches enthält.

Detailer, FaceSwap, FaceProcess, Refiner und zukünftige Module müssen in
beliebiger Reihenfolge kombinierbar oder vollständig weglassbar bleiben.

### Verbindlicher Disabled-Passthrough

Ist ein Modul vollständig deaktiviert, gilt:

```text
GLOBAL ENABLE = OFF
oder
alle lokalen Branches = OFF
```

Dann darf die zugehörige Modulstrecke keine teuren Ressourcen vorbereiten oder
materialisieren.

Der Disabled-Passthrough muss:

- das autoritative IMAGE unverändert weitergeben,
- LOG entsprechend der jeweiligen Schnittstelle unverändert beziehungsweise
  als neutralen Modulblock weitergeben,
- MODEL und PROCESS nicht materialisieren, wenn sie nur für die aktive
  Verarbeitung benötigt würden,
- keine LoRA laden oder anwenden,
- keinen CLIP-Patch erzeugen,
- kein Conditioning aufbauen,
- kein SAM-, Detector-, Restore- oder Swap-Modell laden,
- keine persistente Cache-Datei lesen oder schreiben,
- kein rekursives Upstream-Fingerprinting ausführen.

Ein sichtbarer deaktivierter Node darf kurz als Teil der Graphauswertung
erscheinen. Seine eigentliche Ausführungszeit muss jedoch im
Millisekundenbereich bleiben.

### FaceProcess-Boundary

Für `CMK Face Boundary Cache` ist bestätigt:

```text
FaceProcess global OFF
oder
alle lokalen FaceProcess-Instanzen OFF
```

führt zu:

```text
DISABLED PASSTHROUGH -> CACHE SKIPPED
```

Die Boundary überspringt dabei:

- rekursive Fingerprint-Berechnung,
- Branch-Manifest-Berechnung,
- Lesen und Schreiben des persistenten Caches.

Der Boundary-Key schließt die transportierenden Eingänge `IMAGE` und `LOG`
von seiner eigenen Node-Identität aus. Eine Boundary darf nicht durch
rekursive Kanonisierung des gesamten vorgelagerten Graphen exponentiell
teurer werden.

### Detailer Prepare

Für `CMK Detailer Prepare -Pipe-` gilt bei:

```text
DETAILER GLOBAL ENABLE = OFF
```

ein unmittelbarer Lightweight-Passthrough.

Vor diesem Rückgabepunkt dürfen nicht ausgeführt werden:

- MODEL-Anforderung,
- PROCESS-Anforderung,
- SAM-Laden,
- lokale LoRA-Verarbeitung,
- Übernahme von Prompt oder LoRA-Stack aus dem Sampler,
- CLIP-Veränderung,
- Conditioning-Erzeugung.

Der bestätigte Diagnosehinweis lautet:

```text
GLOBAL DISABLED
-> MODEL / PROCESS / SAM / LORA / CONDITIONING SKIPPED
```

### Cache-Fingerprinting

Persistente Modul-Caches dürfen ihre Signatur nicht aus einer vollständigen
rekursiven Serialisierung des gesamten Upstream-Workflows erzeugen.

Zulässig sind ausschließlich kompakte, modulspezifische Informationen:

- eigene Schema-Version,
- eigene Node-Identität,
- relevante Modulparameter,
- kompakte Branch-Revisionen,
- kompakter Token der unmittelbar vorgelagerten autoritativen Modulgrenze.

Unzulässig sind:

- rekursive Kanonisierung vollständiger IMAGE-, LOG-, PROCESS- oder
  MODEL-Payloads,
- wiederholte Vollgraph-Traversierung in `check_lazy_status()` und
  anschließend erneut in der eigentlichen Boundary-Funktion,
- positionsabhängige Annahmen über die Reihenfolge der Module.

### Diagnoseinstrumentierung

Die temporäre Diagnoseinstrumentierung protokolliert mit Zeitstempel:

```text
INPUT_TYPES
IS_CHANGED
check_lazy_status
eigentliche Node-Funktion
```

Sie dient ausschließlich zur Laufzeitanalyse. Rückgabewerte,
Lazy-Abhängigkeiten und Cache-Entscheidungen dürfen dadurch nicht verändert
werden.

Die Instrumentierung bleibt solange Bestandteil der Diagnostic-Builds, bis
alle Modul-Boundaries architekturweit auf den kompakten,
positionsunabhängigen Mechanismus vereinheitlicht wurden.

## LoRA-Verarbeitung in Sampler, Refiner und Detailer

### Neutrale Auswahl

Optionale lokale LoRA-Dropdowns müssen eine echte Auswahl `None` anbieten.

`None` bedeutet:

- keine lokale Einzel-LoRA laden,
- MODEL und CLIP über diesen lokalen Pfad nicht patchen,
- keinen stillen Rückfall auf den ersten verfügbaren Dateieintrag,
- keine Veränderung eines separat angeschlossenen zentralen LoRA-Stacks.

Verbindliche Standards:

```text
CMK Refiner Prepare SDXL -Pipe-  -> LORA = None
CMK Detailer Prepare -Pipe-      -> LORA = None
```

`CMK Sampler Prepare SDXL -Pipe-` darf weiterhin einen bewusst definierten
1st-Pass-Standard anbieten, muss aber ebenfalls `None` enthalten.

### Modellkompatibilität

Eine LoRA ist nicht allein deshalb mit einem Modell kompatibel, weil beide zur
SDXL-Familie gehören.

Insbesondere gilt:

```text
SDXL Base Checkpoint != SDXL Refiner Checkpoint
```

Eine LoRA kann mit einem 1st-Pass-Checkpoint fehlerfrei funktionieren und im
Refiner dennoch `shape mismatch`-Fehler auslösen.

Deshalb:

- Refiner-LoRAs müssen bewusst und modellspezifisch ausgewählt werden.
- Detailer-LoRAs müssen gegen das im Detailer tatsächlich verwendete Modell
  separat geprüft werden.
- Ein zentraler 1st-Pass-LoRA-Stack darf nicht automatisch auf den Refiner
  übertragen werden.
- `lora key not loaded` und Tensor-Shape-Fehler sind als
  Ressourceninkompatibilität zu behandeln, nicht als Boundary- oder
  Ausführungsfehler der CMK-Modularchitektur.

### Bestätigter Referenzzustand

Folgender Zustand wurde fehlerfrei getestet:

```text
1st-Pass:
  lora_syntax angeschlossen

Refiner:
  LORA = None

Detailer:
  GLOBAL ENABLE = ON
  LORA = None
  USE PROMPT / LORA FROM SAMPLER = OFF
```

Ebenfalls bestätigt:

```text
Detailer GLOBAL ENABLE = OFF
```

verarbeitet weder lokale noch geerbte LoRAs.

Damit sind zentrale LoRA-Syntax, Refiner-Grundpfad,
Detailer-Grundpfad und Disabled-Passthrough funktional voneinander entkoppelt.

### FaceProcess: Wiederverwendung der Cache-Analyse

Die aktive FaceProcess-Strecke darf dieselbe expandierte Promptstruktur nicht
mehrfach innerhalb desselben Workflow-Submissions kanonisieren.

Verbindlich gilt:

- `build_node_fingerprint()` memoisiert Ergebnisse ausschließlich für das
  aktuell übergebene Promptobjekt.
- Ein neues Promptobjekt verwirft die Memoisierung vollständig.
- `CMKFaceBoundaryCache.check_lazy_status()` und
  `CMKFaceBoundaryCache.boundary()` verwenden dieselbe einmal berechnete
  Analyse aus Boundary-Key und Branch-Manifest.
- Vor dem Speichern eines Boundary-Misses wird das Branch-Manifest nicht
  nochmals identisch berechnet.
- Die Memoisierung verändert weder Cache-Key, Revisionstoken,
  Invalidierungsregeln noch funktionale Ausgaben.

Damit bleibt die bisherige vollständige Invalidierungslogik erhalten, während
wiederholte Vollgraph-Traversierungen innerhalb desselben Laufs entfallen.

## FaceSwap-Compositing: getrennte Swap- und Blend-Maske

Image- und Video-FaceSwap verwenden denselben gemeinsamen Paste-back-Kern.

Verbindliche Phase-1-Architektur:

```text
Affine Valid-Support Mask
        +
Face Shape (Landmarks/BBox)
        ↓
Base Mask
        ├── leicht erodiert → SWAP MASK
        └── leicht dilatiert + Gaussian Blur → BLEND MASK
```

`SWAP MASK` definiert ausschließlich den Bereich des Identitätstransfers.
Sie wird adaptiv geringfügig nach innen gezogen, damit Haare, Ohren, Schmuck
und Hintergrund aus dem Target erhalten bleiben.

`BLEND MASK` ist davon getrennt. Sie wird aus der Swap-Maske leicht nach außen
erweitert und weichgezeichnet. Damit kann der Übergang außerhalb des
Identitätskerns stattfinden, ohne den tatsächlich ersetzten Bereich zu
vergrößern.

Bei `feather = 0` wird die Blendbreite automatisch aus der erkannten
Gesichtsgröße abgeleitet. Ein positiver `feather`-Wert bleibt eine bewusste
manuelle Überschreibung.

Die Änderung gilt zentral für:

```text
CMK FaceSwap Image -Pipe-
CMK FaceSwap Video
```

Die FaceSwap-Boundary-Schema-Version und die Video-Engine-Version müssen bei
Änderungen am zentralen Compositing erhöht werden, damit keine Ergebnisse der
alten Paste-back-Logik wiederverwendet werden.

### GPEN im FaceSwap-Pfad

GPEN arbeitet intern auf 512 × 512 Pixeln, während der INSwapper-Patch
typischerweise 128 × 128 Pixel umfasst. Der Enhancer darf deshalb nicht
ungefiltert dominant in den Paste-back-Patch eingehen.

Verbindliche Verarbeitung:

- GPEN-Anteil adaptiv nach lokaler Kantenstärke,
- maximal 42 % in glatten Gesichtsbereichen,
- nur 12 % an starken Kontrastkanten,
- Schutz von Brille, Augenbrauen, Augenkonturen und Zähnen,
- Begrenzung neu erzeugter Hochfrequenzdetails relativ zum Originalpatch,
- Rückskalierung 512 → 128 mit `INTER_AREA`,
- `INTER_CUBIC` ausschließlich beim Hochskalieren.

Lanczos ist für die Rückskalierung des generativ restaurierten GPEN-Patches
nicht zulässig, da synthetische Kanten dadurch Ringing und Treppenartefakte
erzeugen können.
