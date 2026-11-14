# CMK Architecture

**Vertragsstand:** 1.1
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
- Zwei Module dürfen außer über ihre öffentliche funktionale Schnittstelle
  keine Kenntnis voneinander benötigen. Integrationsprobleme eines neu
  hinzugefügten Moduls werden in diesem Modul oder in einem nachweislich
  allgemeinen Vertrag gelöst, nicht durch modulspezifische Sonderpfade in
  bereits bestätigten Vorgängern.

Eine Node ist erst dann als `-Pipe-`-Node fertig, wenn dieser Vertrag erfüllt ist.

### 2.2 Nummerierte Generation, unnummerierter PostProcess

Die gerichtete Generationspipeline besitzt die sichtbaren Schritte 01, 02, 05,
10, 15 und 20. Nach der letzten aktiven Generationsstufe beginnt die
unnummerierte PostProcess-Zone. Detailer, FaceRebuild, FaceProcess, FaceSwap und
Upscale & Save erhalten keine Prozessreihenfolge aus ihrem Namen; maßgeblich ist
allein die Verkabelung des Workflows. MaskDetailer ist der terminale interaktive
Bearbeitungsschritt. Visualizer ist die terminale Visualisierungs- und
Output-Stufe, aber kein Bild-Finalizer.

### 2.3 Einzelne CMK-Nodes in ComfyUI

Alle übrigen CMK-Nodes bleiben über die reguläre ComfyUI-Node-Suche verfügbar.

Sie dürfen:

- native ComfyUI-Typen direkt verwenden;
- einzelne Verarbeitungsschritte offenlegen;
- manuelle Verkabelung und alternative Reihenfolgen erlauben;
- Flexibilität über Schutzmechanismen stellen.

Einzelne Nodes dürfen das geschlossene `-Pipe-`-Ökosystem ergänzen, aber dessen Schutz- und Bedienkonzept nicht aufweichen. Sie bilden keinen eigenständigen Produkt- oder Browserbereich. Eine solche Funktion kann bewusst mehr Eingriffe erlauben als das Apple-like Flow-Frontend, muss aber weiterhin einen klar benannten und gepflegten Einsatzzweck besitzen.

`CMK FaceSwap Image -Pipe-` gehört zum geschlossenen Flow. Im Referenzworkflow wird seine Execute-Node innerhalb von `CMK SwapFace -Pipe-` gemeinsam mit Source-Loader und verpflichtender Boundary komponiert. Der eigenständig gespeicherte Subgraph bleibt eine alternative Flow-Oberfläche desselben vollständig funktionsfähigen Moduls. Geplante Anpassungen an dieser Oberfläche ändern seine Produktzuordnung nicht.

Als einzelne ComfyUI-Node steht dagegen `CMK FaceSwap Image` **ohne** `-Pipe-` zur Verfügung. Diese Variante stellt zusätzliche manuelle Auswahl- und Experimentierschnittstellen bereit.

## 3. Die fünf sichtbaren Transportrollen

Der normale pixelbasierte CMK-Workflow verwendet fünf semantisch getrennte Leitungen:

| Sichtbarer Name | Typ | Verantwortung | Darf nicht enthalten oder bewirken |
|---|---|---|---|
| `MODEL` | `CMK_MODEL_PIPE` | gemeinsame, unveränderte Modellressourcen und kleine Loader-Metadaten | Modulzustand, Bilder, Latents, Logs, lokale Patches |
| `PROCESS SDXL` / `PROCESS Z-IMAGE` | `CMK_PROCESS_SDXL` / `CMK_PROCESS_Z_IMAGE` | familiengebundener, nicht-pixelbasierter Prozesskontext, Masken, Dimensionen, Prompts, LoRA-Angebote und Workflow-Zustand | das authoritative Ergebnisbild, Modellressourcen, Dokumentationslogik oder Wechsel in die andere Modellfamilie |
| `IMAGE` | `IMAGE` | authoritative Pixelinformation zwischen Bildmodulen | versteckte Zustands- oder Logfunktion |
| `LOG` | `CMK_LOG_PIPE` | strukturierte Dokumentation bereits getroffener Entscheidungen und ausgeführter Schritte | Steuerung, Bildveränderung, Modellpatching, Modulaktivierung |
| `VISUAL` | `CMK_VISUAL_PIPE` | geordnete Provider-, Capability-, Channel- und Statusstruktur für die zentrale Darstellung | authoritative Pixelverarbeitung, Processing-Steuerung oder Presentation-State |

Die Rollen beantworten fünf unterschiedliche Fragen:

```text
MODEL   → Welche unveränderten Modellressourcen stehen zur Verfügung?
PROCESS → Welcher Prozesskontext wird angeboten?
IMAGE   → Welches Bild wird tatsächlich weiterverarbeitet?
LOG     → Was wurde dokumentiert?
VISUAL  → Welche ausdrücklich publizierten Darstellungszustände stehen bereit?
```

An jeder öffentlichen CMK-Moduloberfläche liegt `VISUAL` unmittelbar nach
`LOG` und unmittelbar vor dem optionalen Diagnoseausgang. Für Module mit
MODEL-, PROCESS- und Ergebnistransport gilt damit verbindlich:

```text
MODEL / PROCESS / IMAGE oder SAMPLED / LOG / VISUAL / diagnostic
```

PostProcess-Module verändern MODEL nicht. Bei Detailer, FaceProcess und
FaceRebuild (jeweils Standard und Advanced) kommt der öffentliche MODEL-Ausgang
deshalb über `CMK Model Forward -Pipe-` direkt aus dem MODEL-Eingang des Moduls,
nicht aus dem Ergebnis-Gate (OD-2). Eine Anforderung nur von MODEL wertet so
weder Prepare noch Compute, Boundary oder Gate aus. Der MODEL-Eingang des
Forwards ist optional; ohne Checkpoint wird `None` unverändert durchgereicht.
FaceSwap führt MODEL weiterhin über `CMK Result Pack` (eigener Typ
`CMK_RESULT_MODEL`).

Der Eingang des externen `CMK Visualizer` ist optional, damit ein zunächst
unverdrahtet platzierter Visualizer die Prompt-Validierung nicht blockiert.
Ohne Pipe zeigt er einen leeren Wartezustand; erst die angeschlossene `VISUAL`-
Leitung liefert Provider und Channels.

Der `CMK Visualizer` ist zugleich die verbindliche Endstation des Bildzweigs.
Neben `VISUAL` nimmt er die öffentlichen Leitungen `MODEL (opt)`, `PROCESS`,
`IMAGE` und `LOG` entgegen. In der Standardansicht liegen ausschließlich
`SAVE ENABLED` und `filename prefix`; Ausgabeordner, Datumsordner, Upscale-
Schalter, Megapixelgrenzen und 4x-/2x-Modellwahl sind Advanced-Eingaben.
Er verwendet dafür die bestehenden Result-Unpack-, Smart-Upscale- und Save-
Verträge und implementiert weder einen zweiten Cache noch eine abweichende
Speicher- oder Upscale-Logik.

Anschlussgruppen (2.7.1): `Z-IMAGE TURBO` und `SDXL` sind die
Generierungsanschlüsse, `POSTPROCESS` die Anschlüsse der PostProcess-Kette:

| Anzeige | Interner Name | Verwendung |
|---|---|---|
| MODEL ZIT | `POSTPROCESS MODEL ZIT` | Weitergabe des ZIT-Modells (z. B. Multi-Model-Detailer); wird nie angefordert |
| MODEL SDXL | `MODEL (opt)` | unverändert: Result Unpack, Save, Freigabe der SDXL-PostProcess-Spec |
| PROCESS, IMAGE, LOG, VISUAL | unverändert | unverändert |

- `POSTPROCESS MODEL ZIT` ist ein lazy Eingang ohne Anforderung: Der Visualizer
  materialisiert und referenziert dort kein Modell; die ZIT-Freigabe vor SDXL
  bleibt wirksam. Die internen Namen der übrigen Eingänge sind unverändert,
  deshalb bleiben API-Prompts und gespeicherte Workflows gültig. Das Frontend
  ordnet gespeicherte Eingänge nach Namen zu; die Migration alter
  5-Eingang-Visualizer setzt die Gruppe auf Slot 11.
- Normalansicht: Überschrift `POSTPROCESS`, Abstand zur SDXL-Gruppe und eine
  dezente Fläche (`cmk_family_merge_layout.js`, Zuordnung über Eingangsnamen).
- Clean View: Überschrift, Fläche und Abstände entfallen vollständig; die
  kompakte Größe berücksichtigt den neuen Eingang nicht, Clean View bleibt
  damit unverändert.

`enable upscale = false` ist echter Bypass: kein Modell wird geladen, das Bild
bleibt unverändert und es entsteht kein Register. Bei `true` wird das
hochskalierte Ergebnis als reguläres Bildsignal im zusätzlichen Register
`Upscale` publiziert; ein anschließend aktiviertes Save speichert genau dieses
Endergebnis. `SAVE ENABLED = false` führt keinerlei Dateischreibvorgang aus.
Die Grenzwerte beziehen sich auf die Megapixelzahl des Eingangsbildes: bis
einschließlich `limit 4x MP` gilt der 4x-Zweig, anschließend bis einschließlich
`limit 2x MP` der 2x-Zweig. `limit 4x MP = 0.0` deaktiviert den 4x-Zweig.
### 3.0 Upscale & Save

`Upscale & Save` bleibt ein eigenständiges, frei positionierbares Modul für
Zwischenstände. Ab Modulversion 1.4.0 trennt die kanonische Definition drei
Verantwortungen:

- **IMAGE-Weitergabe:** `IMAGE OUT` ist immer `IMAGE IN`, unabhängig von
  Upscale und Save. Die Kette arbeitet nie mit der hochskalierten Kopie weiter.
- **Upscaling:** `CMK_SmartUpscalerPipe` erzeugt nur die zu speichernde Kopie.
  `CMKUpscaleSaveBoundaryCache` materialisiert sie. Bei unveränderten Eingaben
  liefert die Boundary sie aus dem Cache, ohne Upscaler-Ausführung und ohne
  `unload_all_models()`.
- **Saving:** `CMK Save Endpoint` (`CMKSaveEndpoint`) ist ein endständiger
  OUTPUT_NODE ohne verbrauchte Ausgänge und mit Lazy-Eingängen. Er liest das
  Boundary-Ergebnis und schreibt über `CMK_SaveProjectImage.run` bei jeder
  ausgeführten Prompt-Ausführung mit `SAVE ENABLED = true` genau eine neue Datei,
  auch wenn das Bild aus einem Cache stammt. Sein `IS_CHANGED` erreicht keine
  andere Cache-Signatur, weil kein Node seine Ausgaben verbraucht. Bei
  `SAVE ENABLED = false` fordert er keinen Eingang an.

Das Register `Upscale & Save` publiziert der bestehende `CMKVisualProvider`
hinter der Boundary mit den Kanälen `before` (Eingangsbild) und `result`
(gespeicherte Kopie). `SAVE ENABLED = false` ist echter Bypass: Upscaler,
Boundary, Provider und Speichern werden nicht ausgewertet.

Eingebettete Definitionen vor Version 1.4.0 (z. B. in `CMK 2.6 · Full Flow`)
bleiben unverändert lauffähig und behalten ihre bisherige Semantik
(hochskaliertes `IMAGE OUT`, `CMK_SaveProjectImage` vor der Boundary). Sie sind
in `tests/contracts/legacy_registry.json` als Legacy geführt.

### 3.1 Warum IMAGE eine eigene Leitung ist

`IMAGE` liegt bewusst nicht im öffentlichen `PROCESS`-Transport:

- Bildänderungen bleiben im Workflow sichtbar.
- Ein Modul kann kein veraltetes, intern gespiegeltes Bild versehentlich bevorzugen.
- Pixelverarbeitung und Prozessmetadaten werden unabhängig cachebar.
- Standalone-Bildmodule können mit nativen `IMAGE`-Sockets arbeiten.
- Ein Bild muss nicht durch eine Node geleitet werden, die es fachlich gar nicht verarbeitet.

Interne Arbeits-Pipes dürfen zur Ausführung ein Bild referenzieren. Das ändert nicht den öffentlichen Vertrag: Zwischen Modulen bleibt `IMAGE` authoritative.

### 3.2 VISUAL-Provider und zentraler Visualizer

`VISUAL` ist keine zweite `IMAGE`-Leitung. Die Pipe akkumuliert kleine
Provider-Beschreibungen mit stabiler Provider- und Modulidentität, Reihenfolge,
Capabilities, benannten Channels und Lifecycle-Status. Ein Channel darf eine
Referenz auf ein bereits vorhandenes Bildobjekt halten; der Provider erzeugt
dafür keine Pixelkopie. Nur ein expliziter Publish-Schritt macht einen Channel
sichtbar. ContentGuard-Decodes und andere interne Diagnosebilder werden deshalb
nie implizit registriert.

```text
CMK Processing
      ↓
durchgehende VISUAL Pipe
      ↓
Visual Provider Registry / State
      ↓
CMK Visualizer
```

Provider werden capability-driven ausgewertet; der Visualizer enthält keine
Liste bekannter CMK-Module. `CMK Visual Provider -Pipe-` registriert Result- und
optional Source-/Before-/After-Channels. Wiederholte Registrierung derselben
Provider-ID ersetzt nur deren Status und Channels, während die übrige
akkumulierte Pipe erhalten bleibt. Die Provider-ID bezeichnet immer eine
**konkrete Modulinstanz**, nicht einen Modultyp. `branch`, `stage_key`, Label und
`sequence` sind semantische Klassifikation beziehungsweise Sortierinformation
und dürfen nie als eindeutige Identität verwendet werden.

Mehrere Instanzen desselben Moduls dürfen deshalb dieselbe `sequence`, denselben
`stage_key` und dasselbe Label besitzen. Sie bleiben getrennte Provider. Bei
gleicher `sequence` gilt ihre Reihenfolge in der tatsächlich durchlaufenen
`VISUAL`-Kette. Der Visualizer ergänzt nur für die Darstellung fortlaufende
Suffixe (`Detailer`, `Detailer 2`, `Detailer 3`); diese Anzeigenamen verändern
weder Provider-ID noch Cache. `CMK Visual Forward -Pipe-` reicht Module ohne
eigenen visuellen Beitrag identisch durch.

`sequence` bestimmt ausschließlich die grobe sichtbare Registersortierung. Sie
erzwingt keine Prozessreihenfolge und ist kein Zähler, Cachebestandteil oder
Ersatz für die konkrete Instanzidentität.

**Publikation an der Modulgrenze.** In Detailer, FaceProcess, FaceRebuild und
FaceSwap (jeweils Standard und Advanced) registriert weiterhin der Compute-Node
beziehungsweise `CMK SEGS CONCAT` den Provider – nur bei lokal aktiven Zweigen
und vorhandenem Inhalt. Veröffentlicht wird er aber von der Boundary: Ihr
Eingang `VISUAL IN` ist die eingehende Pipe, `VISUAL` die vom Compute-Node
zurückgegebene. Beim MISS speichert die Boundary genau die neu hinzugekommenen
Provider mit vollständigen Metadaten und den Channels als RGB-Bild, wie der
Visualizer sie rendert. Beim HIT hängt sie diese an `VISUAL IN`, ohne den
Compute-Node anzufordern. Im Passthrough reicht sie `VISUAL IN` durch; ein
blockiertes Ergebnis blockiert auch VISUAL. Das Gate erhält `VISUAL ACTIVE`
ausschließlich von der Boundary. Beide VISUAL-Eingänge sind vom Cache-Schlüssel
ausgenommen. Eingebettete Definitionen ohne diese Verdrahtung bleiben
unverändert lauffähig (Legacy).

Das `CMK Module Bypass Gate` fordert die fehlenden Ergebnis-Eingänge
(`MODEL`, `IMAGE`, `LOG`) gemeinsam an und die Hilfs-Eingänge (`DIAGNOSTIC`,
`VISUAL`, `PROCESS`) gemeinsam, sobald `IMAGE` und `LOG` vorliegen. Ein
legitim leeres `MODEL` (ohne Checkpoint) beendet die Anforderung daher nicht
vorzeitig, und ein blockiertes Ergebnis stoppt das Gate weiterhin vor den
Hilfs-Produzenten.

Provider-Auswahl, die per gedrückter Maustaste temporäre Before-/After-Ansicht und abgeschlossene History
sind reiner Frontend-Presentation-State. Sie verändern weder Prompt-Eingaben
noch Processing-Pipes und invalidieren daher keine vorgelagerten Module. Eine
neue ComfyUI-`prompt_id` eröffnet im Frontend eine neue Execution und verwirft
die vorherige History; verspätete Live-Ereignisse werden nur für die aktive
Execution angenommen.

Sampler-Livebilder laufen weiterhin über ComfyUIs bestehenden Latent-Preview-
Callback und WebSocket-Sidechannel. Ein Provider verweist dafür auf eine oder
mehrere zugehörige Sampler-Node-IDs innerhalb derselben konkreten
Subgraph-Instanz. Der Visualizer übernimmt den bereits decodierten
Preview-Frame, folgt dem aktiven Provider und ersetzt ihn nach Abschluss immer
durch dessen autoritativen Final-Channel. Es gibt keinen zweiten Preview-Decode.
Für aktuelle ComfyUI-Versionen ist `b_preview_with_metadata` der authoritative
Live-Kanal: `prompt_id`, `node_id`, `real_node_id`, `parent_node_id` und
`display_node_id` isolieren Execution und Subgraph-Instanz. `progress_state`
liefert Schritt, Gesamtzahl und Running-Status. Das metadatenlose `b_preview`
bleibt ausschließlich als Legacy-Fallback bestehen und wird nicht zusätzlich
verarbeitet, wenn derselbe Blob bereits über den Metadatenkanal eingetroffen ist.

Lifecycle Ownership bleibt getrennt: Processing-Nodes publizieren Bildkanäle
und deren Instanzidentität; alle Entscheidungen über Anzeigen, Folgen,
Behalten, Leeren, Verbergen und Neuaufbau gehören dem `CMK Visualizer`.
ComfyUI-Running-Status, globale Fortschrittsanzeige und bloße Existenz eines
Subgraphs sind ausdrücklich **kein Bildsignal** und dürfen kein Register
anlegen.

Subgraph-Metadaten `cmkVisualProviders` beschreiben ausschließlich Identität,
Label, Reihenfolge, Capabilities, Aktivierung und die Zuordnung von
Live-Node-IDs. Sie bilden einen internen Ereigniskatalog, lösen keine
Verarbeitung aus und erzeugen **keine vorauseilenden Register**. Ein Register
entsteht erst, wenn für seine konkrete Instanz ein Livebild oder ein finaler
Bildchannel eintrifft. Ein Cache-Hit gilt dabei als reguläres finales Bildsignal.
Der abschließende Backend-Payload ist für die Execution autoritativ: nicht darin
enthaltene, deaktivierte Provider dürfen weder leer noch mit einem
durchgereichten Fremdbild fortbestehen.

Der optionale `enable`-Eingang des Providers ist ein reiner `forceInput`-Socket
ohne positionsgebundenes Widget. Unverbunden gilt er als `true`; ein Modul mit
GLOBAL ENABLE verbindet ihn ausdrücklich. Dadurch kann Bypass die Registrierung
verhindern, ohne die gespeicherten Widgetpositionen bestehender Provider zu
verschieben.

Diese Deklarationen werden zusätzlich bei Graphänderungen neu ausgewertet.
Einfügen, Entfernen oder Neuverbinden eines Moduls aktualisiert ausschließlich
den internen Providerkatalog; sichtbare Register entstehen daraus nicht. Es
wird weder ein Prompt gestartet noch ein Processing-Cache invalidiert.
Ereignisserien beim Aufbau eines Subgraphs werden zu einem Refresh
zusammengefasst. Während einer aktiven Execution ist dieser Struktur-Refresh
gesperrt, damit Live- und Final-State nicht durch einen parallelen
Editor-Lifecycle überschrieben werden.

Der erste verbindliche End-to-End-Slice verläuft über die Module 10 und 20:

```text
10 KSampler SDXL 1st Pass
    Provider 1st Pass → result + Sampler-Live-Sidechannel
        ↓
20 Refiner SDXL
    Provider Refiner   → AFTER; BEFORE nur solange die Maustaste gedrückt bleibt
        ↓
CMK Visualizer
```

Modul `15S InstantID Regional SDXL` erweitert diese Kette um den Provider
`Identity` an Position 15. Seine Sampler-Frames werden live dargestellt. Nach
bestandener finaler ContentGuard-Prüfung entsteht ein separater Visual-Decode,
der ausschließlich als `Identity.result` in die VISUAL-Pipe eingeht; der
öffentliche Subgraph erhält keinen zusätzlichen IMAGE-Ausgang. Der interne
Source-Face-Loader bleibt im geöffneten Subgraph vollständig sichtbar, während
sein `image`-Widget zugleich als Bildauswahl auf der äußeren Modul-Node
bereitsteht.

Die Bildobjekte der Provider sind Referenzen auf bereits vorhandene Ergebnisse;
die Publish-Schritte kopieren oder decodieren keine Pixel. Das Refiner-Modul
verwendet die ohnehin erzeugten First-Pass- und Refiner-Decodes. Der Live-Pfad
übernimmt ausschließlich ComfyUIs vorhandenes `b_preview`-Ereignis. Die echte
ComfyUI-`prompt_id` ist die Execution-Identität des Frontends; bei einer neuen
`execution_start`-Meldung werden History, Live-Blob und Auto-Follow-Zustand der
vorherigen Execution verworfen. Ereignisse mit abweichender `prompt_id` werden
ignoriert. Nach `execution_success` oder `execution_error` endet Auto-Follow;
Provider-Navigation und der Klickvergleich bleiben reine UI-Aktionen. Die
Bildreferenzen werden unter dem privaten UI-Feld `cmk_visual_images` übertragen;
dadurch erzeugt ComfyUI unterhalb des Visualizers keine zusätzliche native
Thumbnail-Galerie.

Ein interner `CMK Load Image` mit nach außen geführter Dateiauswahl behandelt
den tatsächlich ausgeführten Eingang als einzige Wahrheitsquelle. Seine innere
Vorschau muss nach der Ausführung genau dieses Bild anzeigen und darf nicht am
gespeicherten Defaultwert seines lokalen Widgets festhalten. Dieser Vertrag
gilt gemeinsam für ControlNet-, InstantID-, FaceRebuild- und FaceSwap-Module;
modulspezifische Synchronisationsregeln sind verboten.

Kompakte Flow-Module im Format `300 × 190` verwenden ein gemeinsames äußeres
UI-Raster. Der sichtbare Widgetblock ist unmittelbar oberhalb der
`Enter subgraph`-Leiste bottom-aligned. Advanced-Varianten mit ausschließlich
einem globalen Schalter erhalten darunter einen rein visuellen, leeren Spacer
in Höhe einer Combo-Zeile. Dieser Spacer ist kein Eingang, wird nicht
serialisiert und besitzt keinerlei Einfluss auf Prompt, Cache oder Ausführung.

### 3.3 Modellfamilien sind mechanisch getrennte Verträge

`01 START HERE` ist der gemeinsame Einstieg und die einzige sichtbare
Modellfamilien-Weiche. Danach gelten zwei inkompatible PROCESS-Typen:

```text
PROCESS SDXL    → CMK_PROCESS_SDXL
PROCESS Z-IMAGE → CMK_PROCESS_Z_IMAGE
```

Die Trennung ist bewusst nominell: Auch bei intern ähnlichen Python-Objekten
darf ComfyUI keine Verbindung zwischen einem SDXL- und einem Z-Image-Modul
zulassen. Der nicht ausgewählte Ausgang von `01 START HERE` liefert keinen
Prozess. Eine spätere Zusammenführung darf erst nach Abschluss der
familienabhängigen Verarbeitung über einen eigenen neutralen Ergebnisvertrag
erfolgen und genau einen aktiven Zweig anfordern.

`CMK Flow · PostProcess Boundary Combined` erhält jeweils
`MODEL / PROCESS / IMAGE / LOG` beider Familien. Die Wahl aus 01 ist bereits
im einzig aktiven PROCESS enthalten; ein separates Routing-Kabel existiert
nicht. 35 fordert zunächst nur beide leichtgewichtigen PROCESS-Durchgänge und
anschließend ausschließlich `MODEL / IMAGE / LOG` des aktiven Zweigs an. Die
vier Ausgänge `CMK_RESULT_MODEL`, `CMK_RESULT_PROCESS`, `CMK_RESULT_IMAGE` und
`CMK_RESULT_LOG` sind familien-neutral. Bei parallel aufgebauten Familien sind
sie die einzig zulässige Übergabe an FaceSwap und 90; ein einzelner SDXL- oder
ZIT-Zweig darf diese beiden gemeinsamen Module auch direkt speisen.

Ein rein pixelbasierter Workflow beginnt mit `CMK Image Load and Resize
-Pipe-`. Sein öffentlicher Ausgangsvertrag folgt der CMK-Reihenfolge
`MODEL / PROCESS / IMAGE / LOG / diagnostic`. Der optionale Eingang
`MODEL SDXL` wird unverändert zum regulären Ausgang `MODEL` durchgereicht; ohne
angeschlossenen Checkpoint bleibt dieser Wert leer. PROCESS wird als
`family_neutral` mit `source_model_family = image` gekennzeichnet. Der
öffentliche PROCESS-Port trägt dennoch den
Typ `CMK_PROCESS_SDXL`, damit derselbe schlanke Bildeingang zusammen mit einem
separaten SDXL-Checkpoint die Standalone-Module 25 und 30 speisen kann. Damit
können außerdem die modellneutralen Module 40 und 90 einen vollständigen
Bildbearbeitungspfad validieren, ohne einen unbenutzten SDXL- oder
ZIT-Checkpoint zu laden. Der neutrale Ergebnisvertrag ist ausschließlich für
diesen eindeutig markierten CMK-Bildeingang zulässig.
Für den rein pixelbasierten Weg zu 40 und 90 ist MODEL optional. Es wird kein
künstlicher Modellkontext und kein besonderer Pixel-Modelltyp erzeugt.

## 4. Proprietäre Arbeits- und Übergabetypen

Die folgenden Typen sind keine zusätzlichen allgemeinen Workflow-Leitungen. Sie schützen konkrete Modulgrenzen:

| Typ | Erzeuger | Zulässiger Hauptverbraucher | Zweck |
|---|---|---|---|
| `CMK_SAMPLER_PIPE` / `SAMPLER` | `CMK Sampler Prepare SDXL -Pipe-` | `CMK KSampler -Pipe-` | vollständig vorbereiteter erster Sampling-Kontext |
| `CMK_SAMPLED_PIPE` / `SAMPLED` | `CMK KSampler -Pipe-` | `CMK Refiner Prepare SDXL -Pipe-` | abgeschlossener First-Pass-Latent samt benötigtem Kontext |
| `CMK_REFINER_PIPE` / `REFINER` | `CMK Refiner Prepare SDXL -Pipe-` | `CMK Refiner -Pipe-` | vollständig vorbereiteter Refiner-Kontext |
| `CMK_DETAILER_PIPE` / `DETAILER` | `CMK Detailer Prepare -Pipe-` | parallele `CMK Smart Detailer -Pipe-` | unveränderlicher gemeinsamer Detailer-Kontext |
| `CMK_FACE_PIPE` / `FACE` | `CMK FaceProcess Prepare -Pipe-` | parallele `CMK FaceProcess -Pipe-` | unveränderlicher gemeinsamer FaceProcess-Kontext |
| `CMK_LOG_BLOCK` / `LOG BLOCK` | parallele Execute-Instanz ohne seriellen Transportvertrag | ausschließlich `CMK LOG CONCAT` | lokaler Logbeitrag für ausdrücklich parallele Merge-Architekturen |

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
PROCESS SDXL + PROCESS Z-IMAGE + IMAGE + LOG + diagnostic
```

Verantwortung:

- erzeugt genau einen aktiven, familiengebundenen `PROCESS`-Kontext;
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
MODEL (opt) + PROCESS + IMAGE_TARGET + LOG + ENABLE
    ↓
[CMK Load Image -Pipe- nur für IMAGE_SOURCE]
    ↓
CMK FaceSwap Image -Pipe-
    ├─→ CMK Boundary Cache → MODEL + IMAGE + LOG
    └─→ CMK Process Forward -Pipe- → PROCESS
```

Modulregeln:

- Das eingehende `IMAGE_TARGET` bleibt das authoritative Workflowbild.
- `CMK Load Image -Pipe-` liefert intern ausschließlich das Referenzbild an `IMAGE_SOURCE`; sein `PROCESS` und `LOG` ersetzen niemals den eingehenden Modulzustand.
- `MODEL` bleibt read-only und wird durch den FaceSwap-Boundary geführt, ohne von FaceSwap verwendet oder verändert zu werden.
- Öffentliche Bild-, Modell- und Log-Ausgänge sowie interne Ergebnis-Previews beziehen `MODEL`, `IMAGE` und `LOG` vom verpflichtenden `CMK Boundary Cache`. `PROCESS` wird ausschließlich über den separaten Forward-Pfad weitergeführt.
- `ENABLE = OFF` ist ein echter Lazy-Passthrough: `IMAGE_TARGET` wird unverändert ausgegeben; interner Source-Loader, Face Detection und Swap werden nicht angefordert.
- Der persistente FaceSwap-Boundary materialisiert `IMAGE` und `LOG` gemeinsam. `MODEL` wird aufgrund seiner read-only Rolle nur aktuell durchgereicht und nicht serialisiert; `PROCESS` gehört nicht zum Cache-Vertrag.
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

#### Gemeinsame PostProcess-Modellquelle (2.7.1)

`CMK PostProcess Model` (`CMKPostProcessModelSourcePipe`) vereint beide
Modellquellen in einem Node mit zwei unabhängigen Ausgängen:

```text
SDXL:          ckpt_name (inkl. None) + vae_name + checkpoint_vae        → MODEL SDXL
Z-IMAGE TURBO: diffusion_model + text_encoder + vae + weight_dtype
               + clip_device                                              → MODEL ZIT
```

- Beide Ausgänge sind die unveränderten Spezifikationen der Einzelquellen
  (`make_postprocess_model_spec`, `make_zit_postprocess_model_spec`); der Node
  lädt nichts. Jede Familie wird nur von ihrem eigenen Verbraucher
  materialisiert, Lifecycle und ZIT-Freigabe bleiben unverändert.
- Beide Familien sind gleichzeitig konfigurierbar; es gibt keinen
  Familienumschalter. Die Oberfläche zeigt die Bereiche `SDXL` und
  `Z-IMAGE TURBO` (`cmk_postprocess_model_source_ui.js`).
- Die Einzelquellen bleiben registriert und unverändert; die SDXL-Einzelquelle
  heißt in der Oberfläche jetzt `CMK PostProcess Model SDXL`.
- Wird in einem Workflow die Quelle gewechselt, ändern sich die
  graphbasierten Cache-Schlüssel nachgelagerter Boundaries einmalig.

#### ZIT-Modellquelle für PostProcess

`CMK Z-Image Turbo PostProcess Model` liefert eine nicht materialisierte
Z-Image-Turbo-Spezifikation. Vorbild ist die SDXL-Spezifikation von
`CMK PostProcess Model SDXL`.

```text
DIFFUSION MODEL + TEXT ENCODER + VAE + weight_dtype + clip_device → MODEL ZIT
```

```python
{
    "model_role": "postprocess",
    "model_family": "z_image_turbo",
    "model_architecture": "z_image_turbo",
    "diffusion_model_name": str,
    "text_encoder_name": str,
    "vae_name": str,
    "weight_dtype": str,
    "clip_device": str,
    "model_pipe_source": "CMK Z-Image Turbo PostProcess Model",
    "materialized": False,
}
```

- Die Spezifikation enthält ausschließlich primitive Werte und dieselben
  Identitätsfelder wie das MODEL von `CMK Z-Image Turbo Loader -Pipe-`.
  Gleiche Dateien ergeben dieselbe Modellidentität.
- MODEL-Forward, Gate und Boundary reichen sie unverändert weiter. Das
  Weiterreichen lädt weder UNet noch Textencoder noch VAE.
- Erst ein aktiver ZIT-Verbraucher materialisiert sie über
  `resolve_zit_postprocess_model`. Das Ergebnis ist eine Kopie; die
  Spezifikation selbst bleibt unverändert.
- SDXL- und ZIT-Spezifikation sind getrennt: SDXL-Helfer erkennen eine
  ZIT-Spezifikation nicht als SDXL, `resolve_postprocess_model` weist sie mit
  einer klaren Meldung zurück und `unload_model_pipe` greift für sie nicht auf
  den SDXL-Ressourcencache zu.

#### ZIT-Ressourcenlebenszyklus

Verbindlich gilt der bestehende ZIT-Lebenszyklus, gemeinsam für den ZIT-Loader
und die PostProcess-Spezifikation:

- Ein gemeinsamer Ressourcencache (`_ZIT_RESOURCE_CACHE`) für Loader und
  Spezifikation; gleiche Dateien werden nicht doppelt geladen.
- UNet und VAE bleiben gemäß der bestehenden Cache-Politik verfügbar. Ein
  Wechsel der Dateien ersetzt den Cache-Eintrag.
- Der Textencoder (Qwen) wird nach dem Encoding freigegeben und bei Bedarf
  neu materialisiert (`ensure_zit_text_encoder` / `evict_zit_text_encoder`).
- Es gibt keine globale Freigabe durch den Visualizer, keine Referenzzählung
  und keine zusätzliche Ressourcenverwaltung.

Wechsel ZIT → SDXL im Multi-Model-Detailer (2.7.1): Auf Unified Memory gibt
ComfyUI ein Modell erst frei, wenn keine Python-Referenz mehr besteht; ein
Entladen verschiebt die Gewichte nur von MPS in den Arbeitsspeicher. Bevor der
ausgeführte SDXL-Zweig des Multi-Model Prepare SDXL materialisiert, ruft er
deshalb `release_zit_resources` auf:

- Leerer ZIT-Ressourcencache: keine Aktion.
- Ist ein Textencoder im Eintrag aktiv oder ein Teil nicht per Weakref
  verfolgbar: `KEPT · IN USE`, Eintrag unverändert.
- Sonst werden die CMK-Referenzen entfernt und per Weakref geprüft, ob UNet
  (ModelPatcher und Gewichte) und VAE (Objekt und `first_stage_model`)
  tatsächlich frei wurden. Ein noch verwendeter Teil wird unverändert
  zurückgelegt und wiederverwendet; ein freigegebener Teil wird nie
  zurückgelegt. Teilweise Freigabe: Der Eintrag enthält nur den verbliebenen
  Teil, `_load_zit_resources` lädt beim nächsten ZIT-Bedarf nur den fehlenden
  Teil (`HIT · PARTIAL RELOAD`).
- Kein Entladen eines noch verwendeten Modells, kein globales Unload; nach
  einer echten Freigabe nur `cleanup_models_gc` und `soft_empty_cache`.
- Konsolenlog: `[CMK Model Lifecycle] ZIT -> SDXL Detailer: ZIT RESOURCES
  RELEASED | PARTIALLY RELEASED | KEPT · IN USE`.

Wirkung: Zwischen getrennten Prompts (ZIT-Lauf, danach SDXL-Lauf) wird ZIT
freigegeben. Innerhalb eines Prompts (ZIT → SDXL → ZIT) hält ComfyUI die
Ausgabe des ersten ZIT-Prepare; dann bleibt ZIT erhalten und wird ohne zweite
Kopie wiederverwendet. Ein Boundary-HIT führt Prepare nicht aus und gibt nichts
frei.

Verbindliche Modellquelle: In eigenständigen PostProcess-Workflows kommt
`MODEL ZIT` ausschließlich aus `CMK Z-Image Turbo PostProcess Model`. Der
`CMK Z-Image Turbo Loader -Pipe-` lädt sofort und gibt UNet und VAE als
Objekte aus; ComfyUI hält diese Ausgabe im Ausgabecache, solange die Node im
Prompt steht. Die Freigabe meldet dann korrekt `KEPT · IN USE`, und SDXL läuft
neben dem weiterhin referenzierten ZIT-Modell.

Realvalidierung (2026-11-12, M1 Pro, 16 GB Unified Memory, ComfyUI 0.45.2,
Standard-Cache `RAM_PRESSURE`, ZIT-Lauf und anschließender SDXL-Lauf ohne
Neustart):

| | ZIT Loader -Pipe- als Quelle | ZIT PostProcess Model als Quelle |
|---|---|---|
| ZIT-Lauf | 8 Steps, 86,35 s/Step, 14:55 min | 8 Steps, 88,03 s/Step, 14:36 min |
| Freigabe vor SDXL | `KEPT · IN USE` | `RELEASED · UNET RELEASED · VAE RELEASED` |
| SDXL-Checkpoint | 90,38 s, Lumina2 7.443,77 MB weiter geladen | 11,39 s |
| SDXL-Sampling | erster Step 123,83 s, abgebrochen | 30 Steps, 3,64 s/Step |
| SDXL gesamt | abgebrochen | 143,59 s |

Der SDXL-Lauf nach ZIT entspricht damit einem regulären SDXL-Lauf
(Referenz ohne vorherigen ZIT-Lauf: 10,78 s Checkpoint, 3,73 s/Step,
143,68 s).

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

Für Inpainting wählt der Anwender bereits in
`CMK Flow · 01 START HERE · Create Image` einen von vier aufgabenbezogenen
Modi:

```text
custom | replace | remove | extend
```

`01 START HERE` besitzt oberhalb der übrigen Einstellungen die zwei
Modellfamilien-Reiter `SDXL` und `Z-IMAGE TURBO`. Der gespeicherte
Backend-Vertrag lautet `model_family = sdxl | z_image_turbo`. SDXL zeigt den
vollständigen bestehenden Text2Image-/Inpaint-Vertrag. Z-Image Turbo bietet
Text2Image sowie ein eindeutig als experimentell gekennzeichnetes, allgemeines
maskiertes Inpaint. Die aufgabenbezogenen SDXL-Inpaint-Modi sind für ZIT nicht
implementiert. Beide Familien verwenden dieselben modellneutral dargestellten
Größenpresets; beim Wechsel vom generischen Standard auf ZIT-Inpaint wird
wegen des zusätzlichen Union-2.1-Patches 768x512 als sichere Ausgangsgröße
gewählt. Eine ausdrücklich gewählte andere Größe bleibt erhalten.

Der Z-Image-Pfad besteht aus einem kombinierten, ausschließlich auf
ComfyUI-Core aufbauenden Loader für Diffusionsmodell, Lumina2-Textencoder und
VAE, `CMK Sampler Prepare Z-Image Turbo -Pipe-`, dem vorhandenen generischen
`CMK KSampler -Pipe-` und `CMK Z-Image Turbo Finalize -Pipe-`. Prepare erzeugt
positives Conditioning, `ConditioningZeroOut`, im Text2Image-Pfad ein
`EmptySD3LatentImage` und das über `ModelSamplingAuraFlow` mit Shift 3 gepatchte
Modell. Experimentelles Inpaint verwendet IMAGE und MASK mit
`InpaintModelConditioning` sowie dem Union-2.1-Modellpatch. Finalize dekodiert
das Ergebnis und führt es als normales `IMAGE` in die gemeinsamen Module
`40/90`.

Das optionale `05 ControlNet ZIT` gibt neben PROCESS und LOG das unveränderte
authoritative IMAGE aus. Das intern erzeugte Canny-Kontrollbild verbleibt im
PROCESS. Dadurch kann `10 ZIT` sein Inpaint-Bild linear vom vorherigen Modul
beziehen, ohne eine parallele Umgehungsleitung von `01 START HERE`.

Der dazugehörige kompakte Subgraph `CMK Flow · 10 KSampler Z-Image Turbo`
kapselt diese vier Stufen vollständig. Sein öffentlicher Vertrag lautet
`PROCESS + IMAGE + LOG -> MODEL + PROCESS + IMAGE + LOG + diagnostic`. IMAGE
wird nur beim experimentellen Inpaint lazy angefordert. Dadurch bleibt
der Z-Pfad frei von SDXL-Refiner- und SDXL-LoRA-Abhängigkeiten und kann nach dem
Decode unmittelbar in die modellneutralen Nachbearbeitungsmodule wechseln.

Die Bildvorbereitung bietet `Fit`, `Crop` und das ausdrücklich
seitenverhältnisändernde `Stretch`. `Fit` und `Crop` erhalten das
Quellseitenverhältnis; `Center`, `Top`, `Bottom`, `Left` oder `Right` bestimmen
die Ausrichtung. Bild und Eingabemaske durchlaufen dieselbe Geometrie. Bei
`Fit` wird die nicht vom Quellbild belegte Zielfläche als Maske erzeugt;
`Extend Image` vereinigt sie mit einer optionalen Eingabemaske und kann daher
die Ziel-Canvas selbst für Outpainting vorbereiten. Bei aktivem Outpaint wächst
die Maske standardmäßig 32 Bildpixel in das Quellbild hinein. Diese
Überlappungszone liefert der Diffusion Kontext beider Seiten und verhindert
eine harte Naht an der ursprünglichen Canvas-Grenze; die Breite ist über
über den Advanced-Parameter `outpaint_overlap` einstellbar.
Synthetische Füllungen wie `noise`, `neutral`, `black` und `white` verwenden
innerhalb dieser Überlappung eine weich auslaufende Füllmaske. Die
authoritative Generationsmaske bleibt vollständig erweitert; nur der
vorbereitende Bildinhalt wird überblendet, damit keine neue harte Füllkante in
das Sampling gelangt.

`01 START HERE · Create Image` zeigt das letzte Bild seines ohnehin erzeugten
Diagnostic-Payloads zusätzlich als native Vorschau direkt auf der
ausgeführten Node. Damit sind Fit-/Crop-Ergebnis und angewandte Maskenfüllung
ohne separate Preview-Node sichtbar. Die vier fachlichen Ausgänge und der
Diagnostic-Vertrag bleiben unverändert.

Die direkte Vorschau enthält ausschließlich das Bild und keinen zusätzlichen
Diagnostic-Titelrahmen. Öffentliche Parameter und Anschlüsse tragen
anwenderverständliche Anzeigenamen in Großschrift; interne Anschlussnamen
bleiben für gespeicherte Workflows stabil. Technische Bezeichnungen in
Kleinschrift sind Advanced-Parametern vorbehalten.

Der optionale STRING-Anschluss `opt_prompt_pos` besitzt kein Textfeld. Sein
Inhalt wird – sofern angeschlossen und nicht leer – mit einem Zeilenumbruch
hinter `PROMPT POS` angefügt und danach als gemeinsamer positiver Prompt durch
den Flow geführt.

Der übergeordnete sichtbare `MODE` ist ein Dropdown mit `Text2Image` als
Standard und `Inpaint`. Im Modus `Text2Image` blendet die Oberfläche sämtliche
Inpaint-spezifischen Einstellungen aus; deren gespeicherte Werte bleiben für
das spätere Zurückschalten erhalten. Alte Workflows mit dem früheren Boolean
`INPAINT_MODE` werden beim Laden auf das Dropdown migriert.

`01 START HERE` schreibt die Auswahl als `inpaint_process_mode` in `PROCESS`.
Der bereits dort vorhandene positive Prompt beschreibt den gewünschten
Bildinhalt; ein zweiter Inpaint-Prompt existiert bewusst nicht. Der Sampler
führt die Aufgabe aus: `custom` übernimmt seine Advanced-Werte unverändert.
Die geführten Modi setzen dagegen vollständige, aufgabenbezogene Lösungen:
`replace` verwendet deterministisches Rauschen, damit die Silhouette des
entfernten Objekts keine semantische Vorgabe für den neuen Inhalt bildet.
Das Preset setzt `denoise` auf `1.00`, aktiviert Noise Mask und Context
Reference, deaktiviert Outpaint und lässt Anwender-Prompt sowie LoRAs aktiv.
`remove` füllt den maskierten Bereich mit deterministischem Rauschen. `01 START
HERE` legt vor der Familienaufteilung die wirksame Promptführung einschließlich
interner Subject-Unterdrückung fest. Ein vorhandener Anwender-Prompt beschreibt
die gewünschte Umgebung; ohne ihn greift eine neutrale interne
Hintergrundführung. Dadurch gilt derselbe Vertrag für SDXL und den ZIT-Finish
von HYBRID. LoRAs bleiben deaktiviert. `extend` setzt die Umgebung mittels
Navier-Stokes fort. Ist der Anwender-Prompt leer, erzeugt `01 START HERE` auch
hier eine neutrale interne Fortsetzungsbeschreibung für beide Familien. Für die
diffusionsbasierten Modi werden passende Kombinationen aus `denoise`,
Noise-Mask und Context Reference gesetzt. `fill_masked_area` ist damit kein
reiner Metadatenwert: `original`, `neutral`, `telea`, `navier-stokes`,
`black`, `white` und `noise` werden vor
der Inpaint-Latent-Erzeugung tatsächlich auf das authoritative Eingangsbild
angewandt. Diese Füllung entsteht bereits in `01 START HERE`; dessen
Diagnostic-Vorschau und dessen öffentlicher `IMAGE`-Ausgang zeigen daher
denselben tatsächlich weitergereichten Bildzustand. Der Sampler wendet die
Füllung nicht nochmals an.

Die vier Modi sind geführte Aufgabenlösungen und keine semantische
Objekterkennung. Für `remove` genügt die vom Anwender gesetzte Maske. Der
KSampler rekonstruiert den maskierten Bereich mit dem Anwender-Prompt oder der
neutralen internen Hintergrundführung und einer zusätzlichen negativen
Subject-Sperre. Der Refiner führt anschließend den regulären zweiten
Diffusionspass mit derselben wirksamen Promptführung aus. Source- und lokale
LoRAs werden nicht geladen. Log und Diagnostic weisen den geführten
Remove-Pfad vollständig aus.
Die tatsächliche Wirkung und die gesetzten Werte sind über `MODE INFO` und den
Tooltip von `PROCESS MODE` direkt in `01 START HERE` dokumentiert.

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
MODEL + PROCESS + IMAGE + LOG + VISUAL + GLOBAL ENABLE
    ↓
MODEL + PROCESS + IMAGE + LOG + VISUAL + diagnostic
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
DETAILER + optional LOG + optional diagnostic
         + LOCAL ENABLE + lokale Detailer-Parameter
    ↓
SEGS DETECTED
SEGS PROCEED
IMAGE PROCEED (optionaler Komfortausgang)
DETAILER IMAGE (internes vollständiges Ergebnis)
diagnostic
LOG
```

Verbindlich gilt:

- `DETAILER` bleibt gemeinsamer unveränderlicher Arbeitskontext und wird nicht seriell durch Execute-Instanzen geführt.
- `opt_log` und `opt_diagnostic` dürfen ausschließlich Dokumentation durch mehrere Instanzen transportieren. Sie beeinflussen weder Pixelberechnung noch Branch-Cache-Schlüssel.
- `IMAGE PROCEED` darf in parallelen Advanced-Modulen deaktiviert bleiben; die Zusammenführung verwendet `SEGS PROCEED`.
- `DETAILER IMAGE` dient Preview-/Provider-Pfaden, darf den verpflichtenden Merge-/Boundary-Ausgang aber nicht ersetzen.
- Bei mehreren internen Samplern deklariert ein Visual Provider alle zugehörigen Live-Node-IDs derselben Subgraph-Instanz und zeigt ihre Frames nacheinander im gemeinsamen Modulregister.

Zusammenführung:

```text
authoritative IMAGE + SEGS 1 ... N
    ↓
CMK SEGS CONCAT
    ↓
zusammengesetztes IMAGE

Basis-LOG + Basis-diagnostic
    ↓
Smart Detailer 1 → LOG + diagnostic
    ↓
Smart Detailer 2 ... N → LOG + diagnostic
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
Der Boundary stempelt das tatsächlich ausgegebene Bild als aktuelles öffentliches
Cache-Artefakt in einer kopierten `PROCESS`-Pipe. Ein nachfolgendes weiteres
Detailer-Modul verwendet damit das unmittelbare Detailer-Ergebnis und nicht
fälschlich weiterhin das ältere Refiner-Artefakt.

### 5.6a Multi-Model-Detailer

Neben den SDXL-Detailer-Nodes bestehen eigenständig registrierte
Multi-Model-Nodes. Die SDXL-Nodes bleiben in Verhalten, Schnittstellen,
gespeicherten Widgetwerten und Cache-Schlüsseln unverändert.

| Generation | Prepare | Execute | DETAILER-Typ |
|---|---|---|---|
| SDXL | `CMKDetailerPreparePipe` | `CMK_SmartDetailerPipe` | `CMK_DETAILER_PIPE` |
| Multi-Model | `CMKDetailerPrepareMultiModelPipe` | `CMK_SmartDetailerMultiModelPipe` | `CMK_DETAILER_MULTIMODEL_PIPE` |

Die getrennten DETAILER-Typen verhindern eine Querverdrahtung der
Generationen.

Prepare Multi-Model:

```text
MODEL ZIT (optional) + MODEL SDXL + PROCESS + IMAGE + optional LOG
    + optional LoRA ZIT + MODEL FAMILY (None / SDXL / Z-Image Turbo)
    ↓
CMK Detailer Prepare Multi-Model -Pipe-
    ↓
DETAILER (Multi-Model) + LOG + diagnostic
```

- `None` ist der vollständige Bypass ohne Modell-, SAM- oder LoRA-Anforderung.
- Nur die Modellquelle der gewählten Familie wird lazy angefordert.
- Jede Familie besitzt eigene Steps, CFG, Sampler, Scheduler und einen eigenen
  Sampling Entry (SDXL 95 %, ZIT 85 % als Ausgangswerte). Der gewählte
  Sampling Entry liegt im DETAILER-Pipe; der Multi-Model Smart Detailer hat
  keinen eigenen Regler.
- Die Cache-Identität des Multi-Model Smart Detailers verwendet ein eigenes
  Schema und berücksichtigt Familie, Sampling Entry, Sampler/Scheduler,
  ZIT-Modellidentität, geladene LoRAs und Model Shift.

Subgraph `CMK Flow · Detailer`:

```text
MODEL ZIT + MODEL SDXL + PROCESS + IMAGE + LOG + VISUAL + MODEL FAMILY
    ↓
MODEL ZIT + MODEL SDXL + PROCESS + IMAGE + LOG + VISUAL + diagnostic
```

- `CMK Detailer Family Enable` übersetzt MODEL FAMILY in das boolesche
  ENABLE von Module Bypass Gate und Preview & Compare.

`CMK Detailer Family Enable` als Steuer-Node (2.8.0), ein Node je Detailer:

- Oberfläche: drei Auswahlflächen `NONE | SDXL | ZIT` statt Dropdown;
  Standardbreite 450 px (`cmk_node_dimensions.js`). Gespeichert und
  übergeben werden die unveränderten Werte `None`, `SDXL`, `Z-Image Turbo`.
- Ausgänge: `ENABLE` (unverändert, Slot 0) und `MODEL FAMILY` (Slot 1) für
  den Eingang `model_family` des Multi-Model Prepare. Ohne Steuer-Node bleibt
  das Dropdown im Prepare maßgeblich; mit Steuer-Node blendet ComfyUI es aus,
  und der Prepare zeigt die Einstellungen der gewählten Familie.
- `cmk_prompt_family_inline` ersetzt vor der Ausführung jeden Link vom
  Ausgang `MODEL FAMILY` durch die wörtliche Auswahl des Steuer-Nodes. Gate,
  Boundary (Bypass-Zustand, Aufbewahrungsslot) und Cache-Schlüssel sehen
  damit denselben Prompt wie mit dem Dropdown. Eine selbst verlinkte Auswahl
  bleibt unverändert.
- Im Standard-Subgraphen ist die Auswahl mit der äußeren MODEL FAMILY
  verbunden; die Auswahlflächen sind dort ausgeblendet, das äußere Layout ist
  unverändert.
- `MODEL SDXL` ist die SDXL-PostProcess-Spezifikation; die Detailer-Boundary
  fordert sie als öffentlichen MODEL-Eingang an.
- `MODEL ZIT` ist die ZIT-PostProcess-Spezifikation. Der Ausgang `MODEL ZIT`
  ist ein Forward für nachgelagerte Multi-Model-Module.
- Aufbewahrung der Detailer-Boundary: Jede Boundary behält pro Node nur den
  zuletzt gespeicherten Eintrag (`latest_by_node.json`). Für den
  Multi-Model-Detailer gilt diese Regel pro Node **und Modellfamilie**
  (Slot `<unique_id>#sdxl` bzw. `<unique_id>#z_image_turbo`). Die Familie
  stammt aus dem `model_family`-Eingang des vorgeschalteten Multi-Model
  Prepare (`_detailer_retention_family`). Ein Wechsel SDXL ↔ ZIT trifft
  daher beide Male; geänderte Parameter ersetzen nur den Eintrag ihrer
  Familie. Cache-Schlüssel, Manifeste und Artefaktformat sind unverändert.
  Der SDXL-Detailer und alle anderen Boundaries behalten den einfachen Slot
  `<unique_id>`. Ein vor 2.7.1 unter dem einfachen Slot gespeicherter Eintrag
  bleibt lesbar und wird bei Wiederverwendung von der Familie übernommen.

### 5.6b Advanced Multi-Model Detailer (2.8.0)

Drei unabhängig konfigurierte Detailer aus einer Vorbereitung; jede
Modellfamilie wird höchstens einmal materialisiert, ZIT vor SDXL.

```text
CMK Detailer Family Enable ×3 ──MODEL FAMILY──┐
MODEL ZIT + MODEL SDXL + PROCESS + IMAGE + LOG ┴→ CMK Detailer Prepare Advanced Multi-Model -Pipe-
    → DETAILER 1/2/3 (Specs, keine Modelle) → CMK Smart Detailer Multi-Model -Pipe- ×3
    → CMK Detailer Merge Multi-Model (lazy, ZIT → SDXL, von None nur die Diagnose) → Detailer-Boundary → Gate
    Merge diagnostic 1/2/3 → CMK Preview Render ×3 · Merge diagnostic → CMK Preview Render
```

- **Advanced Prepare:** gemeinsame Einstellungen (SAM, Prompt aus 1st Pass,
  SDXL-Extras) und je Detailer MODEL FAMILY, Steps, CFG, Sampler, Scheduler,
  Sampling Entry, bei ZIT Model Shift. Promptless, keine LoRA je Detailer.
  Die Pipes enthalten Specs, Einstellungen und die Argumente des
  Multi-Model Prepare, aber weder Modell, VAE, SAM noch Conditioning.
  `ENABLE` ist wahr, solange mindestens eine Familie gewählt ist.
- **Smart Detailer Multi-Model:** Bei einem Branch-Cache-MISS ruft der Hook
  `_materialize_detailer` den unveränderten Multi-Model Prepare für diese
  Konfiguration auf (ZIT: Qwen laden, encodieren, freigeben, dann UNet; SDXL:
  zuvor `release_zit_resources`). Ein Memo je Prompt verwendet ZIT-Conditioning
  nur bei identischer Konfiguration (Encoder, Gerät, Text, keine LoRA) wieder.
  Der SDXL-only Smart Detailer und der Standard-Detailer sind unverändert.
  Der Smart Detailer Multi-Model hat keinen eigenen `ENABLE`-Schalter: Die
  Aktivierung erfolgt nur über MODEL FAMILY bzw. CMK Detailer Family Enable.
  Ein `enable` älterer API-Prompts schaltet weder Branch-Revision noch
  Boundary ab. Eine Migration gespeicherter 2.8-Entwicklungsstände gibt es
  nicht; die 2.8-Subgraphen und -Workflows tragen den aktuellen Vertrag.
  Bei MODEL FAMILY = None zeigt der Prepare nur noch MODEL FAMILY und der
  Smart Detailer (Familie aus dem verbundenen Prepare bzw. dem Advanced-Slot)
  kein Widget; alle Werte bleiben erhalten, die Node-Größe ändert sich nicht.
  Die Boundary des Standard-Subgraphen liest `DETAILER IMAGE`, nicht
  `IMAGE PROCEED`: OUTPUT IMAGE PROCEED = aus funktioniert auch beim
  Boundary-MISS, und Umschalten ändert den Boundary-Schlüssel nicht. Der
  Preview Render liest die Diagnose hinter dem Gate (kein Output-Root am
  Zweig, ein Boundary-HIT führt Prepare und Smart Detailer nicht aus).
  Paket-Subgraph und Showcase folgen bei der Paketübernahme.
- **Branch-Cache je Slot:** Der Schlüssel eines Advanced-Detailers enthält nur
  die gemeinsamen Einstellungen und seinen eigenen Slot; ein `None`-Slot meldet
  `DISABLED_REVISION`.
- **Merge:** fordert die Zweige lazy an, zuerst ZIT, dann SDXL; von einem
  `None`-Slot nur dessen Diagnose (Bypass), zuletzt. Ergebnisse, LOG und
  Diagnose werden nach Slot zugeordnet, unabhängig von der
  Ausführungsreihenfolge und vom Cache-Zustand.
- **Einzel-Diagnosen hinter dem Merge** (2.8.0 Plan B): Der Merge gibt die
  Diagnosen der drei Detailer unverändert an den Ausgängen `diagnostic 1..3`
  weiter (angehängt, die bisherigen Ausgangs-Slots bleiben). Die drei
  Einzel-Previews hängen dort und nicht direkt an den Smart Detailern: Ein
  Output-Node an einem Detailer ist ein eigener Output-Root, den ComfyUI in
  der zufälligen Root-Reihenfolge ausführt, also außerhalb der Reihenfolge
  ZIT → SDXL. Boundary, Cache-Schlüssel und `MERGE_SEMANTICS` sind
  unverändert. `tests/test_detailer_advanced_output_roots.py` verhindert
  Output-Pfade an einem Smart Detailer vorbei am Merge.
- **Boundary:** Alle drei `None` gelten als Bypass; die Aufbewahrung folgt der
  einfachen Regel (ein Eintrag je Boundary).
- **Subgraph** `CMK Flow · Detailer · Advanced`: drei Family Enable über ihren
  Smart Detailern, Prepare links, Merge, Boundary, Gate und Forwards unten.
- **Paket und Flow Browser:** `subgraphs/CMK Flow · Detailer.json` und
  `subgraphs/CMK Flow · Detailer · Advanced.json` sind die abgenommenen
  Vorlagen, unverändert bis auf `extra.CMKFlow`. Im Flows-Katalog ist
  `Detailer` der Primäreintrag, `Detailer · Advanced` seine Variante; die
  SDXL-Detailer bleiben ladbar, sind aber mit `published: false` ausgeblendet.
  MODULE WORKFLOWS 301/302 sind die beiden 2.8-Modul-Workflows. Außen trägt das
  Dropdown die Beschriftung `MODE`; die Anzeige `DISABLE`/`ZIT` des Standard-
  Detailers ist reine Frontend-Darstellung (`web/js/cmk_detailer_mode_ui.js`),
  die gespeicherten Werte bleiben `None`/`Z-Image Turbo`.
- **Globaler Schalter** `detailer_global_enable` (Dropdown `ENABLE`/`DISABLE`,
  Standard `ENABLE`, letzter Prepare-Eingang, am Außenknoten als `ENABLE`):
  Bei `DISABLE` gelten alle drei Detailer als None. Prepare fordert keine
  Modellquelle an, ihr `ENABLE`-Ausgang schaltet Gate und Preview ab, die
  Boundary-Prüfung sieht das Modul als aus. Die Detailer-Einstellungen bleiben
  erhalten; die Branch-Keys enthalten den Schalter nicht.
- **Räumliche Diagnose** (`utils/segs_diagnostic.py`, nur Bilddaten aus
  vorhandenen SEGS, keine Detektion, kein Sampling, keine Modelle): Der Smart
  Detailer ergänzt `Sampling ROI` (`crop_region` neben der Detektionsbox),
  `Mask` (`cropped_mask` projiziert) und `Pasteback` (Gewicht aus `SEGSPaste`,
  feather 5). Ältere Branch-Cache-Einträge erhalten diese Stufen beim HIT aus
  den gecachten SEGS. Der Merge liefert eine eigene Diagnose `01 · INPUT SEGS`,
  `02 · OVERLAP` (Maskenüberschneidung; ROI-Überschneidung nur als Referenz),
  `03 · COMPOSITE` (exakte Alpha-Anteile je Slot) und `04 · FINAL`; die
  Eingänge `diagnostic i` werden mit ihrem Zweig angefordert und nur
  weitergereicht.
- **Maskenbasierter Merge** (`merge_segs_collections_masked`, nur
  `CMK Detailer Merge Multi-Model`): jeder Zweig wird in Slot-Reihenfolge mit
  `SEGSPaste.doit` und den Pasteback-Werten des Smart Detailers
  (`BRANCH_PASTE_FEATHER`/`ALPHA`) auf das bisherige Composite gelegt:
  `previous * (1 - alpha) + processed * alpha`. Die Sampling-ROI bestimmt nicht
  mehr den Übernahmebereich. Vorher muss das Einfügen auf das Ausgangsbild das
  gespeicherte `branch_image` reproduzieren, sonst bricht der Merge ab. Der
  Detailer-Boundary-Key enthält `MERGE_SEMANTICS`, damit kein Ergebnis der
  früheren Rechteck-Regel wiederverwendet wird; andere Detailer-Keys bleiben
  unverändert. `CMK_SEGSConcate` behält `merge_segs_collections`.

### 5.6c MaskDetailer Multi-Model (2.8, Runde 1)

Diskreter Zwei-Modell-Aufbau nach dem Muster des Multi-Model-Detailers
(Referenz `CMK · Detailer.json`, Workflow `CMK · MaskDetailer.json`):
`CMK Detailer Family Enable` → `CMK Mask Detailer · Prepare Multi-Model`
(`CMKMaskDetailerPrepareMultiModel`) → `CMK Mask Detailer · Process Multi-Model`
(`CMKMaskDetailerProcessMultiModel`). Die öffentlichen Ausgänge sind die von
`CMK Mask Detailer Prepare` / `Process` (`CMK_MASK_DETAILER_PROCESS`,
`result_contract` `mask_detailer`); die Legacy-Nodes bleiben unverändert und
teilen sich mit dem Prepare die Helfer für Quelle, Maske, Prompt und Prozess.

- **Familie**: `None` / `SDXL` / `Z-Image Turbo`. Der Prepare fordert lazy nur
  das Modell der aktiven Familie an (`MODEL SDXL` bzw. `MODEL ZIT`); `None` lädt
  nichts, der Process reicht das Bild unverändert durch. ZIT nutzt die
  Bausteine des Detailers (`resolve_zit_postprocess_model`, Text-Encoder
  laden/entladen, `ConditioningZeroOut` als Negativ, `ModelSamplingAuraFlow`).
- **Maske statt Detektion**: kein SAM, kein Detektor; Quelle ist der
  `MASK DETAILER SOURCE` der Intake oder `PROCESS` + `IMAGE` mit Maske (CMK Load
  Image).
- **Prompt / LoRA / Seed**: Quellprompt plus eigener Zusatz, je Familie eine
  lokale LoRA; der Seed kommt aus `PROCESS` (0, wenn keiner vorhanden ist).
  Keine SDXL-Modellpatches, wie beim Legacy-MaskDetailer. Der Prompt läuft
  vor dem Encoden durch die zentrale Übersetzung (`translate_prompt`, wie
  CMK Detailer Prepare): SDXL positiv und negativ, ZIT positiv; der Prozess
  behält den Originaltext. Der Legacy-MaskDetailer übersetzt nicht.
- **Sampling Entry**: eigener Bereich 0–99 % (Denoise = (100 − Entry)/100,
  0.01–1.0, Standard 45 % ≙ 0.55). Die Detailer-Grenze (Denoise ≤ 0.5) gilt hier
  nicht, weil das Inpainting einer gemalten Fläche mehr Denoise braucht.
- **Diagnostik** (§9.1, wie Detailer): Der Prepare liefert eine Textkarte
  (`PLAIN_PREVIEW_NODES`) und wird über `opt_diagnostic` des Process in die
  Modul-Timeline `CMK Flow · MaskDetailer` verkettet (wie beim Smart Detailer).
  Der Process zeigt `01 Sampling ROI` (crop_region und Grenzen der gemalten
  Maske auf dem Quellbild), `02 Mask` (effektive Sampling-Maske nach MASK EXPAND
  als Overlay), `03 Pasteback` (eingefügtes Ergebnis × Pasteback-Gewicht aus
  MASK FEATHER, alpha 255) und `04 Difference` (nur bei Änderung). Die Geometrie
  stammt aus den SEGS, die `SEGSPaste` zurückschreibt (`detailer_spatial_stages`).
  Keine Source-, Detection- oder Result-Stufe. Bei `None` entsteht nur eine
  Textkarte. Die Diagnosen des Legacy-MaskDetailers bleiben unverändert.
- **Subgraph `CMK Flow · MaskDetailer`** (neben dem unveränderten
  `CMK Flow · MaskDetailer SDXL`): dessen Intake-Struktur bleibt –
  Result Unpack → `CMK Mask Detailer Intake` (Capture, Snapshot,
  Zustandsautomat) → `CMK Mask Detailer Flow Gate` (`READY`, Bypass = Unpack) →
  VisualCompare. Ersetzt sind Prepare/Process/Concat durch Prepare/Process
  Multi-Model (Timeline über `opt_diagnostic`). Öffentliche Ports nach dem
  CMK-Port-Vertrag wie `CMK Flow · Detailer`: Eingänge MODEL ZIT, MODEL SDXL,
  PROCESS, IMAGE, LOG, VISUAL, MODE; Ausgänge MODEL ZIT, MODEL SDXL (beide über
  `CMKModelForwardPipe` aus den Eingängen), PROCESS (Mask-Detailer-Prozess der
  Gate), IMAGE, LOG, VISUAL, diagnostic. Dazu
  Family Enable (`ENABLE` → Gate `enabled` und VisualCompare `enable`; ersetzt
  den Eingang `enabled`) und eine innere Preview Render. Intake-spezifisch:
  solange die Intake nicht READY ist (S0–S2, Maske leer), fordert der Prepare
  kein Modell an, materialisiert und sampelt nicht (Status `INTAKE PENDING`,
  Process `NO PAINTED MASK · BYPASS`), denn die Diagnose an Preview Render und
  Ausgang erzwingt die Ausführung von Prepare/Process. Der Seed kommt aus dem
  `PROCESS` des Flows, weil der Snapshot keinen trägt. Die Prepare-Karte nennt
  Intake-Status und Snapshot. Bild und Maske stammen aus dem gesperrten
  Snapshot (Maske = 1 − Alpha), die Diagnose-Geometrie aus den SEGS der
  Verarbeitung.
- Kein Branch- oder Boundary-Cache (wie Legacy).

### 5.7 FaceProcess-Modul

Öffentliche Moduloberfläche:

```text
MODEL + PROCESS + IMAGE + LOG + VISUAL + GLOBAL ENABLE + PROCESS MODE
    ↓
MODEL + PROCESS + IMAGE + LOG + VISUAL + diagnostic
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
+ optional LOG
+ optional diagnostic
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
LOG
diagnostic
```

Die Execute-Instanz führt angeschlossene `opt_log`- und `opt_diagnostic`-
Transporte intern in Anschlussreihenfolge mit ihrer eigenen Dokumentation
zusammen. Diese Transporteingänge sind vom Rechen-Fingerprint ausgeschlossen:
Sie verändern weder Auswahl noch Bildberechnung und dürfen daher einen gültigen
Branch-Cache nicht invalidieren. Auch bei einem Cache-Hit werden jedoch die
aktuellen Transportwerte mit dem gecachten lokalen Beitrag verbunden.

Bewusst nicht vorhanden ist weiterhin:

```text
kein FACE-Ausgang
kein unverändertes IMAGE-Ausgangskabel
```

Im Standardmodul übernimmt die einzelne Execute-Instanz den vollständigen
seriellen `LOG`-/`diagnostic`-Transport. Im Advanced-Modul werden mehrere
Execute-Instanzen ausschließlich dokumentarisch über `opt_log` und
`opt_diagnostic` verkettet; ihre Bildberechnungen bleiben durch die vom
Fingerprint ausgeschlossenen Transporteingänge modular und cache-unabhängig.
`SEGS PROCESSED` mehrerer Instanzen werden weiterhin über `CMK SEGS CONCAT` auf
das gemeinsame authoritative Eingangsbild angewendet. Jede FaceProcess-Instanz
darf darin ausschließlich die ausgewählten und tatsächlich von ihr
verantworteten Gesichter ausgeben. Das cache-stabile Branch-Bild wird über deren
räumlichen SEG-Support übernommen; unveränderte Geschwistergesichter sind kein
Branch-Beitrag.

Der Modulabschluss ist ein verpflichtender `CMK Boundary Cache`:

```text
MODEL + PROCESS + zusammengesetztes IMAGE + zusammengeführtes LOG
    ↓
CMK Boundary Cache
    ↓
öffentliche FaceProcess-Ausgänge, interner Comparer und Visual Provider
```

Der FaceProcess-Boundary serialisiert weiterhin ausschließlich das berechnete
Bild und das zusammengeführte Log. `MODEL` und `PROCESS` werden read-only durch
den aktuellen ComfyUI-Prozess geführt und sind damit nach FaceProcess wieder
frei mit jedem kompatiblen Flow-Modul kombinierbar.

Die äußere Standardoberfläche ist anzeigefrei und besitzt ausschließlich
`FACEPROCESS ENABLE` sowie das kleingeschriebene Combo `process mode`. Der
interne Comparer bleibt Bestandteil des Subgraphen, wird jedoch weder als
Proxy-Widget noch als Bildanzeige auf der äußeren Oberfläche exponiert. Das
äußere `process mode` bleibt ein regulärer Subgraph-Eingang und ist kein
Proxy-Widget. Ist der gleichnamige Eingang des inneren FaceProcess-Nodes
verkabelt, hält dessen dynamische UI vor der Prompt-Serialisierung beide
Parametersätze (Restore und Detailer) serialisierbar. Erst der eingehende Wert
wählt zur Laufzeit den wirksamen Satz. Bei einem unverkabelten Eingang bleibt
die normale dynamische Projektion auf den lokal gewählten Modus erhalten.

Der
`CMK Visual Provider -Pipe-` registriert das Ergebnis unter dem sichtbaren Label
`FaceProcess` mit Sequenz 30. Seine semantische Live-Quellenkennung lautet
ebenfalls `FaceProcess` und wird unabhängig vom gewählten `process mode` auf die
unterstützten konkreten FaceProcess-Node-Typen aufgelöst. Restore und Detailer
erzeugen daher niemals unterschiedliche Registeridentitäten.

### 5.7a FaceProcess 2.8 (in Arbeit: F1–F4)

Neue Nodes neben den unveränderten Legacy-Nodes; die Subgraphen
`CMK Flow · FaceProcess` / `· Advanced` folgen in F5.

```text
CMK FaceProcess Mode Enable ──PROCESS MODE──┐ (None / Restore / SDXL / Z-Image Turbo)
MODEL ZIT + MODEL SDXL + PROCESS + IMAGE + LOG ┴→ CMK FaceProcess Prepare Multi-Model -Pipe-
    → FACE (Spezifikation, keine Modelle) → CMK FaceProcess Multi-Model -Pipe- → CMK Face Boundary Cache
```

- **Mode Enable (F1):** eigene Steuerklasse; `ENABLE` ist aus bei `None`. Der
  Prompt-Handler `cmk_prompt_family_inline` setzt `PROCESS MODE` wie die
  Detailer-Familie als Literal ein; Detailer-Werte und -Prompts unverändert.
- **Prepare (F2):** lädt nichts (kein Modell, SAM, LoRA, Conditioning) und
  fordert je Modus nur an, was der Modus braucht: NONE und RESTORE nur IMAGE
  und LOG; SDXL zusätzlich PROCESS und MODEL SDXL; ZIT PROCESS und MODEL ZIT.
- **REFINE MODE** gehört zum Prepare (sichtbar bei RESTORE, SDXL, ZIT; bei
  NONE gilt Off) und erreicht die Ausführung über FACE; er ist Teil der
  Prepare-Identität des Branch-Schlüssels.
- **RESTORE MODEL / VISIBILITY** gehören ebenfalls zum Prepare (nur bei
  RESTORE sichtbar, VISIBILITY-Default 0.50) und erreichen die Ausführung über
  FACE. Nur RESTORE-Schlüssel enthalten sie (Backend-Identität des gewählten
  Modells); SDXL/ZIT schließen sie aus.
- **Execute bei NONE** zeigt kein Widget; Werte bleiben, keine Größenänderung.
- **OUTPUT IMAGE PROCEED** am Ausführungs-Node wie am Smart Detailer: aus
  bedeutet leerer Ausgang `IMAGE PROCEED`, SEGS, VISUAL und Diagnose bleiben;
  kein Teil des Cache-Schlüssels. Die Boundary liest nicht IMAGE PROCEED,
  sondern den angehängten Ausgang `FACE IMAGE` (wie `DETAILER IMAGE` am Smart
  Detailer Multi-Model), der unabhängig vom Schalter das verarbeitete Bild
  trägt; „aus“ funktioniert so auch beim Boundary-MISS. In Upstream-
  Fingerprints zählt der Schalter dieser beiden Multi-Model-Nodes nur für
  Links, die IMAGE PROCEED lesen (`OUTPUT_SWITCH_SLOTS` in
  `cmk_persistent_cache`); Umschalten hält die Boundary auf HIT.
- **Standard-Subgraph `CMK Flow · FaceProcess` (F5, Entwicklungsstand):**
  äußere Ports wie der Multi-Model Detailer, `MODE` (DISABLE / RESTORE / SDXL
  / ZIT) ist ein Subgraph-Eingang am `process_mode` des Prepare und des Mode
  Enable (dessen `ENABLE` steuert Gate und Compare). Das Frontend spiegelt den
  äußeren MODE wie beim Detailer in den inneren Prepare (beim Umschalten und
  beim Öffnen des Subgraphen), damit die Innenansicht den aktiven Satz zeigt.
  Preview Render und Compare hängen hinter dem Gate, nicht am Zweig.
- **Ausführung (F2):** kein eigener Enable- oder Modus-Schalter. NONE ist der
  vollständige Bypass (keine Detektion, kein Restore/Detailer, kein Refine,
  `DISABLED_REVISION`). Nur beim Branch-MISS wird materialisiert: RESTORE ohne
  Diffusionsmodell und ohne SAM; SDXL über die unveränderte Legacy-Vorbereitung
  nach `release_zit_resources`; ZIT über die ZIT-Vorbereitung des Multi-Model
  Detailer Prepare. Die Verarbeitung selbst ist der unveränderte Legacy-Pfad.
- **Branch-Schlüssel** (`cmk_faceprocess_multimodel_branch_v1`): Modus, eigene
  Einstellungen und der Prepare-Anteil des aktiven Modus. RESTORE ignoriert
  Modellquellen, PROCESS, SDXL-/ZIT- und Detail-Einstellungen; SDXL ignoriert
  ZIT- und Restore-Einstellungen, ZIT entsprechend SDXL. LOG und Diagnose sind
  Transport.
- **Boundary:** `CMKFaceBoundaryCache` unverändert im Schema; die neue
  Ausführung ist zusätzliche Branch-Klasse des Manifests, `process_mode = None`
  des Prepare gilt als Modul aus. Legacy-Schlüssel bleiben gleich.
- **RESTORE (F3):** YOLO erkennt und wählt aus (dieselbe Auswahl wie
  Coverage und Detailer). InsightFace ordnet jedem ausgewählten YOLO-Gesicht
  nur sein vollständiges Face-Objekt zu (`utils/face_matching.py`, Regel
  `iou_kps_margin_v1`: bester Kandidat IoU ≥ 0,30, mindestens 4 von 5
  Keypoints in der YOLO-Box, Abstand zum zweiten Kandidaten ≥ 0,20; ein
  InsightFace-Gesicht für zwei YOLO-Gesichter ist ein Konflikt). Nicht
  zugeordnete oder mehrdeutige Gesichter werden übersprungen, nicht ersetzt,
  und weder restauriert noch beansprucht; LOG und Diagnose unterscheiden „nicht
  erkannt“, „keine Zuordnung“ und „mehrdeutig“. Restauriert wird in der
  2.7-Reihenfolge (aufsteigende Fläche) über `RestoreFaceAdvanced` mit
  `assigned_faces` (106 Landmarken erhalten). Ohne YOLO-Detektor oder mit
  `restore_model = none` ist RESTORE ein Konfigurationsfehler.
- **Restore-Modelle (E5, `pipe/cmk_faceprocess_restore.py`):** GFPGAN mit
  genau der gewählten Datei; CodeFormer mit dem festen CMK-Referenzverhalten
  (Fidelity 0,72, danach 72 % Ergebnis / 28 % Eingang) und nur für die Datei,
  die das Backend tatsächlich lädt; GPEN nur `GPEN-BFR-512.onnx`. Alles andere
  wird abgelehnt, kein stiller Rückfall. Backend und Datei stehen in LOG und
  Diagnose und im Branch-Schlüssel. `restore_facedetection` und
  `restore_codeformer_weight` bietet der 2.8-Node nicht an.
- **SDXL/ZIT-Detailer (F4):** der 2.7-Detailer-Schritt (Detektion mit SAM,
  Detektor-Ersatz bei nicht gesichtsbezogenem Standard-Detektor,
  `SEGSDetailer`, `SEGSPaste` mit `detail_paste_feather`) im 2.8-Node, mit
  derselben YOLO-Auswahlfunktion wie RESTORE.
- **Refine (F4, E9):** feste Werte unverändert, wirkt nur auf die Gesichter des
  Zweigs: RESTORE nur auf zugeordnete und restaurierte, SDXL/ZIT auf die
  ausgewählten. Keine Auswahl heißt kein Refine; die ungefilterte
  Neuerkennung entfällt.
- **Verarbeitungsebenen (F4, `utils/face_layers.py`):** `SEGS PROCESSED` ist
  `CMKFaceLayerSEGS` (Unterklasse der stabilen SEGS) mit `cmk_layers` in
  Verarbeitungsreihenfolge: `quantize` (`round` vor Restore, `trunc` vor Refine
  nach einem Detailer; über die Vereinigung der Zweig-Regionen), `restore` je
  Gesicht (zurückgewarptes Ergebnis, Float32-Pasteback-Alpha, Visibility),
  `detailer` je SEG (Crop wie eingefügt, Feather eingerechnet), `refine`
  (gefiltertes Bild, Alpha 0,75 · Ellipse, vollständiger Wirkbereich).
  `compose_layers` wiederholt die Mischoperationen der 2.7-Pfade und
  reproduziert das Zweigergebnis innerhalb der Regionen exakt; außerhalb liegt
  nur die 2.7-Ganzbild-Quantisierung (0 auf dem 8-Bit-Raster, sonst ≤ 1/255).
  Branch-Schema `…_v3`.
- **Advanced (F5, `pipe/cmk_faceprocess_advanced_multimodel.py`):**
  `CMK FaceProcess Prepare Advanced Multi-Model -Pipe-` liefert FACE 1–3 im
  unveränderten Standard-FACE-Vertrag; jeder Zweig ist die abgenommene
  Ausführung mit eigenem Branch-Cache. Je Zweig MODE (über einen eigenen Mode
  Enable), REFINE, RESTORE MODEL/VISIBILITY und SDXL-/ZIT-Einstellungen;
  gemeinsame Einstellungen und promptloses Verhalten wie beim Advanced
  Multi-Model Detailer, globaler Schalter `faceprocess_global_enable`
  (ENABLE/DISABLE). Der Branch-Schlüssel enthält nur den eigenen Slot und die
  gemeinsamen Werte, die sein Modus liest (`advanced_prepare_excluded_inputs`).
  `CMK FaceProcess Merge Multi-Model` fordert lazy RESTORE → ZIT → SDXL an
  (NONE nur Diagnose, zuletzt) und komponiert in Slotreihenfolge D1 → D2 → D3
  mit `compose_layers` über die F4-Ebenen jedes Zweigs: tatsächliche Restore-,
  Detailer- und Refine-Masken samt Mischarithmetik, jeder Zweig auf dem
  Ergebnis der vorigen. Außerhalb aller Zweigregionen bleibt das Eingangsbild
  unverändert (keine Ganzbild-Quantisierung). Zweige aus einem anderen Bild
  werden abgewiesen. `diagnostic 1..3` reicht die Zweigdiagnosen unverändert
  weiter; die Previews hängen hinter dem Merge (kein Output-Root am Zweig).
  Boundary unverändert; Modul aus bei DISABLE oder drei NONE-Zweigen.
- **Räumliche Diagnostik (F5, `utils/face_diagnostic.py`):** nur aus
  vorhandenen Daten (Quellbild, YOLO-Detektion/-Auswahl, RESTORE-Zuordnung,
  F4-Ebenen); das Bild bleibt unberührt. Payload-Stufen (sichtbar ohne
  Source, Primary-/Refine-Ergebnis und Final, siehe 9.1): Zweig SDXL/ZIT: Source, Detection
  (erkannt/ausgewählt), Sampling ROI, Mask, Pasteback (Detailer-Ebenen-Alpha),
  Detailer, Differenz, Final. RESTORE: Source, Detection · Assignment, Restore
  Region, Restore Alpha, Visibility (Alpha × Visibility), Restore, Differenz,
  Final. Mit Refine zusätzlich Refine Mask (0,75 · Ellipse), Refine und
  Differenzen. NONE ohne. Ältere Branch-Cache-Einträge werden beim HIT aus
  ihren SEGS/Ebenen ergänzt (`with_face_stages`, idempotent). Merge:
  INPUT LAYERS (wirksame Alpha je Zweig), OVERLAP (Alpha-Überlappung, später
  oben), COMPOSITE (exakte Anteile D1 → D2 → D3), FINAL.
- **GFPGAN ist nicht deterministisch:** Der Decoder speist Zufallsrauschen ein;
  dieselbe Eingabe ergibt ohne festen Torch-Seed unterschiedliche Pixel (auch
  in 2.7). Der Branch-Cache hält einen Lauf fest.
- **Bedienung wie der Multi-Model Detailer:** Der Prepare hat exakt dessen
  Parametersatz (nur `process_mode` statt `model_family`), einschließlich
  `SAMPLING ENTRY` (SDXL, 95 %), `ZIT SAMPLING ENTRY` (85 %) und des
  optionalen Eingangs `LoRA ZIT` (nur im ZIT-Modus angefordert). SDXL und ZIT
  materialisieren über den unveränderten `CMKDetailerPrepareMultiModelPipe`:
  dieselbe Prompt- und Seed-Auflösung (1st Pass, sonst lokal; Seed aus PROCESS,
  sonst 0), dieselben LoRA-Regeln (SDXL: 1st Pass gleicher Familie oder lokale
  LoRA; ZIT: ZIT-1st-Pass vor `LoRA ZIT`, nie die SDXL-LoRA) und die
  ZIT-Freigabe vor SDXL. Der Einstieg wird mit der Detailer-Funktion
  `_sampling_entry_to_denoise` in Denoise umgerechnet; `detail_denoise` gibt es
  im 2.8-Node nicht mehr. Branch-Schema `…_v4`.

### 5.7b FaceRebuild v2 (2.8)

Neue Subgraph-Dateien `CMK Flow · FaceRebuild SDXL v2.json` und
`CMK Flow · FaceRebuild SDXL · Advanced v2.json` neben den unveränderten
SDXL-Subgraphen; der sichtbare Titel bleibt. FaceRebuild bleibt ein
SDXL-Modul, ZIT wird nur durchgereicht.

- **Ports** (CMK-Port-Vertrag wie `CMK Flow · Detailer`): Eingänge MODEL ZIT,
  MODEL SDXL, PROCESS, IMAGE, LOG, VISUAL, MODE, danach die Quellgesicht-Dateien;
  Ausgänge MODEL ZIT, MODEL SDXL (beide über `CMKModelForwardPipe` aus den
  Eingängen), PROCESS, IMAGE, LOG, VISUAL, diagnostic. MODEL ZIT wird weder
  materialisiert noch verarbeitet; am Visualizer gehört er an
  `POSTPROCESS MODEL ZIT` (nicht an `MODEL ZIT` der ZIT-Generierungsgruppe).
- **MODE** ENABLE / DISABLE (äußeres Widget `facerebuild_mode`, Darstellung über
  `cmk_detailer_mode_ui.js` wie die Advanced-Module): `CMK FaceRebuild · Mode`
  (`CMKFaceRebuildModeEnable`) liefert `ENABLE` an den Executor
  (`FACEREBUILD ENABLE`), die Bypass-Gate und VisualCompare; DISABLE ist der
  bestehende Modul-Bypass. Der Legacy-Schalter `FACEREBUILD ENABLE` bleibt in den
  SDXL-Subgraphen unverändert.
- **Executoren** `CMKInstantIDFaceRebuildSDXLV2` / `…AdvancedSDXLV2` erben die
  SDXL-Verarbeitung unverändert; nur die Diagnose (`_module_diagnostic`,
  `_bypass_diagnostic`) ist neu. Die Legacy-Executoren liefern bytegleich die
  bisherige Diagnose.
- **Diagnose** (`utils/facerebuild_diagnostic.py`, Geometrie aus dem
  Rebuild-Kontext und den Engine-Masken der Verarbeitung): Prepare-Textkarten
  mit kurzen Bezeichnungen (höchstens 52 Zeichen je Zeile, höchstens 18 Zeilen,
  so passt der Text in die Bildzone); Standard eine Karte, Advanced eine
  Modulkarte („FaceRebuild · Prepare“) und je aktivem Slot eine Karte
  („Slot n · Prepare“) in Verarbeitungsreihenfolge. Die Karten tragen
  `summary_card_layout = footer_rule`: `render_summary_panel` zieht dann den
  Abschnittsstrich über der Fußzeile, an dem `combine_preview_panels` die Karten
  ausrichtet – sonst zählte der ganze Textkörper als Info-Bereich und vergrößerte
  jede Bildkarte. Andere Textkarten bleiben unverändert. Standard: Rebuild ROI (Zielgesicht + Kopf/Hals-ROI),
  Inpaint Mask (Sampling-Maske), Pasteback (eingefügtes Ergebnis ×
  Pasteback-Gewicht, PASTEBACK NECK), Difference nur bei Änderung. Advanced:
  Target Faces (Slots, Selektor, Reihenfolge), Inpaint Masks, Pasteback in
  Pasteback-Reihenfolge (größtes Gesicht zuerst), Overlap nur bei überlappenden
  Slots, Difference. Keine Source-, Detection- oder Rebuilt-ROI-Stufe; Bypass
  und „kein aktiver Slot“ sind Textkarten.
- **Preview-Kette**: Die innere Preview Render hängt an der Bypass-Gate (nach
  Boundary Cache), nicht mehr direkt am Executor; bei DISABLE blockiert die Gate
  die Diagnose (keine alte Diagnose, kein Platzhalter).

### 5.8 Image Resize

`CMK Flow · Image Resize` ist ein modellfreies, familienunabhängiges
PostProcess-Modul. Es übernimmt den vollständigen öffentlichen Vertrag der
PostProcess Boundary und reicht `MODEL` sowie `PROCESS` unverändert weiter:

```text
MODEL (opt) + PROCESS + IMAGE + LOG + VISUAL
    → MODEL + PROCESS + IMAGE + LOG + VISUAL + diagnostic
```

Der zentrale Parameter `scale_method` wählt exklusiv zwischen Skalierung auf
die längste Kante, Skalierung auf eine Ziel-Megapixelzahl und proportionalem
Einpassen mit Padding. Nicht gewählte Verfahren werden nicht aufgerufen. Bei
deaktiviertem `SCALE ENABLE` wird das Eingangsbild identisch durchgereicht und
kein Cache-Eintrag erzeugt. Der vorhandene persistente Boundary-Cache wird mit
einem methodenspezifischen Fingerprint verwendet; Parameter inaktiver Verfahren
gehören deshalb nicht zum Cache-Schlüssel. Erfolgreiche Verarbeitung ergänzt
LOG, VISUAL und das CMK-Diagnostic-Format, ohne Modelle zu laden oder den
Model-Lifecycle zu verändern.

### 5.8a Upscale und Save

`CMK Flow · Upscale & Save` speichert einen Zwischen- oder Endstand und ist frei
in der PostProcess-Kette positionierbar. Verbindlich ist der Vertrag aus
Abschnitt 3.0 (Modulversion 1.4.0): Weitergabe, Upscaling und Speichern sind
getrennte Verantwortungen.

```text
PROCESS + IMAGE + LOG + VISUAL + SAVE ENABLED + FILENAME PREFIX
    ↓
IMAGE (= IMAGE IN) + LOG + VISUAL + diagnostic
```

`CMK Smart Upscaler -Pipe-` erzeugt ausschließlich die zu speichernde Kopie:

```text
IMAGE + LOG → IMAGE + LOG + diagnostic
```

`CMKUpscaleSaveBoundaryCache` materialisiert diese Kopie; bei unveränderten
Eingaben kommt sie aus dem Cache, ohne Upscaler-Ausführung. `CMK Save Endpoint`
ist ein endständiger OUTPUT_NODE ohne verbrauchte Ausgänge und schreibt über
`CMK Save Project Image -Pipe-` bei jeder ausgeführten Prompt-Ausführung mit
`SAVE ENABLED = true` genau eine neue Datei aus dem Boundary-Ergebnis:

```text
PROCESS + IMAGE + LOG + Speicherparameter → Datei (keine verbrauchten Ausgänge)
```

`CMKVisualProvider` hinter der Boundary veröffentlicht das Register
`Upscale & Save` mit den Kanälen `before` und `result`. `SAVE ENABLED = false`
ist echter Bypass ohne Upscaler-, Boundary- oder Speicherausführung.
Eingebettete Definitionen vor Version 1.4.0 bleiben unverändert lauffähig und
behalten ihre bisherige Semantik (Abschnitt 3.0).

Der Zielpfad wird unterhalb von ComfyUIs Output-Verzeichnis in dieser Reihenfolge
gebildet:

```text
OUTPUT FOLDER / Text2Image|Inpaint / optionales Datum / optionales PROJECT FOLDER
```

`Text2Image` beziehungsweise `Inpaint` stammt ausschließlich aus
`PROCESS.boolean_inpaint_mode` und `PROCESS.hybrid_inpaint_mode`. Die Inpaint-Strategien
Custom, Replace, Remove und Extend sowie Outpaint ändern die Kategorie nicht. Fehlen
beide Kennzeichen, ist die Bearbeitungsart unbekannt: Es entsteht kein Kategorieordner,
und das Log weist darauf hin. Relative Pfade werden bereinigt und dürfen das
Output-Verzeichnis nicht verlassen. Speichern darf keine Bildverarbeitung
rückwärts erzwingen, wenn die vorherige Modulgrenze unverändert ist.

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
- den kompakten Artefakttoken des unmittelbar vorgelagerten autoritativen
  Modulergebnisses;
- die eigenen lokalen Parameter;
- die eigene Aktivierung.

Er enthält keine Geschwisterinstanz. Deshalb verändert das Aktivieren von Detailer 2 nicht den Fingerprint von Detailer 1.

Bei `CMK Smart Detailer -Pipe-` sind `output_image_proceed`, `opt_log` und
`opt_diagnostic` ausdrücklich vom Rechen-Fingerprint ausgeschlossen, weil sie
nur optionale Ausgabe beziehungsweise Dokumentation transportieren und nicht
das Detailer-Ergebnis berechnen.

Bei `CMK FaceProcess -Pipe-` sind `opt_log` und `opt_diagnostic` aus demselben
Grund ausgeschlossen. Frische Ausführung, deaktivierter Passthrough und
persistenter Cache-Hit müssen alle denselben vollständigen Rückgabevertrag
`IMAGE/SEGS/ENABLED/LOG/diagnostic` liefern.

Persistente Branch-Caches speichern zusätzlich ein cache-stabiles internes Branch-Artefakt aus vollständig zusammengesetztem Branch-Bild und räumlichem Support der tatsächlich zugeordneten SEG-Crop-Regionen. Der Support darf niemals aus einzelnen Pixel-Differenzen zwischen Quelle und Ergebnis abgeleitet werden. `CMK SEGS CONCAT` verwendet dieses Artefakt bei Cache-Hits direkt. Der öffentliche Wert bleibt strukturell ein Impact-kompatibles `SEGS`; Ports und Kabel ändern sich nicht. Ein Cache-Hit muss pixelidentisch zu einer frischen Branch-Ausführung sein.

Das Branch-Artefakt trägt zusätzlich die Signatur seines autoritativen
Quellbildes. `CMK SEGS CONCAT` muss eine Montage ablehnen, wenn diese Signatur
nicht zum aktuellen Ausgangsbild passt. Diese Ablehnung ist ein letzter
Integritätsschutz; im Normalfall verhindert bereits der vorgelagerte
Artefakttoken den falschen Cache-Hit.

#### Module Boundary Cache

Nach `SEGS CONCAT` und – sofern das Modul unabhängige `LOG BLOCK`-Zweige
verwendet – `LOG CONCAT` materialisiert ein Boundary das vollständige
bildbezogene Modulergebnis. `PROCESS` ist grundsätzlich kein Ein- oder Ausgang
eines `CMK Boundary Cache`, sondern wird parallel über den jeweiligen
Process-Forward-Node weitergeführt.

Ein Boundary-Hit ist nur gültig, wenn zusätzlich zum eigenen Prompt-Fingerprint die gespeicherte Dependency-Manifest-Version exakt mit den aktuell materialisierten Branch-Caches übereinstimmt. Jede Branch-Cache-Datei besitzt dafür einen kleinen Revisionsmarker. Wird ein einzelner Detailer-/FaceProcess-Zweig unter demselben Rechen-Fingerprint neu geschrieben, ändert sich dessen Revision; der Boundary fordert daraufhin `SEGS CONCAT` und die jeweilige LOG-/Diagnostic-Zusammenführung erneut an und materialisiert erst danach ein neues vollständiges Modulergebnis. Ein veralteter Boundary darf niemals ein neu zusammengeführtes Ergebnis übergehen.

Branch- und Module-Boundary-Fingerprints sind instanzgebunden. Zwei formal identische parallele Node-Instanzen besitzen deshalb getrennte persistente Cache-Einträge.

Bei aktivem Modul müssen alle folgenden Verbraucher diesen Boundary verwenden:

- öffentliche Modul-Ausgänge;
- interne Image Comparer;
- nachfolgende Module;
- Save-/Exportpfade.

Bei deaktiviertem Modul gilt dagegen ein harter Lazy-Bypass am öffentlichen
Ergebnis: Nur die unveränderten Eingänge werden angefordert. Der komplette aktive
Pfad einschließlich Prepare, Execute, Merge, Preview und Boundary bleibt ruhig.
Diese Eigenschaft muss jedes Modul mit Bypass selbst garantieren und darf nicht
von der PostProcess Boundary oder einem anderen nachgelagerten Auswahlknoten abhängen.

Damit kann eine Änderung in FaceProcess nicht den gesamten Detailer erneut anfordern, und ein Save-/Comparer-Zweig kann keinen internen Compute-Knoten umgehen.

### 6.3 Unmittelbare Artefaktlinie zwischen Modulen

`PROCESS` darf keine Pixel transportieren, führt aber kompakte Identitäten der
bereits erzeugten öffentlichen Ergebnisse unter `cmk_cache_artifacts`. Jede
erfolgreiche Modulgrenze aktualisiert zusätzlich
`cmk_cache_current_artifact`. Dieser Token bezeichnet ausschließlich das
unmittelbar ausgegebene autoritative Ergebnis der letzten aktiven Stufe.

```text
Modul A erzeugt IMAGE A
    ↓ stamp_artifact(A)
PROCESS.current = Artefakt A
    ↓
Modul B verwendet Artefakt A + eigene wirksame Einstellungen
    ↓ erzeugt IMAGE B und stamp_artifact(B)
PROCESS.current = Artefakt B
```

Damit ist kein Modul an einen namentlich vorausgesetzten Vorgänger gebunden.
Ein Detailer darf auf Refiner, InstantID, 1st Pass, einen anderen Detailer oder
einen künftig ergänzten kompatiblen Bildprozessor folgen. Ein deaktiviertes
Modul stempelt nichts und lässt die vorherige aktuelle Artefaktidentität
unverändert. Die stage-spezifische Historie bleibt erhalten; für den nächsten
Verbraucher hat jedoch `cmk_cache_current_artifact` Vorrang. Alte Workflows ohne
diesen Wert dürfen nur über einen ausdrücklich dokumentierten Legacy-Fallback
auf einen bekannten Stage-Token bedient werden.

Mehrere gleichartige Module dürfen denselben Stage-Key verwenden. Jede aktive
Boundary überschreibt lediglich `current` mit ihrem eigenen Ergebnis und
aktualisiert den Stage-Eintrag. Dadurch funktionieren beliebig lange Folgen
gleichartiger Module ohne Positionsannahme oder Sonderregel.

### 6.4 Cache-Speicherung

Die persistenten Generation-Completion-Ergebnisse liegen paketlokal unter
`cmk-flow/cache/generation_completion/`. Sie bleiben über einen vollständigen
ComfyUI-Neustart erhalten, bis sie durch die Begrenzung des Caches, eine gezielte
Invalidierung oder die manuelle Bereinigung im Flow Browser entfernt werden.

Branch- und Boundary-Caches liegen seit 2.8.0 unter
`<ComfyUI-User-Verzeichnis>/cmk_cache/<scope>/` (standardmäßig
`ComfyUI/user/cmk_cache/`). Alle Scopes lösen diesen Ort über
`cmk_persistent_cache.cache_directory` auf, das ihn aus
`folder_paths.get_user_directory()` ableitet. Bis 2.7 lagen sie unter
`ComfyUI/temp/cmk/`; ComfyUI löscht sein Temp-Verzeichnis jedoch bei jedem Start
und Beenden (`app/assets/lifecycle.cleanup_temp_filesystem`), sodass sie keinen
Neustart überlebten. Alte Temp-Einträge werden nicht übernommen.

Die manuelle Aktion `Clear Persistent Cache` erfasst ausschließlich diese
paketlokalen Generation-Completion-Dateien und die ausdrücklich registrierten
CMK-Scopes unter `cmk_cache/`. In-Memory-Ergebnisse, ComfyUIs normaler
Execution Cache einschließlich RAM-Pressure-Modus sowie allgemeine Modell- und
Ressourcencaches sind nicht Bestandteil dieser Bereinigung.

Aktuelle Implementierung:

| Ebene | gespeicherte Nutzdaten | Format | zusätzliche Laufzeitdaten |
|---|---|---|---|
| Generation Completion | fertiges IMAGE, ressourcenfreier PROCESS, LOG, VISUAL, Metadaten und freigegebener Modelldeskriptor | paketlokales `safetensors` + JSON | keine Modell- oder Samplerreferenzen |
| Refiner Boundary | First-Pass-Bild, Refiner-Bild, LOG | `safetensors` + JSON | `MODEL` und `PROCESS` im aktuellen ComfyUI-Prozess |
| Smart-Detailer-Branch | SEGS, optionales Bild, Diagnostic, LOG BLOCK | lokales Pickle-Bundle + Revisionsmarker | keine öffentliche API |
| FaceProcess-Branch | Bild/SEGS, Status, lokaler Diagnostic-/Logbeitrag | lokales Pickle-Bundle + Revisionsmarker | aktueller vollständiger LOG-/diagnostic-Transport wird nach dem Cache-Lesen ergänzt |
| Detailer Boundary | Modulbild, LOG, Branch-Dependency-Manifest | `safetensors` + JSON + internes Manifest | `MODEL` und `PROCESS` im aktuellen ComfyUI-Prozess |
| FaceProcess Boundary | Modulbild, LOG, Branch-Dependency-Manifest | `safetensors` + JSON + internes Manifest | keine weitere Leitung |
| FaceSwap Boundary | PROCESS, Modulbild, LOG | `safetensors` + JSON | `MODEL` wird read-only aktuell durchgereicht |
| Z-Image Boundary (`z_image_boundary`) | ZIT-Bild, LOG, VISUAL-Provider des ZIT-Samplers (Schema `cmk_z_image_boundary_v3`) | `safetensors` + JSON, VISUAL als `.visual`/`.visual-tensors` | `MODEL` und diagnostic nur im aktuellen ComfyUI-Prozess (kein HIT nach Neustart) |
| Hybrid SDXL Boundary (`sdxl_hybrid_boundary`, max. 8) | SDXL-Handoff-Bild, First-Pass-LOG, VISUAL, diagnostic, Checkpoint-/VAE-Identität, ContentGuard-Status `PASS` / `BYPASS` / `NOT INSPECTED` | Pickle-Envelope mit SHA-256 (Schema `cmk_hybrid_sdxl_image_boundary_v2`) | keine Modellreferenzen |
| InstantID SDXL Handoff (`sdxl_handoff_boundary`, max. 8) | InstantID-Teilsampling für die Fortsetzung in Modul 15 | Pickle | unverändert, eigener Scope |

**Z-Image Boundary und VISUAL (F-03).** Im Modul „10 KSampler Z-Image Turbo“
(ab 0.1.2) führt der VISUAL-Ausgang des ZIT-Samplers in die Boundary
(`VISUAL`, eingehende Pipe als `VISUAL IN`), das Gate erhält VISUAL ausschließlich
vom Boundary-Ausgang. Wie bei den PostProcess-Boundaries (H-3) speichert die
Boundary bei MISS die vom Sampler ergänzten Provider und veröffentlicht sie bei
HIT selbst; VISUAL ist nicht Teil des Schlüssels. Ein HIT fordert daher weder
ZIT-Loader noch Prompt-Vorbereitung noch Sampler an, auch wenn ComfyUIs
Ausführungs-Cache die Sampler-Ausgabe nicht mehr hält. Eingebettete Definitionen
ohne diese Links behalten ihr bisheriges Verhalten.

**Hybrid SDXL Boundary und STOP AFTER HANDOFF.** Jeder tatsächlich berechnete
Hybrid-First-Pass wird unter einem STOP-unabhängigen Schlüssel gespeichert
(START-Felder ohne Wirkung auf den SDXL-Vorpass, darunter STOP, sind
ausgenommen). Bei STOP stellt das SDXL-Gate den Eintrag vor jeder Anforderung von
MODEL oder SAMPLED wieder her: IMAGE, PROCESS, LOG (alle Blöcke plus Block
„SDXL First Pass · Hybrid Boundary“), VISUAL, diagnostic und die finale
Stufenvorschau. MODEL ist dann der freigegebene Generationsdeskriptor
(`model_role = generation_result`, `UNLOADED`), SAMPLED ist blockiert. Die
ZIT-Seite liest denselben Eintrag, aber nie im Prompt, der den First Pass gerade
berechnet hat; dort läuft der frische Handoff einschließlich SDXL-Freigabe.
Die finale Stufenvorschau sendet bei einem HIT das SDXL-Gate
(`CMKFamilyBranchGateSDXLSampled`) statt des nicht ausgeführten KSamplers; der
Visualizer akzeptiert dieses Gate deshalb zusätzlich als Live-Quelle des
First-Pass-Providers (`LIVE_PREVIEW_RELAYS` in `cmk_visualizer.js`), ohne die
Provider-Identität zu ändern.
Ein Eintrag mit `NOT INSPECTED` wird bei aktivem ContentGuard vor jeder Verwendung
geprüft; eine Ablehnung entfernt ihn und führt zur unveränderten Neuberechnung.
STOP-Einträge werden vor dem Speichern geprüft. Der Speicherort liegt unter
`ComfyUI/temp/` und übersteht keinen ComfyUI-Neustart.

**Regionalprompts im SDXL → ZIT-Finish.** Regional Conditioning gilt nur für die
SDXL-Generierung; der SDXL-Refiner darf das zugehörige Conditioning weiterverwenden.
Alle anderen nachgelagerten Verfahren erhalten nur globale semantische Inhalte, ohne
regionale Tensoren und ohne Nachbildung des Layouts im Zielmodell. Regionale Nodes
liegen nur auf dem PROCESS-SDXL-Zweig. Ist der globale positive Prompt leer, liest
`CMKSamplerPrepareZImageTurboPipe` über den versteckten PROMPT die aktiven
Regionaltexte oberhalb des SDXL-Handoffs und führt sie in Kettenreihenfolge zu
einem globalen Text zusammen. Das ist eine feste Grenze: ZIT erhält nur diesen
Text. Koordinaten, Strength, Start/End, Masken, Conditioning und Regionsnamen
bleiben im SDXL-Pfad. Ein vorhandener globaler Prompt hat Vorrang.

**InstantID Handoff und STOP AFTER HANDOFF.** Modul 01 löst den wirksamen
InstantID-Modus an genau einer Stelle auf (`start_instantid_mode`): Inpaint ist
immer `HYBRID INPAINT`, das dort verborgene Widget hat keine Wirkung. STOP gilt
für `HYBRID` und `HYBRID INPAINT`; PROCESS- und Prompt-Auswertung verwenden
dieselbe Normalisierung. Der Handoff-Cache (`sdxl_handoff_boundary`) bedient beide
Modi. Bei Inpaint ist der Eintrag die x0-Schätzung nach der Inpaint Bridge, also
ein Latent ohne `noise_mask`; ContentGuard prüft x0 vor der Bridge. Weil der Prompt
Bild und Maske nur über den Dateinamen benennt, trägt in der Konfiguration
SDXL → InstantID Inpaint jeder Bild-Loader (`_IMAGE_FILE_LOADERS`) seinen
`IS_CHANGED`-Inhaltshash im Fingerprint. Das gilt für alle Fingerprints dieses
Prompts; alle anderen Konfigurationen behalten ihre Schlüssel bitgleich.

Diese Caches sind Beschleuniger, kein portables Projektformat. Nach einem Neustart, einer Codeänderung, einer geänderten Upstream-Konfiguration oder einer Cache-Bereinigung ist ein erster `MISS → STORED` erwartbar.

### 6.5 Warum nicht anders

| Verworfene Lösung | Grund der Ablehnung |
|---|---|
| serielle DETAILER-/FACE-/LOG-Durchleitung | macht frühere Instanzen zu formalen Vorgängern späterer Instanzen |
| nur sternförmige Parallelverkabelung | Merge-Nodes können unveränderte Zweige trotzdem erneut anfordern |
| vollständiges `LOG` an parallelen Execute-Instanzen | erzeugt Verarbeitungsketten und erlaubt Fehlverkabelungen; eine ausdrücklich serielle Modulfolge darf LOG/diagnostic dagegen dokumentarisch durchreichen |
| `STRING` statt `CMK_LOG_BLOCK` | ein beliebiger Text könnte an die Log-Zusammenführung angeschlossen werden; Logblöcke wären nicht typgeschützt |
| unverändertes `IMAGE` durch Execute-Nodes | fachfremder Transport, zusätzliche Invalidierung und Umgehung des Merge-Pfads |
| allein auf den normalen ComfyUI-Cache vertrauen | nicht hinreichend stabil bei Merge-Pfaden, Cache-Eviction und begrenztem Speicher |
| Comparer direkt an Compute-Ausgang | erzeugt einen zweiten, ungeschützten Ausführungspfad |
| sichtbare Sternverkabelung im Hauptworkflow | widerspricht der `-Pipe-`-Produktphilosophie; interne Parallelität darf nicht zur Anwenderaufgabe werden |
| Bilder im öffentlichen PROCESS verstecken | unklare Bildautorität, schwer nachvollziehbare Änderungen und unnötige Kopplung |
| Refiner mit externem First-Pass-IMAGE speisen | redundant; das korrekte First-Pass-Bild wird aus dem unverarbeiteten Sampler-Latent dekodiert |

## 7. LOG-Vertrag

`LOG` dokumentiert. Es verarbeitet nicht.

Ein vollständiger `CMK_LOG_PIPE` wird grundsätzlich an Modulgrenzen
weitergegeben. FaceProcess- und FaceSwap-Zweige geben unabhängige
`CMK_LOG_BLOCK`-Beiträge aus und werden per `CMK LOG CONCAT` gesammelt.
Smart-Detailer dürfen dagegen `opt_log` und `opt_diagnostic` seriell
durchreichen, weil diese Transportkette nicht ihre gemeinsame `DETAILER`-
Arbeitsgrundlage und nicht ihre Rechen-Cacheidentität verändert.

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

### 9.1 Grundregel: Preview zeigt das Ergebnis, Diagnostik den Vorgang (2.8)

Gilt für alle CMK-Module mit Preview- und Diagnosefunktion.

- **Preview / Compare** (Visualizer, Preview & Compare) zeigt die Bilder zur
  Qualitätsbeurteilung: Eingang, Ergebnis, Before/After.
- **Diagnostik** zeigt, wo, wie und mit welcher Wirkung ein Ergebnis entstand:
  Detektion und Auswahl, ROI, Masken, Alpha, Pasteback und Visibility,
  Refine-Masken, Überlappungen, Kompositionsanteile, Differenzen zwischen
  relevanten Stufen (z. B. Source → Primary, Primary → Refine, Source → Final),
  Zuordnungen und Status.
- **Keine Vollbilder in der Diagnosefolge:** Source, Primary-Ergebnis,
  Refine-Ergebnis, Final und die gewöhnliche Node-Vorschau („Preview N“)
  erscheinen nicht als Diagnosekachel. Jede Kachel beantwortet eine eigene
  technische Frage; eine Kachel ohne Informationsgewinn entfällt.
- **Umsetzung:** Die Erzeuger behalten diese Bilder in ihren Payloads (Caches,
  Rekonstruktion, andere Verbraucher); die Diagnose-Renderer (`CMK Preview
  Render`, `CMK Preview Board`, `utils/comfy_preview_output.py`) lassen sie
  aus. Zugeordnet wird je (Node, Stufentitel) in `utils/cmk_diagnostic.py`
  (`PLAIN_IMAGE_STAGES`, `PLAIN_PREVIEW_NODES`), weil derselbe Titel woanders
  technisch sein kann (die „Source“ von `CMK Face Crop` zeigt die Auswahl mit
  Boxen und Landmarken). Gecachte Diagnosen folgen der Regel ohne
  Neuberechnung. Bleibt einer Diagnose kein sichtbarer Schritt, zeigt der
  Renderer ihre Zusammenfassung als Textkarte statt eines Vollbilds.
- Neue Diagnosen benennen ihre Stufen eindeutig; reine Vollbilder einer neuen
  Node werden in die Tabelle aufgenommen. Bildverarbeitung, Caches,
  Lazy-Verträge und Preview/Compare bleiben davon unberührt.

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
CMK Checkpoint VAE Loader -Pipe- → MODEL SDXL (optional)
                                      ↓
CMK Image Load and Resize -Pipe- → MODEL + PROCESS SDXL + IMAGE + LOG + diagnostic
```

Daraus können `CMK Detailer Prepare -Pipe-` beziehungsweise `CMK FaceProcess
Prepare -Pipe-` direkt gespeist werden. Für deren MODEL-Eingang wird der durch
Image Input geschleifte echte SDXL-Modellkontext verwendet. Ohne Checkpoint
bleibt MODEL leer; 40 und 90 validieren den reinen Bildpfad über
PROCESS / IMAGE / LOG.

Die proprietären Arbeits-Pipes schützen die Execute-Schnittstellen, ohne Standalone-Verwendung zu verhindern.

## 12. Nicht verhandelbare Invarianten

Eine Änderung ist architektonisch unzulässig, wenn sie eine der folgenden Regeln verletzt:

1. `MODEL` wird von einem Modul mutiert.
2. Das authoritative Bild wird nur versteckt in `PROCESS` transportiert.
3. `LOG`, `LOG BLOCK` oder `diagnostic` steuert eine Bildberechnung oder ihre
   Cacheidentität.
4. Eine Execute-Instanz schleift `DETAILER` oder `FACE` zur nächsten Instanz
   durch. Serieller `LOG`-/`diagnostic`-Transport ist nur zulässig, wenn er die
   Pixelberechnung und deren Cacheidentität nicht beeinflusst.
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

Unveränderten Workflow erneut ausführen
→ kein Sampling
→ alle tatsächlich publizierten Register und Bilder kommen aus den jeweiligen Modul-Caches

Gleichartigen Bildprozessor ein zweites Mal einschleifen
→ getrennte Provider-Instanzen und Register, zum Beispiel Detailer und Detailer 2
→ gleiche sequence und gleicher Stage-Typ überschreiben einander nicht

Vorgeschalteten Detailer aktivieren oder deaktivieren
→ der nachfolgende Detailer verwendet jeweils einen getrennten Cache für das tatsächlich andere Eingangsbild
→ kein Zusammenführen von SEGS mit abweichender Quellbildsignatur

Modul auf GLOBAL BYPASS stellen
→ unverändertes IMAGE und vorheriger current-Artefakttoken werden durchgereicht
→ kein leeres oder mit einem Fremdbild gefülltes Visualizer-Register
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

#### LOG- und Diagnostic-Transport im parallelen FaceSwap-Modul

`CMK FaceSwap Image -Pipe-` besitzt die optionalen Eingänge `opt_log` und
`opt_diagnostic`. Beide Transporte werden intern mit dem eigenen Beitrag
zusammengeführt; der öffentliche Log-Ausgang ist ein normaler `LOG` vom Typ
`CMK_LOG_PIPE`.

Mehrere FaceSwap-Instanzen bleiben hinsichtlich ihrer Bildberechnung parallel,
werden für den rein dokumentarischen Transport jedoch seriell verkettet:

```text
Haupt-LOG -> FaceSwap A.opt_log -> A.LOG -> FaceSwap B.opt_log
          -> B.LOG -> FaceSwap C.opt_log -> C.LOG -> Boundary

FaceSwap A.diagnostic -> FaceSwap B.opt_diagnostic
FaceSwap B.diagnostic -> FaceSwap C.opt_diagnostic
FaceSwap C.diagnostic -> Modulausgang
```

`opt_log` und `opt_diagnostic` beeinflussen weder Gesichtserkennung,
Swap-Ausführung, SEGS-Zusammenführung noch deren Cache-Identität. Nachträgliche
`CMK LOG CONCAT`- oder `CMK Diagnostic Concat`-Nodes sind innerhalb der
Standard- und Advanced-40-Subgraphen nicht zulässig.

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

For Detailer and FaceProcess modules, GLOBAL OFF or LOCAL OFF is a hard execution boundary. Detector, SAM, model preparation, selection, processing, preview generation and persistent branch caching must be skipped. Disabled branches return the authoritative source image where their interface exposes it, ordinary empty SEGS and only neutral documentation in their declared LOG/diagnostic form. Boundary caches must not persist results when their branch dependency manifest is incomplete; this is the expected state for disabled branches.

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

Umsetzung (H-5): `CMK Face Boundary Cache` nutzt denselben Mechanismus wie die
Detailer-Boundary (K-5): interne Eingänge `IMAGE IN`, `LOG IN` und `diagnostic`,
Ausgang `diagnostic` nach `VISUAL`; das Gate bezieht `DIAGNOSTIC ACTIVE` von der
Boundary. Bei globalem ENABLE und vollständig deaktivierten lokalen Zweigen
gibt die Boundary IMAGE unverändert aus, ergänzt das Eingangs-LOG um den Block
`FaceProcess Boundary` (`STATUS : DISABLED PASSTHROUGH`), liefert eine schlanke
Diagnose und reicht VISUAL unverändert durch; nachgelagerte Module und der
Visualizer laufen. Ein HIT stellt IMAGE, LOG, Diagnose und VISUAL aus dem
gespeicherten Eintrag wieder her, ohne Prepare, Zweige oder Modellauflösung.
Einträge ohne gespeicherte Diagnose werden beim ersten Zugriff einmalig
verworfen. Eingebettete Definitionen ohne die neuen Verbindungen behalten das
frühere Verhalten.

Der FaceProcess-Boundary-Key schließt die transportierenden Eingänge `IMAGE`
und `LOG` (sowie `IMAGE IN`, `LOG IN`, `diagnostic` und VISUAL) von der eigenen
Node-Signatur aus. Die inhaltliche Cache-Validierung
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

### Detailer: Passthrough ohne Prepare (K-5)

`CMK Detailer Boundary Cache` besitzt die internen Eingänge `IMAGE IN` und
`LOG IN` (Modul-Eingänge) sowie `diagnostic` (Diagnose des Compute-Nodes) und
den zusätzlichen Ausgang `diagnostic` nach `VISUAL`. Das Ergebnis-Gate bezieht
`DIAGNOSTIC ACTIVE` von der Boundary, nicht mehr direkt vom Compute-Node.

Bei globalem ENABLE und vollständig deaktivierten lokalen Zweigen fordert die
Boundary ausschließlich `IMAGE IN`, `LOG IN`, MODEL und `VISUAL IN` an. Sie gibt
IMAGE unverändert aus, ergänzt das Eingangs-LOG um den Block
`Detailer Boundary` (`STATUS : DISABLED PASSTHROUGH`), erzeugt eine schlanke
Diagnose ohne Compute und reicht VISUAL unverändert durch. Prepare, Compute und
`resolve_postprocess_model` werden nicht ausgewertet.

Bei aktiver Verarbeitung speichert die Boundary die Diagnose neben IMAGE, LOG
und VISUAL (Schema `cmk_detailer_boundary_v4`) und stellt sie bei einem HIT
ohne Compute wieder her. Die neuen Eingänge sind vom Boundary-Fingerprint
ausgeschlossen; der Schemawechsel führt einmalig zu einem MISS. Eingebettete
Definitionen ohne diese Verbindungen behalten das frühere Verhalten.

### Abhängigkeitsmanifest und Detailer-HIT (K5-B)

Detailer- und FaceProcess-Boundary speichern ein Manifest der Zweig-Revisionen.
Scope und Schlüssel jedes Zweigs kommen ausschließlich aus der Zweig-Klasse
(`dependency_cache_key` von `CMK_SmartDetailerPipe` bzw. `CMKFaceProcessPipe`),
also aus derselben Berechnung, unter der der Zweig seine Revision ablegt. Die
Boundary hält keine eigenen Schema-Konstanten oder Eingangsausschlüsse.

Ein Detailer-HIT übernimmt MODEL aus dem aktuellen Eingang und stellt IMAGE,
LOG, Diagnose und VISUAL aus dem gespeicherten Eintrag wieder her; es gibt
keine Session-Kopie von MODEL mehr. Ein gültiger Eintrag ist damit nach
Eviction und nach einem Neustart nutzbar, solange die Dateien unter
`cmk_cache/` erhalten sind.

Ein lokal deaktivierter Zweig speichert keinen Pickle und veröffentlicht keine
Revision. Seinen Manifest-Zustand liefert deshalb ebenfalls die Zweig-Klasse
(`dependency_revision`, K5-C):

- `enable` im Prompt ausdrücklich `False` ⇒ Marker `disabled`,
- aktiv ⇒ die tatsächlich veröffentlichte Revision,
- aktiv ohne Revision ⇒ Manifest unvollständig (kein HIT),
- verlinkter oder nicht eindeutiger `enable`-Eingang ⇒ aktiv.

Der Zweig-Schlüssel enthält `enable`; ein Umschalten ändert daher den
Manifest-Eintrag (FaceProcess: Boundary-Schlüssel ohne Upstream, Invalidierung
nur über das Manifest) bzw. zusätzlich den Boundary-Schlüssel (Detailer). Nur
neu aktivierte Zweige rechnen; die übrigen nutzen ihren Zweig-Cache. Eine
Boundary hält je Node nur ihren letzten Eintrag; die Rückkehr zu einer früheren
Einstellung berechnet die Boundary einmal aus den Zweig-Caches neu.

### FaceRebuild-Boundary: persistente HITs ohne Session

`CMK FaceRebuild Boundary Cache` hat kein Abhängigkeitsmanifest; der Schlüssel
umfasst den gesamten Compute-Node einschließlich der `FACE n ENABLE`-Schalter.
Ein Eintrag besteht aus IMAGE/LOG, der Diagnose (eigene Sidecar-Datei) und den
VISUAL-Providern (H-3). Ein HIT setzt nur diese Dateien voraus und führt weder
Compute noch Modellauflösung aus; er ist nach Eviction und nach einem Neustart
nutzbar. Einträge ohne gespeicherte Diagnose werden einmalig neu berechnet.

MODEL: Der MODEL-Ausgang der Boundary wird von kanonischen Definitionen nicht
gelesen (OD-2, öffentliches MODEL über `CMK Model Forward -Pipe-`); ein HIT
fordert dann kein MODEL an. Liest eine eingebettete Legacy-Definition den
Ausgang (Gate `MODEL ACTIVE` ← Boundary), erhält sie die Session-Kopie aus
demselben Prozess oder, ohne Session, den angeforderten Eingang; dieser
Eingang kommt vom Compute-Node und führt ihn aus. Schlüssel, Schema und
Subgraphen sind unverändert.

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

### Inaktive Gates vor der Cache-Schlüsselbildung entkoppeln

ComfyUI nimmt jeden verlinkten Vorfahren in die Cache-Signatur einer Node auf,
auch den ACTIVE-Pfad eines Lazy-Gates, das BYPASS wählt. Ohne Gegenmaßnahme
invalidierten Regler und Bildreferenzen eines deaktivierten Moduls jede
nachgelagerte Verarbeitung (ComfyUI-Schlüssel und CMK-Fingerprints).

Der Prompt-Handler `cmk_prompt_gate_pruning.on_prompt` (registriert über
`PromptServer.add_on_prompt_handler`, vor Validierung und Queue) entfernt
deshalb die Eingänge `* ACTIVE` von Gates, deren Entscheidung der Prompt
bereits enthält. Einziger Owner der Entscheidung ist
`cmk_distribution.gate_prompt_state`:

| Gate | inaktiv, wenn |
|---|---|
| `CMKModuleBypassGate` | `ENABLE` literal `False`, oder literal `True` und STOP AFTER HANDOFF wirksam (`stop_after_handoff_prompt_state`, wie zur Laufzeit) |
| `CMKSamplerBypassGate` | `ENABLE` literal `False`, oder `ENABLE` ← `CMKProcessEnableFlag(flag="instantid_enabled")` mit genau einer Modul-01-Quelle, die zugleich die einzige Modul-01-Node oberhalb des Gates ist, und dort `instantid_enabled` literal `False`, eine Nicht-SDXL-Familie oder wirksames STOP AFTER HANDOFF |

Jeder andere Zustand ist `unknown`; dann wird nichts entfernt. Würde die
Bereinigung eine Gate-Entscheidung verändern, unterbleibt sie vollständig.
Der Handler arbeitet auf einer Kopie und gibt bei jedem Fehler die
unveränderte Anfrage zurück. Gespeicherte Workflows und das Frontend bleiben
unberührt; nur der ausführbare Prompt (und damit `/history`) ist bereinigt.

Die entfernten Links stehen in `_meta.cmk_pruned_inputs` des Gates. `_meta`
gehört weder zur ComfyUI-Signatur noch zum CMK-Fingerprint. Ein aus `/history`
erneut gesendeter Prompt erhält die Links vor der erneuten Bereinigung zurück;
wird das Gate dabei aktiviert, ist der ACTIVE-Pfad vollständig vorhanden.
Fehlen die Links ohne gespeicherte Kopie (manuell veränderter API-Prompt),
endet die Ausführung mit einem Fehler, nie mit einem stillen Bypass-Ergebnis.

`CMK_PROMPT_GATE_PRUNING=0` schaltet die Bereinigung ab (Rollback); das
Zurückschreiben gespeicherter Links bleibt aktiv. Einführung und Rückbau
ändern einmalig die Schlüssel nachgelagerter Knoten in Prompts mit inaktiven
Gates; bestehende persistente Einträge bleiben gültig und werden nur nicht
getroffen.

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
