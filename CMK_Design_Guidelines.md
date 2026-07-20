# CMK Design Guidelines

Dieses Dokument ergänzt `ARCHITECTURE.md`. Es definiert ausschließlich Darstellung, Benennung und UI-Verhalten. Schnittstellenrollen und Datenverantwortung werden nur in `ARCHITECTURE.md` verbindlich festgelegt.

## 1. Sichtbare Standardsprache

Standard-Sockets und zentrale Entscheidungen werden kurz und in Großbuchstaben dargestellt:

```text
MODEL
PROCESS
IMAGE
LOG
SAMPLED
SAMPLER
REFINER
DETAILER
FACE
ENABLE
LOG BLOCK
```

Technische Advanced-Parameter dürfen präzise interne Namen verwenden:

```text
bbox_threshold
start_percent
end_percent
stop_at_clip_layer
```

Standard und Advanced dürfen nicht dieselbe Entscheidung doppelt anbieten.

## 2. Node-Namen

- Jede sichtbare Node beginnt mit `CMK`.
- Zusammengesetzte Produktbegriffe bleiben zusammengeschrieben: `FaceProcess`, `FaceSwap`, `ControlNet`.
- Prepare-Nodes: `CMK <Modul> Prepare <Technologie> -Pipe-`.
- Execute-Nodes: `CMK <Funktion> -Pipe-`.
- Zusammenführungen verwenden einheitlich `CONCAT`.
- Interne Caches heißen sichtbar neutral `CMK Boundary Cache`.
- Eine Bezeichnung darf keine nicht vorhandene Funktion versprechen; insbesondere kein `Upscale` im reinen Refiner-Modul.

## 3. Socket-Reihenfolge

Bei `-Pipe-`-Nodes stehen Transport- und Arbeits-Pipes zuerst:

```text
MODEL
PROCESS
IMAGE oder SAMPLED
LOG
```

Danach folgen:

```text
GLOBAL/LOCAL ENABLE
Standardparameter
Advancedparameter
```

Ausgänge folgen derselben semantischen Reihenfolge, soweit der konkrete Node-Vertrag diese Rollen tatsächlich ausgibt.

Ein Compute-Knoten darf keine unveränderten Transportwerte allein aus optischen Gründen durchschleifen.

## 4. Dynamische Eingänge

Dynamische Familien werden fortlaufend nummeriert und zeigen genau einen freien Folgeeingang:

```text
SEGS 1
SEGS 2
SEGS 3
...

LOG BLOCK 1
LOG BLOCK 2
LOG BLOCK 3
...

DIAGNOSTIC 1
DIAGNOSTIC 2
DIAGNOSTIC 3
...
```

Regeln:

- verbundene Eingänge werden niemals automatisch entfernt;
- hinter dem höchsten verbundenen Eingang bleibt genau ein leerer Anschluss;
- unnötige weitere leere Anschlüsse werden entfernt;
- die Node-Höhe wird automatisch angepasst;
- technische interne Namen wie `segs_2` dürfen gespeichert bleiben, die sichtbare Beschriftung ist jedoch `SEGS 2`.

## 5. Enable-Logik

Optionale Module unterscheiden:

```text
GLOBAL ENABLE
LOCAL ENABLE
FUNCTION / MODE
```

- `GLOBAL ENABLE` schaltet das gesamte Modul.
- `LOCAL ENABLE` schaltet eine einzelne parallele Instanz.
- Ein Moduswähler enthält nur reale Funktionen.
- Ein zweiter optionaler Verarbeitungsschritt darf `Off` enthalten, wenn dieser Schritt keinen eigenen Enable-Schalter besitzt.

Diagnostik nennt beide Aktivierungsebenen eindeutig.

## 6. Standard versus Advanced

Standard zeigt die Absicht des Anwenders:

```text
ENABLE
MODEL
MODE
STRENGTH
STEPS
CFG
SAMPLER
SCHEDULER
```

Advanced zeigt technische Feinsteuerung:

```text
Dilation
Drop Size
Sampling-Modus
ZSNR
PAG
FreeU-Einzelwerte
Start-/End-Prozente (`0–100`)
```

Ein Parameter gehört nur dann in Standard, wenn er regelmäßig eine bewusste fachliche Entscheidung verlangt.

Prozentparameter werden in der Advanced-Oberfläche technisch und klein als
`start_percent` und `end_percent` bezeichnet. Sichtbare Werte verwenden den
vertrauten Bereich `0–100`; erst intern werden sie für ComfyUI auf `0.0–1.0`
normiert.

## 6.1 Äußere Standardgröße der Flow-Subgraphen

Die großen nummerierten Flow-Subgraphen verwenden in geschlossenem Zustand
einheitlich:

```text
Breite: 600
Höhe:  1225
```

Die Größe wird sowohl als `size: [600, 1225]` als auch persistent in
`properties.cmkOuterSize: [600, 1225]` gespeichert. Standard- und
Advanced-Varianten weichen davon nur ab, wenn eine ausdrücklich dokumentierte
fachliche UI-Anforderung eine andere Außenfläche benötigt.

## 7. Diagnostik

`diagnostic` ist passiv und lokal.

Eine Diagnose soll mindestens enthalten:

```text
STATUS
GLOBAL ENABLE
LOCAL ENABLE
MODE oder FUNKTION
RELEVANTES ERGEBNIS
FEHLERGRUND oder BYPASS-GRUND
```

Diagnostik darf:

- Vorschauen anzeigen;
- Parameter und effektive Entscheidungen zusammenfassen;
- Fehler klar benennen.

Diagnostik darf nicht:

- den Workflow reparieren;
- Verarbeitungsparameter verändern;
- einen alternativen Bild- oder Pipe-Transport bilden.

## 8. Preview und Comparer

- Preview zeigt nur fachlich relevante Ergebnisse.
- Bei ControlNet ist das verarbeitete ControlNet-Bild die semantische Vorschau, nicht das rohe Referenzbild.
- Ein interner Comparer zeigt verarbeitete Ergebnisse ausschließlich hinter der jeweiligen Modul-Boundary.
- Preview- und Comparer-Nodes sind keine Architekturgrenzen und dürfen Compute-Nodes nicht umgehen.

## 9. Fehlervermeidung

Bevorzugte Reihenfolge:

```text
1. ungültige Verbindung durch Socket-Typ verhindern
2. ungültige Auswahl durch UI verhindern
3. beim Start klar validieren
4. verständliche Fehlermeldung mit konkretem fehlendem Feld
```

Stille Fallbacks sind nur zulässig, wenn sie fachlich eindeutig und im Diagnostic/LOG erkennbar sind.

## 10. LOG-Darstellung

Ein vollständiges `LOG` und ein lokaler `LOG BLOCK` sind sichtbar und technisch verschieden.

```text
LOG       → CMK_LOG_PIPE
LOG BLOCK → CMK_LOG_BLOCK
```

Logblöcke verwenden eine konsistente Struktur:

```python
{
    "order": int,
    "title": str,
    "lines": [str, ...],
    "enabled": bool,
}
```

Logausgaben enthalten keine Tensoren, Masken, Modelle oder Cache-Objekte.

## 11. Kategorien

Die erste Kategorieebene bildet die Produktrolle ab, nicht die Python-Datei:

```text
CMK/Flow
CMK/Toolbox
CMK/Developer
```

Darunter gilt:

```text
CMK/Flow/Input
CMK/Flow/Process
CMK/Flow/Finish

CMK/Toolbox/Image
CMK/Toolbox/Mask & SEGS
CMK/Toolbox/Face
CMK/Toolbox/Video
CMK/Toolbox/ControlNet
CMK/Toolbox/Model & LoRA
CMK/Toolbox/I-O
CMK/Toolbox/Diagnostics

CMK/Developer/Pipe/*
CMK/Developer/Boundary & Cache
CMK/Developer/Diagnostics
CMK/Developer/Legacy
```

`Flow` enthält ausschließlich geführte öffentliche Einstiege, Prozessmodule und den verbindlichen Abschluss. `Toolbox` wird nach fachlicher Anwenderaufgabe gegliedert. `Developer` enthält technische Compose-, Prepare-, Execute-, Peek-, Set-, Forward-, Boundary- und Cache-Infrastruktur.

Boundary-Cache-Nodes sind interne Infrastruktur und werden mit `DEV_ONLY = True` registriert. Sie bleiben für gespeicherte Module und Workflows verfügbar, erscheinen aber nicht in der normalen Anwendersuche. Ihr sichtbarer Laufzeitname ist unabhängig vom jeweiligen Modul einheitlich `CMK Boundary Cache`, damit die Statusanzeige keinen fachlichen Bearbeitungsschritt vortäuscht.

Öffentliche Flow-Anzeigenamen beginnen mit `CMK Flow ·`, damit die Produktrolle auch in der globalen Node-Suche erkennbar bleibt. Python-Klassennamen und Socket-Typen werden dadurch nicht verändert.

Da ComfyUI Subgraphs nicht nach `CATEGORY` gliedert, bilden öffentliche Flow-Subgraphnamen eine sortierbare virtuelle Prozessfolge:

```text
CMK Flow · 01 START HERE · Create Image
CMK Flow · 02 LoRA Stack
CMK Flow · 05 ControlNet (optional)
CMK Flow · 10 KSampler 1st Pass
CMK Flow · 20 Refiner
CMK Flow · 30 Detailer
CMK Flow · 40 FaceSwap
CMK Flow · 50 FaceProcess
CMK Flow · 90 Upscale & Save
```

Nummern kennzeichnen die empfohlene Standardposition und sind kein Ausführungszwang. Zwischenräume bleiben für zukünftige Module reserviert.

Die Nummer ist zugleich ein fester Bestandteil des Modulnamens und der Suche. Anwender können ein Modul im CMK Flow Browser und in der ComfyUI-Suche gezielt über `30`, `40` oder `50` finden. Standard- und Advanced-Varianten behalten deshalb immer dieselbe Modulnummer.

### Unabhängigkeit der Flow-Module

Flow-Module sind innerhalb ihrer passenden Ein- und Ausgänge frei kombinierbar. Jedes Modul verarbeitet ausschließlich die Daten, die an seinen Eingängen anliegen, und reicht das Ergebnis nach vorne weiter. Es darf keine Rückkopplung zu vorgeschalteten Modulen, keine versteckte Abhängigkeit von einem konkreten Vorgänger und keine nur außerhalb des Moduls erfüllte Übergabe geben.

Ein Baustein gehört nur dann zu `CMK Flow`, wenn die für den durchgängigen Transport benötigten Boundaries, Caches und Weitergaben vollständig vorhanden sind. Fehlen diese Voraussetzungen, gehört der Baustein in die `CMK Toolbox` – auch wenn er grundsätzlich mit einzelnen Flow-Daten arbeiten kann. `CMK Toolbox · FaceSwap Image` ist dafür das maßgebliche Beispiel.

Execute-Nodes innerhalb eines Flow-Moduls dürfen eingehende Prozess-, Modell- oder Log-Pipes niemals unverändert oder angereichert zurückgeben. Sie liefern ausschließlich ihre fachlich neu erzeugten Ergebnisse, Diagnosen und einen eigenen Log-Block. Die unveränderte Weitergabe gemeinsamer Flow-Daten erfolgt außerhalb der Execute-Node über die Boundary; Log-Blöcke werden außerhalb der Execute-Node mit dem eingehenden Gesamtlog zusammengeführt. `CMK FaceSwap Image -Pipe-` folgt diesem Vertrag mit `IMAGE PROCEED`, `SEGS PROCESSED`, `LOG BLOCK` und `diagnostic` als Ausgängen.

FaceSwap besitzt zwei getrennte Referenzen. `CMK Flow · FaceSwap Image` ist der empfohlene Einstieg mit einer FaceSwap-Instanz. Nur die erweiterte Referenz trägt den Zusatz `Advanced`; sie erklärt ausschließlich den parallelen Betrieb von drei FaceSwap-Instanzen und deren gemeinsame Bild-, Log- und Boundary-Zusammenführung. Parallele Detailer- oder FaceProcess-Aufbauten gehören nicht in diese Referenz.

Bei FaceSwap sind `IDENTITY STRENGTH` und `BLEND` fachlich getrennt. `IDENTITY STRENGTH` gewichtet die Identitätsinformation vor der INSwapper-Inferenz; `BLEND` mischt erst danach das fertige Swap-Ergebnis mit dem Target. Der neutrale und kompatible Standardwert für beide Regler ist `1.00`.

Die Trennung in empfohlenen Einstieg und `Advanced` gilt entsprechend für den Detailer. Der empfohlene Detailer besitzt einen Ausführungszweig und führt `IMAGE PROCEED` direkt zur Boundary. Advanced besitzt zwei unabhängig konfigurierte Zweige und führt deren `SEGS PROCEED` erst außerhalb der Execute-Nodes zusammen. Beide Varianten bleiben unter einem Haupteintrag im Flow Browser gebündelt.

FaceProcess folgt derselben Trennung. Der empfohlene Einstieg `50 FaceProcess` besitzt einen Execute-Zweig und führt `IMAGE PROCEED` direkt zur Boundary. `50 FaceProcess · Advanced` besitzt kongruent zu `40 FaceSwap · Advanced` bis zu drei unabhängig ausgewählte Gesichts-Zweige und führt deren `SEGS PROCESSED` gemeinsam zusammen.

## 12. Änderungsregel

Eine UI-Änderung ist erst abgeschlossen, wenn:

- bestehende Bezeichnungen auf Widersprüche geprüft wurden;
- dynamische Eingänge nach Laden eines gespeicherten Workflows korrekt normalisiert werden;
- modeabhängig ausgeblendete Widgets in kanonischer Serialisierungsreihenfolge verbleiben und niemals aus `node.widgets` entfernt oder umsortiert werden;
- Standard-/Advanced-Zuordnung geprüft wurde;
- die sichtbare Oberfläche weiterhin den Vertrag aus `ARCHITECTURE.md` abbildet;
- keine zweite, konkurrierende Bezeichnung für dieselbe Rolle eingeführt wurde.

## Wertebereiche als Teil der UI-Verantwortlichkeit

Parametergrenzen sind Bestandteil des CMK-UI-Designs und nicht lediglich technische Backend-Grenzen.

Das UI soll den realen Arbeitsbereich einer Funktion präzise bedienbar machen. Ein großer theoretischer Wertebereich ist zu vermeiden, wenn:

- der praktisch sinnvolle Bereich nur einen kleinen Teil davon umfasst,
- hohe Werte die semantische Aufgabe der Node verlassen,
- Extremwerte überwiegend destruktive oder schwer kontrollierbare Ergebnisse erzeugen,
- ein zu großer Slider- oder Eingabebereich die Feinjustierung des relevanten Bereichs erschwert.

Verbindliche Beispiele:

```text
FaceSwap crop_factor: 1.0–3.0, Default 1.5
Detailer denoise:     maximal 0.5
```

Beim Detailer liegt der typische reale Arbeitsbereich häufig bis ungefähr `0.3`; `0.5` ist bereits als obere Sicherheitsgrenze zu verstehen, nicht als empfohlener Standardwert für jede Anwendung.

UI-Grenzen müssen mit der zentralen Engine-Validierung übereinstimmen. Eine rein visuelle Begrenzung ohne defensive Laufzeitvalidierung ist nicht ausreichend, weil ältere Workflows oder transportierte Pipe-Werte weiterhin außerhalb des aktuellen UI-Bereichs liegen können.

### Modulweite und lokale Freigabe

Nodes mit modulweiter und instanzweiser Schaltung ordnen die Standardparameter so an:

```text
GLOBAL ENABLE
ENABLE
```

`GLOBAL ENABLE` steht sichtbar vor `ENABLE`. Beide Schalter repräsentieren unterschiedliche Bedienebenen und dürfen nicht zusammengelegt werden. Eine Ausführung erfolgt nur bei `GLOBAL ENABLE AND ENABLE`.

### LOG-Ausgänge paralleler Branch-Nodes

Parallele Branch-Nodes dürfen die vollständige eingehende LOG-Pipe nicht durchschleifen. Sie geben ausschließlich einen eigenen Ausgang

```text
LOG BLOCK
```

vom Typ `CMK_LOG_BLOCK` aus. Die Zusammenführung erfolgt sichtbar über `CMK LOG CONCAT`. Dadurch bleiben Dokumentationswege ebenso eindeutig wie Bild-/SEGS-Branches.

### Globale OFF-Schalter als Ausführungsgrenze

Ein globaler OFF-Schalter muss vor jeder teuren Vorbereitung wirken. Es genügt
nicht, den deaktivierten Zustand erst nach Modell-, Detector-, SAM- oder
Conditioning-Aufbau an die Execute-Nodes weiterzugeben.

Upstream-Eingänge, die im global deaktivierten Zustand nicht benötigt werden,
sind lazy auszuführen. Der OFF-Zustand muss einen vollständigen gültigen
Passthrough erzeugen, ohne diese Ressourcen zu initialisieren.

### OFF means no preparation

A visible OFF switch must prevent all expensive preparation and branch caching. A disabled node may light briefly while ComfyUI resolves the graph, but it must not load detectors/models, create previews, stable full-image SEGS payloads or write persistent branch caches.

### Optionale Ressourcen in Dropdowns

Ein Dropdown für eine optionale Modellressource muss eine sichtbare neutrale
Auswahl `None` enthalten. Der Anwender darf nicht gezwungen werden, eine LoRA,
einen Enhancer oder eine vergleichbare optionale Ressource anzuwenden.

`None` muss ein echter funktionaler Bypass sein und darf nicht intern auf den
ersten verfügbaren Eintrag zurückfallen.

### Lokale Detailer-LoRA

Das LoRA-Dropdown eines Detailer-Prepare-Nodes muss `None` anbieten und
standardmäßig neutral sein. Der globale OFF-Schalter hat Vorrang vor sämtlichen
sichtbaren, aber in diesem Zustand inaktiven LoRA- und Prompt-Einstellungen.

## Modul-Deaktivierung

`GLOBAL ENABLE = OFF` ist ein funktionaler Hard-Bypass und keine rein visuelle
UI-Einstellung.

Ein global deaktiviertes Modul darf:

- keine Modelle laden,
- keine LoRAs verarbeiten,
- keine Conditionings erzeugen,
- keine Detectoren oder SAM-Modelle laden,
- keinen persistenten Modul-Cache aufbauen,
- keine rekursive Workflow-Signatur berechnen.

Lokale Parameter dürfen im UI weiterhin sichtbar sein, sind in diesem Zustand
aber vollständig wirkungslos.

## Optionale Modellressourcen

Jede optionale Modellressource benötigt eine neutrale Auswahl:

```text
None
```

Das betrifft insbesondere LoRAs, Enhancer und vergleichbare lokale
Zusatzmodelle.

`None` muss ein echter Bypass sein. Der Code darf bei `None` nicht automatisch
auf den ersten verfügbaren Dateieintrag zurückfallen.

## Modulreihenfolge

CMK-Module dürfen nicht auf eine feste Reihenfolge angewiesen sein. Ein Modul
kann:

- vor oder nach einem anderen Modul stehen,
- mehrfach vorhanden sein,
- vollständig fehlen,
- ausschließlich deaktivierte Branches enthalten.

Boundary-, Cache- und Passthrough-Logik muss deshalb positionsunabhängig und
ausschließlich modulintern entscheiden.

## FaceSwap-Masken

Identitätstransfer und Compositing sind intern getrennte Aufgaben:

- `SWAP MASK` bleibt eng am eigentlichen Gesicht.
- `BLEND MASK` erzeugt ausschließlich den weichen Übergang.
- Haare, Ohren, Schmuck und Hintergrund gehören grundsätzlich zum Target.
- Adaptive interne Standardwerte haben Vorrang vor zusätzlichen UI-Reglern.
