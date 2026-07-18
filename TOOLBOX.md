# CMK Toolbox

Die CMK Toolbox ist neben CMK Flow der zweite eigenständige Produktbereich. Sie ist kein Sammelplatz für interne Flow-Nodes und kein Archiv historischer Experimente. Jede veröffentlichte Toolbox-Funktion benötigt einen verständlichen Einzelzweck, eine fachliche Kategorie und mindestens einen nachvollziehbaren Beispiel- oder Testpfad.

## Produktgrenze

```text
CMK Flow
    geführter, möglichst linearer Gesamtprozess

CMK Toolbox
    frei kombinierbare Bild-, Masken-, Face-, Video- und Diagnosewerkzeuge

CMK Developer
    technische Infrastruktur für Flow und Toolbox
```

Ein `Developer`-Baustein wird nicht allein dadurch zur Toolbox, dass er im Node-Menü sichtbar ist. Umgekehrt darf eine Toolbox-Funktion intern dieselben Engines wie Flow verwenden, ohne dessen geschlossenen Bedienvertrag übernehmen zu müssen.

## Aktuelle Funktionsbereiche

| Bereich | Vorhandene Funktionen | Beispielabdeckung | Pflegebedarf |
|---|---|---|---|
| Image | Smart Detailer, Smart Outpaint Pad, Smart Upscaler | teilweise über ältere Workflows | aktuelle kleine Beispielworkflows fehlen |
| Mask & SEGS | Empty Mask, Image/Mask Switch, SEGS CONCAT | indirekt in Flow-Modulen | eigenständige Anwendungsbeispiele fehlen |
| Face | Face Detection, Crop, Select, Mask, Restore, FaceSwap, FaceProcess | FaceRestore und FaceSwap Image vorhanden | offene und Flow-Varianten klarer dokumentieren |
| Video | Loader, Segmentierung, FaceSwap, Compare, Merge/Save, Metrics | aktueller Video-Referenzworkflow vorhanden | Abhängigkeiten und Minimalworkflow pflegen |
| ControlNet | offene ControlNet-Vorbereitung | hauptsächlich über Flow belegt | eigenständiges Toolbox-Beispiel fehlt |
| Model & LoRA | Checkpoint/VAE- und LoRA-Text-Loader | indirekt über Flow | neutrale Minimalbeispiele fehlen |
| I-O | Dateinamen, Quellpfad, Bild/Text/Video speichern | in mehreren Workflows verwendet | Rollen und gemeinsame Namensregeln vereinheitlichen |
| Diagnostics | Image/Video Metrics, Preview, Summary, strukturierte Logs | punktuell vorhanden | als zusammenhängendes Diagnosepaket dokumentieren |

## Veröffentlichungsregel

Eine Toolbox-Node gilt erst als gepflegt, wenn:

1. ihr Name ohne Kenntnis der Implementierung verständlich ist;
2. ihre Kategorie einer fachlichen Anwenderaufgabe entspricht;
3. ihre Ein- und Ausgänge Tooltips besitzen, wenn die Fortsetzung nicht offensichtlich ist;
4. optionale Ressourcen einen echten neutralen Bypass anbieten;
5. ein kleiner Beispielworkflow oder ein klarer Referenzpfad existiert;
6. fehlende externe Abhängigkeiten lokal und verständlich gemeldet werden;
7. sie nicht versehentlich einen geschützten Flow-Boundary umgeht.

## Nächste Pflegeetappe

Die nächste Toolbox-Runde sollte nicht mit neuen Funktionen beginnen, sondern mit drei kleinen, veröffentlichungsfähigen Wegen:

```text
Toolbox · Face Restore
Toolbox · FaceSwap Image
Toolbox · Video FaceSwap
```

Danach folgen je ein Minimalworkflow für Image/Mask-SEGS, ControlNet und Diagnostics. Nicht belegte historische Subgraphs werden erst nach dieser Bestandsaufnahme archiviert oder entfernt.
