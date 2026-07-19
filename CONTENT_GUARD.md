# CMK FaceSwap ContentGuard

## Zweck

Der ContentGuard ist für alle öffentlichen CMK-FaceSwap-Pfade verpflichtend.
Er begrenzt technisch das Risiko, FaceSwap auf explizite Inhalte oder auf
Personen anzuwenden, deren lokale Altersschätzung keine hinreichend sichere
Erwachsenenklassifikation erlaubt.

## Verbindliche Policy

- Quelle und Ziel werden vor jedem tatsächlichen Swap geprüft.
- Im Videopfad wird jedes decodierte Ziel-Frame auf explizite Inhalte geprüft.
- Jedes tatsächlich ausgewählte Quell- und Zielgesicht benötigt eine gültige
  lokale Altersschätzung von mindestens 25 Jahren.
- Altersschätzungen unter 18 werden mit `CG_AGE_MINOR` blockiert.
- Altersschätzungen von 18 bis 24 werden konservativ mit
  `CG_AGE_UNCERTAIN` blockiert.
- Fehlende oder ungültige Altersschätzungen werden blockiert.
- NudeNet-Treffer ab `0.35` für entblößte Genitalien, Anus, Gesäß oder weibliche
  Brust werden mit `CG_EXPLICIT_CONTENT` blockiert.
- Ein fehlendes Schutzmodell oder ein Inferenzfehler führt zum Abbruch.
- Es gibt keine öffentliche Einstellung zum Abschalten oder Überstimmen des
  Guards.
- Ist die FaceSwap-Node selbst deaktiviert, bleibt sie ein echter Pass-through;
  es wird kein Swap ausgeführt und der Guard nicht geladen.

## Fail-closed und Ausgaben

Ein Guard-Treffer beendet die Verarbeitung vor der jeweiligen Swap-Inferenz.
Bild-Batches liefern bei einem Treffer kein Ergebnis zurück. Der Videopfad
verwirft bei einem Treffer alle in diesem Lauf neu erzeugten Swap-Segmente und
erstellt kein zusammengeführtes Ergebnis. Video-Cache-Signaturen enthalten die
Guard-Version, sodass Ergebnisse aus älteren ungeschützten Engine-Versionen
nicht wiederverwendet werden.

Die Fehlermeldung enthält ausschließlich einen neutralen Diagnosecode und die
Rolle (`source`, `target` oder `target_frame`). Sie gibt keine erkannten
Körperteile oder geschätzten Alterswerte aus.

## Diagnosecodes

| Code | Bedeutung |
|---|---|
| `CG_AGE_MINOR` | Altersschätzung unter 18 |
| `CG_AGE_UNCERTAIN` | Altersschätzung zwischen 18 und 24 |
| `CG_AGE_UNAVAILABLE` | keine verwertbare Altersschätzung |
| `CG_AGE_INVALID` | Altersschätzung außerhalb des gültigen Bereichs |
| `CG_EXPLICIT_CONTENT` | expliziter Inhalt oberhalb der Policy-Schwelle |
| `CG_EXPLICIT_GUARD_UNAVAILABLE` | lokaler Explicit-Guard konnte nicht geladen werden |
| `CG_EXPLICIT_INFERENCE_FAILED` | Explicit-Inferenz ist fehlgeschlagen |
| `CG_SOURCE_IMAGE_REQUIRED` | Quellbild fehlt im geschützten Swap-Pfad |
| `CG_INPUT_INVALID` | Bildformat ist für den Guard ungültig |

## Grenzen

Der ContentGuard ist keine verlässliche Altersfeststellung, keine
Identitätsprüfung und keine Einwilligungsprüfung. Lokale ML-Modelle können
falsch-positive und falsch-negative Ergebnisse liefern. Die konservative
25-Jahre-Grenze reduziert Unsicherheit, beseitigt sie aber nicht. Anwender sind
weiterhin selbst für eine rechtmäßige, einvernehmliche und verantwortungsvolle
Nutzung verantwortlich.

## Lokale Abhängigkeiten

- `nudenet==3.4.2` für die Erkennung expliziter Bildinhalte
- InsightFace `genderage.onnx` aus dem ausgewählten FaceAnalysis-Modellpaket für
  die Altersschätzung

Die jeweiligen Lizenzen und Nutzungsbedingungen der Abhängigkeiten und Modelle
gelten unabhängig von CMK Flow.
