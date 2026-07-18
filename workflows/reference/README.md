# CMK-Flow-Referenzen

Die FaceSwap-Referenzen zeigen denselben Baustein in zwei bewusst unterschiedlich
komplexen Anwendungen.

## CMK Flow · FaceSwap Image

Dies ist der empfohlene Einstieg. Die Referenz enthält eine FaceSwap-Instanz und
zeigt den übersichtlichen Normalfall. Sie eignet sich zum Kennenlernen, als Demo
und als Ausgangsbasis für einen eigenen Workflow.

## CMK Flow · FaceSwap Image · Advanced

Diese Referenz richtet sich an erfahrene Anwender. Bis zu drei FaceSwap-Instanzen
bearbeiten eine linke, mittlere und rechte Zielperson parallel. Ihre Bildbereiche
und Protokolle werden anschließend gemeinsam zusammengeführt und erst danach an
die gemeinsame Flow-Grenze übergeben.

Der mittlere Zweig ist in der Referenz zunächst deaktiviert. Dadurch demonstriert
Advanced zugleich den Betrieb mit zwei Zielpersonen; bei Bedarf kann der dritte
Zweig direkt zugeschaltet werden.

Advanced ist keine grundsätzlich bessere Variante. Sie demonstriert die
Skalierbarkeit von CMK Flow für ein Bild mit mehreren Zielpersonen. Für den
üblichen Einsatz bleibt `CMK Flow · FaceSwap Image` die empfohlene Grundlage.

Detailer und FaceProcess sind nicht Bestandteil dieser beiden Referenzen. Ihre
parallelen Varianten werden getrennt bewertet, damit jede Referenz genau einen
fachlichen Aufbau erklärt.

Im CMK Flow Browser bleibt FaceSwap ein einzelner Haupteintrag. Dort ist
`Empfohlen` voreingestellt; `Advanced` wird erst über die Variantenwahl innerhalb
dieses Eintrags aktiviert.

## CMK Flow · Detailer

Die empfohlene Detailer-Referenz enthält einen Smart-Detailer-Zweig. Sein
`IMAGE PROCEED`-Ergebnis wird direkt an die gemeinsame Flow-Grenze übergeben.

## CMK Flow · Detailer · Advanced

Advanced enthält zwei getrennt konfigurierte Smart-Detailer-Zweige. Beide
arbeiten auf demselben Eingangsbild; ihre `SEGS PROCEED`-Ergebnisse werden vor
der gemeinsamen Flow-Grenze zusammengeführt. Dadurch können zwei unterschiedliche
Detailaufgaben unabhängig voneinander konfiguriert und aktiviert werden.

Wie bei FaceSwap bleibt der kurze Name der empfohlene Einstieg. Advanced erscheint
im Flow Browser ausschließlich als bewusst wählbare Variante desselben Eintrags.

## CMK Flow · FaceProcess

Die empfohlene Referenz enthält einen FaceProcess-Zweig. Sein `IMAGE PROCEED` wird
direkt an die gemeinsame Flow-Grenze übergeben.

## CMK Flow · FaceProcess · Advanced

Advanced enthält bis zu drei getrennt konfigurierte FaceProcess-Zweige für das
linke, mittlere und rechte Zielgesicht. Der mittlere Zweig ist zunächst
deaktiviert. Die `SEGS PROCESSED` aller verwendeten Zweige werden vor der
gemeinsamen Flow-Grenze zusammengesetzt.

Im Browser bleiben beide Varianten unter der gemeinsamen Modulnummer `50`
gebündelt. Die Nummer ist ein fester Bestandteil des Namens und kann direkt als
Suchbegriff verwendet werden.
