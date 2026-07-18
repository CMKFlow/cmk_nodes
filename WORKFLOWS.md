# CMK Subgraphs und Workflows

Dieses Repository versioniert neben den Python-Nodes auch die zugehörigen ComfyUI-Subgraphs und Workflows. Die Dateien stammen aus der CMK-Benutzerkonfiguration, enthalten aber keine Laufzeitlogs, Datenbanken, Manager-Caches oder persönlichen ComfyUI-Einstellungen.

Die aktuelle Nutzungs- und Bereinigungsmatrix steht in [`SUBGRAPH_AUDIT.md`](SUBGRAPH_AUDIT.md).

## Verzeichnisrollen

| Verzeichnis | Rolle |
|---|---|
| `subgraphs/` | Wiederverwendbare ComfyUI-Module und interne Hilfsgraphen |
| `workflows/reference/` | Technische Referenzen einzelner Module und Funktionen |
| `workflows/showcase/` | Kuratierte, im Flow Browser veröffentlichte Beispielworkflows |
| `workflows/examples/` | Interne oder spezialisierte Arbeitsbeispiele; kein Veröffentlichungseinstieg |
| `workflows/archive/` | Historische Stände; nicht für Neuinstallationen vorgesehen |

Die aktuellen veröffentlichten Bildbeispiele liegen unter `workflows/showcase/` und
werden ausschließlich über den Flow Browser als neue Kopie geöffnet. Technische
Modulreferenzen unter `workflows/reference/` werden nicht als allgemeiner Einstieg
angeboten. Der Video-Referenzworkflow bleibt
`workflows/reference/CMK_FaceSwap_Video_Reference_v2.2.json`.

## CMK Flow Browser

Der Menüeintrag `CMK Flow → Flow Browser öffnen` ist der zentrale Einstieg in die veröffentlichten Flow-Module. Der Browser liest ComfyUIs globale Subgraph-Registry, zeigt ausschließlich Einträge aus `custom_nodes.cmk_nodes` mit freigegebenen `CMKFlow`-Metadaten und fügt den ausgewählten Blueprint in den aktuell geöffneten Graphen ein. Zusätzlich erkennt er echte Python-Nodes automatisch, wenn deren Kategorie mit `CMK/Flow/` beginnt. Kategorien unter `CMK/Toolbox/` und `CMK/Developer/` bleiben dadurch ausgeschlossen.

Eine Ablaufnummer ist ausschließlich kuratierten Modulen der geführten Flow-Reihenfolge vorbehalten. Eigenständige, nur grundsätzlich kompatible Bausteine erhalten keine Nummer. Mit `area: "toolbox"` und `published: false` vorbereitete Subgraphen bleiben aus dem Flow-Katalog ausgeschlossen und können später dem Baukasten zugeordnet werden.

Die Registrierung liegt direkt in der jeweiligen Datei unter `extra.CMKFlow`:

```json
{
  "extra": {
    "CMKFlow": {
      "schemaVersion": 2,
      "published": true,
      "displayName": "10 KSampler 1st Pass",
      "category": "Process",
      "domain": "Image Generation",
      "description": "Kurze, anwenderorientierte Beschreibung.",
      "status": "STABLE",
      "version": "1.0.0",
      "author": "CMK Nodes",
      "compatibility": ["SDXL"],
      "features": ["Sampling", "Seed-Management"],
      "order": 10,
      "searchAliases": ["sampler", "first pass"]
    }
  }
}
```

Pflichtfelder eines veröffentlichten Eintrags sind `published`, `category`, `description` und `status`. Zulässige Statuswerte sind `STABLE`, `BETA` und `EXPERIMENTAL`. `order` bestimmt die Sortierung; ohne gültige Zahl wird der Eintrag ans Ende gesetzt. `displayName`, `domain`, `version`, `author`, `compatibility`, `features` und `searchAliases` erweitern die Detailansicht, bleiben für ältere Einträge aber optional. Die Ein- und Ausgänge liest der Browser unmittelbar aus dem Subgraphen und dupliziert sie nicht in den Metadaten.

So wird ein neuer Flow veröffentlicht:

1. Den funktionsfähigen Subgraph als JSON direkt unter `subgraphs/` speichern.
2. `extra.CMKFlow` nach obigem Schema ergänzen und `published` auf `true` setzen.
3. ComfyUI vollständig neu starten, da die globale Subgraph-Liste serverseitig zwischengespeichert wird.
4. Im CMK Flow Browser Kategorie, Suche, Detailansicht und Einfügen prüfen.

Für Entwürfe kann der Metadatenblock bereits vorhanden sein, während `published` auf `false` steht. Neue Flows benötigen keine Änderung an der JavaScript-Erweiterung.

Eine einzelne Python-Node wird für den Flow Browser freigegeben, indem ihre `CATEGORY` gezielt unter `CMK/Flow/Input`, `CMK/Flow/Process` oder `CMK/Flow/Finish` eingeordnet wird. Diese Freigabe sollte nur für kuratierte, anwenderorientierte Nodes erfolgen; Baukasten-Nodes verbleiben unter `CMK/Toolbox/`.

Reale Vorschaubilder liegen unter `web/assets/previews/`. Kuratierte Python-Nodes ordnen ihre Bilder in `web/flow_node_metadata.json` über ein `previews`-Array dem internen Node-Typ zu. Subgraphen registrieren dasselbe Array direkt unter `extra.CMKFlow.previews`. Ein Eintrag besteht aus `src` und einer kurzen Schaltflächenbeschriftung wie `Node`, `Modul` oder `Aufbau`. Mehrere Bilder erscheinen als umschaltbare Galerie. `placementNote` erklärt in Alltagssprache, wo der Baustein in den Flow gehört. Die optionalen Felder `recommendedBefore`, `recommendedAfter` und `dependencyNote` nennen passende Nachbarn und notwendige Verbindungen. Einstellbare Werte der Bedienoberfläche werden nicht nochmals beschrieben; dafür dient die reale Vorschau. Fehlt ein Bild, zeigt der Browser ausdrücklich einen Platzhalter und erzeugt keine schematische Node-Darstellung.

## Zentrale Pipe-Subgraphs

Der geführte Hauptworkflow wird insbesondere aus diesen Subgraphs zusammengesetzt:

```text
CMK Flow · 01 START HERE · Create Image
    ↑
CMK Flow · 02 LoRA Stack + Prompt/Image source
    ↓
CMK Flow · 05 ControlNet (optional)
    ↓
CMK Flow · 10 KSampler 1st Pass
    ↓
CMK Flow · 20 Refiner
    ↓
CMK Flow · 30 Detailer
    ↓
CMK Flow · 40 FaceSwap (optional)
    ↓
CMK Flow · 50 FaceProcess
    ↓
CMK Flow · 90 Upscale & Save
```

`CMK Flow · 90 Upscale & Save` ist der verbindliche Abschluss des Flow-Hauptwegs und wird unter `CMK/Flow/Finish` geführt.

Weitere Subgraphs kapseln Inpaint-, Conditioning-, Masken-, Video-, Dateinamen- und ältere Pipe-In/Pipe-Out-Funktionen.

Subgraphs werden in Workflows über UUIDs referenziert. Um bestehende Workflows kompatibel zu halten, müssen beim Aktualisieren die versionierten JSON-Dateien verwendet werden; ein manuelles Neuerstellen gleichnamiger Subgraphs ist nicht gleichwertig.

`CMK Toolbox · FaceSwap Image` ist eine eigenständig verwendbare Variante und gehört nicht zur geführten Flow-Reihenfolge. Ihm fehlen die vollständigen Boundaries und Weitergaben, die ein frei kombinierbares Flow-Modul benötigt. Es darf deshalb nicht als Alternative zu `CMK Flow · 40 FaceSwap` in den Flow-Hauptweg eingesetzt werden. Die in `CMK Flow · 40 FaceSwap` verwendete Execute-Node bleibt davon unberührt. Auch die offene Node `CMK FaceSwap Image` ohne `-Pipe-` gehört zum Baukasten. Der frühere Subgraphname `SwapFace` wurde entfernt.

Für alle Flow-Module gilt: Daten laufen ausschließlich von den Eingängen zu den Ausgängen und anschließend weiter zum nächsten Modul. Ein Modul darf keine Rückkopplung zu einem vorgeschalteten Modul und keine versteckte Abhängigkeit von dessen internem Aufbau besitzen. Die Reihenfolge ist eine Empfehlung; die korrekten Ein- und Ausgänge sind der Vertrag.

Mehrfach vorkommende Definitionen innerhalb exportierter JSON-Dateien sind nicht automatisch Konflikte: ComfyUI bettet abhängige Subgraph-Definitionen in Exporte ein. Entscheidend ist, ob unterschiedliche öffentliche Subgraphs dieselbe UUID beanspruchen.

## Installation

Bei einer normalen ComfyUI-Installation gilt:

```text
cmk_nodes/subgraphs/*.json
    → bleibt im Node-Pack und wird automatisch als "Subgraph Blueprints" geladen

cmk_nodes/workflows/showcase/*.json
    → über den Flow Browser als neue Kopie öffnen
```

CMK-Subgraphdateien dürfen nicht zusätzlich unter `user/default/subgraphs/` installiert werden. Dieser Ordner ist ComfyUIs separate Registry für benutzereigene Blueprints und erzeugt für dieselben Dateien zweite Einträge unter `Subgraph Blueprints/User`.

Technische Referenz-, Arbeits- und Archivworkflows nur gezielt verwenden. Anschließend ComfyUI vollständig neu starten, damit Subgraph-Definitionen und Node-Erweiterungen gemeinsam neu geladen werden.

## Externe Node-Abhängigkeiten

Die Graphen verwenden neben den CMK-Nodes auch ComfyUI-Core-Nodes und – abhängig vom jeweiligen Graphen – Nodes aus unter anderem:

- ComfyUI Impact Pack und Impact Subpack;
- rgthree/Eclipse-artigen Pipe-, Comparer- und Sampler-Helfern;
- ComfyUI Essentials beziehungsweise Easy-Use-Helfern;
- KJNodes;
- Video Helper Suite;
- WAS/pysssss-, Logic-/Switch- und LoRA-Manager-Erweiterungen;
- Fooocus-Inpaint- und Kontext-Reference-Erweiterungen.

Nicht jeder Workflow benötigt alle Erweiterungen. Maßgeblich sind die beim Laden von ComfyUI als fehlend gemeldeten Node-Typen des konkret verwendeten Workflows.

## Pflege

Bei Änderungen an einem Pipe-Subgraph sind gemeinsam zu prüfen:

1. UUID und öffentliche Ein-/Ausgänge des Subgraphs;
2. proprietäre CMK-Socket-Typen;
3. alle Referenzworkflows, die den Subgraph verwenden;
4. Lazy-Bypass- und Boundary-Cache-Verhalten;
5. gespeicherte Widgetwerte und Proxy-Widgets;
6. vollständiges Laden nach einem ComfyUI-Neustart.
