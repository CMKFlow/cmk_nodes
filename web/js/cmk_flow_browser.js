import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXTENSION_NAME = "CMK.FlowBrowser";
const COMMAND_ID = "cmk.openFlowBrowser";
const NODE_PACK = "custom_nodes.cmk_nodes";
const ALLOWED_STATUS = new Set(["STABLE", "BETA", "EXPERIMENTAL"]);

let browserDataPromise;
let selectedId = null;
let currentCategory = "Alle";
let searchText = "";
let activeTab = "flow";
let toolboxSelectedId = null;
let toolboxCategory = "Alle";
let toolboxSearchText = "";
let referenceSelectedId = null;
let referenceSearchText = "";
let language = localStorage.getItem("cmk-flow-language") || (navigator.language?.toLowerCase().startsWith("de") ? "de" : "en");

const UI_TEXT = {
  de: { toolbox:"Baukasten", all:"Alle", searchFlows:"Flows durchsuchen …", searchToolbox:"Baukasten durchsuchen …", subtitleFlow:"Flow auswählen und in den aktuellen Workflow einfügen.", subtitleToolbox:"Node aus dem Baukasten auswählen und in den aktuellen Workflow einfügen.", noFlows:"Keine passenden Flows gefunden.", noNodes:"Keine passenden Nodes gefunden.", selectFlow:"Flow auswählen, um Details zu sehen.", selectNode:"Node auswählen, um Details zu sehen.", features:"Was dieser Baustein macht", placement:"Empfohlene Platzierung", category:"Kategorie", compatibility:"Kompatibilität", version:"Version", author:"Erstellt von", interfaces:"Ein- und Ausgänge", inputs:"Eingänge", outputs:"Ausgänge", related:"Passende Bausteine davor und danach", before:"Davor", after:"Danach", preview:"So sieht der Baustein aus", insert:"In Workflow einfügen", recommended:"Empfohlen", none:"Keine", previewMissing:"Noch keine reale Vorschau hinterlegt.", canStart:"Kann am Anfang stehen", canFinish:"Kann den Ablauf abschließen", loading:"Flows werden geladen …", loadError:"Browser-Inhalte konnten nicht geladen werden.", videoStorage:"Video-Speicher", videoStorageIntro:"Hier verwaltest du die dauerhaft gespeicherten Arbeitsdateien deiner CMK-Videoprojekte. Du kannst Projekte im Flow öffnen, ihre Speicherordner anzeigen oder nicht mehr benötigte Projektdateien gezielt löschen. Die Segmente bleiben erhalten, damit du später ohne erneute Segmentierung weiterarbeiten kannst.", segments:"Segmente", mergedVideos:"Zusammengeführte Videos", files:"Dateien", technicalLocations:"Technische Speicherorte", deleteAllVideoFiles:"Video-Arbeitsdateien löschen", videoProjects:"Videoprojekte", noVideoProjects:"Keine Video-Arbeitsdateien vorhanden.", mergedWorkingVideos:"zusammengeführte Videos", total:"Gesamt", lastUsed:"Zuletzt verwendet", openInFlow:"▶ Im Flow öffnen", openFolder:"📂 Speicherordner öffnen", deleteProjectFiles:"🗑 Projektdateien löschen", openingFlow:"Video-Workflow wird geöffnet …", folderOpened:"Speicherordner wurde geöffnet.", openFailed:"Öffnen fehlgeschlagen", deleteProjectTitle:"Video-Arbeitsdateien dieses Projekts löschen?", deleteProjectText:(name)=>`Alle gespeicherten Segmente und zusammengeführten Arbeitsvideos für „${name}“ werden gelöscht. Dieser Vorgang kann nicht rückgängig gemacht werden.`, cancel:"Abbrechen", deleteProject:"Projekt löschen", deleteAllTitle:"Alle Video-Arbeitsdateien löschen?", deleteAllText:"Alle gespeicherten Videosegmente und zusammengeführten Arbeitsvideos sämtlicher Projekte werden gelöscht. Laufende Videoprozesse müssen zuvor beendet sein. Dieser Vorgang kann nicht rückgängig gemacht werden.", deleteAll:"Alle Projekte löschen", deleting:"Dateien werden gelöscht …", projectDeleted:(size)=>`Projektdateien gelöscht. ${size} freigegeben.`, allDeleted:"Alle Video-Arbeitsdateien wurden gelöscht.", partialDelete:"Ein Teil der Dateien konnte nicht gelöscht werden.", deleteFailed:"Löschen fehlgeschlagen", storageUnavailable:"Speicherstatus nicht verfügbar" },
  en: { toolbox:"Toolbox", all:"All", searchFlows:"Search flows …", searchToolbox:"Search toolbox …", subtitleFlow:"Select a flow and add it to the current workflow.", subtitleToolbox:"Select a toolbox node and add it to the current workflow.", noFlows:"No matching flows found.", noNodes:"No matching nodes found.", selectFlow:"Select a flow to view its details.", selectNode:"Select a node to view its details.", features:"What this module does", placement:"Recommended placement", category:"Category", compatibility:"Compatibility", version:"Version", author:"Created by", interfaces:"Inputs and outputs", inputs:"Inputs", outputs:"Outputs", related:"Suitable modules before and after", before:"Before", after:"After", preview:"Module preview", insert:"Add to workflow", recommended:"Recommended", none:"None", previewMissing:"No real preview available yet.", canStart:"Can be placed at the beginning", canFinish:"Can complete the workflow", loading:"Loading flows …", loadError:"Browser content could not be loaded.", videoStorage:"Video storage", videoStorageIntro:"Manage the permanently stored working files of your CMK video projects here. You can open projects in Flow, reveal their storage folders, or selectively remove project files you no longer need. Segments remain available so you can continue later without segmenting the video again.", segments:"Segments", mergedVideos:"Merged videos", files:"files", technicalLocations:"Technical storage locations", deleteAllVideoFiles:"Delete video working files", videoProjects:"Video projects", noVideoProjects:"No video working files found.", mergedWorkingVideos:"merged videos", total:"Total", lastUsed:"Last used", openInFlow:"▶ Open in Flow", openFolder:"📂 Open storage folder", deleteProjectFiles:"🗑 Delete project files", openingFlow:"Opening video workflow …", folderOpened:"Storage folder opened.", openFailed:"Open failed", deleteProjectTitle:"Delete this project’s video working files?", deleteProjectText:(name)=>`All stored segments and merged working videos for ‘${name}’ will be deleted. This action cannot be undone.`, cancel:"Cancel", deleteProject:"Delete project", deleteAllTitle:"Delete all video working files?", deleteAllText:"All saved video segments and merged working videos for every project will be deleted. Running video processes must be stopped first. This action cannot be undone.", deleteAll:"Delete all projects", deleting:"Deleting files …", projectDeleted:(size)=>`Project files deleted. ${size} freed.`, allDeleted:"All video working files were deleted.", partialDelete:"Some files could not be deleted.", deleteFailed:"Delete failed", storageUnavailable:"Storage status unavailable" }
};
const t = (key) => UI_TEXT[language]?.[key] || UI_TEXT.de[key] || key;

function addStyles() {
  if (document.getElementById("cmk-flow-browser-styles")) return;

  const style = document.createElement("style");
  style.id = "cmk-flow-browser-styles";
  style.textContent = `
    .cmk-flow-overlay { position: fixed; inset: 0; z-index: 10000; display: grid; place-items: center; padding: 24px; background: rgba(0,0,0,.62); backdrop-filter: blur(3px); }
    [data-cmk-flow-launcher="true"] { min-width: 118px; min-height: 36px; padding-inline: 16px !important; border: 1px solid #43d5d7 !important; background: #1f7779 !important; color: #fff !important; font-size: 14px !important; font-weight: 700 !important; letter-spacing: .015em; }
    [data-cmk-flow-launcher="true"]:hover { border-color: #71edef !important; background: #278e90 !important; }
    [data-cmk-flow-launcher="true"]:focus-visible { outline: 2px solid #71edef !important; outline-offset: 2px; }
    .cmk-flow-browser { width: min(1180px, 96vw); height: min(760px, 92vh); display: grid; grid-template-rows: auto auto 1fr; overflow: hidden; border: 1px solid #354049; border-radius: 16px; background: radial-gradient(circle at 78% 15%, #13272b 0, #11171c 32%, #101419 72%); color: #eef3f4; box-shadow: 0 28px 90px rgba(0,0,0,.62); font: 14px/1.5 system-ui, sans-serif; }
    .cmk-flow-header { display: flex; align-items: center; justify-content: space-between; padding: 18px 24px; border-bottom: 1px solid #293239; background: rgba(11,16,21,.68); }
    .cmk-flow-heading { display: grid; gap: 2px; }
    .cmk-flow-header h2 { margin: 0; font-size: 25px; font-weight: 700; letter-spacing: -.02em; }
    .cmk-flow-subtitle { color: #9eaab1; font-size: 13px; }
    .cmk-flow-header-actions { display: flex; align-items: center; gap: 12px; }
    .cmk-about-open, .cmk-storage-open { min-height: 34px; padding: 0 13px; border: 1px solid #3b474f; border-radius: 8px; background: #182127; color: #c4ced2; font: inherit; font-size: 12px; cursor: pointer; }
    .cmk-about-open:hover, .cmk-storage-open:hover { border-color: #52636b; color: #eef3f4; }
    .cmk-language-switch { display: flex; overflow: hidden; border: 1px solid #3b474f; border-radius: 8px; }
    .cmk-language-button { min-width: 38px; min-height: 32px; border: 0; background: #151d22; color: #7f8d94; font: inherit; font-size: 11px; font-weight: 700; cursor: pointer; }
    .cmk-language-button.is-active { background: #24343a; color: #62d9d6; }
    .cmk-flow-close { border: 0; background: transparent; color: #aeb5bb; font-size: 25px; cursor: pointer; }
    .cmk-about-overlay { position: fixed; inset: 0; z-index: 10001; display: grid; place-items: center; padding: 24px; background: rgba(0,0,0,.68); }
    .cmk-about-dialog { width: min(680px, 92vw); max-height: 86vh; overflow: auto; border: 1px solid #354049; border-radius: 14px; background: #12191e; color: #d4dde0; box-shadow: 0 24px 80px rgba(0,0,0,.68); }
    .cmk-about-header { display: flex; align-items: center; justify-content: space-between; padding: 17px 20px; border-bottom: 1px solid #2d373e; }
    .cmk-about-title { display: flex; align-items: center; gap: 11px; }
    .cmk-about-logo { width: 36px; height: 36px; border: 1px solid #30434d; border-radius: 8px; object-fit: cover; box-shadow: 0 0 14px rgba(38,173,210,.12); }
    .cmk-about-header h3 { margin: 0; color: #eef3f4; font-size: 21px; }
    .cmk-about-body { padding: 20px; color: #bcc7cb; line-height: 1.65; }
    .cmk-about-body p { margin: 0 0 15px; }
    .cmk-about-body p:last-child { margin-bottom: 0; }
    .cmk-about-body strong { color: #eef3f4; }
    .cmk-about-responsibility, .cmk-about-acknowledgements { margin-top: 22px; padding: 18px; border: 1px solid #304147; border-radius: 10px; background: #141f24; }
    .cmk-about-responsibility h4, .cmk-about-acknowledgements h4 { margin: 0 0 10px; color: #62d9d6; font-size: 14px; }
    .cmk-about-acknowledgements a { color: #7cdfe0; }
    .cmk-about-support { margin-top: 22px; padding-top: 20px; border-top: 1px solid #303a41; }
    .cmk-about-donation { margin-top: 18px; padding-top: 17px; border-top: 1px solid #273138; text-align: center; }
    .cmk-about-donation p { margin-bottom: 11px; color: #8f9ca1; font-size: 12px; line-height: 1.5; }
    .cmk-about-donation-actions { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
    .cmk-about-donation-link { display: inline-flex; align-items: center; justify-content: center; padding: 7px 12px; border: 1px solid #3a4951; border-radius: 7px; background: #182126; color: #b8c4c8; font-size: 12px; font-weight: 600; text-decoration: none; transition: border-color .15s ease, background .15s ease, color .15s ease; }
    .cmk-about-donation-link:hover { border-color: #55717a; background: #1c292f; color: #e0e7e9; }
    .cmk-about-donation-link:focus-visible { outline: 2px solid #62d9d6; outline-offset: 2px; }
    .cmk-about-legal { margin-top: 20px; padding-top: 16px; border-top: 1px solid #303a41; color: #88969c; font-size: 12px; line-height: 1.5; }
    .cmk-about-legal p { margin-bottom: 3px; }
    .cmk-storage-summary { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0 20px; }
    .cmk-storage-card { padding: 14px; border: 1px solid #303a42; border-radius: 9px; background: #171f24; }
    .cmk-storage-card strong { display: block; margin-bottom: 4px; color: #dce5e7; }
    .cmk-storage-card span { color: #8f9ba1; font-size: 12px; }
    .cmk-storage-actions { display: flex; align-items: center; justify-content: space-between; gap: 14px; padding-top: 16px; border-top: 1px solid #303a41; }
    .cmk-storage-paths { margin: -7px 0 18px; color: #738087; font-size: 11px; line-height: 1.55; }
    .cmk-storage-paths code { color: #87959b; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .cmk-storage-projects { display: grid; gap: 9px; margin: 0 0 20px; }
    .cmk-storage-projects h4 { margin: 0 0 2px; color: #62d9d6; font-size: 14px; }
    .cmk-storage-project { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 14px; align-items: center; padding: 13px 14px; border: 1px solid #303a42; border-radius: 9px; background: #171f24; }
    .cmk-storage-project-name { display: block; overflow: hidden; color: #dce5e7; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
    .cmk-storage-project-meta { display: grid; gap: 2px; margin-top: 5px; color: #8f9ba1; font-size: 11px; }
    .cmk-storage-project-actions { display: grid; min-width: 180px; gap: 6px; }
    .cmk-storage-project-action { min-height: 31px; padding: 0 10px; border: 1px solid #3a484f; border-radius: 7px; background: #182127; color: #bdc8cc; font: inherit; font-size: 11px; text-align: left; cursor: pointer; }
    .cmk-storage-project-action:hover { border-color: #52636b; color: #eef3f4; }
    .cmk-storage-project-delete { border-color: #594044; background: #292023; color: #d9afb4; }
    .cmk-storage-project-delete:hover { border-color: #86545a; color: #ffd0d4; }
    .cmk-storage-empty { padding: 13px 14px; border: 1px solid #303a42; border-radius: 9px; color: #839097; font-size: 12px; }
    .cmk-storage-clear { min-height: 38px; padding: 0 15px; border: 1px solid #7d4444; border-radius: 8px; background: #3a2424; color: #ffc4c4; font: inherit; font-weight: 650; cursor: pointer; }
    .cmk-storage-clear:hover { border-color: #aa5a5a; background: #4a2929; }
    .cmk-storage-clear:disabled { cursor: wait; opacity: .55; }
    .cmk-storage-status { color: #8f9ba1; font-size: 12px; }
    .cmk-confirm-dialog { width: min(470px, 90vw); }
    .cmk-confirm-actions { display: flex; justify-content: flex-end; gap: 9px; margin-top: 20px; }
    .cmk-confirm-cancel, .cmk-confirm-delete { min-height: 36px; padding: 0 14px; border-radius: 8px; font: inherit; font-weight: 650; cursor: pointer; }
    .cmk-confirm-cancel { border: 1px solid #3b474f; background: #182127; color: #c4ced2; }
    .cmk-confirm-delete { border: 1px solid #8a4848; background: #492727; color: #ffd0d0; }
    .cmk-browser-tabs { display: flex; gap: 4px; padding: 8px 18px 0; border-bottom: 1px solid #293239; background: rgba(11,16,21,.68); }
    .cmk-browser-tab { min-width: 112px; padding: 9px 14px 10px; border: 0; border-bottom: 2px solid transparent; background: transparent; color: #8e9ba2; font: inherit; font-weight: 650; cursor: pointer; }
    .cmk-browser-tab:hover { color: #d5dfe2; }
    .cmk-browser-tab.is-active { border-bottom-color: #42cbc8; color: #eef3f4; }
    .cmk-flow-content { min-height: 0; display: grid; grid-template-columns: 390px 1fr; }
    .cmk-flow-sidebar { min-height: 0; display: grid; grid-template-rows: auto 1fr; border-right: 1px solid #2e383f; background: rgba(14,19,24,.78); }
    .cmk-flow-controls { display: grid; grid-template-columns: minmax(180px, 1fr) 118px; gap: 10px; padding: 18px 18px 12px; }
    .cmk-flow-controls input, .cmk-flow-controls select { min-height: 42px; border: 1px solid #3b474f; border-radius: 9px; padding: 0 12px; background: #1b2228; color: inherit; font: inherit; }
    .cmk-flow-controls input:focus, .cmk-flow-controls select:focus { border-color: #40ccca; outline: 1px solid #40ccca; }
    .cmk-flow-list { min-height: 0; overflow: auto; padding: 4px 12px 16px; }
    .cmk-flow-item { width: 100%; display: grid; grid-template-columns: 48px 1fr; gap: 12px; align-items: center; margin: 0 0 7px; padding: 11px 12px; border: 1px solid #29333a; border-radius: 10px; background: #171e24; color: inherit; text-align: left; cursor: pointer; }
    .cmk-flow-item:hover { border-color: #435159; background: #1c252b; }
    .cmk-flow-item.is-selected { border-color: #39cecb; background: linear-gradient(100deg, #153638, #17484a); box-shadow: inset 3px 0 #44d6d3; }
    .cmk-flow-item-index { display: grid; place-items: center; width: 44px; height: 44px; border: 1px solid #365259; border-radius: 9px; background: #1c2d33; color: #56d9d6; }
    .cmk-flow-item-index svg { width: 25px; height: 25px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
    .cmk-flow-item-name { display: block; font-size: 14px; font-weight: 650; }
    .cmk-flow-item-category { display: block; margin-top: 2px; color: #8e9ba2; font-size: 12px; }
    .cmk-toolbox-item { grid-template-columns: 1fr; padding: 12px 14px; }
    .cmk-toolbox-item.is-featured { border-color: #36575c; box-shadow: inset 3px 0 #43c8c5; }
    .cmk-toolbox-item.is-featured:not(.is-selected):hover { border-color: #4a777c; background: #18272c; }
    .cmk-flow-detail { min-width: 0; overflow: auto; padding: 30px 36px 26px; }
    .cmk-flow-detail-header h3 { margin: 12px 0 2px; font-size: 28px; line-height: 1.2; letter-spacing: -.02em; }
    .cmk-flow-domain { color: #42cbc8; font-size: 14px; font-weight: 650; }
    .cmk-flow-description { max-width: 70ch; margin: 13px 0 24px; color: #b8c2c7; }
    .cmk-flow-badge { display: inline-block; padding: 3px 8px; border-radius: 99px; background: #273238; color: #bceff0; font-size: 11px; font-weight: 700; letter-spacing: .06em; }
    .cmk-flow-badge[data-status="BETA"] { background: #3b3122; color: #ffd58a; }
    .cmk-flow-badge[data-status="EXPERIMENTAL"] { background: #3d2938; color: #ffb9e5; }
    .cmk-flow-variants { display: flex; flex-wrap: wrap; gap: 7px; margin: 0 0 18px; }
    .cmk-flow-variant { padding: 7px 11px; border: 1px solid #34444a; border-radius: 8px; background: #182127; color: #aebbc0; font: inherit; font-size: 12px; cursor: pointer; }
    .cmk-flow-variant:hover { border-color: #4e6b70; color: #e3eeee; }
    .cmk-flow-variant.is-selected { border-color: #42cbc8; background: #18383a; color: #64ddd9; }
    .cmk-flow-section-title { margin: 0 0 9px; color: #52d3cf; font-size: 13px; font-weight: 700; }
    .cmk-flow-detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 18px; }
    .cmk-flow-card { min-width: 0; padding: 15px 16px; border: 1px solid #303a42; border-radius: 11px; background: rgba(25,32,38,.8); }
    .cmk-flow-card ul { display: grid; gap: 7px; margin: 0; padding: 0; list-style: none; color: #c4ccd0; }
    .cmk-flow-card li { display: grid; grid-template-columns: 14px 1fr; gap: 7px; }
    .cmk-flow-card li::before { content: ""; width: 7px; height: 7px; margin-top: 7px; border: 2px solid #43c8c5; border-radius: 50%; }
    .cmk-flow-feature-title { display: block; color: #d5dcdf; font-weight: 650; }
    .cmk-flow-feature-description { display: block; margin-top: 2px; color: #8f9aa0; font-size: 12px; line-height: 1.4; }
    .cmk-flow-feature-callout { display: grid; grid-template-columns: 42px 1fr; gap: 12px; margin-top: 14px; padding-top: 14px; border-top: 1px solid #344047; }
    .cmk-flow-feature-callout::before { content: "✓"; display: grid; width: 32px; height: 32px; place-items: center; border: 2px solid #43c8c5; border-radius: 50%; color: #43c8c5; font-size: 19px; font-weight: 700; }
    .cmk-flow-feature-callout-title { color: #52d3cf; font-size: 13px; font-weight: 700; }
    .cmk-flow-feature-callout-text { margin-top: 2px; color: #9aa5aa; font-size: 12px; line-height: 1.4; }
    .cmk-flow-card .is-optional { color: #8d989f; }
    .cmk-flow-meta { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 0 0 18px; }
    .cmk-flow-meta-item { padding: 10px 11px; border-left: 2px solid #34444a; background: rgba(19,26,31,.65); }
    .cmk-flow-meta-label { display: block; color: #7f8d94; font-size: 10px; text-transform: uppercase; letter-spacing: .06em; }
    .cmk-flow-meta-value { display: block; margin-top: 3px; overflow: hidden; color: #d2d9dc; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
    .cmk-flow-interface { margin-bottom: 20px; }
    .cmk-flow-sequence { margin-bottom: 20px; padding: 14px 16px; border: 1px solid #304047; border-radius: 10px; background: rgba(20,31,36,.82); }
    .cmk-flow-sequence-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .cmk-flow-sequence-label { display: block; margin-bottom: 4px; color: #7f8d94; font-size: 10px; text-transform: uppercase; letter-spacing: .06em; }
    .cmk-flow-sequence-value { color: #c6d1d5; font-size: 12px; }
    .cmk-flow-sequence-links { display: flex; flex-wrap: wrap; gap: 6px; }
    .cmk-flow-sequence-link { padding: 4px 8px; border: 1px solid #385057; border-radius: 6px; background: #18272c; color: #65d8d5; font: inherit; font-size: 12px; text-align: left; cursor: pointer; }
    .cmk-flow-sequence-link:hover { border-color: #46c9c6; background: #1b3438; color: #91efec; }
    .cmk-flow-sequence-text { color: #9eabb0; }
    .cmk-flow-dependency-note { margin: 11px 0 0; padding-top: 10px; border-top: 1px solid #2d3a40; color: #9fb0b6; font-size: 12px; }
    .cmk-flow-sockets { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .cmk-flow-socket-group { min-width: 0; padding: 12px 14px; border: 1px solid #2c373e; border-radius: 9px; background: #141b20; }
    .cmk-flow-socket-label { display: block; margin-bottom: 5px; color: #7f8c93; font-size: 10px; text-transform: uppercase; letter-spacing: .07em; }
    .cmk-flow-socket-values { color: #bac5ca; font-size: 12px; overflow-wrap: anywhere; }
    .cmk-flow-preview { margin-bottom: 20px; }
    .cmk-flow-preview-stage { min-height: 190px; display: grid; place-items: center; padding: 22px; overflow: hidden; border: 1px solid #2d3940; border-radius: 11px; background-color: #10161a; background-image: radial-gradient(#29353b 1px, transparent 1px); background-size: 16px 16px; }
    .cmk-flow-preview-gallery { width: 100%; display: grid; gap: 12px; justify-items: center; }
    .cmk-flow-preview-image { display: block; max-width: 100%; max-height: 390px; border: 1px solid #465159; border-radius: 9px; object-fit: contain; box-shadow: 0 12px 34px rgba(0,0,0,.42); }
    .cmk-flow-preview-tabs { display: flex; flex-wrap: wrap; justify-content: center; gap: 7px; }
    .cmk-flow-preview-tab { min-height: 30px; padding: 0 12px; border: 1px solid #3b484f; border-radius: 7px; background: #1a2328; color: #aebbc0; font: inherit; font-size: 12px; cursor: pointer; }
    .cmk-flow-preview-tab:hover { border-color: #52636b; color: #e2e9eb; }
    .cmk-flow-preview-tab.is-active { border-color: #42cbc8; background: #153638; color: #62e0dc; }
    .cmk-flow-preview-missing { color: #75838a; font-size: 12px; }
    .cmk-flow-actions { display: flex; justify-content: flex-end; padding-top: 2px; }
    .cmk-toolbox-detail { max-width: 760px; }
    .cmk-toolbox-detail.is-featured { max-width: none; }
    .cmk-toolbox-detail h3 { margin: 5px 0 2px; font-size: 25px; line-height: 1.2; }
    .cmk-toolbox-category { color: #8e9ba2; font-size: 12px; }
    .cmk-toolbox-description { margin: 18px 0 24px; color: #b8c2c7; }
    .cmk-toolbox-highlight { margin-bottom: 20px; }
    .cmk-toolbox-sockets { margin-bottom: 24px; }
    .cmk-flow-insert { min-width: 230px; min-height: 44px; padding: 0 20px; border: 0; border-radius: 9px; background: #35cfca; color: #071112; font: inherit; font-weight: 750; cursor: pointer; }
    .cmk-flow-insert:hover { background: #38d4d6; }
    .cmk-flow-empty, .cmk-flow-error { padding: 22px 14px; color: #9da5ab; }
    .cmk-flow-error { color: #ffb4b4; }
    @media (max-width: 850px) { .cmk-flow-browser { height: 94vh; } .cmk-flow-content { grid-template-columns: 1fr; grid-template-rows: minmax(250px, 42%) 1fr; } .cmk-flow-sidebar { border-right: 0; border-bottom: 1px solid #30353a; } .cmk-flow-detail-grid, .cmk-flow-sockets { grid-template-columns: 1fr; } .cmk-flow-meta { grid-template-columns: 1fr 1fr; } .cmk-storage-project { grid-template-columns: 1fr; } .cmk-storage-project-actions { min-width: 0; } }
  `;
  document.head.append(style);
}

async function fetchJson(path) {
  const response = await api.fetchApi(path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function fetchNodeMetadata() {
  try {
    const response = await fetch("/extensions/cmk_nodes/flow_node_metadata.json", { cache: "no-store" });
    if (!response.ok) return {};
    return (await response.json()).nodes || {};
  } catch {
    return {};
  }
}

async function fetchToolboxMetadata() {
  try {
    const response = await fetch("/extensions/cmk_nodes/toolbox_node_metadata.json", { cache: "no-store" });
    if (!response.ok) return {};
    return (await response.json()).nodes || {};
  } catch {
    return {};
  }
}

async function fetchEnglishContent() {
  try {
    const response = await fetch("/extensions/cmk_nodes/browser_content_en.json", { cache: "no-store" });
    return response.ok ? await response.json() : { flows: {}, toolbox: {} };
  } catch {
    return { flows: {}, toolbox: {} };
  }
}

async function fetchReferenceRegistry() {
  try {
    return await fetchJson("/cmk/showcase-workflows");
  } catch {
    return { workflows: [] };
  }
}

function normalizePreviews(metadata = {}) {
  const rawPreviews = Array.isArray(metadata.previews)
    ? metadata.previews
    : metadata.preview
      ? [{ src: metadata.preview, label: "Vorschau" }]
      : [];

  return rawPreviews
    .map((preview, index) => typeof preview === "string" ? { src: preview } : preview)
    .filter((preview) => preview?.src)
    .map((preview, index) => ({
      src: `/extensions/cmk_nodes/${preview.src}`,
      label: preview.label || `Ansicht ${index + 1}`,
    }));
}

function isCableInput(spec) {
  const type = spec?.[0];
  if (Array.isArray(type)) return false;
  if (spec?.[1]?.forceInput) return true;
  return !new Set(["BOOLEAN", "COMBO", "FLOAT", "IMAGEUPLOAD", "INT", "STRING"]).has(String(type || ""));
}

function discoverCuratedNodes(nodeRegistry, nodeMetadata, englishContent) {
  return Object.entries(nodeRegistry)
    .filter(([, nodeDef]) => String(nodeDef?.category || "").startsWith("CMK/Flow/"))
    .map(([nodeType, nodeDef]) => {
      const metadata = { ...(nodeMetadata[nodeType] || {}), ...(language === "en" ? englishContent.flows?.[nodeType] : {}) };
      const displayName = nodeDef.display_name || nodeDef.name || nodeType;
      const category = nodeDef.category.split("/").at(-1);
      const numericPrefix = Number(displayName.match(/(?:·\s*)?(\d{1,2})\b/)?.[1]);
      const categoryOrder = { Input: 4, Process: 70, Finish: 95 }[category] ?? 80;
      const required = Object.entries(nodeDef.input?.required || {}).filter(([, spec]) => isCableInput(spec)).map(([name]) => name);
      const optional = Object.entries(nodeDef.input?.optional || {}).filter(([, spec]) => isCableInput(spec)).map(([name]) => name);
      const inputDetails = [
        ...Object.entries(nodeDef.input?.required || {}),
        ...Object.entries(nodeDef.input?.optional || {}),
      ].map(([name, spec]) => ({ name, type: Array.isArray(spec?.[0]) ? "Auswahl" : String(spec?.[0] || "") }));

      return {
        entryId: `node:${nodeType}`,
        kind: "node",
        nodeType,
        marker: "NODE",
        name: displayName,
        displayName: displayName.replace(/^CMK Flow\s*·\s*/, ""),
        category,
        domain: "Flow Node",
        description: metadata.description || nodeDef.description || "Ein direkt einsetzbarer Baustein für CMK Flow.",
        status: "STABLE",
        version: "—",
        author: "CMK Nodes",
        compatibility: [],
        features: Array.isArray(metadata.features) ? metadata.features : ["Direkt als einzelne Node einsetzbar"],
        inputs: [...required, ...optional.map((name) => `${name} (optional)`) ],
        inputDetails,
        outputs: (nodeDef.output_name || nodeDef.output || []).map(String),
        order: Number.isFinite(numericPrefix) ? numericPrefix : categoryOrder,
        searchAliases: [nodeType, category, "node"],
        icon: metadata.icon || "",
        recommendedBefore: Array.isArray(metadata.recommendedPredecessors)
          ? metadata.recommendedPredecessors
          : (Array.isArray(metadata.recommendedBefore) ? metadata.recommendedBefore : []),
        recommendedAfter: Array.isArray(metadata.recommendedSuccessors)
          ? metadata.recommendedSuccessors
          : (Array.isArray(metadata.recommendedAfter) ? metadata.recommendedAfter : []),
        dependencyNote: metadata.dependencyNote || "",
        placementNote: metadata.placementNote || "Kann an der passenden Stelle in einen CMK Flow eingefügt werden.",
        previews: normalizePreviews(metadata),
        previewAlt: metadata.previewAlt || `${displayName} Vorschau`,
      };
    });
}

function conciseDescription(value) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) return language === "en" ? "A directly usable node from the CMK Toolbox." : "Direkt einsetzbare Node aus dem CMK-Baukasten.";
  const firstSentence = text.match(/^.*?[.!?](?:\s|$)/)?.[0]?.trim() || text;
  return firstSentence.length > 220 ? `${firstSentence.slice(0, 217).trim()}…` : firstSentence;
}

function toolboxFallbackDescription(category) {
  if (language === "en") {
    const english = { ControlNet:"Prepares ControlNet data for use in the workflow.", Diagnostics:"Supports inspection, display, or forwarding of diagnostic and log data.", Face:"Tool for detecting or selectively processing faces.", "I-O":"Tool for project and media input or output.", Image:"Tool for processing and forwarding image data.", "Mask & SEGS":"Processes masks or detected image regions for subsequent steps.", "Model & LoRA":"Prepares models or LoRA data for the workflow.", Video:"Tool for loading, processing, or saving video data." };
    return english[category] || "A directly usable node from the CMK Toolbox.";
  }
  const descriptions = {
    ControlNet: "Bereitet ControlNet-Daten für die weitere Verwendung im Workflow auf.",
    Diagnostics: "Unterstützt die Prüfung, Darstellung oder Weitergabe von Diagnose- und Protokolldaten.",
    Face: "Werkzeug zur Erkennung oder gezielten Verarbeitung von Gesichtern.",
    "I-O": "Werkzeug für die Ein- oder Ausgabe von Projekt- und Mediendaten.",
    Image: "Werkzeug zur gezielten Verarbeitung und Weitergabe von Bilddaten.",
    "Mask & SEGS": "Verarbeitet Masken oder erkannte Bildbereiche für nachfolgende Arbeitsschritte.",
    "Model & LoRA": "Bereitet Modelle oder LoRA-Daten für den Workflow vor.",
    Video: "Werkzeug zum Laden, Verarbeiten oder Speichern von Videodaten.",
  };
  return descriptions[category] || "Direkt einsetzbare Node aus dem CMK-Baukasten.";
}

function compactSocketNames(values) {
  const numberedGroups = new Map();
  const regular = [];

  for (const value of values) {
    const match = String(value).match(/^(.*)_(\d+)( \(optional\))?$/);
    if (!match) {
      regular.push(value);
      continue;
    }
    const [, base, number, optional = ""] = match;
    const key = `${base}\u0000${optional}`;
    const group = numberedGroups.get(key) || { base, optional, numbers: [] };
    group.numbers.push(Number(number));
    numberedGroups.set(key, group);
  }

  for (const group of numberedGroups.values()) {
    group.numbers.sort((a, b) => a - b);
    const unnumbered = `${group.base}${group.optional}`;
    const unnumberedIndex = regular.indexOf(unnumbered);
    if (group.numbers[0] === 2 && unnumberedIndex !== -1) {
      regular.splice(unnumberedIndex, 1);
      group.numbers.unshift(1);
    }
    const consecutive = group.numbers.every((number, index) => index === 0 || number === group.numbers[index - 1] + 1);
    if (group.numbers.length >= 3 && consecutive) {
      regular.push(`${group.base}_${group.numbers[0]}–${group.numbers.at(-1)} (${language === "en" ? "dynamic" : "dynamisch"}${group.optional ? ", optional" : ""})`);
    } else {
      regular.push(...group.numbers.map((number) => `${group.base}_${number}${group.optional}`));
    }
  }

  return regular;
}

function discoverToolboxNodes(nodeRegistry, toolboxMetadata, englishContent) {
  return Object.entries(nodeRegistry)
    .filter(([, nodeDef]) => String(nodeDef?.category || "").startsWith("CMK/Toolbox/"))
    .map(([nodeType, nodeDef]) => {
      const name = nodeDef.display_name || nodeDef.name || nodeType;
      const category = String(nodeDef.category).replace(/^CMK\/Toolbox\/?/, "") || "Allgemein";
      const metadata = toolboxMetadata[nodeType] || {};
      const required = Object.entries(nodeDef.input?.required || {})
        .filter(([, spec]) => isCableInput(spec))
        .map(([input]) => input);
      const optional = Object.entries(nodeDef.input?.optional || {})
        .filter(([, spec]) => isCableInput(spec))
        .map(([input]) => `${input} (optional)`);
      return {
        entryId: `toolbox:${nodeType}`,
        kind: "node",
        nodeType,
        displayName: name,
        category,
        description: conciseDescription((language === "en" && englishContent.toolbox?.[nodeType]) || metadata.description || nodeDef.description || toolboxFallbackDescription(category)),
        highlight: language === "en" ? (metadata.highlight_en || metadata.highlight || "") : (metadata.highlight || ""),
        previews: normalizePreviews({
          previews: (metadata.previews || []).map((preview) => ({
            ...preview,
            label: language === "en" ? (preview.label_en || preview.label) : preview.label,
          })),
        }),
        previewAlt: name,
        special: metadata.special === true,
        inputs: [...required, ...optional],
        outputs: (nodeDef.output_name || nodeDef.output || []).map(String),
        searchAliases: [nodeType, nodeDef.category],
      };
    })
    .sort((a, b) => a.category.localeCompare(b.category, "de") || a.displayName.localeCompare(b.displayName, "de"));
}

async function discoverFlows() {
  const [registry, nodeRegistry, nodeMetadata, toolboxMetadata, englishContent, referenceRegistry] = await Promise.all([
    fetchJson("/global_subgraphs"),
    fetchJson("/object_info"),
    fetchNodeMetadata(),
    fetchToolboxMetadata(),
    fetchEnglishContent(),
    fetchReferenceRegistry(),
  ]);
  const cmkEntries = Object.entries(registry).filter(([, entry]) => entry?.info?.node_pack === NODE_PACK);

  const candidates = await Promise.all(cmkEntries.map(async ([entryId, entry]) => {
    const fullEntry = await fetchJson(`/global_subgraphs/${encodeURIComponent(entryId)}`);
    const data = typeof fullEntry.data === "string" ? JSON.parse(fullEntry.data) : fullEntry.data;
    const baseMetadata = data?.extra?.CMKFlow;
    const metadata = { ...baseMetadata, ...(language === "en" ? englishContent.flows?.[entry.name] : {}) };
    if (!metadata?.published) return null;

    const status = String(metadata.status || "").toUpperCase();
    if (!metadata.description || !metadata.category || !ALLOWED_STATUS.has(status)) {
      console.warn(`[${EXTENSION_NAME}] Ignoriere unvollständige Flow-Metadaten:`, entry.name);
      return null;
    }

    return {
      entryId,
      kind: "subgraph",
      marker: String(metadata.order).padStart(2, "0"),
      blueprintId: data?.definitions?.subgraphs?.find((definition) => definition.name === entry.name)?.id
        || data?.definitions?.subgraphs?.[0]?.id,
      blueprintItems: {
        nodes: data?.nodes || [],
        subgraphs: data?.definitions?.subgraphs || [],
      },
      name: entry.name,
      displayName: metadata.displayName || entry.name.replace(/^CMK Flow\s*·\s*/, ""),
      category: metadata.category,
      domain: metadata.domain || metadata.category,
      description: metadata.description,
      status,
      version: metadata.version || "—",
      author: metadata.author || "CMK Nodes",
      compatibility: Array.isArray(metadata.compatibility) ? metadata.compatibility : [],
      features: Array.isArray(metadata.features) ? metadata.features : [],
      featureCallout: metadata.featureCallout || null,
      inputs: (data?.nodes?.[0]?.inputs || []).map((input) => input.name),
      inputDetails: (data?.nodes?.[0]?.inputs || []).map((input) => ({ name: input.name, type: String(input.type || "") })),
      outputs: (data?.nodes?.[0]?.outputs || []).map((output) => output.name),
      order: Number.isFinite(metadata.order) ? metadata.order : 999,
      searchAliases: Array.isArray(metadata.searchAliases) ? metadata.searchAliases : [],
      icon: metadata.icon || "",
      recommendedBefore: Array.isArray(metadata.recommendedPredecessors)
        ? metadata.recommendedPredecessors
        : (Array.isArray(metadata.recommendedBefore) ? metadata.recommendedBefore : []),
      recommendedAfter: Array.isArray(metadata.recommendedSuccessors)
        ? metadata.recommendedSuccessors
        : (Array.isArray(metadata.recommendedAfter) ? metadata.recommendedAfter : []),
      dependencyNote: metadata.dependencyNote || "",
      placementNote: metadata.placementNote || "Kann an der passenden Stelle in einen CMK Flow eingefügt werden.",
      previews: normalizePreviews(metadata),
      previewAlt: metadata.previewAlt || `${entry.name} Vorschau`,
      variantOf: metadata.variantOf || "",
      variantLabel: metadata.variantLabel || "",
    };
  }));

  const discovered = candidates.filter(Boolean);
  const variants = discovered.filter((flow) => flow.variantOf);
  const primary = discovered.filter((flow) => !flow.variantOf);
  for (const flow of primary) {
    flow.variants = [flow, ...variants.filter((variant) => variant.variantOf === flow.name)];
  }

  return {
    flows: [...primary, ...discoverCuratedNodes(nodeRegistry, nodeMetadata, englishContent)]
      .sort((a, b) => a.order - b.order || a.name.localeCompare(b.name, "de")),
    toolbox: discoverToolboxNodes(nodeRegistry, toolboxMetadata, englishContent),
    references: Array.isArray(referenceRegistry.workflows) ? referenceRegistry.workflows : [],
  };
}

function loadBrowserData() {
  browserDataPromise ??= discoverFlows().catch((error) => {
    browserDataPromise = undefined;
    throw error;
  });
  return browserDataPromise;
}

function focusLoadedWorkflow() {
  const canvas = app.canvas;
  const graph = canvas?.graph || app.graph;
  const nodes = Array.isArray(graph?._nodes) ? graph._nodes : [];
  const surface = canvas?.canvas;
  const ds = canvas?.ds;
  if (!nodes.length || !surface || !ds) return;

  const bounds = nodes.reduce((result, node) => {
    const x = Number(node?.pos?.[0]) || 0;
    const y = Number(node?.pos?.[1]) || 0;
    const width = Math.max(Number(node?.size?.[0]) || 0, 40);
    const height = Math.max(Number(node?.size?.[1]) || 0, 30);
    result.minX = Math.min(result.minX, x);
    result.minY = Math.min(result.minY, y);
    result.maxX = Math.max(result.maxX, x + width);
    result.maxY = Math.max(result.maxY, y + height);
    return result;
  }, { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity });

  const viewport = surface.getBoundingClientRect?.();
  const viewportWidth = viewport?.width || surface.clientWidth || surface.width;
  const viewportHeight = viewport?.height || surface.clientHeight || surface.height;
  const edgeInset = (side) => {
    if (!viewport || typeof document.elementsFromPoint !== "function") return 0;
    const x = side === "left" ? viewport.left + 4 : viewport.right - 4;
    const y = viewport.top + viewport.height / 2;
    const candidates = document.elementsFromPoint(x, y);
    let inset = 0;
    for (const element of candidates) {
      if (element === surface || element === document.body || element === document.documentElement) continue;
      if (element.contains?.(surface) || surface.contains?.(element)) continue;
      const rect = element.getBoundingClientRect?.();
      if (!rect || rect.height < viewport.height * 0.35 || rect.width > viewport.width * 0.7) continue;
      if (side === "left" && rect.left <= viewport.left + 12 && rect.right > viewport.left) {
        inset = Math.max(inset, Math.min(rect.right - viewport.left, viewport.width));
      }
      if (side === "right" && rect.right >= viewport.right - 12 && rect.left < viewport.right) {
        inset = Math.max(inset, Math.min(viewport.right - rect.left, viewport.width));
      }
    }
    return inset;
  };
  const leftInset = edgeInset("left");
  const rightInset = edgeInset("right");
  const visibleWidth = Math.max(viewportWidth - leftInset - rightInset, 1);
  const contentWidth = Math.max(bounds.maxX - bounds.minX, 1);
  const contentHeight = Math.max(bounds.maxY - bounds.minY, 1);
  const padding = 96;
  const scale = Math.max(0.05, Math.min(1, (visibleWidth - padding) / contentWidth, (viewportHeight - padding) / contentHeight));
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;

  ds.scale = scale;
  ds.offset[0] = (leftInset + visibleWidth / 2) / scale - centerX;
  ds.offset[1] = viewportHeight / (2 * scale) - centerY;
  canvas.setDirty?.(true, true);
}

function focusLoadedWorkflowAfterRender() {
  requestAnimationFrame(() => requestAnimationFrame(focusLoadedWorkflow));
  // Some ComfyUI builds finish switching the workflow tab after loadGraphData resolves.
  window.setTimeout(focusLoadedWorkflow, 180);
}

function placeAndAddFlow(flow) {
  const canvas = app.canvas;
  const mouse = canvas?.graph_mouse;
  const position = Array.isArray(mouse) ? [mouse[0], mouse[1]] : [0, 0];

  if (flow.kind === "node") {
    const node = LiteGraph.createNode(flow.nodeType);
    const graph = canvas?.graph || app.graph;
    if (!node || !graph) throw new Error("Die ausgewählte Node konnte nicht erzeugt werden.");
    node.pos = position;
    graph.beforeChange?.();
    graph.add(node);
    graph.afterChange?.();
    canvas?.selectNode?.(node, false);
    canvas?.setDirty?.(true, true);
    return;
  }

  // This is the same ComfyUI mechanism used by the node library for blueprints:
  // it imports the node together with its subgraph definition into the active graph.
  if (typeof canvas?._deserializeItems === "function") {
    // _deserializeItems mutates its input while remapping subgraph and node IDs.
    // Always provide a fresh plain-JSON copy so runtime Maps/proxies from one
    // insertion can never contaminate the browser registry or a later paste.
    const blueprintItems = JSON.parse(JSON.stringify(flow.blueprintItems));
    const results = canvas._deserializeItems(blueprintItems, { position });
    const node = results?.nodes?.values?.().next?.().value
      || results?.created?.find?.((item) => item?.type === flow.blueprintId || item?.subgraph);
    if (!node) throw new Error("Der Flow konnte nicht in den aktuellen Graphen eingefügt werden.");
    canvas.selectNode?.(node, false);
    canvas.setDirty?.(true, true);
    return;
  }

  // Compatibility fallback for older ComfyUI frontends that registered blueprints
  // directly as LiteGraph node types.
  const nodeTypes = [flow.blueprintId, `SubgraphBlueprint.${flow.entryId}`].filter(Boolean);
  const node = nodeTypes.map((nodeType) => LiteGraph.createNode(nodeType)).find(Boolean);
  if (!node) throw new Error("Diese ComfyUI-Version unterstützt das Einfügen dieses Flow-Blueprints nicht.");

  const graph = canvas?.graph || app.graph;
  node.pos = position;
  graph.beforeChange?.();
  graph.add(node);
  graph.afterChange?.();
  canvas?.selectNode?.(node, false);
  canvas?.setDirty?.(true, true);
}

function fillTextList(container, values, emptyText) {
  container.replaceChildren();
  const items = values.length ? values : [emptyText];
  for (const value of items) {
    const item = document.createElement("li");
    if (value && typeof value === "object") {
      const content = document.createElement("div");
      const title = document.createElement("span");
      title.className = "cmk-flow-feature-title";
      title.textContent = value.title || "";
      content.append(title);
      if (value.description) {
        const description = document.createElement("span");
        description.className = "cmk-flow-feature-description";
        description.textContent = value.description;
        content.append(description);
      }
      item.append(content);
    } else {
      item.textContent = value;
    }
    if (!values.length) item.className = "is-optional";
    container.append(item);
  }
}

function setMetaValue(detail, key, value) {
  detail.querySelector(`[data-meta="${key}"]`).textContent = value || "—";
}

function renderNodePreview(container, flow) {
  container.replaceChildren();
  if (!flow.previews?.length) {
    const missing = document.createElement("span");
    missing.className = "cmk-flow-preview-missing";
    missing.textContent = t("previewMissing");
    container.append(missing);
    return;
  }

  const gallery = document.createElement("div");
  gallery.className = "cmk-flow-preview-gallery";

  const image = document.createElement("img");
  image.className = "cmk-flow-preview-image";
  image.alt = flow.previewAlt;
  image.loading = "lazy";
  gallery.append(image);

  const tabs = document.createElement("div");
  tabs.className = "cmk-flow-preview-tabs";
  tabs.setAttribute("role", "group");
  tabs.setAttribute("aria-label", "Vorschauansicht auswählen");

  const buttons = flow.previews.map((preview, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "cmk-flow-preview-tab";
    button.textContent = preview.label;
    button.addEventListener("click", () => selectPreview(index));
    tabs.append(button);
    return button;
  });

  const selectPreview = (index) => {
    const preview = flow.previews[index];
    image.src = preview.src;
    image.alt = `${flow.previewAlt} – ${preview.label}`;
    buttons.forEach((button, buttonIndex) => {
      const active = buttonIndex === index;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  };

  if (flow.previews.length > 1) gallery.append(tabs);
  container.append(gallery);
  selectPreview(0);
}

function flowIcon(flow) {
  const value = `${flow.icon} ${flow.displayName}`.toLocaleLowerCase("de");
  let body;
  if (value.includes("lora")) body = '<path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/>';
  else if (value.includes("checkpoint") || value.includes("vae")) body = '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z"/><path d="m4.3 7.7 7.7 4.4 7.7-4.4M12 12v9"/>';
  else if (value.includes("controlnet")) body = '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3"/>';
  else if (value.includes("ksampler")) body = '<path d="M3 12h3l2.2-6 4 12 2.5-8 2 2H21"/>';
  else if (value.includes("refiner")) body = '<path d="m12 3 1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3Z"/><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z"/>';
  else if (value.includes("detailer")) body = '<path d="M8 3H3v5m13-5h5v5M8 21H3v-5m13 5h5v-5"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1"/>';
  else if (value.includes("faceswap")) body = '<path d="M7 7h9l-3-3m4 13H8l3 3"/><path d="M19 8a7 7 0 0 1 0 8M5 16a7 7 0 0 1 0-8"/>';
  else if (value.includes("faceprocess") || value.includes("face process")) body = '<circle cx="12" cy="12" r="8"/><path d="M9 10h.01M15 10h.01M9 15c1.8 1.3 4.2 1.3 6 0"/>';
  else if (value.includes("save") || value.includes("upscale")) body = '<path d="M12 16V4m0 0L7 9m5-5 5 5"/><path d="M5 14v6h14v-6"/>';
  else if (value.includes("create")) body = '<path d="M12 3v18M3 12h18"/><circle cx="12" cy="12" r="8"/>';
  else body = '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8" cy="9" r="2"/><path d="m21 15-5-5L5 20"/>';
  return `<svg viewBox="0 0 24 24" aria-hidden="true">${body}</svg>`;
}

function recommendationKey(value) {
  return String(value).toLocaleLowerCase("de")
    .replace(/^cmk flow\s*·?\s*/, "")
    .replace(/\s*\(optional\)\s*/g, "")
    .replace(/[^a-z0-9äöüß]+/g, " ")
    .trim();
}

function renderFlowBrowser(root, flows) {
  const list = root.querySelector(".cmk-flow-list");
  const detail = root.querySelector(".cmk-flow-detail");
  const categorySelect = root.querySelector("select");
  const categories = [t("all"), ...new Set(flows.map((flow) => flow.category))];
  if (!categories.includes(currentCategory)) currentCategory = t("all");
  categorySelect.innerHTML = categories.map((category) => `<option>${category}</option>`).join("");
  categorySelect.value = categories.includes(currentCategory) ? currentCategory : t("all");
  const selectedVariants = new Map();

  const update = () => {
    const listScrollTop = list.scrollTop;
    const term = searchText.trim().toLocaleLowerCase("de");
    const visible = flows.filter((flow) => {
      if (currentCategory !== t("all") && flow.category !== currentCategory) return false;
      if (!term) return true;
      return [flow.name, flow.displayName, flow.description, flow.category, flow.domain, ...flow.searchAliases]
        .join(" ").toLocaleLowerCase("de").includes(term);
    });

    if (!visible.some((flow) => flow.entryId === selectedId)) selectedId = visible[0]?.entryId ?? null;
    list.replaceChildren();
    if (!visible.length) {
      list.innerHTML = `<div class="cmk-flow-empty">${t("noFlows")}</div>`;
    } else {
      for (const flow of visible) {
        const button = document.createElement("button");
        button.className = `cmk-flow-item${flow.entryId === selectedId ? " is-selected" : ""}`;
        button.innerHTML = `<span class="cmk-flow-item-index"></span><span><span class="cmk-flow-item-name"></span><span class="cmk-flow-item-category"></span></span>`;
        button.querySelector(".cmk-flow-item-index").innerHTML = flowIcon(flow);
        button.querySelector(".cmk-flow-item-name").textContent = flow.displayName;
        button.querySelector(".cmk-flow-item-category").textContent = flow.category;
        button.addEventListener("click", () => { selectedId = flow.entryId; update(); });
        list.append(button);
      }
    }
    list.scrollTop = listScrollTop;

    const selectedPrimary = flows.find((flow) => flow.entryId === selectedId);
    if (!selectedPrimary) {
      detail.innerHTML = `<div class="cmk-flow-empty">${t("selectFlow")}</div>`;
      return;
    }
    const variantOptions = selectedPrimary.variants || [selectedPrimary];
    const selected = variantOptions.find((flow) => flow.entryId === selectedVariants.get(selectedPrimary.entryId)) || selectedPrimary;
    detail.innerHTML = `
      <div class="cmk-flow-detail-header">
        <span class="cmk-flow-badge"></span>
        <h3></h3>
        <div class="cmk-flow-domain"></div>
        <p class="cmk-flow-description"></p>
      </div>
      <div class="cmk-flow-variants" hidden></div>
      <div class="cmk-flow-detail-grid">
        <section><h4 class="cmk-flow-section-title">${t("features")}</h4><div class="cmk-flow-card"><ul data-list="features"></ul><div class="cmk-flow-feature-callout" hidden><div><div class="cmk-flow-feature-callout-title"></div><div class="cmk-flow-feature-callout-text"></div></div></div></div></section>
        <section><h4 class="cmk-flow-section-title">${t("placement")}</h4><div class="cmk-flow-card cmk-flow-placement"></div></section>
      </div>
      <div class="cmk-flow-meta">
        <div class="cmk-flow-meta-item"><span class="cmk-flow-meta-label">${t("category")}</span><span class="cmk-flow-meta-value" data-meta="category"></span></div>
        <div class="cmk-flow-meta-item"><span class="cmk-flow-meta-label">${t("compatibility")}</span><span class="cmk-flow-meta-value" data-meta="compatibility"></span></div>
        <div class="cmk-flow-meta-item"><span class="cmk-flow-meta-label">${t("version")}</span><span class="cmk-flow-meta-value" data-meta="version"></span></div>
        <div class="cmk-flow-meta-item"><span class="cmk-flow-meta-label">${t("author")}</span><span class="cmk-flow-meta-value" data-meta="author"></span></div>
      </div>
      <section class="cmk-flow-interface">
        <h4 class="cmk-flow-section-title">${t("interfaces")}</h4>
        <div class="cmk-flow-sockets">
          <div class="cmk-flow-socket-group"><span class="cmk-flow-socket-label">${t("inputs")}</span><div class="cmk-flow-socket-values" data-sockets="inputs"></div></div>
          <div class="cmk-flow-socket-group"><span class="cmk-flow-socket-label">${t("outputs")}</span><div class="cmk-flow-socket-values" data-sockets="outputs"></div></div>
        </div>
      </section>
      <section class="cmk-flow-sequence">
        <h4 class="cmk-flow-section-title">${t("related")}</h4>
        <div class="cmk-flow-sequence-grid">
          <div><span class="cmk-flow-sequence-label">${t("before")}</span><div class="cmk-flow-sequence-value" data-sequence="before"></div></div>
          <div><span class="cmk-flow-sequence-label">${t("after")}</span><div class="cmk-flow-sequence-value" data-sequence="after"></div></div>
        </div>
        <p class="cmk-flow-dependency-note"></p>
      </section>
      <section class="cmk-flow-preview">
        <h4 class="cmk-flow-section-title">${t("preview")}</h4>
        <div class="cmk-flow-preview-stage"></div>
      </section>
      <div class="cmk-flow-actions"><button class="cmk-flow-insert">${t("insert")}</button></div>`;
    const badge = detail.querySelector(".cmk-flow-badge");
    badge.textContent = selected.status;
    badge.dataset.status = selected.status;
    detail.querySelector("h3").textContent = selected.displayName;
    detail.querySelector(".cmk-flow-domain").textContent = selected.domain;
    detail.querySelector(".cmk-flow-description").textContent = selected.description;
    const variantBar = detail.querySelector(".cmk-flow-variants");
    if (variantOptions.length > 1) {
      variantBar.hidden = false;
      for (const variant of variantOptions) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `cmk-flow-variant${variant.entryId === selected.entryId ? " is-selected" : ""}`;
        button.textContent = variant === selectedPrimary ? t("recommended") : (variant.variantLabel || variant.displayName);
        button.addEventListener("click", () => {
          selectedVariants.set(selectedPrimary.entryId, variant.entryId);
          update();
          requestAnimationFrame(() => detail.scrollTo({ top: 0 }));
        });
        variantBar.append(button);
      }
    }
    fillTextList(detail.querySelector('[data-list="features"]'), selected.features, "Noch keine Funktionsübersicht hinterlegt");
    const featureCallout = detail.querySelector(".cmk-flow-feature-callout");
    if (selected.featureCallout?.title && selected.featureCallout?.text) {
      featureCallout.hidden = false;
      featureCallout.querySelector(".cmk-flow-feature-callout-title").textContent = selected.featureCallout.title;
      featureCallout.querySelector(".cmk-flow-feature-callout-text").textContent = selected.featureCallout.text;
    }
    detail.querySelector(".cmk-flow-placement").textContent = selected.placementNote;
    setMetaValue(detail, "category", selected.category);
    setMetaValue(detail, "compatibility", selected.compatibility.join(", "));
    setMetaValue(detail, "version", selected.version);
    setMetaValue(detail, "author", selected.author);
    detail.querySelector('[data-sockets="inputs"]').textContent = compactSocketNames(selected.inputs).join(" · ") || t("none");
    detail.querySelector('[data-sockets="outputs"]').textContent = selected.outputs.join(" · ") || t("none");
    const sequence = detail.querySelector(".cmk-flow-sequence");
    const renderRecommendations = (container, recommendations, fallback) => {
      container.replaceChildren();
      const values = recommendations.length ? recommendations : [fallback];
      const links = document.createElement("div");
      links.className = "cmk-flow-sequence-links";
      for (const value of values) {
        const key = recommendationKey(value);
        const exactTarget = flows.find((flow) => recommendationKey(flow.displayName) === key);
        const target = exactTarget || flows.find((flow) => {
          if (flow.entryId === selected.entryId) return false;
          const candidate = recommendationKey(flow.displayName);
          return candidate.includes(key) || key.includes(candidate);
        });
        const element = document.createElement(target ? "button" : "span");
        element.textContent = value;
        if (target) {
          element.type = "button";
          element.className = "cmk-flow-sequence-link";
          element.addEventListener("click", () => {
            selectedId = target.entryId;
            currentCategory = t("all");
            searchText = "";
            categorySelect.value = t("all");
            root.querySelector("input").value = "";
            update();
            requestAnimationFrame(() => {
              list.querySelector(".cmk-flow-item.is-selected")?.scrollIntoView({ block: "nearest", behavior: "smooth" });
              detail.scrollTo({ top: 0, behavior: "smooth" });
            });
          });
        } else {
          element.className = "cmk-flow-sequence-text";
        }
        links.append(element);
      }
      container.append(links);
    };
    renderRecommendations(sequence.querySelector('[data-sequence="before"]'), selected.recommendedBefore, t("canStart"));
    renderRecommendations(sequence.querySelector('[data-sequence="after"]'), selected.recommendedAfter, t("canFinish"));
    const note = sequence.querySelector(".cmk-flow-dependency-note");
    note.textContent = selected.dependencyNote;
    note.hidden = !selected.dependencyNote;
    renderNodePreview(detail.querySelector(".cmk-flow-preview-stage"), selected);
    detail.querySelector(".cmk-flow-insert").addEventListener("click", () => {
      try {
        placeAndAddFlow(selected);
        root.remove();
      } catch (error) {
        detail.insertAdjacentHTML("beforeend", `<div class="cmk-flow-error"></div>`);
        detail.querySelector(".cmk-flow-error").textContent = error.message;
      }
    });
  };

  root.querySelector("input").value = searchText;
  root.querySelector("input").addEventListener("input", (event) => { searchText = event.target.value; update(); });
  categorySelect.addEventListener("change", (event) => { currentCategory = event.target.value; update(); });
  update();
}

function renderToolboxBrowser(root, nodes) {
  const list = root.querySelector(".cmk-flow-list");
  const detail = root.querySelector(".cmk-flow-detail");
  const input = root.querySelector(".cmk-flow-controls input");
  const categorySelect = root.querySelector(".cmk-flow-controls select");
  const categories = [t("all"), ...new Set(nodes.map((node) => node.category))];
  if (!categories.includes(toolboxCategory)) toolboxCategory = t("all");
  categorySelect.innerHTML = categories.map((category) => `<option></option>`).join("");
  [...categorySelect.options].forEach((option, index) => { option.textContent = categories[index]; });
  categorySelect.value = categories.includes(toolboxCategory) ? toolboxCategory : t("all");

  const update = () => {
    const listScrollTop = list.scrollTop;
    const term = toolboxSearchText.trim().toLocaleLowerCase("de");
    const visible = nodes.filter((node) => {
      if (toolboxCategory !== t("all") && node.category !== toolboxCategory) return false;
      if (!term) return true;
      return [node.displayName, node.description, node.category, ...node.searchAliases]
        .join(" ").toLocaleLowerCase("de").includes(term);
    });

    if (!visible.some((node) => node.entryId === toolboxSelectedId)) toolboxSelectedId = visible[0]?.entryId ?? null;
    list.replaceChildren();
    if (!visible.length) {
      list.innerHTML = `<div class="cmk-flow-empty">${t("noNodes")}</div>`;
    } else {
      for (const node of visible) {
        const button = document.createElement("button");
        button.className = `cmk-flow-item cmk-toolbox-item${node.special ? " is-featured" : ""}${node.entryId === toolboxSelectedId ? " is-selected" : ""}`;
        button.innerHTML = '<span><span class="cmk-flow-item-name"></span><span class="cmk-flow-item-category"></span></span>';
        button.querySelector(".cmk-flow-item-name").textContent = node.displayName;
        button.querySelector(".cmk-flow-item-category").textContent = node.category;
        button.addEventListener("click", () => {
          toolboxSelectedId = node.entryId;
          update();
          requestAnimationFrame(() => detail.scrollTo({ top: 0 }));
        });
        list.append(button);
      }
    }
    list.scrollTop = listScrollTop;

    const selected = nodes.find((node) => node.entryId === toolboxSelectedId);
    if (!selected) {
      detail.innerHTML = `<div class="cmk-flow-empty">${t("selectNode")}</div>`;
      return;
    }

    detail.innerHTML = `
      <div class="cmk-toolbox-detail">
        <span class="cmk-toolbox-category"></span>
        <h3></h3>
        <p class="cmk-toolbox-description"></p>
        <section class="cmk-toolbox-highlight" hidden><h4 class="cmk-flow-section-title">${language === "de" ? "Warum dieses Werkzeug besonders ist" : "Why this tool stands out"}</h4><div class="cmk-flow-card"></div></section>
        <div class="cmk-flow-sockets cmk-toolbox-sockets">
          <div class="cmk-flow-socket-group"><span class="cmk-flow-socket-label">${t("inputs")}</span><div class="cmk-flow-socket-values" data-sockets="inputs"></div></div>
          <div class="cmk-flow-socket-group"><span class="cmk-flow-socket-label">${t("outputs")}</span><div class="cmk-flow-socket-values" data-sockets="outputs"></div></div>
        </div>
        <section class="cmk-toolbox-preview cmk-flow-preview" hidden><h4 class="cmk-flow-section-title">${t("preview")}</h4><div class="cmk-flow-preview-stage"></div></section>
        <div class="cmk-flow-actions"><button class="cmk-flow-insert">${t("insert")}</button></div>
      </div>`;
    detail.querySelector(".cmk-toolbox-category").textContent = selected.category;
    detail.querySelector(".cmk-toolbox-detail").classList.toggle("is-featured", selected.special);
    detail.querySelector("h3").textContent = selected.displayName;
    detail.querySelector(".cmk-toolbox-description").textContent = selected.description;
    const highlight = detail.querySelector(".cmk-toolbox-highlight");
    highlight.hidden = !selected.highlight;
    highlight.querySelector(".cmk-flow-card").textContent = selected.highlight;
    detail.querySelector('[data-sockets="inputs"]').textContent = compactSocketNames(selected.inputs).join(" · ") || t("none");
    detail.querySelector('[data-sockets="outputs"]').textContent = selected.outputs.join(" · ") || t("none");
    const preview = detail.querySelector(".cmk-toolbox-preview");
    preview.hidden = !selected.previews.length;
    if (selected.previews.length) renderNodePreview(preview.querySelector(".cmk-flow-preview-stage"), selected);
    detail.querySelector(".cmk-flow-insert").addEventListener("click", () => {
      try {
        placeAndAddFlow(selected);
        root.remove();
      } catch (error) {
        detail.insertAdjacentHTML("beforeend", '<div class="cmk-flow-error"></div>');
        detail.querySelector(".cmk-flow-error").textContent = error.message;
      }
    });
  };

  input.value = toolboxSearchText;
  input.addEventListener("input", (event) => { toolboxSearchText = event.target.value; update(); });
  categorySelect.addEventListener("change", (event) => { toolboxCategory = event.target.value; update(); });
  update();
}

function renderReferenceBrowser(root, references) {
  const list = root.querySelector(".cmk-flow-list");
  const detail = root.querySelector(".cmk-flow-detail");
  const input = root.querySelector(".cmk-flow-controls input");
  const categorySelect = root.querySelector(".cmk-flow-controls select");
  categorySelect.hidden = true;
  const labels = language === "de"
    ? { empty:"Keine passenden Beispiel-Workflows gefunden.", catalogEmpty:"Die ersten beispielhaften CMK-Workflows entstehen derzeit und werden hier später veröffentlicht.", choose:"Beispiel-Workflow auswählen.", what:"Was dieser Workflow macht", cmk:"Das ist besonders CMK", preview:"Vorschau", open:"Als Kopie öffnen", confirm:"Der Beispiel-Workflow wird in einem neuen, ungespeicherten Workflow geöffnet. Fortfahren?" }
    : { empty:"No matching example workflows found.", catalogEmpty:"The first CMK example workflows are currently being created and will be published here later.", choose:"Select an example workflow.", what:"What this workflow does", cmk:"What makes it distinctly CMK", preview:"Preview", open:"Open as copy", confirm:"The example workflow will open in a new, unsaved workflow. Continue?" };

  const localizeReference = (reference) => ({
    ...reference,
    categoryLabel: language === "en" ? (reference.category_en || reference.category) : reference.category,
    descriptionLabel: language === "en" ? (reference.description_en || reference.description) : reference.description,
    highlightLabel: language === "en" ? (reference.cmkHighlight_en || reference.cmkHighlight) : reference.cmkHighlight,
    previews: normalizePreviews({
      previews: (reference.previews || []).map((preview) => ({
        ...preview,
        label: language === "en" ? (preview.label_en || preview.label) : preview.label,
      })),
    }),
    previewAlt: reference.name,
  });

  const update = () => {
    const scrollTop = list.scrollTop;
    const term = referenceSearchText.trim().toLocaleLowerCase(language);
    const visible = references.map(localizeReference).filter((reference) => !term || `${reference.name} ${reference.categoryLabel} ${reference.descriptionLabel} ${reference.highlightLabel}`.toLocaleLowerCase(language).includes(term));
    if (!visible.some((reference) => reference.filename === referenceSelectedId)) referenceSelectedId = visible[0]?.filename || null;
    list.replaceChildren();
    if (!visible.length) list.innerHTML = `<div class="cmk-flow-empty">${references.length ? labels.empty : labels.catalogEmpty}</div>`;
    for (const reference of visible) {
      const button = document.createElement("button");
      button.className = `cmk-flow-item cmk-toolbox-item${reference.filename === referenceSelectedId ? " is-selected" : ""}`;
      button.innerHTML = '<span><span class="cmk-flow-item-name"></span><span class="cmk-flow-item-category"></span></span>';
      button.querySelector(".cmk-flow-item-name").textContent = reference.name;
      button.querySelector(".cmk-flow-item-category").textContent = reference.categoryLabel;
      button.addEventListener("click", () => { referenceSelectedId = reference.filename; update(); requestAnimationFrame(() => detail.scrollTo({ top: 0 })); });
      list.append(button);
    }
    list.scrollTop = scrollTop;

    const selectedRaw = references.find((reference) => reference.filename === referenceSelectedId);
    const selected = selectedRaw ? localizeReference(selectedRaw) : null;
    if (!selected) { detail.innerHTML = `<div class="cmk-flow-empty">${labels.choose}</div>`; return; }
    detail.innerHTML = `<div class="cmk-toolbox-detail">
      <span class="cmk-toolbox-category"></span><h3></h3>
      <section><h4 class="cmk-flow-section-title">${labels.what}</h4><div class="cmk-flow-card cmk-showcase-description"></div></section>
      <section><h4 class="cmk-flow-section-title">${labels.cmk}</h4><div class="cmk-flow-card cmk-showcase-highlight"></div></section>
      <section class="cmk-flow-preview"><h4 class="cmk-flow-section-title">${labels.preview}</h4><div class="cmk-flow-preview-stage"></div></section>
      <div class="cmk-flow-actions"><button class="cmk-flow-insert">${labels.open}</button></div>
    </div>`;
    detail.querySelector(".cmk-toolbox-category").textContent = selected.categoryLabel;
    detail.querySelector("h3").textContent = selected.name;
    detail.querySelector(".cmk-showcase-description").textContent = selected.descriptionLabel;
    detail.querySelector(".cmk-showcase-highlight").textContent = selected.highlightLabel;
    renderNodePreview(detail.querySelector(".cmk-flow-preview-stage"), selected);
    detail.querySelector(".cmk-flow-insert").addEventListener("click", async () => {
      if (!window.confirm(labels.confirm)) return;
      try {
        const workflow = await fetchJson(`/cmk/showcase-workflows?file=${encodeURIComponent(selected.filename)}`);
        if (typeof app.loadGraphData !== "function") {
          throw new Error(language === "de" ? "Diese ComfyUI-Version kann die Referenz nicht sicher als Kopie öffnen." : "This ComfyUI version cannot safely open the reference as a copy.");
        }
        await app.loadGraphData(workflow, true, false, selected.name, { openSource: "cmk-showcase" });
        root.remove();
        focusLoadedWorkflowAfterRender();
      } catch (error) {
        detail.insertAdjacentHTML("beforeend", '<div class="cmk-flow-error"></div>');
        detail.querySelector(".cmk-flow-error").textContent = error.message;
      }
    });
  };
  input.value = referenceSearchText;
  input.addEventListener("input", (event) => { referenceSearchText = event.target.value; update(); });
  update();
}

function renderBrowser(root, data) {
  const controls = root.querySelector(".cmk-flow-controls");
  const subtitle = root.querySelector(".cmk-flow-subtitle");
  const activate = (tab) => {
    activeTab = tab;
    root.querySelectorAll(".cmk-browser-tab").forEach((button) => {
      const selected = button.dataset.cmkTab === tab;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-selected", String(selected));
    });
    const toolbox = tab === "toolbox";
    const references = tab === "references";
    subtitle.textContent = references
      ? (language === "de" ? "Beispiel-Workflows entdecken und als ungespeicherte Kopie öffnen." : "Explore example workflows and open them as an unsaved copy.")
      : toolbox ? t("subtitleToolbox") : t("subtitleFlow");
    controls.innerHTML = `<input type="search"><select aria-label="${t("category")}"></select>`;
    const input = controls.querySelector("input");
    input.placeholder = references ? (language === "de" ? "Beispiel-Workflows durchsuchen …" : "Search example workflows …") : toolbox ? t("searchToolbox") : t("searchFlows");
    input.setAttribute("aria-label", input.placeholder.replace(" …", ""));
    root.querySelector(".cmk-flow-list").replaceChildren();
    root.querySelector(".cmk-flow-detail").replaceChildren();
    if (references) renderReferenceBrowser(root, data.references);
    else if (toolbox) renderToolboxBrowser(root, data.toolbox);
    else renderFlowBrowser(root, data.flows);
    input.focus();
  };

  root.querySelectorAll(".cmk-browser-tab").forEach((button) => {
    button.addEventListener("click", () => activate(button.dataset.cmkTab));
  });
  activate(activeTab);
}

function openAboutDialog(root) {
  root.querySelector(".cmk-about-overlay")?.remove();
  const overlay = document.createElement("div");
  overlay.className = "cmk-about-overlay";
  const aboutBody = language === "de" ? `
        <p>CMK Flow begann mit dem Wunsch nach einer einzigen nativen Custom Node. Als sichtbar wurde, was mit Custom Nodes möglich ist, wuchs daraus ein Projekt. Dieses Projekt entwickelte sich schließlich zu einem Produkt.</p>
        <p><strong>CMK Flow wurde von Carsten Kirschner konzipiert und mit Unterstützung moderner KI-gestützter Werkzeuge für die Softwareentwicklung entwickelt.</strong></p>
        <p>Ein erheblicher Teil des Quellcodes wurde mit KI-Unterstützung erzeugt, analysiert und weiterentwickelt. CMK Flow entstand jedoch <strong>nicht</strong> aus einem einzigen Prompt. Es ist das Ergebnis zahlloser Architekturentscheidungen, Tests, Fehlersuchen, Überarbeitungen und vieler Stunden konzentrierter Entwicklungsarbeit.</p>
        <p>KI diente während des gesamten Projekts als leistungsfähiges Werkzeug zur Umsetzung. Produktvision, Systemarchitektur, Benutzererlebnis und grundlegende Gestaltungsphilosophie wurden in einem fortlaufenden, vom Menschen geführten Entwicklungsprozess bestimmt.</p>
        <p><strong>CMK Flow verbindet menschliche Kreativität, durchdachtes Produktdesign und KI-gestützte Softwareentwicklung. Es zeigt, wie menschliche Erfahrung und moderne KI gemeinsam Software schaffen können, die keiner von beiden allein hätte hervorbringen können.</strong></p>
        <div class="cmk-about-acknowledgements">
          <h4>Inspiration &amp; Anerkennung</h4>
          <p>Der <a href="https://github.com/willmiao/ComfyUI-Lora-Manager" target="_blank" rel="noopener noreferrer">ComfyUI LoRA Manager von willmiao</a> hat die Erwartung an den CMK Flow Browser wesentlich mitgeprägt: überraschend direkt aus ComfyUI eine komfortable Browseroberfläche zu öffnen und dort Checkpoints, LoRAs, Metadaten und Civitai-Inhalte zugänglich zu machen, zeigte eindrucksvoll, wie funktional eine integrierte Verwaltungsoberfläche sein kann.</p>
          <p>CMK Flow ist ein eigenständiges Projekt und steht in keiner offiziellen Verbindung zum LoRA Manager. Diese Nennung würdigt die gestalterische Inspiration und die umfangreiche Arbeit hinter diesem Open-Source-Projekt.</p>
        </div>
        <div class="cmk-about-responsibility">
          <h4>Verantwortung</h4>
          <p>CMK Flow will erwachsenen Menschen nicht vorschreiben, was sie privat und einvernehmlich gestalten. Gleichzeitig toleriert CMK keinerlei Veröffentlichung manipulierter Darstellungen realer Personen ohne deren ausdrückliche Einwilligung – insbesondere nicht in sexualisiertem oder entwürdigendem Zusammenhang.</p>
          <p>Technische Schutzmaßnahmen können Verantwortung unterstützen, aber nicht ersetzen. Wer CMK Flow verwendet, bleibt für einen respektvollen, einvernehmlichen und rechtmäßigen Umgang mit den erzeugten Inhalten verantwortlich.</p>
          <p>Eine gewisse Ironie bleibt: Ausgerechnet der Wunsch nach einer nativen FaceSwap-Engine gab den Anstoß für das gesamte Projekt.</p>
        </div>
        <div class="cmk-about-support">
          <p><strong>CMK Flow ist und bleibt vollständig kostenlos nutzbar.</strong></p>
          <p>Wenn CMK Flow deine Arbeit erleichtert, dir Zeit gespart oder einfach Freude bereitet hat, kannst du die weitere Entwicklung freiwillig unterstützen.</p>
          <p><strong>Danke, dass du CMK Flow verwendest.</strong></p>
        </div>
        <div class="cmk-about-legal">
          <p>Copyright © 2026 Carsten Kirschner</p>
          <p>Lizenziert unter GNU GPL v3 oder neuer (GPL-3.0-or-later).</p>
        </div>
        <div class="cmk-about-donation">
          <p>Eine Unterstützung ist vollständig freiwillig und hat keinen Einfluss auf den Funktionsumfang von CMK Flow.</p>
          <div class="cmk-about-donation-actions">
            <a class="cmk-about-donation-link" href="https://github.com/CMKFlow/cmk_nodes" target="_blank" rel="noopener noreferrer">Quellcode &amp; Dokumentation</a>
            <a class="cmk-about-donation-link" href="https://paypal.me/CMKFlow" target="_blank" rel="noopener noreferrer">Freiwillig via PayPal unterstützen</a>
          </div>
        </div>` : `
        <p>CMK Flow began with the wish for a single native custom node. Once it became clear what custom nodes could make possible, that node grew into a project. The project eventually evolved into a product.</p>
        <p><strong>CMK Flow was conceived by Carsten Kirschner and developed with the support of modern AI-assisted software engineering tools.</strong></p>
        <p>A significant portion of the source code was generated, analyzed, and refined with AI assistance. However, CMK Flow was <strong>not</strong> created from a single prompt. It is the result of countless architectural decisions, testing, debugging, refactoring, and many hours of focused development.</p>
        <p>AI served as a powerful implementation tool throughout the project. The product vision, system architecture, user experience, and overall design philosophy were defined through an ongoing human-led development process.</p>
        <p><strong>CMK Flow represents the combination of human creativity, thoughtful product design, and AI-assisted software engineering. It demonstrates how human expertise and modern AI can work together to create software that neither could have produced alone.</strong></p>
        <div class="cmk-about-acknowledgements">
          <h4>Inspiration &amp; acknowledgement</h4>
          <p><a href="https://github.com/willmiao/ComfyUI-Lora-Manager" target="_blank" rel="noopener noreferrer">ComfyUI LoRA Manager by willmiao</a> strongly influenced the expectations behind the CMK Flow Browser. Its unexpectedly seamless transition from ComfyUI into a comfortable browser interface for checkpoints, LoRAs, metadata, and Civitai content demonstrated how capable an integrated management experience could be.</p>
          <p>CMK Flow is an independent project and has no official affiliation with LoRA Manager. This acknowledgement recognizes the design inspiration and the substantial work behind that open-source project.</p>
        </div>
        <div class="cmk-about-responsibility">
          <h4>Responsibility</h4>
          <p>CMK Flow does not seek to dictate what consenting adults create in private. At the same time, CMK does not tolerate the publication of manipulated depictions of real people without their explicit consent, especially in a sexualized or degrading context.</p>
          <p>Technical safeguards can support responsible conduct, but they cannot replace it. Everyone using CMK Flow remains responsible for treating generated content respectfully, consensually, and lawfully.</p>
          <p>There is a certain irony in the fact that the entire project began with the wish for a native FaceSwap engine.</p>
        </div>
        <div class="cmk-about-support">
          <p><strong>CMK Flow is, and will remain, completely free to use.</strong></p>
          <p>If CMK Flow has made your work easier, saved you time, or simply brought you joy, you can support its continued development with a voluntary contribution.</p>
          <p><strong>Thank you for using CMK Flow.</strong></p>
        </div>
        <div class="cmk-about-legal">
          <p>Copyright © 2026 Carsten Kirschner</p>
          <p>Licensed under GNU GPL v3 or later (GPL-3.0-or-later).</p>
        </div>
        <div class="cmk-about-donation">
          <p>Support is entirely voluntary and does not affect the features available in CMK Flow.</p>
          <div class="cmk-about-donation-actions">
            <a class="cmk-about-donation-link" href="https://github.com/CMKFlow/cmk_nodes" target="_blank" rel="noopener noreferrer">Source code &amp; documentation</a>
            <a class="cmk-about-donation-link" href="https://paypal.me/CMKFlow" target="_blank" rel="noopener noreferrer">Support voluntarily via PayPal</a>
          </div>
        </div>`;
  overlay.innerHTML = `
    <section class="cmk-about-dialog" role="dialog" aria-modal="true" aria-labelledby="cmk-about-title">
      <header class="cmk-about-header">
        <div class="cmk-about-title">
          <img class="cmk-about-logo" src="/extensions/cmk_nodes/assets/brand/cmk-logo.png" alt="" width="36" height="36">
          <h3 id="cmk-about-title">About CMK Flow</h3>
        </div>
        <button class="cmk-flow-close" type="button" aria-label="Close">×</button>
      </header>
      <div class="cmk-about-body">${aboutBody}</div>
    </section>`;
  root.append(overlay);
  const close = () => overlay.remove();
  overlay.querySelector(".cmk-flow-close").addEventListener("click", close);
  overlay.addEventListener("click", (event) => { if (event.target === overlay) close(); });
  overlay.querySelector(".cmk-flow-close").focus();
}

function formatStorageSize(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = value / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && size >= 1024; index += 1) {
    size /= 1024;
    unit = units[index];
  }
  return `${size.toLocaleString(language, { maximumFractionDigits: 1 })} ${unit}`;
}

function escapeStorageText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function confirmStorageAction(root, title, message, confirmLabel) {
  return new Promise((resolve) => {
    const confirmation = document.createElement("div");
    confirmation.className = "cmk-about-overlay";
    confirmation.innerHTML = `
      <section class="cmk-about-dialog cmk-confirm-dialog" role="alertdialog" aria-modal="true">
        <header class="cmk-about-header"><h3></h3></header>
        <div class="cmk-about-body">
          <p></p>
          <div class="cmk-confirm-actions">
            <button class="cmk-confirm-cancel" type="button">${t("cancel")}</button>
            <button class="cmk-confirm-delete" type="button"></button>
          </div>
        </div>
      </section>`;
    confirmation.querySelector("h3").textContent = title;
    confirmation.querySelector("p").textContent = message;
    confirmation.querySelector(".cmk-confirm-delete").textContent = confirmLabel;
    root.append(confirmation);
    const finish = (result) => { confirmation.remove(); resolve(result); };
    confirmation.querySelector(".cmk-confirm-cancel").addEventListener("click", () => finish(false));
    confirmation.querySelector(".cmk-confirm-delete").addEventListener("click", () => finish(true));
    confirmation.addEventListener("click", (event) => { if (event.target === confirmation) finish(false); });
    confirmation.querySelector(".cmk-confirm-cancel").focus();
  });
}

async function openVideoStorageDialog(root) {
  root.querySelector(".cmk-about-overlay")?.remove();
  const overlay = document.createElement("div");
  overlay.className = "cmk-about-overlay";
  overlay.innerHTML = `
    <section class="cmk-about-dialog" role="dialog" aria-modal="true" aria-labelledby="cmk-storage-title">
      <header class="cmk-about-header">
        <h3 id="cmk-storage-title">${t("videoStorage")}</h3>
        <button class="cmk-flow-close" type="button" aria-label="Close">×</button>
      </header>
      <div class="cmk-about-body">
        <p>${t("videoStorageIntro")}</p>
        <div class="cmk-storage-summary"><div class="cmk-storage-card">…</div><div class="cmk-storage-card">…</div></div>
        <div class="cmk-storage-projects"><h4>${t("videoProjects")}</h4><div class="cmk-storage-empty">…</div></div>
        <div class="cmk-storage-paths">${t("technicalLocations")}:<br><code>output/video/segments</code><br><code>output/video/merged</code></div>
        <div class="cmk-storage-actions">
          <span class="cmk-storage-status"></span>
          <button class="cmk-storage-clear" type="button">${t("deleteAllVideoFiles")}</button>
        </div>
      </div>
    </section>`;
  root.append(overlay);
  const close = () => overlay.remove();
  overlay.querySelector(".cmk-flow-close").addEventListener("click", close);
  overlay.addEventListener("click", (event) => { if (event.target === overlay) close(); });

  const status = overlay.querySelector(".cmk-storage-status");
  const clearButton = overlay.querySelector(".cmk-storage-clear");
  const renderStats = (locations = {}) => {
    const labels = { segments: t("segments"), merged: t("mergedVideos"), files: t("files") };
    overlay.querySelector(".cmk-storage-summary").innerHTML = ["segments", "merged"].map((name) => {
      const item = locations[name] || {};
      return `<div class="cmk-storage-card"><strong>${labels[name]}</strong><span>${Number(item.files) || 0} ${labels.files} · ${formatStorageSize(item.bytes)}</span></div>`;
    }).join("");
  };

  const renderProjects = (projects = []) => {
    const container = overlay.querySelector(".cmk-storage-projects");
    const heading = `<h4>${t("videoProjects")}</h4>`;
    if (!projects.length) {
      container.innerHTML = `${heading}<div class="cmk-storage-empty">${t("noVideoProjects")}</div>`;
      return;
    }
    const dateFormatter = new Intl.DateTimeFormat(language === "de" ? "de-DE" : "en", { dateStyle: "medium", timeStyle: "short" });
    container.innerHTML = heading + projects.map((project) => {
      const name = escapeStorageText(project.display_name);
      const modified = project.last_modified ? dateFormatter.format(new Date(project.last_modified)) : "—";
      return `<article class="cmk-storage-project">
        <div><span class="cmk-storage-project-name" title="${name}">${name}</span>
          <div class="cmk-storage-project-meta">
            <span>${Number(project.segment_files) || 0} ${t("segments")} · ${formatStorageSize(project.segment_bytes)}</span>
            <span>${Number(project.merged_files) || 0} ${t("mergedWorkingVideos")} · ${formatStorageSize(project.merged_bytes)}</span>
            <span>${t("total")}: ${formatStorageSize(project.total_bytes)}</span>
            <span>${t("lastUsed")}: ${modified}</span>
          </div>
        </div>
        <div class="cmk-storage-project-actions" data-project-id="${escapeStorageText(project.id)}">
          <button class="cmk-storage-project-action cmk-storage-project-flow" type="button">${t("openInFlow")}</button>
          <button class="cmk-storage-project-action cmk-storage-project-folder" type="button">${t("openFolder")}</button>
          <button class="cmk-storage-project-action cmk-storage-project-delete" type="button">${t("deleteProjectFiles")}</button>
        </div>
      </article>`;
    }).join("");
  };

  const loadStorage = async () => {
    const response = await fetchJson("/cmk/video-storage");
    renderStats(response.locations);
    renderProjects(response.projects);
  };

  overlay.querySelector(".cmk-storage-projects").addEventListener("click", async (event) => {
    const button = event.target.closest(".cmk-storage-project-action");
    if (!button) return;
    const projectCard = button.closest(".cmk-storage-project");
    const projectId = button.closest(".cmk-storage-project-actions").dataset.projectId;
    const name = projectCard.querySelector(".cmk-storage-project-name").textContent;

    if (button.classList.contains("cmk-storage-project-flow")) {
      button.disabled = true;
      status.textContent = t("openingFlow");
      try {
        const response = await fetchJson(`/cmk/video-storage/project/workflow?id=${encodeURIComponent(projectId)}`);
        if (typeof app.loadGraphData !== "function") throw new Error("loadGraphData unavailable");
        await app.loadGraphData(response.workflow, true, false, `CMK Video · ${name}`, { openSource: "cmk-video-project" });
        root.remove();
        focusLoadedWorkflowAfterRender();
      } catch (error) {
        console.error("CMK video project workflow could not be opened", error);
        status.textContent = `${t("openFailed")}: ${error.message}`;
        button.disabled = false;
      }
      return;
    }

    if (button.classList.contains("cmk-storage-project-folder")) {
      button.disabled = true;
      try {
        const response = await api.fetchApi("/cmk/video-storage/project/open-folder", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: projectId }),
        });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        status.textContent = t("folderOpened");
      } catch (error) {
        console.error("CMK video project folder could not be opened", error);
        status.textContent = `${t("openFailed")}: ${error.message}`;
      } finally {
        button.disabled = false;
      }
      return;
    }

    if (!await confirmStorageAction(root, t("deleteProjectTitle"), t("deleteProjectText")(name), t("deleteProject"))) return;
    button.disabled = true;
    status.textContent = t("deleting");
    try {
      const response = await api.fetchApi("/cmk/video-storage/project/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: projectId }),
      });
      const result = await response.json();
      if (!response.ok && response.status !== 207) throw new Error(`${response.status} ${response.statusText}`);
      if (result.failures?.length) {
        console.warn("CMK video project cleanup was incomplete", result.failures);
        status.textContent = t("partialDelete");
      } else {
        status.textContent = t("projectDeleted")(formatStorageSize(result.freed_bytes));
      }
      await loadStorage();
    } catch (error) {
      console.error("CMK video project cleanup failed", error);
      status.textContent = `${t("deleteFailed")}: ${error.message}`;
    } finally {
      button.disabled = false;
    }
  });

  try {
    await loadStorage();
  } catch (error) {
    status.textContent = `${t("storageUnavailable")}: ${error.message}`;
    clearButton.disabled = true;
  }

  clearButton.addEventListener("click", async () => {
    if (!await confirmStorageAction(root, t("deleteAllTitle"), t("deleteAllText"), t("deleteAll"))) return;
    clearButton.disabled = true;
    status.textContent = t("deleting");
    try {
      const response = await api.fetchApi("/cmk/video-storage/clear", { method: "POST" });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      await loadStorage();
      status.textContent = t("allDeleted");
    } catch (error) {
      console.error("CMK video storage cleanup failed", error);
      status.textContent = `${t("deleteFailed")}: ${error.message}`;
    } finally {
      clearButton.disabled = false;
    }
  });
  overlay.querySelector(".cmk-flow-close").focus();
}

async function openFlowBrowser() {
  document.querySelector(".cmk-flow-overlay")?.remove();
  addStyles();

  const root = document.createElement("div");
  root.className = "cmk-flow-overlay";
  root.innerHTML = `
    <section class="cmk-flow-browser" role="dialog" aria-modal="true" aria-labelledby="cmk-flow-title">
      <header class="cmk-flow-header">
        <div class="cmk-flow-heading"><h2 id="cmk-flow-title">CMK Flow</h2><span class="cmk-flow-subtitle">${t("subtitleFlow")}</span></div>
        <div class="cmk-flow-header-actions"><div class="cmk-language-switch" aria-label="Language"><button class="cmk-language-button" data-language="de">DE</button><button class="cmk-language-button" data-language="en">EN</button></div><button class="cmk-storage-open" type="button">${t("videoStorage")}</button><button class="cmk-about-open" type="button">About CMK Flow</button><button class="cmk-flow-close" type="button" aria-label="Schließen">×</button></div>
      </header>
      <nav class="cmk-browser-tabs" role="tablist" aria-label="Bereich">
        <button class="cmk-browser-tab" type="button" role="tab" data-cmk-tab="flow">Flow</button>
        <button class="cmk-browser-tab" type="button" role="tab" data-cmk-tab="toolbox">${t("toolbox")}</button>
        <button class="cmk-browser-tab" type="button" role="tab" data-cmk-tab="references">${language === "de" ? "Referenzen" : "References"}</button>
      </nav>
      <div class="cmk-flow-content">
        <aside class="cmk-flow-sidebar">
          <div class="cmk-flow-controls"><input type="search" placeholder="${t("searchFlows")}" aria-label="${t("searchFlows")}"><select aria-label="${t("category")}"></select></div>
          <div class="cmk-flow-list"><div class="cmk-flow-empty">${t("loading")}</div></div>
        </aside>
        <div class="cmk-flow-detail"></div>
      </div>
    </section>`;
  document.body.append(root);

  const close = () => root.remove();
  root.querySelector(".cmk-flow-header > .cmk-flow-header-actions .cmk-flow-close").addEventListener("click", close);
  root.querySelector(".cmk-storage-open").addEventListener("click", () => openVideoStorageDialog(root));
  root.querySelector(".cmk-about-open").addEventListener("click", () => openAboutDialog(root));
  root.querySelectorAll(".cmk-language-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.language === language);
    button.addEventListener("click", async () => {
      if (button.dataset.language === language) return;
      language = button.dataset.language;
      localStorage.setItem("cmk-flow-language", language);
      browserDataPromise = undefined;
      root.remove();
      await openFlowBrowser();
    });
  });
  root.addEventListener("click", (event) => { if (event.target === root) close(); });
  root.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const about = root.querySelector(".cmk-about-overlay");
    if (about) about.remove();
    else close();
  });

  try {
    renderBrowser(root, await loadBrowserData());
  } catch (error) {
    root.querySelector(".cmk-flow-list").innerHTML = `<div class="cmk-flow-error">${t("loadError")}</div>`;
    root.querySelector(".cmk-flow-detail").textContent = error.message;
    console.error(`[${EXTENSION_NAME}]`, error);
  }
}

app.registerExtension({
  name: EXTENSION_NAME,
  commands: [{ id: COMMAND_ID, label: "CMK Flow öffnen", menubarLabel: "Flow Browser öffnen", function: openFlowBrowser }],
  menuCommands: [{ path: ["CMK Flow"], commands: [COMMAND_ID] }],
  async setup() {
    if (!app.menu?.settingsGroup || document.querySelector("[data-cmk-flow-launcher]")) return;

    addStyles();
    const ComfyButton = window.comfyAPI?.button?.ComfyButton;
    if (!ComfyButton) {
      console.warn(`[${EXTENSION_NAME}] ComfyUI button API is unavailable; use the CMK Flow menu command instead.`);
      return;
    }
    const launcher = new ComfyButton({
      action: openFlowBrowser,
      tooltip: "Kuratierte CMK Flow-Module öffnen",
      content: "CMK Flow",
    });
    launcher.element.dataset.cmkFlowLauncher = "true";
    app.menu.settingsGroup.append(launcher);
  },
});
