import { app } from "../../../scripts/app.js";
import {
    applyStartPromptHeights,
    configureStartBase,
    getWidget,
    prepareStartMode,
    setWidgetVisible,
    startLayoutMode,
    startVisibleNames,
} from "./cmk_flow_start_guidance_v51.js";
import {
    ENGINE_SDXL,
    ENGINE_ZIT,
    FINISH_NONE,
    FINISH_REFINER,
    FINISH_ZIT,
    STRATEGY_IDENTITY,
    STRATEGY_STANDARD,
    allowedFinishes,
    modelFamilyFor,
    normalizeSelection,
    stopSupported,
} from "./cmk_flow_start_v2_core.js";

const NODE_CLASS = "CMKPipeCreateImage";
const UI_VERSION = 2;
const TARGET_SIZE = [600, 1225];
const MINIMUM_SIZE = [450, 700];
const CONTROL_HEIGHT = 36;
const PANEL_HEIGHT = 60;
const FINISH_PANEL_HEIGHT = 80;
const GENERATION_MODE_HEIGHT = PANEL_HEIGHT * 2 + FINISH_PANEL_HEIGHT;
const GENERATION_MODE_PANEL = "GENERATION MODE";
const ENGINE_PANEL = "GENERATION ENGINE";
const STRATEGY_PANEL = "GENERATION STRATEGY";
const FINISH_PANEL = "FINISH";
const LEGACY_FAMILY_PANEL = "MODEL FAMILY TABS";
// Kept for serialization and synchronised from the v2 state, never shown.
const V2_HIDDEN_WIDGETS = new Set(["model_family", "instantid_enabled", LEGACY_FAMILY_PANEL]);
const STYLE_ID = "cmk-start-v2-style";

function isV2(node) {
    const target = node && (
        node.comfyClass === NODE_CLASS
        || node.type === NODE_CLASS
        || node.constructor?.comfyClass === NODE_CLASS
        || node.constructor?.nodeData?.name === NODE_CLASS
    );
    return Boolean(target && Number(node.properties?.cmkStartUiVersion) === UI_VERSION);
}

function isInpaint(node) {
    const value = getWidget(node, "INPAINT_MODE")?.value;
    return typeof value === "boolean"
        ? value
        : String(value ?? "Text2Image").trim().toLowerCase() === "inpaint";
}

function contextKey(engine, strategy) {
    return `${engine === ENGINE_ZIT ? "zit" : "sdxl"}:${strategy === STRATEGY_IDENTITY ? "identity" : "standard"}`;
}

function resolutionToken(value, fallback) {
    return String(value ?? fallback).trim().split(/\s+/).at(-1);
}

function currentSelection(node) {
    const family = String(getWidget(node, "model_family")?.value ?? "SDXL").trim().toLowerCase();
    const identity = Boolean(getWidget(node, "instantid_enabled")?.value);
    if (family.includes("z-image")) {
        return { engine: ENGINE_ZIT, strategy: STRATEGY_STANDARD, finish: FINISH_NONE };
    }
    return {
        engine: ENGINE_SDXL,
        strategy: identity ? STRATEGY_IDENTITY : STRATEGY_STANDARD,
        finish: family === "hybrid" ? FINISH_ZIT : FINISH_REFINER,
    };
}

function persistentState(node) {
    node.properties ??= {};
    const inferred = currentSelection(node);
    const stored = node.properties.cmkStartV2State;
    const state = stored && typeof stored === "object" ? stored : {};
    state.engine ??= inferred.engine;
    state.strategy ??= inferred.strategy;
    state.sdxlStrategy ??= inferred.strategy;
    state.finishByContext ??= {};
    state.finishByContext[contextKey(inferred.engine, inferred.strategy)] ??= inferred.finish;
    state.resolutionByEngine ??= {};
    state.customInpaint ??= {};
    state.generationModeExpanded ??= false;
    node.properties.cmkStartV2State = state;
    return state;
}

function installStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.cmk-start-v2-mode { width:100%; height:${CONTROL_HEIGHT}px !important; min-height:${CONTROL_HEIGHT}px !important; max-height:${CONTROL_HEIGHT}px !important; box-sizing:border-box; display:flex; flex:0 0 ${CONTROL_HEIGHT}px; align-items:center; gap:7px; padding:0 10px; border:1px solid rgba(255,255,255,.08); border-radius:5px; background:rgba(255,255,255,.025); color:var(--input-text,#ddd); font-family:inherit; font-size:12px; font-weight:600; line-height:1; cursor:pointer; }
.cmk-start-v2-mode-chevron { width:10px; flex:0 0 10px; text-align:center; }
.cmk-start-v2-mode-label { white-space:nowrap; }
.cmk-start-v2-mode-summary { min-width:0; margin-left:auto; overflow:hidden; color:var(--input-text,#ddd); font-weight:500; text-overflow:ellipsis; white-space:nowrap; }
.cmk-start-v2-section { display:grid; gap:6px; width:100%; box-sizing:border-box; }
.cmk-start-v2-section-heading { height:16px; color:var(--input-text,#ddd); font-family:inherit; font-size:11px; font-weight:600; line-height:16px; white-space:nowrap; }
.cmk-start-v2-control { width:100%; box-sizing:border-box; padding-left:12px; }
.cmk-start-v2-tabs { display:grid; grid-template-columns:1fr 1fr; gap:7px; }
.cmk-start-v2-tabs button { height:${CONTROL_HEIGHT}px; border:1px solid rgba(255,255,255,.08); border-radius:5px; background:rgba(255,255,255,.025); color:var(--input-text,#ddd); font-family:inherit; font-size:12px; font-weight:600; cursor:pointer; }
.cmk-start-v2-tabs button.is-active { border-color:rgba(255,255,255,.08); background:rgba(255,255,255,.025); color:#62d9d6; }
.cmk-start-v2-tabs button:disabled { cursor:not-allowed; opacity:.35; }
.cmk-start-v2-finish-wrap { position:relative; display:grid; gap:5px; }
.cmk-start-v2-finish-wrap::after { position:absolute; top:0; right:18px; height:${CONTROL_HEIGHT}px; display:flex; align-items:center; color:var(--input-text,#ddd); content:"▾"; pointer-events:none; }
.cmk-start-v2-finish { width:100%; height:${CONTROL_HEIGHT}px; box-sizing:border-box; appearance:none; padding:0 30px 0 10px; border:1px solid rgba(255,255,255,.08); border-radius:5px; background:rgba(255,255,255,.025); color:var(--input-text,#ddd); font-family:inherit; font-size:12px; font-weight:600; line-height:${CONTROL_HEIGHT}px; }
.cmk-start-v2-finish:disabled { opacity:.45; cursor:not-allowed; }
.cmk-start-v2-finish-description { overflow:hidden; color:color-mix(in srgb, var(--input-text,#ddd) 62%, transparent); font-family:inherit; font-size:10px; font-weight:400; line-height:15px; text-overflow:ellipsis; white-space:nowrap; }
`;
    document.head.append(style);
}

// Fixed-height DOM controls take no part in the constrained layout: without
// computeLayoutSize the frontend renders their grid row as min-content, so only
// the prompt fields grow with the node.
function fixedDomWidget(node, name, type, element, height, minimumWidth = 410) {
    const widget = node.addDOMWidget(name, type, element, {
        hideOnZoom: false,
        getMinHeight: () => height,
        getHeight: () => height,
    });
    widget.serialize = false;
    widget.computeSize = (width) => [Math.max(Number(width) || minimumWidth, minimumWidth), height];
    widget.computeLayoutSize = undefined;
    return widget;
}

function tabPanel(node, name, choices, onSelect) {
    const root = document.createElement("div");
    root.className = "cmk-start-v2-section";
    const heading = document.createElement("div");
    heading.className = "cmk-start-v2-section-heading";
    heading.textContent = name;
    const controls = document.createElement("div");
    controls.className = "cmk-start-v2-control cmk-start-v2-tabs";
    root.append(heading, controls);
    const buttons = choices.map(({ value, label }) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.style.height = `${CONTROL_HEIGHT}px`;
        button.style.fontSize = "12px";
        button.addEventListener("click", () => onSelect(value));
        controls.append(button);
        return { button, value };
    });
    const widget = fixedDomWidget(node, name, `cmk_start_v2_${name.toLowerCase().replaceAll(" ", "_")}`, root, PANEL_HEIGHT);
    return { widget, root, heading, controls, buttons };
}

function pathSummary(selection) {
    if (selection.engine === ENGINE_ZIT) return "ZIT";
    const stages = ["SDXL"];
    if (selection.strategy === STRATEGY_IDENTITY) stages.push("InstantID");
    stages.push(selection.finish === FINISH_ZIT ? "ZIT" : "Refiner");
    return stages.join(" → ");
}

function finishDescription(selection) {
    if (selection.engine === ENGINE_ZIT) return "Direct Z-Image Turbo generation";
    const source = selection.strategy === STRATEGY_IDENTITY
        ? "SDXL InstantID generation"
        : "SDXL generation";
    const finish = selection.finish === FINISH_ZIT
        ? "Z-Image Turbo finishing pass"
        : "SDXL Refiner finishing pass";
    return `${source} → ${finish}`;
}

function setNodeHeight(node, height) {
    const width = Number(node?.size?.[0]) || TARGET_SIZE[0];
    node.setSize?.([width, Math.max(MINIMUM_SIZE[1], Math.round(height))]);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function installHeightHooks(node, state) {
    if (node._cmkStartV2HeightState) return;
    const applied = state.generationModeExpanded ? GENERATION_MODE_HEIGHT : 0;
    node._cmkStartV2HeightState = { applied, suspended: false };
    node._cmkBeforeCleanViewChange = () => {
        const heightState = node._cmkStartV2HeightState;
        if (!heightState || heightState.suspended || heightState.applied <= 0) return;
        setNodeHeight(node, Number(node.size?.[1]) - heightState.applied);
        heightState.suspended = true;
    };
    node._cmkAfterCleanViewChange = () => {
        const heightState = node._cmkStartV2HeightState;
        if (!heightState?.suspended) return;
        setNodeHeight(node, Number(node.size?.[1]) + heightState.applied);
        heightState.suspended = false;
    };
    node._cmkBaseSizeForTemporaryUi = (size) => {
        const heightState = node._cmkStartV2HeightState;
        const offset = heightState?.suspended ? 0 : Number(heightState?.applied) || 0;
        return [Number(size[0]), Math.max(MINIMUM_SIZE[1], Number(size[1]) - offset)];
    };
}

function setGenerationHeight(node, expanded) {
    const heightState = node._cmkStartV2HeightState;
    if (!heightState) return;
    const next = expanded ? GENERATION_MODE_HEIGHT : 0;
    if (heightState.applied === next) return;
    const currentHeight = Number(node.size?.[1]) || TARGET_SIZE[1];
    setNodeHeight(node, currentHeight - heightState.applied + next);
    heightState.applied = next;
}

function renderModeHeader(node, state, selection) {
    const mode = node._cmkStartV2GenerationMode;
    if (!mode) return;
    mode.chevron.textContent = state.generationModeExpanded ? "▾" : "▸";
    mode.summary.textContent = pathSummary(selection);
    mode.button.setAttribute("aria-expanded", String(state.generationModeExpanded));
}

function toggleGenerationMode(node, state) {
    state.generationModeExpanded = !state.generationModeExpanded;
    layoutV2(node, state);
    setGenerationHeight(node, state.generationModeExpanded);
    renderModeHeader(node, state, normalizeSelection(
        state.engine,
        state.strategy,
        state.finishByContext[contextKey(state.engine, state.strategy)],
    ));
    node.properties.cmkStartV2State = state;
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function installModeHeader(node, state) {
    if (node._cmkStartV2GenerationMode) return node._cmkStartV2GenerationMode;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "cmk-start-v2-mode";
    for (const property of ["height", "min-height", "max-height"]) {
        button.style.setProperty(property, `${CONTROL_HEIGHT}px`, "important");
    }
    const chevron = document.createElement("span");
    chevron.className = "cmk-start-v2-mode-chevron";
    const label = document.createElement("span");
    label.className = "cmk-start-v2-mode-label";
    label.textContent = "GENERATION MODE";
    const summary = document.createElement("span");
    summary.className = "cmk-start-v2-mode-summary";
    button.append(chevron, label, summary);
    button.addEventListener("click", () => toggleGenerationMode(node, state));
    const widget = fixedDomWidget(
        node,
        GENERATION_MODE_PANEL,
        "cmk_start_v2_generation_mode",
        button,
        CONTROL_HEIGHT,
    );
    node._cmkStartV2GenerationMode = { widget, button, chevron, summary };
    return node._cmkStartV2GenerationMode;
}

function rememberResolution(node, state) {
    const resolution = getWidget(node, "resolution");
    if (!resolution) return;
    state.resolutionByEngine[state.engine] = resolutionToken(
        resolution.value,
        state.engine === ENGINE_ZIT ? "1024x1024" : "1152x832",
    );
}

function restoreResolution(node, state) {
    const resolution = getWidget(node, "resolution");
    if (!resolution) return;
    const fallback = state.engine === ENGINE_ZIT ? "1024x1024" : "1152x832";
    const remembered = state.resolutionByEngine[state.engine];
    const values = resolution.options?.values;
    resolution.value = Array.isArray(values) && values.includes(remembered) ? remembered : fallback;
    state.resolutionByEngine[state.engine] = resolution.value;
}

function setDisabled(widget, disabled) {
    if (!widget) return;
    widget.disabled = Boolean(disabled);
    widget.options ??= {};
    widget.options.disabled = Boolean(disabled);
}

function installPanels(node, state) {
    if (node._cmkStartV2Panels) return node._cmkStartV2Panels;
    const engine = tabPanel(node, ENGINE_PANEL, [
        { value: ENGINE_SDXL, label: "SDXL" },
        { value: ENGINE_ZIT, label: "Z-IMAGE TURBO" },
    ], (value) => {
        rememberResolution(node, state);
        if (state.engine === ENGINE_SDXL) state.sdxlStrategy = state.strategy;
        state.engine = value;
        state.strategy = value === ENGINE_ZIT ? STRATEGY_STANDARD : (state.sdxlStrategy || STRATEGY_STANDARD);
        refresh(node, true);
    });
    const strategy = tabPanel(node, STRATEGY_PANEL, [
        { value: STRATEGY_STANDARD, label: "STANDARD" },
        { value: STRATEGY_IDENTITY, label: "IDENTITY" },
    ], (value) => {
        if (state.engine === ENGINE_ZIT) return;
        state.strategy = value;
        state.sdxlStrategy = value;
        refresh(node, true);
    });
    const finishSection = document.createElement("div");
    finishSection.className = "cmk-start-v2-section";
    const finishHeading = document.createElement("div");
    finishHeading.className = "cmk-start-v2-section-heading";
    finishHeading.textContent = FINISH_PANEL;
    const finishRoot = document.createElement("div");
    finishRoot.className = "cmk-start-v2-control cmk-start-v2-finish-wrap";
    const select = document.createElement("select");
    select.className = "cmk-start-v2-finish";
    select.style.height = `${CONTROL_HEIGHT}px`;
    select.style.fontSize = "12px";
    const description = document.createElement("div");
    description.className = "cmk-start-v2-finish-description";
    finishRoot.append(select, description);
    finishSection.append(finishHeading, finishRoot);
    select.addEventListener("change", () => {
        state.finishByContext[contextKey(state.engine, state.strategy)] = select.value;
        refresh(node, true);
    });
    const finishWidget = fixedDomWidget(node, FINISH_PANEL, "cmk_start_v2_finish", finishSection, FINISH_PANEL_HEIGHT);
    node._cmkStartV2Panels = {
        engine,
        strategy,
        finish: {
            widget: finishWidget,
            root: finishSection,
            heading: finishHeading,
            controls: finishRoot,
            select,
            description,
        },
    };
    return node._cmkStartV2Panels;
}

function renderPanels(node, state, selection, stopAllowed) {
    const panels = node._cmkStartV2Panels;
    if (!panels) return;
    for (const item of panels.engine.buttons) {
        item.button.classList.toggle("is-active", item.value === selection.engine);
    }
    for (const item of panels.strategy.buttons) {
        item.button.classList.toggle("is-active", item.value === selection.strategy);
        item.button.disabled = selection.engine === ENGINE_ZIT && item.value === STRATEGY_IDENTITY;
    }
    const options = allowedFinishes(selection.engine, selection.strategy);
    panels.finish.select.replaceChildren(...options.map((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        return option;
    }));
    panels.finish.select.value = selection.finish;
    panels.finish.description.textContent = finishDescription(selection);
    const stop = getWidget(node, "stop_after_handoff");
    panels.finish.select.disabled = Boolean(stopAllowed && stop?.value);
    renderModeHeader(node, state, selection);
}

// The v2 layout applies the shared visibility rules of the start node to the
// v2 controls; the legacy family tabs and family/identity switches never take
// part.  The widget list is rebuilt only when mode, visibility or order really
// change, and prompt geometry is set only on a mode change, so value-only
// changes (STOP, IMAGE FIT) leave the layout untouched.
function layoutV2(node, state) {
    const panels = node._cmkStartV2Panels;
    const header = node._cmkStartV2GenerationMode?.widget;
    if (!panels || !header) return;
    const ui = node._cmkStartUi;
    const mode = startLayoutMode(node);
    const modeChanged = ui.visibleMode !== mode;
    if (modeChanged) prepareStartMode(node, mode);
    const controls = [header, panels.engine.widget, panels.strategy.widget, panels.finish.widget];
    const stop = getWidget(node, "stop_after_handoff");
    const own = new Set([...controls, stop]);
    const content = startVisibleNames(node, mode)
        .filter((name) => !V2_HIDDEN_WIDGETS.has(name))
        .map((name) => getWidget(node, name))
        .filter((widget) => widget && !own.has(widget));
    const placed = [...controls, ...(stop ? [stop] : []), ...content];
    const placedSet = new Set(placed);
    const order = [...placed, ...(node.widgets ?? []).filter((widget) => !placedSet.has(widget))];
    const shown = new Set([header, ...(stop ? [stop] : []), ...content]);
    if (state.generationModeExpanded) controls.slice(1).forEach((widget) => shown.add(widget));
    const unchanged = !modeChanged
        && order.length === node.widgets?.length
        && order.every((widget, index) => node.widgets[index] === widget)
        && order.every((widget) => (widget.hidden === true) !== shown.has(widget));
    if (unchanged) return;
    for (const widget of order) setWidgetVisible(widget, shown.has(widget));
    if (modeChanged) applyStartPromptHeights(node, mode, shown);
    node.widgets = order;
    node._cmkStartVisibleWidgets = shown;
    ui.visibleMode = mode;
    node.graph?.trigger?.("node:widget:changed", { nodeId: node.id });
}

function syncCustomInpaintState(node, state) {
    for (const [property, nodeKey] of [
        ["fillMaskedArea", "_cmkCustomFillMaskedArea"],
        ["outpaint", "_cmkCustomOutpaint"],
        ["resizeMode", "_cmkCustomResizeMode"],
    ]) {
        if (node[nodeKey] === undefined && state.customInpaint[property] !== undefined) {
            node[nodeKey] = state.customInpaint[property];
        }
        if (node[nodeKey] !== undefined) state.customInpaint[property] = node[nodeKey];
    }
}

function refresh(node, restoreEngineResolution = false) {
    if (!isV2(node) || node._cmkStartV2Refreshing || !node._cmkStartV2Panels) return;
    node._cmkStartV2Refreshing = true;
    try {
        const state = persistentState(node);
        const key = contextKey(state.engine, state.strategy);
        const requestedFinish = state.finishByContext[key];
        const selection = normalizeSelection(state.engine, state.strategy, requestedFinish);
        state.engine = selection.engine;
        state.strategy = selection.strategy;
        state.finishByContext[contextKey(selection.engine, selection.strategy)] = selection.finish;

        const family = getWidget(node, "model_family");
        const identity = getWidget(node, "instantid_enabled");
        if (family) family.value = modelFamilyFor(selection.engine, selection.strategy, selection.finish);
        if (identity) identity.value = selection.engine === ENGINE_SDXL && selection.strategy === STRATEGY_IDENTITY;

        layoutV2(node, state);
        if (restoreEngineResolution) restoreResolution(node, state);
        else rememberResolution(node, state);

        const instantidMode = getWidget(node, "instantid_mode")?.value;
        const stopAllowed = stopSupported({
            ...selection,
            inpaint: isInpaint(node),
            instantidMode,
        });
        const stop = getWidget(node, "stop_after_handoff");
        if (stop && !stopAllowed) stop.value = false;
        setDisabled(stop, !stopAllowed);
        renderPanels(node, state, selection, stopAllowed);
        syncCustomInpaintState(node, state);
        node.properties.cmkStartV2State = state;
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    } finally {
        node._cmkStartV2Refreshing = false;
    }
}

function wrapRefreshCallback(node, name, after) {
    const widget = getWidget(node, name);
    if (!widget || widget._cmkStartV2CallbackInstalled) return;
    const original = widget.callback;
    widget.callback = function () {
        const result = original?.apply(this, arguments);
        after?.();
        refresh(node);
        return result;
    };
    widget._cmkStartV2CallbackInstalled = true;
}

export function configureV2(node) {
    if (!isV2(node)) return;
    node._cmkStartPromptMinimumScale = 0.5;
    node._cmkStartPromptMinimumWidth = 410;
    node.min_size = [...MINIMUM_SIZE];
    node.minSize = [...MINIMUM_SIZE];
    if (!node._cmkStartV2GetMinSizeInstalled) {
        node.getMinSize = () => [...MINIMUM_SIZE];
        node._cmkStartV2GetMinSizeInstalled = true;
    }
    installStyle();
    // Every layout request of the shared controls (including legacy
    // callbacks) is answered by the v2 layout, never by the v1 rebuild.
    node._cmkStartLayout = () => refresh(node);
    configureStartBase(node, true);
    const state = persistentState(node);
    installHeightHooks(node, state);
    installModeHeader(node, state);
    installPanels(node, state);
    syncCustomInpaintState(node, state);
    wrapRefreshCallback(node, "INPAINT_MODE");
    wrapRefreshCallback(node, "instantid_mode");
    wrapRefreshCallback(node, "stop_after_handoff");
    wrapRefreshCallback(node, "resolution", () => rememberResolution(node, state));
    if (!node._cmkStartV2SerializationInstalled) {
        const original = node.onSerialize;
        node.onSerialize = function () {
            const current = persistentState(this);
            rememberResolution(this, current);
            syncCustomInpaintState(this, current);
            this.properties.cmkStartUiVersion = UI_VERSION;
            this.properties.cmkStartV2State = current;
            return original?.apply(this, arguments);
        };
        node._cmkStartV2SerializationInstalled = true;
    }
    node.title = "CMK Flow · 01 START HERE v2 · Create Image";
    refresh(node);
    if (!node._cmkStartV2InitialSizeApplied && !node.__cmkLoadedFromWorkflow) {
        node.setSize?.([...TARGET_SIZE]);
        node._cmkStartV2InitialSizeApplied = true;
    }
}

function schedule(node) {
    for (const delay of [0, 50, 200, 500, 1000]) {
        setTimeout(() => configureV2(node), delay);
    }
}

app.registerExtension({
    name: "cmk.flow.start.guidance.v2",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;
        for (const hook of ["onNodeCreated", "onConfigure", "onAdded"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                schedule(this);
                return result;
            };
        }
    },
    nodeCreated(node) {
        if (isV2(node)) schedule(node);
    },
    loadedGraphNode(node) {
        if (isV2(node)) schedule(node);
    },
});
