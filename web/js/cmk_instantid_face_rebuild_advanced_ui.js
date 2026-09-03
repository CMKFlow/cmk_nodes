import { app } from "../../../scripts/app.js";
import { installSourceTargetSlider } from "./cmk_source_target_slider.js";

const NODE_CLASS = "CMKInstantIDFaceRebuildAdvancedSDXL";
const STANDARD_NODE_CLASS = "CMKInstantIDFaceRebuildSDXL";
const NODE_SIZE = [600, 1400];
const STANDARD_NODE_SIZE = [580, 1080];
const ADVANCED = new Set([
    "DETECT MODEL", "DETECT SIZE", "instantid_model", "SAMPLING START",
    "controlnet_model", "provider", "cfg", "identity_noise",
]);
const TECHNICAL_LABELS = {
    "DETECT MODEL": "detect model",
    "DETECT SIZE": "detect size",
    instantid_model: "instantid model",
    controlnet_model: "controlnet model",
    provider: "provider",
    cfg: "cfg",
    identity_noise: "identity noise",
    sampler: "SAMPLER",
    scheduler: "SCHEDULER",
};

function isTarget(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS || node.type === NODE_CLASS ||
        node.constructor?.comfyClass === NODE_CLASS ||
        node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function isStandardTarget(node) {
    return Boolean(node) && (
        node.comfyClass === STANDARD_NODE_CLASS || node.type === STANDARD_NODE_CLASS ||
        node.constructor?.comfyClass === STANDARD_NODE_CLASS ||
        node.constructor?.nodeData?.name === STANDARD_NODE_CLASS
    );
}

function lineElement() {
    const element = document.createElement("div");
    element.style.width = "100%";
    element.style.height = "17px";
    element.style.boxSizing = "border-box";
    element.style.borderTop = "1px solid rgba(255,255,255,.48)";
    element.style.pointerEvents = "none";
    return element;
}

function installStableSerialization(node, canonical) {
    if (node._cmkFaceRebuildAdvancedSerialization) return;
    const original = node.onSerialize;
    node.onSerialize = function(data) {
        const result = original?.apply(this, arguments);
        if (this.serialize_widgets) {
            data.widgets_values = canonical
                .filter((widget) => widget?.serialize !== false)
                .map((widget) => widget?.value ?? null);
        }
        return result;
    };
    node._cmkFaceRebuildAdvancedSerialization = true;
}

function configureStandard(node) {
    if (!isStandardTarget(node) || !Array.isArray(node.widgets)) return;
    if (node._cmkFaceRebuildStandardUi) {
        node.setSize?.(STANDARD_NODE_SIZE);
        return;
    }
    node._cmkFaceRebuildStandardUi = true;
    const canonical = [...node.widgets];
    const byName = new Map(canonical.filter((widget) => widget?.name).map((widget) => [widget.name, widget]));
    installStableSerialization(node, canonical);

    for (const widget of canonical) {
        if (!widget?.name) continue;
        widget.advanced = ADVANCED.has(widget.name);
        widget.label = TECHNICAL_LABELS[widget.name] ?? widget.name.toUpperCase();
    }
    const separators = Array.from({ length: 3 }, (_value, index) => {
        if (typeof node.addDOMWidget !== "function") return null;
        const separator = node.addDOMWidget(
            `CMK FACE REBUILD STANDARD SEPARATOR ${index + 1}`,
            "cmk_face_rebuild_separator",
            lineElement(),
            { hideOnZoom: false, getMinHeight: () => 17, getHeight: () => 17 },
        );
        separator.serialize = false;
        return separator;
    }).filter(Boolean);

    const sampling = byName.get("SAMPLING START");
    const balance = installSourceTargetSlider(node, sampling, "CMK SOURCE TARGET");
    const names = [
        "FACEREBUILD ENABLE", "|",
        "TARGET FACE", "PASTEBACK NECK", "BALANCE", "IDENTITY STRENGTH", "POSE STRENGTH", "|",
        "HEAD AREA", "NECK AREA", "MASK FEATHER", "WORKING RESOLUTION", "|",
        "TOTAL STEPS", "sampler", "scheduler",
    ];
    let separatorIndex = 0;
    const visible = names.map((name) => {
        if (name === "|") return separators[separatorIndex++];
        if (name === "BALANCE") return balance;
        return byName.get(name);
    }).filter(Boolean);
    const included = new Set(visible);
    node.widgets = [
        ...visible,
        ...canonical.filter((widget) => !included.has(widget) && widget !== sampling),
    ];
    node.properties ??= {};
    node.properties.cmkFixedSize = [...STANDARD_NODE_SIZE];
    node.properties.cmkOuterSize = [...STANDARD_NODE_SIZE];
    node.properties.cmkManualSize = [...STANDARD_NODE_SIZE];
    node.setSize?.(STANDARD_NODE_SIZE);
    node.setDirtyCanvas?.(true, true);
}

function configure(node) {
    if (!isTarget(node) || !Array.isArray(node.widgets)) return;
    if (node._cmkFaceRebuildAdvancedUi) {
        node.setSize?.(NODE_SIZE);
        return;
    }
    node._cmkFaceRebuildAdvancedUi = true;
    const canonical = [...node.widgets];
    const byName = new Map(canonical.filter((widget) => widget?.name).map((widget) => [widget.name, widget]));
    installStableSerialization(node, canonical);

    for (const widget of canonical) {
        if (!widget?.name) continue;
        widget.advanced = ADVANCED.has(widget.name);
        widget.label = TECHNICAL_LABELS[widget.name] ?? widget.name.toUpperCase();
    }
    for (const input of node.inputs ?? []) {
        if (!input?.name) continue;
        const label = input.name === "diagnostic" ? "diagnostic" : input.name.toUpperCase();
        input.label = label;
        input.localized_name = label;
    }
    for (const output of node.outputs ?? []) {
        if (!output?.name) continue;
        const label = output.name === "diagnostic" ? "diagnostic" : output.name.toUpperCase();
        output.label = label;
        output.localized_name = label;
    }

    const separators = Array.from({ length: 5 }, (_value, index) => {
        if (typeof node.addDOMWidget !== "function") return null;
        const separator = node.addDOMWidget(
            `CMK FACE REBUILD SEPARATOR ${index + 1}`,
            "cmk_face_rebuild_separator",
            lineElement(),
            { hideOnZoom: false, getMinHeight: () => 17, getHeight: () => 17 },
        );
        separator.serialize = false;
        return separator;
    }).filter(Boolean);

    const balanceWidgets = new Map();
    const balanceCanonical = new Set();
    for (let index = 1; index <= 3; index += 1) {
        const canonical = byName.get(`SAMPLING START ${index}`);
        const display = installSourceTargetSlider(node, canonical, `CMK SOURCE TARGET ${index}`);
        if (display) {
            balanceWidgets.set(index, display);
            balanceCanonical.add(canonical);
        }
    }

    const names = [
        "FACEREBUILD ENABLE", "|",
        "FACE 1 ENABLE", "TARGET FACE 1", "PASTEBACK NECK 1", "BALANCE 1", "IDENTITY STRENGTH 1", "POSE STRENGTH 1", "|",
        "FACE 2 ENABLE", "TARGET FACE 2", "PASTEBACK NECK 2", "BALANCE 2", "IDENTITY STRENGTH 2", "POSE STRENGTH 2", "|",
        "FACE 3 ENABLE", "TARGET FACE 3", "PASTEBACK NECK 3", "BALANCE 3", "IDENTITY STRENGTH 3", "POSE STRENGTH 3", "|",
        "HEAD AREA", "NECK AREA", "MASK FEATHER", "WORKING RESOLUTION", "|",
        "TOTAL STEPS", "sampler", "scheduler",
    ];
    let separatorIndex = 0;
    const visible = names.map((name) => {
        if (name === "|") return separators[separatorIndex++];
        const balanceMatch = /^BALANCE ([1-3])$/.exec(name);
        return balanceMatch ? balanceWidgets.get(Number(balanceMatch[1])) : byName.get(name);
    }).filter(Boolean);
    const included = new Set(visible);
    node.widgets = [
        ...visible,
        ...canonical.filter((widget) => !included.has(widget) && !balanceCanonical.has(widget)),
    ];
    node.properties ??= {};
    node.properties.cmkFixedSize = [...NODE_SIZE];
    node.properties.cmkOuterSize = [...NODE_SIZE];
    node.properties.cmkManualSize = [...NODE_SIZE];
    node.setSize?.(NODE_SIZE);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function schedule(node) {
    for (const delay of [0, 50, 200]) setTimeout(() => {
        if (isStandardTarget(node)) configureStandard(node);
        else configure(node);
    }, delay);
}

app.registerExtension({
    name: "cmk.instantid.face_rebuild.advanced_ui.v1",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS && nodeData?.name !== STANDARD_NODE_CLASS) return;
        for (const hook of ["onNodeCreated", "onConfigure", "onAdded"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function() {
                const result = original?.apply(this, arguments);
                schedule(this);
                return result;
            };
        }
    },
    nodeCreated(node) { if (isTarget(node) || isStandardTarget(node)) schedule(node); },
    loadedGraphNode(node) { if (isTarget(node) || isStandardTarget(node)) schedule(node); },
});
