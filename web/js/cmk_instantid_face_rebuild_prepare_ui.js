import { app } from "../../../scripts/app.js";
import { installSourceTargetSlider } from "./cmk_source_target_slider.js";

const PREPARE_TYPE = "CMKInstantIDFaceRebuildPreparePipe";
const PASTEBACK_TYPE = "CMKInstantIDFaceRebuildPastebackPipe";
const DETAILER_TYPE = "CMKInstantIDFaceDetailerSDXL";
const NECK_TITLE = "PASTEBACK NECK";
const PREPARE_SIZE = [430, 550];
const PREPARE_WIDGETS = [
    "FACEREBUILD ENABLE",
    "ENABLE",
    "TARGET FACE",
    "HEAD AREA",
    "NECK AREA",
    "MASK FEATHER",
    "WORKING RESOLUTION",
    "DETECT MODEL",
    "DETECT SIZE",
];

function findNeckSwitch(node) {
    const pasteback = node?.graph?._nodes?.find((candidate) => candidate?.type === PASTEBACK_TYPE);
    return pasteback?.widgets?.find((widget) => widget?.name === NECK_TITLE);
}

function neckValue(node) {
    return Boolean(findNeckSwitch(node)?.value ?? false);
}

function setNeckValue(node, value) {
    const targetWidget = findNeckSwitch(node);
    if (!targetWidget) return;

    const next = Boolean(value);
    if (Boolean(targetWidget.value) === next) return;
    targetWidget.value = next;
    targetWidget.callback?.(next, targetWidget);
    app.graph?.setDirtyCanvas?.(true, true);
}

function serializedPrepareValues(node) {
    return PREPARE_WIDGETS.map((name) => node.widgets?.find((widget) => widget?.name === name)?.value);
}

function installNeckProxy(node) {
    if (!node || node.__cmkPastebackNeckWidget) return;

    const widget = node.addWidget("toggle", NECK_TITLE, false, (value) => {
        setNeckValue(node, value);
    });
    widget.options = { ...(widget.options || {}), serialize: false };
    widget.serialize = false;
    node.__cmkPastebackNeckWidget = widget;

    for (const candidate of node.widgets ?? []) {
        if (!candidate?.name) continue;
        candidate.label = candidate.name === "DETECT MODEL"
            ? "detect model"
            : candidate.name === "DETECT SIZE"
                ? "detect size"
                : candidate.name.toUpperCase();
    }
    for (const input of node.inputs ?? []) {
        if (!input?.name) continue;
        const label = input.name.toUpperCase();
        input.label = label;
        input.localized_name = label;
    }
    for (const output of node.outputs ?? []) {
        if (!output?.name) continue;
        const label = output.name === "diagnostic" ? "diagnostic" : output.name.toUpperCase();
        output.label = label;
        output.localized_name = label;
    }

    requestAnimationFrame(() => {
        const targetIndex = node.widgets?.findIndex((candidate) => candidate?.name === "TARGET FACE") ?? -1;
        const proxyIndex = node.widgets?.indexOf(widget) ?? -1;
        if (targetIndex >= 0 && proxyIndex >= 0 && proxyIndex !== targetIndex + 1) {
            node.widgets.splice(proxyIndex, 1);
            node.widgets.splice(targetIndex + 1, 0, widget);
        }
        widget.value = neckValue(node);
        node.setSize?.(PREPARE_SIZE);
        node.setDirtyCanvas?.(true, true);
    });
}

function configureDetailerLabels(node) {
    if (!node || node.type !== DETAILER_TYPE) return;
    const sampling = node.widgets?.find((widget) => widget?.name === "SAMPLING START");
    const balance = installSourceTargetSlider(node, sampling, "CMK SOURCE TARGET");
    balance?._cmkSyncFromCanonical?.();
    for (const widget of node.widgets ?? []) {
        if (widget?.name === "sampler") widget.label = "SAMPLER";
        if (widget?.name === "scheduler") widget.label = "SCHEDULER";
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "cmk.instantid.face_rebuild.prepare_ui",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === DETAILER_TYPE) {
            const originalCreated = nodeType.prototype.onNodeCreated;
            const originalConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onNodeCreated = function () {
                const result = originalCreated?.apply(this, arguments);
                for (const delay of [0, 50, 200]) setTimeout(() => configureDetailerLabels(this), delay);
                return result;
            };
            nodeType.prototype.onConfigure = function () {
                const result = originalConfigure?.apply(this, arguments);
                for (const delay of [0, 50, 200]) setTimeout(() => configureDetailerLabels(this), delay);
                return result;
            };
            return;
        }
        if (nodeData?.name !== PREPARE_TYPE && nodeData?.python_module !== PREPARE_TYPE) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onNodeCreated = function () {
            originalCreated?.apply(this, arguments);
            installNeckProxy(this);
        };
        nodeType.prototype.onSerialize = function (data) {
            originalSerialize?.apply(this, arguments);
            data.widgets_values = serializedPrepareValues(this);
        };
    },
    nodeCreated(node) {
        if (node?.type === DETAILER_TYPE) configureDetailerLabels(node);
    },
    loadedGraphNode(node) {
        if (node?.type === DETAILER_TYPE) configureDetailerLabels(node);
    },
});
