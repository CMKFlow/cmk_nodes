import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// CMK ControlNet Prepare deliberately avoids ComfyUI's native image_upload
// flag. That flag also renders the raw reference image inside the node. CMK
// reserves this area exclusively for the processed ControlNet result, so a
// separate upload button writes into the normal reference-image selector.

const CMK_NODE_CLASSES = new Set([
    "CMKControlNetPrepare",
    "CMKControlNetPreparePipe",
]);
const BUTTON_WIDGET_NAME = "select_reference_image";

function isCMKControlNetPrepare(node) {
    return node && (
        CMK_NODE_CLASSES.has(node.comfyClass) ||
        CMK_NODE_CLASSES.has(node.type)
    );
}

function findReferenceWidget(node) {
    return findWidget(node, "reference_image") || findWidget(node, "REFERENCE IMAGE");
}

function findWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name);
}

function moveWidgetAfter(node, widget, afterName) {
    if (!node?.widgets || !widget) return;
    const widgets = node.widgets;
    const from = widgets.indexOf(widget);
    const after = widgets.findIndex((candidate) => candidate?.name === afterName);
    if (from < 0 || after < 0) return;

    widgets.splice(from, 1);
    const insertAt = widgets.findIndex((candidate) => candidate?.name === afterName) + 1;
    widgets.splice(insertAt, 0, widget);
}

async function uploadReferenceImage(file) {
    const body = new FormData();
    body.append("image", file);
    body.append("type", "input");
    body.append("overwrite", "false");

    const response = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
    });

    if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
    }

    const data = await response.json();
    if (data?.subfolder) {
        return `${data.subfolder}/${data.name}`;
    }
    return data?.name || file.name;
}

function openReferenceFileDialog(node) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/png,image/jpeg,image/webp,image/bmp,image/tiff";
    input.style.display = "none";

    input.onchange = async () => {
        const file = input.files?.[0];
        document.body.removeChild(input);
        if (!file) return;

        const refWidget = findReferenceWidget(node);
        const buttonWidget = findWidget(node, BUTTON_WIDGET_NAME);

        try {
            if (buttonWidget) buttonWidget.value = "Wird übernommen …";
            const uploadedName = await uploadReferenceImage(file);
            if (refWidget) refWidget.value = uploadedName;
            if (buttonWidget) buttonWidget.value = "Referenzbild auswählen …";
            node.setDirtyCanvas?.(true, true);
        } catch (error) {
            console.error("[CMK] Reference image upload failed", error);
            if (buttonWidget) buttonWidget.value = "Auswahl fehlgeschlagen";
            setTimeout(() => {
                if (buttonWidget) buttonWidget.value = "Referenzbild auswählen …";
                node.setDirtyCanvas?.(true, true);
            }, 1500);
        }
    };

    document.body.appendChild(input);
    input.click();
}

function addReferencePickerButton(node) {
    if (!isCMKControlNetPrepare(node)) return;
    if (findWidget(node, BUTTON_WIDGET_NAME)) return;

    const button = node.addWidget(
        "button",
        BUTTON_WIDGET_NAME,
        "Referenzbild auswählen …",
        () => openReferenceFileDialog(node)
    );

    button.serialize = false;
    moveWidgetAfter(node, button, findReferenceWidget(node)?.name);
    node.setDirtyCanvas?.(true, true);
}

function normalizePickerValue(node, values) {
    if (!Array.isArray(values)) return values;

    const pickerIndex = node?.widgets?.findIndex((widget) => widget?.name === BUTTON_WIDGET_NAME) ?? -1;
    if (pickerIndex < 0) return values;

    const functionalCount = node.widgets.length - 1;
    if (values.length === functionalCount) {
        const normalized = [...values];
        normalized.splice(pickerIndex, 0, null);
        return normalized;
    }

    if (values.length === node.widgets.length && values.at(-1) == null) {
        const normalized = [...values];
        normalized.pop();
        normalized.splice(pickerIndex, 0, null);
        return normalized;
    }

    if (values.length === node.widgets.length) {
        const withoutPicker = values.filter((_, index) => index !== pickerIndex);
        const valid = (
            typeof withoutPicker[0] === "boolean" &&
            typeof withoutPicker[4] === "boolean" &&
            typeof withoutPicker[5] === "string" &&
            typeof withoutPicker[6] === "number" &&
            typeof withoutPicker[7] === "number" &&
            typeof withoutPicker[8] === "number" &&
            typeof withoutPicker[9] === "number" &&
            typeof withoutPicker[10] === "boolean"
        );
        if (valid) return values;
    }

    // Recover files affected by repeated positional shifts. The first four
    // values precede the picker and are stable. Locate the last recognisable
    // preprocessor and retain any numeric settings that still follow it.
    const head = values.slice(0, pickerIndex);
    const tail = values.slice(pickerIndex).filter((value) => value != null);
    let preprocessorIndex = -1;
    for (let index = 0; index < tail.length; index += 1) {
        if (typeof tail[index] === "string" && (tail[index] === "none" || tail[index].endsWith("Preprocessor"))) {
            preprocessorIndex = index;
        }
    }
    const preprocessor = preprocessorIndex >= 0 ? tail[preprocessorIndex] : "none";
    const beforePreprocessor = preprocessorIndex >= 0 ? tail.slice(0, preprocessorIndex) : tail;
    const afterPreprocessor = preprocessorIndex >= 0 ? tail.slice(preprocessorIndex + 1) : [];
    const booleans = tail.filter((value) => typeof value === "boolean");
    const numbers = afterPreprocessor.filter((value) => typeof value === "number");
    const applyMask = beforePreprocessor.find((value) => typeof value === "boolean") ?? false;
    const strength = numbers[0] ?? 0.60;
    const resolution = numbers[1] ?? 768;
    const start = numbers[2] ?? 0.00;
    const end = numbers[3] ?? 1.00;
    const invert = booleans.at(-1) ?? false;
    return [...head, null, applyMask, preprocessor, strength, resolution, start, end, invert];
}

app.registerExtension({
    name: "cmk.controlnet.prepare.reference_picker.v31",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!CMK_NODE_CLASSES.has(nodeData.name)) return;

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (Array.isArray(info?.widgets_values)) {
                info = { ...info, widgets_values: normalizePickerValue(this, info.widgets_values) };
            }
            return originalOnConfigure?.call(this, info);
        };

        const originalOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(info) {
            const result = originalOnSerialize?.call(this, info);
            const pickerIndex = this.widgets?.findIndex((widget) => widget?.name === BUTTON_WIDGET_NAME) ?? -1;
            if (pickerIndex >= 0 && Array.isArray(info?.widgets_values)) {
                const [pickerValue] = info.widgets_values.splice(pickerIndex, 1);
                info.widgets_values.push(pickerValue ?? null);
            }
            return result;
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            addReferencePickerButton(this);
            return result;
        };
    },

    nodeCreated(node) {
        addReferencePickerButton(node);
    },
});
