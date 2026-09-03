import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// CMK ControlNet Prepare deliberately avoids ComfyUI's native image_upload
// flag. That flag also renders the raw reference image inside the node. CMK
// reserves this area exclusively for the processed ControlNet result, so a
// separate upload button writes into the normal reference-image selector.

const CMK_NODE_CLASSES = new Set([
    "CMKControlNetPrepare",
    "CMKControlNetPreparePipe",
    "CMKZITControlNetPreparePipe",
]);
const CMK_COMBINED_CONTROLNET_TYPES = new Set([
    "CMKCombinedControlNetPreparePipe",
]);
const BUTTON_WIDGET_NAME = "select_reference_image";
const PACKAGED_REFERENCE = "CMK Package · controlnet_reference.png";

const PERCENT_WIDGET_LABELS = new Map([
    ["controlnet_start_percent", "start_percent"],
    ["controlnet_end_percent", "end_percent"],
    ["start_percent", "start_percent"],
    ["end_percent", "end_percent"],
]);

function isCMKControlNetPrepare(node) {
    return node && (
        CMK_NODE_CLASSES.has(node.comfyClass) ||
        CMK_NODE_CLASSES.has(node.type) ||
        CMK_COMBINED_CONTROLNET_TYPES.has(node.comfyClass) ||
        CMK_COMBINED_CONTROLNET_TYPES.has(node.type) ||
        node.title === "CMK Flow · 05 Combined ControlNet (optional)"
    );
}

function isCombinedControlNet(node) {
    return Boolean(node && (
        CMK_COMBINED_CONTROLNET_TYPES.has(node.comfyClass) ||
        CMK_COMBINED_CONTROLNET_TYPES.has(node.type) ||
        node.title === "CMK Flow · 05 Combined ControlNet (optional)"
    ));
}

function findReferenceWidget(node) {
    return findWidget(node, "reference_image") || findWidget(node, "REFERENCE IMAGE");
}

function findWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name);
}

function controlNetEnabled(node) {
    return Boolean(findWidget(node, "ENABLE")?.value ?? findWidget(node, "enable")?.value);
}

function clearNodePreview(node) {
    node.imgs = [];
    node.imageIndex = null;
    node.setDirtyCanvas?.(true, true);
}

function clarifyPercentLabels(node) {
    if (!isCMKControlNetPrepare(node)) return;
    for (const widget of node.widgets || []) {
        const label = PERCENT_WIDGET_LABELS.get(widget?.name);
        if (!label) continue;
        widget.label = label;
        widget.options = { ...(widget.options || {}), label };
    }
    node.setDirtyCanvas?.(true, true);
}

function clarifyPercentSchema(nodeData) {
    for (const [name, label] of PERCENT_WIDGET_LABELS) {
        const spec = nodeData?.input?.required?.[name];
        if (!Array.isArray(spec)) continue;
        spec[1] = { ...(spec[1] || {}), label };
    }
}

function migrateLegacyFractions(node, values, info) {
    if (!Array.isArray(values)) return values;
    if (Number(info?.properties?.cmkControlNetPercentScale) === 100) return values;

    const migrated = [...values];
    for (const widget of node.widgets || []) {
        if (!PERCENT_WIDGET_LABELS.has(widget?.name)) continue;
        const index = node.widgets.indexOf(widget);
        const value = migrated[index];
        if (typeof value === "number" && value >= 0 && value <= 1) {
            migrated[index] = value * 100;
        }
    }
    info.properties = { ...(info.properties || {}), cmkControlNetPercentScale: 100 };
    return migrated;
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

function showCombinedReferencePreview(node, filename) {
    if (!isCMKControlNetPrepare(node) || !filename) return;
    if (!controlNetEnabled(node)) {
        clearNodePreview(node);
        return;
    }

    const normalized = String(filename).replaceAll("\\\\", "/");
    if (normalized === PACKAGED_REFERENCE) {
        const preview = new Image();
        preview.onload = () => {
            node.imgs = [preview];
            node.imageIndex = 0;
            node.setDirtyCanvas?.(true, true);
        };
        preview.src = typeof api?.apiURL === "function"
            ? api.apiURL("/cmk/reference-assets/controlnet_reference.png")
            : "/cmk/reference-assets/controlnet_reference.png";
        return;
    }
    const separator = normalized.lastIndexOf("/");
    const descriptor = {
        filename: separator >= 0 ? normalized.slice(separator + 1) : normalized,
        subfolder: separator >= 0 ? normalized.slice(0, separator) : "",
        type: "input",
    };

    // Feed the selection through ComfyUI's native execution-preview handler.
    // This creates exactly the same image area as a backend {ui:{images:…}}
    // response; the processed ControlNet result replaces it after execution.
    if (typeof node.onExecuted === "function") {
        node.onExecuted({ images: [descriptor] });
        node.setDirtyCanvas?.(true, true);
        return;
    }

    const params = new URLSearchParams(descriptor);
    const preview = new Image();
    preview.onload = () => {
        node.imgs = [preview];
        node.imageIndex = 0;
        node.setDirtyCanvas?.(true, true);
    };
    preview.onerror = () => {
        // Keep the last processed ControlNet preview when the selected input
        // is temporarily unavailable (for example while an upload settles).
    };
    const path = `/view?${params.toString()}`;
    preview.src = typeof api?.apiURL === "function" ? api.apiURL(path) : path;
}

function setupCombinedReferencePreview(node) {
    if (!isCMKControlNetPrepare(node) || node.__cmkCombinedReferencePreview) return;
    const widget = findReferenceWidget(node);
    if (!widget) return;
    const enableWidget = findWidget(node, "ENABLE") || findWidget(node, "enable");

    node.__cmkCombinedReferencePreview = true;
    const originalCallback = widget.callback;
    widget.callback = function(value) {
        const result = originalCallback?.apply(this, arguments);
        showCombinedReferencePreview(node, value);
        return result;
    };
    if (enableWidget) {
        const originalEnableCallback = enableWidget.callback;
        enableWidget.callback = function(value) {
            const result = originalEnableCallback?.apply(this, arguments);
            if (Boolean(value)) showCombinedReferencePreview(node, widget.value);
            else clearNodePreview(node);
            return result;
        };
    }
    if (controlNetEnabled(node)) showCombinedReferencePreview(node, widget.value);
    else clearNodePreview(node);
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
            if (refWidget) {
                refWidget.value = uploadedName;
                refWidget.callback?.(uploadedName);
            }
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

function normalizeZITPickerValue(node, values) {
    if (!Array.isArray(values)) return values;

    const pickerIndex = node?.widgets?.findIndex(
        (widget) => widget?.name === BUTTON_WIDGET_NAME
    ) ?? -1;
    if (pickerIndex < 0) return values;

    const functionalCount = node.widgets.length - 1;
    if (values.length === functionalCount) {
        const normalized = [...values];
        // Recover the first ZIT draft, where the picker occupied one value
        // position during configure. Its visible signature is unambiguous:
        // resolution=STRENGTH, low_threshold=resolution,
        // high_threshold=low_threshold, MODEL PATCH=high_threshold.
        if (
            functionalCount === 9 &&
            typeof normalized[8] === "number" &&
            typeof normalized[6] === "number" &&
            normalized[6] > 1
        ) {
            normalized[5] = normalized[6];
            normalized[6] = normalized[7];
            normalized[7] = normalized[8];
            normalized[8] = "Z-Image-Turbo-Fun-Controlnet-Union.safetensors";
        }
        normalized.splice(pickerIndex, 0, null);
        return normalized;
    }

    if (values.length === node.widgets.length && values.at(-1) == null) {
        const normalized = [...values];
        normalized.pop();
        normalized.splice(pickerIndex, 0, null);
        return normalized;
    }

    return values;
}

function normalizeCombinedPickerValue(node, values) {
    if (!Array.isArray(values)) return values;

    const pickerIndex = node?.widgets?.findIndex(
        (widget) => widget?.name === BUTTON_WIDGET_NAME
    ) ?? -1;
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

    return values;
}

app.registerExtension({
    name: "cmk.controlnet.prepare.reference_picker.v33",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (
            !CMK_NODE_CLASSES.has(nodeData.name) &&
            !CMK_COMBINED_CONTROLNET_TYPES.has(nodeData.name)
        ) return;

        if (CMK_COMBINED_CONTROLNET_TYPES.has(nodeData.name)) {
            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                if (Array.isArray(info?.widgets_values)) {
                    info = {
                        ...info,
                        widgets_values: normalizeCombinedPickerValue(
                            this,
                            info.widgets_values
                        ),
                    };
                }
                return originalOnConfigure?.call(this, info);
            };

            const originalOnSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(info) {
                const result = originalOnSerialize?.call(this, info);
                const pickerIndex = this.widgets?.findIndex(
                    (widget) => widget?.name === BUTTON_WIDGET_NAME
                ) ?? -1;
                if (
                    pickerIndex >= 0 &&
                    Array.isArray(info?.widgets_values) &&
                    info.widgets_values.length === this.widgets.length
                ) {
                    info.widgets_values.splice(pickerIndex, 1);
                }
                return result;
            };
            return;
        }

        // The SDXL node has a long-lived positional migration contract. ZIT
        // shares the reference picker and preview UX, but has its own smaller
        // canonical widget list and must never receive SDXL value migrations.
        if (nodeData.name === "CMKZITControlNetPreparePipe") {
            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                if (Array.isArray(info?.widgets_values)) {
                    info = {
                        ...info,
                        widgets_values: normalizeZITPickerValue(
                            this,
                            info.widgets_values
                        ),
                    };
                }
                return originalOnConfigure?.call(this, info);
            };

            const originalOnSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(info) {
                const result = originalOnSerialize?.call(this, info);
                const pickerIndex = this.widgets?.findIndex(
                    (widget) => widget?.name === BUTTON_WIDGET_NAME
                ) ?? -1;
                if (
                    pickerIndex >= 0 &&
                    Array.isArray(info?.widgets_values) &&
                    info.widgets_values.length === this.widgets.length
                ) {
                    info.widgets_values.splice(pickerIndex, 1);
                }
                return result;
            };

            return;
        }
        clarifyPercentSchema(nodeData);

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (Array.isArray(info?.widgets_values)) {
                info = { ...info, widgets_values: normalizePickerValue(this, info.widgets_values) };
                info.widgets_values = migrateLegacyFractions(this, info.widgets_values, info);
            }
            return originalOnConfigure?.call(this, info);
        };

        const originalOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(info) {
            const result = originalOnSerialize?.call(this, info);
            info.properties = { ...(info.properties || {}), cmkControlNetPercentScale: 100 };
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
            clarifyPercentLabels(this);
            return result;
        };
    },

    nodeCreated(node) {
        addReferencePickerButton(node);
        setTimeout(() => setupCombinedReferencePreview(node), 0);
        clarifyPercentLabels(node);
    },
});
