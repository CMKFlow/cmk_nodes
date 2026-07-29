import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKPipeCreateImage";
const GUIDE_NAME = "NEXT STEP";
const GUIDE_TEXT = "05 ControlNet (optional)  or  10 KSampler 1st Pass";
const MODE_INFO_NAME = "MODE INFO";
const MODE_INFO = {
    "custom": "Manual prompts and Advanced settings.",
    "replace object": "Prompt and LoRAs describe the replacement.",
    "remove object": "Prompt-free reconstruction from the surrounding image.",
    "extend image": "Prompt describes the extension; Fit creates canvas and mask.",
};
const MODE_INFO_TOOLTIP = {
    "custom": "Uses the user prompts and the technical Sampler Advanced values without guided overrides.",
    "replace object": "Noise removes the old silhouette; user prompt and LoRAs describe the new content; denoise 1.00, noise mask ON, context reference ON; outpaint OFF.",
    "remove object": "Local LaMa reconstructs the masked area from the surrounding image. No prompt, LoRA, KSampler or Refiner is used.",
    "extend image": "Fit creates the outpaint canvas and mask. Navier-Stokes continues the surroundings as preparation; user prompt describes the extended scene; denoise 1.00, noise mask ON, context reference ON.",
};
const GUIDED_FILL = {
    "replace object": "noise",
    "remove object": "lama",
    "extend image": "navier-stokes",
};
const GUIDED_OUTPAINT = {
    "replace object": false,
    "remove object": false,
    "extend image": true,
};
const GUIDED_RESIZE = {
    "extend image": "Fit",
};
const INPAINT_ONLY_WIDGETS = new Set([
    "outpaint_on",
    "outpaint_overlap",
    "mask_fill_holes",
    "fill_masked_area",
    "process_mode",
    MODE_INFO_NAME,
]);

const CROP_POSITION_WIDGET = "crop_position";
const USER_WIDGET_LABELS = {
    "PROMPT POS": "PROMPT POS",
    "PROMPT NEG": "PROMPT NEG",
    "INPAINT_MODE": "MODE",
    "resolution": "IMAGE SIZE",
    "swap_dimensions": "SWAP WIDTH / HEIGHT",
    "upscale_method": "RESIZE QUALITY",
    "outpaint_on": "OUTPAINT",
    "mask_fill_holes": "FILL MASK HOLES",
    "fill_masked_area": "MASK FILL",
    "process_mode": "PROCESS MODE",
    "resize_mode": "IMAGE FIT",
    "crop_position": "IMAGE POSITION",
};
const USER_INPUT_LABELS = {
    "PROCESS": "PROCESS",
    "IMAGE": "IMAGE",
    "MASK": "MASK",
    "FILENAME STRING": "FILENAME",
    "LOG": "LOG",
    "lora_stack": "LORA STACK",
    "lora_syntax": "ACTIVE LORAS",
    "opt_prompt_pos": "ADDITIONAL PROMPT",
};
const USER_OUTPUT_LABELS = {
    "PROCESS": "PROCESS",
    "IMAGE": "IMAGE",
    "LOG": "LOG",
    "diagnostic": "DIAGNOSTIC",
};

function isTarget(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS ||
        node.type === NODE_CLASS ||
        node.constructor?.comfyClass === NODE_CLASS ||
        node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function captureWidgets(node) {
    node._cmkStartUi ??= {
        widgetsByName: new Map(),
        canonicalOrder: [],
        visibleMode: null,
        rebuilding: false,
    };
    const state = node._cmkStartUi;
    for (const widget of node.widgets ?? []) {
        if (!widget?.name) continue;
        if (!state.widgetsByName.has(widget.name)) {
            state.canonicalOrder.push(widget.name);
        }
        state.widgetsByName.set(widget.name, widget);
    }
    return state;
}

function getWidget(node, name) {
    return captureWidgets(node).widgetsByName.get(name) ?? null;
}

function isInpaintMode(node) {
    const value = getWidget(node, "INPAINT_MODE")?.value;
    if (typeof value === "boolean") return value;
    return String(value ?? "Text2Image").trim().toLowerCase() === "inpaint";
}

function rebuildModeWidgets(node, force = false) {
    const state = captureWidgets(node);
    if (state.rebuilding) return;
    const mode = isInpaintMode(node) ? "inpaint" : "text2image";
    if (!force && state.visibleMode === mode) return;

    state.rebuilding = true;
    try {
        node.widgets = state.canonicalOrder
            .filter((name) => mode === "inpaint" || !INPAINT_ONLY_WIDGETS.has(name))
            .filter((name) => {
                if (name !== CROP_POSITION_WIDGET) return true;
                const resizeMode = String(
                    state.widgetsByName.get("resize_mode")?.value ?? "Fit"
                ).trim().toLowerCase();
                return resizeMode !== "stretch";
            })
            .map((name) => state.widgetsByName.get(name))
            .filter(Boolean);
        state.visibleMode = mode;
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    } finally {
        state.rebuilding = false;
    }
}

function configure(node) {
    if (!isTarget(node) || typeof node.addWidget !== "function") return;
    captureWidgets(node);
    for (const widget of node.widgets ?? []) {
        if (USER_WIDGET_LABELS[widget?.name]) {
            widget.label = USER_WIDGET_LABELS[widget.name];
        }
    }
    for (const input of node.inputs ?? []) {
        if (USER_INPUT_LABELS[input?.name]) {
            input.label = USER_INPUT_LABELS[input.name];
            input.localized_name = USER_INPUT_LABELS[input.name];
        }
    }
    for (const output of node.outputs ?? []) {
        if (USER_OUTPUT_LABELS[output?.name]) {
            output.label = USER_OUTPUT_LABELS[output.name];
            output.localized_name = USER_OUTPUT_LABELS[output.name];
        }
    }
    if (node.properties && "cmkStartPreferredWidth" in node.properties) {
        delete node.properties.cmkStartPreferredWidth;
    }

    const flowMode = getWidget(node, "INPAINT_MODE");
    if (flowMode) {
        flowMode.label = "MODE";
        if (typeof flowMode.value === "boolean") {
            flowMode.value = flowMode.value ? "Inpaint" : "Text2Image";
        }
        if (!flowMode._cmkModeVisibilityCallbackInstalled) {
            const originalModeCallback = flowMode.callback;
            flowMode.callback = function () {
                const result = originalModeCallback?.apply(this, arguments);
                for (const delay of [0, 50, 200]) {
                    setTimeout(() => rebuildModeWidgets(node, true), delay);
                }
                return result;
            };
            flowMode._cmkModeVisibilityCallbackInstalled = true;
        }
    }

    const resizeMode = getWidget(node, "resize_mode");
    if (resizeMode && !resizeMode._cmkResizeVisibilityCallbackInstalled) {
        const originalResizeCallback = resizeMode.callback;
        resizeMode.callback = function () {
            const result = originalResizeCallback?.apply(this, arguments);
            rebuildModeWidgets(node, true);
            return result;
        };
        resizeMode._cmkResizeVisibilityCallbackInstalled = true;
    }

    const processMode = getWidget(node, "process_mode");
    if (processMode) {
        processMode.label = "PROCESS MODE";
        processMode.advanced = false;
        processMode.hidden = false;
        const processModeTooltip = [
            "Guided inpaint preset; no semantic object recognition.",
            "Custom: Sampler Advanced values unchanged.",
            "Replace Object: noise fill, denoise 1.00, noise mask ON, context reference ON and outpaint OFF; user prompt and LoRAs remain active.",
            "Remove Object: local LaMa reconstructs from the image context. Diffusion, source prompts and all LoRAs are bypassed; no user prompt required.",
            "Extend Image: Fit creates canvas and mask; Navier-Stokes fill, denoise 1.00, noise mask ON, context reference ON.",
            "Guided modes override fill_masked_area. Custom leaves all technical values selectable.",
        ].join("\n");
        processMode.tooltip = processModeTooltip;
        processMode.options = {
            ...(processMode.options ?? {}),
            tooltip: processModeTooltip,
        };
        if (!processMode._cmkModeInfoCallbackInstalled) {
            const originalCallback = processMode.callback;
            processMode.callback = function () {
                const result = originalCallback?.apply(this, arguments);
                const modeKey = String(processMode.value ?? "Custom").trim().toLowerCase();
                const fillWidget = getWidget(node, "fill_masked_area");
                if (fillWidget) {
                    if (modeKey === "custom") {
                        fillWidget.value = node._cmkCustomFillMaskedArea ?? fillWidget.value;
                    } else {
                        if (node._cmkLastProcessMode === "custom") {
                            node._cmkCustomFillMaskedArea = fillWidget.value;
                        }
                        fillWidget.value = GUIDED_FILL[modeKey] ?? fillWidget.value;
                    }
                }
                const outpaintWidget = getWidget(node, "outpaint_on");
                if (outpaintWidget) {
                    if (modeKey === "custom") {
                        outpaintWidget.value = node._cmkCustomOutpaint ?? outpaintWidget.value;
                    } else {
                        if (node._cmkLastProcessMode === "custom") {
                            node._cmkCustomOutpaint = outpaintWidget.value;
                        }
                        outpaintWidget.value = GUIDED_OUTPAINT[modeKey] ?? outpaintWidget.value;
                    }
                }
                const resizeWidget = getWidget(node, "resize_mode");
                if (resizeWidget) {
                    if (modeKey === "custom") {
                        resizeWidget.value = node._cmkCustomResizeMode ?? resizeWidget.value;
                    } else if (GUIDED_RESIZE[modeKey]) {
                        if (node._cmkLastProcessMode === "custom") {
                            node._cmkCustomResizeMode = resizeWidget.value;
                        }
                        resizeWidget.value = GUIDED_RESIZE[modeKey];
                    }
                }
                node._cmkLastProcessMode = modeKey;
                modeInfo.value = MODE_INFO[modeKey] ?? MODE_INFO.custom;
                modeInfo.tooltip = MODE_INFO_TOOLTIP[modeKey] ?? MODE_INFO_TOOLTIP.custom;
                modeInfo.options = {
                    ...(modeInfo.options ?? {}),
                    tooltip: modeInfo.tooltip,
                };
                node.setDirtyCanvas?.(true, true);
                app.graph?.setDirtyCanvas?.(true, true);
                return result;
            };
            processMode._cmkModeInfoCallbackInstalled = true;
        }
    }

    let modeInfo = getWidget(node, MODE_INFO_NAME);
    if (!modeInfo) {
        modeInfo = node.addWidget("text", MODE_INFO_NAME, "", () => {}, {
            serialize: false,
        });
        captureWidgets(node);
    }
    const modeKey = String(processMode?.value ?? "Custom").trim().toLowerCase();
    const fillWidget = getWidget(node, "fill_masked_area");
    if (fillWidget && modeKey !== "custom") {
        fillWidget.value = GUIDED_FILL[modeKey] ?? fillWidget.value;
    }
    const outpaintWidget = getWidget(node, "outpaint_on");
    if (outpaintWidget && modeKey !== "custom") {
        outpaintWidget.value = GUIDED_OUTPAINT[modeKey] ?? outpaintWidget.value;
    }
    const resizeWidget = getWidget(node, "resize_mode");
    if (resizeWidget && GUIDED_RESIZE[modeKey]) {
        resizeWidget.value = GUIDED_RESIZE[modeKey];
    }
    node._cmkLastProcessMode = modeKey;
    modeInfo.value = MODE_INFO[modeKey] ?? MODE_INFO.custom;
    modeInfo.label = "MODE INFO →";
    modeInfo.disabled = true;
    modeInfo.serialize = false;
    modeInfo.tooltip = MODE_INFO_TOOLTIP[modeKey] ?? MODE_INFO_TOOLTIP.custom;
    modeInfo.options = {
        ...(modeInfo.options ?? {}),
        tooltip: modeInfo.tooltip,
    };

    let guide = getWidget(node, GUIDE_NAME);
    if (!guide) {
        guide = node.addWidget("text", GUIDE_NAME, GUIDE_TEXT, () => {}, {
            serialize: false,
        });
        captureWidgets(node);
    }

    guide.value = GUIDE_TEXT;
    guide.label = "NEXT STEP →";
    guide.disabled = true;
    guide.serialize = false;

    rebuildModeWidgets(node, true);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function schedule(node) {
    for (const delay of [0, 50, 200, 500]) setTimeout(() => configure(node), delay);
}

app.registerExtension({
    name: "cmk.flow.start.guidance.v21",

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
        if (isTarget(node)) schedule(node);
    },

    loadedGraphNode(node) {
        if (isTarget(node)) schedule(node);
    },
});
