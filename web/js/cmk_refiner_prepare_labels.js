import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKRefinerPrepareSDXLPipe";
const NODE_SELECTOR = "[data-node-id]";
const LABEL_SELECTOR = '[data-testid="widget-layout-field-label"]';

// Standard widgets are visual anchors in the expanded node. Advanced widgets
// intentionally retain their technical lower-case/snake_case names.
const STANDARD_LABELS = {
    ckpt_name: "CHECKPOINT",
    use_prompt_lora_from_sampler: "USE PROMPT FROM 1ST PASS",
    use_lora_from_1st_pass: "USE LORA FROM 1ST PASS",
    lora_name: "LORA",
    steps: "LOCAL STEPS",
    cfg: "CFG",
};

function updateSamplingWidgets(node) {
    // Keep every widget structurally intact. The current Vue renderer does not
    // reliably remount widgets after they were converted/removed, so only the
    // rendered DOM rows are toggled below.
    scan();
}

function installSamplingToggle(node) {
    const inherit = node.widgets?.find((widget) => widget?.name === "sampling_source");
    if (!inherit || inherit._cmkRefinerToggleInstalled) return;
    inherit._cmkRefinerToggleInstalled = true;
    const originalCallback = inherit.callback;
    inherit.callback = function(value) {
        const result = originalCallback?.apply(this, arguments);
        updateSamplingWidgets(node);
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
        for (const delay of [0, 50]) {
            setTimeout(() => updateSamplingWidgets(node), delay);
        }
        return result;
    };
}

function isTarget(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS ||
        node.type === NODE_CLASS ||
        node.constructor?.comfyClass === NODE_CLASS ||
        node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function nodeForElement(element) {
    const nodeElement = element?.closest?.(NODE_SELECTOR);
    const id = nodeElement?.dataset?.nodeId;
    if (id == null) return null;
    for (const graph of [app.canvas?.graph, app.graph, app.rootGraph]) {
        for (const candidate of [id, Number(id)]) {
            if (candidate === "" || Number.isNaN(candidate)) continue;
            const node = graph?.getNodeById?.(candidate);
            if (node) return node;
        }
    }
    return null;
}

function decorateNodeElement(nodeElement) {
    const node = nodeForElement(nodeElement);
    if (!isTarget(node)) return;
    const source = node.widgets?.find((widget) => widget?.name === "sampling_source");
    const showLocal = source?.value === "Local settings";
    const localLabels = new Set(["steps", "LOCAL STEPS", "sampler", "scheduler"]);
    for (const label of nodeElement.querySelectorAll(LABEL_SELECTOR)) {
        if (!localLabels.has(label.textContent?.trim())) continue;
        if (label.parentElement) label.parentElement.style.display = showLocal ? "" : "none";
    }
}

function scan(root = document) {
    if (root instanceof Element) {
        const owner = root.matches(NODE_SELECTOR) ? root : root.closest(NODE_SELECTOR);
        if (owner) decorateNodeElement(owner);
    }
    for (const nodeElement of root.querySelectorAll?.(NODE_SELECTOR) || []) {
        decorateNodeElement(nodeElement);
    }
}

function installDomBehavior() {
    if (window._cmkRefinerPrepareDomInstalled) return;
    window._cmkRefinerPrepareDomInstalled = true;
    const observer = new MutationObserver((records) => {
        for (const record of records) {
            for (const added of record.addedNodes) {
                if (added instanceof Element) scan(added);
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    scan();
}

function configure(node) {
    if (!isTarget(node) || !Array.isArray(node.widgets)) return;

    for (const widget of node.widgets) {
        const label = STANDARD_LABELS[widget?.name];
        if (label) widget.label = label;
    }

    for (const input of node.inputs ?? []) {
        if (input?.name === "prompt_pos_input") {
            input.label = "OPT PROMPT POS";
            input.localized_name = "OPT PROMPT POS";
        } else if (input?.name === "prompt_neg_input") {
            input.label = "OPT PROMPT NEG";
            input.localized_name = "OPT PROMPT NEG";
        }
    }

    installSamplingToggle(node);
    updateSamplingWidgets(node);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function migrateLegacyWidgetValues(data) {
    const values = data?.widgets_values;
    if (!Array.isArray(values)) return data;
    const migrated = [...values];

    // Legacy order contained two multiline prompt widgets at positions 4/5.
    if (
        typeof migrated[4] === "string"
        && typeof migrated[5] === "string"
        && typeof migrated[6] === "number"
    ) {
        migrated.splice(4, 2);

        // The first split-version stored USE LORA at the very end. Older
        // workflows have no separate value and receive the safe OFF default.
        const hasSplitLoraValue = (
            typeof migrated.at(-1) === "boolean"
            && typeof migrated.at(-2) === "boolean"
        );
        const loraFromFirstPass = hasSplitLoraValue ? migrated.pop() : false;
        migrated.splice(1, 0, loraFromFirstPass);
        data = { ...data, widgets_values: migrated };
    }

    // The dual-start version adds the InstantID-specific value immediately
    // after the existing start percentage. Preserve the legacy start value
    // and seed the new slot with the current InstantID candidate (90%).
    const current = data?.widgets_values;
    if (Array.isArray(current) && current.length === 14) {
        const dualStart = [...current];
        dualStart.splice(7, 0, 90);
        data = { ...data, widgets_values: dualStart };
    }

    // Move the inheritance switch from the old final position to the start
    // of the sampling block. This keeps all following widget values aligned.
    const ordered = data?.widgets_values;
    if (Array.isArray(ordered) && ordered.length === 15 && typeof ordered[14] === "boolean") {
        const reordered = [...ordered];
        const inheritSampling = reordered.pop();
        reordered.splice(5, 0, inheritSampling);
        data = { ...data, widgets_values: reordered };
    }

    // Boolean versions map losslessly onto the explicit dropdown.
    const dropdown = data?.widgets_values;
    if (Array.isArray(dropdown) && typeof dropdown[5] === "boolean") {
        const converted = [...dropdown];
        converted[5] = converted[5] ? "1st pass" : "Local settings";
        return { ...data, widgets_values: converted };
    }
    return data;
}

app.registerExtension({
    name: "cmk.refiner.prepare.ui.v7",

    setup() {
        installDomBehavior();
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            for (const delay of [0, 50, 200]) setTimeout(() => configure(this), delay);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(data) {
            const migrated = migrateLegacyWidgetValues(data);
            const result = originalOnConfigure?.call(this, migrated);
            for (const delay of [0, 50, 200]) setTimeout(() => configure(this), delay);
            return result;
        };
    },

    nodeCreated(node) {
        for (const delay of [0, 50, 200]) setTimeout(() => configure(node), delay);
    },
});
