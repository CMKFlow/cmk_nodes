import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKInstantIDSamplerSDXLPipe";
const NODE_SELECTOR = "[data-node-id]";
const LABEL_SELECTOR = '[data-testid="widget-layout-field-label"]';
const ROW_CLASS = "cmk-instantid-sampling-section";

function isTarget(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS
        || node.type === NODE_CLASS
        || node.constructor?.comfyClass === NODE_CLASS
        || node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function nodeForElement(element) {
    const nodeElement = element?.closest?.(NODE_SELECTOR);
    const id = nodeElement?.dataset?.nodeId;
    if (id == null) return null;
    const candidates = [id, Number(id)].filter(
        (value) => value !== "" && !Number.isNaN(value),
    );
    for (const graph of [app.canvas?.graph, app.graph, app.rootGraph]) {
        for (const candidate of candidates) {
            const node = graph?.getNodeById?.(candidate);
            if (node) return node;
        }
    }
    return null;
}

function decorateNodeElement(nodeElement) {
    const node = nodeForElement(nodeElement);
    if (!isTarget(node)) return;
    const referenceMode = node.widgets?.find(
        (widget) => widget?.name === "conditioning_mode",
    )?.value === "InstantID reference";
    for (const label of nodeElement.querySelectorAll(LABEL_SELECTOR)) {
        const name = label.textContent?.trim();
        if (name === "conditioning_mode") {
            label.parentElement?.classList.add(ROW_CLASS);
        } else if (name === "reference_conditioning_weight" && label.parentElement) {
            label.parentElement.style.display = referenceMode ? "" : "none";
        }
    }
}

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    widget._cmkInstantIDVisibility ??= {
        type: widget.type,
        computeSize: widget.computeSize,
        draw: widget.draw,
        hidden: widget.hidden,
    };
    const original = widget._cmkInstantIDVisibility;
    if (visible) {
        widget.type = original.type;
        widget.computeSize = original.computeSize;
        widget.draw = original.draw;
        widget.hidden = original.hidden;
    } else {
        widget.type = "converted-widget";
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
        widget.hidden = true;
    }
}

function updateConditioningVisibility(node) {
    const mode = node.widgets?.find((widget) => widget?.name === "conditioning_mode");
    const weight = node.widgets?.find(
        (widget) => widget?.name === "reference_conditioning_weight",
    );
    setWidgetVisible(weight, mode?.value === "InstantID reference");
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    scan();
}

function configureNode(node) {
    if (!isTarget(node) || !Array.isArray(node.widgets)) return;
    const mode = node.widgets.find((widget) => widget?.name === "conditioning_mode");
    const standardLabels = {
        identity_strength: "IDENTITY STRENGTH",
        target_face: "TARGET FACE",
        pose_strength: "POSE STRENGTH",
        instantid_start: "INSTANTID START",
        instantid_end: "INSTANTID END",
        noise: "IDENTITY EMBEDDING NOISE",
        reference_start_at_step: "SAMPLING START AT STEP",
    };
    for (const widget of node.widgets) {
        if (standardLabels[widget?.name]) widget.label = standardLabels[widget.name];
    }
    for (const input of node.inputs ?? []) {
        if (input?.name !== "source_face") continue;
        input.label = "SOURCE FACE";
        input.localized_name = "SOURCE FACE";
    }
    if (mode && !mode._cmkInstantIDVisibilityInstalled) {
        mode._cmkInstantIDVisibilityInstalled = true;
        const originalCallback = mode.callback;
        mode.callback = function(value) {
            const result = originalCallback?.apply(this, arguments);
            updateConditioningVisibility(node);
            return result;
        };
    }
    updateConditioningVisibility(node);
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

function installStyles() {
    if (document.getElementById("cmk-instantid-sampler-style")) return;
    const style = document.createElement("style");
    style.id = "cmk-instantid-sampler-style";
    style.textContent = `
      .${ROW_CLASS} {
        position: relative;
        margin-top: 8px;
        padding-top: 10px;
      }
      .${ROW_CLASS}::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: rgba(255, 255, 255, 0.18);
        pointer-events: none;
      }
    `;
    document.head.append(style);
}

function installDomBehavior() {
    if (window._cmkInstantIDSamplerDomInstalled) return;
    window._cmkInstantIDSamplerDomInstalled = true;
    installStyles();

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

app.registerExtension({
    name: "cmk.instantid.sampler.ui.v7",

    setup() {
        installDomBehavior();
    },

    nodeCreated(node) {
        if (!isTarget(node)) return;
        for (const delay of [0, 50, 200]) {
            setTimeout(() => {
                configureNode(node);
                scan();
            }, delay);
        }
    },
});
