import { app } from "../../../scripts/app.js";

// FaceProcess 2.8 operating mode (F1). Presentation only: stored and sent
// COMBO values stay "None" / "Restore" / "SDXL" / "Z-Image Turbo".
//
// - CMK FaceProcess Mode Enable: four choice fields NONE | RESTORE | SDXL | ZIT
//   replace the dropdown, styled like CMK Detailer Family Enable.
// - Outer FaceProcess subgraphs: the mode dropdown is labelled MODE,
//   right-aligned, with short option labels (getOptionLabel).
// The Detailer controls keep their own extensions and are not touched here.
export const MODE_CONTROL_CLASS = "CMKFaceProcessModeEnable";
export const MODE_WIDGET = "process_mode";
export const MODE_CHOICES = [["None", "NONE"], ["Restore", "RESTORE"], ["SDXL", "SDXL"], ["Z-Image Turbo", "ZIT"]];

// Subgraph name -> outer widget and its option labels. The widget names are
// those the F5 subgraphs promote (Standard: the Mode Enable selection;
// Advanced: the global switch of the FaceProcess Advanced Prepare).
export const OUTER_MODES = {
    "CMK Flow · FaceProcess": {
        widget: MODE_WIDGET,
        labels: { None: "DISABLE", Restore: "RESTORE", SDXL: "SDXL", "Z-Image Turbo": "ZIT" },
    },
    "CMK Flow · FaceProcess · Advanced": {
        widget: "faceprocess_global_enable",
        labels: { ENABLE: "ENABLE", DISABLE: "DISABLE" },
    },
};

const CHOICE_NAME = "cmk_faceprocess_mode_choice";
const ACCENT = "rgb(88, 190, 230)";
const ROW_ATTRIBUTE = "data-cmk-faceprocess-mode-select";
const STYLE_ID = "cmk-faceprocess-mode-style";

function nodeClass(node) {
    return node?.comfyClass ?? node?.type ?? node?.constructor?.comfyClass ?? node?.constructor?.nodeData?.name;
}

function setWidgetVisible(target, visible) {
    if (!target) return;
    target.options ??= {};
    target._cmkModeVisibility ??= {
        computeSize: target.computeSize,
        draw: target.draw,
        hidden: target.hidden,
        optionsHidden: target.options.hidden,
    };
    const original = target._cmkModeVisibility;
    if (visible) {
        target.computeSize = original.computeSize;
        target.draw = original.draw;
        target.hidden = original.hidden;
        target.options.hidden = original.optionsHidden;
        return;
    }
    // The widget stays in node.widgets: its value is part of the prompt.
    target.computeSize = () => [0, -4];
    target.draw = () => {};
    target.hidden = true;
    target.options.hidden = true;
}

function selectionLinked(node) {
    return Boolean(node.inputs?.some((input) => input?.widget?.name === MODE_WIDGET && input.link != null));
}

function render(root) {
    for (const button of root.querySelectorAll("button")) {
        const active = button.dataset.value === root._cmkValue;
        button.setAttribute("aria-pressed", active ? "true" : "false");
        button.style.color = active ? ACCENT : "rgba(214, 219, 226, 0.5)";
        button.style.border = active ? "1px solid rgba(88, 190, 230, 0.6)" : "1px solid rgba(255, 255, 255, 0.09)";
        button.style.fontWeight = active ? "700" : "500";
    }
}

export function installModeChoice(node) {
    const original = node.widgets?.find((target) => target?.name === MODE_WIDGET);
    if (!original || typeof node.addDOMWidget !== "function") return;
    let panel = node.widgets.find((target) => target?.name === CHOICE_NAME);
    if (!panel) {
        const root = document.createElement("div");
        root.className = "cmk-faceprocess-mode-choice";
        Object.assign(root.style, {
            display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px",
            width: "100%", boxSizing: "border-box", padding: "2px 0",
        });
        for (const [value, label] of MODE_CHOICES) {
            const button = document.createElement("button");
            button.type = "button";
            button.dataset.value = value;
            button.textContent = label;
            button.title = value;
            Object.assign(button.style, {
                height: "40px", borderRadius: "6px", cursor: "pointer", appearance: "none",
                background: "rgba(255, 255, 255, 0.025)", boxShadow: "none", outline: "none",
                fontFamily: "inherit", fontSize: "13px", letterSpacing: "0.08em",
                transition: "color 120ms ease, border-color 120ms ease",
            });
            button.addEventListener("pointerdown", (event) => event.stopPropagation());
            button.addEventListener("click", (event) => {
                event.stopPropagation();
                if (root._cmkValue === value) return;
                root._cmkValue = value;
                original.value = value;
                render(root);
                try { original.callback?.(value, app.canvas, node); } catch (_) {}
                node.setDirtyCanvas?.(true, true);
                app.graph?.setDirtyCanvas?.(true, true);
            });
            root.append(button);
        }
        root._cmkValue = String(original.value ?? "None");
        panel = node.addDOMWidget(CHOICE_NAME, CHOICE_NAME, root, {
            serialize: false,
            hideOnZoom: false,
            getMinHeight: () => 46,
            getHeight: () => 46,
            getValue: () => root._cmkValue,
            setValue: (value) => { root._cmkValue = String(value ?? "None"); render(root); },
        });
        panel.serialize = false;
        panel.computeLayoutSize = undefined;
        panel._cmkRoot = root;
    }
    const root = panel._cmkRoot;
    root._cmkValue = String(original.value ?? "None");
    render(root);
    // A linked selection (Standard subgraph: MODE from the outer widget) is
    // decided elsewhere; ComfyUI hides the dropdown, so do the same.
    setWidgetVisible(original, false);
    setWidgetVisible(panel, !selectionLinked(node));
    node.setDirtyCanvas?.(true, true);
}

export function outerModeSpec(node) {
    if (!node?.isSubgraphNode?.()) return null;
    return OUTER_MODES[node.subgraph?.name] ?? null;
}

export function applyOuterMode(node) {
    const spec = outerModeSpec(node);
    if (!spec) return false;
    const widget = node.widgets?.find((item) => item?.name === spec.widget);
    if (!widget) return false;
    widget.label = "MODE";
    widget.options ??= {};
    if (widget.options.cmkModeLabels !== spec.labels) {
        widget.options.cmkModeLabels = spec.labels;
        widget.options.getOptionLabel = (value) => spec.labels[value] ?? value;
    }
    return true;
}

function graphNode(id) {
    return [app.canvas?.graph, app.graph, app.rootGraph]
        .map((graph) => graph?.getNodeById?.(id) ?? graph?.getNodeById?.(Number(id)))
        .find(Boolean);
}

function decorate() {
    for (const root of document.querySelectorAll(".lg-node[data-node-id]")) {
        const node = graphNode(root.getAttribute("data-node-id"));
        const spec = outerModeSpec(node);
        if (!spec) continue;
        applyOuterMode(node);
        const row = root.querySelector(`.lg-node-widget[data-widget-name="${spec.widget}"]`);
        if (row && !row.hasAttribute(ROW_ATTRIBUTE)) row.setAttribute(ROW_ATTRIBUTE, "");
    }
}

function installStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.lg-node-widget[${ROW_ATTRIBUTE}] [data-testid="widget-select-default-trigger"] > span { text-align: right; }
`;
    document.head.appendChild(style);
}

function refresh(node) {
    if (nodeClass(node) === MODE_CONTROL_CLASS) installModeChoice(node);
    else applyOuterMode(node);
}

// Late passes cover widgets the frontend re-creates after creation.
function schedule(node) {
    for (const delay of [0, 50, 200]) setTimeout(() => refresh(node), delay);
}

app.registerExtension({
    name: "cmk.faceprocess.mode.ui.v1",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== MODE_CONTROL_CLASS) return;
        for (const hook of ["onNodeCreated", "onConfigure", "onAdded", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                schedule(this);
                return result;
            };
        }
    },

    setup() {
        installStyle();
        new MutationObserver(decorate).observe(document.documentElement, { childList: true, subtree: true });
    },

    nodeCreated(node) {
        schedule(node);
    },

    loadedGraphNode(node) {
        schedule(node);
    },
});
