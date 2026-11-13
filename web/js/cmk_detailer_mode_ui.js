import { app } from "../../../scripts/app.js";

// MODE dropdown of the outer Detailer subgraphs (2.8.0). Presentation only:
// the stored and sent COMBO value stays the backend value; the select shows
// its label through the frontend's options.getOptionLabel and right-aligns it.
const MODE_WIDGETS = {
    "CMK Flow · Detailer": {
        widget: "model_family",
        labels: { None: "DISABLE", SDXL: "SDXL", "Z-Image Turbo": "ZIT" },
    },
    "CMK Flow · Detailer · Advanced": {
        widget: "detailer_global_enable",
        labels: { ENABLE: "ENABLE", DISABLE: "DISABLE" },
    },
};
const ROW_ATTRIBUTE = "data-cmk-mode-select";
const STYLE_ID = "cmk-detailer-mode-style";

export function modeSpec(node) {
    if (!node?.isSubgraphNode?.()) return null;
    return MODE_WIDGETS[node.subgraph?.name] ?? null;
}

export function applyMode(node) {
    const spec = modeSpec(node);
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
        const spec = modeSpec(node);
        if (!spec) continue;
        applyMode(node);
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

function schedule(node) {
    for (const delay of [0, 50, 200]) setTimeout(() => applyMode(node), delay);
}

app.registerExtension({
    name: "cmk.detailer.mode.ui.v1",

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
