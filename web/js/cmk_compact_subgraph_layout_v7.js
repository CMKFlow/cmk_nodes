import { app } from "../../../scripts/app.js";

const STYLE_ID = "cmk-compact-subgraph-bottom-widgets-v7";
const NODE_CLASS = "cmk-compact-subgraph-bottom-widgets";
const ADVANCED_NODE_CLASS = "cmk-compact-subgraph-advanced";
const NODE_SELECTOR = '.lg-node[data-node-id], [data-testid^="node-body-"]';
const OBSOLETE_SPACER_NAME = "cmk_advanced_combo_spacer";
const UPSCALE_SAVE_TITLE = "CMK Flow · 90 Upscale & Save";
const UPSCALE_SAVE_MIN_HEIGHT_REDUCTION = 2;

function installStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .${NODE_CLASS} [data-testid="node-widgets"] {
            margin-top: auto !important;
        }
        .${NODE_CLASS}.${ADVANCED_NODE_CLASS} [data-testid="node-widgets"] {
            padding-bottom: 30px !important;
        }
    `;
    document.head.appendChild(style);
}

function nodeIdFromElement(element) {
    const root = element?.closest?.(".lg-node") || element;
    const body = root?.querySelector?.('[data-testid^="node-body-"]');
    const testId = body?.getAttribute?.("data-testid") || "";
    return root?.getAttribute?.("data-node-id")
        ?? testId.match(/^node-body-(.+)$/)?.[1]
        ?? null;
}

function graphNode(nodeId) {
    for (const graph of [app.canvas?.graph, app.graph, app.rootGraph]) {
        for (const candidate of [nodeId, Number(nodeId)]) {
            if (candidate == null || candidate === "" || Number.isNaN(candidate)) continue;
            const node = graph?.getNodeById?.(candidate);
            if (node) return node;
        }
    }
    return null;
}

function isCmkVisualSubgraph(node) {
    return Boolean(
        node?.isSubgraphNode?.()
        && Array.isArray(node?.properties?.cmkVisualProviders),
    );
}

function isAdvancedSubgraph(node) {
    return node?.properties?.cmkVisualProviders?.some(
        (provider) => String(provider?.stage_key || "").endsWith(".advanced"),
    ) ?? false;
}

function compactUpscaleSaveMinimumHeight(node) {
    if (String(node?.title || "") !== UPSCALE_SAVE_TITLE) return;
    if (node.__cmkUpscaleSaveCompactMinimum) return;
    node.__cmkUpscaleSaveCompactMinimum = true;
    const originalComputeSize = node.computeSize;
    if (typeof originalComputeSize !== "function") return;
    node.computeSize = function(out) {
        const measured = originalComputeSize.call(this, out);
        if (!Array.isArray(measured)) return measured;
        const result = [Number(measured[0]), Number(measured[1])];
        if (Number.isFinite(result[1])) {
            result[1] = Math.max(1, result[1] - UPSCALE_SAVE_MIN_HEIGHT_REDUCTION);
        }
        return result;
    };
    node.setDirtyCanvas?.(true, true);
}

function replaceObsoleteSpacer(node) {
    const spacers = (node?.widgets || []).filter(
        (widget) => widget?.name === OBSOLETE_SPACER_NAME,
    );
    if (spacers.some((widget) => widget.__cmkInvisibleSentinel)) return;
    for (const widget of spacers) node.ensureWidgetRemoved?.(widget);
    if (typeof node?.addCustomWidget !== "function") return;
    node.addCustomWidget({
        name: OBSOLETE_SPACER_NAME,
        type: "converted-widget",
        value: null,
        hidden: true,
        options: { hidden: true },
        serialize: false,
        __cmkInvisibleSentinel: true,
        computeSize: () => [0, -4],
        draw() {},
    });
}

function decorate(element) {
    const root = element?.closest?.(".lg-node") || element;
    if (!root?.classList) return;
    const node = graphNode(nodeIdFromElement(root));
    const isTarget = isCmkVisualSubgraph(node);
    root.classList.toggle(NODE_CLASS, isTarget);
    root.classList.toggle(ADVANCED_NODE_CLASS, isTarget && isAdvancedSubgraph(node));
    if (isTarget) {
        replaceObsoleteSpacer(node);
        compactUpscaleSaveMinimumHeight(node);
    }
}

function scan() {
    installStyle();
    for (const element of document.querySelectorAll(NODE_SELECTOR)) decorate(element);
}

app.registerExtension({
    name: "cmk.compact_subgraph_layout.v7",
    setup() {
        scan();
        const observer = new MutationObserver(scan);
        observer.observe(document.documentElement, { childList: true, subtree: true });
        setInterval(scan, 500);
    },
    nodeCreated() {
        queueMicrotask(scan);
    },
    loadedGraphNode() {
        queueMicrotask(scan);
    },
});
