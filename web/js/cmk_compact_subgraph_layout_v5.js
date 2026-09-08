import { app } from "../../../scripts/app.js";

const STYLE_ID = "cmk-compact-subgraph-bottom-widgets-v4";
const NODE_CLASS = "cmk-compact-subgraph-bottom-widgets";
const NODE_SELECTOR = '.lg-node[data-node-id], [data-testid^="node-body-"]';
const OBSOLETE_SPACER_NAME = "cmk_advanced_combo_spacer";

function installStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .${NODE_CLASS} [data-testid="node-widgets"] {
            margin-top: auto !important;
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

function hideObsoleteSpacer(node) {
    for (const widget of node?.widgets || []) {
        if (widget?.name !== OBSOLETE_SPACER_NAME) continue;
        widget.options ||= {};
        widget.options.hidden = true;
        widget.hidden = true;
    }
}

function decorate(element) {
    const root = element?.closest?.(".lg-node") || element;
    if (!root?.classList) return;
    const node = graphNode(nodeIdFromElement(root));
    const isTarget = isCmkVisualSubgraph(node);
    root.classList.toggle(NODE_CLASS, isTarget);
    if (isTarget) hideObsoleteSpacer(node);
}

function scan() {
    installStyle();
    for (const element of document.querySelectorAll(NODE_SELECTOR)) decorate(element);
}

app.registerExtension({
    name: "cmk.compact_subgraph_layout.v5",
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
