import { app } from "../../../scripts/app.js";

const COMBINED_TYPE = "CMKFamilyResultMergePipe";
const BOUNDARY_TYPES = new Set([
    COMBINED_TYPE,
    "CMKPostProcessBoundarySDXLPipe",
    "CMKPostProcessBoundaryZITPipe",
]);
const ADVANCED_WIDGETS = new Set([
    "postprocess_checkpoint",
    "postprocess_vae",
    "use_checkpoint_vae",
]);
function nodeClass(node) {
    return node?.comfyClass
        ?? node?.constructor?.comfyClass
        ?? node?.constructor?.nodeData?.name
        ?? node?.type;
}

function isBoundary(node) {
    return BOUNDARY_TYPES.has(nodeClass(node));
}

function configureAdvancedWidgets(node) {
    if (!isBoundary(node) || !Array.isArray(node.widgets)) return;
    for (const widget of node.widgets) {
        if (!widget) continue;
        const advanced = ADVANCED_WIDGETS.has(widget.name);
        widget.advanced = advanced;
        widget.options ??= {};
        widget.options.advanced = advanced;
    }
    node.widgets = [...node.widgets];
    node.setDirtyCanvas?.(true, true);
}

function scheduleConfigure(node) {
    for (const delay of [0, 50, 200]) {
        setTimeout(() => configureAdvancedWidgets(node), delay);
    }
}

function decorate() {
    for (const root of document.querySelectorAll(".lg-node[data-node-id]")) {
        const id = root.getAttribute("data-node-id");
        const node = [app.canvas?.graph, app.graph, app.rootGraph]
            .map((graph) => graph?.getNodeById?.(id) ?? graph?.getNodeById?.(Number(id)))
            .find(Boolean);
        const target = nodeClass(node) === COMBINED_TYPE;
        for (const slot of root.querySelectorAll(".lg-slot--input")) {
            const isSdxlStart = target && slot.textContent.trim() === "MODEL SDXL";
            slot.classList.toggle("cmk-family-block-start", isSdxlStart);
        }
    }
}

app.registerExtension({
    name: "cmk.family_merge_layout",
    setup() {
        if (!document.getElementById("cmk-family-merge-spacing")) {
            const style = document.createElement("style");
            style.id = "cmk-family-merge-spacing";
            style.textContent = '.lg-node:not([data-collapsed]) .cmk-family-block-start { margin-top: 20px !important; }';
            document.head.appendChild(style);
        }
        new MutationObserver(decorate).observe(document.documentElement, {
            childList: true, subtree: true,
        });
        decorate();
    },
    nodeCreated(node) {
        if (isBoundary(node)) scheduleConfigure(node);
    },
    loadedGraphNode(node) {
        scheduleConfigure(node);
    },
    afterConfigureGraph() {
        // Slot order is part of the Python node contract. Never reorder live
        // input arrays here: frontend releases differ in how assigned slot
        // views are resolved, and moving them can retarget serialized links.
        for (const graph of new Set([app.canvas?.graph, app.graph, app.rootGraph])) {
            for (const node of graph?._nodes || []) {
                configureAdvancedWidgets(node);
            }
        }
        queueMicrotask(decorate);
    },
});
