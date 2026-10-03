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
const ORDER = [
    "postprocess_checkpoint", "postprocess_vae", "use_checkpoint_vae",
    "MODEL ZIT", "PROCESS ZIT", "IMAGE ZIT", "LOG ZIT", "VISUAL ZIT",
    "MODEL SDXL", "PROCESS SDXL", "IMAGE SDXL", "LOG SDXL", "VISUAL SDXL",
];

function nodeClass(node) {
    return node?.comfyClass
        ?? node?.constructor?.comfyClass
        ?? node?.constructor?.nodeData?.name
        ?? node?.type;
}

function isBoundary(node) {
    return BOUNDARY_TYPES.has(nodeClass(node));
}

// Reorder the existing slot objects, preserving their links and metadata.
function arrangeInputs(node) {
    if (nodeClass(node) !== COMBINED_TYPE || !node.inputs) return;
    const previous = [...node.inputs];
    const rank = (input) => {
        const index = ORDER.indexOf(input.name);
        return index < 0 ? ORDER.length : index;
    };
    const sorted = [...previous].sort((a, b) => rank(a) - rank(b));
    if (sorted.every((input, index) => input === previous[index])) return;
    node.inputs.splice(0, node.inputs.length, ...sorted);
    for (const [index, input] of node.inputs.entries()) {
        if (input.link == null) continue;
        const links = node.graph?.links;
        const link = links?.get?.(input.link) ?? links?.[input.link];
        if (link) link.target_slot = index;
    }
    node.setDirtyCanvas?.(true, true);
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

function configure(node) {
    arrangeInputs(node);
    configureAdvancedWidgets(node);
}

function scheduleConfigure(node) {
    for (const delay of [0, 50, 200]) {
        setTimeout(() => configure(node), delay);
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
        // Graph links are fully available only after graph configuration.
        for (const graph of new Set([app.canvas?.graph, app.graph, app.rootGraph])) {
            for (const node of graph?._nodes || []) {
                configure(node);
                if (nodeClass(node) !== COMBINED_TYPE) continue;
                for (const [index, input] of (node.inputs || []).entries()) {
                    if (input.link == null) continue;
                    const link = graph.links?.get?.(input.link) ?? graph.links?.[input.link];
                    if (link) link.target_slot = index;
                }
            }
        }
        queueMicrotask(decorate);
    },
});
