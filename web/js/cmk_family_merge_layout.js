import { app } from "../../../scripts/app.js";

const TYPE = "CMKFamilyResultMergePipe";
const ORDER = [
    "MODEL SDXL", "PROCESS SDXL", "IMAGE SDXL", "LOG SDXL", "VISUAL SDXL",
    "MODEL ZIT", "PROCESS ZIT", "IMAGE ZIT", "LOG ZIT", "VISUAL ZIT",
];

// Reorder the existing slot objects, preserving their links and metadata.
function arrangeInputs(node) {
    if (node?.type !== TYPE || !node.inputs) return;
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

function decorate() {
    for (const root of document.querySelectorAll(".lg-node[data-node-id]")) {
        const id = root.getAttribute("data-node-id");
        const node = [app.canvas?.graph, app.graph, app.rootGraph]
            .map((graph) => graph?.getNodeById?.(id) ?? graph?.getNodeById?.(Number(id)))
            .find(Boolean);
        const target = node?.type === TYPE;
        for (const slot of root.querySelectorAll(".lg-slot--input")) {
            const isZitStart = target && slot.textContent.trim() === "MODEL ZIT";
            slot.classList.toggle("cmk-family-block-start", isZitStart);
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
        if (node?.type === TYPE) queueMicrotask(() => arrangeInputs(node));
    },
    loadedGraphNode(node) {
        arrangeInputs(node);
    },
    afterConfigureGraph() {
        // Graph links are fully available only after graph configuration.
        for (const graph of new Set([app.canvas?.graph, app.graph, app.rootGraph])) {
            for (const node of graph?._nodes || []) {
                arrangeInputs(node);
                if (node.type !== TYPE) continue;
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
