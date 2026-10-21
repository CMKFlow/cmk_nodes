import { app } from "../../../scripts/app.js";

export const CMK_COLOR_PALETTE = Object.freeze({
    CONTROLNET: Object.freeze({
        title: "#202A2D",
        background: "#222F32",
        frame: "#263237",
    }),
    IDENTITY: Object.freeze({
        title: "#222A33",
        background: "#293341",
        frame: "#28313D",
    }),
    DETAILER: Object.freeze({
        title: "#2C2822",
        background: "#373128",
        frame: "#332D26",
    }),
    OUTPUT: Object.freeze({
        title: "#222D25",
        background: "#27372B",
        frame: "#2A392E",
    }),
});

export const CMK_COLOR_ROLE_PROPERTY = "cmkColorRole";

export function applyCmkSubgraphColor(node) {
    const colors = CMK_COLOR_PALETTE[node?.properties?.[CMK_COLOR_ROLE_PROPERTY]];
    if (!colors) return false;

    node.color = colors.title;
    node.bgcolor = colors.background;
    for (const group of node.subgraph?.groups || []) group.color = colors.frame;
    node.setDirtyCanvas?.(true, true);
    node.subgraph?.setDirtyCanvas?.(true, true);
    return true;
}

function applyGraphColors(graph, visited = new Set()) {
    if (!graph || visited.has(graph)) return;
    visited.add(graph);

    for (const node of graph.nodes || graph._nodes || []) {
        applyCmkSubgraphColor(node);
        if (node?.isSubgraphNode?.()) applyGraphColors(node.subgraph, visited);
    }
}

app.registerExtension({
    name: "cmk.subgraph.colors",

    nodeCreated(node) {
        applyCmkSubgraphColor(node);
    },

    loadedGraphNode(node) {
        applyCmkSubgraphColor(node);
    },

    afterConfigureGraph() {
        applyGraphColors(app.rootGraph || app.graph);
    },
});
