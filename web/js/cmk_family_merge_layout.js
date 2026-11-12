import { app } from "../../../scripts/app.js";

const POSTPROCESS_SETUP_TYPE = "CMKPostProcessSetupPipe";
const VISUALIZER_TYPE = "CMKVisualizer";
const FAMILY_BLOCK_STARTS = new Map([
    [POSTPROCESS_SETUP_TYPE, new Set(["MODEL SDXL"])],
]);
// Visualizer rows are matched by input name: the labels "MODEL ZIT" and
// "MODEL SDXL" occur in a generation group and in the POSTPROCESS group.
const VISUALIZER_SDXL_START = "MODEL SDXL";
const VISUALIZER_POSTPROCESS_INPUTS = [
    "POSTPROCESS MODEL ZIT", "MODEL (opt)", "PROCESS", "IMAGE", "LOG", "VISUAL",
];
const POSTPROCESS_PANEL_CLASS = "cmk-postprocess-panel";
const POSTPROCESS_START_CLASS = "cmk-postprocess-start";
const POSTPROCESS_END_CLASS = "cmk-postprocess-end";
const POSTPROCESS_SLOTS_CLASS = "cmk-postprocess-slots";
const CLEAN_VIEW_ATTRIBUTE = "data-cmk-clean-view";
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
    return nodeClass(node) === POSTPROCESS_SETUP_TYPE;
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

function clearPostProcessGroup(root) {
    root.querySelector(`.${POSTPROCESS_PANEL_CLASS}`)?.remove();
    root.querySelector(`.${POSTPROCESS_SLOTS_CLASS}`)?.classList.remove(POSTPROCESS_SLOTS_CLASS);
    for (const slot of root.querySelectorAll(`.${POSTPROCESS_START_CLASS}, .${POSTPROCESS_END_CLASS}`)) {
        slot.classList.remove(POSTPROCESS_START_CLASS, POSTPROCESS_END_CLASS);
    }
}

// Normal view only: section heading, separation and a subdued background for
// the POSTPROCESS inputs. Clean View keeps its own unchanged presentation.
function decorateVisualizer(root, node) {
    const rows = Array.from(root.querySelectorAll(".lg-slot--input"));
    const names = (node.inputs || []).filter((input) => !input?.widget).map((input) => input?.name);
    const cleanView = root.getAttribute(CLEAN_VIEW_ATTRIBUTE) === "true"
        || node?.properties?.cmkCleanView === true;
    rows.forEach((row, index) => {
        row.classList.toggle("cmk-family-block-start", !cleanView && names[index] === VISUALIZER_SDXL_START);
    });
    const first = names.indexOf(VISUALIZER_POSTPROCESS_INPUTS[0]);
    const groupRows = first < 0 ? [] : rows.slice(first, first + VISUALIZER_POSTPROCESS_INPUTS.length);
    const complete = groupRows.length === VISUALIZER_POSTPROCESS_INPUTS.length
        && VISUALIZER_POSTPROCESS_INPUTS.every((name, offset) => names[first + offset] === name);
    const container = groupRows[0]?.parentElement?.parentElement;
    if (cleanView || !complete || !container || root.hasAttribute("data-collapsed")) {
        clearPostProcessGroup(root);
        return;
    }
    for (const slot of root.querySelectorAll(`.${POSTPROCESS_START_CLASS}, .${POSTPROCESS_END_CLASS}`)) {
        if (slot !== groupRows[0]) slot.classList.remove(POSTPROCESS_START_CLASS);
        if (slot !== groupRows.at(-1)) slot.classList.remove(POSTPROCESS_END_CLASS);
    }
    groupRows[0].classList.add(POSTPROCESS_START_CLASS);
    groupRows.at(-1).classList.add(POSTPROCESS_END_CLASS);
    container.classList.add(POSTPROCESS_SLOTS_CLASS);
    let panel = container.querySelector(`:scope > .${POSTPROCESS_PANEL_CLASS}`);
    if (!panel) {
        panel = document.createElement("div");
        panel.className = POSTPROCESS_PANEL_CLASS;
        panel.setAttribute("aria-hidden", "true");
        panel.textContent = "POSTPROCESS";
        container.prepend(panel);
    }
    const last = groupRows.at(-1);
    const top = groupRows[0].offsetTop - 26;
    const height = last.offsetTop + last.offsetHeight - top + 5;
    panel.style.top = `${top}px`;
    panel.style.height = `${height}px`;
}

function decorate() {
    for (const root of document.querySelectorAll(".lg-node[data-node-id]")) {
        const id = root.getAttribute("data-node-id");
        const node = [app.canvas?.graph, app.graph, app.rootGraph]
            .map((graph) => graph?.getNodeById?.(id) ?? graph?.getNodeById?.(Number(id)))
            .find(Boolean);
        if (nodeClass(node) === VISUALIZER_TYPE) {
            decorateVisualizer(root, node);
            continue;
        }
        const blockStarts = FAMILY_BLOCK_STARTS.get(nodeClass(node));
        for (const slot of root.querySelectorAll(".lg-slot--input")) {
            const isBlockStart = blockStarts?.has(slot.textContent.trim()) ?? false;
            slot.classList.toggle("cmk-family-block-start", isBlockStart);
        }
    }
}

app.registerExtension({
    name: "cmk.family_merge_layout",
    setup() {
        if (!document.getElementById("cmk-family-merge-spacing")) {
            const style = document.createElement("style");
            style.id = "cmk-family-merge-spacing";
            style.textContent = `
.lg-node:not([data-collapsed]) .cmk-family-block-start { margin-top: 20px !important; }
.lg-node:not([data-collapsed]):not([data-cmk-clean-view="true"]) .cmk-postprocess-start { margin-top: 36px !important; }
.lg-node:not([data-collapsed]):not([data-cmk-clean-view="true"]) .cmk-postprocess-end { margin-bottom: 10px !important; }
.lg-node:not([data-cmk-clean-view="true"]) .cmk-postprocess-slots { position: relative; z-index: 0; }
.cmk-postprocess-panel {
    position: absolute; left: 4px; right: 4px; z-index: -1; box-sizing: border-box;
    padding: 5px 0 0 10px; pointer-events: none; user-select: none;
    border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 6px;
    background: rgba(255, 255, 255, 0.035);
    color: rgba(214, 219, 226, 0.62); font-size: 10px; font-weight: 600;
    letter-spacing: 0.14em; line-height: 14px; white-space: nowrap;
}
[data-cmk-clean-view="true"] .cmk-postprocess-panel, .lg-node[data-collapsed] .cmk-postprocess-panel { display: none !important; }
`;
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
