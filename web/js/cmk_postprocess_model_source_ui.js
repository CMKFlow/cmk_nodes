import { app } from "../../../scripts/app.js";

// CMK PostProcess Model (2.7.1): one node, two model families. The settings
// are shown in two separated sections that line up with the outputs
// MODEL SDXL and MODEL ZIT. Presentation only: widget names, values and the
// serialized order are those of the backend definition.
// CMK Flow · 30 PostProcess Setup v2 (2.8) has the same settings under the
// names of the Setup and shows them the same way.
const NODE_SPECS = {
    CMKPostProcessModelSourcePipe: {
        labels: {
            ckpt_name: "CHECKPOINT",
            vae_name: "VAE",
            checkpoint_vae: "CHECKPOINT VAE",
            diffusion_model: "DIFFUSION MODEL",
            text_encoder: "TEXT ENCODER",
            vae: "VAE",
            weight_dtype: "WEIGHT DTYPE",
            clip_device: "CLIP DEVICE",
        },
        sections: [
            { title: "SDXL", first: "ckpt_name" },
            { title: "Z-IMAGE TURBO", first: "diffusion_model" },
        ],
    },
    CMKPostProcessSetupV2Pipe: {
        labels: {
            postprocess_checkpoint: "CHECKPOINT",
            postprocess_vae: "VAE",
            use_checkpoint_vae: "CHECKPOINT VAE",
            postprocess_zit_model: "DIFFUSION MODEL",
            postprocess_zit_text_encoder: "TEXT ENCODER",
            postprocess_zit_vae: "VAE",
            postprocess_zit_weight_dtype: "WEIGHT DTYPE",
            postprocess_zit_clip_device: "CLIP DEVICE",
        },
        sections: [
            { title: "SDXL", first: "postprocess_checkpoint" },
            { title: "Z-IMAGE TURBO", first: "postprocess_zit_model" },
        ],
    },
};
const STYLE_ID = "cmk-postprocess-model-source-style";
const SECTION_CLASS = "cmk-pp-model-section";
const GRID_CLASS = "cmk-pp-model-grid";

function nodeClass(node) {
    return node?.comfyClass ?? node?.constructor?.comfyClass ?? node?.type;
}

function applyLabels(node) {
    const labels = NODE_SPECS[nodeClass(node)]?.labels ?? {};
    for (const widget of node.widgets ?? []) {
        const label = labels[widget?.name];
        if (label) widget.label = label;
    }
    node.setDirtyCanvas?.(true, true);
}

function graphNode(id) {
    return [app.canvas?.graph, app.graph, app.rootGraph]
        .map((graph) => graph?.getNodeById?.(id) ?? graph?.getNodeById?.(Number(id)))
        .find(Boolean);
}

function rowWidgetName(row) {
    return row.querySelector("[data-widget-name]")?.getAttribute("data-widget-name") ?? null;
}

function decorate() {
    for (const root of document.querySelectorAll(".lg-node[data-node-id]")) {
        const node = graphNode(root.getAttribute("data-node-id"));
        const spec = NODE_SPECS[nodeClass(node)];
        if (!spec) continue;
        const grid = root.querySelector(".lg-node-widgets");
        const rows = Array.from(grid?.children ?? []).filter((row) => rowWidgetName(row));
        if (!grid || !rows.length) continue;
        grid.classList.add(GRID_CLASS);
        const starts = spec.sections.map((section) => rows.findIndex((row) => rowWidgetName(row) === section.first));
        rows.forEach((row) => {
            row.classList.remove(SECTION_CLASS);
            row.removeAttribute("data-cmk-section");
        });
        starts.forEach((start, index) => {
            if (start < 0) return;
            const end = index + 1 < starts.length && starts[index + 1] > start ? starts[index + 1] - 1 : rows.length - 1;
            const row = rows[start];
            row.classList.add(SECTION_CLASS);
            row.setAttribute("data-cmk-section", spec.sections[index].title);
            const last = rows[end];
            const height = last.offsetTop + last.offsetHeight - row.offsetTop;
            row.style.setProperty("--cmk-pp-section-height", `${height}px`);
        });
    }
}

function installStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.${GRID_CLASS} { position: relative; z-index: 0; }
.${GRID_CLASS} > .${SECTION_CLASS} { position: relative; margin-top: 34px; }
.${GRID_CLASS} > .${SECTION_CLASS}:first-child { margin-top: 22px; }
.${GRID_CLASS} > .${SECTION_CLASS}::before {
    content: attr(data-cmk-section); position: absolute; left: 14px; top: -19px;
    color: rgba(214, 219, 226, 0.62); font-size: 10px; font-weight: 600;
    letter-spacing: 0.14em; line-height: 14px; white-space: nowrap; pointer-events: none;
}
.${GRID_CLASS} > .${SECTION_CLASS}::after {
    content: ""; position: absolute; z-index: -1; left: 4px; right: -8px; top: -22px;
    height: calc(var(--cmk-pp-section-height, 0px) + 27px); box-sizing: border-box;
    border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 6px;
    background: rgba(255, 255, 255, 0.035); pointer-events: none;
}
`;
    document.head.appendChild(style);
}

function schedule(node) {
    for (const delay of [0, 50, 200]) setTimeout(() => { applyLabels(node); decorate(); }, delay);
}

app.registerExtension({
    name: "cmk.postprocess_model_source.ui.v1",

    setup() {
        installStyle();
        new MutationObserver(decorate).observe(document.documentElement, { childList: true, subtree: true });
    },

    nodeCreated(node) {
        if (NODE_SPECS[nodeClass(node)]) schedule(node);
    },

    loadedGraphNode(node) {
        if (NODE_SPECS[nodeClass(node)]) schedule(node);
    },
});
