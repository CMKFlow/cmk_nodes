import { app } from "../../../scripts/app.js";

// CMK Detailer Prepare Advanced Multi-Model -Pipe- (2.8.0): one shared
// section and one section per Detailer. Each Detailer shows only the
// settings of its model family. Presentation only: widget names, values and
// the serialized order are those of the backend definition.
const NODE_CLASS = "CMKDetailerPrepareAdvancedMultiModelPipe";
const SLOTS = [1, 2, 3];
const SHARED = ["sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler"];
const SDXL_SHARED = ["stop_at_clip_layer", "pag_scale", "sampling", "zsnr", "freeu_enabled",
    "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2"];
const SDXL_SLOT = ["steps", "cfg", "sampling_entry", "sampler", "scheduler"];
const ZIT_SLOT = ["zit_steps", "zit_cfg", "zit_sampling_entry", "zit_sampler", "zit_scheduler", "zit_model_shift"];
const LABELS = {
    sam_model_name: "SAM MODEL",
    sam_device_mode: "SAM DEVICE",
    use_prompt_lora_from_sampler: "USE PROMPT FROM 1ST PASS",
    detailer_global_enable: "ENABLE",
    model_family: "MODEL FAMILY",
    steps: "STEPS", cfg: "CFG", sampling_entry: "SAMPLING ENTRY AT %", sampler: "sampler", scheduler: "scheduler",
    zit_steps: "STEPS", zit_cfg: "CFG", zit_sampling_entry: "SAMPLING ENTRY AT %", zit_sampler: "sampler",
    zit_scheduler: "scheduler", zit_model_shift: "MODEL SHIFT",
    stop_at_clip_layer: "CLIP SKIP (SDXL)", pag_scale: "PAG SCALE (SDXL)", sampling: "SAMPLING (SDXL)",
    zsnr: "ZSNR (SDXL)", freeu_enabled: "USE FREEU (SDXL)", freeu_b1: "FREEU B1", freeu_b2: "FREEU B2",
    freeu_s1: "FREEU S1", freeu_s2: "FREEU S2",
};
const STYLE_ID = "cmk-detailer-advanced-style";
const SECTION_CLASS = "cmk-advanced-section";
const GRID_CLASS = "cmk-advanced-grid";

const slotName = (name, slot) => `${name}_${slot}`;
const baseName = (name) => name.replace(/_[123]$/, "");

function nodeClass(node) {
    return node?.comfyClass ?? node?.constructor?.comfyClass ?? node?.type;
}

function widget(node, name) {
    return node.widgets?.find((item) => item?.name === name);
}

function family(node, slot) {
    const value = widget(node, slotName("model_family", slot))?.value;
    return value === "SDXL" || value === "Z-Image Turbo" ? value : "None";
}

function setVisible(target, visible) {
    if (!target) return;
    target.options ??= {};
    target._cmkAdvancedVisibility ??= {
        computeSize: target.computeSize, draw: target.draw, hidden: target.hidden, optionsHidden: target.options.hidden,
    };
    const original = target._cmkAdvancedVisibility;
    if (visible) {
        target.computeSize = original.computeSize;
        target.draw = original.draw;
        target.hidden = original.hidden;
        target.options.hidden = original.optionsHidden;
        return;
    }
    target.computeSize = () => [0, -4];
    target.draw = () => {};
    target.hidden = true;
    target.options.hidden = true;
}

function linked(node, name) {
    return Boolean(node.inputs?.some((input) => input?.widget?.name === name && input.link != null));
}

export function refreshAdvanced(node) {
    if (!Array.isArray(node?.widgets)) return;
    const families = SLOTS.map((slot) => family(node, slot));
    for (const target of node.widgets) {
        const name = target?.name ?? "";
        const label = LABELS[baseName(name)];
        if (label) target.label = label;
        const slot = Number(name.match(/_([123])$/)?.[1]);
        let visible = true;
        if (slot && name !== slotName("model_family", slot)) {
            const selected = families[slot - 1];
            if (SDXL_SLOT.includes(baseName(name))) visible = selected === "SDXL";
            else if (ZIT_SLOT.includes(baseName(name))) visible = selected === "Z-Image Turbo";
        } else if (slot) {
            // A family linked to a CMK Detailer Family Enable is decided there.
            visible = !linked(node, name);
        } else if (SDXL_SHARED.includes(name)) {
            visible = families.includes("SDXL");
        }
        setVisible(target, visible);
    }
    // The node is first laid out with every widget visible and grows to that
    // height; return to the remembered size once the family sets are hidden.
    const manual = node.properties?.cmkManualSize;
    if (Array.isArray(manual) && Number(manual[1]) > 0 && node.size?.[1] > Number(manual[1])) {
        node.setSize?.([node.size[0], Number(manual[1]) - 30]);
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    queueMicrotask(decorate);
}

function sections() {
    return [
        { title: "SHARED", names: [...SHARED, ...SDXL_SHARED] },
        ...SLOTS.map((slot) => ({
            title: `DETAILER ${slot}`,
            names: [slotName("model_family", slot), ...[...SDXL_SLOT, ...ZIT_SLOT].map((name) => slotName(name, slot))],
            slot,
        })),
    ];
}

function graphNode(id) {
    return [app.canvas?.graph, app.graph, app.rootGraph]
        .map((graph) => graph?.getNodeById?.(id) ?? graph?.getNodeById?.(Number(id)))
        .find(Boolean);
}

function rowName(row) {
    return row.querySelector("[data-widget-name]")?.getAttribute("data-widget-name") ?? null;
}

function decorate() {
    for (const root of document.querySelectorAll(".lg-node[data-node-id]")) {
        const node = graphNode(root.getAttribute("data-node-id"));
        if (nodeClass(node) !== NODE_CLASS) continue;
        const grid = root.querySelector(".lg-node-widgets");
        const rows = Array.from(grid?.children ?? []).filter((row) => rowName(row));
        if (!grid || !rows.length) continue;
        grid.classList.add(GRID_CLASS);
        for (const row of rows) {
            row.classList.remove(SECTION_CLASS);
            row.removeAttribute("data-cmk-section");
        }
        for (const section of sections()) {
            const members = rows.filter((row) => section.names.includes(rowName(row)));
            if (!members.length) continue;
            const first = members[0];
            const last = members.at(-1);
            let title = section.title;
            if (section.slot) {
                const selected = family(node, section.slot);
                title += ` · ${selected === "Z-Image Turbo" ? "ZIT" : selected.toUpperCase()}`;
            }
            first.classList.add(SECTION_CLASS);
            first.setAttribute("data-cmk-section", title);
            first.style.setProperty("--cmk-advanced-section-height", `${last.offsetTop + last.offsetHeight - first.offsetTop}px`);
        }
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
    height: calc(var(--cmk-advanced-section-height, 0px) + 27px); box-sizing: border-box;
    border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 6px;
    background: rgba(255, 255, 255, 0.035); pointer-events: none;
}
`;
    document.head.appendChild(style);
}

function install(node) {
    node._cmkRefreshAdvanced = () => refreshAdvanced(node);
    for (const slot of SLOTS) {
        const target = widget(node, slotName("model_family", slot));
        if (!target || target._cmkAdvancedCallback) continue;
        target._cmkAdvancedCallback = true;
        const original = target.callback;
        target.callback = function () {
            const result = original?.apply(this, arguments);
            refreshAdvanced(node);
            return result;
        };
    }
    refreshAdvanced(node);
}

function schedule(node) {
    for (const delay of [0, 50, 200]) setTimeout(() => install(node), delay);
}

app.registerExtension({
    name: "cmk.detailer.advanced.ui.v1",

    setup() {
        installStyle();
        new MutationObserver(decorate).observe(document.documentElement, { childList: true, subtree: true });
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;
        for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                schedule(this);
                return result;
            };
        }
    },

    nodeCreated(node) {
        if (nodeClass(node) === NODE_CLASS) schedule(node);
    },

    loadedGraphNode(node) {
        if (nodeClass(node) === NODE_CLASS) schedule(node);
    },
});
