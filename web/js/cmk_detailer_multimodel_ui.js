import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// UI of the Multi-Model Detailer generation, also used by the Multi-Model
// Mask Detailer (same Family Enable, family-dependent widgets and NONE
// bypass). The SDXL-only Detailer nodes keep their own extensions
// (cmk_detailer_prepare_labels.js, cmk_smart_detailer_labels.js).
const PREPARE_CLASS = "CMKDetailerPrepareMultiModelPipe";
const DETAILER_CLASS = "CMK_SmartDetailerMultiModelPipe";
const MASK_PREPARE_CLASS = "CMKMaskDetailerPrepareMultiModel";
const MASK_PROCESS_CLASS = "CMKMaskDetailerProcessMultiModel";
const FAMILY_WIDGET = "model_family";

const PREPARE_LABELS = {
    model_family: "MODEL FAMILY",
    sam_model_name: "SAM MODEL",
    sam_device_mode: "SAM DEVICE",
    use_prompt_lora_from_sampler: "USE PROMPT FROM 1ST PASS",
    use_lora_from_1st_pass: "USE LORA FROM 1ST PASS",
    use_1st_pass_sampling: "USE SAMPLING FROM 1ST PASS",
    lora_name: "LORA",
    prompt_pos: "PROMPT POS",
    prompt_neg: "PROMPT NEG",
    steps: "STEPS",
    cfg: "CFG",
    sampling_entry: "SAMPLING ENTRY AT %",
    zit_steps: "STEPS",
    zit_cfg: "CFG",
    zit_sampling_entry: "SAMPLING ENTRY AT %",
    // Same captions as the SDXL widgets, which show their input names.
    zit_sampler: "sampler",
    zit_scheduler: "scheduler",
    zit_model_shift: "MODEL SHIFT",
    freeu_enabled: "USE FREEU",
};

const DETAILER_LABELS = {
    output_image_proceed: "OUTPUT IMAGE PROCEED",
    model_name: "DETECT MODEL",
    bbox_threshold: "BBOX THRESHOLD",
    crop_factor: "CROP FACTOR",
    guide_size: "GUIDE SIZE",
};

const FAMILY_WIDGETS = {
    SDXL: new Set([
        "use_1st_pass_sampling", "lora_name", "strength_model", "strength_clip",
        "prompt_neg", "steps", "cfg", "sampling_entry", "sampler", "scheduler",
        "stop_at_clip_layer", "pag_scale", "sampling", "zsnr", "freeu_enabled",
        "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2",
    ]),
    "Z-Image Turbo": new Set([
        "zit_steps", "zit_cfg", "zit_sampling_entry", "zit_sampler", "zit_scheduler",
        "zit_model_shift",
    ]),
};

const MASK_PREPARE_LABELS = {
    model_family: "MODEL FAMILY",
    fill_mask_holes: "FILL MASK HOLES",
    solid_mask: "SOLID MASK",
    lora_name: "LORA",
    lora_strength: "LORA STRENGTH",
    zit_lora_name: "LORA",
    zit_lora_strength: "LORA STRENGTH",
    prompt_pos: "PROMPT ADDITION",
    steps: "STEPS",
    cfg: "CFG",
    sampling_entry: "SAMPLING ENTRY AT %",
    zit_steps: "STEPS",
    zit_cfg: "CFG",
    zit_sampling_entry: "SAMPLING ENTRY AT %",
    zit_sampler: "sampler",
    zit_scheduler: "scheduler",
    zit_model_shift: "MODEL SHIFT",
};

const MASK_FAMILY_WIDGETS = {
    SDXL: new Set(["lora_name", "lora_strength", "steps", "cfg", "sampling_entry", "sampler", "scheduler"]),
    "Z-Image Turbo": new Set([
        "zit_lora_name", "zit_lora_strength", "zit_steps", "zit_cfg", "zit_sampling_entry", "zit_sampler",
        "zit_scheduler", "zit_model_shift",
    ]),
};

// Per Prepare class its captions and family widget sets; per execution class
// its captions and the input that carries the Prepare.
const PREPARE_SPECS = {
    [PREPARE_CLASS]: { labels: PREPARE_LABELS, families: FAMILY_WIDGETS },
    [MASK_PREPARE_CLASS]: { labels: MASK_PREPARE_LABELS, families: MASK_FAMILY_WIDGETS },
};
const EXECUTION_SPECS = {
    [DETAILER_CLASS]: { labels: DETAILER_LABELS, input: "DETAILER" },
    [MASK_PROCESS_CLASS]: {
        labels: { mask_expand: "MASK EXPAND", mask_feather: "MASK FEATHER" },
        input: "MASK DETAILER PROCESS",
    },
};

// CMK Detailer Family Enable: one control per Detailer. Three choice fields
// replace the dropdown; the stored value stays the model family name.
const FAMILY_CONTROL_CLASS = "CMKDetailerFamilyEnable";
const ADVANCED_PREPARE_CLASS = "CMKDetailerPrepareAdvancedMultiModelPipe";
const FAMILY_CHOICES = [["None", "NONE"], ["SDXL", "SDXL"], ["Z-Image Turbo", "ZIT"]];
const FAMILY_TABS_NAME = "cmk_family_choice";
const FAMILY_ACCENT = "rgb(88, 190, 230)";

const previewNodes = new Set();

function nodeClass(node) {
    return node?.comfyClass ?? node?.type ?? node?.constructor?.comfyClass ?? node?.constructor?.nodeData?.name;
}

function setWidgetVisible(target, visible) {
    if (!target) return;
    target.options ??= {};
    target._cmkMultiModelVisibility ??= {
        computeSize: target.computeSize,
        draw: target.draw,
        hidden: target.hidden,
        optionsHidden: target.options.hidden,
    };
    const original = target._cmkMultiModelVisibility;
    if (visible) {
        target.computeSize = original.computeSize;
        target.draw = original.draw;
        target.hidden = original.hidden;
        target.options.hidden = original.optionsHidden;
        return;
    }
    // Keep the widget (and its value) in node.widgets: it is still part of
    // the prompt. Vue nodes filter options.hidden; the canvas gets no row.
    target.computeSize = () => [0, -4];
    target.draw = () => {};
    target.hidden = true;
    target.options.hidden = true;
}

function applyLabels(node, labels) {
    for (const target of node.widgets ?? []) {
        const label = labels[target?.name];
        if (label) target.label = label;
    }
}

function familyValue(node) {
    const value = node.widgets?.find((target) => target?.name === FAMILY_WIDGET)?.value;
    return value === "SDXL" || value === "Z-Image Turbo" ? value : "None";
}

// Switching families only hides widgets; no value is reset and, like
// FaceProcess, the node keeps its size (the prompt field takes the space).
// None is the complete bypass: no Prepare setting has an effect, so only the
// MODEL FAMILY row remains.
function refreshPrepare(node) {
    if (!Array.isArray(node.widgets)) return;
    const spec = PREPARE_SPECS[nodeClass(node)];
    applyLabels(node, spec.labels);
    const family = familyValue(node);
    const active = spec.families[family] ?? new Set();
    const specific = new Set(Object.values(spec.families).flatMap((names) => [...names]));
    for (const target of node.widgets) {
        const visible = target?.name === FAMILY_WIDGET
            || (family !== "None" && (!specific.has(target?.name) || active.has(target.name)));
        setWidgetVisible(target, visible);
    }
    for (const detailer of linkedDetailers(node)) refreshDetailer(detailer);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function linkOf(graph, id) {
    return graph?.links?.get?.(id) ?? graph?.links?.[id];
}

function linkedDetailers(node) {
    const graph = node.graph ?? app.graph;
    return (node.outputs ?? []).flatMap((output) => output?.links ?? [])
        .map((id) => linkOf(graph, id))
        .map((link) => (link ? graph.getNodeById?.(link.target_id) : null))
        .filter((target) => nodeClass(target) in EXECUTION_SPECS);
}

// The Smart Detailer has no switch of its own: its family is the one of the
// Prepare (Advanced: of the Detailer slot) feeding DETAILER, or null when the
// input is not linked to one.
function detailerFamily(node) {
    const name = EXECUTION_SPECS[nodeClass(node)].input;
    const input = node.inputs?.find((item) => item?.name === name && item.link != null);
    const graph = node.graph ?? app.graph;
    const link = input ? linkOf(graph, input.link) : null;
    const source = link ? graph.getNodeById?.(link.origin_id) : null;
    const type = nodeClass(source);
    if (type in PREPARE_SPECS) return familyValue(source);
    if (type !== ADVANCED_PREPARE_CLASS) return null;
    const value = source.widgets?.find((item) => item?.name === `${FAMILY_WIDGET}_${Number(link.origin_slot) + 1}`)?.value;
    return value === "SDXL" || value === "Z-Image Turbo" ? value : "None";
}

// NONE bypasses the Detailer completely: no widget has an effect, so none is
// shown (values stay and are serialized; the node keeps its size).
function refreshDetailer(node) {
    if (!Array.isArray(node.widgets)) return;
    applyLabels(node, EXECUTION_SPECS[nodeClass(node)].labels);
    const bypass = detailerFamily(node) === "None";
    for (const target of node.widgets) setWidgetVisible(target, !bypass);
    node.setDirtyCanvas?.(true, true);
}

function installFamilyCallback(node) {
    const family = node.widgets?.find((target) => target?.name === FAMILY_WIDGET);
    if (!family || family._cmkMultiModelCallback) return;
    family._cmkMultiModelCallback = true;
    const original = family.callback;
    family.callback = function () {
        const result = original?.apply(this, arguments);
        refreshPrepare(node);
        return result;
    };
}

// Inside a subgraph MODEL FAMILY is linked to a subgraph input, so the inner
// widget keeps a stale value while the prompt uses the outer one. Mirror the
// outer choice into the inner Prepare so the inner view shows the active set.
// Instances share one definition, so the entered instance is mirrored again
// when the canvas opens its subgraph (subgraph-opened, detail.fromNode).
function innerPrepares(host) {
    const inner = host?.subgraph;
    const nodes = inner?.nodes ?? inner?._nodes ?? [];
    return nodes.filter((node) => nodeClass(node) in PREPARE_SPECS).map((node) => [inner, node]);
}

function hostFamilyWidget(host, inner, prepare) {
    const input = prepare.inputs?.find((item) => item?.name === FAMILY_WIDGET && item.link != null);
    if (!input) return null;
    const link = inner.links?.get?.(input.link) ?? inner.links?.[input.link];
    if (!link || Number(link.origin_id) !== -10) return null;
    const name = inner.inputs?.[link.origin_slot]?.name;
    return host.widgets?.find((item) => item?.name === name) ?? null;
}

function syncHostFamily(host) {
    for (const [inner, prepare] of innerPrepares(host)) {
        const hostWidget = hostFamilyWidget(host, inner, prepare);
        const widget = prepare.widgets?.find((item) => item?.name === FAMILY_WIDGET);
        if (!hostWidget || !widget) continue;
        widget.value = hostWidget.value;
        refreshPrepare(prepare);
        if (!hostWidget._cmkMultiModelHost) {
            hostWidget._cmkMultiModelHost = true;
            const original = hostWidget.callback;
            hostWidget.callback = function () {
                const result = original?.apply(this, arguments);
                syncHostFamily(host);
                return result;
            };
        }
    }
}

function familyInputLinked(node) {
    return Boolean(node.inputs?.some((input) => input?.widget?.name === FAMILY_WIDGET && input.link != null));
}

// A Prepare whose MODEL FAMILY is linked to the control shows the control's
// family (its own dropdown is hidden by ComfyUI and would be stale).
function mirrorToPrepares(node) {
    const value = node.widgets?.find((target) => target?.name === FAMILY_WIDGET)?.value;
    const graph = node.graph ?? app.graph;
    for (const linkId of node.outputs?.[1]?.links ?? []) {
        const link = graph?.links?.get?.(linkId) ?? graph?.links?.[linkId];
        const target = link ? graph.getNodeById?.(link.target_id) : null;
        const type = nodeClass(target);
        if (!(type in PREPARE_SPECS) && type !== ADVANCED_PREPARE_CLASS) continue;
        // Standard: model_family; Advanced: model_family_1/2/3 of the linked input.
        const name = target.inputs?.[link.target_slot]?.widget?.name ?? FAMILY_WIDGET;
        const widget = target.widgets?.find((item) => item?.name === name);
        if (widget && value != null && widget.value !== value) widget.value = value;
        if (type in PREPARE_SPECS) refreshPrepare(target);
        else target._cmkRefreshAdvanced?.();
    }
}

function renderFamilyChoice(root) {
    for (const button of root.querySelectorAll("button")) {
        const active = button.dataset.value === root._cmkValue;
        button.setAttribute("aria-pressed", active ? "true" : "false");
        button.style.color = active ? FAMILY_ACCENT : "rgba(214, 219, 226, 0.5)";
        button.style.border = active ? "1px solid rgba(88, 190, 230, 0.6)" : "1px solid rgba(255, 255, 255, 0.09)";
        button.style.fontWeight = active ? "700" : "500";
    }
}

function installFamilyChoice(node) {
    const original = node.widgets?.find((target) => target?.name === FAMILY_WIDGET);
    if (!original || typeof node.addDOMWidget !== "function") return;
    let panel = node.widgets.find((target) => target?.name === FAMILY_TABS_NAME);
    if (!panel) {
        const root = document.createElement("div");
        root.className = "cmk-family-choice";
        Object.assign(root.style, {
            display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "8px",
            width: "100%", boxSizing: "border-box", padding: "2px 0",
        });
        for (const [value, label] of FAMILY_CHOICES) {
            const button = document.createElement("button");
            button.type = "button";
            button.dataset.value = value;
            button.textContent = label;
            button.title = value;
            Object.assign(button.style, {
                height: "40px", borderRadius: "6px", cursor: "pointer", appearance: "none",
                background: "rgba(255, 255, 255, 0.025)", boxShadow: "none", outline: "none",
                fontFamily: "inherit", fontSize: "13px", letterSpacing: "0.08em",
                transition: "color 120ms ease, border-color 120ms ease",
            });
            button.addEventListener("pointerdown", (event) => event.stopPropagation());
            button.addEventListener("click", (event) => {
                event.stopPropagation();
                if (root._cmkValue === value) return;
                root._cmkValue = value;
                original.value = value;
                renderFamilyChoice(root);
                try { original.callback?.(value, app.canvas, node); } catch (_) {}
                mirrorToPrepares(node);
                node.setDirtyCanvas?.(true, true);
                app.graph?.setDirtyCanvas?.(true, true);
            });
            root.append(button);
        }
        root._cmkValue = String(original.value ?? "None");
        panel = node.addDOMWidget(FAMILY_TABS_NAME, "cmk_family_choice", root, {
            serialize: false,
            hideOnZoom: false,
            getMinHeight: () => 46,
            getHeight: () => 46,
            getValue: () => root._cmkValue,
            setValue: (value) => { root._cmkValue = String(value ?? "None"); renderFamilyChoice(root); },
        });
        panel.serialize = false;
        panel.computeLayoutSize = undefined;
        panel._cmkRoot = root;
    }
    const root = panel._cmkRoot;
    root._cmkValue = String(original.value ?? "None");
    renderFamilyChoice(root);
    // A linked selection (Standard subgraph: MODEL FAMILY from the outer
    // widget) is decided elsewhere; ComfyUI hides the dropdown, so do the same.
    setWidgetVisible(original, false);
    setWidgetVisible(panel, !familyInputLinked(node));
    mirrorToPrepares(node);
    node.setDirtyCanvas?.(true, true);
}

function refresh(node) {
    if (node?.subgraph) {
        syncHostFamily(node);
        return;
    }
    const type = nodeClass(node);
    if (type in PREPARE_SPECS) {
        installFamilyCallback(node);
        refreshPrepare(node);
    } else if (type === FAMILY_CONTROL_CLASS) {
        installFamilyChoice(node);
    } else if (type in EXECUTION_SPECS) {
        node._cmkRefreshDetailer = () => refreshDetailer(node);
        refreshDetailer(node);
    }
}

// Late passes cover widgets whose labels the frontend re-creates after creation.
function schedule(node) {
    for (const delay of [0, 50, 200]) setTimeout(() => refresh(node), delay);
}

function track(node) {
    if (nodeClass(node) === DETAILER_CLASS) previewNodes.add(node);
    schedule(node);
}

function clearPreviousPreviewsAtWorkflowStart() {
    for (const node of previewNodes) {
        node.imgs = [];
        node.imageIndex = null;
        node.setDirtyCanvas?.(true, true);
    }
}

// MODEL ZIT is optional in the backend contract but belongs above MODEL SDXL;
// like CMK Image Resize, list it first and keep it optional.
function registerModelZitFirst(nodeData) {
    const required = nodeData?.input?.required;
    const optional = nodeData?.input?.optional;
    const modelZit = optional?.["MODEL ZIT"];
    if (!required || !modelZit) return;
    const [type, options = {}] = modelZit;
    nodeData.input.required = { "MODEL ZIT": [type, { ...options, isOptional: true }], ...required };
    delete optional["MODEL ZIT"];
    delete nodeData.input_order;
}

// The input MODEL was renamed to MODEL SDXL and MODEL ZIT moved to the top.
// Saved inputs are restored by position, so saved workflows and subgraph
// definitions are migrated on the graph data, links included, before load.
const SOCKET_ORDER = ["MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "LoRA ZIT"];

function migratePrepareInputs(node) {
    if (node?.type !== PREPARE_CLASS || !Array.isArray(node.inputs)) return null;
    const inputs = node.inputs.map((input) => (
        input?.name === "MODEL" && !node.inputs.some((item) => item?.name === "MODEL SDXL")
            ? { ...input, name: "MODEL SDXL", localized_name: "MODEL SDXL", label: undefined }
            : input
    ));
    const sockets = SOCKET_ORDER.map((name) => inputs.find((input) => input?.name === name)).filter(Boolean);
    const ordered = [...sockets, ...inputs.filter((input) => !SOCKET_ORDER.includes(input?.name))];
    if (ordered.every((input, index) => input === node.inputs[index])) return null;
    node.inputs = ordered;
    return new Map(ordered.flatMap((input, index) => (input?.link != null ? [[input.link, index]] : [])));
}

function retargetLinks(links, nodeId, slots) {
    for (const link of links ?? []) {
        if (Array.isArray(link)) {
            if (String(link[3]) === String(nodeId) && slots.has(link[0])) link[4] = slots.get(link[0]);
        } else if (link && String(link.target_id) === String(nodeId) && slots.has(link.id)) {
            link.target_slot = slots.get(link.id);
        }
    }
}

function migrateGraphData(graph) {
    for (const node of graph?.nodes ?? []) {
        const slots = migratePrepareInputs(node);
        if (slots) retargetLinks(graph.links, node.id, slots);
    }
}

const TRACKED_CLASSES = new Set([...Object.keys(PREPARE_SPECS), ...Object.keys(EXECUTION_SPECS), FAMILY_CONTROL_CLASS]);

app.registerExtension({
    name: "cmk.detailer.multimodel.ui.v1",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (![...TRACKED_CLASSES].includes(nodeData.name)) return;
        if (nodeData.name in PREPARE_SPECS) registerModelZitFirst(nodeData);
        for (const hook of ["onNodeCreated", "onConfigure", "onAdded", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                schedule(this);
                return result;
            };
        }
        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            previewNodes.delete(this);
            return originalRemoved?.apply(this, arguments);
        };
    },

    nodeCreated(node) {
        if (TRACKED_CLASSES.has(nodeClass(node)) || innerPrepares(node).length) track(node);
    },

    loadedGraphNode(node) {
        if (TRACKED_CLASSES.has(nodeClass(node)) || innerPrepares(node).length) track(node);
    },

    beforeConfigureGraph(graphData) {
        migrateGraphData(graphData);
        for (const definition of graphData?.definitions?.subgraphs ?? []) migrateGraphData(definition);
    },

    setup() {
        api.addEventListener("execution_start", clearPreviousPreviewsAtWorkflowStart);
        app.canvas?.canvas?.addEventListener("subgraph-opened", (event) => {
            const host = event.detail?.fromNode;
            if (innerPrepares(host).length) syncHostFamily(host);
        });
    },
});
