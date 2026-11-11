import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// UI of the Multi-Model Detailer generation. The SDXL-only Detailer nodes
// keep their own extensions (cmk_detailer_prepare_labels.js,
// cmk_smart_detailer_labels.js).
const PREPARE_CLASS = "CMKDetailerPrepareMultiModelPipe";
const DETAILER_CLASS = "CMK_SmartDetailerMultiModelPipe";
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
    enable: "ENABLE",
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
const FAMILY_SPECIFIC = new Set([...FAMILY_WIDGETS.SDXL, ...FAMILY_WIDGETS["Z-Image Turbo"]]);

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
function refreshPrepare(node) {
    if (!Array.isArray(node.widgets)) return;
    applyLabels(node, PREPARE_LABELS);
    const active = FAMILY_WIDGETS[familyValue(node)] ?? new Set();
    for (const target of node.widgets) {
        setWidgetVisible(target, !FAMILY_SPECIFIC.has(target?.name) || active.has(target.name));
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
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
    return nodes.filter((node) => nodeClass(node) === PREPARE_CLASS).map((node) => [inner, node]);
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

function refresh(node) {
    if (node?.subgraph) {
        syncHostFamily(node);
        return;
    }
    const type = nodeClass(node);
    if (type === PREPARE_CLASS) {
        installFamilyCallback(node);
        refreshPrepare(node);
    } else if (type === DETAILER_CLASS && Array.isArray(node.widgets)) {
        applyLabels(node, DETAILER_LABELS);
        node.setDirtyCanvas?.(true, true);
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

app.registerExtension({
    name: "cmk.detailer.multimodel.ui.v1",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== PREPARE_CLASS && nodeData.name !== DETAILER_CLASS) return;
        if (nodeData.name === PREPARE_CLASS) registerModelZitFirst(nodeData);
        for (const hook of ["onNodeCreated", "onConfigure", "onAdded"]) {
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
        const type = nodeClass(node);
        if (type === PREPARE_CLASS || type === DETAILER_CLASS || innerPrepares(node).length) track(node);
    },

    loadedGraphNode(node) {
        const type = nodeClass(node);
        if (type === PREPARE_CLASS || type === DETAILER_CLASS || innerPrepares(node).length) track(node);
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
