import { app } from "../../../scripts/app.js";

// FaceProcess 2.8 operating mode (F1). Presentation only: stored and sent
// COMBO values stay "None" / "Restore" / "SDXL" / "Z-Image Turbo".
//
// - CMK FaceProcess Mode Enable: four choice fields NONE | RESTORE | SDXL | ZIT
//   replace the dropdown, styled like CMK Detailer Family Enable.
// - Outer FaceProcess subgraphs: the mode dropdown is labelled MODE,
//   right-aligned, with short option labels (getOptionLabel).
// - CMK FaceProcess Prepare Multi-Model / Multi-Model -Pipe-: CMK labels and
//   only the widgets the operating mode uses, like the Multi-Model Detailer
//   Prepare (values stay; hidden widgets are still serialised).
// The Detailer controls keep their own extensions and are not touched here.
export const MODE_CONTROL_CLASS = "CMKFaceProcessModeEnable";
export const MODE_WIDGET = "process_mode";
export const MODE_CHOICES = [["None", "NONE"], ["Restore", "RESTORE"], ["SDXL", "SDXL"], ["Z-Image Turbo", "ZIT"]];

// Subgraph name -> outer widget and its option labels. The widget names are
// those the F5 subgraphs promote (Standard: the Mode Enable selection;
// Advanced: the global switch of the FaceProcess Advanced Prepare).
export const OUTER_MODES = {
    "CMK Flow · FaceProcess": {
        widget: MODE_WIDGET,
        labels: { None: "DISABLE", Restore: "RESTORE", SDXL: "SDXL", "Z-Image Turbo": "ZIT" },
    },
    "CMK Flow · FaceProcess · Advanced": {
        widget: "faceprocess_global_enable",
        labels: { ENABLE: "ENABLE", DISABLE: "DISABLE" },
    },
};

export const PREPARE_CLASS = "CMKFaceProcessPrepareMultiModelPipe";
export const EXECUTE_CLASS = "CMKFaceProcessMultiModelPipe";

// Same captions as CMK Detailer Prepare Multi-Model (cmk_detailer_multimodel_ui.js).
const PREPARE_LABELS = {
    process_mode: "PROCESS MODE",
    refine_mode: "REFINE MODE",
    restore_model: "RESTORE MODEL",
    restore_visibility: "VISIBILITY",
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
    zit_sampler: "sampler",
    zit_scheduler: "scheduler",
    zit_model_shift: "MODEL SHIFT",
    freeu_enabled: "USE FREEU",
};
// Widgets per operating mode: Refine for every active mode, the restore
// model and its visibility for RESTORE; NONE shows nothing.
const DIFFUSION_SHARED = ["refine_mode", "sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler", "use_lora_from_1st_pass", "prompt_pos"];
const PREPARE_WIDGETS = {
    Restore: new Set(["refine_mode", "restore_model", "restore_visibility"]),
    SDXL: new Set([...DIFFUSION_SHARED,
        "use_1st_pass_sampling", "lora_name", "strength_model", "strength_clip", "prompt_neg", "steps", "cfg",
        "sampling_entry", "sampler", "scheduler", "stop_at_clip_layer", "pag_scale", "sampling", "zsnr",
        "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2"]),
    "Z-Image Turbo": new Set([...DIFFUSION_SHARED,
        "zit_steps", "zit_cfg", "zit_sampling_entry", "zit_sampler", "zit_scheduler", "zit_model_shift"]),
};

const EXECUTE_LABELS = {
    output_image_proceed: "OUTPUT IMAGE PROCEED",
    select_face: "SELECT FACE",
    detect_model: "DETECT MODEL",
    detect_bbox_threshold: "BBOX THRESHOLD",
    detect_crop_factor: "CROP FACTOR",
    detail_guide_size: "GUIDE SIZE",
};
const EXECUTE_COMMON = ["output_image_proceed", "select_face", "detect_model", "detect_bbox_threshold", "detect_bbox_dilation",
    "detect_crop_factor", "detect_drop_size"];
const DETAIL_WIDGETS = ["detail_guide_size", "detail_guide_size_for", "detail_max_size", "detail_noise_mask",
    "detail_force_inpaint", "detail_paste_feather"];
const EXECUTE_WIDGETS = {
    Restore: new Set(EXECUTE_COMMON),
    SDXL: new Set([...EXECUTE_COMMON, ...DETAIL_WIDGETS]),
    "Z-Image Turbo": new Set([...EXECUTE_COMMON, ...DETAIL_WIDGETS]),
};

const CHOICE_NAME = "cmk_faceprocess_mode_choice";
const ACCENT = "rgb(88, 190, 230)";
const ROW_ATTRIBUTE = "data-cmk-faceprocess-mode-select";
const STYLE_ID = "cmk-faceprocess-mode-style";

function nodeClass(node) {
    return node?.comfyClass ?? node?.type ?? node?.constructor?.comfyClass ?? node?.constructor?.nodeData?.name;
}

function setWidgetVisible(target, visible) {
    if (!target) return;
    target.options ??= {};
    target._cmkModeVisibility ??= {
        computeSize: target.computeSize,
        draw: target.draw,
        hidden: target.hidden,
        optionsHidden: target.options.hidden,
    };
    const original = target._cmkModeVisibility;
    if (visible) {
        target.computeSize = original.computeSize;
        target.draw = original.draw;
        target.hidden = original.hidden;
        target.options.hidden = original.optionsHidden;
        return;
    }
    // The widget stays in node.widgets: its value is part of the prompt.
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

function graphOf(node) {
    return node.graph ?? app.graph;
}

function linkOf(graph, id) {
    return graph?.links?.get?.(id) ?? graph?.links?.[id];
}

function linkedTargets(node, outputName, type) {
    const graph = graphOf(node);
    const output = node.outputs?.find((item) => item?.name === outputName);
    return (output?.links ?? [])
        .map((id) => linkOf(graph, id))
        .filter(Boolean)
        .map((link) => [graph.getNodeById?.(link.target_id), link])
        .filter(([target]) => nodeClass(target) === type);
}

function prepareMode(prepare) {
    const value = prepare?.widgets?.find((item) => item?.name === MODE_WIDGET)?.value;
    return MODE_CHOICES.some(([mode]) => mode === value) ? value : "None";
}

export function refreshPrepare(node) {
    if (!Array.isArray(node.widgets)) return;
    applyLabels(node, PREPARE_LABELS);
    const active = PREPARE_WIDGETS[prepareMode(node)] ?? new Set();
    for (const target of node.widgets) {
        if (target?.name === MODE_WIDGET) continue;
        setWidgetVisible(target, active.has(target?.name));
    }
    for (const [execute] of linkedTargets(node, "FACE", EXECUTE_CLASS)) refreshExecute(execute);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function executePrepare(node) {
    const input = node.inputs?.find((item) => item?.name === "FACE" && item.link != null);
    const link = input ? linkOf(graphOf(node), input.link) : null;
    const source = link ? graphOf(node).getNodeById?.(link.origin_id) : null;
    return nodeClass(source) === PREPARE_CLASS ? source : null;
}

export function refreshExecute(node) {
    if (!Array.isArray(node.widgets)) return;
    applyLabels(node, EXECUTE_LABELS);
    const prepare = executePrepare(node);
    // Without a Prepare the mode is unknown: keep every widget reachable.
    const active = prepare ? (EXECUTE_WIDGETS[prepareMode(prepare)] ?? new Set()) : null;
    for (const target of node.widgets) setWidgetVisible(target, active === null || active.has(target?.name));
    node.setDirtyCanvas?.(true, true);
}

// The control's selection reaches the Prepare as a link; its own widget would
// stay stale, so mirror the selection into every linked Prepare (as the
// Detailer Family Enable does) and refresh the projection.
function mirrorToPrepares(node) {
    const value = node.widgets?.find((target) => target?.name === MODE_WIDGET)?.value;
    for (const [prepare, link] of linkedTargets(node, "PROCESS MODE", PREPARE_CLASS)) {
        const name = prepare.inputs?.[link.target_slot]?.widget?.name ?? MODE_WIDGET;
        const widget = prepare.widgets?.find((item) => item?.name === name);
        if (widget && value != null && widget.value !== value) widget.value = value;
        refreshPrepare(prepare);
    }
}

function installPrepareCallback(node) {
    const widget = node.widgets?.find((target) => target?.name === MODE_WIDGET);
    if (!widget || widget._cmkFaceProcessMode) return;
    widget._cmkFaceProcessMode = true;
    const original = widget.callback;
    widget.callback = function () {
        const result = original?.apply(this, arguments);
        refreshPrepare(node);
        return result;
    };
}

function selectionLinked(node) {
    return Boolean(node.inputs?.some((input) => input?.widget?.name === MODE_WIDGET && input.link != null));
}

function render(root) {
    for (const button of root.querySelectorAll("button")) {
        const active = button.dataset.value === root._cmkValue;
        button.setAttribute("aria-pressed", active ? "true" : "false");
        button.style.color = active ? ACCENT : "rgba(214, 219, 226, 0.5)";
        button.style.border = active ? "1px solid rgba(88, 190, 230, 0.6)" : "1px solid rgba(255, 255, 255, 0.09)";
        button.style.fontWeight = active ? "700" : "500";
    }
}

export function installModeChoice(node) {
    const original = node.widgets?.find((target) => target?.name === MODE_WIDGET);
    if (!original || typeof node.addDOMWidget !== "function") return;
    let panel = node.widgets.find((target) => target?.name === CHOICE_NAME);
    if (!panel) {
        const root = document.createElement("div");
        root.className = "cmk-faceprocess-mode-choice";
        Object.assign(root.style, {
            display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px",
            width: "100%", boxSizing: "border-box", padding: "2px 0",
        });
        for (const [value, label] of MODE_CHOICES) {
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
                render(root);
                try { original.callback?.(value, app.canvas, node); } catch (_) {}
                mirrorToPrepares(node);
                node.setDirtyCanvas?.(true, true);
                app.graph?.setDirtyCanvas?.(true, true);
            });
            root.append(button);
        }
        root._cmkValue = String(original.value ?? "None");
        panel = node.addDOMWidget(CHOICE_NAME, CHOICE_NAME, root, {
            serialize: false,
            hideOnZoom: false,
            getMinHeight: () => 46,
            getHeight: () => 46,
            getValue: () => root._cmkValue,
            setValue: (value) => { root._cmkValue = String(value ?? "None"); render(root); },
        });
        panel.serialize = false;
        panel.computeLayoutSize = undefined;
        panel._cmkRoot = root;
    }
    const root = panel._cmkRoot;
    root._cmkValue = String(original.value ?? "None");
    render(root);
    // A linked selection (Standard subgraph: MODE from the outer widget) is
    // decided elsewhere; ComfyUI hides the dropdown, so do the same.
    setWidgetVisible(original, false);
    setWidgetVisible(panel, !selectionLinked(node));
    mirrorToPrepares(node);
    node.setDirtyCanvas?.(true, true);
}

export function outerModeSpec(node) {
    if (!node?.isSubgraphNode?.()) return null;
    return OUTER_MODES[node.subgraph?.name] ?? null;
}

export function applyOuterMode(node) {
    const spec = outerModeSpec(node);
    if (!spec) return false;
    const widget = node.widgets?.find((item) => item?.name === spec.widget);
    if (!widget) return false;
    widget.label = "MODE";
    widget.options ??= {};
    if (widget.options.cmkModeLabels !== spec.labels) {
        widget.options.cmkModeLabels = spec.labels;
        widget.options.getOptionLabel = (value) => spec.labels[value] ?? value;
    }
    return true;
}

function graphNode(id) {
    return [app.canvas?.graph, app.graph, app.rootGraph]
        .map((graph) => graph?.getNodeById?.(id) ?? graph?.getNodeById?.(Number(id)))
        .find(Boolean);
}

function decorate() {
    for (const root of document.querySelectorAll(".lg-node[data-node-id]")) {
        const node = graphNode(root.getAttribute("data-node-id"));
        const spec = outerModeSpec(node);
        if (!spec) continue;
        applyOuterMode(node);
        const row = root.querySelector(`.lg-node-widget[data-widget-name="${spec.widget}"]`);
        if (row && !row.hasAttribute(ROW_ATTRIBUTE)) row.setAttribute(ROW_ATTRIBUTE, "");
    }
}

function installStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.lg-node-widget[${ROW_ATTRIBUTE}] [data-testid="widget-select-default-trigger"] > span { text-align: right; }
`;
    document.head.appendChild(style);
}

// Inside CMK Flow · FaceProcess the Prepare's PROCESS MODE is linked to the
// subgraph input MODE, so the inner widget keeps a stale value while the
// prompt uses the outer one. As in the Multi-Model Detailer, mirror the outer
// choice into the inner Prepare so the inner view shows the active set (and
// the Execute follows it). Instances share one definition, so the entered
// instance is mirrored again when the canvas opens its subgraph.
function innerPrepares(host) {
    const inner = host?.subgraph;
    const nodes = inner?.nodes ?? inner?._nodes ?? [];
    return nodes.filter((node) => nodeClass(node) === PREPARE_CLASS).map((node) => [inner, node]);
}

function hostModeWidget(host, inner, prepare) {
    const input = prepare.inputs?.find((item) => item?.name === MODE_WIDGET && item.link != null);
    if (!input) return null;
    const link = linkOf(inner, input.link);
    if (!link || Number(link.origin_id) !== -10) return null;
    const name = inner.inputs?.[link.origin_slot]?.name;
    return host.widgets?.find((item) => item?.name === name) ?? null;
}

export function syncHostMode(host) {
    for (const [inner, prepare] of innerPrepares(host)) {
        const hostWidget = hostModeWidget(host, inner, prepare);
        const widget = prepare.widgets?.find((item) => item?.name === MODE_WIDGET);
        if (!hostWidget || !widget) continue;
        widget.value = hostWidget.value;
        refreshPrepare(prepare);
        if (!hostWidget._cmkFaceProcessHost) {
            hostWidget._cmkFaceProcessHost = true;
            const original = hostWidget.callback;
            hostWidget.callback = function () {
                const result = original?.apply(this, arguments);
                syncHostMode(host);
                return result;
            };
        }
    }
}

function refresh(node) {
    const type = nodeClass(node);
    if (type === MODE_CONTROL_CLASS) installModeChoice(node);
    else if (type === PREPARE_CLASS) {
        installPrepareCallback(node);
        refreshPrepare(node);
    } else if (type === EXECUTE_CLASS) refreshExecute(node);
    else {
        applyOuterMode(node);
        if (node?.subgraph) syncHostMode(node);
    }
}

// Late passes cover widgets the frontend re-creates after creation.
function schedule(node) {
    for (const delay of [0, 50, 200]) setTimeout(() => refresh(node), delay);
}

app.registerExtension({
    name: "cmk.faceprocess.mode.ui.v1",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (![MODE_CONTROL_CLASS, PREPARE_CLASS, EXECUTE_CLASS].includes(nodeData.name)) return;
        for (const hook of ["onNodeCreated", "onConfigure", "onAdded", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                schedule(this);
                return result;
            };
        }
    },

    setup() {
        installStyle();
        new MutationObserver(decorate).observe(document.documentElement, { childList: true, subtree: true });
        app.canvas?.canvas?.addEventListener?.("subgraph-opened", (event) => {
            const host = event.detail?.fromNode;
            if (innerPrepares(host).length) syncHostMode(host);
        });
    },

    nodeCreated(node) {
        schedule(node);
    },

    loadedGraphNode(node) {
        schedule(node);
    },
});
