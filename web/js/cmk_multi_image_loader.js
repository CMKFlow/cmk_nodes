import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {
    NODE_TITLE,
    NODE_TYPE,
    PROPERTY,
    SLOTS,
    injectReferences,
    migrateOutputNames,
    normalizeState,
} from "./cmk_multi_image_loader_core.js";

const FILE_SOURCE_CLASS = "CMKImageFileLoader";
const STYLE_ID = "cmk-multi-image-loader-style";
const NODE_SIZE = [360, 420];

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.cmk-multi-image-slots { display: flex; gap: 6px; width: 100%; box-sizing: border-box; }
.cmk-multi-image-slot {
    flex: 1 1 0; min-width: 0; display: flex; align-items: center; justify-content: center; gap: 6px;
    height: 26px; border-radius: 6px; border: 1px solid rgba(255,255,255,.18);
    background: rgba(255,255,255,.04); color: var(--fg-color, #ddd); font-size: 11px; font-weight: 600;
}
.cmk-multi-image-slot[data-selected="true"] { border-color: var(--p-primary-color, #4a9eff); background: rgba(74,158,255,.18); }
.cmk-multi-image-slot-label { cursor: pointer; white-space: nowrap; padding: 4px 0; }
.cmk-multi-image-slot-dot {
    width: 10px; height: 10px; flex: 0 0 10px; border-radius: 50%; cursor: pointer; box-sizing: border-box;
    border: 2px solid #3fbf6a; background: #3fbf6a;
}
.cmk-multi-image-slot-dot[data-enabled="false"] { background: transparent; border-color: rgba(255,255,255,.45); }
`;
    document.head.append(style);
}

async function imageFileChoices() {
    try {
        const response = await api.fetchApi(`/object_info/${FILE_SOURCE_CLASS}`);
        const data = await response.json();
        const choices = data?.[FILE_SOURCE_CLASS]?.input?.required?.image?.[0];
        if (Array.isArray(choices)) return choices;
    } catch (_) {}
    return [];
}

function state(node) {
    node.properties ||= {};
    node.properties[PROPERTY] = normalizeState(node.properties[PROPERTY]);
    return node.properties[PROPERTY];
}

function renderSlots(node) {
    const ui = node._cmkMultiImage;
    if (!ui) return;
    const current = state(node);
    ui.buttons.forEach(({ root, dot }, index) => {
        root.dataset.selected = String(current.selected === index);
        dot.dataset.enabled = String(current.slots[index].enabled);
        dot.title = current.slots[index].enabled ? "Active · click to disable" : "Inactive · click to enable";
    });
}

// The single file selection shows the selected slot.  Writing the widget does
// not change the stored references of any other slot.
function showSelectedFile(node) {
    const ui = node._cmkMultiImage;
    if (!ui?.combo) return;
    const current = state(node);
    ui.syncing = true;
    try {
        ui.combo.value = current.slots[current.selected].file;
        ui.combo.callback?.(ui.combo.value);
    } finally {
        ui.syncing = false;
    }
    node.setDirtyCanvas?.(true, true);
}

function buildSlotBar(node) {
    const root = document.createElement("div");
    root.className = "cmk-multi-image-slots";
    const buttons = SLOTS.map((slot, index) => {
        const button = document.createElement("div");
        button.className = "cmk-multi-image-slot";
        const label = document.createElement("span");
        label.className = "cmk-multi-image-slot-label";
        label.textContent = slot.short;
        label.title = `${slot.output} · select`;
        const dot = document.createElement("span");
        dot.className = "cmk-multi-image-slot-dot";
        // Selection and activation are separate targets and never trigger each other.
        label.addEventListener("pointerdown", (event) => event.stopPropagation());
        dot.addEventListener("pointerdown", (event) => event.stopPropagation());
        label.addEventListener("click", (event) => {
            event.stopPropagation();
            state(node).selected = index;
            renderSlots(node);
            showSelectedFile(node);
        });
        dot.addEventListener("click", (event) => {
            event.stopPropagation();
            const slotState = state(node).slots[index];
            slotState.enabled = !slotState.enabled;
            renderSlots(node);
            node.setDirtyCanvas?.(true, true);
        });
        button.append(label, dot);
        root.append(button);
        return { root: button, label, dot };
    });
    return { root, buttons };
}

async function installUi(node) {
    if (node._cmkMultiImage) return;
    ensureStyle();
    state(node);
    const bar = buildSlotBar(node);
    node._cmkMultiImage = { buttons: bar.buttons, combo: null, syncing: false };
    const barWidget = node.addDOMWidget("SLOTS", "cmk_multi_image_slots", bar.root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 30,
        getHeight: () => 30,
    });
    barWidget.serialize = false;

    const { ComfyWidgets } = await import("../../../scripts/widgets.js");
    const choices = await imageFileChoices();
    const combo = ComfyWidgets.COMBO(node, "image", [choices, { image_upload: true }], app).widget;
    combo.serialize = false;
    combo.options = { ...(combo.options || {}), serialize: false };
    node._cmkMultiImage.combo = combo;
    const upload = ComfyWidgets.IMAGEUPLOAD(node, "upload", ["IMAGEUPLOAD", { imageInputName: "image" }], app);
    if (upload?.widget) upload.widget.serialize = false;
    // The upload widget replaces the combo callback (preview); wrap it afterwards.
    const previewCallback = combo.callback;
    combo.callback = function (value) {
        const result = previewCallback?.apply(this, arguments);
        const ui = node._cmkMultiImage;
        if (ui && !ui.syncing) {
            const current = state(node);
            current.slots[current.selected].file = String(combo.value ?? value ?? "");
        }
        return result;
    };

    renderSlots(node);
    showSelectedFile(node);
    node.setSize?.([Math.max(node.size?.[0] ?? 0, NODE_SIZE[0]), Math.max(node.size?.[1] ?? 0, NODE_SIZE[1])]);
}

function registerNode() {
    const LiteGraph = window.LiteGraph;
    const LGraphNode = window.LGraphNode ?? window.comfyAPI?.litegraph?.LGraphNode;
    if (!LiteGraph || !LGraphNode || LiteGraph.registered_node_types?.[NODE_TYPE]) return;

    class CMKMultiImageLoaderNode extends LGraphNode {
        constructor(title) {
            super(title);
            // Frontend only: the prompt receives the references directly at the consumers.
            this.isVirtualNode = true;
            this.serialize_widgets = false;
            this.properties = { [PROPERTY]: normalizeState(null) };
            for (const slot of SLOTS) this.addOutput(slot.output, "STRING");
            this.size = [...NODE_SIZE];
            queueMicrotask(() => installUi(this));
        }

        onConfigure() {
            migrateOutputNames(this);
            // The former default title; a title the user set stays.
            if (this.title === "CMK Multi Image Loader") this.title = NODE_TITLE;
            state(this);
            renderSlots(this);
            showSelectedFile(this);
        }
    }
    CMKMultiImageLoaderNode.title = NODE_TITLE;
    CMKMultiImageLoaderNode.category = "CMK/Flow/Input";
    CMKMultiImageLoaderNode.description = "Manages four image_file references (CN, ID A, ID B, ID C). No image processing.";
    LiteGraph.registerNodeType(NODE_TYPE, CMKMultiImageLoaderNode);
}

function installPromptInjection() {
    if (app._cmkMultiImageInjection) return;
    const original = app.graphToPrompt.bind(app);
    app.graphToPrompt = async function (...args) {
        const result = await original(...args);
        injectReferences(args[0] ?? app.rootGraph ?? app.graph, result?.output);
        return result;
    };
    app._cmkMultiImageInjection = true;
}

// The frontend builds a placeholder definition for frontend-only nodes.  Give it
// the CMK name, category and source (badge "cmk-flow"), and the input spec of
// the CMK Image File Loader so the file widget renders as in that loader.
function describeNode(definitions) {
    const list = Array.isArray(definitions) ? definitions : Object.values(definitions ?? {});
    const definition = list.find((item) => item?.name === NODE_TYPE);
    if (!definition) return;
    const fileSource = list.find((item) => item?.name === FILE_SOURCE_CLASS);
    const imageSpec = fileSource?.input?.required?.image;
    Object.assign(definition, {
        display_name: NODE_TITLE,
        category: "CMK/Flow/Input",
        python_module: "custom_nodes.cmk-flow",
        description: "Manages four image_file references (CN, ID A, ID B, ID C). No image processing.",
        output: SLOTS.map(() => "STRING"),
        output_name: SLOTS.map((slot) => slot.output),
        output_is_list: SLOTS.map(() => false),
    });
    if (imageSpec) definition.input = { required: { image: imageSpec }, optional: {} };
}

app.registerExtension({
    name: "cmk.multi_image_loader",
    registerCustomNodes() {
        registerNode();
    },
    beforeRegisterVueAppNodeDefs(definitions) {
        describeNode(definitions);
    },
    setup() {
        installPromptInjection();
    },
});
