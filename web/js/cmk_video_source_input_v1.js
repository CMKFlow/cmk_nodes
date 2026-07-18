import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS = "CMKSplitVideoIntoSegments";
const VIDEO_WIDGET = "VIDEO";

function isTarget(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS ||
        node.type === NODE_CLASS ||
        node.constructor?.comfyClass === NODE_CLASS ||
        node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function make(tag) {
    return document.createElement(tag);
}

function styleSelect(select) {
    select.style.width = "100%";
    select.style.minWidth = "0";
    select.style.height = "32px";
    select.style.borderRadius = "8px";
    select.style.border = "1px solid rgba(255,255,255,.18)";
    select.style.background = "var(--comfy-input-bg, #222)";
    select.style.color = "var(--input-text, #ddd)";
    select.style.padding = "0 8px";
}

function updateOptions(select, values, current) {
    const unique = [...new Set([current, ...(values || [])].filter(Boolean))].sort((a, b) =>
        String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" })
    );
    select.replaceChildren();
    for (const value of unique) {
        const option = make("option");
        option.value = String(value);
        option.textContent = String(value);
        select.append(option);
    }
    select.value = String(current || "");
}

async function uploadVideo(file) {
    const body = new FormData();
    // ComfyUI's established generic input upload endpoint uses the field name
    // "image" for arbitrary uploaded media.
    body.append("image", file);
    body.append("type", "input");
    body.append("subfolder", "video");
    body.append("overwrite", "true");
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response?.ok) throw new Error(`video upload failed (${response?.status ?? "unknown"})`);
    const result = await response.json();
    const name = String(result?.name ?? file.name ?? "").replaceAll("\\", "/");
    const subfolder = String(result?.subfolder ?? "video").replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
    return subfolder ? `${subfolder}/${name}` : name;
}

function resizeNode(node) {
    try {
        const computed = node.computeSize?.();
        const width = Math.max(Number(node.size?.[0]) || 0, 520);
        const height = Math.max(Number(computed?.[1]) || Number(node.size?.[1]) || 0, 260);
        node.setSize?.([width, height]);
    } catch (_) {}
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function installVideoSource(node) {
    if (!isTarget(node)) return null;
    if (node._cmkVideoSourceV1?.widget) return node._cmkVideoSourceV1;

    const original = (node.widgets ?? []).find((widget) => widget?.name === VIDEO_WIDGET);
    if (!original) return null;

    const root = make("div");
    root.style.display = "grid";
    root.style.gridTemplateColumns = "145px minmax(0,1fr) 38px";
    root.style.gap = "8px";
    root.style.alignItems = "center";
    root.style.boxSizing = "border-box";
    root.style.width = "100%";
    root.style.height = "38px";
    root.style.padding = "2px 0";
    root.style.fontFamily = "inherit";

    const label = make("span");
    label.textContent = "VIDEO";
    label.style.fontSize = "12px";
    label.style.opacity = ".75";

    const select = make("select");
    styleSelect(select);
    updateOptions(select, original.options?.values, original.value);

    const button = make("button");
    button.type = "button";
    button.textContent = "📁";
    button.title = "Video auswählen oder nach input/video hochladen";
    button.style.height = "32px";
    button.style.width = "38px";
    button.style.padding = "0";
    button.style.borderRadius = "8px";
    button.style.border = "1px solid rgba(255,255,255,.18)";
    button.style.background = "var(--comfy-input-bg, #222)";
    button.style.color = "var(--input-text, #ddd)";
    button.style.cursor = "pointer";

    const fileInput = make("input");
    fileInput.type = "file";
    fileInput.accept = "video/*,.mp4,.mov,.mkv,.webm,.avi,.m4v,.mpeg,.mpg";
    fileInput.style.display = "none";

    root.append(label, select, button, fileInput);

    const state = { node, root, select, button, fileInput, widget: null, original };

    Object.defineProperty(root, "value", {
        configurable: true,
        get() { return String(select.value || ""); },
        set(value) {
            const next = String(value || "");
            updateOptions(select, original.options?.values, next);
        },
    });

    const panel = node.addDOMWidget(VIDEO_WIDGET, "cmk_video_source", root, {
        hideOnZoom: false,
        getMinHeight: () => 38,
        getHeight: () => 38,
        getValue: () => root.value,
        setValue: (value) => { root.value = value; },
    });
    state.widget = panel;
    panel.serialize = true;
    panel.serializeValue = async () => root.value;
    panel.computeSize = (width) => [Math.max(Number(width) || 520, 520), 38];

    const originalIndex = node.widgets.indexOf(original);
    const panelIndex = node.widgets.indexOf(panel);
    if (panelIndex >= 0) node.widgets.splice(panelIndex, 1);
    if (originalIndex >= 0) node.widgets.splice(originalIndex, 1, panel);
    else node.widgets.unshift(panel);

    const commit = (value) => {
        root.value = value;
        try { panel.callback?.(root.value, node, panel); } catch (_) {}
        try { node.onWidgetChanged?.(VIDEO_WIDGET, root.value, panel, panel); } catch (_) {}
        node.setDirtyCanvas?.(true, true);
    };

    select.addEventListener("change", () => commit(select.value));
    button.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", async () => {
        const file = fileInput.files?.[0];
        if (!file) return;
        button.disabled = true;
        button.textContent = "…";
        try {
            const value = await uploadVideo(file);
            if (Array.isArray(original.options?.values) && !original.options.values.includes(value)) {
                original.options.values.push(value);
            }
            commit(value);
        } catch (error) {
            console.error("CMK video upload failed", error);
            button.textContent = "!";
            setTimeout(() => { button.textContent = "📁"; }, 1200);
        } finally {
            fileInput.value = "";
            button.disabled = false;
            if (button.textContent === "…") button.textContent = "📁";
        }
    });

    node._cmkVideoSourceV1 = state;
    root.value = original.value;
    setTimeout(() => resizeNode(node), 0);
    return state;
}

function ensureInstalled(node) {
    const state = installVideoSource(node);
    if (state?.widget) {
        state.root.value = state.widget.value ?? state.root.value;
        resizeNode(node);
    }
}

app.registerExtension({
    name: "cmk.video.source.v1",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            ensureInstalled(this);
            return result;
        };

        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            ensureInstalled(this);
            const result = originalConfigure?.apply(this, arguments);
            ensureInstalled(this);
            const state = this._cmkVideoSourceV1;
            if (state?.widget) state.root.value = state.widget.value;
            return result;
        };

        const originalSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
            const result = originalSerialize?.apply(this, arguments);
            const state = this._cmkVideoSourceV1;
            if (data && state?.widget) {
                data.widgets_values = (this.widgets ?? [])
                    .filter((widget) => widget?.serialize !== false)
                    .map((widget) => widget === state.widget ? state.root.value : widget?.value);
            }
            return result;
        };
    },
    nodeCreated(node) { ensureInstalled(node); },
    loadedGraphNode(node) { ensureInstalled(node); },
});
