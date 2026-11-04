import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS = "CMKSwapImageLoaderPipe";
const PAIR_WIDGET = "IMAGE PAIR";
const DEFAULT_PAIR = { target: "", source: "" };

function isTarget(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS ||
        node.type === NODE_CLASS ||
        node.constructor?.comfyClass === NODE_CLASS ||
        node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function decodePair(value) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
        return {
            target: String(value.target ?? ""),
            source: String(value.source ?? ""),
        };
    }
    try {
        const parsed = JSON.parse(String(value || "{}"));
        return {
            target: String(parsed?.target ?? ""),
            source: String(parsed?.source ?? ""),
        };
    } catch (_) {
        return { ...DEFAULT_PAIR };
    }
}

function encodePair(pair) {
    return JSON.stringify({
        target: String(pair?.target ?? ""),
        source: String(pair?.source ?? ""),
    });
}

function splitAnnotatedName(rawValue) {
    let value = String(rawValue || "").trim();
    let type = "input";
    const annotated = value.match(/\s\[(input|output|temp)\]$/i);
    if (annotated) {
        type = annotated[1].toLowerCase();
        value = value.slice(0, annotated.index).trim();
    }
    value = value.replaceAll("\\", "/");
    const slash = value.lastIndexOf("/");
    return {
        filename: slash >= 0 ? value.slice(slash + 1) : value,
        subfolder: slash >= 0 ? value.slice(0, slash) : "",
        type,
    };
}

function imageViewUrl(value) {
    const parsed = splitAnnotatedName(value);
    if (!parsed.filename) return "";
    const params = new URLSearchParams({
        filename: parsed.filename,
        type: parsed.type,
        rand: String(Date.now()),
    });
    if (parsed.subfolder) params.set("subfolder", parsed.subfolder);
    const path = `/view?${params.toString()}`;
    return typeof api?.apiURL === "function" ? api.apiURL(path) : path;
}

function normaliseObjectInfo(payload) {
    const info = payload?.["CMKImageLoadAndResizePipe"] ?? payload;
    const spec = info?.input?.required?.IMAGE ?? info?.input?.required?.image;
    const values = Array.isArray(spec?.[0]) ? spec[0] : [];
    return values.map((item) => String(item));
}

async function fetchImageOptions() {
    for (const route of [
        "/object_info/CMKImageLoadAndResizePipe",
        "/api/object_info/CMKImageLoadAndResizePipe",
    ]) {
        try {
            const response = await api.fetchApi(route);
            if (!response?.ok) continue;
            const values = normaliseObjectInfo(await response.json());
            if (values.length) return values;
        } catch (_) {}
    }
    return [];
}

function makeElement(tag, className = "") {
    const element = document.createElement(tag);
    if (className) element.className = className;
    return element;
}

function applyStyles(root) {
    root.style.display = "grid";
    root.style.gridTemplateColumns = "repeat(2, minmax(0, 1fr))";
    root.style.gap = "10px";
    root.style.boxSizing = "border-box";
    root.style.width = "100%";
    root.style.minHeight = "350px";
    root.style.setProperty("--comfy-widget-min-height", "350px");
    root.style.setProperty("--comfy-widget-height", "350px");
    root.style.padding = "2px";
    root.style.fontFamily = "inherit";
}

function createImageCard(title) {
    const card = makeElement("div", "cmk-swap-loader-card");
    card.style.display = "grid";
    card.style.gridTemplateRows = "auto auto minmax(210px, 1fr)";
    card.style.gap = "7px";
    card.style.minWidth = "0";
    card.style.padding = "9px";
    card.style.border = "1px solid rgba(255,255,255,.14)";
    card.style.borderRadius = "10px";
    card.style.background = "rgba(0,0,0,.16)";
    card.style.boxSizing = "border-box";

    const heading = makeElement("div");
    heading.textContent = title;
    heading.style.fontSize = "12px";
    heading.style.fontWeight = "700";
    heading.style.letterSpacing = ".04em";
    heading.style.opacity = ".9";

    const controls = makeElement("div");
    controls.style.display = "grid";
    controls.style.gridTemplateColumns = "minmax(0,1fr) auto";
    controls.style.gap = "6px";

    const select = makeElement("select");
    select.style.width = "100%";
    select.style.minWidth = "0";
    select.style.height = "32px";
    select.style.borderRadius = "8px";
    select.style.border = "1px solid rgba(255,255,255,.18)";
    select.style.background = "var(--comfy-input-bg, #222)";
    select.style.color = "var(--input-text, #ddd)";
    select.style.padding = "0 8px";

    const button = makeElement("button");
    button.type = "button";
    button.textContent = "LOAD";
    button.style.height = "32px";
    button.style.padding = "0 12px";
    button.style.borderRadius = "8px";
    button.style.border = "1px solid rgba(255,255,255,.18)";
    button.style.background = "var(--comfy-input-bg, #2b2b2b)";
    button.style.color = "var(--input-text, #ddd)";
    button.style.cursor = "pointer";

    const fileInput = makeElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.style.display = "none";

    const previewFrame = makeElement("div");
    previewFrame.style.position = "relative";
    previewFrame.style.display = "flex";
    previewFrame.style.alignItems = "center";
    previewFrame.style.justifyContent = "center";
    previewFrame.style.minHeight = "210px";
    previewFrame.style.overflow = "hidden";
    previewFrame.style.borderRadius = "8px";
    previewFrame.style.background = "rgba(0,0,0,.35)";

    const preview = makeElement("img");
    preview.alt = title;
    preview.style.display = "none";
    preview.style.width = "100%";
    preview.style.height = "100%";
    preview.style.maxHeight = "300px";
    preview.style.objectFit = "contain";

    const empty = makeElement("div");
    empty.textContent = "NO IMAGE";
    empty.style.fontSize = "11px";
    empty.style.opacity = ".45";

    controls.append(select, button, fileInput);
    previewFrame.append(preview, empty);
    card.append(heading, controls, previewFrame);

    return { card, select, button, fileInput, preview, empty };
}

function updateOptions(select, values, current) {
    const unique = [...new Set([current, ...values].filter(Boolean))].sort((a, b) =>
        a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" })
    );
    const fragment = document.createDocumentFragment();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select image…";
    fragment.append(placeholder);
    for (const value of unique) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        fragment.append(option);
    }
    select.replaceChildren(fragment);
    select.value = current || "";
}

function updatePreview(card, value) {
    const url = imageViewUrl(value);
    if (!url) {
        card.preview.removeAttribute("src");
        card.preview.style.display = "none";
        card.empty.style.display = "block";
        return;
    }
    card.preview.onload = () => {
        card.preview.style.display = "block";
        card.empty.style.display = "none";
    };
    card.preview.onerror = () => {
        card.preview.style.display = "none";
        card.empty.style.display = "block";
        card.empty.textContent = "PREVIEW UNAVAILABLE";
    };
    card.empty.textContent = "LOADING…";
    card.empty.style.display = "block";
    card.preview.src = url;
}

async function uploadImage(file) {
    const body = new FormData();
    body.append("image", file);
    body.append("type", "input");
    body.append("overwrite", "true");
    const response = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
    });
    if (!response?.ok) {
        throw new Error(`upload failed (${response?.status ?? "unknown"})`);
    }
    const result = await response.json();
    let value = String(result?.name ?? file.name ?? "");
    const subfolder = String(result?.subfolder ?? "").replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
    if (subfolder) value = `${subfolder}/${value}`;
    const type = String(result?.type ?? "input");
    if (type && type !== "input") value = `${value} [${type}]`;
    return value;
}

function resizeNode(node) {
    try {
        const computed = node.computeSize?.();
        const width = Math.max(Number(node.size?.[0]) || 0, 720);
        const height = Math.max(Number(computed?.[1]) || Number(node.size?.[1]) || 0, 470);
        node.setSize?.([width, height]);
    } catch (_) {}
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function installPairWidget(node) {
    if (!isTarget(node)) return null;
    if (node._cmkSwapImagePairV1?.widget) return node._cmkSwapImagePairV1;

    const original = (node.widgets ?? []).find((widget) => widget?.name === PAIR_WIDGET);
    if (!original) return null;

    const root = makeElement("div", "cmk-swap-image-loader-pair");
    applyStyles(root);
    const targetCard = createImageCard("TARGET IMAGE");
    const sourceCard = createImageCard("SOURCE IMAGE");
    root.append(targetCard.card, sourceCard.card);

    const state = {
        node,
        root,
        targetCard,
        sourceCard,
        pair: decodePair(original.value),
        options: [],
        widget: null,
        removed: false,
    };

    const render = () => {
        updateOptions(targetCard.select, state.options, state.pair.target);
        updateOptions(sourceCard.select, state.options, state.pair.source);
        updatePreview(targetCard, state.pair.target);
        updatePreview(sourceCard, state.pair.source);
    };

    const setPairValue = (value, notify = true) => {
        state.pair = decodePair(value);
        const serialized = encodePair(state.pair);
        root._cmkValue = serialized;
        render();
        if (notify) {
            try {
                node.onWidgetChanged?.(PAIR_WIDGET, serialized, state.widget, state.widget);
            } catch (_) {}
            node.setDirtyCanvas?.(true, true);
        }
    };

    Object.defineProperty(root, "value", {
        configurable: true,
        get() {
            return root._cmkValue ?? encodePair(state.pair);
        },
        set(value) {
            setPairValue(value, false);
        },
    });
    root._cmkValue = encodePair(state.pair);

    const panelWidget = node.addDOMWidget(
        PAIR_WIDGET,
        "cmk_swap_image_pair",
        root,
        {
            hideOnZoom: false,
            getMinHeight: () => 350,
            getHeight: () => 350,
            getValue: () => root.value,
            setValue: (value) => {
                root.value = value;
            },
        },
    );
    state.widget = panelWidget;
    panelWidget.serialize = true;
    panelWidget.serializeValue = async () => root.value;
    panelWidget.computeSize = (width) => [Math.max(Number(width) || 720, 720), 350];

    const originalIndex = node.widgets.indexOf(original);
    const panelIndex = node.widgets.indexOf(panelWidget);
    if (panelIndex >= 0) node.widgets.splice(panelIndex, 1);
    if (originalIndex >= 0) node.widgets.splice(originalIndex, 1, panelWidget);
    else node.widgets.push(panelWidget);

    const commit = (key, value) => {
        state.pair[key] = String(value || "");
        root._cmkValue = encodePair(state.pair);
        render();
        try {
            panelWidget.callback?.(root._cmkValue, node, panelWidget);
        } catch (_) {}
        try {
            node.onWidgetChanged?.(PAIR_WIDGET, root._cmkValue, panelWidget, panelWidget);
        } catch (_) {}
        node.setDirtyCanvas?.(true, true);
    };

    targetCard.select.addEventListener("change", () => commit("target", targetCard.select.value));
    sourceCard.select.addEventListener("change", () => commit("source", sourceCard.select.value));
    targetCard.button.addEventListener("click", () => targetCard.fileInput.click());
    sourceCard.button.addEventListener("click", () => sourceCard.fileInput.click());

    const bindUpload = (card, key) => {
        card.fileInput.addEventListener("change", async () => {
            const file = card.fileInput.files?.[0];
            if (!file) return;
            card.button.disabled = true;
            card.button.textContent = "…";
            try {
                const value = await uploadImage(file);
                if (!state.options.includes(value)) state.options.push(value);
                commit(key, value);
            } catch (error) {
                console.error("CMK Swap Image Loader upload failed", error);
                card.empty.textContent = "UPLOAD FAILED";
                card.empty.style.display = "block";
            } finally {
                card.fileInput.value = "";
                card.button.disabled = false;
                card.button.textContent = "LOAD";
            }
        });
    };
    bindUpload(targetCard, "target");
    bindUpload(sourceCard, "source");

    const refreshOptions = async () => {
        const values = await fetchImageOptions();
        if (state.removed) return;
        state.options = values;
        render();
    };
    targetCard.select.addEventListener("focus", refreshOptions);
    sourceCard.select.addEventListener("focus", refreshOptions);

    node._cmkSwapImagePairV1 = state;
    setPairValue(original.value, false);
    refreshOptions();
    setTimeout(() => resizeNode(node), 0);
    setTimeout(() => resizeNode(node), 150);
    return state;
}

function ensureInstalled(node) {
    if (!isTarget(node)) return;
    installPairWidget(node);
    const state = node._cmkSwapImagePairV1;
    if (state?.widget) {
        state.root.value = state.widget.value ?? state.root.value;
        resizeNode(node);
    }
}

app.registerExtension({
    name: "cmk.swap_image_loader.pair.v1",

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
            const state = this._cmkSwapImagePairV1;
            if (state?.widget) state.root.value = state.widget.value;
            return result;
        };

        const originalSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
            const result = originalSerialize?.apply(this, arguments);
            const state = this._cmkSwapImagePairV1;
            if (data && state?.widget) {
                const serializable = (this.widgets ?? [])
                    .filter((widget) => widget?.serialize !== false)
                    .map((widget) => widget === state.widget ? state.root.value : widget?.value);
                data.widgets_values = serializable;
            }
            return result;
        };

        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            const state = this._cmkSwapImagePairV1;
            if (state) state.removed = true;
            return originalRemoved?.apply(this, arguments);
        };
    },

    nodeCreated(node) {
        ensureInstalled(node);
    },

    loadedGraphNode(node) {
        ensureInstalled(node);
    },
});
