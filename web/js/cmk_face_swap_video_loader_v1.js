import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS = "CMKFaceSwapVideoLoader";
const WIDGET_NAME = "MEDIA SOURCES";
const DEFAULT_MEDIA = { video: "", source: "" };

function isTarget(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS ||
        node.type === NODE_CLASS ||
        node.constructor?.comfyClass === NODE_CLASS ||
        node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function decodeMedia(value) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
        return {
            video: String(value.video ?? ""),
            source: String(value.source ?? ""),
        };
    }
    try {
        const parsed = JSON.parse(String(value || "{}"));
        return {
            video: String(parsed?.video ?? ""),
            source: String(parsed?.source ?? ""),
        };
    } catch (_) {
        return { ...DEFAULT_MEDIA };
    }
}

function encodeMedia(media) {
    return JSON.stringify({
        video: String(media?.video ?? ""),
        source: String(media?.source ?? ""),
    });
}

function make(tag, text = "") {
    const element = document.createElement(tag);
    if (text) element.textContent = text;
    return element;
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

function mediaUrl(value, formatPrefix) {
    const parsed = splitAnnotatedName(value);
    if (!parsed.filename) return "";
    const extension = parsed.filename.includes(".")
        ? parsed.filename.split(".").pop().toLowerCase()
        : (formatPrefix === "video" ? "mp4" : "png");
    const params = new URLSearchParams({
        filename: parsed.filename,
        type: parsed.type,
        format: `${formatPrefix}/${extension}`,
        rand: String(Date.now()),
    });
    if (parsed.subfolder) params.set("subfolder", parsed.subfolder);
    const path = `/view?${params.toString()}`;
    return typeof api?.apiURL === "function" ? api.apiURL(path) : path;
}

async function fetchVideoOptions() {
    for (const route of [
        `/object_info/${NODE_CLASS}`,
        `/api/object_info/${NODE_CLASS}`,
    ]) {
        try {
            const response = await api.fetchApi(route);
            if (!response?.ok) continue;
            const payload = await response.json();
            const info = payload?.[NODE_CLASS] ?? payload;
            // The public UI uses MEDIA SOURCES, so obtain the universal Split list.
            const splitResponse = await api.fetchApi("/object_info/CMKSplitVideoIntoSegments");
            if (!splitResponse?.ok) continue;
            const splitPayload = await splitResponse.json();
            const splitInfo = splitPayload?.CMKSplitVideoIntoSegments ?? splitPayload;
            const spec = splitInfo?.input?.required?.VIDEO;
            const values = Array.isArray(spec?.[0]) ? spec[0] : [];
            if (values.length) return values.map(String);
        } catch (_) {}
    }
    return [];
}

async function fetchImageOptions() {
    for (const route of [
        "/object_info/CMKImageLoadAndResizePipe",
        "/api/object_info/CMKImageLoadAndResizePipe",
    ]) {
        try {
            const response = await api.fetchApi(route);
            if (!response?.ok) continue;
            const payload = await response.json();
            const info = payload?.CMKImageLoadAndResizePipe ?? payload;
            const spec = info?.input?.required?.IMAGE ?? info?.input?.required?.image;
            const values = Array.isArray(spec?.[0]) ? spec[0] : [];
            if (values.length) return values.map(String);
        } catch (_) {}
    }
    return [];
}

async function uploadFile(file, subfolder = "") {
    const body = new FormData();
    body.append("image", file);
    body.append("type", "input");
    body.append("overwrite", "true");
    if (subfolder) body.append("subfolder", subfolder);
    const response = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
    });
    if (!response?.ok) {
        throw new Error(`upload failed (${response?.status ?? "unknown"})`);
    }
    const result = await response.json();
    let value = String(result?.name ?? file.name ?? "");
    const resultSubfolder = String(result?.subfolder ?? subfolder ?? "")
        .replaceAll("\\", "/")
        .replace(/^\/+|\/+$/g, "");
    if (resultSubfolder) value = `${resultSubfolder}/${value}`;
    const type = String(result?.type ?? "input");
    if (type && type !== "input") value = `${value} [${type}]`;
    return value;
}

function updateOptions(select, values, current, placeholderText) {
    const unique = [...new Set([current, ...values].filter(Boolean))].sort((a, b) =>
        a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" })
    );
    const fragment = document.createDocumentFragment();
    const placeholder = make("option", placeholderText);
    placeholder.value = "";
    fragment.append(placeholder);
    for (const value of unique) {
        const option = make("option", value);
        option.value = value;
        fragment.append(option);
    }
    select.replaceChildren(fragment);
    select.value = current || "";
}

function createCard(title, kind) {
    const card = make("div");
    card.style.cssText = [
        "display:grid",
        "grid-template-rows:auto auto minmax(250px,1fr)",
        "gap:7px",
        "min-width:0",
        "padding:9px",
        "box-sizing:border-box",
        "border:1px solid rgba(255,255,255,.14)",
        "border-radius:10px",
        "background:rgba(0,0,0,.16)",
    ].join(";");

    const heading = make("div", title);
    heading.style.cssText = "font-size:12px;font-weight:700;letter-spacing:.04em;opacity:.9";

    const controls = make("div");
    controls.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px";

    const select = make("select");
    select.style.cssText = [
        "width:100%",
        "min-width:0",
        "height:32px",
        "padding:0 8px",
        "border:1px solid rgba(255,255,255,.18)",
        "border-radius:8px",
        "background:var(--comfy-input-bg,#222)",
        "color:var(--input-text,#ddd)",
    ].join(";");

    const button = make("button", "LOAD");
    button.type = "button";
    button.style.cssText = [
        "height:32px",
        "padding:0 12px",
        "border:1px solid rgba(255,255,255,.18)",
        "border-radius:8px",
        "background:var(--comfy-input-bg,#2b2b2b)",
        "color:var(--input-text,#ddd)",
        "cursor:pointer",
    ].join(";");

    const fileInput = make("input");
    fileInput.type = "file";
    fileInput.accept = kind === "video" ? "video/*" : "image/*";
    fileInput.style.display = "none";

    const mediaFrame = make("div");
    mediaFrame.style.cssText = [
        "position:relative",
        "display:flex",
        "align-items:center",
        "justify-content:center",
        "min-height:250px",
        "overflow:hidden",
        "border-radius:8px",
        "background:#000",
    ].join(";");

    const media = kind === "video" ? make("video") : make("img");
    if (kind === "video") {
        media.controls = true;
        media.preload = "metadata";
        media.playsInline = true;
    } else {
        media.alt = title;
    }
    media.style.cssText = "display:none;width:100%;height:100%;max-height:320px;object-fit:contain";

    const empty = make("div", kind === "video" ? "NO VIDEO" : "NO IMAGE");
    empty.style.cssText = "font-size:11px;opacity:.45";

    controls.append(select, button, fileInput);
    mediaFrame.append(media, empty);
    card.append(heading, controls, mediaFrame);
    return { card, select, button, fileInput, media, empty, kind, url: "" };
}

function updatePreview(card, value) {
    const url = mediaUrl(value, card.kind);
    if (!url) {
        card.media.removeAttribute("src");
        card.media.style.display = "none";
        card.empty.textContent = card.kind === "video" ? "NO VIDEO" : "NO IMAGE";
        card.empty.style.display = "block";
        card.url = "";
        return;
    }
    card.media.onload = card.kind === "image" ? () => {
        card.media.style.display = "block";
        card.empty.style.display = "none";
    } : null;
    if (card.kind === "video") {
        card.media.onloadedmetadata = () => {
            card.media.style.display = "block";
            card.empty.style.display = "none";
        };
    }
    card.media.onerror = () => {
        card.media.style.display = "none";
        card.empty.textContent = "PREVIEW UNAVAILABLE";
        card.empty.style.display = "block";
    };
    card.empty.textContent = "LOADING…";
    card.empty.style.display = "block";
    card.url = url;
    card.media.src = url;
    if (card.kind === "video") card.media.load();
}

function resizeNode(node) {
    requestAnimationFrame(() => {
        try {
            const computed = node.computeSize?.();
            const width = Math.max(Number(node.size?.[0]) || 0, 980);
            const height = Math.max(Number(computed?.[1]) || Number(node.size?.[1]) || 0, 700);
            node.setSize?.([width, height]);
        } catch (_) {}
        node.setDirtyCanvas?.(true, true);
        node.graph?.setDirtyCanvas?.(true, true);
    });
}

function install(node) {
    if (!isTarget(node) || node._cmkFaceSwapVideoLoaderV1) return;

    const original = (node.widgets ?? []).find((widget) => widget?.name === WIDGET_NAME);
    if (!original) return;

    const root = make("div");
    root.style.cssText = [
        "display:grid",
        "grid-template-columns:repeat(2,minmax(0,1fr))",
        "gap:10px",
        "width:100%",
        "height:390px",
        "padding:2px",
        "box-sizing:border-box",
        "font-family:inherit",
    ].join(";");

    const videoCard = createCard("SOURCE VIDEO", "video");
    const imageCard = createCard("SOURCE IMAGE", "image");
    root.append(videoCard.card, imageCard.card);

    const state = {
        node,
        root,
        videoCard,
        imageCard,
        media: decodeMedia(original.value),
        videoOptions: [],
        imageOptions: [],
        widget: null,
        removed: false,
    };

    const render = () => {
        updateOptions(videoCard.select, state.videoOptions, state.media.video, "Select video…");
        updateOptions(imageCard.select, state.imageOptions, state.media.source, "Select image…");
        updatePreview(videoCard, state.media.video);
        updatePreview(imageCard, state.media.source);
    };

    Object.defineProperty(root, "value", {
        configurable: true,
        get() { return encodeMedia(state.media); },
        set(value) {
            state.media = decodeMedia(value);
            render();
        },
    });

    const panel = node.addDOMWidget(WIDGET_NAME, "cmk_face_swap_video_loader", root, {
        hideOnZoom: false,
        getMinHeight: () => 390,
        getHeight: () => 390,
        getValue: () => root.value,
        setValue: (value) => { root.value = value; },
    });
    state.widget = panel;
    panel.serialize = true;
    panel.serializeValue = async () => root.value;
    panel.computeSize = (width) => [Math.max(Number(width) || 980, 980), 390];

    const originalIndex = node.widgets.indexOf(original);
    const panelIndex = node.widgets.indexOf(panel);
    if (panelIndex >= 0) node.widgets.splice(panelIndex, 1);
    if (originalIndex >= 0) node.widgets.splice(originalIndex, 1, panel);
    else node.widgets.push(panel);

    const commit = (key, value) => {
        state.media[key] = String(value || "");
        render();
        const serialized = root.value;
        try { panel.callback?.(serialized, node, panel); } catch (_) {}
        try { node.onWidgetChanged?.(WIDGET_NAME, serialized, panel, panel); } catch (_) {}
        node.setDirtyCanvas?.(true, true);
    };

    videoCard.select.addEventListener("change", () => commit("video", videoCard.select.value));
    imageCard.select.addEventListener("change", () => commit("source", imageCard.select.value));
    videoCard.button.addEventListener("click", () => videoCard.fileInput.click());
    imageCard.button.addEventListener("click", () => imageCard.fileInput.click());

    videoCard.fileInput.addEventListener("change", async () => {
        const file = videoCard.fileInput.files?.[0];
        if (!file) return;
        videoCard.button.disabled = true;
        videoCard.button.textContent = "…";
        try {
            const value = await uploadFile(file, "video");
            if (!state.videoOptions.includes(value)) state.videoOptions.push(value);
            commit("video", value);
        } catch (error) {
            console.error("CMK FaceSwap Video Loader video upload failed", error);
            videoCard.empty.textContent = "UPLOAD FAILED";
            videoCard.empty.style.display = "block";
        } finally {
            videoCard.fileInput.value = "";
            videoCard.button.disabled = false;
            videoCard.button.textContent = "LOAD";
        }
    });

    imageCard.fileInput.addEventListener("change", async () => {
        const file = imageCard.fileInput.files?.[0];
        if (!file) return;
        imageCard.button.disabled = true;
        imageCard.button.textContent = "…";
        try {
            const value = await uploadFile(file);
            if (!state.imageOptions.includes(value)) state.imageOptions.push(value);
            commit("source", value);
        } catch (error) {
            console.error("CMK FaceSwap Video Loader image upload failed", error);
            imageCard.empty.textContent = "UPLOAD FAILED";
            imageCard.empty.style.display = "block";
        } finally {
            imageCard.fileInput.value = "";
            imageCard.button.disabled = false;
            imageCard.button.textContent = "LOAD";
        }
    });

    const refreshVideo = async () => {
        state.videoOptions = await fetchVideoOptions();
        if (!state.removed) render();
    };
    const refreshImage = async () => {
        state.imageOptions = await fetchImageOptions();
        if (!state.removed) render();
    };
    videoCard.select.addEventListener("focus", refreshVideo);
    imageCard.select.addEventListener("focus", refreshImage);

    node._cmkFaceSwapVideoLoaderV1 = state;
    root.value = original.value;
    refreshVideo();
    refreshImage();
    setTimeout(() => resizeNode(node), 0);
    setTimeout(() => resizeNode(node), 150);
}

function ensure(node) {
    if (!isTarget(node)) return;
    install(node);
    const state = node._cmkFaceSwapVideoLoaderV1;
    if (state?.widget) {
        state.root.value = state.widget.value ?? state.root.value;
        resizeNode(node);
    }
}

app.registerExtension({
    name: "cmk.face_swap_video_loader.v1",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            ensure(this);
            return result;
        };

        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            ensure(this);
            const result = configure?.apply(this, arguments);
            ensure(this);
            const state = this._cmkFaceSwapVideoLoaderV1;
            if (state?.widget) state.root.value = state.widget.value;
            return result;
        };

        const serialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
            const result = serialize?.apply(this, arguments);
            const state = this._cmkFaceSwapVideoLoaderV1;
            if (data && state?.widget) {
                const values = (this.widgets ?? [])
                    .filter((widget) => widget?.serialize !== false)
                    .map((widget) => widget === state.widget ? state.root.value : widget?.value);
                data.widgets_values = values;
            }
            return result;
        };

        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            const state = this._cmkFaceSwapVideoLoaderV1;
            if (state) state.removed = true;
            return removed?.apply(this, arguments);
        };
    },
    nodeCreated(node) { ensure(node); },
    loadedGraphNode(node) { ensure(node); },
});
