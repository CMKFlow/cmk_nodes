import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const TARGETS = new Map([
    ["CMKSplitVideoIntoSegments", { title: "SOURCE VIDEO", inputWidget: "VIDEO", inputType: "input" }],
    ["CMKMergeAndSaveVideo", { title: "MERGED VIDEO" }],
]);

function isTarget(node, nodeClass) {
    return Boolean(node) && (
        node.comfyClass === nodeClass ||
        node.type === nodeClass ||
        node.constructor?.comfyClass === nodeClass ||
        node.constructor?.nodeData?.name === nodeClass
    );
}

function make(tag, text = "") {
    const element = document.createElement(tag);
    if (text) element.textContent = text;
    return element;
}

function descriptorUrl(descriptor) {
    if (!descriptor?.filename) return "";
    const params = new URLSearchParams({
        filename: String(descriptor.filename),
        type: String(descriptor.type || "output"),
        format: String(descriptor.format || "video/mp4"),
        rand: String(Date.now()),
    });
    if (descriptor.subfolder) params.set("subfolder", String(descriptor.subfolder));
    const path = `/view?${params.toString()}`;
    return typeof api?.apiURL === "function" ? api.apiURL(path) : path;
}

function descriptorFromInput(value) {
    const normalized = String(value || "").replaceAll("\\", "/").replace(/^\/+/, "");
    if (!normalized) return null;
    const parts = normalized.split("/").filter(Boolean);
    const filename = parts.pop();
    if (!filename) return null;
    const extension = filename.includes(".") ? filename.split(".").pop().toLowerCase() : "mp4";
    return {
        filename,
        subfolder: parts.join("/"),
        type: "input",
        format: `video/${extension || "mp4"}`,
    };
}

function currentWidgetValue(node, name) {
    return (node.widgets || []).find((widget) => widget?.name === name)?.value ?? "";
}

function setSource(state, descriptor) {
    const url = descriptorUrl(descriptor);
    if (!url || state.url === url) return;
    state.url = url;
    state.video.src = url;
    state.video.load();
}

function compactToContent(node, minWidth = 600) {
    requestAnimationFrame(() => {
        try {
            const computed = node.computeSize?.();
            const width = Math.max(Number(node.size?.[0]) || 0, minWidth);
            const height = Math.max(Number(computed?.[1]) || 0, 380);
            node.setSize?.([width, height]);
        } catch (_) {}
        node.setDirtyCanvas?.(true, true);
        node.graph?.setDirtyCanvas?.(true, true);
    });
}

function installPlayer(node, config) {
    if (node._cmkInlineVideoPlayerV1) return node._cmkInlineVideoPlayerV1;

    const root = make("div");
    root.style.cssText = [
        "display:grid",
        "grid-template-rows:auto minmax(250px,1fr)",
        "gap:7px",
        "width:100%",
        "height:310px",
        "box-sizing:border-box",
        "padding:8px",
        "border:1px solid rgba(255,255,255,.14)",
        "border-radius:10px",
        "background:rgba(0,0,0,.14)",
    ].join(";");

    const title = make("div", config.title || "VIDEO");
    title.style.cssText = "font-size:12px;font-weight:700;letter-spacing:.04em;opacity:.9";

    const video = make("video");
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;
    video.style.cssText = "display:block;width:100%;height:100%;min-height:250px;object-fit:contain;background:#000;border-radius:8px";

    root.append(title, video);
    const widget = node.addDOMWidget("cmk_inline_video_player", "CMK_INLINE_VIDEO_PLAYER", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 310,
        getHeight: () => 310,
    });
    widget.serialize = false;
    widget.computeSize = (width) => [Math.max(Number(width) || 600, 600), 310];

    const state = { node, root, video, widget, url: "" };
    node._cmkInlineVideoPlayerV1 = state;

    if (config.inputWidget) {
        const descriptor = descriptorFromInput(currentWidgetValue(node, config.inputWidget));
        if (descriptor) setSource(state, descriptor);
    }
    compactToContent(node);
    return state;
}

function installHooks(nodeType, nodeClass, config) {
    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = originalCreated?.apply(this, arguments);
        installPlayer(this, config);
        return result;
    };

    const originalExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        const result = originalExecuted?.apply(this, arguments);
        const state = installPlayer(this, config);
        const descriptor = message?.cmk_video_player?.[0];
        if (descriptor) setSource(state, descriptor);
        compactToContent(this);
        return result;
    };

    if (config.inputWidget) {
        const originalChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name, value) {
            const result = originalChanged?.apply(this, arguments);
            if (name === config.inputWidget) {
                const state = installPlayer(this, config);
                const descriptor = descriptorFromInput(value);
                if (descriptor) setSource(state, descriptor);
            }
            return result;
        };

        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            const result = originalConfigure?.apply(this, arguments);
            const state = installPlayer(this, config);
            const descriptor = descriptorFromInput(currentWidgetValue(this, config.inputWidget));
            if (descriptor) setSource(state, descriptor);
            compactToContent(this);
            return result;
        };
    }
}

app.registerExtension({
    name: "cmk.video.inline.players.v1",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const config = TARGETS.get(nodeData.name);
        if (!config) return;
        installHooks(nodeType, nodeData.name, config);
    },
    nodeCreated(node) {
        for (const [nodeClass, config] of TARGETS) {
            if (isTarget(node, nodeClass)) {
                installPlayer(node, config);
                break;
            }
        }
    },
    loadedGraphNode(node) {
        for (const [nodeClass, config] of TARGETS) {
            if (isTarget(node, nodeClass)) {
                const state = installPlayer(node, config);
                if (config.inputWidget) {
                    const descriptor = descriptorFromInput(currentWidgetValue(node, config.inputWidget));
                    if (descriptor) setSource(state, descriptor);
                }
                compactToContent(node);
                break;
            }
        }
    },
});
