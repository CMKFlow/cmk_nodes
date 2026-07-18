import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS = "CMKVideoCompare";

function make(tag, text = "") {
    const element = document.createElement(tag);
    if (text) element.textContent = text;
    return element;
}

function descriptorUrl(descriptor) {
    if (!descriptor?.filename) return "";
    const params = new URLSearchParams({
        filename: descriptor.filename,
        type: descriptor.type || "output",
        format: descriptor.format || "video/mp4",
        rand: String(Date.now()),
    });
    if (descriptor.subfolder) params.set("subfolder", descriptor.subfolder);
    const path = `/view?${params}`;
    return typeof api?.apiURL === "function" ? api.apiURL(path) : path;
}

function playerCard(title) {
    const card = make("div");
    card.style.cssText = "display:grid;grid-template-rows:auto minmax(220px,1fr);gap:7px;padding:9px;border:1px solid rgba(255,255,255,.14);border-radius:10px;background:rgba(0,0,0,.16);min-width:0";
    const heading = make("div", title);
    heading.style.cssText = "font-size:12px;font-weight:700;letter-spacing:.04em;opacity:.9";
    const video = make("video");
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;
    video.style.cssText = "display:block;width:100%;height:100%;min-height:220px;object-fit:contain;background:#000;border-radius:8px";
    card.append(heading, video);
    return { card, video };
}

function setSource(video, url) {
    if (!url) return;
    video.src = url;
    video.load();
}

function installCompare(node) {
    if (node._cmkComparePlayersV1) return node._cmkComparePlayersV1;

    const root = make("div");
    root.style.cssText = "display:grid;grid-template-columns:1fr 1fr;grid-template-rows:minmax(260px,1fr) auto;gap:10px;width:100%;height:100%;padding:2px;box-sizing:border-box";
    const source = playerCard("SOURCE");
    const result = playerCard("RESULT");

    const bar = make("div");
    bar.style.cssText = "grid-column:1/-1;display:grid;grid-template-columns:auto 1fr auto auto;gap:10px;align-items:center;padding:8px;border:1px solid rgba(255,255,255,.14);border-radius:9px;background:rgba(0,0,0,.18)";
    const play = make("button", "▶");
    const seek = make("input");
    seek.type = "range";
    seek.min = "0";
    seek.max = "1000";
    seek.value = "0";
    const time = make("span", "0:00 / 0:00");
    const audio = make("select");
    for (const [value, label] of [["source", "SOURCE AUDIO"], ["result", "RESULT AUDIO"], ["mute", "MUTE"]]) {
        const option = make("option", label);
        option.value = value;
        audio.append(option);
    }
    audio.value = "source";
    bar.append(play, seek, time, audio);
    root.append(source.card, result.card, bar);

    const dom = node.addDOMWidget("cmk_video_compare", "CMK_VIDEO_COMPARE", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 500,
        getHeight: () => 500,
    });
    dom.serialize = false;

    const hardSync = () => { result.video.currentTime = source.video.currentTime; };
    play.onclick = () => {
        if (source.video.paused) {
            hardSync();
            Promise.allSettled([source.video.play(), result.video.play()]);
            play.textContent = "❚❚";
        } else {
            source.video.pause();
            result.video.pause();
            play.textContent = "▶";
        }
    };
    seek.oninput = () => {
        const duration = Math.min(source.video.duration || 0, result.video.duration || Infinity);
        const target = duration * Number(seek.value) / 1000;
        source.video.currentTime = target;
        result.video.currentTime = target;
    };
    audio.onchange = () => {
        source.video.muted = audio.value !== "source";
        result.video.muted = audio.value !== "result";
    };
    audio.onchange();

    source.video.addEventListener("timeupdate", () => {
        const duration = Math.min(source.video.duration || 0, result.video.duration || Infinity);
        if (duration > 0) seek.value = String(Math.round(source.video.currentTime / duration * 1000));
        time.textContent = `${source.video.currentTime.toFixed(2)} / ${Number.isFinite(duration) ? duration.toFixed(2) : "--"}`;
        if (!result.video.paused && Math.abs(result.video.currentTime - source.video.currentTime) > 0.08) {
            result.video.currentTime = source.video.currentTime;
        }
    });
    for (const video of [source.video, result.video]) {
        video.addEventListener("pause", () => {
            if (source.video.paused && result.video.paused) play.textContent = "▶";
        });
    }

    node._cmkComparePlayersV1 = { source: source.video, result: result.video, dom };
    setTimeout(() => {
        node.setSize?.([Math.max(Number(node.size?.[0]) || 0, 900), Math.max(Number(node.size?.[1]) || 0, 600)]);
        node.setDirtyCanvas?.(true, true);
    }, 0);
    return node._cmkComparePlayersV1;
}

app.registerExtension({
    name: "cmk.video.compare.players.v1",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            installCompare(this);
            return result;
        };
        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const resultValue = originalExecuted?.apply(this, arguments);
            const state = installCompare(this);
            const descriptors = message?.cmk_compare_videos || [];
            if (descriptors[0]) setSource(state.source, descriptorUrl(descriptors[0]));
            if (descriptors[1]) setSource(state.result, descriptorUrl(descriptors[1]));
            return resultValue;
        };
    },
    nodeCreated(node) {
        if (node.comfyClass === NODE_CLASS) installCompare(node);
    },
});
