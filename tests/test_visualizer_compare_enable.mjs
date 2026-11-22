// CMK 2.8 Preview & Compare: a disabled module's Compare is no display target.
// displayNodeEnabled() reads the real CMK 2.8 control by its contract (MODE
// off at DISABLE, model family / process mode off at None) instead of a
// Boolean of a widget string; a disabled Compare takes no live preview of
// another module; an interrupted run leaves no live image as its state.
import assert from "node:assert/strict";
import { register } from "node:module";

register("./helpers/cmk_frontend_test_loader.mjs", import.meta.url);

const listeners = {};
let extension = null;
const element = () => new Proxy({ style: {}, children: [], dataset: {} }, {
    get(target, key) {
        if (key in target) return target[key];
        if (key === "replaceChildren") return (...items) => { target.children = items; };
        if (["append", "appendChild"].includes(key)) return (...items) => { target.children.push(...items); };
        if (key === "querySelector") return () => null;
        return () => {};
    },
});
globalThis.__cmkTestApp = { registerExtension(value) { extension = value; }, graph: { _nodes: [] } };
globalThis.__cmkTestApi = { addEventListener(name, handler) { listeners[name] = handler; }, apiURL: (path) => path };
globalThis.document = { createElement: element, head: { append() {}, appendChild() {} }, getElementById: () => null };
let urls = 0;
globalThis.URL.createObjectURL = () => `blob:${++urls}`;
globalThis.URL.revokeObjectURL = () => {};
globalThis.setTimeout = () => 0;

const { displayNodeEnabled, visualState } = await import("../web/js/cmk_visualizer.js");

class CompareNode {
    addDOMWidget() {}
}
await extension.beforeRegisterNodeDef(CompareNode, { name: "CMKVisualCompare" });

function graphWith(nodes, links) {
    return {
        _nodes: nodes,
        links: Object.fromEntries(links.map((link) => [link.id, link])),
        getNodeById: (id) => nodes.find((node) => String(node.id) === String(id)) || null,
    };
}
function compareFrom(control, value, extra = {}) {
    const origin = {
        id: 2, type: control, comfyClass: control,
        widgets: [{ name: extra.widget, value }], inputs: extra.inputs || [], outputs: [{ name: "ENABLE" }],
    };
    const compare = Object.assign(new CompareNode(), {
        id: 3, type: "CMKVisualCompare", comfyClass: "CMKVisualCompare",
        inputs: [{ name: "enable", link: 1 }], properties: {},
    });
    compare.onNodeCreated();
    const graph = graphWith([origin, compare], [{ id: 1, origin_id: 2, origin_slot: 0, target_id: 3, target_slot: 0 }, ...(extra.links || [])]);
    origin.graph = graph;
    compare.graph = graph;
    return { origin, compare, graph };
}

// 1. Every CMK 2.8 control by its contract.
const contracts = [
    ["CMKDetailerFamilyEnable", "model_family", { None: false, SDXL: true, "Z-Image Turbo": true }],
    ["CMKFaceProcessModeEnable", "process_mode", { None: false, Restore: true, SDXL: true, "Z-Image Turbo": true }],
    ["CMKFaceRebuildModeEnable", "facerebuild_mode", { DISABLE: false, ENABLE: true }],
    ["CMKFaceSwapModeEnable", "faceswap_mode", { DISABLE: false, ENABLE: true }],
    ["CMKMaskDetailerModeEnable", "maskdetailer_mode", { DISABLE: false, ENABLE: true }],
    ["CMKUpscaleSaveModeEnable", "upscale_save_mode", { DISABLE: false, ENABLE: true }],
    ["CMKDetailerAdvancedModeEnable", "detailer_global_enable", { DISABLE: false, ENABLE: true }],
    ["CMKFaceProcessAdvancedModeEnable", "faceprocess_global_enable", { DISABLE: false, ENABLE: true }],
    // Advanced subgraphs saved before the 2.8 MODE node.
    ["CMKDetailerPrepareAdvancedMultiModelPipe", "detailer_global_enable", { DISABLE: false, ENABLE: true }],
    ["CMKFaceProcessPrepareAdvancedMultiModelPipe", "faceprocess_global_enable", { DISABLE: false, ENABLE: true }],
];
for (const [control, widget, expected] of contracts) {
    for (const [value, enabled] of Object.entries(expected)) {
        const { compare } = compareFrom(control, value, { widget });
        assert.equal(displayNodeEnabled(compare), enabled, `${control} = ${value}`);
    }
}

// 2. The control widget promoted to the outer MODE: the outer value decides.
{
    const { origin, compare } = compareFrom("CMKDetailerAdvancedModeEnable", "ENABLE", {
        widget: "detailer_global_enable",
        inputs: [{ name: "detailer_global_enable", widget: { name: "detailer_global_enable" }, link: 7 }],
        links: [{ id: 7, origin_id: -10, origin_slot: 6, target_id: 2, target_slot: 0 }],
    });
    const outer = {
        id: 8627, inputs: Array.from({ length: 7 }, (_, index) => (index === 6
            ? { name: "detailer_global_enable", widget: { name: "detailer_global_enable" } } : { name: `in${index}` })),
        widgets: [{ name: "detailer_global_enable", value: "DISABLE" }],
        subgraph: { nodes: [origin, compare] },
    };
    globalThis.__cmkTestApp.graph = { _nodes: [outer] };
    assert.equal(displayNodeEnabled(compare), false, "outer MODE DISABLE, stale inner ENABLE");
    outer.widgets[0].value = "ENABLE";
    assert.equal(displayNodeEnabled(compare), true, "outer MODE ENABLE");
}

// 3. Unknown producers keep their former reading; no Boolean of strings.
{
    const { compare } = compareFrom("SomeOtherNode", false, { widget: "ENABLE" });
    assert.equal(displayNodeEnabled(compare), false, "a Boolean widget of the output name still counts");
}

// 4. Live preview of another module and interrupted runs.
const sampler = { id: 7652, type: "CMKKSamplerPipe", comfyClass: "CMKKSamplerPipe", inputs: [], outputs: [], widgets: [],
    properties: { cmkVisualProviders: [{ provider_id: "sampling", key: "sampling", live_node_id: "7652", live_node_ids: ["7652"], live_node_resolved: true, capabilities: { live: true } }] } };
const disabled = compareFrom("CMKDetailerAdvancedModeEnable", "DISABLE", { widget: "detailer_global_enable" });
const enabled = compareFrom("CMKDetailerAdvancedModeEnable", "ENABLE", { widget: "detailer_global_enable" });
enabled.compare.id = 4;
globalThis.__cmkTestApp.graph = { _nodes: [sampler, disabled.origin, disabled.compare, enabled.compare] };
const run = (name, detail = {}) => listeners[name]({ detail: { prompt_id: "run-1", ...detail } });

run("execution_start");
run("b_preview_with_metadata", { blob: { size: 1 }, node_id: "7652" });
assert.equal(visualState(disabled.compare).providers.length, 0, "disabled Compare takes no foreign live image");
assert.equal(visualState(enabled.compare).providers.length, 1, "enabled Compare keeps its live preview");
assert.ok(visualState(enabled.compare).providers[0].liveUrl);

run("execution_interrupted");
for (const node of [disabled.compare, enabled.compare]) {
    const state = visualState(node);
    assert.equal(state.runActive, false);
    assert.equal(state.live, null);
    assert.equal(state.providers.length, 0, "no live image stays after an interrupt");
}

// A regular result of the last completed run comes back after an interrupt;
// a result this run already delivered stays.
const previous = { providers: [{ provider_id: "sampling", channels: [{ name: "after", image_index: 0 }], executedUrl: null }],
    images: [{ filename: "done.png" }], providerId: "sampling", channel: null };
visualState(enabled.compare).completedSnapshot = previous;
run("execution_start");
run("b_preview_with_metadata", { blob: { size: 2 }, node_id: "7652" });
run("execution_interrupted");
assert.deepEqual(visualState(enabled.compare).images, [{ filename: "done.png" }], "last completed result restored");
assert.equal(visualState(enabled.compare).providers[0].liveUrl, null);

run("execution_start");
run("executed", { node_id: "7652", output: { images: [{ filename: "new.png", type: "temp", subfolder: "" }] } });
run("execution_interrupted");
assert.ok(visualState(enabled.compare).providers[0].executedUrl, "a result of the interrupted run stays");

// Success and error keep their behavior (handlers unchanged).
assert.equal(typeof listeners.execution_success, "function");
assert.equal(typeof listeners.execution_error, "function");
console.log("visualizer compare enable: OK");
