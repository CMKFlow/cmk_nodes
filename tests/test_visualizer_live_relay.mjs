// STOP AFTER HANDOFF, Hybrid boundary HIT: the SDXL first pass is not sampled;
// CMKFamilyBranchGateSDXLSampled publishes the restored image as the stage
// preview.  The Visualizer must map that preview to the first-pass provider
// (live node CMKKSamplerPipe) without changing the provider identity.
import assert from "node:assert/strict";
import { register } from "node:module";

register("./helpers/cmk_frontend_test_loader.mjs", import.meta.url);

const listeners = {};
globalThis.__cmkTestApp = { registerExtension() {}, graph: { _nodes: [] } };
globalThis.__cmkTestApi = { addEventListener(name, handler) { listeners[name] = handler; } };
globalThis.document = {
    createElement: () => ({ style: {}, set textContent(value) { this.text = value; } }),
    head: { append() {} },
};

const { remappedDeclaredProvider, providerForNode } = await import("../web/js/cmk_visualizer.js");

const node = (id, type) => ({ id, type, comfyClass: type, widgets: [] });
const firstPass = {
    id: 8430,
    subgraph: {
        nodes: [
            node(7414, "CMKCheckpointVAELoaderPipe"),
            node(7415, "CMKKSamplerPipe"),
            node(7416, "CMKSamplerPrepareSDXLPipe"),
            node(7417, "CMKFamilyBranchGateSDXLSampled"),
        ],
    },
};
const declared = {
    provider_id: "cmk-CMKVisualProvider-6302", key: "first-pass", label: "1st-Pass SDXL",
    sequence: 10, live_node_id: "7415", capabilities: { preview: true, live: true },
};

const provider = remappedDeclaredProvider(firstPass, declared);
// Identity unchanged: the KSampler stays the live node and names the provider.
assert.equal(provider.provider_id, "cmk-CMKKSamplerPipe-8430-7415");
assert.equal(provider.live_node_id, "8430:7415");
assert.deepEqual(provider.live_node_ids, ["8430:7415", "8430:7417"]);

const state = { catalog: [provider] };
// Fresh STOP run: the KSampler's preview (unchanged).
assert.equal(providerForNode(state, "8430:7415"), provider);
// Boundary HIT: the gate's preview reaches the same provider.
assert.equal(providerForNode(state, "8430:7417"), provider);
// Other inner nodes still do not.
assert.equal(providerForNode(state, "8430:7416"), null);

// ZIT sampling subgraph has no SDXL sampled gate: unchanged single live node.
const zit = {
    id: 8431,
    subgraph: { nodes: [node(7652, "CMKKSamplerPipe"), node(7655, "CMKFamilyBranchGateZImage")] },
};
const zitProvider = remappedDeclaredProvider(zit, { key: "sampling", provider_id: "x", capabilities: { live: true } });
assert.deepEqual(zitProvider.live_node_ids, ["8431:7652"]);
assert.equal(providerForNode({ catalog: [zitProvider] }, "8431:7655"), null);

// A provider whose live type is not the KSampler gets no relay.
const refiner = {
    id: 8438,
    subgraph: { nodes: [node(7809, "CMKRefinerPipe"), node(7417, "CMKFamilyBranchGateSDXLSampled")] },
};
assert.deepEqual(remappedDeclaredProvider(refiner, { key: "refiner", capabilities: { live: true } }).live_node_ids,
    ["8438:7809"]);

console.log("visualizer live relay: ok");
