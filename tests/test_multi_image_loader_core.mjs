// CMK Multi Image Loader: the prompt injection that replaces a shared backend
// ancestor by literal references at every consumer.
import assert from "node:assert/strict";

const core = await import("../web/js/cmk_multi_image_loader_core.js");
const { injectReferences, MultiImageInjectionError, missingFileValue, normalizeState, NODE_TYPE, PROPERTY } = core;

let nextLink = 1;
function graph() {
    const nodes = [];
    const links = new Map();
    return {
        nodes, links, inputs: [],
        getNodeById: (id) => nodes.find((node) => node.id === id),
        add(node) { nodes.push(node); return node; },
        link(source, slot, target, targetSlot) {
            const id = nextLink++;
            links.set(id, { id, origin_id: source.id, origin_slot: slot, target_id: target.id, target_slot: targetSlot });
            if (source.outputs) (source.outputs[slot].links ??= []).push(id);
            return id;
        },
    };
}
const loader = (id, slots, extra = {}) => ({
    id, type: NODE_TYPE, mode: 0, outputs: [{}, {}, {}, {}],
    properties: { [PROPERTY]: { slots, selected: 3 } }, ...extra,
});
const consumer = (id, type = "CMKLoadImage") => ({ id, type, comfyClass: type, mode: 0, inputs: [{ name: "image" }, { name: "opt_image_file" }], outputs: [] });
const promptOf = (...entries) => Object.fromEntries(entries.map(([id, type]) => [id, { class_type: type, inputs: { image: "own.png" } }]));
const SLOTS4 = [
    { file: "cn.png", enabled: true }, { file: "a.png", enabled: true },
    { file: "b.png", enabled: false }, { file: "", enabled: true },
];

// Direct consumers, one source used twice, disabled and empty slots.
{
    const root = graph();
    const multi = root.add(loader(1, SLOTS4));
    const [cn, a1, a2, b, c] = [10, 11, 12, 13, 14].map((id) => root.add(consumer(id)));
    root.link(multi, 0, cn, 1); root.link(multi, 1, a1, 1); root.link(multi, 1, a2, 1);
    root.link(multi, 2, b, 1); root.link(multi, 3, c, 1);
    const prompt = promptOf(["10", "CMKLoadImage"], ["11", "CMKLoadImage"], ["12", "CMKLoadImage"], ["13", "CMKLoadImage"], ["14", "CMKLoadImage"]);
    injectReferences(root, prompt);
    assert.equal(prompt["10"].inputs.opt_image_file, "cn.png");
    assert.equal(prompt["11"].inputs.opt_image_file, "a.png");
    assert.equal(prompt["12"].inputs.opt_image_file, "a.png");
    assert.equal(prompt["13"].inputs.opt_image_file, "", "disabled slot: local fallback");
    assert.equal(prompt["14"].inputs.opt_image_file, missingFileValue(3), "active slot without file");
    assert.ok(!(NODE_TYPE in Object.fromEntries(Object.values(prompt).map((entry) => [entry.class_type, 1]))));
}

// Consumers inside a module, inside a nested subgraph, and leaving a subgraph.
{
    const root = graph();
    const inner = graph();
    const deep = graph();
    const multi = root.add(loader(1, SLOTS4));
    const deepConsumer = deep.add(consumer(7));
    deep.inputs = [{ linkIds: [] }];
    deep.inputs[0].linkIds.push(deep.link({ id: -10 }, 0, deepConsumer, 1));
    const nested = inner.add({ id: 5, mode: 0, inputs: [{ name: "opt_image_file" }], outputs: [], isSubgraphNode: () => true, subgraph: deep });
    const moduleConsumer = inner.add(consumer(6));
    inner.inputs = [{ linkIds: [] }, { linkIds: [] }];
    inner.inputs[0].linkIds.push(inner.link({ id: -10 }, 0, moduleConsumer, 1));
    inner.inputs[1].linkIds.push(inner.link({ id: -10 }, 1, nested, 0));
    const module = root.add({ id: 30, mode: 0, inputs: [{ name: "opt_image_file" }, { name: "opt_image_file_1" }], outputs: [], isSubgraphNode: () => true, subgraph: inner });
    root.link(multi, 1, module, 0);
    root.link(multi, 0, module, 1);
    const prompt = promptOf(["30:6", "CMKLoadImage"], ["30:5:7", "CMKLoadImage"]);
    injectReferences(root, prompt);
    assert.equal(prompt["30:6"].inputs.opt_image_file, "a.png");
    assert.equal(prompt["30:5:7"].inputs.opt_image_file, "cn.png");
}

// A loader inside a subgraph instance reaching consumers in its parent graph.
{
    const root = graph();
    const inner = graph();
    const multi = inner.add(loader(2, SLOTS4));
    const module = root.add({ id: 40, mode: 0, inputs: [], outputs: [{}], isSubgraphNode: () => true, subgraph: inner });
    inner.link(multi, 1, { id: -20 }, 0);
    const outside = root.add(consumer(41));
    root.link(module, 0, outside, 1);
    const prompt = promptOf(["41", "CMKLoadImage"]);
    injectReferences(root, prompt);
    assert.equal(prompt["41"].inputs.opt_image_file, "a.png");
}

// Two loaders never mix their slots; muted consumers are skipped.
{
    const root = graph();
    const first = root.add(loader(1, SLOTS4));
    const second = root.add(loader(2, [{ file: "x.png", enabled: true }, { file: "y.png", enabled: true }, { file: "", enabled: false }, { file: "", enabled: false }]));
    const [p, q, muted] = [10, 11, 12].map((id) => root.add(consumer(id)));
    muted.mode = 2;
    root.link(first, 1, p, 1); root.link(second, 1, q, 1); root.link(second, 0, muted, 1);
    const prompt = promptOf(["10", "CMKLoadImage"], ["11", "CMKLoadImage"]);
    injectReferences(root, prompt);
    assert.equal(prompt["10"].inputs.opt_image_file, "a.png");
    assert.equal(prompt["11"].inputs.opt_image_file, "y.png");
}

// A connected consumer that is missing in the prompt is a technical error.
{
    const root = graph();
    const multi = root.add(loader(1, SLOTS4));
    const target = root.add(consumer(10));
    root.link(multi, 1, target, 1);
    assert.throws(() => injectReferences(root, {}), MultiImageInjectionError);
    assert.throws(() => injectReferences(root, promptOf(["10", "CMKImageFileLoader"])), MultiImageInjectionError);
}

// State normalization: complete, fresh copies.
{
    const stored = { slots: [{ file: "a.png", enabled: false }], selected: 9 };
    const state = normalizeState(stored);
    assert.deepEqual(state.slots.map((slot) => [slot.file, slot.enabled]), [["a.png", false], ["", true], ["", true], ["", true]]);
    assert.equal(state.selected, 0);
    state.slots[0].file = "changed.png";
    assert.equal(stored.slots[0].file, "a.png");
}

console.log("CMK Multi Image Loader core: OK");
