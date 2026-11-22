// CMK Flow · properties.cmkFixedSize pins the outer node size: it survives a
// graph load (computeSize growth, content fit of the inline video player) and
// both Clean View switches. Nodes without the property keep the previous
// behaviour (expanded size restored on Clean View OFF, content fit on load).
import assert from "node:assert/strict";
import { register } from "node:module";

register("./helpers/cmk_frontend_test_loader.mjs", import.meta.url);

const noop = () => {};
const element = () => ({
    style: {},
    setAttribute: noop,
    append: noop,
    prepend: noop,
    addEventListener: noop,
    load: noop,
    classList: { add: noop, remove: noop, toggle: noop },
});
globalThis.Element = class {};
globalThis.HTMLElement = class {};
globalThis.document = {
    head: { appendChild: noop, append: noop },
    createElement: element,
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    documentElement: {},
};
globalThis.requestAnimationFrame = (callback) => callback();
globalThis.CSS = { escape: String };
const extensions = [];
globalThis.__cmkTestApp = {
    registerExtension: (extension) => extensions.push(extension),
    graph: { setDirtyCanvas: noop },
    canvas: { graph: null },
    rootGraph: null,
};
globalThis.__cmkTestApi = { apiURL: (path) => path };

const { fixedSize, applyFixedSize } = await import("../web/js/cmk_fixed_size.js");
const { setCleanView } = await import("../web/js/cmk_clean_view_prototype.js");
await import("../web/js/cmk_video_inline_players_v1.js");
const cleanView = extensions.find((extension) => extension.name === "cmk.clean_view.prototype.v1");
const inlinePlayers = extensions.find((extension) => extension.name === "cmk.video.inline.players.v1");

function mergeNode(properties) {
    return {
        id: 8,
        type: "CMKMergeAndSaveVideo",
        comfyClass: "CMKMergeAndSaveVideo",
        title: "CMK Merge and Save Video",
        size: [700, 750],
        properties: JSON.parse(JSON.stringify(properties)),
        flags: {},
        inputs: [{ name: "IMAGE" }, { name: "LOG" }],
        outputs: [{ name: "VIDEO" }, { name: "LOG" }],
        widgets: [],
        setSize(size) { this.size = [...size]; },
        computeSize: () => [620, 560],
        setDirtyCanvas: noop,
        addDOMWidget: () => ({}),
    };
}

// Node part of ComfyApp.loadGraphDataNow: grow to computeSize(), then the
// loadedGraphNode handlers.
function load(node) {
    const computed = node.computeSize();
    node.setSize([Math.max(node.size[0], computed[0]), Math.max(node.size[1], computed[1])]);
    cleanView.loadedGraphNode(node);
    inlinePlayers.loadedGraphNode(node);
    return node;
}

const PINNED = {
    cmkCleanView: true,
    cmkFixedSize: [700, 750],
    cmkCleanViewSize: [700, 750],
    cmkCleanViewExpandedSize: [700, 750],
    cmkCleanViewExpandedHeight: 750,
};

// Helper contract.
assert.deepEqual(fixedSize({ properties: { cmkFixedSize: [470.4, 750] } }), [470, 750]);
assert.equal(fixedSize({ properties: { cmkFixedSize: [0, 750] } }), null);
assert.equal(fixedSize({ properties: {} }), null);
assert.equal(applyFixedSize({ properties: {}, size: [1, 1] }), false);

// Load: content fit (computeSize height 560) and computeSize growth do not
// change a pinned node, in either view.
for (const cleanViewOn of [true, false]) {
    const node = load(mergeNode({ ...PINNED, cmkCleanView: cleanViewOn }));
    assert.deepEqual(node.size, [700, 750], `load with Clean View ${cleanViewOn}`);
}

// A larger computeSize than the pinned size is undone on load as well.
{
    const node = mergeNode(PINNED);
    node.computeSize = () => [900, 1100];
    load(node);
    assert.deepEqual(node.size, [700, 750]);
}

// Clean View OFF → ON → OFF keeps the outer size, and the remembered sizes
// follow the pinned one (also when they were stale before).
{
    const node = mergeNode({ ...PINNED, cmkCleanViewExpandedSize: [650, 700], cmkCleanViewExpandedHeight: 700 });
    setCleanView(node, false);
    assert.deepEqual(node.size, [700, 750]);
    setCleanView(node, true);
    assert.deepEqual(node.size, [700, 750]);
    setCleanView(node, false);
    assert.deepEqual(node.size, [700, 750]);
    assert.deepEqual(node.properties.cmkCleanViewExpandedSize, [700, 750]);
    assert.equal(node.properties.cmkCleanViewExpandedHeight, 750);
    assert.deepEqual(node.properties.cmkCleanViewSize, [700, 750]);
}

// Unpinned nodes: unchanged behaviour.
{
    const node = mergeNode({
        cmkCleanView: true,
        cmkCleanViewSize: [700, 750],
        cmkCleanViewExpandedSize: [650, 700],
        cmkCleanViewExpandedHeight: 700,
    });
    setCleanView(node, false);
    assert.deepEqual(node.size, [650, 700], "Clean View OFF restores the expanded size");
    const loaded = mergeNode({ cmkCleanView: false });
    inlinePlayers.loadedGraphNode(loaded);
    assert.deepEqual(loaded.size, [700, 560], "content fit still applies without a pinned size");
}

console.log("fixed node size: OK");
