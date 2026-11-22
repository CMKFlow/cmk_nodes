// CMK Flow · 30 PostProcess Setup v2: Clean View keeps the compact and the
// expanded size the user chose, also when the graph is loaded again (workflow
// tab switch, page reload); the legacy Setup keeps its fixed 230 × 100.
import assert from "node:assert/strict";
import { register } from "node:module";

register("./helpers/cmk_frontend_test_loader.mjs", import.meta.url);

const noop = () => {};
globalThis.Element = class {};
globalThis.HTMLElement = class {};
globalThis.document = {
    head: { appendChild: noop },
    createElement: () => ({ style: {}, setAttribute: noop }),
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    documentElement: {},
};
globalThis.requestAnimationFrame = noop;
globalThis.CSS = { escape: String };
const extensions = [];
globalThis.__cmkTestApp = {
    registerExtension: (extension) => extensions.push(extension),
    graph: { setDirtyCanvas: noop },
    canvas: { graph: null },
    rootGraph: null,
};

const { setCleanView } = await import("../web/js/cmk_clean_view_prototype.js");
const cleanView = extensions.find((extension) => extension.name === "cmk.clean_view.prototype.v1");

// LiteGraph's estimate over every port and widget row of the Setup v2.
const COMPUTED = [286, 406];

// The node part of ComfyApp.loadGraphDataNow, run on every workflow tab
// switch and page load: configure, grow to computeSize(), loadedGraphNode.
function loadAgain(node) {
    const loaded = setupNode(node.type);
    loaded.size = [...node.size];
    loaded.properties = JSON.parse(JSON.stringify(node.properties));
    const computed = loaded.computeSize();
    loaded.setSize([Math.max(loaded.size[0], computed[0]), Math.max(loaded.size[1], computed[1])]);
    cleanView.loadedGraphNode(loaded);
    return loaded;
}

function setupNode(type) {
    return {
        id: 1,
        type,
        comfyClass: type,
        title: "CMK Flow · 30 PostProcess Setup",
        size: [600, 500],
        properties: {},
        flags: {},
        inputs: ["MODEL ZIT", "PROCESS ZIT", "MODEL SDXL", "PROCESS SDXL"].map((name) => ({ name })),
        outputs: ["MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL"].map((name) => ({ name })),
        setSize(size) { this.size = [...size]; },
        computeSize: () => [...COMPUTED],
        setDirtyCanvas: noop,
    };
}

// A drag in Clean View stores the compact size (CMK Node Dimensions applySize).
function dragCompact(node, size) {
    node.setSize(size);
    node.properties.cmkCleanViewSize = [...size];
}

// A drag in the normal view stores the expanded size.
function dragExpanded(node, size) {
    node.setSize(size);
    node.properties.cmkCleanViewExpandedSize = [...size];
    node.properties.cmkCleanViewExpandedHeight = size[1];
}

{
    const node = setupNode("CMKPostProcessSetupV2Pipe");
    dragExpanded(node, [680, 555]);
    setCleanView(node, true);
    assert.deepEqual(node.size, [230, 100], "first compact size");
    dragCompact(node, [326, 155]);
    for (let round = 0; round < 4; round += 1) {
        setCleanView(node, false);
        assert.deepEqual(node.size, [680, 555], "expanded size restored");
        setCleanView(node, true);
        assert.deepEqual(node.size, [326, 155], "the chosen compact size is kept");
    }
    setCleanView(node, false);
    dragExpanded(node, [720, 600]);
    setCleanView(node, true);
    assert.deepEqual(node.size, [326, 155]);
    assert.deepEqual(node.properties.cmkCleanViewExpandedSize, [720, 600]);
    setCleanView(node, false);
    assert.deepEqual(node.size, [720, 600]);

    // Saved and loaded in either state: the properties carry both sizes.
    const loaded = setupNode("CMKPostProcessSetupV2Pipe");
    loaded.properties = JSON.parse(JSON.stringify(node.properties));
    loaded.size = [720, 600];
    setCleanView(loaded, true);
    assert.deepEqual(loaded.size, [326, 155]);
    setCleanView(loaded, false);
    assert.deepEqual(loaded.size, [720, 600]);
}

{
    // A chosen height below the computeSize() estimate survives every reload,
    // in both states and across repeated switches.
    let node = setupNode("CMKPostProcessSetupV2Pipe");
    dragExpanded(node, [680, 346]);
    node = loadAgain(node);
    assert.deepEqual(node.size, [680, 346], "normal view: no jump to the computeSize() height");
    setCleanView(node, true);
    dragCompact(node, [386, 201]);
    node = loadAgain(node);
    assert.deepEqual(node.size, [386, 201], "Clean View: compact size kept");
    for (let round = 0; round < 3; round += 1) {
        setCleanView(node, false);
        node = loadAgain(node);
        assert.deepEqual(node.size, [680, 346]);
        setCleanView(node, true);
        node = loadAgain(node);
        assert.deepEqual(node.size, [386, 201]);
    }
    setCleanView(node, false);
    dragExpanded(node, [700, 300]);
    node = loadAgain(node);
    assert.deepEqual(node.size, [700, 300], "the last chosen size counts");
    assert.deepEqual(node.properties.cmkCleanViewSize, [386, 201]);

    const legacy = loadAgain(Object.assign(setupNode("CMKPostProcessSetupPipe"), {
        size: [680, 346], properties: { cmkCleanViewExpandedSize: [680, 346] },
    }));
    assert.deepEqual(legacy.size, [680, 406], "the legacy Setup is unchanged");
}

{
    const legacy = setupNode("CMKPostProcessSetupPipe");
    setCleanView(legacy, true);
    dragCompact(legacy, [326, 155]);
    setCleanView(legacy, false);
    setCleanView(legacy, true);
    assert.deepEqual(legacy.size, [230, 100], "the legacy Setup is unchanged");
}

console.log("postprocess setup v2 clean view: OK");
