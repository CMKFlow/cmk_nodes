import assert from "node:assert/strict";
import { register } from "node:module";

register("./helpers/cmk_frontend_test_loader.mjs", import.meta.url);

class MockStyle {
    constructor() { this.values = new Map(); }
    getPropertyValue(name) { return this.values.get(name) ?? ""; }
    setProperty(name, value) { this.values.set(name, value); }
}

class MockHTMLElement {
    constructor() {
        this.style = new MockStyle();
        this.offsetHeight = 1500;
    }
}

globalThis.HTMLElement = MockHTMLElement;
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30 };
globalThis.CSS = { escape: (value) => String(value) };
globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.window = { addEventListener() {}, removeEventListener() {}, setTimeout };
globalThis.document = {
    querySelector: () => new MockHTMLElement(),
    addEventListener() {},
    createElement: () => ({}),
};
globalThis.__cmkTestApp = {
    registerExtension() {},
    graph: { setDirtyCanvas() {} },
};

const { minimumSize, normalizeSize } = await import("../web/js/cmk_node_dimensions.js");

const v2 = {
    id: 1,
    type: "CMKPipeCreateImage",
    comfyClass: "CMKPipeCreateImage",
    properties: { cmkStartUiVersion: 2 },
    size: [600, 1225],
};
assert.deepEqual(minimumSize(v2, 450), [450, 700]);
assert.deepEqual(normalizeSize(v2, 450, 700), [450, 700]);
assert.deepEqual(normalizeSize(v2, 449, 699), [450, 700]);
assert.deepEqual(normalizeSize(v2, 520, 760), [520, 760]);

const v1 = {
    id: 2,
    type: "CMKPipeCreateImage",
    comfyClass: "CMKPipeCreateImage",
    properties: {},
    size: [600, 960],
};
assert.deepEqual(minimumSize(v1, 400), [400, 1470]);

console.log("01 START HERE v2 dimensions: OK");
