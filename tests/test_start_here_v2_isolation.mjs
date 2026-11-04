// 01 START HERE v2: isolation from the legacy v1 UI.
//
// v2 must never show the legacy family tabs or legacy controls after any
// widget callback; value-only changes (STOP, IMAGE FIT) must not rebuild the
// layout (prompt geometry and widget order stay untouched).  v1 keeps its
// recorded layout and both versions keep their positional serialization
// (fixtures/frontend/start_here_reference.json, recorded before the change;
// CMK_RECORD=1 re-records it).
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { register } from "node:module";

register("./helpers/cmk_frontend_test_loader.mjs", import.meta.url);

const REFERENCE = new URL("./fixtures/frontend/start_here_reference.json", import.meta.url);
const RECORD = process.env.CMK_RECORD === "1";

class MockElement {
    constructor(tagName = "div") {
        this.tagName = tagName.toUpperCase();
        this.children = [];
        this.listeners = new Map();
        this.style = { setProperty(name, value) { this[name] = value; }, removeProperty(name) { delete this[name]; } };
        this.classList = { values: new Set(), add(...v) { v.forEach((x) => this.values.add(x)); }, toggle(v, on) { if (on) this.values.add(v); else this.values.delete(v); }, contains(v) { return this.values.has(v); } };
        this.dataset = {};
        this.value = "";
        this.disabled = false;
    }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.append(child); return child; }
    replaceChildren(...children) { this.children = [...children]; }
    addEventListener(name, callback) { this.listeners.set(name, callback); }
    dispatch(name) { this.listeners.get(name)?.({ target: this }); }
    click() { this.dispatch("click"); }
    setAttribute(name, value) { this[name] = String(value); }
    querySelectorAll() { return []; }
    querySelector() { return null; }
    closest() { return null; }
    matches() { return false; }
}

const head = new MockElement("head");
head.append = () => {};
globalThis.Element = MockElement;
globalThis.document = {
    head,
    body: new MockElement("body"),
    createElement: (tagName) => new MockElement(tagName),
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
};
globalThis.requestAnimationFrame = (callback) => callback();
globalThis.__cmkTestApp = {
    registerExtension() {},
    graph: { setDirtyCanvas() {} },
    canvas: { graph: null },
    rootGraph: null,
};

const { configureV2 } = await import("../web/js/cmk_flow_start_guidance_v2.js");
const { configureStartBase } = await import("../web/js/cmk_flow_start_guidance_v51.js");
const { modelFamilyFor } = await import("../web/js/cmk_flow_start_v2_core.js");

function widget(name, value, options = {}) {
    return {
        name,
        value,
        type: options.type ?? "combo",
        options: { ...(options.options ?? {}) },
        callback: options.callback,
        computeSize: () => [560, 28],
        draw() {},
        serialize: options.serialize,
    };
}

function canonicalWidgets() {
    return [
        widget("GLOBAL PROMPT POS", "positive", { type: "text" }),
        widget("PROMPT NEG", "negative", { type: "text" }),
        widget("INPAINT_MODE", "Text2Image"),
        widget("resolution", "1152x832", { options: { values: ["1024x1024", "1152x832", "768x512"] } }),
        widget("swap_dimensions", false),
        widget("upscale_method", "lanczos"),
        widget("device", "cpu"),
        widget("outpaint_on", false),
        widget("mask_fill_holes", false),
        widget("fill_masked_area", "neutral"),
        widget("process_mode", "Custom"),
        widget("resize_mode", "Fit", { options: { values: ["Fit", "Crop", "Stretch"] } }),
        widget("crop_position", "Center"),
        widget("outpaint_overlap", 32),
        widget("model_family", "SDXL"),
        widget("instantid_enabled", false),
        widget("instantid_mode", "HYBRID"),
        widget("inpaint_mask_expand", 16),
        widget("hybrid_sdxl_handoff", "90%"),
        widget("hybrid_zit_denoise", "0.20"),
        widget("unload_models_after_use", true),
        widget("HYBRID BALANCE", 2),
        widget("live_preview_zit", true),
        widget("content_guard_enabled", true),
        widget("controlnet_enabled", false),
        widget("instantid_handoff_percent", 70),
        widget("stop_after_handoff", false),
    ];
}

function fakeNode(v2) {
    const triggers = [];
    return {
        id: 1,
        type: "CMKPipeCreateImage",
        comfyClass: "CMKPipeCreateImage",
        properties: v2 ? { cmkStartUiVersion: 2 } : {},
        widgets: canonicalWidgets(),
        inputs: [],
        outputs: [],
        size: [600, 1225],
        flags: {},
        serialize_widgets: true,
        triggers,
        graph: { trigger(name) { triggers.push(name); } },
        addDOMWidget(name, type, element, options = {}) {
            const result = widget(name, options.getValue?.() ?? "", { type, options });
            result.element = element;
            this.widgets.push(result);
            return result;
        },
        addWidget(type, name, value, callback, options = {}) {
            const result = widget(name, value, { type, callback, options, serialize: options.serialize });
            this.widgets.push(result);
            return result;
        },
        setSize(size) { this.size = [...size]; },
        setDirtyCanvas() {},
    };
}

const byName = (node, name) => node.widgets.find((candidate) => candidate.name === name);
const settle = () => new Promise((resolve) => setTimeout(resolve, 300));
const shown = (node, name) => {
    const target = byName(node, name);
    return Boolean(target && target.hidden !== true && node._cmkStartVisibleWidgets?.has(target));
};
const visibleNames = (node) => node.widgets.filter((w) => w.hidden !== true && node._cmkStartVisibleWidgets?.has(w)).map((w) => w.name);

async function fire(node, name, value) {
    const target = byName(node, name);
    target.value = value;
    target.callback?.(value, node, target);
    await settle();
}

function serialized(node) {
    const data = { properties: node.properties };
    node.onSerialize(data);
    return data.widgets_values;
}

// -- reference (recorded before the change) --------------------------------
const MODES = [
    ["SDXL", "Text2Image"], ["SDXL", "Inpaint"], ["Hybrid", "Text2Image"],
    ["Hybrid", "Inpaint"], ["Z-Image Turbo", "Text2Image"], ["Z-Image Turbo", "Inpaint"],
];
const current = { v1: {}, serialization: {} };
for (const [family, mode] of MODES) {
    const v1 = fakeNode(false);
    byName(v1, "model_family").value = family;
    byName(v1, "INPAINT_MODE").value = mode;
    configureStartBase(v1);
    await settle();
    current.v1[`${family}/${mode}`] = visibleNames(v1);
}
{
    const v1 = fakeNode(false);
    configureStartBase(v1);
    await settle();
    current.serialization.v1 = serialized(v1);
    const v2 = fakeNode(true);
    configureV2(v2);
    await settle();
    current.serialization.v2 = serialized(v2);
}
if (RECORD) {
    writeFileSync(REFERENCE, JSON.stringify(current, null, 2) + "\n");
    console.log("recorded", REFERENCE.pathname);
    process.exit(0);
}
const reference = JSON.parse(readFileSync(REFERENCE, "utf8"));
const failures = [];
const check = (label, fn) => {
    try { fn(); } catch (error) { failures.push(`${label}: ${error.message.split("\n")[0]}`); }
};

check("v1 layout unchanged in all six modes", () => assert.deepEqual(current.v1, reference.v1));
check("positional serialization unchanged (v1, v2)", () => assert.deepEqual(current.serialization, reference.serialization));

// -- v2 isolation ----------------------------------------------------------
function assertV2(node, label) {
    check(`${label}: v2 controls on top`, () => assert.deepEqual(node.widgets.slice(0, 5).map((w) => w.name), [
        "GENERATION MODE", "GENERATION ENGINE", "GENERATION STRATEGY", "FINISH", "stop_after_handoff",
    ]));
    check(`${label}: every widget exactly once`, () => assert.equal(new Set(node.widgets).size, node.widgets.length));
    for (const legacy of ["MODEL FAMILY TABS", "instantid_enabled", "model_family"]) {
        check(`${label}: ${legacy} not shown`, () => assert.equal(shown(node, legacy), false));
    }
    check(`${label}: stop shown`, () => assert.equal(shown(node, "stop_after_handoff"), true));
    const state = node.properties.cmkStartV2State;
    const finish = state.finishByContext[`${state.engine === "Z-Image Turbo" ? "zit" : "sdxl"}:${state.strategy === "IDENTITY" ? "identity" : "standard"}`];
    check(`${label}: model_family follows v2 state`, () => assert.equal(byName(node, "model_family").value, modelFamilyFor(state.engine, state.strategy, finish)));
    check(`${label}: instantid_enabled follows v2 state`, () => assert.equal(byName(node, "instantid_enabled").value, state.engine === "SDXL" && state.strategy === "IDENTITY"));
}

const node = fakeNode(true);
configureV2(node);
await settle();
assertV2(node, "initial");
for (const [name, value] of [
    ["instantid_enabled", true],
    ["model_family", "Hybrid"],
    ["INPAINT_MODE", "Inpaint"],
    ["resize_mode", "Crop"],
    ["process_mode", "Replace Object"],
    ["process_mode", "Custom"],
    ["INPAINT_MODE", "Text2Image"],
    ["instantid_mode", "ZERO-PASS"],
    ["resolution", "1024x1024"],
]) {
    await fire(node, name, value);
    assertV2(node, `after ${name}=${value}`);
}
const tabs = byName(node, "MODEL FAMILY TABS");
tabs?.element?.children?.[2]?.click?.();
await settle();
assertV2(node, "after legacy tab click");

// -- value-only changes keep the layout ------------------------------------
async function valueOnly(label, name, value, prepare) {
    const target = fakeNode(true);
    configureV2(target);
    await settle();
    await prepare?.(target);
    const prompts = ["GLOBAL PROMPT POS", "PROMPT NEG"].map((n) => byName(target, n));
    prompts.forEach((prompt) => { prompt.computedHeight = 123; });
    const order = target.widgets;
    const before = visibleNames(target);
    target.triggers.length = 0;
    await fire(target, name, value);
    check(`${label}: prompt geometry untouched`, () => assert.deepEqual(prompts.map((p) => p.computedHeight), [123, 123]));
    check(`${label}: widget list not rebuilt`, () => assert.equal(target.widgets, order));
    check(`${label}: visible widgets unchanged`, () => assert.deepEqual(visibleNames(target), before));
    check(`${label}: no layout event`, () => assert.deepEqual(target.triggers.filter((t) => t === "node:widget:changed"), []));
    assertV2(target, label);
}
await valueOnly("STOP on (SDXL → ZIT)", "stop_after_handoff", true, async (target) => {
    target._cmkStartV2Panels.finish.select.value = "Z-Image Turbo";
    target._cmkStartV2Panels.finish.select.dispatch("change");
    await settle();
});
await valueOnly("IMAGE FIT Crop (Inpaint)", "resize_mode", "Crop", async (target) => {
    await fire(target, "INPAINT_MODE", "Inpaint");
});

// -- layout-relevant value change still applies -----------------------------
{
    const target = fakeNode(true);
    configureV2(target);
    await settle();
    await fire(target, "INPAINT_MODE", "Inpaint");
    check("IMAGE FIT Stretch hides crop_position", () => assert.equal(shown(target, "crop_position"), true));
    await fire(target, "resize_mode", "Stretch");
    check("IMAGE FIT Stretch hides crop_position (after)", () => assert.equal(shown(target, "crop_position"), false));
    assertV2(target, "after Stretch");
}

if (failures.length) {
    console.error(`01 START HERE v2 isolation: ${failures.length} failure(s)`);
    failures.forEach((failure) => console.error(" -", failure));
    process.exit(1);
}
console.log("01 START HERE v2 isolation: OK");
