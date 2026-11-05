import assert from "node:assert/strict";
import { register } from "node:module";

register("./helpers/cmk_frontend_test_loader.mjs", import.meta.url);

class MockClassList {
    constructor() { this.values = new Set(); }
    add(...values) { values.forEach((value) => this.values.add(value)); }
    toggle(value, enabled) {
        if (enabled) this.values.add(value);
        else this.values.delete(value);
    }
    contains(value) { return this.values.has(value); }
}

class MockStyle {
    setProperty(name, value) { this[name] = value; }
    removeProperty(name) { delete this[name]; }
}

class MockElement {
    constructor(tagName = "div") {
        this.tagName = tagName.toUpperCase();
        this.children = [];
        this.listeners = new Map();
        this.style = new MockStyle();
        this.classList = new MockClassList();
        this.dataset = {};
        this.value = "";
        this.disabled = false;
        this.id = "";
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

const elementsById = new Map();
const head = new MockElement("head");
const nodeWidgetGrid = new MockElement("div");
const nodeBody = new MockElement("div");
nodeBody.querySelector = (selector) => (
    selector === '[data-testid="node-widgets"]' ? nodeWidgetGrid : null
);
const rememberElement = (element) => {
    if (element?.id) elementsById.set(element.id, element);
};
head.append = (...children) => children.forEach(rememberElement);
head.appendChild = (child) => { rememberElement(child); return child; };

globalThis.Element = MockElement;
globalThis.document = {
    head,
    body: new MockElement("body"),
    createElement: (tagName) => new MockElement(tagName),
    getElementById: (id) => elementsById.get(id) ?? null,
    querySelector: (selector) => (
        String(selector).includes('data-testid="node-body-1"') ? nodeBody : null
    ),
    querySelectorAll: () => [],
};
globalThis.requestAnimationFrame = (callback) => callback();

const extensions = [];
globalThis.__cmkTestApp = {
    registerExtension(extension) { extensions.push(extension); },
    graph: { setDirtyCanvas() {} },
    canvas: { graph: null },
    rootGraph: null,
};

const { configureV2 } = await import("../web/js/cmk_flow_start_guidance_v2.js");
const { setCleanView } = await import("../web/js/cmk_clean_view_prototype.js");

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
        widget("GLOBAL PROMPT POS", "persistent positive", { type: "text" }),
        widget("PROMPT NEG", "persistent negative", { type: "text" }),
        widget("INPAINT_MODE", "Text2Image"),
        widget("resolution", "1152x832", { options: { values: ["1024x1024", "1152x832", "768x512"] } }),
        widget("swap_dimensions", false),
        widget("upscale_method", "lanczos"),
        widget("device", "cpu"),
        widget("outpaint_on", false),
        widget("mask_fill_holes", false),
        widget("fill_masked_area", "neutral"),
        widget("process_mode", "Custom"),
        widget("resize_mode", "Fit"),
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

function fakeNode(v2 = true) {
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
        graph: { trigger() {} },
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

function byName(node, name) {
    return node.widgets.find((candidate) => candidate.name === name);
}

function count(node, name) {
    return node.widgets.filter((candidate) => candidate.name === name).length;
}

function assertV2Layout(node) {
    const names = node.widgets.map((candidate) => candidate.name);
    assert.deepEqual(names.slice(0, 5), [
        "GENERATION MODE",
        "GENERATION ENGINE",
        "GENERATION STRATEGY",
        "FINISH",
        "stop_after_handoff",
    ]);
    assert.equal(count(node, "GENERATION MODE"), 1);
    assert.equal(count(node, "GENERATION ENGINE"), 1);
    assert.equal(count(node, "GENERATION STRATEGY"), 1);
    assert.equal(count(node, "FINISH"), 1);
    assert.equal(count(node, "stop_after_handoff"), 1);
    const legacy = byName(node, "MODEL FAMILY TABS");
    assert.ok(legacy);
    assert.equal(legacy.hidden, true);
    assert.equal(legacy.options.hidden, true);
    const stop = byName(node, "stop_after_handoff");
    assert.notEqual(stop.hidden, true);
    assert.notEqual(stop.options.hidden, true);
    const expanded = node.properties.cmkStartV2State.generationModeExpanded;
    for (const name of ["GENERATION ENGINE", "GENERATION STRATEGY", "FINISH"]) {
        const control = byName(node, name);
        assert.equal(control.hidden === true, !expanded);
        assert.equal(control.options.hidden === true, !expanded);
    }
    assert.deepEqual(node.getMinSize(), [450, 700]);
    assert.deepEqual(node.min_size, [450, 700]);
    assert.deepEqual(node.minSize, [450, 700]);
    // The v2 layout never writes inline styles onto the frontend's node body
    // or widget grid: those are lost on every Vue remount and made the
    // geometry depend on whether a refresh ran after mounting (STOP bug).
    assert.equal(nodeBody.dataset.cmkStartV2, undefined);
    assert.deepEqual({ ...nodeBody.style }, {});
    assert.deepEqual({ ...nodeWidgetGrid.style }, {});
}

function assertPromptMinima(node, mode) {
    const family = byName(node, "model_family").value;
    const positiveHeight = family === "Hybrid"
        ? (mode === "Inpaint" ? 67 : 143)
        : family === "Z-Image Turbo"
            ? (mode === "Inpaint" ? 158 : 181)
            : (mode === "Inpaint" ? 54 : 132);
    const expected = { "GLOBAL PROMPT POS": positiveHeight };
    if (family !== "Z-Image Turbo") {
        expected["PROMPT NEG"] = mode === "Inpaint" ? 36 : 45.5;
    }
    for (const [name, height] of Object.entries(expected)) {
        const prompt = byName(node, name);
        assert.equal(prompt.options.getMinHeight(), height);
        assert.equal(prompt.options.getMaxHeight, undefined);
        assert.equal(prompt.value, name === "GLOBAL PROMPT POS" ? "persistent positive" : "persistent negative");
        assert.equal(prompt.type, "text");
    }
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 260));

async function switchMode(node, value) {
    const mode = byName(node, "INPAINT_MODE");
    mode.value = value;
    mode.callback?.(value, node, mode);
    await settle();
    assertV2Layout(node);
    assertPromptMinima(node, value);
}

const node = fakeNode(true);
configureV2(node);
configureV2(node);
assertV2Layout(node);
assert.deepEqual(node.size, [600, 1225]);
assertPromptMinima(node, "Text2Image");
assert.equal(node.properties.cmkStartV2State.generationModeExpanded, false);
assert.equal(node._cmkStartV2GenerationMode.chevron.textContent, "▸");
assert.equal(node._cmkStartV2GenerationMode.summary.textContent, "SDXL → Refiner");
assert.equal(node._cmkStartV2GenerationMode.button["aria-expanded"], "false");
assert.equal(node._cmkStartV2Panels.engine.buttons[0].button.style.height, "36px");
assert.equal(node._cmkStartV2Panels.strategy.buttons[0].button.style.height, "36px");
assert.equal(node._cmkStartV2Panels.finish.select.style.height, "36px");
assert.equal(node._cmkStartV2Panels.engine.buttons[0].button.style.fontSize, "12px");
assert.equal(node._cmkStartV2Panels.finish.select.style.fontSize, "12px");
assert.equal(node._cmkStartV2GenerationMode.button.style.height, "36px");
assert.equal(node._cmkStartV2GenerationMode.button.style["min-height"], "36px");
assert.equal(node._cmkStartV2GenerationMode.button.style["max-height"], "36px");
assert.equal(node._cmkStartV2GenerationMode.widget.options.getMinHeight(), 36);
// No computeLayoutSize: the frontend renders the header row as min-content, so
// only the prompt rows take the free node height.
assert.equal(node._cmkStartV2GenerationMode.widget.computeLayoutSize, undefined);
for (const panel of [node._cmkStartV2Panels.engine, node._cmkStartV2Panels.strategy, node._cmkStartV2Panels.finish]) {
    assert.equal(panel.widget.computeLayoutSize, undefined);
}
const sectionHeadings = [
    node._cmkStartV2Panels.engine.heading,
    node._cmkStartV2Panels.strategy.heading,
    node._cmkStartV2Panels.finish.heading,
];
assert.deepEqual(sectionHeadings.map((heading) => heading.textContent), [
    "GENERATION ENGINE",
    "GENERATION STRATEGY",
    "FINISH",
]);
assert.deepEqual(new Set(sectionHeadings.map((heading) => heading.className)), new Set([
    "cmk-start-v2-section-heading",
]));
assert.deepEqual([
    node._cmkStartV2Panels.engine.controls.className,
    node._cmkStartV2Panels.strategy.controls.className,
    node._cmkStartV2Panels.finish.controls.className,
], [
    "cmk-start-v2-control cmk-start-v2-tabs",
    "cmk-start-v2-control cmk-start-v2-tabs",
    "cmk-start-v2-control cmk-start-v2-finish-wrap",
]);
const v2Style = elementsById.get("cmk-start-v2-style").textContent;
assert.match(v2Style, /cmk-start-v2-control[^}]+padding-left:12px/);
assert.match(v2Style, /cmk-start-v2-section-heading[^}]+font-size:11px[^}]+font-weight:600/);
assert.match(v2Style, /button\.is-active[^}]+background:rgba\(255,255,255,\.025\)/);
assert.match(v2Style, /button\.is-active[^}]+color:#62d9d6/);

const promptGeometryBeforeToggle = [
    byName(node, "GLOBAL PROMPT POS").options.getMinHeight,
    byName(node, "PROMPT NEG").options.getMinHeight,
];
node._cmkStartV2GenerationMode.button.click();
assert.deepEqual(node.size, [600, 1425]);
assert.deepEqual(node._cmkBaseSizeForTemporaryUi([450, 900]), [450, 700]);
assert.equal(node._cmkStartV2GenerationMode.chevron.textContent, "▾");
assertV2Layout(node);
node._cmkStartV2GenerationMode.button.click();
assert.deepEqual(node.size, [600, 1225]);
assertV2Layout(node);
assert.deepEqual([
    byName(node, "GLOBAL PROMPT POS").options.getMinHeight,
    byName(node, "PROMPT NEG").options.getMinHeight,
], promptGeometryBeforeToggle);
for (let index = 0; index < 5; index += 1) {
    node._cmkStartV2GenerationMode.button.click();
    node._cmkStartV2GenerationMode.button.click();
}
assert.deepEqual(node.size, [600, 1225]);

for (let index = 0; index < 3; index += 1) {
    await switchMode(node, "Inpaint");
    await switchMode(node, "Text2Image");
}
assert.equal(byName(node, "GLOBAL PROMPT POS").value, "persistent positive");
assert.equal(byName(node, "PROMPT NEG").value, "persistent negative");

node.setSize([450, 700]);
configureV2(node);
assert.deepEqual(node.size, [450, 700]);
assertV2Layout(node);

node._cmkStartV2GenerationMode.button.click();
assert.deepEqual(node.size, [450, 900]);
node._cmkStartV2Panels.engine.buttons.find(({ value }) => value === "Z-Image Turbo").button.click();
assertV2Layout(node);
assert.equal(node._cmkStartV2GenerationMode.summary.textContent, "ZIT");
node._cmkStartV2Panels.engine.buttons.find(({ value }) => value === "SDXL").button.click();
assertV2Layout(node);

node._cmkStartV2Panels.strategy.buttons.find(({ value }) => value === "IDENTITY").button.click();
assertV2Layout(node);
assert.equal(node._cmkStartV2GenerationMode.summary.textContent, "SDXL → InstantID → Refiner");
node._cmkStartV2Panels.strategy.buttons.find(({ value }) => value === "STANDARD").button.click();
assertV2Layout(node);

const finish = node._cmkStartV2Panels.finish.select;
finish.value = "Z-Image Turbo";
finish.dispatch("change");
assert.equal(node._cmkStartV2GenerationMode.summary.textContent, "SDXL → ZIT");
assert.equal(node._cmkStartV2Panels.finish.description.textContent, "SDXL generation → Z-Image Turbo finishing pass");
const stop = byName(node, "stop_after_handoff");
const geometry = (target) => ({
    size: [...target.size],
    order: target.widgets.map((widget) => widget.name),
    hidden: target.widgets.map((widget) => widget.hidden === true),
    prompts: ["GLOBAL PROMPT POS", "PROMPT NEG"].map((name) => {
        const prompt = byName(target, name);
        return [prompt.computeSize, prompt.options.getMinHeight?.(), prompt.options.rows, prompt.computedHeight];
    }),
});
const beforeStop = geometry(node);
let layoutChanges = 0;
const originalTrigger = node.graph.trigger;
node.graph.trigger = function (event, ...rest) {
    if (event === "node:widget:changed") layoutChanges += 1;
    return originalTrigger?.call(this, event, ...rest);
};
for (const value of [true, false, true, false]) {
    stop.value = value;
    stop.callback?.(value, node, stop);
    assert.equal(finish.disabled, value);
    assertV2Layout(node);
    // STOP is a value-only change: same size, order, visibility and prompt geometry.
    assert.deepEqual(geometry(node), beforeStop);
}
assert.equal(layoutChanges, 0);
node.graph.trigger = originalTrigger;

node.flags.advanced = true;
node.setSize([450, 990]);
node._cmkStartV2GenerationMode.button.click();
assert.deepEqual(node.size, [450, 790]);
assertV2Layout(node);
node._cmkStartV2GenerationMode.button.click();
assert.deepEqual(node.size, [450, 990]);
node.flags.advanced = false;
node.setSize([450, 900]);
node._cmkStartV2GenerationMode.button.click();
assert.deepEqual(node.size, [450, 700]);
assertV2Layout(node);

node.setSize([500, 750]);
node.properties.cmkCleanViewSize = [450, 700];
node.inputs = [{ name: "MODEL" }];
node.outputs = [{ name: "PROCESS" }];
node._cmkStartV2GenerationMode.button.click();
assert.deepEqual(node.size, [500, 950]);
setCleanView(node, true);
assert.deepEqual(node.size, [450, 900]);
assert.deepEqual(node.properties.cmkCleanViewExpandedSize, [500, 750]);
setCleanView(node, false);
assert.deepEqual(node.size, [500, 950]);
node._cmkStartV2GenerationMode.button.click();
assert.deepEqual(node.size, [500, 750]);

const serialized = { properties: node.properties };
node.onSerialize(serialized);
assert.equal(serialized.properties.cmkStartUiVersion, 2);
const loaded = fakeNode(true);
loaded.properties = JSON.parse(JSON.stringify(serialized.properties));
loaded.__cmkLoadedFromWorkflow = true;
loaded.size = [500, 750];
configureV2(loaded);
assertV2Layout(loaded);
assert.deepEqual(loaded.size, [500, 750]);
assert.equal(loaded.properties.cmkStartV2State.generationModeExpanded, false);
await switchMode(loaded, "Inpaint");
await switchMode(loaded, "Text2Image");

loaded._cmkStartV2GenerationMode.button.click();
assert.deepEqual(loaded.size, [500, 950]);
const expandedSerialized = { properties: loaded.properties };
loaded.onSerialize(expandedSerialized);
const expandedLoaded = fakeNode(true);
expandedLoaded.properties = JSON.parse(JSON.stringify(expandedSerialized.properties));
expandedLoaded.__cmkLoadedFromWorkflow = true;
expandedLoaded.size = [500, 950];
configureV2(expandedLoaded);
assertV2Layout(expandedLoaded);
assert.deepEqual(expandedLoaded.size, [500, 950]);
assert.equal(expandedLoaded._cmkStartV2GenerationMode.chevron.textContent, "▾");
expandedLoaded._cmkStartV2GenerationMode.button.click();
assert.deepEqual(expandedLoaded.size, [500, 750]);

const v1 = fakeNode(false);
const originalV1Widgets = [...v1.widgets];
configureV2(v1);
assert.deepEqual(v1.widgets, originalV1Widgets);
assert.equal(count(v1, "GENERATION ENGINE"), 0);
assert.equal(count(v1, "GENERATION STRATEGY"), 0);
assert.equal(count(v1, "FINISH"), 0);

assert.equal(extensions.length, 3);
console.log("01 START HERE v2 lifecycle/rebuild regression: OK");
