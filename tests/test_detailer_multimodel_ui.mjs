import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const sourceUrl = new URL("../web/js/cmk_detailer_multimodel_ui.js", import.meta.url);
const source = readFileSync(sourceUrl, "utf8")
  .replace('import { app } from "../../../scripts/app.js";', "")
  .replace('import { api } from "../../../scripts/api.js";', "");

let extension = null;
const listeners = {};
const app = {
  graph: { setDirtyCanvas() {} },
  registerExtension(value) { extension = value; },
};
const api = { addEventListener(name, callback) { listeners[name] = callback; } };
// Minimal DOM for the Family Enable choice fields.
class FakeElement {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.style = {}; this.dataset = {}; this.children = [];
    this.attributes = {}; this.listeners = {}; this.textContent = ""; this.className = "";
  }
  append(...items) { this.children.push(...items); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  addEventListener(name, callback) { (this.listeners[name] ??= []).push(callback); }
  querySelectorAll(selector) { return selector === "button" ? this.children.filter((c) => c.tagName === "BUTTON") : []; }
  click() { for (const callback of this.listeners.click ?? []) callback({ stopPropagation() {} }); }
}
const document = { createElement: (tag) => new FakeElement(tag) };
vm.runInNewContext(source, { app, api, document, setTimeout(callback) { callback(); } });
assert.ok(extension, "Multi-Model Detailer UI extension was not registered");
extension.setup();

// Backend widget order of CMKDetailerPrepareMultiModelPipe; it is also the
// display order (no reordering, so widgets_values stay positional).
const PREPARE = [
  ["model_family", "None"], ["sam_model_name", "sam_vit_b_01ec64.pth"], ["sam_device_mode", "CPU"],
  ["use_prompt_lora_from_sampler", true], ["use_lora_from_1st_pass", false], ["use_1st_pass_sampling", true],
  ["lora_name", "None"], ["strength_model", 1], ["strength_clip", 1], ["prompt_pos", ""], ["prompt_neg", ""],
  ["steps", 30], ["cfg", 7], ["sampling_entry", 95], ["zit_steps", 8], ["zit_cfg", 1], ["zit_sampling_entry", 85],
  ["sampler", "euler"], ["scheduler", "simple"], ["zit_sampler", "res_multistep"], ["zit_scheduler", "simple"],
  ["stop_at_clip_layer", -2], ["pag_scale", 2.5], ["sampling", "v_prediction"], ["zsnr", true],
  ["zit_model_shift", 3], ["freeu_enabled", true], ["freeu_b1", 1.3], ["freeu_b2", 1.4], ["freeu_s1", 0.9],
  ["freeu_s2", 0.2],
];
const DETAILER = [
  ["output_image_proceed", true], ["model_name", "bbox/hand_yolov8s.pt"],
  ["bbox_threshold", 0.5], ["bbox_dilation", 0], ["crop_factor", 3], ["drop_size", 10], ["guide_size", 512],
  ["guide_size_for", true], ["max_size", 768], ["noise_mask", true], ["force_inpaint", true],
];
const SDXL_ONLY = ["use_1st_pass_sampling", "lora_name", "strength_model", "strength_clip", "prompt_neg",
  "steps", "cfg", "sampling_entry", "sampler", "scheduler", "stop_at_clip_layer", "pag_scale", "sampling",
  "zsnr", "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2"];
const ZIT_ONLY = ["zit_steps", "zit_cfg", "zit_sampling_entry", "zit_sampler", "zit_scheduler", "zit_model_shift"];
const SHARED = ["model_family", "sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler",
  "use_lora_from_1st_pass", "prompt_pos"];

function nodeType(comfyClass, widgets) {
  return class {
    constructor() {
      this.comfyClass = comfyClass;
      this.size = [450, 830];
      this.widgets = widgets.map(([name, value]) => ({
        name, value, options: {}, computeSize: () => [200, 24], draw() {},
      }));
    }
    configure(data) {
      data.widgets_values.forEach((value, index) => { this.widgets[index].value = value; });
      this.onConfigure?.(data);
    }
    serialize() {
      return JSON.parse(JSON.stringify({ widgets_values: this.widgets.map((widget) => widget.value) }));
    }
  };
}

const PrepareNode = nodeType("CMKDetailerPrepareMultiModelPipe", PREPARE);
const DetailerNode = nodeType("CMK_SmartDetailerMultiModelPipe", DETAILER);
const prepareDef = {
  name: "CMKDetailerPrepareMultiModelPipe",
  input: {
    required: { "MODEL SDXL": ["CMK_MODEL_PIPE", { lazy: true }], PROCESS: ["CMK_RESULT_PROCESS", { lazy: true }] },
    optional: { LOG: ["CMK_LOG_PIPE", { lazy: true }], "MODEL ZIT": ["CMK_MODEL_PIPE", { lazy: true }] },
  },
  input_order: { required: ["MODEL SDXL", "PROCESS"], optional: ["LOG", "MODEL ZIT"] },
};
extension.beforeRegisterNodeDef(PrepareNode, prepareDef);

// MODEL ZIT is listed first, still optional; MODEL SDXL follows.
assert.deepEqual(Object.keys(prepareDef.input.required), ["MODEL ZIT", "MODEL SDXL", "PROCESS"]);
assert.equal(prepareDef.input.required["MODEL ZIT"][1].isOptional, true);
assert.equal("MODEL ZIT" in prepareDef.input.optional, false);
assert.equal(prepareDef.input_order, undefined);

// Saved MODEL sockets (before the rename) become MODEL SDXL, the sockets take
// the current order and their links follow, in workflows and subgraph definitions.
{
  const savedPrepare = (id, model, zit) => ({
    id, type: "CMKDetailerPrepareMultiModelPipe",
    inputs: [
      { name: "MODEL", label: "MODEL", link: model }, { name: "PROCESS", link: model + 1 },
      { name: "IMAGE", link: model + 2 }, { name: "LOG", link: model + 3 }, { name: "MODEL ZIT", link: zit },
      { name: "LoRA ZIT", link: null }, { name: "model_family", widget: { name: "model_family" }, link: null },
    ],
  });
  const graphData = {
    nodes: [savedPrepare(5, 10, 20), { id: 6, type: "Other", inputs: [{ name: "MODEL", link: 30 }] }],
    links: [[10, 1, 0, 5, 0, "CMK_MODEL_PIPE"], [11, 1, 1, 5, 1, "X"], [12, 1, 2, 5, 2, "IMAGE"],
      [13, 1, 3, 5, 3, "LOG"], [20, 2, 0, 5, 4, "CMK_MODEL_PIPE"], [30, 1, 0, 6, 0, "CMK_MODEL_PIPE"]],
    definitions: { subgraphs: [{ nodes: [savedPrepare(8, 40, 50)], links: [
      { id: 40, target_id: 8, target_slot: 0 }, { id: 50, target_id: 8, target_slot: 4 }] }] },
  };
  extension.beforeConfigureGraph(graphData);
  const migrated = JSON.parse(JSON.stringify(graphData));
  const prepare = migrated.nodes[0];
  assert.deepEqual(prepare.inputs.map((input) => `${input.name}:${input.link}`),
    ["MODEL ZIT:20", "MODEL SDXL:10", "PROCESS:11", "IMAGE:12", "LOG:13", "LoRA ZIT:null", "model_family:null"]);
  assert.deepEqual(migrated.links.map((link) => link[4]), [1, 2, 3, 4, 0, 0]);
  assert.equal(migrated.nodes[1].inputs[0].name, "MODEL", "other node types stay untouched");
  const inner = migrated.definitions.subgraphs[0];
  assert.deepEqual(inner.nodes[0].inputs.slice(0, 2).map((input) => input.name), ["MODEL ZIT", "MODEL SDXL"]);
  assert.deepEqual(inner.links.map((link) => link.target_slot), [1, 0]);
  // Already migrated data stays as it is.
  const snapshot = JSON.stringify(graphData);
  extension.beforeConfigureGraph(graphData);
  assert.equal(JSON.stringify(graphData), snapshot);
}

extension.beforeRegisterNodeDef(DetailerNode, { name: "CMK_SmartDetailerMultiModelPipe" });

function create(Type) {
  const node = new Type();
  node.onNodeCreated?.();
  extension.nodeCreated(node);
  return node;
}

function load(Type, values) {
  const node = new Type();
  node.onNodeCreated?.();
  extension.nodeCreated(node);
  node.configure({ widgets_values: [...values] });
  extension.loadedGraphNode(node);
  return node;
}

const byName = (node, name) => node.widgets.find((widget) => widget.name === name);
const visible = (node, name) => !byName(node, name).hidden;

function select(node, family) {
  const widget = byName(node, "model_family");
  widget.value = family;
  widget.callback(family);
}

function assertState(node, family) {
  assert.equal(byName(node, "model_family").value, family);
  // None is the complete bypass: only MODEL FAMILY remains visible.
  for (const name of SHARED) assert.equal(visible(node, name), name === "model_family" || family !== "None", `${family}: ${name} shared`);
  for (const name of SDXL_ONLY) assert.equal(visible(node, name), family === "SDXL", `${family}: ${name}`);
  for (const name of ZIT_ONLY) assert.equal(visible(node, name), family === "Z-Image Turbo", `${family}: ${name}`);
}

// New Prepare: starts as None, labels follow the CMK style.
{
  const node = create(PrepareNode);
  assertState(node, "None");
  assert.equal(byName(node, "model_family").label, "MODEL FAMILY");
  assert.equal(byName(node, "sampling_entry").label, "SAMPLING ENTRY AT %");
  assert.equal(byName(node, "zit_sampling_entry").label, "SAMPLING ENTRY AT %");
  assert.equal(byName(node, "zit_sampler").label, "sampler");
  assert.equal(node.widgets.length, PREPARE.length, "hidden widgets stay part of the prompt");
  assert.deepEqual(node.widgets.map((widget) => widget.name), PREPARE.map(([name]) => name),
    "display order is the backend order");
}

// Switching keeps every family value; None hides without resetting.
{
  const node = create(PrepareNode);
  select(node, "SDXL");
  assertState(node, "SDXL");
  byName(node, "sampler").value = "euler";
  byName(node, "scheduler").value = "karras";
  byName(node, "sampling_entry").value = 93;
  const size = [...node.size];
  select(node, "Z-Image Turbo");
  assertState(node, "Z-Image Turbo");
  assert.deepEqual([...node.size], size, "switching keeps the node size");
  byName(node, "zit_sampling_entry").value = 82;
  byName(node, "zit_scheduler").value = "beta";
  select(node, "None");
  assertState(node, "None");
  select(node, "SDXL");
  assert.equal(byName(node, "scheduler").value, "karras");
  assert.equal(byName(node, "sampling_entry").value, 93);
  select(node, "Z-Image Turbo");
  assert.equal(byName(node, "zit_scheduler").value, "beta");
  assert.equal(byName(node, "zit_sampling_entry").value, 82);

  // Save and reload: positional values, identical state.
  const saved = node.serialize().widgets_values;
  const reloaded = load(PrepareNode, saved);
  assertState(reloaded, "Z-Image Turbo");
  assert.deepEqual(reloaded.serialize().widgets_values, saved);
  select(reloaded, "SDXL");
  assert.equal(byName(reloaded, "sampling_entry").value, 93);
}

// Smart Detailer Multi-Model: CMK labels, no sampling entry widget, previews cleared per run.
{
  const node = create(DetailerNode);
  assert.equal(byName(node, "model_name").label, "DETECT MODEL");
  assert.equal(byName(node, "denoise"), undefined);
  node.imgs = ["old"];
  listeners.execution_start();
  assert.equal(node.imgs.length, 0);
  assert.equal(byName(node, "enable"), undefined, "no second activation path");
}

// Smart Detailer Multi-Model: NONE of the feeding Prepare (Advanced: of its
// Detailer slot) hides every widget; values and node size stay.
{
  const nodes = new Map();
  const links = new Map();
  const graph = { links, getNodeById: (id) => nodes.get(id) };
  const add = (node, id, outputs) => {
    Object.assign(node, { id, graph, outputs, size: [450, 830],
      setSize() { throw new Error("the Smart Detailer must not be resized"); } });
    nodes.set(id, node);
    return node;
  };
  const prepare = add(create(PrepareNode), 1, [{ name: "DETAILER", links: [5] }]);
  const detailer = add(create(DetailerNode), 2, []);
  detailer.inputs = [{ name: "DETAILER", link: 5 }];
  links.set(5, { id: 5, origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0 });
  byName(detailer, "crop_factor").value = 2.5;
  const shownDetailer = () => detailer.widgets.filter((w) => !w.hidden).map((w) => w.name);
  extension.loadedGraphNode(detailer);
  assert.deepEqual(shownDetailer(), [], "NONE: no Smart Detailer widget");
  select(prepare, "SDXL");
  assert.deepEqual(shownDetailer(), DETAILER.map(([name]) => name), "SDXL: every widget");
  select(prepare, "None");
  assert.deepEqual(shownDetailer(), []);
  select(prepare, "Z-Image Turbo");
  assert.deepEqual(shownDetailer(), DETAILER.map(([name]) => name), "ZIT: every widget");
  assert.equal(byName(detailer, "crop_factor").value, 2.5, "values survive NONE");
  assert.deepEqual(detailer.serialize().widgets_values.length, DETAILER.length, "hidden widgets are still serialized");
  assert.deepEqual(detailer.size, [450, 830], "no automatic resize");
  // Advanced Prepare: DETAILER 2 follows model_family_2.
  const advanced = add({ comfyClass: "CMKDetailerPrepareAdvancedMultiModelPipe",
    widgets: [1, 2, 3].map((slot) => ({ name: `model_family_${slot}`, value: slot === 2 ? "None" : "SDXL" })) }, 3, []);
  links.set(6, { id: 6, origin_id: 3, origin_slot: 1, target_id: 2, target_slot: 0 });
  detailer.inputs[0].link = 6;
  detailer._cmkRefreshDetailer();
  assert.deepEqual(shownDetailer(), [], "Advanced slot 2 is NONE");
  advanced.widgets[1].value = "SDXL";
  detailer._cmkRefreshDetailer();
  assert.deepEqual(shownDetailer(), DETAILER.map(([name]) => name));
  // Not linked to a Prepare: the family is unknown, every widget stays reachable.
  detailer.inputs[0].link = null;
  detailer._cmkRefreshDetailer();
  assert.deepEqual(shownDetailer(), DETAILER.map(([name]) => name));
}

// No ENABLE migration (2.8 development states need none): the graph hook
// only reorders the Prepare sockets and leaves the Smart Detailer untouched.
{
  const values = DETAILER.map(([, value]) => value);
  const graphData = { nodes: [{ id: 7, type: "CMK_SmartDetailerMultiModelPipe", widgets_values: [...values],
    inputs: [{ name: "DETAILER", link: 1 }] }], links: [[1, 2, 0, 7, 0, "X"]], definitions: { subgraphs: [] } };
  const before = JSON.stringify(graphData);
  extension.beforeConfigureGraph(graphData);
  assert.equal(JSON.stringify(graphData), before);
  const node = new DetailerNode();
  node.configure({ widgets_values: [...values] });
  assert.equal(byName(node, "model_name").value, "bbox/hand_yolov8s.pt");
}

// CMK Detailer Family Enable: NONE | SDXL | ZIT replace the dropdown; the
// stored value stays the model family name.
const FAMILIES = ["None", "SDXL", "Z-Image Turbo"];
class ControlNode {
  constructor() {
    this.comfyClass = "CMKDetailerFamilyEnable";
    this.size = [450, 130];
    this.widgets = [{ name: "model_family", value: "None", options: { values: FAMILIES }, computeSize: () => [200, 24], draw() {}, callback() {} }];
    this.inputs = [{ name: "model_family", widget: { name: "model_family" }, link: null }];
    this.outputs = [{ name: "ENABLE", links: [] }, { name: "MODEL FAMILY", links: [] }];
  }
  addDOMWidget(name, type, element, options) {
    const widget = { name, type, element, options: { ...options }, computeSize: () => [200, 46], draw() {} };
    this.widgets.push(widget);
    return widget;
  }
  configure(data) {
    data.widgets_values.forEach((value, index) => { this.widgets[index].value = value; });
    this.onConfigure?.(data);
  }
  serialize() {
    return { widgets_values: this.widgets.filter((w) => w.options?.serialize !== false).map((w) => w.value) };
  }
}
extension.beforeRegisterNodeDef(ControlNode, { name: "CMKDetailerFamilyEnable" });
const buttons = (node) => byName(node, "cmk_family_choice").element.querySelectorAll("button");
const pressed = (node) => buttons(node).filter((b) => b.getAttribute("aria-pressed") === "true").map((b) => b.textContent);
const press = (node, label) => buttons(node).find((b) => b.textContent === label).click();
{
  const first = create(ControlNode);
  const second = create(ControlNode);
  assert.deepEqual(buttons(first).map((b) => [b.textContent, b.dataset.value]),
    [["NONE", "None"], ["SDXL", "SDXL"], ["ZIT", "Z-Image Turbo"]]);
  assert.ok(byName(first, "model_family").hidden, "the dropdown is replaced, not duplicated");
  assert.equal(byName(first, "cmk_family_choice").options.serialize, false, "the choice fields are not a prompt input");
  assert.deepEqual(pressed(first), ["NONE"]);
  press(first, "SDXL");
  assert.equal(byName(first, "model_family").value, "SDXL");
  assert.deepEqual(pressed(first), ["SDXL"]);
  const active = buttons(first).find((b) => b.textContent === "SDXL");
  assert.equal(active.style.color, "rgb(88, 190, 230)", "active text in the CMK accent");
  assert.match(active.style.border, /rgba\(88, 190, 230/, "active outline");
  assert.equal(active.style.background, "rgba(255, 255, 255, 0.025)", "no accent fill");
  press(first, "ZIT");
  assert.equal(byName(first, "model_family").value, "Z-Image Turbo", "ZIT is only the caption");
  // Instances are independent.
  assert.equal(byName(second, "model_family").value, "None");
  assert.deepEqual(pressed(second), ["NONE"]);
  // Save and load keep the selection; only model_family is stored.
  const saved = first.serialize().widgets_values;
  assert.deepEqual(saved, ["Z-Image Turbo"]);
  const reloaded = new ControlNode();
  reloaded.onNodeCreated?.(); extension.nodeCreated(reloaded);
  reloaded.configure({ widgets_values: saved });
  extension.loadedGraphNode(reloaded);
  assert.deepEqual(pressed(reloaded), ["ZIT"]);
  // A linked selection (Standard subgraph) hides the choice fields.
  const linked = new ControlNode();
  linked.inputs[0].link = 7;
  linked.onNodeCreated?.(); extension.nodeCreated(linked);
  assert.ok(byName(linked, "cmk_family_choice").hidden);
}
// A Prepare linked to MODEL FAMILY follows the control.
{
  const control = create(ControlNode);
  const prepare = create(PrepareNode);
  const graph = { links: { 9: { target_id: 42 } }, getNodeById: (id) => (id === 42 ? prepare : null) };
  control.graph = graph;
  control.outputs[1].links = [9];
  press(control, "SDXL");
  assertState(prepare, "SDXL");
  press(control, "ZIT");
  assertState(prepare, "Z-Image Turbo");
}

// Multi-Model Mask Detailer: the same Family Enable, family widget sets of its
// own, NONE shows only MODEL FAMILY, the Process follows its Prepare.
{
  const MASK_PREPARE = [["model_family", "None"], ["fill_mask_holes", false], ["solid_mask", false],
    ["lora_name", "None"], ["lora_strength", 1], ["zit_lora_name", "None"], ["zit_lora_strength", 1], ["prompt_pos", ""],
    ["steps", 30], ["cfg", 7], ["sampling_entry", 45], ["zit_steps", 8], ["zit_cfg", 1], ["zit_sampling_entry", 45],
    ["sampler", "euler"], ["scheduler", "karras"], ["zit_sampler", "res_multistep"], ["zit_scheduler", "simple"],
    ["zit_model_shift", 3]];
  const MaskPrepare = nodeType("CMKMaskDetailerPrepareMultiModel", MASK_PREPARE);
  const MaskProcess = nodeType("CMKMaskDetailerProcessMultiModel", [["mask_expand", 8], ["mask_feather", 12]]);
  const maskDef = {
    name: "CMKMaskDetailerPrepareMultiModel",
    input: { required: { "MODEL SDXL": ["CMK_MODEL_PIPE", { lazy: true }] },
      optional: { "MODEL ZIT": ["CMK_MODEL_PIPE", { lazy: true }], PROCESS: ["CMK_RESULT_PROCESS"] } },
  };
  extension.beforeRegisterNodeDef(MaskPrepare, maskDef);
  extension.beforeRegisterNodeDef(MaskProcess, { name: "CMKMaskDetailerProcessMultiModel" });
  assert.deepEqual(Object.keys(maskDef.input.required), ["MODEL ZIT", "MODEL SDXL"], "MODEL ZIT above MODEL SDXL");

  const nodes = new Map();
  const links = new Map();
  const graph = { links, getNodeById: (id) => nodes.get(id) };
  const prepare = create(MaskPrepare);
  Object.assign(prepare, { id: 11, graph, outputs: [{ name: "MODEL", links: [] }, { name: "MASK DETAILER PROCESS", links: [15] }] });
  const process = create(MaskProcess);
  Object.assign(process, { id: 12, graph, inputs: [{ name: "MODEL", link: null }, { name: "MASK DETAILER PROCESS", link: 15 }] });
  nodes.set(11, prepare); nodes.set(12, process);
  links.set(15, { id: 15, origin_id: 11, origin_slot: 1, target_id: 12, target_slot: 1 });
  const shown = (node) => node.widgets.filter((w) => !w.hidden).map((w) => w.name);
  const always = ["model_family", "fill_mask_holes", "solid_mask", "prompt_pos"];
  select(prepare, "None");
  assert.deepEqual(shown(prepare), ["model_family"], "NONE: only MODEL FAMILY");
  assert.deepEqual(shown(process), [], "NONE: the Process shows nothing");
  select(prepare, "SDXL");
  assert.deepEqual(shown(prepare), [...always.slice(0, 3), "lora_name", "lora_strength", "prompt_pos", "steps", "cfg",
    "sampling_entry", "sampler", "scheduler"]);
  assert.deepEqual(shown(process), ["mask_expand", "mask_feather"]);
  select(prepare, "Z-Image Turbo");
  assert.deepEqual(shown(prepare), [...always.slice(0, 3), "zit_lora_name", "zit_lora_strength", "prompt_pos", "zit_steps",
    "zit_cfg", "zit_sampling_entry", "zit_sampler", "zit_scheduler", "zit_model_shift"]);
  assert.equal(byName(prepare, "zit_lora_name").label, "LORA");
  assert.equal(byName(prepare, "prompt_pos").label, "PROMPT ADDITION");
  assert.equal(byName(prepare, "zit_sampling_entry").label, "SAMPLING ENTRY AT %");
  assert.equal(byName(process, "mask_feather").label, "MASK FEATHER");
  assert.equal(byName(prepare, "steps").value, 30, "values survive family switches");
  // CMK Detailer Family Enable drives the Mask Detailer Prepare as the Detailer Prepare.
  const control = create(ControlNode);
  control.graph = { links: { 21: { target_id: 11, target_slot: 2 } }, getNodeById: (id) => nodes.get(id) };
  control.outputs[1].links = [21];
  press(control, "SDXL");
  assert.equal(byName(prepare, "model_family").value, "SDXL");
  assert.ok(shown(prepare).includes("lora_name"));

  // CMK Flow · MaskDetailer: MODEL FAMILY of the inner Prepare is linked to a
  // subgraph input; the inner view follows the outer MODE.
  const inner = create(MaskPrepare);
  inner.inputs = [{ name: "model_family", link: 40 }];
  select(inner, "None");
  const host = { subgraph: { nodes: [inner], links: new Map([[40, { origin_id: -10, origin_slot: 1 }]]),
    inputs: [{ name: "MODEL ZIT" }, { name: "model_family" }] }, widgets: [{ name: "model_family", value: "Z-Image Turbo" }] };
  extension.loadedGraphNode(host);
  await new Promise((resolve) => setTimeout(resolve, 250));
  assert.equal(byName(inner, "model_family").value, "Z-Image Turbo", "inner Prepare mirrors the outer MODE");
  assert.ok(shown(inner).includes("zit_steps"));
  host.widgets[0].value = "SDXL";
  host.widgets[0].callback();
  assert.ok(shown(inner).includes("steps") && !shown(inner).includes("zit_steps"));
}

console.log("detailer multi-model UI contract: OK");
