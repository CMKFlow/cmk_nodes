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
  ["enable", true], ["output_image_proceed", true], ["model_name", "bbox/hand_yolov8s.pt"],
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
  for (const name of SHARED) assert.ok(visible(node, name), `${family}: ${name} shared`);
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

console.log("detailer multi-model UI contract: OK");
