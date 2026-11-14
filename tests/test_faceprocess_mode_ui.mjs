import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

// FaceProcess 2.8 F1 (web/js/cmk_faceprocess_mode_ui.js): the Mode Enable
// choice fields and the outer MODE labels. Presentation only; COMBO values
// stay None / Restore / SDXL / Z-Image Turbo. Detailer controls are untouched.
const source = readFileSync(new URL("../web/js/cmk_faceprocess_mode_ui.js", import.meta.url), "utf8")
  .replace('import { app } from "../../../scripts/app.js";', "")
  .replaceAll("export const ", "const ")
  .replaceAll("export function ", "function ");

let extension = null;
const canvasListeners = {};
const app = { graph: { setDirtyCanvas() {} }, registerExtension(value) { extension = value; },
  canvas: { canvas: { addEventListener(name, callback) { canvasListeners[name] = callback; } } } };
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
const styles = [];
const document = {
  createElement: (tag) => new FakeElement(tag),
  getElementById: () => null,
  head: { appendChild(node) { styles.push(node.textContent); } },
  documentElement: {},
  querySelectorAll: () => [],
};
class MutationObserver { observe() {} }
vm.runInNewContext(source, { app, document, MutationObserver, setTimeout(callback) { callback(); } });
assert.ok(extension, "FaceProcess mode UI extension was not registered");
extension.setup();
assert.match(styles.join(""), /data-cmk-faceprocess-mode-select\] \[data-testid="widget-select-default-trigger"\] > span \{ text-align: right; \}/);

const MODES = ["None", "Restore", "SDXL", "Z-Image Turbo"];
class ControlNode {
  constructor() {
    this.comfyClass = "CMKFaceProcessModeEnable";
    this.size = [450, 130];
    this.widgets = [{ name: "process_mode", value: "None", options: { values: MODES }, computeSize: () => [200, 24], draw() {}, callback() {} }];
    this.inputs = [{ name: "process_mode", widget: { name: "process_mode" }, link: null }];
    this.outputs = [{ name: "ENABLE", links: [] }, { name: "PROCESS MODE", links: [] }];
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
extension.beforeRegisterNodeDef(ControlNode, { name: "CMKFaceProcessModeEnable" });
const byName = (node, name) => node.widgets.find((w) => w.name === name);
const create = () => { const node = new ControlNode(); node.onNodeCreated?.(); return node; };
const buttons = (node) => byName(node, "cmk_faceprocess_mode_choice").element.querySelectorAll("button");
const pressed = (node) => buttons(node).filter((b) => b.getAttribute("aria-pressed") === "true").map((b) => b.textContent);
const press = (node, label) => buttons(node).find((b) => b.textContent === label).click();

// Mode Enable: NONE | RESTORE | SDXL | ZIT replace the dropdown.
{
  const node = create();
  assert.deepEqual(buttons(node).map((b) => [b.textContent, b.dataset.value]),
    [["NONE", "None"], ["RESTORE", "Restore"], ["SDXL", "SDXL"], ["ZIT", "Z-Image Turbo"]]);
  assert.ok(byName(node, "process_mode").hidden, "the dropdown is replaced, not duplicated");
  assert.equal(byName(node, "cmk_faceprocess_mode_choice").options.serialize, false, "choice fields are no prompt input");
  assert.deepEqual(pressed(node), ["NONE"]);
  for (const [label, value] of [["RESTORE", "Restore"], ["ZIT", "Z-Image Turbo"], ["SDXL", "SDXL"], ["NONE", "None"]]) {
    press(node, label);
    assert.equal(byName(node, "process_mode").value, value, `${label} stores ${value}`);
    assert.deepEqual(pressed(node), [label]);
  }
  press(node, "RESTORE");
  const active = buttons(node).find((b) => b.textContent === "RESTORE");
  assert.equal(active.style.color, "rgb(88, 190, 230)", "active text in the CMK accent");
  // Save and reload: one positional value, the selection survives.
  const saved = node.serialize().widgets_values;
  assert.deepEqual(saved, ["Restore"]);
  const reloaded = create();
  reloaded.configure({ widgets_values: saved });
  assert.deepEqual(pressed(reloaded), ["RESTORE"]);
}

// A linked selection (decided by the outer MODE) hides the choice fields.
{
  const node = new ControlNode();
  node.inputs[0].link = 7;
  node.onNodeCreated?.();
  assert.ok(byName(node, "cmk_faceprocess_mode_choice").hidden);
  assert.ok(byName(node, "process_mode").hidden);
}

// Outer subgraphs: MODE with short labels, values unchanged.
function subgraphNode(name, widgetName, value, values) {
  return { isSubgraphNode: () => true, subgraph: { name }, widgets: [{ name: widgetName, label: "OLD", value, options: { values } }] };
}
{
  const node = subgraphNode("CMK Flow · FaceProcess", "process_mode", "Restore", MODES);
  extension.loadedGraphNode(node);
  const widget = node.widgets[0];
  assert.equal(widget.label, "MODE");
  assert.deepEqual(widget.options.values.map(widget.options.getOptionLabel), ["DISABLE", "RESTORE", "SDXL", "ZIT"]);
  assert.equal(widget.value, "Restore", "the COMBO value is not translated");
  assert.deepEqual(widget.options.values, MODES);
}
{
  const node = subgraphNode("CMK Flow · FaceProcess · Advanced", "faceprocess_global_enable", "ENABLE", ["ENABLE", "DISABLE"]);
  extension.nodeCreated(node);
  const widget = node.widgets[0];
  assert.equal(widget.label, "MODE");
  assert.deepEqual(widget.options.values.map(widget.options.getOptionLabel), ["ENABLE", "DISABLE"]);
}

// Untouched: legacy FaceProcess and Detailer subgraphs, and the Detailer control.
{
  const others = [
    subgraphNode("CMK Flow · FaceProcess SDXL", "face_global_enable", true, [true, false]),
    subgraphNode("CMK Flow · Detailer", "model_family", "None", ["None", "SDXL", "Z-Image Turbo"]),
    subgraphNode("CMK Flow · Detailer · Advanced", "detailer_global_enable", "ENABLE", ["ENABLE", "DISABLE"]),
  ];
  for (const node of others) {
    extension.loadedGraphNode(node);
    assert.equal(node.widgets[0].label, "OLD");
    assert.equal(node.widgets[0].options.getOptionLabel, undefined);
  }
  const detailerControl = { comfyClass: "CMKDetailerFamilyEnable", widgets: [{ name: "model_family", label: "OLD", value: "None", options: {} }],
    addDOMWidget() { throw new Error("must not decorate the Detailer control"); } };
  extension.nodeCreated(detailerControl);
  assert.equal(detailerControl.widgets.length, 1);
}

// Prepare / execution: only the widgets the operating mode uses, Detailer captions.
const PREPARE_WIDGETS = ["process_mode", "refine_mode", "restore_model", "restore_visibility", "sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler",
  "use_lora_from_1st_pass", "use_1st_pass_sampling", "lora_name", "strength_model", "strength_clip", "prompt_pos",
  "prompt_neg", "steps", "cfg", "sampling_entry", "zit_steps", "zit_cfg", "zit_sampling_entry", "sampler", "scheduler",
  "zit_sampler", "zit_scheduler", "stop_at_clip_layer", "pag_scale", "sampling", "zsnr", "zit_model_shift",
  "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2"];
const EXECUTE_WIDGETS = ["output_image_proceed", "select_face", "detect_model", "detect_bbox_threshold", "detect_bbox_dilation",
  "detect_crop_factor", "detect_drop_size", "detail_guide_size", "detail_guide_size_for", "detail_max_size", "detail_noise_mask", "detail_force_inpaint", "detail_paste_feather"];
const widget = (name, value = 0) => ({ name, value, options: {}, computeSize: () => [200, 24], draw() {} });
const links = new Map();
const nodes = new Map();
const graph = { links, getNodeById: (id) => nodes.get(id) };
function graphNode(id, comfyClass, widgetNames, inputs = [], outputs = []) {
  // Geometry stays the user's: mode switches never resize a node.
  const node = { id, comfyClass, graph, widgets: widgetNames.map((name) => widget(name)), inputs, outputs, size: [450, 900],
    setSize() { throw new Error(`node ${id} must not be resized`); } };
  nodes.set(id, node);
  return node;
}
function connect(id, origin, originSlot, target, targetSlot) {
  links.set(id, { id, origin_id: origin.id, origin_slot: originSlot, target_id: target.id, target_slot: targetSlot });
  (origin.outputs[originSlot].links ??= []).push(id);
  target.inputs[targetSlot].link = id;
}
const control = new ControlNode();
control.id = 1; control.graph = graph; control.outputs = [{ name: "ENABLE", links: [] }, { name: "PROCESS MODE", links: [] }];
nodes.set(1, control);
const prepare = graphNode(2, "CMKFaceProcessPrepareMultiModelPipe", PREPARE_WIDGETS,
  [{ name: "process_mode", widget: { name: "process_mode" }, link: null }],
  [{ name: "FACE", links: [] }, { name: "LOG", links: [] }, { name: "diagnostic", links: [] }]);
const execute = graphNode(3, "CMKFaceProcessMultiModelPipe", EXECUTE_WIDGETS, [{ name: "FACE", link: null }], []);
connect(10, control, 1, prepare, 0);
connect(11, prepare, 0, execute, 0);
for (const node of [prepare, execute]) extension.nodeCreated(node);
control.onNodeCreated?.();
const shown = (node) => node.widgets.filter((w) => !w.hidden).map((w) => w.name);
const SHARED = ["refine_mode", "sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler", "use_lora_from_1st_pass", "prompt_pos"];
const expected = {
  NONE: { prepare: ["process_mode"], execute: [] },
  RESTORE: { prepare: ["process_mode", "refine_mode", "restore_model", "restore_visibility"], execute: ["output_image_proceed",
    "select_face", "detect_model", "detect_bbox_threshold", "detect_bbox_dilation", "detect_crop_factor", "detect_drop_size"] },
  SDXL: { prepare: ["process_mode", ...PREPARE_WIDGETS.filter((n) => !n.startsWith("zit_") && !n.startsWith("restore_") && n !== "process_mode")] },
  ZIT: { prepare: ["process_mode", ...SHARED, "zit_steps", "zit_cfg", "zit_sampling_entry",
    "zit_sampler", "zit_scheduler", "zit_model_shift"] },
};
const DETAILER_EXECUTE = ["output_image_proceed", "select_face", "detect_model", "detect_bbox_threshold", "detect_bbox_dilation",
  "detect_crop_factor", "detect_drop_size", "detail_guide_size", "detail_guide_size_for", "detail_max_size",
  "detail_noise_mask", "detail_force_inpaint", "detail_paste_feather"];
for (const [label, value] of [["SDXL", "SDXL"], ["ZIT", "Z-Image Turbo"], ["RESTORE", "Restore"], ["NONE", "None"]]) {
  press(control, label);
  assert.equal(byName(prepare, "process_mode").value, value, `${label}: the selection is mirrored into the linked Prepare`);
  assert.deepEqual(shown(prepare), expected[label].prepare, `${label}: Prepare widgets`);
  assert.deepEqual(shown(execute), expected[label].execute ?? DETAILER_EXECUTE, `${label}: execution widgets`);
}
press(control, "ZIT");
assert.equal(byName(prepare, "zit_sampling_entry").label, "SAMPLING ENTRY AT %");
assert.equal(byName(prepare, "prompt_pos").label, "PROMPT POS");
assert.equal(byName(prepare, "restore_visibility").label, "VISIBILITY");
assert.equal(byName(prepare, "restore_model").label, "RESTORE MODEL");
assert.deepEqual([prepare.size, execute.size], [[450, 900], [450, 900]], "no automatic resize");
assert.equal(byName(prepare, "refine_mode").label, "REFINE MODE");
assert.equal(byName(execute, "output_image_proceed").label, "OUTPUT IMAGE PROCEED");
assert.equal(byName(prepare, "steps").value, 0, "switching hides widgets but keeps their values");

// CMK Flow · FaceProcess: MODE is a subgraph input linked to the inner
// Prepare's PROCESS MODE (as MODEL FAMILY in the Multi-Model Detailer). The
// inner widget would stay stale, so the outer choice is mirrored into the
// Prepare on change and when the subgraph is opened; the Execute follows.
{
  const innerLinks = new Map();
  const innerNodes = new Map();
  const inner = { name: "CMK Flow · FaceProcess", links: innerLinks, getNodeById: (id) => innerNodes.get(id),
    inputs: [{ name: "MODEL ZIT" }, { name: "MODEL_1" }, { name: "PROCESS_1" }, { name: "IMAGE_1" }, { name: "LOG BYPASS" },
      { name: "VISUAL IN" }, { name: "process_mode" }] };
  const make = (id, comfyClass, names, inputs, outputs) => {
    const node = { id, comfyClass, graph: inner, widgets: names.map((name) => widget(name)), inputs, outputs, size: [450, 900],
      setSize() { throw new Error(`node ${id} must not be resized`); } };
    innerNodes.set(id, node);
    inner.nodes = [...innerNodes.values()];
    return node;
  };
  const innerPrepare = make(8601, "CMKFaceProcessPrepareMultiModelPipe", PREPARE_WIDGETS,
    [{ name: "process_mode", widget: { name: "process_mode" }, link: 23006 }], [{ name: "FACE", links: [23011] }]);
  const innerExecute = make(8602, "CMKFaceProcessMultiModelPipe", EXECUTE_WIDGETS, [{ name: "FACE", link: 23011 }], []);
  innerLinks.set(23006, { id: 23006, origin_id: -10, origin_slot: 6, target_id: 8601, target_slot: 0 });
  innerLinks.set(23011, { id: 23011, origin_id: 8601, origin_slot: 0, target_id: 8602, target_slot: 0 });
  byName(innerPrepare, "process_mode").value = "Restore";
  for (const node of [innerPrepare, innerExecute]) extension.nodeCreated(node);
  const hostMode = { ...widget("process_mode", "SDXL"), options: { values: MODES } };
  const host = { id: 8555, subgraph: inner, isSubgraphNode: () => true, widgets: [hostMode] };
  extension.nodeCreated(host);
  assert.equal(byName(innerPrepare, "process_mode").value, "SDXL", "the outer MODE reaches the inner Prepare");
  assert.deepEqual(shown(innerPrepare), expected.SDXL.prepare);
  assert.deepEqual(shown(innerExecute), DETAILER_EXECUTE);
  assert.equal(hostMode.label, "MODE");
  for (const [label, value] of [["NONE", "None"], ["RESTORE", "Restore"], ["ZIT", "Z-Image Turbo"]]) {
    hostMode.value = value;
    hostMode.callback(value);
    assert.equal(byName(innerPrepare, "process_mode").value, value, `${label}: mirrored on change`);
    assert.deepEqual(shown(innerPrepare), expected[label].prepare, `${label}: inner Prepare widgets`);
    assert.deepEqual(shown(innerExecute), expected[label].execute ?? DETAILER_EXECUTE, `${label}: inner execution widgets`);
  }
  hostMode.value = "Restore";  // changed elsewhere, e.g. another instance of the shared definition
  canvasListeners["subgraph-opened"]({ detail: { fromNode: host } });
  assert.equal(byName(innerPrepare, "process_mode").value, "Restore", "mirrored again when the subgraph opens");
  assert.deepEqual(shown(innerPrepare), expected.RESTORE.prepare);
  assert.equal(byName(innerPrepare, "steps").value, 0, "values stay");
  assert.deepEqual([innerPrepare.size, innerExecute.size], [[450, 900], [450, 900]], "no automatic resize");
  // A subgraph without a linked FaceProcess Prepare is left alone.
  const other = { id: 9, subgraph: { name: "CMK Flow · Detailer", nodes: [], links: new Map(), inputs: [] },
    isSubgraphNode: () => true, widgets: [widget("model_family", "SDXL")] };
  extension.nodeCreated(other);
  assert.equal(other.widgets[0].callback, undefined);
}

console.log("faceprocess mode UI contract: OK");
