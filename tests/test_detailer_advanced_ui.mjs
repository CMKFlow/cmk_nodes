import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

// Frontend contract of CMK Detailer Prepare Advanced Multi-Model -Pipe-
// (web/js/cmk_detailer_advanced_ui.js): family switch, dynamic visibility of
// the settings, three independent Detailers, save and load.
const sourceUrl = new URL("../web/js/cmk_detailer_advanced_ui.js", import.meta.url);
const source = readFileSync(sourceUrl, "utf8")
  .replace('import { app } from "../../../scripts/app.js";', "")
  .replace("export function refreshAdvanced", "function refreshAdvanced");

let extension = null;
const app = {
  graph: { setDirtyCanvas() {} },
  registerExtension(value) { extension = value; },
};
const document = {
  getElementById: () => null,
  createElement: () => ({ style: {}, textContent: "" }),
  head: { appendChild() {} },
  documentElement: {},
  querySelectorAll: () => [],
};
class MutationObserver { observe() {} }
vm.runInNewContext(source, {
  app, document, MutationObserver, queueMicrotask: () => {}, setTimeout(callback) { callback(); },
});
assert.ok(extension, "Advanced Prepare UI extension was not registered");
extension.setup();

const SHARED = [["sam_model_name", "sam_vit_b_01ec64.pth"], ["sam_device_mode", "CPU"], ["use_prompt_lora_from_sampler", false],
  ["stop_at_clip_layer", -2], ["pag_scale", 2.5], ["sampling", "v_prediction"], ["zsnr", true], ["freeu_enabled", true],
  ["freeu_b1", 1.3], ["freeu_b2", 1.4], ["freeu_s1", 0.9], ["freeu_s2", 0.2]];
const SDXL_SLOT = ["steps", "cfg", "sampling_entry", "sampler", "scheduler"];
const ZIT_SLOT = ["zit_steps", "zit_cfg", "zit_sampling_entry", "zit_sampler", "zit_scheduler", "zit_model_shift"];
const SDXL_SHARED = ["stop_at_clip_layer", "pag_scale", "sampling", "zsnr", "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2"];
// Backend widget order: shared, then per Detailer its family and settings.
const WIDGETS = [...SHARED];
for (const slot of [1, 2, 3]) {
  WIDGETS.push([`model_family_${slot}`, "None"]);
  for (const name of SDXL_SLOT) WIDGETS.push([`${name}_${slot}`, name === "steps" ? 30 : 1]);
  for (const name of ZIT_SLOT) WIDGETS.push([`${name}_${slot}`, name === "zit_steps" ? 8 : 1]);
}
WIDGETS.push(["detailer_global_enable", "ENABLE"]);

class AdvancedNode {
  constructor() {
    this.comfyClass = "CMKDetailerPrepareAdvancedMultiModelPipe";
    this.size = [450, 760];
    this.properties = {};
    this.widgets = WIDGETS.map(([name, value]) => ({ name, value, options: {}, computeSize: () => [200, 24], draw() {} }));
    this.inputs = [1, 2, 3].map((slot) => ({ name: `model_family_${slot}`, widget: { name: `model_family_${slot}` }, link: null }));
  }
  setSize(size) { this.size = [...size]; }
  configure(data) {
    data.widgets_values.forEach((value, index) => { this.widgets[index].value = value; });
    this.onConfigure?.(data);
  }
  serialize() {
    return JSON.parse(JSON.stringify({ widgets_values: this.widgets.map((widget) => widget.value) }));
  }
}
extension.beforeRegisterNodeDef(AdvancedNode, { name: "CMKDetailerPrepareAdvancedMultiModelPipe" });

function create() {
  const node = new AdvancedNode();
  node.onNodeCreated?.();
  extension.nodeCreated(node);
  return node;
}
const byName = (node, name) => node.widgets.find((widget) => widget.name === name);
const visible = (node, name) => !byName(node, name).hidden;
function select(node, slot, family) {
  const widget = byName(node, `model_family_${slot}`);
  widget.value = family;
  widget.callback(family);
}
function assertSlot(node, slot, family) {
  for (const name of SDXL_SLOT) assert.equal(visible(node, `${name}_${slot}`), family === "SDXL", `D${slot} ${family}: ${name}`);
  for (const name of ZIT_SLOT) assert.equal(visible(node, `${name}_${slot}`), family === "Z-Image Turbo", `D${slot} ${family}: ${name}`);
}

// New node: all Detailers None, only the shared basics visible, CMK labels.
{
  const node = create();
  for (const slot of [1, 2, 3]) assertSlot(node, slot, "None");
  for (const [name] of SHARED.slice(0, 3)) assert.ok(visible(node, name), `${name} shared`);
  for (const name of SDXL_SHARED) assert.ok(!visible(node, name), `${name} without an SDXL Detailer`);
  assert.equal(byName(node, "model_family_2").label, "MODEL FAMILY");
  assert.equal(byName(node, "zit_sampling_entry_3").label, "SAMPLING ENTRY AT %");
  assert.equal(byName(node, "pag_scale").label, "PAG SCALE (SDXL)");
  assert.equal(byName(node, "detailer_global_enable").label, "ENABLE");
  assert.ok(visible(node, "detailer_global_enable"), "global switch is never hidden by a family");
  assert.equal(node.widgets.length, WIDGETS.length, "hidden widgets stay part of the prompt");
}

// Three independent Detailers; switching keeps every value.
{
  const node = create();
  select(node, 1, "Z-Image Turbo");
  select(node, 2, "SDXL");
  assertSlot(node, 1, "Z-Image Turbo");
  assertSlot(node, 2, "SDXL");
  assertSlot(node, 3, "None");
  for (const name of SDXL_SHARED) assert.ok(visible(node, name), `${name} with an SDXL Detailer`);
  byName(node, "zit_steps_1").value = 6;
  byName(node, "steps_2").value = 42;
  select(node, 3, "Z-Image Turbo");
  byName(node, "zit_steps_3").value = 11;
  byName(node, "detailer_global_enable").value = "DISABLE";
  assertSlot(node, 3, "Z-Image Turbo");
  assert.equal(byName(node, "zit_steps_1").value, 6, "D1 unaffected by D3");
  select(node, 2, "None");
  assertSlot(node, 2, "None");
  for (const name of SDXL_SHARED) assert.ok(!visible(node, name), `${name} hidden again`);
  select(node, 2, "SDXL");
  assert.equal(byName(node, "steps_2").value, 42, "value kept across a family switch");

  // Save and load: positional values, identical state.
  const saved = node.serialize().widgets_values;
  const reloaded = new AdvancedNode();
  reloaded.onNodeCreated?.();
  extension.nodeCreated(reloaded);
  reloaded.configure({ widgets_values: [...saved] });
  extension.loadedGraphNode(reloaded);
  assertSlot(reloaded, 1, "Z-Image Turbo");
  assertSlot(reloaded, 2, "SDXL");
  assertSlot(reloaded, 3, "Z-Image Turbo");
  assert.deepEqual(reloaded.serialize().widgets_values, saved);
  assert.equal(byName(reloaded, "zit_steps_3").value, 11);
  assert.equal(byName(reloaded, "detailer_global_enable").value, "DISABLE", "global DISABLE is saved and restored");
  assertSlot(reloaded, 2, "SDXL");
}

// A family linked to a CMK Detailer Family Enable hides its dropdown only.
{
  const node = new AdvancedNode();
  node.inputs[0].link = 9;
  node.onNodeCreated?.();
  extension.nodeCreated(node);
  assert.ok(!visible(node, "model_family_1"));
  assert.ok(visible(node, "model_family_2"));
}

// The node returns to its remembered size after the family sets are hidden.
{
  const node = new AdvancedNode();
  node.size = [450, 1286];
  node.properties.cmkManualSize = [450, 790];
  node.onNodeCreated?.();
  extension.nodeCreated(node);
  assert.deepEqual(node.size, [450, 760]);
}

console.log("detailer advanced UI contract: OK");
