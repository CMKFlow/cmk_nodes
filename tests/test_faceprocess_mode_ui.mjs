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
const app = { graph: { setDirtyCanvas() {} }, registerExtension(value) { extension = value; } };
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

console.log("faceprocess mode UI contract: OK");
