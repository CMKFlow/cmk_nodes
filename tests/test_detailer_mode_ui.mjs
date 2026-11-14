import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

// MODE dropdown of the outer Detailer subgraphs (web/js/cmk_detailer_mode_ui.js):
// visible label and option labels only; COMBO values and other nodes untouched.
const source = readFileSync(new URL("../web/js/cmk_detailer_mode_ui.js", import.meta.url), "utf8")
  .replace('import { app } from "../../../scripts/app.js";', "")
  .replace("export function modeSpec", "function modeSpec")
  .replace("export function applyMode", "function applyMode");

let extension = null;
const app = { registerExtension(value) { extension = value; } };
const styles = [];
const document = {
  getElementById: () => null,
  createElement: () => ({ style: {}, textContent: "" }),
  head: { appendChild(node) { styles.push(node.textContent); } },
  documentElement: {},
  querySelectorAll: () => [],
};
class MutationObserver { observe() {} }
vm.runInNewContext(source, { app, document, MutationObserver, setTimeout(callback) { callback(); } });
assert.ok(extension, "MODE UI extension was not registered");
extension.setup();
assert.match(styles.join(""), /data-cmk-mode-select\] \[data-testid="widget-select-default-trigger"\] > span \{ text-align: right; \}/);

function subgraphNode(name, widgetName, value, values) {
  return {
    isSubgraphNode: () => true,
    subgraph: { name },
    widgets: [{ name: widgetName, label: "OLD", value, options: { values } }],
  };
}

// Standard: MODE, DISABLE / SDXL / ZIT, value stays the model family.
{
  const node = subgraphNode("CMK Flow · Detailer", "model_family", "None", ["None", "SDXL", "Z-Image Turbo"]);
  extension.loadedGraphNode(node);
  const widget = node.widgets[0];
  assert.equal(widget.label, "MODE");
  assert.deepEqual(widget.options.values.map(widget.options.getOptionLabel), ["DISABLE", "SDXL", "ZIT"]);
  assert.equal(widget.value, "None", "the COMBO value is not translated");
  assert.deepEqual(widget.options.values, ["None", "SDXL", "Z-Image Turbo"]);
}

// Mask Detailer (2.8): the same MODE as the Standard Detailer.
{
  const node = subgraphNode("CMK Flow · MaskDetailer", "model_family", "SDXL", ["None", "SDXL", "Z-Image Turbo"]);
  extension.loadedGraphNode(node);
  const widget = node.widgets[0];
  assert.equal(widget.label, "MODE");
  assert.deepEqual(widget.options.values.map(widget.options.getOptionLabel), ["DISABLE", "SDXL", "ZIT"]);
  assert.equal(widget.value, "SDXL");
}

// FaceRebuild v2 (same title as the SDXL subgraphs): MODE ENABLE / DISABLE on
// facerebuild_mode; the legacy subgraph's ENABLE toggle keeps its label.
for (const name of ["CMK Flow · FaceRebuild SDXL", "CMK Flow · FaceRebuild SDXL · Advanced",
  // inserted from the blueprint library: named after the file
  "CMK Flow · FaceRebuild SDXL v2", "CMK Flow · FaceRebuild SDXL · Advanced v2"]) {
  const v2 = subgraphNode(name, "facerebuild_mode", "ENABLE", ["ENABLE", "DISABLE"]);
  extension.loadedGraphNode(v2);
  assert.equal(v2.widgets[0].label, "MODE");
  assert.deepEqual(v2.widgets[0].options.values.map(v2.widgets[0].options.getOptionLabel), ["ENABLE", "DISABLE"]);
  const legacy = subgraphNode(name, "FACEREBUILD ENABLE", true, undefined);
  extension.loadedGraphNode(legacy);
  assert.equal(legacy.widgets[0].label, "OLD", "the legacy toggle is untouched");
  assert.equal(legacy.widgets[0].options.getOptionLabel, undefined);
}

// Advanced: MODE, ENABLE / DISABLE unchanged.
{
  const node = subgraphNode("CMK Flow · Detailer · Advanced", "detailer_global_enable", "DISABLE", ["ENABLE", "DISABLE"]);
  extension.nodeCreated(node);
  const widget = node.widgets[0];
  assert.equal(widget.label, "MODE");
  assert.deepEqual(widget.options.values.map(widget.options.getOptionLabel), ["ENABLE", "DISABLE"]);
  assert.equal(widget.value, "DISABLE");
}

// Other dropdowns are untouched: another subgraph, and interior nodes with
// the same widget names (Family Enable, Prepare).
{
  const other = subgraphNode("CMK Flow · Detailer SDXL", "model_family", "SDXL", ["None", "SDXL"]);
  const interior = { comfyClass: "CMKDetailerFamilyEnable", widgets: [{ name: "model_family", label: "OLD", value: "None", options: { values: ["None"] } }] };
  for (const node of [other, interior]) {
    extension.loadedGraphNode(node);
    assert.equal(node.widgets[0].label, "OLD");
    assert.equal(node.widgets[0].options.getOptionLabel, undefined);
  }
}

console.log("detailer mode UI contract: OK");
