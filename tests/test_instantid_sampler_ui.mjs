import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const sourceUrl = new URL("../web/js/cmk_instantid_sampler_ui.js", import.meta.url);
const source = readFileSync(sourceUrl, "utf8").replace(
  'import { app } from "../../../scripts/app.js";',
  "",
);

let extension = null;
const app = {
  graph: { setDirtyCanvas() {}, getNodeById() { return null; } },
  registerExtension(value) {
    extension = value;
  },
};

class Element {
  constructor({ textContent = "", parentElement = null } = {}) {
    this.textContent = textContent;
    this.parentElement = parentElement;
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.attributes = {};
  }
  closest(selector) {
    if (selector === "[data-node-id]") return this.nodeElement ?? null;
    return this.fieldRow ?? this.parentElement;
  }
  querySelectorAll() { return []; }
  matches() { return false; }
  setAttribute(name, value) { this.attributes[name] = value; }
}
const document = {
  nodeElements: [],
  querySelectorAll() { return this.nodeElements; },
};

vm.runInNewContext(source, {
  app,
  document,
  Element,
  window: {},
  MutationObserver: class {},
  setTimeout(callback) { callback(); },
});

assert.ok(extension, "InstantID UI extension was not registered");

for (const comfyClass of [
  "CMKInstantIDSamplerSDXLPipe",
  "CMKInstantIDRegionalAdvancedSDXLPipe",
]) {
  const referenceStart = {
    name: "reference_start_at_step",
    value: 8,
    type: "number",
    options: { advanced: true },
    computeSize: () => [200, 24],
    draw() {},
    hidden: false,
  };
  const identityStrength = {
    name: "identity_strength",
    value: 1.0,
    type: "number",
    options: {},
    computeSize: () => [200, 24],
    draw() {},
    hidden: false,
  };
  const node = {
    comfyClass,
    widgets: [
      identityStrength,
      { name: "conditioning_mode", value: "InstantID reference", options: {} },
      { name: "reference_conditioning_weight", value: 0.75, options: {} },
      referenceStart,
    ],
    inputs: [
      { name: "source_face", type: "IMAGE" },
      {
        name: "reference_start_at_step",
        type: "INT",
        widget: { name: "reference_start_at_step" },
      },
    ],
    size: [450, 800],
    setDirtyCanvas() {},
  };
  const row = new Element();
  const label = new Element({ textContent: "reference_start_at_step", parentElement: row });
  const nodeElement = new Element();
  nodeElement.dataset.nodeId = "1";
  nodeElement.nodeElement = nodeElement;
  nodeElement.matches = (selector) => selector === "[data-node-id]";
  nodeElement.querySelectorAll = (selector) => (
    selector === "*" || selector === '[data-testid="widget-layout-field-label"]'
      ? (selector === "*" ? [row, label] : [label])
      : []
  );
  label.nodeElement = nodeElement;
  label.fieldRow = row;
  row.nodeElement = nodeElement;
  document.nodeElements = [nodeElement];
  app.graph.getNodeById = () => node;

  extension.nodeCreated(node);

  assert.equal(referenceStart.value, 8, `${comfyClass}: stored value changed`);
  assert.equal(referenceStart.type, "hidden", `${comfyClass}: widget remains visible`);
  assert.equal(referenceStart.hidden, true, `${comfyClass}: hidden flag missing`);
  assert.equal(referenceStart.options.hidden, true, `${comfyClass}: DOM hidden flag missing`);
  assert.deepEqual(Array.from(referenceStart.computeSize()), [0, -4], `${comfyClass}: row still occupies space`);
  const legacyInput = node.inputs.find((input) => input.name === "reference_start_at_step");
  assert.equal(legacyInput.hidden, true, `${comfyClass}: input row remains visible`);
  assert.equal(legacyInput.options.hidden, true, `${comfyClass}: input row hidden option missing`);
  assert.equal(legacyInput.widget.hidden, true, `${comfyClass}: mirrored widget row remains visible`);
  assert.equal(
    legacyInput.widget.options.hidden,
    true,
    `${comfyClass}: mirrored widget hidden option missing`,
  );
  assert.equal(node.inputs[0].hidden, undefined, `${comfyClass}: unrelated input row was hidden`);
  assert.equal(row.hidden, true, `${comfyClass}: rendered field row remains visible`);
  assert.equal(row.style.display, "none", `${comfyClass}: rendered field row occupies layout space`);
  assert.equal(row.attributes["aria-hidden"], "true", `${comfyClass}: rendered row is exposed to accessibility`);
  assert.equal(identityStrength.type, "number", `${comfyClass}: unrelated widget type changed`);
  assert.equal(identityStrength.hidden, false, `${comfyClass}: unrelated widget was hidden`);
  assert.equal(
    identityStrength.label,
    comfyClass === "CMKInstantIDSamplerSDXLPipe" ? "IDENTITY STRENGTH" : undefined,
    `${comfyClass}: unrelated label behavior changed`,
  );
  assert.deepEqual(node.size, [450, 800], `${comfyClass}: node geometry changed`);
}

assert.equal(source.includes('reference_start_at_step: "SAMPLING START AT STEP"'), false);
console.log("InstantID sampler UI tests passed");
