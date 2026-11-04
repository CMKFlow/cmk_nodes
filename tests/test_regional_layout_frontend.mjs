import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  automaticPreviewSignature, commitHistory, createHistory, createLatestRequestGuard,
  firstImageReference, hitRegionIndices, moveBox, normalizeBackgroundSource,
  normalizeLayout, packagedReferenceAssetPath, prepareImageGeometry,
  processDimensionsFromWidgets, redoHistory, regionalCanvasGeometry,
  regionalLayoutAvailable, resolveAutomaticPreviewSource, resizeBox, serializeLayout,
  undoHistory,
} from "../web/js/cmk_regional_layout_core.js";

function widget(name, value) { return { name, value }; }

function graph(nodes, links) {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const value = { id: "graph", links: new Map(links.map((link) => [link.id, link])), getNodeById: (id) => byId.get(id) };
  for (const node of nodes) node.graph = value;
  return value;
}

function directFlow({ enabled = false, source = "Reference Image", external = true, reroutes = false } = {}) {
  const loader = { id: 1, type: "LoadImage", widgets: [widget("image", "base.png [input]")] };
  const referenceLoader = { id: 2, type: "LoadImage", widgets: [widget("image", "external.png [input]")] };
  const imageReroute = { id: 3, type: "Reroute", inputs: [{ name: "", link: 101 }] };
  const start = { id: 4, type: "CMKPipeCreateImage", inputs: [{ name: "IMAGE", link: reroutes ? 102 : 101 }], widgets: [
    widget("model_family", "SDXL"), widget("controlnet_enabled", enabled), widget("resolution", "1152x832"),
  ] };
  const control = { id: 5, type: "CMKControlNetPreparePipe", inputs: [
    { name: "PROCESS", type: "CMK_PROCESS_SDXL", link: 103 },
    { name: "REFERENCE IMAGE INPUT", link: external ? 104 : null },
  ], widgets: [widget("ENABLE", enabled), widget("IMAGE SOURCE", source), widget("REFERENCE IMAGE", "internal.png [input]")] };
  const processReroute = { id: 6, type: "Reroute", inputs: [{ name: "", link: 105 }] };
  const layoutNode = { id: 7, type: "CMKRegionalLayoutSDXL", inputs: [
    { name: "PROCESS", type: "CMK_PROCESS_SDXL", link: reroutes ? 106 : 105 },
  ] };
  graph([loader, referenceLoader, imageReroute, start, control, processReroute, layoutNode], [
    { id: 101, origin_id: loader.id },
    { id: 102, origin_id: imageReroute.id },
    { id: 103, origin_id: start.id },
    { id: 104, origin_id: referenceLoader.id },
    { id: 105, origin_id: control.id },
    { id: 106, origin_id: processReroute.id },
  ]);
  return { loader, referenceLoader, start, control, layoutNode };
}

const base = normalizeLayout({ width: 1280, height: 720, regions: [
  { id: "back", name: "Back", prompt: "sky", box: { x: 0.1, y: 0.1, width: 0.6, height: 0.6 } },
  { id: "front", name: "Front", prompt: "person", strength: 1.4, box: { x: 0.3, y: 0.3, width: 0.3, height: 0.3 } },
] });

test("frontend registration uses 07 name and enforces the pure-ZIT edit lock", () => {
  const source = readFileSync(new URL("../web/js/cmk_regional_layout_ui.js", import.meta.url), "utf8");
  const loader = readFileSync(new URL("../web/js/cmk_regional_layout_loader_v1.js", import.meta.url), "utf8");
  assert.match(source, /CMK Flow · 07 Regional Layout/);
  assert.match(source, /setBackgroundHint\("not available"\)/);
  assert.match(source, /if \(editingLocked\) return;/);
  assert.match(source, /new ResizeObserver\(resizeCanvas\)\.observe\(root\)/);
  assert.match(source, /_cmkRegionalLayoutSyncCanvas\?\.\(\)/);
  assert.match(loader, /cmk-regional-layout-v5/);
});

test("canvas lifecycle ignores hidden graph width and restores without an editor action", () => {
  const initial = regionalCanvasGeometry(600, 1152, 832);
  assert.ok(initial);
  let applied = initial;
  const hidden = regionalCanvasGeometry(0, 1152, 832);
  if (hidden) applied = hidden;
  assert.deepEqual(applied, initial);
  const restored = regionalCanvasGeometry(600, 1152, 832);
  if (restored) applied = restored;
  assert.deepEqual(applied, initial);
  assert.ok(applied.width > 1);
  assert.ok(applied.height > 1);
});

test("canvas geometry is available with zero regions and preserves normalized data", () => {
  const empty = normalizeLayout({ width: 1024, height: 1024, regions: [] });
  assert.equal(empty.regions.length, 0);
  assert.deepEqual(regionalCanvasGeometry(600, empty.width, empty.height), { width: 390, height: 390 });
});

test("packaged references use the existing CMK asset endpoint", () => {
  assert.equal(
    packagedReferenceAssetPath("CMK Package · controlnet_reference.png"),
    "/cmk/reference-assets/controlnet_reference.png",
  );
  assert.equal(packagedReferenceAssetPath("controlnet_reference.png"), null);
  assert.equal(packagedReferenceAssetPath("CMK Package · ../controlnet_reference.png"), null);
  const asset = readFileSync(new URL("../assets/references/controlnet_reference.png", import.meta.url));
  assert.ok(asset.length > 0);
});

test("overlap hit testing exposes every region from front to back", () => {
  assert.deepEqual(hitRegionIndices(base.regions, 0.4, 0.4), [1, 0]);
});

test("moving and resizing remain inside normalized bounds", () => {
  assert.deepEqual(moveBox(base.regions[1].box, 0.8, 0.8), { x: 0.7, y: 0.7, width: 0.3, height: 0.3 });
  assert.deepEqual(resizeBox(base.regions[1].box, "nw", -0.5, -0.5), { x: 0, y: 0, width: 0.6, height: 0.6 });
});

test("undo and redo restore prompt, strength, order and geometry", () => {
  let history = createHistory(base);
  const changed = normalizeLayout(JSON.parse(serializeLayout(base)));
  changed.regions.reverse();
  changed.regions[0].prompt = "changed";
  changed.regions[0].strength = 0.5;
  changed.regions[0].box.x = 0.55;
  history = commitHistory(history, changed);
  history = undoHistory(history);
  assert.equal(history.present.regions[0].id, "back");
  assert.equal(history.present.regions[1].prompt, "person");
  history = redoHistory(history);
  assert.equal(history.present.regions[0].id, "front");
  assert.equal(history.present.regions[0].prompt, "changed");
  assert.equal(history.present.regions[0].strength, 0.5);
  assert.equal(history.present.regions[0].box.x, 0.55);
});

test("workflow serialization round-trips more than three regions", () => {
  const layout = normalizeLayout({ width: 1536, height: 640, regions: Array.from({ length: 5 }, (_, index) => ({
    id: `r${index}`, name: `Area ${index}`, prompt: `Prompt ${index}`, strength: 0.5 + index / 10,
    box: { x: index / 10, y: 0.1, width: 0.2, height: 0.4 },
  })) });
  const restored = normalizeLayout(serializeLayout(layout));
  assert.equal(restored.width, 1536);
  assert.equal(restored.height, 640);
  assert.equal(restored.regions.length, 5);
  assert.deepEqual(restored.regions.map((region) => [region.id, region.prompt, region.strength]), layout.regions.map((region) => [region.id, region.prompt, region.strength]));
});

test("strength offset defaults to zero and survives workflow serialization", () => {
  assert.equal(normalizeLayout({ regions: [] }).strength_offset, 0);
  const restored = normalizeLayout(serializeLayout({ strength_offset: 1.75, regions: [] }));
  assert.equal(restored.strength_offset, 1.75);
  assert.equal(normalizeLayout({ strength_offset: 9, regions: [] }).strength_offset, 2);
});

test("START HERE resolution and swap determine editor dimensions", () => {
  assert.deepEqual(processDimensionsFromWidgets("SDXL 1152x832", false), { width: 1152, height: 832 });
  assert.deepEqual(processDimensionsFromWidgets("1024 × 1536", true), { width: 1536, height: 1024 });
});

test("dimension changes preserve normalized region geometry", () => {
  const resized = normalizeLayout({ ...base, width: 832, height: 1216 });
  assert.deepEqual(resized.regions.map((region) => region.box), base.regions.map((region) => region.box));
});

test("background source defaults to Manual and accepts Automatic", () => {
  assert.equal(normalizeBackgroundSource(), "manual");
  assert.equal(normalizeBackgroundSource("Manual"), "manual");
  assert.equal(normalizeBackgroundSource("Automatic"), "automatic");
  assert.equal(normalizeBackgroundSource("unsupported"), "manual");
});

test("automatic image batches use only their first resolvable reference", () => {
  assert.equal(firstImageReference(["first.png [input]", "second.png [input]"]), "first.png [input]");
  assert.deepEqual(firstImageReference([{ filename: "mask.png", subfolder: "clipspace", type: "input" }]), {
    filename: "mask.png", subfolder: "clipspace", type: "input",
  });
  assert.equal(firstImageReference([]), null);
  assert.equal(firstImageReference({ invalid: true }), null);
});

test("01 → 07 uses the START HERE image and follows reroutes", () => {
  const flow = directFlow({ enabled: false, reroutes: true });
  const source = resolveAutomaticPreviewSource(flow.layoutNode);
  assert.equal(source.kind, "start");
  assert.equal(source.reference, "base.png [input]");
  assert.equal(source.start, flow.start);
});

test("active SDXL ControlNet prioritizes external then internal references", () => {
  const externalFlow = directFlow({ enabled: true, external: true });
  const external = resolveAutomaticPreviewSource(externalFlow.layoutNode);
  assert.equal(external.kind, "controlnet-reference");
  assert.equal(external.reference, "external.png [input]");

  const internalFlow = directFlow({ enabled: true, external: false });
  const internal = resolveAutomaticPreviewSource(internalFlow.layoutNode);
  assert.equal(internal.kind, "controlnet-reference");
  assert.equal(internal.reference, "internal.png [input]");
});

test("inactive ControlNet falls back to 01 and Base Image explicitly uses 01", () => {
  const inactive = directFlow({ enabled: false });
  assert.equal(resolveAutomaticPreviewSource(inactive.layoutNode).kind, "start");
  const base = directFlow({ enabled: true, source: "Base Image" });
  const source = resolveAutomaticPreviewSource(base.layoutNode);
  assert.equal(source.kind, "controlnet-base");
  assert.equal(source.reference, "base.png [input]");
});

test("05 Combined resolves its effective SDXL source inside the subgraph", () => {
  const loader = { id: 20, type: "LoadImage", widgets: [widget("image", "base.png [input]")] };
  const start = { id: 21, type: "CMKPipeCreateImage", inputs: [{ name: "IMAGE", link: 201 }], widgets: [
    widget("model_family", "Hybrid"), widget("controlnet_enabled", true),
  ] };
  const innerReference = { id: 30, type: "CMKLoadImage", widgets: [widget("image", "combined.png [input]")] };
  const flag = { id: 31, type: "CMKProcessEnableFlag", widgets: [widget("flag", "controlnet_enabled")] };
  const innerControl = { id: 32, type: "CMKCombinedControlNetPreparePipe", inputs: [
    { name: "ENABLE", link: 301 }, { name: "REFERENCE IMAGE INPUT", link: 302 },
  ], widgets: [widget("IMAGE SOURCE", "Reference Image"), widget("REFERENCE IMAGE", "internal.png [input]")] };
  graph([innerReference, flag, innerControl], [
    { id: 301, origin_id: flag.id }, { id: 302, origin_id: innerReference.id },
  ]);
  const combined = { id: 22, type: "combined-subgraph-uuid", inputs: [
    { name: "PROCESS SDXL", type: "CMK_PROCESS_SDXL", link: 202 },
  ], subgraph: { _nodes: [innerReference, flag, innerControl] } };
  const layoutNode = { id: 23, type: "CMKRegionalLayoutSDXL", inputs: [
    { name: "PROCESS", type: "CMK_PROCESS_SDXL", link: 203 },
  ] };
  graph([loader, start, combined, layoutNode], [
    { id: 201, origin_id: loader.id }, { id: 202, origin_id: start.id }, { id: 203, origin_id: combined.id },
  ]);
  const source = resolveAutomaticPreviewSource(layoutNode);
  assert.equal(source.kind, "controlnet-reference");
  assert.equal(source.reference, "combined.png [input]");
  assert.equal(source.family, "hybrid");
});

test("pure ZIT is unavailable while SDXL and Hybrid reactivate layout", () => {
  const flow = directFlow({ enabled: false });
  const family = flow.start.widgets.find((item) => item.name === "model_family");
  family.value = "Z-Image Turbo";
  assert.deepEqual(
    { unavailable: resolveAutomaticPreviewSource(flow.layoutNode).unavailable, message: resolveAutomaticPreviewSource(flow.layoutNode).message },
    { unavailable: true, message: "not available" },
  );
  assert.equal(regionalLayoutAvailable(family.value), false);
  family.value = "SDXL";
  assert.equal(resolveAutomaticPreviewSource(flow.layoutNode).unavailable, undefined);
  assert.equal(regionalLayoutAvailable(family.value), true);
  family.value = "Hybrid";
  assert.equal(resolveAutomaticPreviewSource(flow.layoutNode).family, "hybrid");
  assert.equal(regionalLayoutAvailable(family.value), true);
});

test("Fit, Crop and Stretch reproduce target geometry and crop position", () => {
  assert.deepEqual(prepareImageGeometry(1000, 500, 1000, 1000, "Fit", "Center"), {
    mode: "fit", x: 0, y: 250, width: 1000, height: 500, targetWidth: 1000, targetHeight: 1000,
  });
  assert.deepEqual(prepareImageGeometry(1000, 500, 1000, 1000, "Fit", "Bottom"), {
    mode: "fit", x: 0, y: 500, width: 1000, height: 500, targetWidth: 1000, targetHeight: 1000,
  });
  assert.deepEqual(prepareImageGeometry(1000, 500, 1000, 1000, "Crop", "Right"), {
    mode: "crop", x: -1000, y: 0, width: 2000, height: 1000, targetWidth: 1000, targetHeight: 1000,
  });
  assert.deepEqual(prepareImageGeometry(1000, 500, 832, 1216, "Stretch", "Center"), {
    mode: "stretch", x: 0, y: 0, width: 832, height: 1216, targetWidth: 832, targetHeight: 1216,
  });
});

test("ControlNet preview uses ComfyUI's centered target crop without changing target size", () => {
  assert.deepEqual(prepareImageGeometry(1000, 500, 1152, 832, "Crop", "Center", "Custom"), {
    mode: "crop", x: -256, y: 0, width: 1664, height: 832, targetWidth: 1152, targetHeight: 832,
  });
});

test("Expand forces Fit and geometry uses Python half-to-even rounding", () => {
  assert.equal(prepareImageGeometry(1000, 500, 1000, 1000, "Crop", "Center", "Expand").mode, "fit");
  assert.equal(prepareImageGeometry(1000, 500, 1000, 1000, "Stretch", "Center", "Extend Image").mode, "fit");
  assert.equal(prepareImageGeometry(5, 2, 4, 1, "Fit").width, 2);
});

test("automatic preview signature reacts to image, mask snapshot and geometry changes", () => {
  const initial = automaticPreviewSignature({ sourceId: 4, imageReference: "source.png", width: 1024, height: 1024, resizeMode: "Fit" });
  assert.notEqual(initial, automaticPreviewSignature({ sourceId: 4, imageReference: "other.png", width: 1024, height: 1024, resizeMode: "Fit" }));
  assert.notEqual(initial, automaticPreviewSignature({ sourceId: 4, imageReference: "clipspace-painted-masked-2.png", width: 1024, height: 1024, resizeMode: "Fit" }));
  assert.notEqual(initial, automaticPreviewSignature({ sourceId: 4, imageReference: "source.png", width: 1216, height: 832, resizeMode: "Fit" }));
  assert.notEqual(initial, automaticPreviewSignature({ sourceId: 4, imageReference: "source.png", width: 1024, height: 1024, resizeMode: "Crop" }));
  assert.notEqual(initial, automaticPreviewSignature({ sourceId: 4, imageReference: "source.png", width: 1024, height: 1024, resizeMode: "Fit", imageRevision: 1 }));
  assert.notEqual(initial, automaticPreviewSignature({ sourceId: 4, imageReference: "source.png", width: 1024, height: 1024, resizeMode: "Fit", sourceKind: "controlnet-reference" }));
});

test("latest request guard rejects stale asynchronous image loads", () => {
  const guard = createLatestRequestGuard();
  const first = guard.begin();
  const second = guard.begin();
  assert.equal(guard.isCurrent(first), false);
  assert.equal(guard.isCurrent(second), true);
  guard.invalidate();
  assert.equal(guard.isCurrent(second), false);
});

test("background calculations never mutate region data", () => {
  const before = serializeLayout(base);
  prepareImageGeometry(1920, 1080, base.width, base.height, "Crop", "Left", "Custom");
  automaticPreviewSignature({ imageReference: "source.png", width: base.width, height: base.height });
  assert.equal(serializeLayout(base), before);
});
