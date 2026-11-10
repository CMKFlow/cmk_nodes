// START HERE: saved values and the ContentGuard slot, BASELINE and RELEASE.
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { alignGuardValues } from "../web/js/cmk_start_guard_values.js";

const ORDER = [
  ["GLOBAL PROMPT POS", "string"], ["PROMPT NEG", "string"], ["INPAINT_MODE", "string"], ["resolution", "string"],
  ["swap_dimensions", "boolean"], ["upscale_method", "string"], ["device", "string"], ["outpaint_on", "boolean"],
  ["mask_fill_holes", "boolean"], ["fill_masked_area", "string"], ["process_mode", "string"], ["resize_mode", "string"],
  ["crop_position", "string"], ["outpaint_overlap", "number"], ["model_family", "string"], ["instantid_enabled", "boolean"],
  ["instantid_mode", "string"], ["inpaint_mask_expand", "number"], ["hybrid_sdxl_handoff", "string"],
  ["hybrid_zit_denoise", "string"], ["unload_models_after_use", "boolean"], ["HYBRID BALANCE", "number"],
  ["live_preview_zit", "boolean"], ["content_guard_enabled", "boolean"], ["controlnet_enabled", "boolean"],
  ["instantid_handoff_percent", "number"], ["stop_after_handoff", "boolean"],
].map(([name, kind]) => ({ name, kind }));
const GUARD = ORDER.findIndex((widget) => widget.name === "content_guard_enabled");
const BASELINE = ORDER;
const RELEASE = ORDER.filter((_, index) => index !== GUARD);
const STOP_R = RELEASE.findIndex((widget) => widget.name === "stop_after_handoff");
const HANDOFF_R = RELEASE.findIndex((widget) => widget.name === "instantid_handoff_percent");
const CN_R = RELEASE.findIndex((widget) => widget.name === "controlnet_enabled");

const saved27 = ["pos", "", "Inpaint", "1344x768", false, "lanczos", "cpu", true, true, "neutral", "Custom", "Fit", "Center", 32,
  "Hybrid", false, "HYBRID", 16, "90%", "0.20", true, 2, true, false, true, 30, true];
const saved26 = saved27.filter((_, index) => index !== GUARD);

assert.deepEqual(alignGuardValues(BASELINE, saved27), saved27, "baseline keeps its own layout");
assert.deepEqual(alignGuardValues(RELEASE, saved26), saved26, "release keeps its own layout");
assert.deepEqual(alignGuardValues(RELEASE, saved27), saved26, "release drops exactly the guard slot");
const inserted = alignGuardValues(BASELINE, saved26);
assert.equal(inserted[GUARD], true, "a release save loads with the guard's safe default");
assert.deepEqual(inserted.filter((_, index) => index !== GUARD), saved26);

const aligned = alignGuardValues(RELEASE, saved27);
assert.equal(aligned[CN_R], true);
assert.equal(aligned[HANDOFF_R], 30);
assert.equal(aligned[STOP_R], true);
assert.equal(typeof aligned[STOP_R], "boolean");

// Older and foreign layouts stay exactly as saved.
const legacy25 = [...saved27.slice(0, 22), "AUTO", false, "AUTO"];
assert.deepEqual(alignGuardValues(RELEASE, legacy25), legacy25);
assert.deepEqual(alignGuardValues(BASELINE, legacy25), legacy25);
const beforeStop = saved27.slice(0, 26);
assert.deepEqual(alignGuardValues(BASELINE, beforeStop), beforeStop, "baseline save from before STOP");
assert.deepEqual(alignGuardValues(RELEASE, beforeStop), saved26.slice(0, 25), "the same save loaded in release");
const wrongHead = [...saved27]; wrongHead[13] = "not a number";
assert.deepEqual(alignGuardValues(RELEASE, wrongHead), wrongHead, "a head that does not fit is not touched");
const ambiguous = saved27.slice(0, GUARD + 1);
assert.deepEqual(alignGuardValues(RELEASE, ambiguous), ambiguous, "one value after the slot is ambiguous");
assert.equal(alignGuardValues(RELEASE, undefined), undefined);

// Every shipped CMK 2.7 START HERE node.
const folder = new URL("../workflows/showcase/", import.meta.url);
let checked = 0;
for (const name of readdirSync(folder)) {
  if (!name.endsWith(".json") || name.startsWith("CMK 2.6") || name === "CMK FaceSwap Video.json") continue;
  const workflow = JSON.parse(readFileSync(new URL(encodeURIComponent(name), folder), "utf8"));
  for (const node of workflow.nodes.filter((item) => item.type === "CMKPipeCreateImage")) {
    if (node.widgets_values.length !== 27) continue;
    const values = node.widgets_values;
    assert.deepEqual(alignGuardValues(BASELINE, values), values, name);
    const release = alignGuardValues(RELEASE, values);
    assert.equal(release.length, RELEASE.length, name);
    assert.deepEqual(release, values.filter((_, index) => index !== GUARD), name);
    assert.equal(typeof release[STOP_R], "boolean", name);
    checked += 1;
  }
}
assert.ok(checked >= 18, `checked ${checked}`);
console.log("START HERE guard values: OK");
