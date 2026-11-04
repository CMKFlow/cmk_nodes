import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
    ENGINE_SDXL,
    ENGINE_ZIT,
    FINISH_NONE,
    FINISH_REFINER,
    FINISH_ZIT,
    STRATEGY_IDENTITY,
    STRATEGY_STANDARD,
    allowedFinishes,
    effectiveStop,
    modelFamilyFor,
    normalizeSelection,
    stopSupported,
} from "../web/js/cmk_flow_start_v2_core.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (relative) => fs.readFileSync(path.join(root, relative), "utf8");

assert.deepEqual(
    allowedFinishes(ENGINE_SDXL, STRATEGY_STANDARD),
    [FINISH_REFINER, FINISH_ZIT],
);
assert.deepEqual(
    allowedFinishes(ENGINE_SDXL, STRATEGY_IDENTITY),
    [FINISH_REFINER],
);
assert.deepEqual(
    allowedFinishes(ENGINE_ZIT, STRATEGY_STANDARD),
    [FINISH_NONE],
);
assert.deepEqual(
    normalizeSelection(ENGINE_ZIT, STRATEGY_IDENTITY, FINISH_ZIT),
    { engine: ENGINE_ZIT, strategy: STRATEGY_STANDARD, finish: FINISH_NONE },
);

assert.equal(modelFamilyFor(ENGINE_SDXL, STRATEGY_STANDARD, FINISH_REFINER), "SDXL");
assert.equal(modelFamilyFor(ENGINE_SDXL, STRATEGY_STANDARD, FINISH_ZIT), "Hybrid");
assert.equal(modelFamilyFor(ENGINE_ZIT, STRATEGY_STANDARD, FINISH_NONE), "Z-Image Turbo");

const stopCases = [
    [{ engine: ENGINE_SDXL, strategy: STRATEGY_STANDARD, finish: FINISH_REFINER, inpaint: false, instantidMode: "HYBRID" }, false],
    [{ engine: ENGINE_SDXL, strategy: STRATEGY_STANDARD, finish: FINISH_ZIT, inpaint: false, instantidMode: "HYBRID" }, true],
    [{ engine: ENGINE_SDXL, strategy: STRATEGY_STANDARD, finish: FINISH_ZIT, inpaint: true, instantidMode: "HYBRID" }, true],
    [{ engine: ENGINE_SDXL, strategy: STRATEGY_IDENTITY, finish: FINISH_REFINER, inpaint: false, instantidMode: "HYBRID" }, true],
    [{ engine: ENGINE_SDXL, strategy: STRATEGY_IDENTITY, finish: FINISH_REFINER, inpaint: false, instantidMode: "ZERO-PASS" }, false],
    [{ engine: ENGINE_SDXL, strategy: STRATEGY_IDENTITY, finish: FINISH_REFINER, inpaint: true, instantidMode: "HYBRID" }, false],
    [{ engine: ENGINE_ZIT, strategy: STRATEGY_STANDARD, finish: FINISH_NONE, inpaint: false, instantidMode: "HYBRID" }, false],
];
for (const [configuration, expected] of stopCases) {
    assert.equal(stopSupported(configuration), expected);
    assert.equal(effectiveStop(true, configuration), expected);
    assert.equal(effectiveStop(false, configuration), false);
}

const frontend = read("web/js/cmk_flow_start_guidance_v2.js");
const legacy = read("web/js/cmk_flow_start_guidance_v51.js");
const browser = read("web/js/cmk_flow_browser.js");
const dimensions = read("web/js/cmk_node_dimensions.js");
const metadata = JSON.parse(read("web/flow_node_metadata.json"));
const backend = read("pipe/cmk_pipe_image.py");

assert.match(frontend, /cmkStartUiVersion/);
// v2 reuses the shared start-node setup, but every layout request of the
// shared controls is answered by the v2 layout; v2 never runs the v1 rebuild.
assert.match(frontend, /node\._cmkStartLayout = \(\) => refresh\(node\);\s+configureStartBase\(node, true\)/);
assert.doesNotMatch(frontend, /rebuildModeWidgets/);
assert.match(legacy, /function requestStartLayout\(node\)/);
assert.doesNotMatch(
    legacy.slice(legacy.indexOf("export function configureStartBase")),
    /rebuildModeWidgets\(node, true\)/,
);
assert.match(frontend, /widget\.serialize = false/);
assert.match(frontend, /TARGET_SIZE = \[600, 1225\]/);
assert.match(frontend, /MINIMUM_SIZE = \[450, 700\]/);
assert.match(frontend, /_cmkStartPromptMinimumScale = 0\.5/);
assert.match(frontend, /overflowY = "auto"/);
assert.doesNotMatch(frontend, /MOSS/i);
assert.match(legacy, /Number\(node\.properties\?\.cmkStartUiVersion\) === 2/);
assert.match(legacy, /widgets_values = widgets/);

const propertyIndex = browser.indexOf("Object.assign(node.properties");
const graphAddIndex = browser.indexOf("graph.add(node)", propertyIndex);
assert.ok(propertyIndex >= 0 && graphAddIndex > propertyIndex);
assert.deepEqual(
    metadata.nodes.CMKPipeCreateImage.nodeProperties,
    { cmkStartUiVersion: 2 },
);
assert.equal(metadata.nodes.CMKPipeCreateImage.displayName, "01 START HERE v2 · Create Image");
assert.match(dimensions, /"CMKPipeCreateImage:v2": \[600, 1225\]/);
assert.match(dimensions, /"CMKPipeCreateImage:v2": \[450, 700\]/);

const returnTypes = backend.match(/RETURN_TYPES = \(([^]*?)\n    \)/)?.[1];
assert.ok(returnTypes);
assert.equal((returnTypes.match(/"CMK_/g) || []).length, 5);
assert.match(returnTypes, /"IMAGE"/);

console.log("01 START HERE v2 frontend/browser/persistence contracts: OK");
