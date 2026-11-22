// CMK Flow Browser · References tab: search-only entries (sidecar
// "searchOnly": true) are absent from the category view and found by search;
// the published sidecars carry the flag for exactly the four 2.8 Advanced
// module workflows.
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { register } from "node:module";

register("./helpers/cmk_frontend_test_loader.mjs", import.meta.url);

globalThis.localStorage = { getItem: () => null, setItem: () => {} };
Object.defineProperty(globalThis, "navigator", { value: { language: "de-DE" }, configurable: true });
globalThis.__cmkTestApp = { registerExtension: () => {} };
globalThis.__cmkTestApi = {};

const { filterReferences } = await import("../web/js/cmk_flow_browser.js");

const reference = (name, extra = {}) => ({
    name,
    categoryLabel: "MODULE WORKFLOWS",
    descriptionLabel: "",
    highlightLabel: "",
    infoLabel: "",
    ...extra,
});
const references = [
    reference("CMK 2.8 · Detailer · Modul"),
    reference("CMK 2.8 · Detailer Advanced · Modul", { searchOnly: true }),
    reference("CMK 2.8 · FaceSwap · Modul"),
];
const names = (list) => list.map((item) => item.name);

// Category view (no search term): search-only entries are not listed.
assert.deepEqual(names(filterReferences(references, "")), [
    "CMK 2.8 · Detailer · Modul",
    "CMK 2.8 · FaceSwap · Modul",
]);

// Search: search-only entries are found like every other entry.
assert.deepEqual(names(filterReferences(references, "advanced")), ["CMK 2.8 · Detailer Advanced · Modul"]);
assert.deepEqual(names(filterReferences(references, "detailer")), [
    "CMK 2.8 · Detailer · Modul",
    "CMK 2.8 · Detailer Advanced · Modul",
]);
// The text of the category also matches, so a category search lists them too.
assert.equal(filterReferences(references, "module workflows").length, 3);
// searchOnly must be literally true.
assert.equal(filterReferences([reference("X", { searchOnly: "yes" })], "").length, 1);

// Package sidecars: exactly the four Advanced module workflows, published.
const metadataDir = new URL("../workflows/showcase/metadata/", import.meta.url);
const searchOnly = readdirSync(metadataDir)
    .filter((file) => file.endsWith(".json"))
    .map((file) => [file, JSON.parse(readFileSync(new URL(file, metadataDir), "utf-8"))])
    .filter(([, data]) => data.searchOnly === true);
assert.deepEqual(searchOnly.map(([file]) => file).sort(), [
    "CMK 2.8 · Detailer Advanced · Modul.json",
    "CMK 2.8 · FaceProcess Advanced · Modul.json",
    "CMK 2.8 · FaceRebuild Advanced SDXL · Modul.json",
    "CMK 2.8 · FaceSwap Advanced · Modul.json",
]);
for (const [file, data] of searchOnly) {
    assert.equal(data.published, true, file);
    // Their own names are found by the browser search.
    const entry = reference(data.displayName, { searchOnly: true });
    assert.equal(filterReferences([entry], "advanced").length, 1, file);
    assert.equal(filterReferences([entry], "").length, 0, file);
}

console.log("flow browser search-only: OK");
