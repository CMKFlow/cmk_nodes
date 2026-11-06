// The source/target slider style must never leak to other sliders: an
// unscoped "div.grid:has(...)" rule matched the page-level comfyui-body grid,
// so one visible HYBRID BALANCE slider (v1 START HERE) labelled the v2
// HYBRID HANDOFF slider of another node "ZIT".
import assert from "node:assert/strict";

let styleText = "";
globalThis.document = {
    getElementById: () => null,
    createElement: () => ({ set textContent(value) { styleText = value; } }),
    head: { appendChild() {} },
};

const { installSourceTargetSlider } = await import("../web/js/cmk_source_target_slider.js");
installSourceTargetSlider({}, { value: 0, options: {} }, "CMK HYBRID BALANCE", "SDXL", "ZIT", 0, 4, 2);
assert.ok(styleText.includes('content: "ZIT"'), "HYBRID BALANCE target label is installed");
assert.ok(styleText.includes('content: "TARGET"'), "SAMPLING START target label is installed");
assert.ok(styleText.includes('content: "IDENTITY " attr(data-cmk-identity-share)'), "IDENTITY label is installed");

const selectors = styleText
    .split("}")
    .map((rule) => rule.split("{")[0].replace(/\s+/g, " ").trim())
    .filter(Boolean);
assert.ok(selectors.length >= 14, selectors.length);
for (const selector of selectors) {
    assert.ok(
        selector.startsWith('[data-testid="node-widget"]'),
        `rule not scoped to one widget row: ${selector}`,
    );
}
// The target labels address only the flex row whose slider is the owning one.
const labelRules = selectors.filter((selector) => selector.endsWith("::after"));
assert.equal(labelRules.length, 3);
for (const selector of labelRules) {
    assert.match(selector, /^\[data-testid="node-widget"\] div\.grid:has\(\[data-slot="slider"\]\[aria-label\^="(HYBRID BALANCE|SAMPLING START|IDENTITY SHARE)"\]\)/);
}
console.log("source/target slider scope: ok");
