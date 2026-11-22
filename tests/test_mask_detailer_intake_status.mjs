// CMK Mask Detailer · Intake: initial status of the frontend extension.
// A freshly inserted Intake (image "None") shows NO SNAPSHOT, a pending
// incoming candidate LIVE · CAPTURE CURRENT, a locked snapshot LOCKED; the
// IMAGE / LOG / PROCESS sockets are left untouched by the setup.
import assert from "node:assert/strict";
import { register } from "node:module";

register("./helpers/cmk_frontend_test_loader.mjs", import.meta.url);

const extensions = [];
globalThis.__cmkTestApp = { registerExtension: (extension) => extensions.push(extension), graph: { setDirtyCanvas() {} } };
globalThis.__cmkTestApi = { fetchApi: async () => ({ ok: true, status: 200 }) };

await import("../web/js/cmk_mask_detailer_intake.js");
const extension = extensions.find((item) => item.name === "cmk.mask-detailer.terminal-intake.v1");

function intake(imageValue) {
    const inputs = [
        { name: "PROCESS", type: "*" },
        { name: "IMAGE", type: "IMAGE" },
        { name: "LOG", type: "CMK_LOG_PIPE" },
        { name: "capture_current", type: "BOOLEAN", widget: { name: "capture_current" } },
        { name: "image", type: "COMBO", widget: { name: "image" } },
    ];
    return {
        type: "CMKMaskDetailerIntake",
        comfyClass: "CMKMaskDetailerIntake",
        properties: {},
        inputs,
        widgets: [
            { name: "capture_current", type: "toggle", value: false },
            { name: "image", type: "combo", value: imageValue, options: { values: ["None"] } },
        ],
        addWidget(type, name, value) {
            const widget = { type, name, value };
            this.widgets.push(widget);
            return widget;
        },
        setDirtyCanvas() {},
    };
}

async function created(imageValue) {
    const node = intake(imageValue);
    const sockets = JSON.stringify(node.inputs);
    extension.nodeCreated(node);
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(JSON.stringify(node.inputs), sockets, "setup must not change the sockets");
    return node.widgets.find((widget) => widget.name === "status").value;
}

assert.equal(await created("None"), "NO SNAPSHOT");
assert.equal(await created(""), "NO SNAPSHOT");
assert.equal(await created("cmk_mask_detailer/incoming-7783-7572.png"), "LIVE · CAPTURE CURRENT");
assert.equal(await created("clipspace-painted-masked-1794728649506.png [input]"), "LOCKED");

console.log("mask detailer intake status: OK");
