import { app } from "../../../scripts/app.js";

const NODE_NAME = "CMKPreviewBoard";
const MAX_INPUTS = 32;

function isDiagnosticInput(input) {
    return input?.type === "CMK_DIAGNOSTIC" && (input.name === "diagnostic_1" || /^diagnostic_\d+$/.test(input.name));
}

function inputNumber(input) {
    if (input.name === "diagnostic_1") return 1;
    const match = /^diagnostic_(\d+)$/.exec(input.name);
    return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
}

function applyLabels(node) {
    for (const input of node?.inputs || []) {
        const match = /^diagnostic_(\d+)$/.exec(input.name || "");
        if (match) input.label = `DIAGNOSTIC ${match[1]}`;
    }
}

function normalize(node) {
    if (!node?.inputs) return;
    const family = node.inputs.filter(isDiagnosticInput);
    if (!family.length) return;

    const connected = family.filter((input) => input.link != null);
    const highestConnected = connected.length
        ? Math.max(...connected.map(inputNumber))
        : 1;

    for (let index = node.inputs.length - 1; index >= 0; index--) {
        const input = node.inputs[index];
        if (!isDiagnosticInput(input) || input.name === "diagnostic_1") continue;
        const number = inputNumber(input);
        if (number > highestConnected + 1 && input.link == null) node.removeInput(index);
    }

    const refreshed = node.inputs.filter(isDiagnosticInput);
    const hasTrailing = refreshed.some(
        (input) => inputNumber(input) === highestConnected + 1 && input.link == null,
    );

    if (!hasTrailing && highestConnected < MAX_INPUTS) {
        node.addInput(`diagnostic_${highestConnected + 1}`, "CMK_DIAGNOSTIC");
    }

    applyLabels(node);
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "cmk.preview_board.dynamic_inputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                queueMicrotask(() => normalize(this));
                return result;
            };
        }
    },
});
