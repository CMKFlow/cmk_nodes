import { app } from "../../../scripts/app.js";

const NODE_NAME = "CMKLogConcat";
const MAX_INPUTS = 32;

function isLogBlockInput(input) {
    return input?.name === "log_block" || /^log_block_\d+$/.test(input?.name || "");
}

function inputNumber(input) {
    if (input?.name === "log_block") return 1;
    const match = /^log_block_(\d+)$/.exec(input?.name || "");
    return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
}

function applyLabels(node) {
    for (const input of node?.inputs || []) {
        if (input.name === "LOG") input.label = "LOG";
        else if (input.name === "log_block") input.label = "LOG BLOCK 1";
        else {
            const match = /^log_block_(\d+)$/.exec(input.name || "");
            if (match) input.label = `LOG BLOCK ${match[1]}`;
        }
    }
}

function normalize(node) {
    if (!node?.inputs) return;
    const family = node.inputs.filter(isLogBlockInput);
    if (!family.length) return;

    const connected = family.filter((input) => input.link != null);
    const highestConnected = connected.length
        ? Math.max(...connected.map(inputNumber))
        : 1;

    for (let index = node.inputs.length - 1; index >= 0; index--) {
        const input = node.inputs[index];
        if (!isLogBlockInput(input) || input.name === "log_block") continue;
        const number = inputNumber(input);
        if (number > highestConnected + 1 && input.link == null) node.removeInput(index);
    }

    const refreshed = node.inputs.filter(isLogBlockInput);
    const hasTrailing = refreshed.some(
        (input) => inputNumber(input) === highestConnected + 1 && input.link == null,
    );

    if (!hasTrailing && highestConnected < MAX_INPUTS) {
        node.addInput(`log_block_${highestConnected + 1}`, "CMK_LOG_BLOCK");
    }

    applyLabels(node);
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "cmk.log_concat.dynamic_inputs",
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
