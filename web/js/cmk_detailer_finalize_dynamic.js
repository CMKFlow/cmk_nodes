import { app } from "../../../scripts/app.js";

const NODE_NAME = "CMKDetailerFinalizePipe";
const MAX_INPUTS = 32;

function isNumbered(input, base) {
    return input?.name === base || new RegExp(`^${base}_(\\d+)$`).test(input?.name || "");
}

function inputNumber(input, base) {
    if (input?.name === base) return 1;
    const match = new RegExp(`^${base}_(\\d+)$`).exec(input?.name || "");
    return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
}

function normalizeFamily(node, base, type) {
    if (!node?.inputs) return;
    const family = node.inputs.filter((input) => isNumbered(input, base));
    if (!family.length) return;

    const connected = family.filter((input) => input.link != null);
    const highestConnected = connected.length
        ? Math.max(...connected.map((input) => inputNumber(input, base)))
        : 1;

    for (let index = node.inputs.length - 1; index >= 0; index--) {
        const input = node.inputs[index];
        if (!isNumbered(input, base) || input.name === base) continue;
        const number = inputNumber(input, base);
        if (number > highestConnected + 1 && input.link == null) node.removeInput(index);
    }

    const refreshed = node.inputs.filter((input) => isNumbered(input, base));
    const hasTrailing = refreshed.some(
        (input) => inputNumber(input, base) === highestConnected + 1 && input.link == null,
    );
    if (!hasTrailing && highestConnected < MAX_INPUTS) {
        node.addInput(`${base}_${highestConnected + 1}`, type);
    }
}

function normalize(node) {
    normalizeFamily(node, "segs", "SEGS");
    normalizeFamily(node, "log_pipe", "STRING");
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "cmk.detailer.finalize.dynamic_inputs",
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
