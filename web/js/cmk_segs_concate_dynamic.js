import { app } from "../../../scripts/app.js";

const NODE_NAME = "CMK_SEGSConcate";
const MAX_SEGS_INPUTS = 32;

function isSegsInput(input) {
    return input?.type === "SEGS" && (input.name === "segs" || /^segs_\d+$/.test(input.name));
}

function segsInputNumber(input) {
    if (input.name === "segs") return 1;
    const match = /^segs_(\d+)$/.exec(input.name);
    return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
}

function nextSegsName(node) {
    const numbers = node.inputs
        .filter(isSegsInput)
        .map(segsInputNumber)
        .filter(Number.isFinite);
    const next = Math.max(1, ...numbers) + 1;
    return next <= MAX_SEGS_INPUTS ? `segs_${next}` : null;
}

function applyLabels(node) {
    for (const input of node?.inputs || []) {
        if (input.name === "image") input.label = "IMAGE";
        else if (input.name === "segs") input.label = "SEGS 1";
        else {
            const match = /^segs_(\d+)$/.exec(input.name || "");
            if (match) input.label = `SEGS ${match[1]}`;
            else if (input.name === "feather") input.label = "FEATHER";
            else if (input.name === "alpha") input.label = "ALPHA";
        }
    }
}

function normalizeSegsInputs(node) {
    if (!node?.inputs) return;

    const segsInputs = node.inputs.filter(isSegsInput);
    if (!segsInputs.length) return;

    const connected = segsInputs.filter((input) => input.link != null);
    const highestConnected = connected.length
        ? Math.max(...connected.map(segsInputNumber))
        : 1;

    // Keep every slot up to the highest connected one and exactly one empty
    // trailing slot. This avoids losing valid links while keeping the node tidy.
    for (let index = node.inputs.length - 1; index >= 0; index--) {
        const input = node.inputs[index];
        if (!isSegsInput(input) || input.name === "segs") continue;

        const number = segsInputNumber(input);
        if (number > highestConnected + 1 && input.link == null) {
            node.removeInput(index);
        }
    }

    const currentSegsInputs = node.inputs.filter(isSegsInput);
    const hasTrailingEmpty = currentSegsInputs.some(
        (input) => segsInputNumber(input) === highestConnected + 1 && input.link == null,
    );

    if (!hasTrailingEmpty && highestConnected < MAX_SEGS_INPUTS) {
        const name = nextSegsName(node);
        if (name) node.addInput(name, "SEGS");
    }

    applyLabels(node);
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "cmk.segs_concate.dynamic_inputs",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            queueMicrotask(() => normalizeSegsInputs(this));
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            queueMicrotask(() => normalizeSegsInputs(this));
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            queueMicrotask(() => normalizeSegsInputs(this));
            return result;
        };
    },
});
