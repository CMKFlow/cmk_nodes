import { app } from "../../../scripts/app.js";

const NODE_NAMES = new Set(["CMKPreviewBoard", "CMKDiagnosticConcat"]);
const CONCAT_NODE = "CMKDiagnosticConcat";
const PREVIEW_NODE = "CMKPreviewBoard";
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

function isConcat(node) {
    return (
        node?.comfyClass === CONCAT_NODE ||
        node?.type === CONCAT_NODE ||
        node?.constructor?.comfyClass === CONCAT_NODE ||
        node?.constructor?.nodeData?.name === CONCAT_NODE
    );
}

function isPreviewBoard(node) {
    return (
        node?.comfyClass === PREVIEW_NODE ||
        node?.type === PREVIEW_NODE ||
        node?.constructor?.comfyClass === PREVIEW_NODE ||
        node?.constructor?.nodeData?.name === PREVIEW_NODE
    );
}

function removeLegacyTitle(node) {
    if (!isConcat(node)) return;

    for (let index = (node.inputs?.length || 0) - 1; index >= 0; index--) {
        if (node.inputs[index]?.name === "title") node.removeInput(index);
    }
    for (let index = (node.widgets?.length || 0) - 1; index >= 0; index--) {
        if (node.widgets[index]?.name === "title") node.widgets.splice(index, 1);
    }
}

function normalize(node, configuredSize = null) {
    if (!node?.inputs) return;
    // Preserve the workflow/manual dimensions before changing the dynamic
    // sockets. computeSize() still sees the complete 32-input node definition
    // in some frontend paths and would otherwise restore the bogus ~680 px
    // minimum height.
    const preservedSize = Array.isArray(configuredSize) && configuredSize.length === 2
        ? [Number(configuredSize[0]), Number(configuredSize[1])]
        : Array.isArray(node.size)
        ? [Number(node.size[0]), Number(node.size[1])]
        : null;
    removeLegacyTitle(node);
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
    const fixedSize = node.properties?.cmkFixedSize;
    if (Array.isArray(fixedSize) && fixedSize.length === 2) {
        node.setSize([Number(fixedSize[0]), Number(fixedSize[1])]);
        node.setDirtyCanvas?.(true, true);
        return;
    }
    if (isPreviewBoard(node) && preservedSize?.every(Number.isFinite)) {
        node.setSize(preservedSize);
        node.setDirtyCanvas?.(true, true);
        return;
    }
    const size = node.computeSize();
    // onConfigure receives the workflow's persisted node.size. Dynamic input
    // normalization may increase the minimum dimensions, but must never reset
    // a height the user resized manually. Width already followed this rule.
    node.setSize([
        Math.max(node.size[0], size[0]),
        Math.max(node.size[1], size[1]),
    ]);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "cmk.preview_board.dynamic_inputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) return;
        for (const hook of ["onNodeCreated", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                queueMicrotask(() => normalize(this));
                return result;
            };
        }

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (configuration) {
            // Tab changes reconstruct the graph. The frontend can expand the
            // node while onConfigure runs, so retain the serialized size from
            // the workflow instead of reading this.size afterwards.
            const configuredSize = Array.isArray(configuration?.size)
                ? [...configuration.size]
                : null;
            const result = originalOnConfigure?.apply(this, arguments);
            queueMicrotask(() => normalize(this, configuredSize));
            return result;
        };
    },
});
