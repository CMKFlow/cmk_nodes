import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKPipeCreateImage";
const GUIDE_NAME = "NEXT STEP";
const GUIDE_TEXT = "05 ControlNet (optional)  or  10 KSampler 1st Pass";

function isTarget(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS ||
        node.type === NODE_CLASS ||
        node.constructor?.comfyClass === NODE_CLASS ||
        node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function configure(node) {
    if (!isTarget(node) || typeof node.addWidget !== "function") return;

    let guide = node.widgets?.find((widget) => widget?.name === GUIDE_NAME);
    if (!guide) {
        guide = node.addWidget("text", GUIDE_NAME, GUIDE_TEXT, () => {}, {
            serialize: false,
        });
    }

    guide.value = GUIDE_TEXT;
    guide.label = "NEXT STEP →";
    guide.disabled = true;
    guide.serialize = false;

    if (Array.isArray(node.size)) {
        node.size[0] = Math.max(Number(node.size[0]) || 0, 430);
    }

    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function schedule(node) {
    for (const delay of [0, 50, 200]) setTimeout(() => configure(node), delay);
}

app.registerExtension({
    name: "cmk.flow.start.guidance.v1",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        for (const hook of ["onNodeCreated", "onConfigure", "onAdded"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                schedule(this);
                return result;
            };
        }
    },

    nodeCreated(node) {
        if (isTarget(node)) schedule(node);
    },

    loadedGraphNode(node) {
        if (isTarget(node)) schedule(node);
    },
});
