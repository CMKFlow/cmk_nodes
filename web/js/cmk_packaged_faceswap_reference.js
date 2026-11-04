import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASSES = new Set(["CMKLoadImage", "CMKImageLoadAndResizePipe"]);
const PACKAGED_REFERENCES = new Map([
    ["CMK Package · face_reference.png", "face_reference.png"],
    ["CMK Package · faceswap_reference.png", "faceswap_reference.png"],
    ["CMK Package · inpaint_reference.png", "inpaint_reference.png"],
    ["CMK Package · remove_reference.png", "remove_reference.png"],
]);

function isTarget(node) {
    return node && (
        NODE_CLASSES.has(node.comfyClass) ||
        NODE_CLASSES.has(node.type) ||
        NODE_CLASSES.has(node.constructor?.comfyClass)
    );
}

function showPackagedReference(node, value) {
    const filename = PACKAGED_REFERENCES.get(String(value || ""));
    if (!isTarget(node) || !filename) return;
    const preview = new Image();
    preview.onload = () => {
        node.imgs = [preview];
        node.imageIndex = 0;
        node.setDirtyCanvas?.(true, true);
    };
    preview.src = typeof api?.apiURL === "function"
        ? api.apiURL(`/cmk/reference-assets/${filename}`)
        : `/cmk/reference-assets/${filename}`;
}

function install(node) {
    if (!isTarget(node) || node._cmkPackagedFaceSwapReference) return;
    const widget = node.widgets?.find((item) => item?.name === "image" || item?.name === "IMAGE");
    if (!widget) return;
    node._cmkPackagedFaceSwapReference = true;
    const original = widget.callback;
    widget.callback = function (value) {
        const result = original?.apply(this, arguments);
        showPackagedReference(node, value);
        return result;
    };
    showPackagedReference(node, widget.value);
}

app.registerExtension({
    name: "cmk.packaged_faceswap_reference",
    nodeCreated(node) {
        install(node);
    },
    loadedGraphNode(node) {
        install(node);
        const widget = node.widgets?.find((item) => item?.name === "image" || item?.name === "IMAGE");
        showPackagedReference(node, widget?.value);
    },
});
