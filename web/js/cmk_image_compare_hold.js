import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const CMK_FLOW_TITLE = /^CMK Flow\s*[·-]/i;
const NODE_SELECTOR = "[data-node-id]";
const VIEWPORT_SELECTOR = '[data-testid="image-compare-viewport"]';
const RESETTABLE_PREVIEW_FLOW = /^CMK Flow\s*[·-]\s*(10|15|20|23|25|30|40|42|90)\b/i;
let executionActive = false;

function isCmkFlow(node) {
    return CMK_FLOW_TITLE.test(String(node?.title || node?.type || ""));
}

function isNativeCompare(node) {
    return node?.constructor?.comfyClass === "ImageCompare" || node?.type === "ImageCompare";
}

function ownsCompareProxy(node) {
    return Boolean(node?.properties?.proxyWidgets?.some?.(
        (entry) => Array.isArray(entry) && entry[1] === "compare_view",
    ));
}

function isCompareOwner(node) {
    return isCmkFlow(node) || isNativeCompare(node) || ownsCompareProxy(node);
}

function enforceClickCompare(node) {
    if (!isCompareOwner(node)) return;
    node.properties ||= {};
    node.properties.comparer_mode = "Click";
    node.properties.default_output = "b";
}

function isResettablePreviewFlow(node, nodeElement = null) {
    const identity = [node?.title, node?.type, nodeElement?.textContent]
        .filter(Boolean)
        .join(" ");
    return RESETTABLE_PREVIEW_FLOW.test(identity);
}

function resetViewport(viewport) {
    viewport.classList.add("cmk-preview-reset");
    viewport.classList.remove("cmk-preview-refreshed");
}

function clearModuleComparePreviews() {
    executionActive = true;
    // Proxy widgets of subgraphs are not consistently children of the visible
    // outer node. Reset the actual compare viewport instead of its node wrapper.
    for (const viewport of document.querySelectorAll(VIEWPORT_SELECTOR)) {
        resetViewport(viewport);
    }

    for (const nodeElement of document.querySelectorAll(NODE_SELECTOR)) {
        const node = nodeForElement(nodeElement);
        if (!isResettablePreviewFlow(node, nodeElement)) continue;
        if (node) {
            node.imgs = [];
            node.imageIndex = null;
        }
        node?.setDirtyCanvas?.(true, true);
    }
}

function revealUpdatedPreview(record) {
    const element = record.target instanceof Element ? record.target : record.target?.parentElement;
    const viewport = element?.closest?.(VIEWPORT_SELECTOR);
    if (!viewport?.classList.contains("cmk-preview-reset")) return;
    const addedImage = [...(record.addedNodes || [])].some((node) =>
            node instanceof HTMLImageElement || node?.querySelector?.("img")
        ) || (record.type === "attributes" && record.target instanceof HTMLImageElement);
    if (addedImage) viewport.classList.add("cmk-preview-refreshed");
}

function finishModuleComparePreviews() {
    executionActive = false;
    for (const viewport of document.querySelectorAll(
        `${VIEWPORT_SELECTOR}.cmk-preview-reset.cmk-preview-refreshed`,
    )) {
        viewport.classList.remove("cmk-preview-reset", "cmk-preview-refreshed");
    }
}

function ownsDynamicStartUi(node) {
    return Boolean(
        node?.widgets?.some?.((widget) => widget?.name === "INPAINT_MODE"),
    );
}

function validSize(value) {
    return Array.isArray(value) && Number(value[0]) > 0 && Number(value[1]) > 0;
}

function nodeForElement(element) {
    const nodeElement = element?.closest?.(NODE_SELECTOR);
    const id = nodeElement?.dataset?.nodeId;
    if (id == null) return null;
    const candidates = [id, Number(id)].filter((value) => value !== "" && !Number.isNaN(value));
    for (const graph of [app.canvas?.graph, app.graph, app.rootGraph]) {
        for (const candidate of candidates) {
            const node = graph?.getNodeById?.(candidate);
            if (node) return node;
        }
    }
    return null;
}

function restoreCmkSize(node, nodeElement) {
    if (!isCmkFlow(node)) return;
    nodeElement.classList.add("cmk-flow-node");
    // 01 START HERE deliberately changes its visible widget set between
    // Text2Image and Inpaint. Restoring cmkOuterSize on every resulting DOM
    // mutation would overwrite the width selected manually by the user.
    if (ownsDynamicStartUi(node)) return;
    node.properties ||= {};
    if (!validSize(node.properties.cmkOuterSize)) {
        node.properties.cmkOuterSize = [Number(node.size?.[0]) || 600, Number(node.size?.[1]) || 1225];
        return;
    }
    const stored = node.properties.cmkOuterSize.map(Number);
    if (Number(node.size?.[0]) !== stored[0] || Number(node.size?.[1]) !== stored[1]) {
        node.setSize?.(stored);
    }
}

function prepareNodeElement(nodeElement) {
    const node = nodeForElement(nodeElement);
    if (!node) return;
    enforceClickCompare(node);
    const compareViewports = nodeElement.querySelectorAll(VIEWPORT_SELECTOR);
    if (compareViewports.length > 0) {
        restoreCmkSize(node, nodeElement);
    }

    for (const viewport of compareViewports) {
        if (isCompareOwner(node)) {
            viewport.classList.add("cmk-hold-compare");
            viewport.title = "RESULT · Maustaste gedrückt halten für SOURCE";
        }
        // Some proxy widgets (notably 10/20) replace their DOM viewport after
        // execution_start. Such a replacement must inherit the current reset.
        if (executionActive) viewport.classList.add("cmk-preview-reset");
    }
}

function scan(root = document) {
    if (root instanceof Element) {
        const owner = root.matches(NODE_SELECTOR) ? root : root.closest(NODE_SELECTOR);
        if (owner) prepareNodeElement(owner);
    }
    for (const nodeElement of root.querySelectorAll?.(NODE_SELECTOR) || []) prepareNodeElement(nodeElement);
}

function installStyles() {
    if (document.getElementById("cmk-hold-compare-style")) return;
    const style = document.createElement("style");
    style.id = "cmk-hold-compare-style";
    style.textContent = `
      .cmk-hold-compare img:nth-of-type(2) { clip-path: inset(0 0 0 0) !important; }
      .cmk-hold-compare.cmk-show-source img:nth-of-type(2) { clip-path: inset(0 100% 0 0) !important; }
      .cmk-hold-compare > [role="presentation"] { display: none !important; }
      ${VIEWPORT_SELECTOR}.cmk-preview-reset { visibility: hidden !important; }
      [data-node-id]:not(.outline-node-stroke-executing):has(.cmk-hold-compare) img.pointer-events-none,
      [data-node-id]:not(.outline-node-stroke-executing):has(.cmk-hold-compare) img.pointer-events-none + div {
        display: none !important;
      }
    `;
    document.head.append(style);
}

function installDomBehavior() {
    if (window._cmkHoldCompareDomInstalled) return;
    window._cmkHoldCompareDomInstalled = true;
    installStyles();

    document.addEventListener("pointerdown", (event) => {
        const viewport = event.target?.closest?.(".cmk-hold-compare");
        if (viewport && event.button === 0) {
            viewport.classList.add("cmk-show-source");
            event.preventDefault();
            return;
        }

        const nodeElement = event.target?.closest?.(NODE_SELECTOR);
        const resizeHandle = event.target?.closest?.('[role="button"]');
        if (!nodeElement || !resizeHandle || !/cursor-.*-resize/.test(String(resizeHandle.className))) return;
        const node = nodeForElement(nodeElement);
        if (!isCmkFlow(node)) return;
        if (ownsDynamicStartUi(node)) return;
        if (!nodeElement.querySelector(VIEWPORT_SELECTOR)) return;

        const remember = () => setTimeout(() => {
            if (!validSize(node.size)) return;
            node.properties ||= {};
            node.properties.cmkOuterSize = [Number(node.size[0]), Number(node.size[1])];
        }, 100);
        window.addEventListener("pointerup", remember, { once: true });
        window.addEventListener("pointercancel", remember, { once: true });
    }, true);

    const showResult = () => {
        for (const viewport of document.querySelectorAll(".cmk-hold-compare.cmk-show-source")) {
            viewport.classList.remove("cmk-show-source");
        }
    };
    window.addEventListener("pointerup", showResult, true);
    window.addEventListener("pointercancel", showResult, true);
    window.addEventListener("blur", showResult);

    const observer = new MutationObserver((records) => {
        for (const record of records) {
            revealUpdatedPreview(record);
            for (const added of record.addedNodes) {
                if (added instanceof Element) scan(added);
            }
        }
    });
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["src"],
    });
    scan();
    api.addEventListener("execution_start", clearModuleComparePreviews);
    api.addEventListener("execution_success", finishModuleComparePreviews);
    api.addEventListener("execution_error", () => { executionActive = false; });
    api.addEventListener("execution_interrupted", () => { executionActive = false; });
}

app.registerExtension({
    name: "cmk.image.compare.hold.v8",
    nodeCreated(node) {
        enforceClickCompare(node);
    },
    loadedGraphNode(node) {
        enforceClickCompare(node);
    },
    setup() {
        installDomBehavior();
    },
});
