import { app } from "../../../scripts/app.js";

const CMK_FLOW_TITLE = /^CMK Flow\s*[·-]/i;
const NODE_SELECTOR = "[data-node-id]";
const VIEWPORT_SELECTOR = '[data-testid="image-compare-viewport"]';

function isCmkFlow(node) {
    return CMK_FLOW_TITLE.test(String(node?.title || node?.type || ""));
}

function isNativeCompare(node) {
    return node?.constructor?.comfyClass === "ImageCompare" || node?.type === "ImageCompare";
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
    restoreCmkSize(node, nodeElement);

    for (const viewport of nodeElement.querySelectorAll(VIEWPORT_SELECTOR)) {
        if (isCmkFlow(node) || isNativeCompare(node)) {
            viewport.classList.add("cmk-hold-compare");
            viewport.title = "RESULT · Maustaste gedrückt halten für SOURCE";
        }
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
            for (const added of record.addedNodes) {
                if (added instanceof Element) scan(added);
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    scan();
}

app.registerExtension({
    name: "cmk.image.compare.hold.v3",
    setup() {
        installDomBehavior();
    },
});
