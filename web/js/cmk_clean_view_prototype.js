import { app } from "../../../scripts/app.js";

const EXTENSION_NAME = "cmk.clean_view.prototype.v1";
const STYLE_ID = "cmk-clean-view-prototype-style";
const ROOT_CLASS = "cmk-clean-view-prototype";
const WRAPPER_CLASS = "cmk-clean-view-slot-wrapper";
const CONTRACT_CLASS = "cmk-clean-view-contract-slot";
const BAR_CLASS = "cmk-clean-view-connections-bar";
const PROPERTY = "cmkCleanView";
const COMPACT_SIZE_PROPERTY = "cmkCleanViewSize";
const EXPANDED_HEIGHT_PROPERTY = "cmkCleanViewExpandedHeight";
const EXPANDED_SIZE_PROPERTY = "cmkCleanViewExpandedSize";
const MENU_ON = "CMK · Clean View";
const MENU_OFF = "CMK · Exit Clean View";
const CANONICAL_PATTERN = /^(MODEL|PROCESS|IMAGE|LOG|VISUAL|SAMPLED)(?: (?:SDXL|ZIT|Z-IMAGE TURBO))?$/;
const SLOT_COLOR_FALLBACKS = {
    MODEL: "#b39ddb",
    PROCESS: "#d9b86c",
    IMAGE: "#64b5f6",
    MASK: "#81c784",
    SAMPLED: "#ff8a65",
    LOG: "#b0bec5",
    VISUAL: "#4dd0e1",
    DIAGNOSTIC: "#ef6c75",
    BOOLEAN: "#8bc34a",
    STRING: "#f2c94c",
};

let scanScheduled = false;

function text(value) {
    return String(value ?? "").trim();
}

function slotName(slot) {
    return text(slot?.name || slot?.localized_name || slot?.label).toUpperCase();
}

function isCanonical(slot) {
    return CANONICAL_PATTERN.test(slotName(slot));
}

function slotType(slot) {
    const value = Array.isArray(slot?.type) ? slot.type[0] : slot?.type;
    return text(value).toUpperCase();
}

function colorMapValue(map, key) {
    if (!map || !key) return null;
    if (typeof map.get === "function") return map.get(key) || map.get(key.toLowerCase()) || null;
    return map[key] || map[key.toLowerCase()] || null;
}

function slotColor(slot) {
    const type = slotType(slot);
    const direct = slot?.color_on || slot?.color || slot?.color_off;
    if (typeof direct === "string" && direct) return direct;

    for (const map of [
        app.canvas?.default_connection_color_byType,
        app.canvas?.link_type_colors,
        globalThis.LGraphCanvas?.link_type_colors,
        globalThis.LiteGraph?.link_type_colors,
    ]) {
        const color = colorMapValue(map, type);
        if (typeof color === "string" && color) return color;
    }

    for (const [token, color] of Object.entries(SLOT_COLOR_FALLBACKS)) {
        if (type.includes(token) || slotName(slot).includes(token)) return color;
    }
    return "#94a3b8";
}

function slotColorFill(slots) {
    const colors = [...new Set((slots || []).map(slotColor).filter(Boolean))];
    if (colors.length === 0) return "#94a3b8";
    if (colors.length === 1) return colors[0];
    const stops = colors.flatMap((color, index) => {
        const start = (index / colors.length) * 100;
        const end = ((index + 1) / colors.length) * 100;
        return [`${color} ${start}%`, `${color} ${end}%`];
    });
    return `linear-gradient(to bottom, ${stops.join(", ")})`;
}

function nodeIdentity(node) {
    return [
        node?.title,
        node?.type,
        node?.comfyClass,
        node?.constructor?.comfyClass,
        node?.constructor?.nodeData?.name,
        node?.constructor?.nodeData?.display_name,
    ].map(text);
}

function isCmkNode(node) {
    return nodeIdentity(node).some((value) => /^CMK(?:\s|_|·|-)/i.test(value))
        || (
            Array.isArray(node?.properties?.cmkOuterSize)
            && Number(node.properties.cmkOuterSize[0]) > 0
            && Number(node.properties.cmkOuterSize[1]) > 0
        );
}

function isFlowStart(node) {
    return nodeIdentity(node).some((value) =>
        value === "CMKPipeCreateImage" || /01 START HERE/i.test(value),
    );
}

function isLoadImage(node) {
    return nodeIdentity(node).some((value) =>
        value === "CMKLoadImage" || /^CMK LOAD IMAGE$/i.test(value),
    );
}

function isRefiner20(node) {
    return node?.properties?.cmkVisualProviders?.some?.((provider) =>
        Number(provider?.sequence) === 20,
    ) || nodeIdentity(node).some((value) => value === "CMKRefinerPipe");
}

function isMaskDetailer95(node) {
    return node?.properties?.cmkVisualProviders?.some?.((provider) =>
        Number(provider?.sequence) === 95,
    ) || nodeIdentity(node).some((value) => value === "CMKMaskDetailer");
}

function isBoundary35Combined(node) {
    return nodeIdentity(node).some((value) => value === "CMKFamilyResultMergePipe");
}

function isCleanSlot(node, slot) {
    // Clean View is now deliberately connector-free. The aggregate docking
    // points retain the wiring presence and type colors without reserving a
    // permanent row for rarely changed individual ports.
    return Boolean(slot);
}

function visibleInputSlots(node) {
    return (node?.inputs || []).filter((input) => !input?.widget);
}

function eligible(node) {
    if (!isCmkNode(node)) return false;
    const sockets = [...visibleInputSlots(node), ...(node?.outputs || [])];
    return sockets.filter((slot) => isCleanSlot(node, slot)).length >= 2;
}

function graphNode(nodeId) {
    for (const graph of [app.canvas?.graph, app.graph, app.rootGraph]) {
        for (const candidate of [nodeId, Number(nodeId)]) {
            if (candidate == null || candidate === "" || Number.isNaN(candidate)) continue;
            const node = graph?.getNodeById?.(candidate);
            if (node) return node;
        }
    }
    return null;
}

function nodeForRoot(root) {
    return graphNode(root?.getAttribute?.("data-node-id"));
}

function cleanEnabled(node) {
    return node?.properties?.[PROPERTY] === true;
}

function validSize(value) {
    return Array.isArray(value)
        && Number.isFinite(Number(value[0]))
        && Number.isFinite(Number(value[1]))
        && Number(value[0]) > 0
        && Number(value[1]) > 0;
}

function rememberCompactSize(node, size = node?.size) {
    if (!validSize(size)) return false;
    node.properties ||= {};
    node.properties[COMPACT_SIZE_PROPERTY] = [
        Math.round(Number(size[0])),
        Math.round(Number(size[1])),
    ];
    return true;
}

function rememberExpandedSize(node, size = node?.size) {
    if (!validSize(size)) return false;
    const expanded = [
        Math.round(Number(size[0])),
        Math.round(Number(size[1])),
    ];
    node.properties ||= {};
    node.properties[EXPANDED_SIZE_PROPERTY] = expanded;
    node.properties[EXPANDED_HEIGHT_PROPERTY] = expanded[1];
    return true;
}

function applyCompactSize(node, size, remember = false) {
    if (!validSize(size)) return false;
    const target = [Math.round(Number(size[0])), Math.round(Number(size[1]))];
    if (remember) rememberCompactSize(node, target);
    if (Number(node?.size?.[0]) === target[0] && Number(node?.size?.[1]) === target[1]) {
        return false;
    }
    node.setSize?.(target);
    node.setDirtyCanvas?.(true, true);
    return true;
}

function restoreCompactSize(node) {
    if (!cleanEnabled(node)) return false;
    if (node.__cmkManualResizeActive) return false;
    return applyCompactSize(node, node?.properties?.[COMPACT_SIZE_PROPERTY]);
}

function scheduleCompactSizeRestore(node) {
    for (const delay of [0, 60, 200, 750]) {
        window.setTimeout(() => {
            if (!cleanEnabled(node)) return;
            if (restoreCompactSize(node)) app.graph?.setDirtyCanvas?.(true, true);
        }, delay);
    }
}

function clearNativeCollapse(node) {
    if (node?.flags?.collapsed !== true) return false;
    node.flags.collapsed = false;
    return true;
}

function installStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .${ROOT_CLASS} .${WRAPPER_CLASS} {
            position: relative !important;
            min-height: 28px !important;
            padding-top: 28px !important;
        }

        .${ROOT_CLASS} .${WRAPPER_CLASS} .${CONTRACT_CLASS} {
            position: absolute !important;
            top: 2px !important;
            z-index: 2 !important;
            width: 24px !important;
            height: 24px !important;
            min-height: 24px !important;
            margin: 0 !important;
            padding: 0 !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }

        .${ROOT_CLASS} .${WRAPPER_CLASS} .${CONTRACT_CLASS}.lg-slot--input {
            left: 0 !important;
            right: auto !important;
        }

        .${ROOT_CLASS} .${WRAPPER_CLASS} .${CONTRACT_CLASS}.lg-slot--output {
            right: 0 !important;
            left: auto !important;
        }

        .${ROOT_CLASS} .${BAR_CLASS} {
            position: absolute;
            inset: 0 0 auto 0;
            z-index: 1;
            box-sizing: border-box;
            display: flex;
            height: 28px;
            align-items: center;
            justify-content: center;
            border-bottom: 1px solid color-mix(in srgb, var(--border-color, #777) 42%, transparent);
            pointer-events: none;
        }

        .${ROOT_CLASS} .${BAR_CLASS}::before,
        .${ROOT_CLASS} .${BAR_CLASS}::after {
            position: absolute;
            top: 7px;
            box-sizing: border-box;
            width: 8px;
            height: 14px;
            border: 1px solid rgba(12, 15, 18, .9);
            border-radius: 999px;
            background: var(--cmk-clean-input-fill, #94a3b8);
            filter: saturate(.62) brightness(.88);
            opacity: .78;
            content: "";
        }

        .${ROOT_CLASS} .${BAR_CLASS}::before { left: -4px; }
        .${ROOT_CLASS} .${BAR_CLASS}::after {
            right: -4px;
            background: var(--cmk-clean-output-fill, #94a3b8);
        }
        .${ROOT_CLASS} .${BAR_CLASS}:not(.cmk-clean-view-has-input)::before {
            display: none;
        }
        .${ROOT_CLASS} .${BAR_CLASS}:not(.cmk-clean-view-has-output)::after {
            display: none;
        }
    `;
    document.head.appendChild(style);
}

function slotWrapper(body) {
    return Array.from(body?.children || []).find((element) =>
        element?.querySelector?.(":scope .lg-slot--input, :scope .lg-slot--output"),
    ) || null;
}

function clearDecoration(root) {
    root?.classList?.remove(ROOT_CLASS);
    const wrapper = root?.querySelector?.(`.${WRAPPER_CLASS}`);
    wrapper?.classList?.remove(WRAPPER_CLASS);
    for (const slot of root?.querySelectorAll?.(`.${CONTRACT_CLASS}`) || []) {
        slot.classList.remove(CONTRACT_CLASS);
        slot.style.removeProperty("margin");
    }
    root?.querySelector?.(`.${BAR_CLASS}`)?.remove();
}

function markCanonicalRows(node, rows, slots) {
    rows.forEach((row, index) => {
        const hidden = isCleanSlot(node, slots[index]);
        row.classList.toggle(CONTRACT_CLASS, hidden);
        if (hidden) row.style.setProperty("margin", "0", "important");
        else row.style.removeProperty("margin");
    });
}

function addConnectionsBar(root, wrapper, node) {
    let bar = wrapper.querySelector(`:scope > .${BAR_CLASS}`);
    if (!bar) {
        bar = document.createElement("div");
        bar.className = BAR_CLASS;
        bar.setAttribute("aria-hidden", "true");
        wrapper.prepend(bar);
    }
    const inputSlots = visibleInputSlots(node).filter((slot) => isCleanSlot(node, slot));
    const outputSlots = (node.outputs || []).filter((slot) => isCleanSlot(node, slot));
    const hasInput = inputSlots.length > 0;
    const hasOutput = outputSlots.length > 0;
    bar.classList.toggle("cmk-clean-view-has-input", hasInput);
    bar.classList.toggle("cmk-clean-view-has-output", hasOutput);
    bar.style.setProperty("--cmk-clean-input-fill", slotColorFill(inputSlots));
    bar.style.setProperty("--cmk-clean-output-fill", slotColorFill(outputSlots));
    root.classList.add(ROOT_CLASS);
    wrapper.classList.add(WRAPPER_CLASS);
}

function decorate(root) {
    const node = nodeForRoot(root);
    if (!node || !eligible(node)) {
        clearDecoration(root);
        return;
    }
    // The desktop can rebuild the DOM without recreating the graph node. Keep
    // the context-menu hook alive even while Clean View is disabled.
    installMenu(node);
    if (!cleanEnabled(node)) {
        clearDecoration(root);
        return;
    }
    // CMK Clean View is the only compact state while it is active. A native
    // ComfyUI collapse flag can be persisted independently by older workflows
    // and otherwise wins again when the desktop rebuilds the node DOM.
    if (clearNativeCollapse(node)) {
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    }
    if (isBoundary35Combined(node)
        && !validSize(node?.properties?.[COMPACT_SIZE_PROPERTY])) {
        rememberCompactSize(node, [230, 100]);
    }

    const body = root.querySelector(`[data-testid="node-body-${CSS.escape(String(node.id))}"]`)
        || root.querySelector('[data-testid^="node-body-"]');
    const wrapper = slotWrapper(body);
    if (!wrapper) return;

    const inputRows = Array.from(wrapper.querySelectorAll(".lg-slot--input"));
    const outputRows = Array.from(wrapper.querySelectorAll(".lg-slot--output"));
    markCanonicalRows(node, inputRows, visibleInputSlots(node));
    markCanonicalRows(node, outputRows, node.outputs || []);
    addConnectionsBar(root, wrapper, node);

    // The Vue renderer measures the undecorated port rows during some mount
    // sequences and can replace a manually selected compact height with that
    // transient content height. Remember the graph size before that happens,
    // then restore it only after the contract rows have been removed from the
    // normal layout flow.
    if (!validSize(node?.properties?.[COMPACT_SIZE_PROPERTY])) {
        rememberCompactSize(node);
    }
    restoreCompactSize(node);
    if (!root.__cmkCleanViewSizeRestoreScheduled) {
        root.__cmkCleanViewSizeRestoreScheduled = true;
        scheduleCompactSizeRestore(node);
    }
}

function scan() {
    scanScheduled = false;
    installStyle();
    for (const root of document.querySelectorAll('.lg-node[data-node-id]')) decorate(root);
}

function scheduleScan() {
    if (scanScheduled) return;
    scanScheduled = true;
    requestAnimationFrame(scan);
}

function compactHeight(node) {
    const inputs = visibleInputSlots(node);
    const outputs = node?.outputs || [];
    const normalRows = Math.max(inputs.length, outputs.length);
    const specialRows = Math.max(
        inputs.filter((slot) => !isCleanSlot(node, slot)).length,
        outputs.filter((slot) => !isCleanSlot(node, slot)).length,
    );
    return Math.max(0, normalRows * 24 - (28 + specialRows * 24));
}

function resizeForState(node, enabled) {
    const width = Number(node?.size?.[0]);
    const height = Number(node?.size?.[1]);
    if (!Number.isFinite(width) || !Number.isFinite(height)) return;

    if (enabled) {
        rememberExpandedSize(node, [width, height]);
        if (isBoundary35Combined(node)) {
            applyCompactSize(node, [230, 100], true);
            return;
        }
        const rememberedCompact = node.properties?.[COMPACT_SIZE_PROPERTY];
        if (validSize(rememberedCompact)) {
            applyCompactSize(node, rememberedCompact);
            return;
        }
        const reduction = compactHeight(node);
        const compact = [width, reduction > 0 ? Math.max(60, height - reduction) : height];
        applyCompactSize(node, compact, true);
    } else {
        const expandedSize = node.properties?.[EXPANDED_SIZE_PROPERTY];
        if (validSize(expandedSize)) {
            node.setSize?.([Number(expandedSize[0]), Number(expandedSize[1])]);
            return;
        }
        const expandedHeight = Number(node?.properties?.[EXPANDED_HEIGHT_PROPERTY]);
        if (Number.isFinite(expandedHeight) && expandedHeight > 0) {
            node.setSize?.([width, expandedHeight]);
        }
    }
}

function setCleanView(node, enabled) {
    if (!eligible(node)) return;
    node.properties ||= {};
    const collapseCleared = enabled && clearNativeCollapse(node);
    if (cleanEnabled(node) === enabled) {
        if (collapseCleared) {
            node.setDirtyCanvas?.(true, true);
            app.graph?.setDirtyCanvas?.(true, true);
            scheduleScan();
        }
        return;
    }
    resizeForState(node, enabled);
    node.properties[PROPERTY] = enabled;
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    scheduleScan();
}

function addMenuOption(node, options) {
    if (!eligible(node) || !Array.isArray(options)) return;
    const enabled = cleanEnabled(node);
    const label = enabled ? MENU_OFF : MENU_ON;
    if (options.some((option) => option?.content === label)) return;
    options.unshift({
        content: label,
        callback: () => setCleanView(node, !enabled),
    });
}

function installMenu(node) {
    if (!eligible(node)) return;
    if (node.__cmkCleanViewPrototypeMenu
        && node.__cmkCleanViewPrototypeMenu === node.getExtraMenuOptions) return;
    const original = typeof node.getExtraMenuOptions === "function"
        ? node.getExtraMenuOptions
        : null;
    const wrapped = function(canvas, options) {
        const result = original?.apply(this, arguments);
        addMenuOption(this, options);
        return result;
    };
    node.__cmkCleanViewPrototypeMenu = wrapped;
    node.getExtraMenuOptions = wrapped;
}

function isOuterCmkFlowSubgraph(node) {
    return isCmkNode(node)
        && Array.isArray(node?.properties?.cmkOuterSize)
        && eligible(node);
}

function rebindFlowSubgraphMenus() {
    const graphs = [app.canvas?.graph, app.graph, app.rootGraph];
    const seen = new Set();
    for (const graph of graphs) {
        for (const node of graph?._nodes || []) {
            if (seen.has(node) || !isOuterCmkFlowSubgraph(node)) continue;
            seen.add(node);
            installMenu(node);
        }
    }
}

app.registerExtension({
    name: EXTENSION_NAME,

    setup() {
        installStyle();
        scheduleScan();
        rebindFlowSubgraphMenus();
        window.setInterval(rebindFlowSubgraphMenus, 750);
        const observer = new MutationObserver(scheduleScan);
        observer.observe(document.documentElement, {
            childList: true,
            subtree: true,
        });
    },

    nodeCreated(node) {
        installMenu(node);
        scheduleScan();
    },

    loadedGraphNode(node) {
        installMenu(node);
        scheduleScan();
    },
});
