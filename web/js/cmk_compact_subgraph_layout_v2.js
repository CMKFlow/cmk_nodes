import { app } from "../../../scripts/app.js";

const COMPACT_FLOW_SIZE = [450, 230];
const SUBGRAPH_FOOTER_HEIGHT = 24;
const ADVANCED_SPACER_NAME = "cmk_advanced_combo_spacer";
const WIDGET_ROW_HEIGHT = 26;

function text(value) {
    return String(value ?? "");
}

function nodeTitle(node) {
    return text(
        node?.title
        || node?.getTitle?.()
        || node?.constructor?.title
        || node?.constructor?.type,
    );
}

function hasVisualProviderContract(node) {
    return Array.isArray(node?.properties?.cmkVisualProviders);
}

function isCompactFlowModule(node) {
    return (
        (nodeTitle(node).startsWith("CMK Flow · ") || hasVisualProviderContract(node)) &&
        (node?.isSubgraphNode?.() || hasVisualProviderContract(node))
    );
}

function isAdvancedFlowModule(node) {
    if (nodeTitle(node).includes(" · Advanced")) return true;
    return (node?.properties?.cmkVisualProviders || []).some((provider) =>
        text(provider?.stage_key).endsWith(".advanced"),
    );
}

function visibleWidgets(node) {
    return (node?.widgets || []).filter((widget) => {
        if (!widget || widget.hidden || widget.type === "converted-widget") return false;
        const size = widget.computeSize?.(COMPACT_FLOW_SIZE[0]);
        return !Array.isArray(size) || Number(size[1]) > 0;
    });
}

function ensureAdvancedSpacer(node) {
    if (!isCompactFlowModule(node) || !isAdvancedFlowModule(node)) return;
    if (node.widgets?.some((widget) => widget?.name === ADVANCED_SPACER_NAME)) return;
    if (visibleWidgets(node).length !== 1 || typeof node.addCustomWidget !== "function") return;
    node.addCustomWidget({
        name: ADVANCED_SPACER_NAME,
        type: "cmk-layout-spacer",
        value: null,
        serialize: false,
        computeSize(width) {
            return [Math.max(Number(width) || 0, 1), WIDGET_ROW_HEIGHT];
        },
        draw(ctx, _node, width, y, height) {
            const x = 10;
            const boxWidth = Math.max(1, Number(width) - 20);
            const boxHeight = Math.max(1, Number(height) - 4);
            ctx.save();
            ctx.beginPath();
            ctx.roundRect(x, Number(y) + 2, boxWidth, boxHeight, 7);
            ctx.fillStyle = globalThis.LiteGraph?.WIDGET_BGCOLOR || "#222";
            ctx.fill();
            ctx.strokeStyle = globalThis.LiteGraph?.WIDGET_OUTLINE_COLOR || "#444";
            ctx.stroke();
            ctx.restore();
        },
    });
}

function layoutHeight(widget) {
    const computed = Number(widget?.computedHeight);
    if (computed > 0) return computed;
    const measured = widget?.computeSize?.(COMPACT_FLOW_SIZE[0]);
    const height = Array.isArray(measured) ? Number(measured[1]) : 0;
    return (height > 0 ? height : WIDGET_ROW_HEIGHT) + 4;
}

function applyLayout(node) {
    if (!isCompactFlowModule(node)) return false;
    ensureAdvancedSpacer(node);
    const widgets = visibleWidgets(node);
    if (!widgets.length) return false;
    const totalHeight = widgets.reduce((sum, widget) => sum + layoutHeight(widget), 0);
    const footerTop = Number(node.size[1]) - SUBGRAPH_FOOTER_HEIGHT;
    const startY = footerTop - totalHeight;
    node.widgets_start_y = startY;
    let y = startY;
    for (const widget of widgets) {
        widget.y = y;
        y += layoutHeight(widget);
    }
    if (!node.__cmkCompactLayoutReported) {
        node.__cmkCompactLayoutReported = true;
        console.info(`[CMK Compact Layout] ${nodeTitle(node)} | footer ${footerTop}`);
    }
    node.setDirtyCanvas?.(true, true);
    return true;
}

function install(node) {
    if (!isCompactFlowModule(node) || node.__cmkCompactLayoutV2) return;
    node.__cmkCompactLayoutV2 = true;
    ensureAdvancedSpacer(node);
    const originalResize = node.onResize;
    node.onResize = function() {
        const result = originalResize?.apply(this, arguments);
        this.arrange?.();
        return result;
    };
    const originalArrange = node.arrange;
    node.arrange = function() {
        const result = originalArrange?.apply(this, arguments);
        applyLayout(this);
        this._arrangeWidgetInputSlots?.();
        return result;
    };
    node.arrange?.();
}

function retryInstall(node) {
    for (const delay of [0, 50, 250, 1000]) {
        setTimeout(() => {
            install(node);
        }, delay);
    }
}

app.registerExtension({
    name: "cmk.compact_subgraph_layout.v2",
    setup() {
        setInterval(() => {
            for (const node of app?.graph?._nodes || []) {
                if (isCompactFlowModule(node)) install(node);
            }
        }, 500);
    },
    nodeCreated(node) {
        retryInstall(node);
    },
    loadedGraphNode(node) {
        retryInstall(node);
    },
});
