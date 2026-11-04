import { app } from "../../../scripts/app.js";

const CMK_NODE_CLASS = "CMK_VideoQuickMetrics";
const CMK_DISPLAY_NAME = "CMK Video QuickMetrics";

const METRIC_WIDGETS = [
    "width",
    "height",
    "pixels_per_frame",
    "fps",
    "bit_depth",
    "frames",
    "duration",
];

function isQuickMetrics(node) {
    return node && (
        node.comfyClass === CMK_NODE_CLASS ||
        node.type === CMK_NODE_CLASS ||
        node.constructor?.nodeData?.name === CMK_NODE_CLASS ||
        node.title === CMK_DISPLAY_NAME
    );
}

function isQuickMetricsNodeData(nodeData) {
    return nodeData && (
        nodeData.name === CMK_NODE_CLASS ||
        nodeData.display_name === CMK_DISPLAY_NAME
    );
}

function findWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name);
}

function ensureMetricWidgets(node) {
    if (!isQuickMetrics(node)) return;

    for (const name of METRIC_WIDGETS) {
        let widget = findWidget(node, name);
        if (!widget) {
            widget = node.addWidget("text", name, "—", () => {});
            widget.serialize = false;
        }
        widget.disabled = true;
        widget.readOnly = true;
    }

    node.setDirtyCanvas?.(true, true);
}

function firstScalar(value) {
    if (Array.isArray(value)) return value.length ? String(value[0]) : undefined;
    if (value === null || value === undefined) return undefined;
    if (typeof value === "object") return undefined;
    return String(value);
}

function normalizeMetrics(message) {
    const candidates = [
        message,
        message?.ui,
        message?.output,
        message?.output?.ui,
        message?.data,
        message?.data?.ui,
    ].filter(Boolean);

    for (const candidate of candidates) {
        if (candidate.metrics && typeof candidate.metrics === "object" && !Array.isArray(candidate.metrics)) {
            return candidate.metrics;
        }

        const direct = {};
        let directCount = 0;
        for (const name of METRIC_WIDGETS) {
            const value = firstScalar(candidate[name]);
            if (value !== undefined) {
                direct[name] = value;
                directCount += 1;
            }
        }
        if (directCount) return direct;

        const text = candidate.text;
        if (Array.isArray(text)) {
            const parsed = {};
            for (const item of text) {
                if (typeof item !== "string") continue;
                const index = item.indexOf(":");
                if (index < 0) continue;
                const key = item.slice(0, index).trim();
                const value = item.slice(index + 1).trim();
                if (METRIC_WIDGETS.includes(key)) parsed[key] = value;
            }
            if (Object.keys(parsed).length) return parsed;
        }
    }

    return {};
}

function updateMetricWidgets(node, message) {
    if (!isQuickMetrics(node)) return;
    ensureMetricWidgets(node);

    const metrics = normalizeMetrics(message);
    for (const name of METRIC_WIDGETS) {
        const widget = findWidget(node, name);
        if (!widget) continue;
        widget.value = metrics[name] ?? "—";
    }

    node.setDirtyCanvas?.(true, true);
}

function attachInstanceHandler(node) {
    if (!isQuickMetrics(node) || node.__cmkVideoQuickMetricsPatched) return;
    node.__cmkVideoQuickMetricsPatched = true;

    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function(message) {
        const result = originalOnExecuted?.apply(this, arguments);
        updateMetricWidgets(this, message);
        return result;
    };

    ensureMetricWidgets(node);
}

app.registerExtension({
    name: "cmk.video.quickmetrics.widgets.v27_26",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isQuickMetricsNodeData(nodeData)) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            setTimeout(() => attachInstanceHandler(this), 0);
            return result;
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            const result = originalOnExecuted?.apply(this, arguments);
            updateMetricWidgets(this, message);
            return result;
        };
    },

    nodeCreated(node) {
        setTimeout(() => attachInstanceHandler(node), 0);
    },
});
