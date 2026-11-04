import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {
    CONCAT_NODE_CLASS,
    STACKER_NODE_CLASS,
    collectToggleLoraNames,
    concatenatedToggles,
    createRevisionGuard,
} from "./cmk_trigger_words_concatenate_core.js";

const pendingGraphs = new WeakSet();
const revisionGuard = createRevisionGuard();

function graphId(graph) {
    return graph?.id ?? "root";
}

async function triggerWordsFor(name) {
    const response = await api.fetchApi(
        `/lm/loras/get-trigger-words?name=${encodeURIComponent(name)}`,
        { cache: "no-store" },
    );
    if (!response.ok) throw new Error(`Trigger words request failed for ${name}`);
    const data = await response.json();
    if (!data?.success) throw new Error(data?.error || `Trigger words request failed for ${name}`);
    return Array.isArray(data?.trigger_words)
        ? data.trigger_words.filter((word) => typeof word === "string")
        : [];
}

async function synchronizeToggle(toggle) {
    const names = collectToggleLoraNames(toggle);
    if (names === null) return;

    const revision = revisionGuard.next(toggle);
    try {
        const groups = await Promise.all(names.map(triggerWordsFor));
        if (!revisionGuard.isCurrent(toggle, revision)) return;
        const message = groups.flat().join(",, ");
        api.dispatchEvent(new CustomEvent("trigger_word_update", {
            detail: { id: toggle.id, graph_id: String(graphId(toggle.graph)), message },
        }));
    } catch (error) {
        if (revisionGuard.isCurrent(toggle, revision)) {
            console.error("CMK TriggerWords Concatenate live update failed:", error);
        }
    }
}

function scheduleGraphSync(graph) {
    if (!graph || pendingGraphs.has(graph)) return;
    pendingGraphs.add(graph);
    queueMicrotask(() => {
        pendingGraphs.delete(graph);
        for (const toggle of concatenatedToggles(graph)) synchronizeToggle(toggle);
    });
}

function wrapWidgetCallback(node, widget) {
    if (!widget || widget.callback?._cmkTriggerWordsConcatenate) return;
    const original = widget.callback;
    const callback = function () {
        const result = original?.apply(this, arguments);
        scheduleGraphSync(node.graph);
        return result;
    };
    callback._cmkTriggerWordsConcatenate = true;
    widget.callback = callback;
}

function observeStacker(node) {
    if (!node || node.comfyClass !== STACKER_NODE_CLASS) return;
    wrapWidgetCallback(
        node,
        node.lorasWidget || node.widgets?.find((widget) => widget?.name === "loras"),
    );
    wrapWidgetCallback(
        node,
        node.inputWidget || node.widgets?.find((widget) => widget?.name === "text"),
    );
}

function scheduleObservation(node) {
    for (const delay of [0, 50, 200]) {
        setTimeout(() => {
            observeStacker(node);
            scheduleGraphSync(node?.graph);
        }, delay);
    }
}

function wrapLifecycle(nodeType, observe) {
    for (const hook of ["onNodeCreated", "onConfigure", "onAdded", "onConnectionsChange"]) {
        const original = nodeType.prototype[hook];
        nodeType.prototype[hook] = function () {
            const result = original?.apply(this, arguments);
            if (observe) scheduleObservation(this);
            else scheduleGraphSync(this.graph);
            return result;
        };
    }
}

app.registerExtension({
    name: "cmk.trigger_words_concatenate.live.v1",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === CONCAT_NODE_CLASS) wrapLifecycle(nodeType, false);
        if (nodeData?.name === STACKER_NODE_CLASS) wrapLifecycle(nodeType, true);
        if (nodeData?.name === "TriggerWord Toggle (LoraManager)") wrapLifecycle(nodeType, false);
    },
    nodeCreated(node) {
        if (node?.comfyClass === STACKER_NODE_CLASS) scheduleObservation(node);
        else if (
            node?.comfyClass === CONCAT_NODE_CLASS ||
            node?.comfyClass === "TriggerWord Toggle (LoraManager)"
        ) scheduleGraphSync(node.graph);
    },
    loadedGraphNode(node) {
        if (node?.comfyClass === STACKER_NODE_CLASS) scheduleObservation(node);
        else scheduleGraphSync(node?.graph);
    },
});
