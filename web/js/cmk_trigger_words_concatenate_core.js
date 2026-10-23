export const CONCAT_NODE_CLASS = "CMKTriggerWordsConcatenate";
export const STACKER_NODE_CLASS = "Lora Stacker (LoraManager)";
export const TOGGLE_NODE_CLASS = "TriggerWord Toggle (LoraManager)";

export function getGraphLink(graph, linkId) {
    if (!graph || graph.links == null || linkId == null) return null;
    return typeof graph.links.get === "function"
        ? graph.links.get(linkId) || null
        : graph.links[linkId] || null;
}

function nodeClass(node) {
    return node?.comfyClass || node?.type || node?.constructor?.comfyClass || "";
}

function sourceForInput(node, input) {
    const link = getGraphLink(node?.graph, input?.link);
    if (!link) return null;
    const source = node.graph?.getNodeById?.(link.origin_id);
    return source ? { node: source, output: Number(link.origin_slot) } : null;
}

function isActive(node) {
    return node?.mode === undefined || node.mode === 0 || node.mode === 3;
}

function isTriggerWordsOutput(node, outputIndex) {
    const name = String(node?.outputs?.[outputIndex]?.name || "").toLowerCase();
    return name === "trigger_words" || name === "trigger words";
}

function collectLoraStackNames(node, visited) {
    if (!node || !isActive(node)) return [];
    const key = node.id ?? node;
    if (visited.has(key)) return [];
    visited.add(key);

    const names = [];
    for (const input of node.inputs || []) {
        if (input?.type !== "LORA_STACK" || input.link == null) continue;
        const source = sourceForInput(node, input);
        if (source) names.push(...collectLoraStackNames(source.node, visited));
    }

    const widget = node.lorasWidget || node.widgets?.find((item) => item?.name === "loras");
    if (nodeClass(node) === STACKER_NODE_CLASS && Array.isArray(widget?.value)) {
        for (const lora of widget.value) {
            if (lora?.active && typeof lora.name === "string" && lora.name) names.push(lora.name);
        }
    }
    return names;
}

function collectTriggerSource(source, visited) {
    if (!source?.node) return [];
    const currentClass = nodeClass(source.node);
    const key = `${source.node.id ?? "node"}:${source.output}`;
    if (visited.has(key)) return [];
    visited.add(key);

    if (currentClass === STACKER_NODE_CLASS) {
        if (!isTriggerWordsOutput(source.node, source.output)) return [];
        return collectLoraStackNames(source.node, new Set());
    }
    if (currentClass !== CONCAT_NODE_CLASS) return [];

    const names = [];
    for (const inputName of ["trigger_words_1", "trigger_words_2"]) {
        const input = source.node.inputs?.find((item) => item?.name === inputName);
        const upstream = sourceForInput(source.node, input);
        if (upstream) names.push(...collectTriggerSource(upstream, visited));
    }
    return names;
}

export function collectToggleLoraNames(toggleNode) {
    const input = toggleNode?.inputs?.find((item) => item?.name === "trigger_words");
    const source = sourceForInput(toggleNode, input);
    if (!source || nodeClass(source.node) !== CONCAT_NODE_CLASS) return null;
    return collectTriggerSource(source, new Set());
}

export function graphNodes(graph) {
    if (Array.isArray(graph?._nodes)) return graph._nodes;
    if (Array.isArray(graph?.nodes)) return graph.nodes;
    return [];
}

export function concatenatedToggles(graph) {
    return graphNodes(graph).filter(
        (node) => nodeClass(node) === TOGGLE_NODE_CLASS && collectToggleLoraNames(node) !== null,
    );
}

export function createRevisionGuard() {
    const revisions = new WeakMap();
    return {
        next(target) {
            const revision = (revisions.get(target) || 0) + 1;
            revisions.set(target, revision);
            return revision;
        },
        isCurrent(target, revision) {
            return revisions.get(target) === revision;
        },
    };
}
