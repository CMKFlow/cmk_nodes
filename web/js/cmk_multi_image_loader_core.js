// CMK Multi Image Loader: frontend-only manager for four image_file references.
//
// The node never reaches the backend prompt.  After ComfyUI built the prompt,
// every connected consumer input receives the reference of its own slot as a
// literal value, so the four consumers share no ancestor and keep independent
// ComfyUI and CMK cache identities.

export const NODE_TYPE = "CMKMultiImageLoader";
export const PROPERTY = "cmkMultiImageLoader";
export const SLOTS = [
    { output: "CONTROLNET", short: "CN" },
    { output: "IDENTITY A", short: "ID A" },
    { output: "IDENTITY B", short: "ID B" },
    { output: "IDENTITY C", short: "ID C" },
];

const MUTED = 2;
const BYPASSED = 4;
const SUBGRAPH_INPUT_NODE = -10;
const SUBGRAPH_OUTPUT_NODE = -20;

export function defaultState() {
    return { slots: SLOTS.map(() => ({ file: "", enabled: true })), selected: 0 };
}

// A fresh, complete copy: duplicated nodes and loaded workflows never share
// slot objects.
export function normalizeState(value) {
    const state = defaultState();
    if (!value || typeof value !== "object") return state;
    const slots = Array.isArray(value.slots) ? value.slots : [];
    state.slots = SLOTS.map((_slot, index) => ({
        file: String(slots[index]?.file ?? ""),
        enabled: slots[index]?.enabled !== false,
    }));
    const selected = Number(value.selected);
    state.selected = Number.isInteger(selected) && selected >= 0 && selected < SLOTS.length ? selected : 0;
    return state;
}

export function missingFileValue(index) {
    return `CMK Multi Image Loader · ${SLOTS[index].short}: no file selected`;
}

// The value a slot hands to its consumers.
export function slotValue(state, index) {
    const slot = state.slots[index];
    if (!slot.enabled) return "";
    return slot.file || missingFileValue(index);
}

function linkById(graph, id) {
    const links = graph?.links;
    if (!links) return null;
    return typeof links.get === "function" ? links.get(id) : links[id];
}

function linksOf(output) {
    return Array.isArray(output?.links) ? output.links : [];
}

export class MultiImageInjectionError extends Error {}

// Consumer inputs fed by output `slot` of `node` in `graph`.  `path` holds the
// subgraph nodes from the root graph down to `graph` ({node, parentGraph}).
export function slotConsumers(graph, node, slot, path = []) {
    const found = [];
    const visit = (currentGraph, sourceLinks, currentPath, seen) => {
        for (const linkId of sourceLinks) {
            const link = linkById(currentGraph, linkId);
            if (!link) continue;
            const key = `${currentPath.map((step) => step.node.id).join(":")}|${linkId}`;
            if (seen.has(key)) continue;
            seen.add(key);
            if (link.target_id === SUBGRAPH_OUTPUT_NODE) {
                // Leaves the subgraph through one of its outputs.
                const step = currentPath.at(-1);
                if (!step) continue;
                visit(step.parentGraph, linksOf(step.node.outputs?.[link.target_slot]), currentPath.slice(0, -1), seen);
                continue;
            }
            const target = currentGraph.getNodeById(link.target_id);
            if (!target || target.mode === MUTED) continue;
            if (target.isSubgraphNode?.()) {
                const subgraph = target.subgraph;
                const entry = subgraph?.inputs?.[link.target_slot];
                visit(subgraph, entry?.linkIds ?? [], [...currentPath, { node: target, parentGraph: currentGraph }], seen);
                continue;
            }
            if (target.isVirtualNode || target.mode === BYPASSED) {
                // Reroutes and bypassed nodes pass the reference on unchanged.
                for (const output of target.outputs ?? []) visit(currentGraph, linksOf(output), currentPath, seen);
                continue;
            }
            const input = target.inputs?.[link.target_slot];
            found.push({
                id: [...currentPath.map((step) => step.node.id), target.id].join(":"),
                input: input?.name,
                classType: target.comfyClass ?? target.type,
            });
        }
    };
    visit(graph, linksOf(node.outputs?.[slot]), path, new Set());
    return found;
}

// Every Multi Image Loader in the root graph and in the subgraph instances
// below it, with the path that addresses its consumers in the prompt.
export function findLoaders(rootGraph) {
    const loaders = [];
    const walk = (graph, path, depth) => {
        if (!graph || depth > 32) return;
        for (const node of graph.nodes ?? graph._nodes ?? []) {
            if (node.type === NODE_TYPE && node.mode !== MUTED) loaders.push({ graph, node, path });
            if (node.isSubgraphNode?.() && node.mode !== MUTED) {
                walk(node.subgraph, [...path, { node, parentGraph: graph }], depth + 1);
            }
        }
    };
    walk(rootGraph, [], 0);
    return loaders;
}

// Writes the references into `output` (the API prompt) and verifies every
// consumer.  Throws MultiImageInjectionError instead of letting a connected
// consumer fall back to its local file selection.
export function injectReferences(rootGraph, output) {
    const report = [];
    const assigned = new Map();
    for (const { graph, node, path } of findLoaders(rootGraph)) {
        const state = normalizeState(node.properties?.[PROPERTY]);
        SLOTS.forEach((slot, index) => {
            const value = slotValue(state, index);
            for (const consumer of slotConsumers(graph, node, index, path)) {
                const entry = output?.[consumer.id];
                const where = `${slot.output} → node ${consumer.id} (${consumer.classType ?? "unknown"})`;
                if (!entry || typeof entry !== "object" || !entry.inputs) {
                    throw new MultiImageInjectionError(
                        `CMK Multi Image Loader: ${where} is missing in the prompt; the reference cannot be delivered.`,
                    );
                }
                if (consumer.classType && entry.class_type !== consumer.classType) {
                    throw new MultiImageInjectionError(
                        `CMK Multi Image Loader: ${where} resolved to ${entry.class_type}.`,
                    );
                }
                if (!consumer.input) {
                    throw new MultiImageInjectionError(`CMK Multi Image Loader: ${where} has no input name.`);
                }
                const key = `${consumer.id}|${consumer.input}`;
                if (assigned.has(key) && assigned.get(key) !== value) {
                    throw new MultiImageInjectionError(
                        `CMK Multi Image Loader: ${where} is fed by two different references.`,
                    );
                }
                assigned.set(key, value);
                entry.inputs[consumer.input] = value;
                report.push({ slot: slot.output, id: consumer.id, input: consumer.input, value });
            }
        });
    }
    return report;
}
