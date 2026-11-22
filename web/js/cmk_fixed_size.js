// CMK fixed node size (opt-in per node instance).
//
// A workflow can pin the outer size of a node with properties.cmkFixedSize.
// That size wins over every automatic size calculation (content fit after a
// load or execution, LiteGraph's computeSize() growth in loadGraphDataNow)
// and over the Clean View ON/OFF switch, which then keeps the outer size.
// A deliberate resize by the user (drag, CMK Node Dimensions) updates the
// fixed size. Nodes without the property behave exactly as before.

export const FIXED_SIZE_PROPERTY = "cmkFixedSize";

export function fixedSize(node) {
    const value = node?.properties?.[FIXED_SIZE_PROPERTY];
    if (!Array.isArray(value)) return null;
    const size = [Math.round(Number(value[0])), Math.round(Number(value[1]))];
    return Number.isFinite(size[0]) && Number.isFinite(size[1]) && size[0] > 0 && size[1] > 0
        ? size
        : null;
}

// Applies the fixed size; returns false when the node has none, so callers
// fall back to their own sizing.
export function applyFixedSize(node) {
    const size = fixedSize(node);
    if (!size) return false;
    if (Number(node?.size?.[0]) !== size[0] || Number(node?.size?.[1]) !== size[1]) {
        node.setSize?.([...size]);
        node.setDirtyCanvas?.(true, true);
    }
    return true;
}
