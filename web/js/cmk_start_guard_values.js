// Saved START HERE values with or without the ContentGuard slot.
//
// content_guard_enabled exists only in BASELINE and sits inside the widget
// list, so a workflow saved in one distribution mode holds one value more or
// less than the other mode expects.  The slot is located through the complete
// widget order, and the layout is accepted only if every saved value fits the
// widget it would belong to; anything else stays as saved.

export const GUARD_WIDGET = "content_guard_enabled";
export const GUARD_PREDECESSOR = "live_preview_zit";

function fits(value, kind) {
    return value === null || value === undefined || kind === "undefined" || typeof value === kind;
}

function tailFits(values, kinds) {
    return values.length <= kinds.length && values.every((value, index) => fits(value, kinds[index]));
}

// widgets: [{name, kind}] of the node's serialized widgets in saved order.
export function alignGuardValues(widgets, values) {
    if (!Array.isArray(values)) return values;
    const names = widgets.map((widget) => widget.name);
    const hasGuard = names.includes(GUARD_WIDGET);
    const slot = hasGuard ? names.indexOf(GUARD_WIDGET) : names.indexOf(GUARD_PREDECESSOR) + 1;
    if (slot <= 0 || values.length < slot) return values;
    const kinds = widgets.map((widget) => widget.kind);
    if (!values.slice(0, slot).every((value, index) => fits(value, kinds[index]))) return values;
    const following = kinds.slice(hasGuard ? slot + 1 : slot);
    const tail = values.slice(slot);
    const savedWithGuard = tailFits(tail, ["boolean", ...following]);
    const savedWithoutGuard = tailFits(tail, following);
    if (savedWithGuard === savedWithoutGuard) return values;
    if (!hasGuard && savedWithGuard) return [...values.slice(0, slot), ...tail.slice(1)];
    if (hasGuard && savedWithoutGuard) return [...values.slice(0, slot), true, ...tail];
    return values;
}
