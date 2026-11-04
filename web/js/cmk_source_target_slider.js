const MIN_BALANCE = -2;
const MAX_BALANCE = 2;
const STYLE_ID = "cmk-source-target-native-slider-style";

function ensureNativeSliderStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        [data-testid="node-widget"]:has([data-slot="slider"][aria-label^="SAMPLING START"]) {
            align-items: center;
        }
        div.grid:has([data-slot="slider"][aria-label^="SAMPLING START"])
            > [data-testid="widget-layout-field-label"] {
            font-size: 11px;
            font-weight: 600;
            color: var(--fg-color, #ddd);
        }
        div.grid:has([data-slot="slider"][aria-label^="SAMPLING START"])
            > .relative > div {
            border: 0 !important;
            box-shadow: none !important;
        }
        div.grid:has([data-slot="slider"][aria-label^="SAMPLING START"])
            div.flex:has(> [data-slot="slider"]) {
            gap: 10px !important;
            padding: 0 2px !important;
            background: transparent !important;
        }
        div.grid:has([data-slot="slider"][aria-label^="SAMPLING START"])
            div.flex:has(> [data-slot="slider"])::after {
            content: "TARGET";
            flex: 0 0 auto;
            font-size: 11px;
            font-weight: 600;
            color: var(--fg-color, #ddd);
        }
        div.grid:has([data-slot="slider"][aria-label^="SAMPLING START"])
            [data-slot="slider-range"] {
            display: none !important;
        }
        div.grid:has([data-slot="slider"][aria-label^="SAMPLING START"])
            .w-16:has(input[aria-label^="SAMPLING START"]) {
            display: none !important;
        }
        [data-testid="node-widget"]:has([data-slot="slider"][aria-label^="HYBRID BALANCE"]) {
            align-items: center;
        }
        div.grid:has([data-slot="slider"][aria-label^="HYBRID BALANCE"])
            > [data-testid="widget-layout-field-label"] {
            font-size: 11px;
            font-weight: 600;
            color: var(--fg-color, #ddd);
        }
        div.grid:has([data-slot="slider"][aria-label^="HYBRID BALANCE"])
            > .relative > div {
            border: 0 !important;
            box-shadow: none !important;
        }
        div.grid:has([data-slot="slider"][aria-label^="HYBRID BALANCE"])
            div.flex:has(> [data-slot="slider"]) {
            gap: 10px !important;
            padding: 0 2px !important;
            background: transparent !important;
        }
        div.grid:has([data-slot="slider"][aria-label^="HYBRID BALANCE"])
            div.flex:has(> [data-slot="slider"])::after {
            content: "ZIT";
            flex: 0 0 auto;
            font-size: 11px;
            font-weight: 600;
            color: var(--fg-color, #ddd);
        }
        div.grid:has([data-slot="slider"][aria-label^="HYBRID BALANCE"])
            [data-slot="slider-range"] {
            display: none !important;
        }
        div.grid:has([data-slot="slider"][aria-label^="HYBRID BALANCE"])
            .w-16:has(input[aria-label^="HYBRID BALANCE"]) {
            display: none !important;
        }
    `;
    document.head.appendChild(style);
}

function clampBalance(value, minimum = MIN_BALANCE, maximum = MAX_BALANCE, fallback = 0) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < minimum || numeric > maximum) return fallback;
    return Math.max(minimum, Math.min(maximum, Math.round(numeric)));
}

export function installSourceTargetSlider(
    node,
    widget,
    visibleName,
    leftLabel = "SOURCE",
    rightLabel = "TARGET",
    minimum = MIN_BALANCE,
    maximum = MAX_BALANCE,
    fallback = 0,
) {
    if (!node || !widget) return null;
    ensureNativeSliderStyle();
    const marker = `_cmkSourceTarget_${visibleName}`;
    if (node[marker]) return node[marker];

    // Existing subgraphs stored an absolute start step (normally 10). The new
    // control stores only the relative SOURCE/TARGET balance, centered at zero.
    widget.value = clampBalance(widget.value, minimum, maximum, fallback);
    widget.type = "slider";
    widget.label = leftLabel;
    widget.options = {
        ...(widget.options || {}),
        min: minimum,
        max: maximum,
        step: 1,
        precision: 0,
    };
    widget._cmkSyncFromCanonical = () => {
        widget.value = clampBalance(widget.value, minimum, maximum, fallback);
    };

    node[marker] = widget;
    return widget;
}
