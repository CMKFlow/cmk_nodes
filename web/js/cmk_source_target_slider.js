const MIN_BALANCE = -2;
const MAX_BALANCE = 2;
const STYLE_ID = "cmk-source-target-native-slider-style";

function ensureNativeSliderStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        div.grid:has([data-slot="slider"][aria-label^="SAMPLING START"]) {
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
    `;
    document.head.appendChild(style);
}

function clampBalance(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < MIN_BALANCE || numeric > MAX_BALANCE) return 0;
    return Math.max(MIN_BALANCE, Math.min(MAX_BALANCE, Math.round(numeric)));
}

function drawSourceTarget(ctx, _node, width, y, height) {
    const centerY = y + height * 0.5;
    const lineStart = 82;
    const lineEnd = Math.max(lineStart + 40, width - 82);
    const ratio = (clampBalance(this.value) - MIN_BALANCE) / (MAX_BALANCE - MIN_BALANCE);
    const knobX = lineStart + ratio * (lineEnd - lineStart);

    ctx.save();
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.fillStyle = "#ddd";
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillText("SOURCE", 10, centerY);
    ctx.textAlign = "right";
    ctx.fillText("TARGET", width - 10, centerY);

    ctx.beginPath();
    ctx.moveTo(lineStart, centerY);
    ctx.lineTo(lineEnd, centerY);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(190, 198, 210, .62)";
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(knobX, centerY, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#8aa9d6";
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(225, 232, 242, .78)";
    ctx.stroke();
    ctx.restore();
}

export function installSourceTargetSlider(node, widget, visibleName) {
    if (!node || !widget) return null;
    ensureNativeSliderStyle();
    const marker = `_cmkSourceTarget_${visibleName}`;
    if (node[marker]) return node[marker];

    // Existing subgraphs stored an absolute start step (normally 10). The new
    // control stores only the relative SOURCE/TARGET balance, centered at zero.
    widget.value = clampBalance(widget.value);
    widget.type = "slider";
    widget.label = "SOURCE";
    widget.options = {
        ...(widget.options || {}),
        min: MIN_BALANCE,
        max: MAX_BALANCE,
        step: 1,
        precision: 0,
    };
    widget.draw = drawSourceTarget;
    widget.computeSize = (width) => [Math.max(Number(width) || 300, 300), 30];
    widget._cmkSyncFromCanonical = () => {
        widget.value = clampBalance(widget.value);
    };

    node[marker] = widget;
    return widget;
}
