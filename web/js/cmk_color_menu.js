import { app } from "../../../scripts/app.js";
import {
    CMK_COLOR_PALETTE,
    CMK_COLOR_ROLE_PROPERTY,
} from "./cmk_subgraph_colors.js";

const MENU_LABEL = "CMK · Color";
const HEX_COLOR_PATTERN = /^#[0-9A-F]{6}$/i;
const PRESETS = Object.freeze([
    Object.freeze({ label: "ControlNet · Petrol", role: "CONTROLNET" }),
    Object.freeze({ label: "Identity · Slate Blue", role: "IDENTITY" }),
    Object.freeze({ label: "Detailer · Umbra", role: "DETAILER" }),
    Object.freeze({ label: "Output · Green", role: "OUTPUT" }),
]);

function make(tag, className, label) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (label != null) element.textContent = label;
    return element;
}

function normalizeHex(value) {
    const candidate = String(value ?? "").trim().toUpperCase();
    return HEX_COLOR_PATTERN.test(candidate) ? candidate : null;
}

function isCentrallyManaged(node) {
    return Boolean(CMK_COLOR_PALETTE[node?.properties?.[CMK_COLOR_ROLE_PROPERTY]]);
}

function markChanged(target, update) {
    const graph = target?.graph || app.canvas?.graph || app.graph;
    graph?.beforeChange?.();
    try {
        update();
    } finally {
        graph?.afterChange?.();
    }
    target?.setDirtyCanvas?.(true, true);
    graph?.setDirtyCanvas?.(true, true);
}

function applyPreset(target, role) {
    const colors = CMK_COLOR_PALETTE[role];
    if (!colors || isCentrallyManaged(target)) return;
    markChanged(target, () => {
        target.color = colors.title;
        target.bgcolor = colors.background;
    });
}

function applyCustomColor(target, value) {
    const color = normalizeHex(value);
    if (!color || isCentrallyManaged(target)) return false;
    markChanged(target, () => {
        target.color = color;
    });
    return true;
}

function installStyles() {
    if (document.getElementById("cmk-color-dialog-style")) return;
    const style = make("style");
    style.id = "cmk-color-dialog-style";
    style.textContent = `
      .cmk-color-backdrop{position:fixed;inset:0;z-index:100000;display:grid;place-items:center;background:rgba(0,0,0,.55)}
      .cmk-color-dialog{width:min(390px,calc(100vw - 32px));padding:18px;border:1px solid #59616d;border-radius:12px;background:#202226;color:#f1f3f5;box-shadow:0 18px 55px rgba(0,0,0,.55);font:13px system-ui,sans-serif}
      .cmk-color-dialog h2{margin:0 0 16px;font-size:17px}.cmk-color-field{display:grid;gap:6px;color:#c8cdd5}.cmk-color-row{display:grid;grid-template-columns:1fr 92px;gap:12px;align-items:stretch}.cmk-color-hex{width:100%;box-sizing:border-box;padding:8px 10px;border:1px solid #59616d;border-radius:7px;background:#15171a;color:#fff;font:inherit;text-transform:uppercase}.cmk-color-hex[aria-invalid="true"]{border-color:#d86464}.cmk-color-preview{display:grid;place-items:center;min-height:36px;border:1px solid #707784;border-radius:7px;color:#fff;text-shadow:0 1px 2px #000;background-image:linear-gradient(45deg,#777 25%,transparent 25%),linear-gradient(-45deg,#777 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#777 75%),linear-gradient(-45deg,transparent 75%,#777 75%);background-size:12px 12px;background-position:0 0,0 6px,6px -6px,-6px 0}
      .cmk-color-picker-row{display:flex;align-items:center;gap:10px;margin-top:14px;color:#c8cdd5}.cmk-color-picker{width:56px;height:34px;padding:2px;border:1px solid #59616d;border-radius:7px;background:#30343a;cursor:pointer}.cmk-color-note{min-height:18px;margin-top:8px;color:#e28a8a;font-size:12px}.cmk-color-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:12px}.cmk-color-actions button{padding:7px 12px;border:1px solid #59616d;border-radius:7px;background:#30343a;color:#fff;cursor:pointer}.cmk-color-actions button:hover{background:#3b4149}.cmk-color-actions .primary{background:#315f91;border-color:#4a7db2}.cmk-color-actions button:disabled{opacity:.45;cursor:not-allowed}
    `;
    document.head.append(style);
}

function openCustomColorDialog(target) {
    if (isCentrallyManaged(target)) return;
    installStyles();

    const fallback = CMK_COLOR_PALETTE.CONTROLNET.title;
    const initial = normalizeHex(target?.color) || fallback;
    const backdrop = make("div", "cmk-color-backdrop");
    const dialog = make("div", "cmk-color-dialog");
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    dialog.setAttribute("aria-labelledby", "cmk-color-dialog-title");

    const title = make("h2", "", "CMK · Custom Color");
    title.id = "cmk-color-dialog-title";
    const field = make("label", "cmk-color-field", "HEX Color");
    const row = make("div", "cmk-color-row");
    const hexInput = make("input", "cmk-color-hex");
    hexInput.type = "text";
    hexInput.value = initial;
    hexInput.maxLength = 7;
    hexInput.spellcheck = false;
    hexInput.autocomplete = "off";
    hexInput.inputMode = "text";
    const preview = make("div", "cmk-color-preview", "Preview");
    row.append(hexInput, preview);
    field.append(row);

    const pickerRow = make("label", "cmk-color-picker-row", "Color Picker");
    const picker = make("input", "cmk-color-picker");
    picker.type = "color";
    picker.value = initial;
    pickerRow.append(picker);

    const note = make("div", "cmk-color-note");
    note.setAttribute("aria-live", "polite");
    const actions = make("div", "cmk-color-actions");
    const cancelButton = make("button", "", "Cancel");
    const applyButton = make("button", "primary", "Apply");
    cancelButton.type = "button";
    applyButton.type = "button";
    actions.append(cancelButton, applyButton);
    dialog.append(title, field, pickerRow, note, actions);
    backdrop.append(dialog);
    document.body.append(backdrop);

    const render = () => {
        const color = normalizeHex(hexInput.value);
        const valid = Boolean(color);
        hexInput.setAttribute("aria-invalid", String(!valid));
        applyButton.disabled = !valid;
        note.textContent = valid ? "" : "Use a six-digit HEX color such as #A1B2C3.";
        if (valid) {
            preview.style.background = color;
            picker.value = color;
        }
        return color;
    };
    const close = () => backdrop.remove();

    hexInput.addEventListener("input", render);
    hexInput.addEventListener("blur", () => {
        const color = normalizeHex(hexInput.value);
        if (color) hexInput.value = color;
        render();
    });
    picker.addEventListener("input", () => {
        hexInput.value = picker.value.toUpperCase();
        render();
    });
    cancelButton.addEventListener("click", close);
    applyButton.addEventListener("click", () => {
        const color = render();
        if (!color || !applyCustomColor(target, color)) return;
        hexInput.value = color;
        close();
    });
    backdrop.addEventListener("pointerdown", (event) => {
        if (event.target === backdrop) close();
    });
    dialog.addEventListener("keydown", (event) => {
        if (event.key === "Escape") close();
        if (event.key === "Enter" && !applyButton.disabled) {
            event.preventDefault();
            applyButton.click();
        }
    });

    render();
    hexInput.focus();
    hexInput.select();
}

function createColorMenu(target) {
    return {
        content: MENU_LABEL,
        has_submenu: true,
        submenu: {
            options: [
                ...PRESETS.map(({ label, role }) => ({
                    content: label,
                    callback: () => applyPreset(target, role),
                })),
                null,
                {
                    content: "Custom Color...",
                    callback: () => openCustomColorDialog(target),
                },
            ],
        },
    };
}

app.registerExtension({
    name: "cmk.color.menu.v1",

    getNodeMenuItems(node) {
        return isCentrallyManaged(node) ? [] : [createColorMenu(node)];
    },
});
