import { app } from "../../../scripts/app.js";

const STORAGE_KEY = "cmk-node-size-defaults-v1";
const MENU_LABEL = "CMK · Node-Dimensionen …";
const MIN_WIDTH = 140;
const MIN_HEIGHT = 60;
const BUILTIN_DEFAULTS = {
    CMKCheckpointVAELoader: [600, 240],
    CMKControlNetPrepare: [540, 420],
    CMKCombinedControlNetPreparePipe: [600, 1225],
    CMKPipeCreateImage: [600, 1060],
    "CMK Flow · 02 SDXL LoRA Stack": [600, 125],
    CMKControlNetPreparePipe: [600, 1225],
    CMKZITControlNetPreparePipe: [600, 1225],
    CMKCheckpointVAELoaderPipe: [600, 165],
    CMKImageLoadAndResizePipe: [600, 860],
    CMKLoadImage: [600, 1225],
    CMKSwapImageLoaderPipe: [1200, 800],
    CMKFamilyResultMergePipe: [300, 250],
    CMKDiagnosticConcat: [320, 135],
    CMK_FaceProcess: [540, 1015],
    CMKFaceSwapImage: [400, 395],
    CMKFaceSwapVideo: [540, 435],
    CMKFaceSwapVideoLoader: [1180, 810],
    CMKLoRATextLoader: [280, 145],
    CMKMergeAndSaveVideo: [600, 780],
    CMK_SmartDetailer: [540, 775],
    CMK_SourcePathInfo: [240, 100],
    CMKSplitVideoIntoSegments: [600, 910],
    CMKVideoCompare: [1180, 600],
};

function text(value) {
    return String(value ?? "");
}

function isCmkNode(node) {
    const candidates = [
        node?.title,
        node?.type,
        node?.comfyClass,
        node?.constructor?.comfyClass,
        node?.constructor?.nodeData?.name,
        node?.constructor?.nodeData?.display_name,
    ];
    return candidates.some((value) => /^CMK(?:\s|_|·|-)/i.test(text(value)));
}

function sizeKey(node) {
    return text(
        node?.comfyClass ||
        node?.constructor?.comfyClass ||
        node?.type ||
        node?.title,
    ).trim();
}

function validSize(value) {
    return (
        Array.isArray(value) &&
        Number.isFinite(Number(value[0])) &&
        Number.isFinite(Number(value[1])) &&
        Number(value[0]) > 0 &&
        Number(value[1]) > 0
    );
}

function loadDefaults() {
    try {
        const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
        return value && typeof value === "object" && !Array.isArray(value) ? value : {};
    } catch (_) {
        return {};
    }
}

function saveDefaults(value) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
}

function minimumSize(node) {
    const declared = node?.constructor?.min_size || node?.min_size || node?.minSize;
    return [
        Math.max(MIN_WIDTH, Number(declared?.[0]) || 0),
        Math.max(MIN_HEIGHT, Number(declared?.[1]) || 0),
    ];
}

function normalizeSize(node, width, height) {
    const minimum = minimumSize(node);
    return [
        Math.max(minimum[0], Math.round(Number(width) || minimum[0])),
        Math.max(minimum[1], Math.round(Number(height) || minimum[1])),
    ];
}

function applySize(node, width, height) {
    const size = normalizeSize(node, width, height);
    node.setSize?.(size);
    node.properties ||= {};
    node.properties.cmkOuterSize = [...size];
    node.properties.cmkManualSize = [...size];
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    return size;
}

function contentSize(node) {
    try {
        const computed = node.computeSize?.();
        if (validSize(computed)) return normalizeSize(node, computed[0], computed[1]);
    } catch (error) {
        console.warn("[CMK] Could not compute node content size", error);
    }
    return normalizeSize(node, node.size?.[0], node.size?.[1]);
}

function make(tag, className, label) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (label != null) element.textContent = label;
    return element;
}

function installStyles() {
    if (document.getElementById("cmk-node-dimensions-style")) return;
    const style = make("style");
    style.id = "cmk-node-dimensions-style";
    style.textContent = `
      .cmk-size-backdrop{position:fixed;inset:0;z-index:100000;display:grid;place-items:center;background:rgba(0,0,0,.55)}
      .cmk-size-dialog{width:min(430px,calc(100vw - 32px));padding:18px;border:1px solid #59616d;border-radius:12px;background:#202226;color:#f1f3f5;box-shadow:0 18px 55px rgba(0,0,0,.55);font:13px system-ui,sans-serif}
      .cmk-size-dialog h2{margin:0 0 4px;font-size:17px}.cmk-size-subtitle{margin:0 0 16px;color:#abb2bd;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
      .cmk-size-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.cmk-size-field{display:grid;gap:5px;color:#c8cdd5}.cmk-size-field input{width:100%;box-sizing:border-box;padding:8px 10px;border:1px solid #59616d;border-radius:7px;background:#15171a;color:#fff;font:inherit}
      .cmk-size-lock{display:flex;align-items:center;gap:8px;margin:13px 0;color:#d7dbe1}.cmk-size-actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}.cmk-size-actions button{padding:7px 10px;border:1px solid #59616d;border-radius:7px;background:#30343a;color:#fff;cursor:pointer}.cmk-size-actions button:hover{background:#3b4149}.cmk-size-actions .primary{margin-left:auto;background:#315f91;border-color:#4a7db2}.cmk-size-note{min-height:18px;margin-top:10px;color:#9fa8b5;font-size:12px}
    `;
    document.head.append(style);
}

function openSizeDialog(node) {
    installStyles();
    const initial = normalizeSize(node, node.size?.[0], node.size?.[1]);
    let ratio = initial[0] / initial[1];

    const backdrop = make("div", "cmk-size-backdrop");
    const dialog = make("div", "cmk-size-dialog");
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    dialog.append(make("h2", "", "Node-Dimensionen"));
    dialog.append(make("div", "cmk-size-subtitle", text(node.title || node.type)));

    const grid = make("div", "cmk-size-grid");
    const widthField = make("label", "cmk-size-field", "BREITE (px)");
    const heightField = make("label", "cmk-size-field", "HÖHE (px)");
    const widthInput = make("input");
    const heightInput = make("input");
    for (const input of [widthInput, heightInput]) {
        input.type = "number";
        input.min = "1";
        input.step = "1";
    }
    widthInput.value = String(initial[0]);
    heightInput.value = String(initial[1]);
    widthField.append(widthInput);
    heightField.append(heightInput);
    grid.append(widthField, heightField);
    dialog.append(grid);

    const lockLabel = make("label", "cmk-size-lock");
    const lockInput = make("input");
    lockInput.type = "checkbox";
    lockInput.checked = false;
    lockLabel.append(lockInput, document.createTextNode("Seitenverhältnis beibehalten"));
    dialog.append(lockLabel);

    let syncing = false;
    widthInput.addEventListener("input", () => {
        if (!lockInput.checked || syncing) return;
        syncing = true;
        heightInput.value = String(Math.max(1, Math.round(Number(widthInput.value) / ratio)));
        syncing = false;
    });
    heightInput.addEventListener("input", () => {
        if (!lockInput.checked || syncing) return;
        syncing = true;
        widthInput.value = String(Math.max(1, Math.round(Number(heightInput.value) * ratio)));
        syncing = false;
    });

    const note = make("div", "cmk-size-note");
    const actions = make("div", "cmk-size-actions");
    const fitButton = make("button", "", "Auf Inhalt anpassen");
    const defaultButton = make("button", "", "Als Typ-Standard speichern");
    const clearButton = make("button", "", "Typ-Standard löschen");
    const cancelButton = make("button", "", "Abbrechen");
    const applyButton = make("button", "primary", "Übernehmen");
    for (const button of [fitButton, defaultButton, clearButton, cancelButton, applyButton]) {
        button.type = "button";
    }
    actions.append(fitButton, defaultButton, clearButton, cancelButton, applyButton);
    dialog.append(actions, note);
    backdrop.append(dialog);
    document.body.append(backdrop);

    const close = () => backdrop.remove();
    const applyFields = () => {
        const size = applySize(node, widthInput.value, heightInput.value);
        widthInput.value = String(size[0]);
        heightInput.value = String(size[1]);
        ratio = size[0] / size[1];
        return size;
    };

    fitButton.addEventListener("click", () => {
        const size = contentSize(node);
        widthInput.value = String(size[0]);
        heightInput.value = String(size[1]);
        ratio = size[0] / size[1];
        applySize(node, size[0], size[1]);
        note.textContent = `Inhalt: ${size[0]} × ${size[1]} px`;
    });
    defaultButton.addEventListener("click", () => {
        const size = applyFields();
        const key = sizeKey(node);
        const defaults = loadDefaults();
        defaults[key] = size;
        saveDefaults(defaults);
        note.textContent = `Standard für ${key} gespeichert.`;
    });
    clearButton.addEventListener("click", () => {
        const key = sizeKey(node);
        const defaults = loadDefaults();
        delete defaults[key];
        saveDefaults(defaults);
        note.textContent = `Standard für ${key} gelöscht.`;
    });
    cancelButton.addEventListener("click", close);
    applyButton.addEventListener("click", () => {
        applyFields();
        close();
    });
    backdrop.addEventListener("pointerdown", (event) => {
        if (event.target === backdrop) close();
    });
    dialog.addEventListener("keydown", (event) => {
        if (event.key === "Escape") close();
        if (event.key === "Enter" && event.target?.tagName === "INPUT") {
            applyFields();
            close();
        }
    });
    setTimeout(() => widthInput.focus(), 0);
}

function addMenuOption(node, options) {
    if (!isCmkNode(node) || !Array.isArray(options)) return;
    if (options.some((option) => option?.content === MENU_LABEL)) return;
    options.unshift({
        content: MENU_LABEL,
        callback: () => openSizeDialog(node),
    });
}

function installMenu(node) {
    if (!isCmkNode(node) || node.__cmkDimensionsMenuInstalled) return;
    node.__cmkDimensionsMenuInstalled = true;
    const original = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function(canvas, options) {
        const result = original?.apply(this, arguments);
        addMenuOption(this, options);
        return result;
    };
}

function applyTypeDefault(node) {
    if (!isCmkNode(node)) return;
    const key = sizeKey(node);
    const saved = (
        loadDefaults()[key] ||
        BUILTIN_DEFAULTS[key] ||
        BUILTIN_DEFAULTS[text(node?.title).trim()]
    );
    if (!validSize(saved)) return;
    setTimeout(() => {
        if (node.__cmkLoadedFromWorkflow) return;
        applySize(node, saved[0], saved[1]);
    }, 0);
}

app.registerExtension({
    name: "cmk.node.dimensions.v1",

    beforeRegisterNodeDef(nodeType) {
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            this.__cmkLoadedFromWorkflow = true;
            return originalConfigure?.apply(this, arguments);
        };

        const original = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
            const result = original?.apply(this, arguments);
            addMenuOption(this, options);
            return result;
        };
    },

    nodeCreated(node) {
        installMenu(node);
        applyTypeDefault(node);
    },
});
