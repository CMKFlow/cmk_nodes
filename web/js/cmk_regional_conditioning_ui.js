import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKRegionalConditioningSDXL";
const NODE_SELECTOR = '[data-node-id], [data-testid^="node-body-"]';
const LABEL_SELECTOR = 'label, [data-testid="widget-layout-field-label"]';
const POSITIONS = ["TOP LEFT", "TOP", "TOP RIGHT", "LEFT", "CENTER", "RIGHT", "BOTTOM LEFT", "BOTTOM", "BOTTOM RIGHT"];
const EXTENTS = { SMALL: 0.33, MEDIUM: 0.50, LARGE: 0.67 };
const DEFAULT_NODE_WIDTH = 600;
const DEFAULT_NODE_HEIGHT = 1225;

function isTarget(node) {
  return Boolean(node) && (
    node.comfyClass === NODE_CLASS || node.type === NODE_CLASS ||
    node.constructor?.comfyClass === NODE_CLASS || node.constructor?.nodeData?.name === NODE_CLASS
  );
}

function geometry(position, size) {
  const extent = EXTENTS[String(size || "MEDIUM")] ?? EXTENTS.MEDIUM;
  const horizontal = String(position).includes("LEFT") ? "LEFT" : String(position).includes("RIGHT") ? "RIGHT" : "CENTER";
  const vertical = String(position).startsWith("TOP") ? "TOP" : String(position).startsWith("BOTTOM") ? "BOTTOM" : "CENTER";
  const x = horizontal === "LEFT" ? 0 : horizontal === "RIGHT" ? 1 - extent : (1 - extent) / 2;
  const y = vertical === "TOP" ? 0 : vertical === "BOTTOM" ? 1 - extent : (1 - extent) / 2;
  return [x, y, extent, extent];
}

function setStructuralVisibility(widget, visible) {
  if (!widget) return;
  widget.options ??= {};
  widget._cmkRegionalLayout ??= {
    type: widget.type,
    computeSize: widget.computeSize,
    draw: widget.draw,
    hidden: widget.hidden,
    optionsHidden: widget.options.hidden,
  };
  const original = widget._cmkRegionalLayout;
  if (visible) {
    widget.type = original.type;
    widget.computeSize = original.computeSize;
    widget.draw = original.draw;
    widget.hidden = original.hidden;
    widget.options.hidden = original.optionsHidden;
  } else {
    widget.type = "converted-widget";
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    widget.hidden = true;
    widget.options.hidden = true;
  }
}

function setPromptHeight(widget, height) {
  if (!widget) return;
  widget._cmkRegionalPromptComputeSize ??= widget.computeSize;
  widget.computeSize = (width) => [Math.max(Number(width) || 560, 560), height];
  // Multiline STRING inputs are DOM widgets in current ComfyUI frontends.
  // Their layout ignores computeSize and distributes space from these options.
  widget.options ??= {};
  widget.options.getMinHeight = () => height;
  delete widget.options.getMaxHeight;
  delete widget.options.getHeight;
  widget.options.rows = Math.max(2, Math.round((height - 24) / 14));
  widget.computedHeight = undefined;
  for (const element of [widget.element, widget.inputEl]) {
    if (!element?.style) continue;
    element.style.setProperty("--comfy-widget-min-height", `${height}px`);
    element.style.removeProperty("--comfy-widget-max-height");
    element.style.removeProperty("--comfy-widget-height");
    element.style.height = "100%";
    element.style.minHeight = `${height}px`;
    element.setAttribute?.("rows", String(widget.options.rows));
  }
}

function nodeForElement(element) {
  const nodeElement = element?.closest?.(NODE_SELECTOR);
  const testId = nodeElement?.getAttribute?.("data-testid") || "";
  const id = nodeElement?.dataset?.nodeId ?? testId.match(/^node-body-(.+)$/)?.[1];
  if (id == null) return null;
  for (const graph of [app.canvas?.graph, app.graph, app.rootGraph]) {
    for (const candidate of [id, Number(id)]) {
      if (candidate === "" || Number.isNaN(candidate)) continue;
      const node = graph?.getNodeById?.(candidate);
      if (node) return node;
    }
  }
  return null;
}

function decorateNodeElement(nodeElement) {
  const node = nodeForElement(nodeElement);
  if (!isTarget(node)) return;
  const advanced = node.properties?.cmkRegionalMode === "ADVANCED";
  const standardOnly = new Set(["POSITION", "SIZE"]);
  const advancedOnly = new Set(["X", "Y", "WIDTH", "HEIGHT", "END"]);
  for (const label of nodeElement.querySelectorAll(LABEL_SELECTOR)) {
    const text = label.textContent?.trim();
    const visible = standardOnly.has(text) ? !advanced : advancedOnly.has(text) ? advanced : true;
    const row = label.parentElement;
    if (!row) continue;
    row.style.display = visible ? "" : "none";
    if (text === "AREA PROMPT") {
      const promptHeight = advanced ? 72 : 96;
      row.style.height = "auto";
      row.style.minHeight = `${promptHeight}px`;
      const textarea = row.querySelector("textarea");
      if (textarea) {
        textarea.style.height = "100%";
        textarea.style.minHeight = `${promptHeight - 8}px`;
        textarea.style.resize = "none";
      }
    }
  }
}

function scan(root = document) {
  if (root instanceof Element) {
    const owner = root.matches(NODE_SELECTOR) ? root : root.closest(NODE_SELECTOR);
    if (owner) decorateNodeElement(owner);
  }
  for (const nodeElement of root.querySelectorAll?.(NODE_SELECTOR) || []) decorateNodeElement(nodeElement);
}

function installDomBehavior() {
  if (window._cmkRegionalConditioningDomInstalled) return;
  window._cmkRegionalConditioningDomInstalled = true;
  const observer = new MutationObserver((records) => {
    for (const record of records) for (const added of record.addedNodes) {
      if (added instanceof Element) scan(added);
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
  scan();
}

function makeBand(text, line = false) {
  const root = document.createElement("div");
  root.style.width = "100%";
  root.style.boxSizing = "border-box";
  root.style.height = line ? "17px" : "30px";
  root.style.display = "flex";
  root.style.alignItems = "center";
  root.style.color = line ? "transparent" : "#e5e5e5";
  root.style.fontSize = "12px";
  root.style.fontWeight = "700";
  root.style.letterSpacing = ".08em";
  if (line) root.style.borderTop = "1px solid rgba(255,255,255,.16)";
  root.textContent = text;
  return root;
}

function installStableSerialization(node, canonical) {
  if (node._cmkRegionalSerializationInstalled) return;
  const original = node.onSerialize;
  node.onSerialize = function(data) {
    const result = original?.apply(this, arguments);
    if (this.serialize_widgets) {
      data.widgets_values = canonical
        .filter((widget) => widget?.serialize !== false)
        .map((widget) => widget?.value ?? null);
    }
    return result;
  };
  node._cmkRegionalSerializationInstalled = true;
}

function configure(node, applyDefaultSize = false) {
  if (!isTarget(node) || !Array.isArray(node.widgets) || typeof node.addDOMWidget !== "function") return;
  if (node._cmkRegionalUiInstalled) {
    node._cmkRegionalRebuild?.(applyDefaultSize);
    return;
  }
  node._cmkRegionalUiInstalled = true;
  node.properties ??= {};
  node.properties.cmkRegionalMode ??= "STANDARD";
  const canonical = [...node.widgets];
  const byName = new Map(canonical.filter((widget) => widget?.name).map((widget) => [widget.name, widget]));
  installStableSerialization(node, canonical);

  const tabsRoot = document.createElement("div");
  tabsRoot.style.display = "grid"; tabsRoot.style.gridTemplateColumns = "1fr 1fr";
  tabsRoot.style.gap = "6px"; tabsRoot.style.width = "100%";
  const tabButtons = ["STANDARD", "ADVANCED"].map((mode) => {
    const button = document.createElement("button");
    button.type = "button"; button.textContent = mode; button.style.height = "30px";
    button.style.borderRadius = "5px"; button.style.cursor = "pointer";
    button.style.font = "600 12px system-ui"; tabsRoot.append(button); return button;
  });
  const tabs = node.addDOMWidget("CMK REGION MODE", "cmk_region_tabs", tabsRoot, {
    hideOnZoom: false, getMinHeight: () => 36, getHeight: () => 36,
  });
  tabs.serialize = false;

  const headers = [];
  const separators = [];
  for (let index = 1; index <= 3; index++) {
    const header = node.addDOMWidget(`CMK REGION ${index}`, "cmk_region_header", makeBand(`REGION ${index}`), {
      hideOnZoom: false, getMinHeight: () => 30, getHeight: () => 30,
    });
    header.serialize = false; headers.push(header);
    if (index < 3) {
      const separator = node.addDOMWidget(`CMK REGION SEPARATOR ${index}`, "cmk_region_separator", makeBand("", true), {
        hideOnZoom: false, getMinHeight: () => 17, getHeight: () => 17,
      });
      separator.serialize = false; separators.push(separator);
    }
  }

  const syncPreset = (index) => {
    const position = byName.get(`region_${index}_position`);
    const size = byName.get(`region_${index}_size`);
    if (!position || position.value === "CUSTOM") return;
    const values = geometry(position.value, size?.value);
    ["x", "y", "width", "height"].forEach((name, offset) => {
      const widget = byName.get(`region_${index}_${name}`);
      if (widget) widget.value = values[offset];
    });
  };
  for (let index = 1; index <= 3; index++) {
    for (const name of ["position", "size"]) {
      const widget = byName.get(`region_${index}_${name}`);
      const callback = widget?.callback;
      if (widget) widget.callback = function() {
        const result = callback?.apply(this, arguments); syncPreset(index); return result;
      };
    }
    for (const name of ["x", "y", "width", "height"]) {
      const widget = byName.get(`region_${index}_${name}`);
      const callback = widget?.callback;
      if (widget) widget.callback = function() {
        const result = callback?.apply(this, arguments);
        const position = byName.get(`region_${index}_position`);
        if (node.properties.cmkRegionalMode === "ADVANCED" && position) position.value = "CUSTOM";
        return result;
      };
    }
  }

  const rebuild = (applyDefaultSize = false) => {
    const advanced = node.properties.cmkRegionalMode === "ADVANCED";
    tabButtons.forEach((button, index) => {
      const active = advanced === (index === 1);
      button.style.border = active ? "1px solid #61cddd" : "1px solid rgba(255,255,255,.16)";
      button.style.background = active ? "#244a52" : "#25292c";
      button.style.color = active ? "#fff" : "#aeb4b8";
    });
    for (let index = 1; index <= 3; index++) {
      for (const name of ["position", "size"]) {
        setStructuralVisibility(byName.get(`region_${index}_${name}`), !advanced);
      }
      for (const name of ["x", "y", "width", "height", "end"]) {
        setStructuralVisibility(byName.get(`region_${index}_${name}`), advanced);
      }
      setStructuralVisibility(byName.get(`region_${index}_area_prompt`), true);
      setStructuralVisibility(byName.get(`region_${index}_strength`), true);
      setPromptHeight(byName.get(`region_${index}_area_prompt`), advanced ? 72 : 96);
    }
    const visibleNames = advanced
      ? ["x", "y", "width", "height", "end", "area_prompt", "strength"]
      : ["position", "size", "area_prompt", "strength"];
    const visible = [tabs];
    const visibleWidgets = new Set();
    for (let index = 1; index <= 3; index++) {
      visible.push(headers[index - 1]);
      for (const name of visibleNames) {
        const widget = byName.get(`region_${index}_${name}`);
        if (widget) { visible.push(widget); visibleWidgets.add(widget); }
      }
      if (index < 3) visible.push(separators[index - 1]);
    }
    // Keep inactive values available to graph-to-prompt and workflow
    // serialization, but move them behind the complete visible layout. Their
    // zero-height rows can no longer create gaps between visible controls.
    const inactive = canonical.filter((widget) => !visibleWidgets.has(widget));
    node.widgets = [...visible, ...inactive];
    // Let LiteGraph derive the node height from the currently visible widget
    // set. This removes both the inactive Vue rows and the fixed bottom area.
    if (applyDefaultSize && !node._cmkRegionalLoadedFromWorkflow) {
      node.setSize?.([DEFAULT_NODE_WIDTH, DEFAULT_NODE_HEIGHT]);
    }
    node.graph?.trigger?.("node:widget:changed", { nodeId: node.id });
    node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true);
    for (const delay of [0, 40, 120]) setTimeout(() => scan(), delay);
  };
  node._cmkRegionalRebuild = rebuild;
  tabButtons.forEach((button, index) => button.addEventListener("click", () => {
    node.properties.cmkRegionalMode = index === 1 ? "ADVANCED" : "STANDARD";
    if (index === 1) for (let region = 1; region <= 3; region++) syncPreset(region);
    // A mode switch only rearranges the contents. Keep the user's current
    // node height; flexible prompt rows absorb the available space.
    rebuild(false);
  }));

  for (const widget of canonical) {
    const name = widget?.name || "";
    if (name.includes("_position")) widget.label = "POSITION";
    else if (name.includes("_size")) widget.label = "SIZE";
    else if (name.includes("_width")) widget.label = "WIDTH";
    else if (name.includes("_height")) widget.label = "HEIGHT";
    else if (name.includes("_strength")) widget.label = "STRENGTH";
    else if (name.includes("_area_prompt")) widget.label = "AREA PROMPT";
    else if (name.includes("_end")) widget.label = "END";
    else if (name.endsWith("_x")) widget.label = "X";
    else if (name.endsWith("_y")) widget.label = "Y";
  }
  for (let index = 1; index <= 3; index++) syncPreset(index);
  rebuild(applyDefaultSize);
}

function schedule(node, applyDefaultSize = false) {
  for (const delay of [0, 50, 200]) {
    setTimeout(() => configure(node, delay === 0 && applyDefaultSize), delay);
  }
}

app.registerExtension({
  name: import.meta.url.includes("cmk-layout-v5")
    ? "cmk.regional.conditioning.ui.v5"
    : "cmk.regional.conditioning.ui.v1",
  setup() { installDomBehavior(); },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_CLASS) return;
    for (const hook of ["onNodeCreated", "onConfigure", "onAdded"]) {
      const original = nodeType.prototype[hook];
      nodeType.prototype[hook] = function() {
        if (hook === "onConfigure") this._cmkRegionalLoadedFromWorkflow = true;
        const result = original?.apply(this, arguments);
        schedule(this, hook === "onNodeCreated");
        return result;
      };
    }
  },
  nodeCreated(node) { if (isTarget(node)) schedule(node, true); },
  loadedGraphNode(node) {
    if (!isTarget(node)) return;
    node._cmkRegionalLoadedFromWorkflow = true;
    schedule(node, false);
  },
});
