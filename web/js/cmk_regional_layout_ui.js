import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {
  automaticPreviewSignature, clamp, cloneLayout, commitHistory, createHistory,
  createLatestRequestGuard, hitRegionIndices, makeRegion, moveBox,
  normalizeBackgroundSource, normalizeLayout, prepareImageGeometry,
  packagedReferenceAssetPath, processDimensionsFromWidgets, redoHistory,
  regionalCanvasGeometry, resizeBox, serializeLayout, undoHistory,
  resolveAutomaticPreviewSource, upstreamStartNode,
} from "./cmk_regional_layout_core.js?cmk-regional-layout-v5";

const NODE_CLASS = "CMKRegionalLayoutSDXL";
const DISPLAY_NAME = "CMK Flow · 07 Regional Layout";
const COLORS = ["#61cddd", "#f3a65a", "#d582f0", "#70d884", "#f06f7e", "#8ca9ff", "#e1d45b", "#54d5b0"];
const PANEL_HEIGHT = 802;

function target(node) {
  return node?.comfyClass === NODE_CLASS || node?.type === NODE_CLASS ||
    node?.constructor?.comfyClass === NODE_CLASS || node?.constructor?.nodeData?.name === NODE_CLASS ||
    node?.title === DISPLAY_NAME;
}

function widgetValue(node, name, fallback) {
  return node?.widgets?.find((widget) => widget?.name === name)?.value ?? fallback;
}

function imageReferenceParts(reference) {
  if (reference && typeof reference === "object") {
    return {
      filename: String(reference.filename || ""),
      subfolder: String(reference.subfolder || ""),
      type: String(reference.type || "input"),
    };
  }
  const match = String(reference || "").trim().match(/^(.*?)(?:\s+\[(input|output|temp)\])?$/i);
  const path = String(match?.[1] || "").replace(/^[/\\]+/, "");
  const slash = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return {
    filename: slash >= 0 ? path.slice(slash + 1) : path,
    subfolder: slash >= 0 ? path.slice(0, slash).replaceAll("\\", "/") : "",
    type: String(match?.[2] || "input").toLowerCase(),
  };
}

function automaticImageUrl(reference, revision = 0) {
  const value = typeof reference === "string" ? reference.trim() : "";
  const packaged = packagedReferenceAssetPath(value);
  if (packaged) return api.apiURL(packaged);
  const parts = imageReferenceParts(reference);
  if (!parts.filename) return null;
  const params = new URLSearchParams({ filename: parts.filename, type: parts.type, channel: "rgb" });
  if (parts.subfolder) params.set("subfolder", parts.subfolder);
  if (revision) params.set("cmk_preview_revision", String(revision));
  return api.apiURL(`/view?${params.toString()}`);
}

function labelProcessInput(node) {
  const input = node?.inputs?.find((slot) => slot?.name === "PROCESS");
  if (input) input.localized_name = "PROCESS SDXL";
}

function button(label, title = "") {
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = label;
  element.title = title;
  Object.assign(element.style, {
    minHeight: "28px", border: "1px solid rgba(255,255,255,.16)", borderRadius: "5px",
    background: "#25292c", color: "#e5e5e5", cursor: "pointer", font: "600 11px system-ui",
  });
  return element;
}

function field(label, input) {
  const root = document.createElement("label");
  root.style.display = "grid";
  root.style.gap = "4px";
  root.style.color = "#b9c0c4";
  root.style.font = "600 11px system-ui";
  const caption = document.createElement("span");
  caption.textContent = label;
  root.append(caption, input);
  return root;
}

function input(type = "text") {
  const element = document.createElement("input");
  element.type = type;
  Object.assign(element.style, {
    width: "100%", boxSizing: "border-box", minHeight: "30px", border: "1px solid rgba(255,255,255,.15)",
    borderRadius: "4px", background: "#171a1c", color: "#eee", padding: "5px 7px",
  });
  return element;
}

function hideWidget(widget) {
  if (!widget) return;
  widget.options ??= {};
  widget.options.hidden = true;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
  widget.draw = () => {};
}

function watchPreviewWidget(owner, widget) {
  if (!owner || !widget || widget.__cmkRegionalPreviewWatcher) return;
  const original = widget.callback;
  widget.callback = function() {
    owner.__cmkRegionalPreviewRevision = Number(owner.__cmkRegionalPreviewRevision || 0) + 1;
    return original?.apply(this, arguments);
  };
  widget.__cmkRegionalPreviewWatcher = true;
}

function editor(node) {
  if (node._cmkRegionalLayoutEditor || typeof node.addDOMWidget !== "function") return;
  const canonical = [...(node.widgets || [])];
  const byName = new Map(canonical.map((widget) => [widget?.name, widget]));
  const widthWidget = byName.get("width");
  const heightWidget = byName.get("height");
  const dataWidget = byName.get("regions_json");
  if (!widthWidget || !heightWidget || !dataWidget) return;
  node._cmkRegionalLayoutEditor = true;
  for (const widget of [widthWidget, heightWidget, dataWidget]) hideWidget(widget);
  labelProcessInput(node);

  node.properties ??= {};
  let history = createHistory(normalizeLayout(dataWidget.value, widthWidget.value, heightWidget.value));
  let selectedId = history.present.regions.some((region) => region.id === node.properties.cmkRegionalSelectedId)
    ? node.properties.cmkRegionalSelectedId
    : history.present.regions[0]?.id || null;
  let drag = null;
  let background = null;
  let automaticBackground = null;
  let automaticGeometry = null;
  let automaticSignature = "";
  const automaticRequests = createLatestRequestGuard();
  let editingLocked = false;
  let lastHits = "";
  let hitCycle = 0;

  const root = document.createElement("div");
  Object.assign(root.style, { width: "100%", height: `${PANEL_HEIGHT}px`, boxSizing: "border-box", padding: "8px", color: "#e5e5e5", font: "12px system-ui", overflow: "hidden" });
  const processSize = document.createElement("div");
  Object.assign(processSize.style, { minHeight: "26px", display: "grid", placeItems: "center", marginBottom: "7px", border: "1px solid rgba(97,205,221,.32)", borderRadius: "5px", background: "rgba(97,205,221,.08)", color: "#bdeff4", font: "700 11px system-ui", letterSpacing: ".04em" });
  processSize.textContent = "PROCESS SIZE · waiting for 01 START HERE";
  const toolbar = document.createElement("div");
  Object.assign(toolbar.style, { display: "grid", gridTemplateColumns: "repeat(9, minmax(0, 1fr))", gap: "5px", marginBottom: "7px" });
  const add = button("＋", "Add region");
  const duplicate = button("⧉", "Duplicate selected region");
  const remove = button("−", "Delete selected region");
  const undo = button("↶", "Undo");
  const redo = button("↷", "Redo");
  const up = button("↑", "Move region earlier");
  const down = button("↓", "Move region later");
  const reference = button("BG", "Choose reference background");
  const backgroundSource = document.createElement("select");
  for (const [value, label] of [["manual", "Manual"], ["automatic", "Automatic"]]) {
    const option = document.createElement("option"); option.value = value; option.textContent = label; backgroundSource.append(option);
  }
  Object.assign(backgroundSource.style, { minWidth: "0", border: "1px solid rgba(255,255,255,.16)", borderRadius: "5px", background: "#25292c", color: "#e5e5e5", font: "600 10px system-ui" });
  backgroundSource.value = normalizeBackgroundSource(node.properties.cmkRegionalBackgroundSource);
  toolbar.append(add, duplicate, remove, undo, redo, up, down, reference, backgroundSource);

  const stage = document.createElement("div");
  Object.assign(stage.style, { position: "relative", width: "100%", background: "#111416", border: "1px solid rgba(255,255,255,.16)", borderRadius: "5px", overflow: "hidden", boxSizing: "border-box" });
  const canvas = document.createElement("canvas");
  Object.assign(canvas.style, { width: "100%", height: "100%", display: "block", touchAction: "none", cursor: "crosshair" });
  const backgroundHint = document.createElement("div");
  Object.assign(backgroundHint.style, { position: "absolute", inset: "0", display: "none", placeItems: "center", padding: "18px", textAlign: "center", color: "#f0c879", background: "rgba(10,12,14,.5)", font: "600 12px system-ui", pointerEvents: "none", zIndex: "1" });
  stage.append(canvas, backgroundHint);

  const selection = document.createElement("select");
  Object.assign(selection.style, { width: "100%", marginTop: "7px", minHeight: "30px", borderRadius: "4px", background: "#171a1c", color: "#eee", border: "1px solid rgba(255,255,255,.15)" });
  const name = input();
  const prompt = document.createElement("textarea");
  Object.assign(prompt.style, { width: "100%", boxSizing: "border-box", minHeight: "164px", resize: "none", border: "1px solid rgba(255,255,255,.15)", borderRadius: "4px", background: "#171a1c", color: "#eee", padding: "7px" });
  const strength = input("number"); strength.min = "0"; strength.max = "2"; strength.step = "0.05";
  const strengthOffset = input("range"); strengthOffset.min = "0"; strengthOffset.max = "2"; strengthOffset.step = "0.05";
  const details = document.createElement("details");
  details.style.marginTop = "7px";
  const summary = document.createElement("summary"); summary.textContent = "ADVANCED"; summary.style.cursor = "pointer";
  const strengthOffsetField = field("Strength Offset · +0.00", strengthOffset);
  strengthOffsetField.style.marginTop = "6px";
  const strengthOffsetCaption = strengthOffsetField.querySelector("span");
  const geometry = document.createElement("div");
  Object.assign(geometry.style, { display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "6px", marginTop: "6px" });
  const detailInputs = {};
  for (const key of ["x", "y", "width", "height", "end"]) {
    const element = input("number"); element.min = key === "width" || key === "height" ? "0.01" : "0"; element.max = "1"; element.step = "0.01";
    detailInputs[key] = element; geometry.append(field(key.toUpperCase(), element));
  }
  details.append(summary, strengthOffsetField, geometry);
  const editGrid = document.createElement("div");
  Object.assign(editGrid.style, { display: "grid", gridTemplateColumns: "2fr 1fr", gap: "7px", marginTop: "7px" });
  editGrid.append(field("NAME", name), field("STRENGTH", strength));
  root.append(processSize, toolbar, stage, selection, editGrid, field("POSITIVE PROMPT", prompt), details);

  const panel = node.addDOMWidget("CMK REGIONAL LAYOUT", "cmk_regional_layout", root, {
    serialize: false, hideOnZoom: false, getMinHeight: () => PANEL_HEIGHT, getHeight: () => PANEL_HEIGHT, getMaxHeight: () => PANEL_HEIGHT,
  });
  panel.serialize = false;
  panel.computeSize = (width) => [Math.max(520, Number(width) || 520), PANEL_HEIGHT];
  panel.computeLayoutSize = undefined;

  const layout = () => history.present;
  const selectedIndex = () => layout().regions.findIndex((region) => region.id === selectedId);
  const selected = () => layout().regions[selectedIndex()] || null;

  function save() {
    node.properties.cmkRegionalSelectedId = selectedId;
    dataWidget.value = serializeLayout(history.present);
    dataWidget.callback?.(dataWidget.value, app.canvas, node);
    node.graph?.trigger?.("node:widget:changed", { nodeId: node.id });
    node.setDirtyCanvas?.(true, true);
  }

  function commit(mutator) {
    if (editingLocked) return;
    const next = cloneLayout(layout());
    mutator(next);
    history = commitHistory(history, next);
    save(); render();
  }

  function setPresent(next) {
    if (editingLocked) return;
    history = { ...history, present: cloneLayout(next) };
    save(); render();
  }

  function resizeCanvas() {
    const geometry = regionalCanvasGeometry(root.clientWidth, layout().width, layout().height);
    // ComfyUI unmounts DOM widgets while another graph/subgraph is active.
    // Keep the last usable dimensions instead of persisting the transient 0px state.
    if (!geometry) return false;
    stage.style.width = `${geometry.width}px`;
    stage.style.height = `${geometry.height}px`;
    stage.style.marginInline = "auto";
    const rect = stage.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    renderCanvas();
    return true;
  }

  function renderCanvas() {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.width / dpr;
    const height = canvas.height / dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#111416"; ctx.fillRect(0, 0, width, height);
    const sourceMode = normalizeBackgroundSource(node.properties.cmkRegionalBackgroundSource);
    if (!editingLocked && sourceMode === "manual" && background?.complete) {
      const scale = Math.min(width / background.naturalWidth, height / background.naturalHeight);
      const dw = background.naturalWidth * scale, dh = background.naturalHeight * scale;
      ctx.globalAlpha = 0.55; ctx.drawImage(background, (width - dw) / 2, (height - dh) / 2, dw, dh); ctx.globalAlpha = 1;
    } else if (!editingLocked && sourceMode === "automatic" && automaticBackground?.complete && automaticGeometry) {
      const sx = width / automaticGeometry.targetWidth;
      const sy = height / automaticGeometry.targetHeight;
      ctx.globalAlpha = 0.55;
      ctx.drawImage(
        automaticBackground,
        automaticGeometry.x * sx, automaticGeometry.y * sy,
        automaticGeometry.width * sx, automaticGeometry.height * sy,
      );
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = "rgba(255,255,255,.08)"; ctx.lineWidth = 1;
    for (let part = 1; part < 4; part++) {
      ctx.beginPath(); ctx.moveTo(width * part / 4, 0); ctx.lineTo(width * part / 4, height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, height * part / 4); ctx.lineTo(width, height * part / 4); ctx.stroke();
    }
    layout().regions.forEach((region, index) => {
      const box = region.box; const color = COLORS[index % COLORS.length];
      const x = box.x * width, y = box.y * height, w = box.width * width, h = box.height * height;
      ctx.fillStyle = `${color}2b`; ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = color; ctx.lineWidth = region.id === selectedId ? 3 : 1.5; ctx.strokeRect(x, y, w, h);
      const label = `${index + 1} · ${String(region.name || "Region").slice(0, 22)}`;
      ctx.font = "600 12px system-ui"; const labelWidth = Math.min(w, ctx.measureText(label).width + 12);
      ctx.fillStyle = color; ctx.fillRect(x, y, labelWidth, 22);
      ctx.fillStyle = "#101315"; ctx.fillText(label, x + 6, y + 15, Math.max(0, labelWidth - 8));
      if (region.id === selectedId) {
        ctx.fillStyle = "#fff";
        for (const [hx, hy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) ctx.fillRect(hx - 4, hy - 4, 8, 8);
      }
    });
  }

  function render() {
    if (selectedIndex() < 0) selectedId = layout().regions[0]?.id || null;
    node.properties.cmkRegionalSelectedId = selectedId;
    selection.replaceChildren();
    layout().regions.forEach((region, index) => {
      const option = document.createElement("option"); option.value = region.id; option.textContent = `${index + 1} · ${region.name || `Region ${index + 1}`}`; selection.append(option);
    });
    selection.value = selectedId || "";
    const region = selected();
    for (const control of [selection, name, prompt, strength, ...Object.values(detailInputs)]) {
      control.disabled = editingLocked || !region;
    }
    name.value = region?.name || ""; prompt.value = region?.prompt || ""; strength.value = region?.strength ?? 1;
    strengthOffset.value = layout().strength_offset ?? 0;
    if (strengthOffsetCaption) strengthOffsetCaption.textContent = `Strength Offset · +${Number(layout().strength_offset ?? 0).toFixed(2)}`;
    if (region) {
      for (const key of ["x", "y", "width", "height"]) detailInputs[key].value = region.box[key].toFixed(3);
      detailInputs.end.value = Number(region.end ?? 1).toFixed(2);
    }
    for (const control of [add, duplicate, remove, up, down, reference, backgroundSource, strengthOffset]) {
      control.disabled = editingLocked;
    }
    undo.disabled = editingLocked || !history.past.length;
    redo.disabled = editingLocked || !history.future.length;
    summary.style.pointerEvents = editingLocked ? "none" : "auto";
    summary.style.opacity = editingLocked ? "0.55" : "1";
    canvas.style.cursor = editingLocked ? "not-allowed" : "crosshair";
    resizeCanvas();
  }

  function setBackgroundHint(message = "", pending = false) {
    backgroundHint.textContent = message;
    backgroundHint.style.display = message ? "grid" : "none";
    backgroundHint.style.color = pending ? "#bdeff4" : "#f0c879";
  }

  function watchPreviewSource(source) {
    const start = source?.start;
    for (const widgetName of [
      "model_family", "resolution", "swap_dimensions", "process_mode",
      "resize_mode", "crop_position", "controlnet_enabled",
    ]) watchPreviewWidget(start, start?.widgets?.find((widget) => widget?.name === widgetName));
    const control = source?.controlNode;
    for (const widgetName of ["IMAGE SOURCE", "REFERENCE IMAGE", "ENABLE", "enable"]) {
      watchPreviewWidget(control, control?.widgets?.find((widget) => widget?.name === widgetName));
    }
    watchPreviewWidget(source?.node, source?.widget);
  }

  function sourceRevision(source) {
    return [source?.start, source?.controlNode, source?.node]
      .filter((item, index, values) => item && values.indexOf(item) === index)
      .reduce((total, item) => total + Number(item.__cmkRegionalPreviewRevision || 0), 0);
  }

  function syncAvailability() {
    const source = resolveAutomaticPreviewSource(node);
    watchPreviewSource(source);
    const nextLocked = source?.unavailable === true;
    if (nextLocked) {
      const changed = !editingLocked;
      editingLocked = true;
      drag = null;
      automaticRequests.invalidate();
      automaticBackground = null;
      automaticGeometry = null;
      setBackgroundHint("not available");
      if (changed) render();
      else renderCanvas();
      return source;
    }
    if (editingLocked) {
      editingLocked = false;
      automaticSignature = "";
      setBackgroundHint();
      render();
    }
    return source;
  }

  function point(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: clamp((event.clientX - rect.left) / rect.width), y: clamp((event.clientY - rect.top) / rect.height) };
  }

  function handleAt(region, p) {
    if (!region) return null;
    const rect = canvas.getBoundingClientRect();
    const px = 9 / rect.width, py = 9 / rect.height;
    const box = region.box;
    const horizontal = Math.abs(p.x - box.x) <= px ? "w" : Math.abs(p.x - box.x - box.width) <= px ? "e" : "";
    const vertical = Math.abs(p.y - box.y) <= py ? "n" : Math.abs(p.y - box.y - box.height) <= py ? "s" : "";
    return horizontal && vertical ? vertical + horizontal : null;
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (editingLocked) return;
    event.preventDefault(); canvas.setPointerCapture(event.pointerId);
    const p = point(event); const before = cloneLayout(layout());
    const handle = handleAt(selected(), p);
    if (handle) { drag = { mode: "resize", handle, start: p, before }; return; }
    const hits = hitRegionIndices(layout().regions, p.x, p.y);
    if (hits.length) {
      const key = hits.map((index) => layout().regions[index].id).join("|");
      hitCycle = key === lastHits ? (hitCycle + 1) % hits.length : 0; lastHits = key;
      selectedId = layout().regions[hits[hitCycle]].id; render();
      drag = { mode: "move", start: p, before, original: { ...selected().box } }; return;
    }
    const region = makeRegion(layout().regions.length + 1, { x: p.x, y: p.y, width: 0.01, height: 0.01 });
    selectedId = region.id;
    const next = cloneLayout(layout()); next.regions.push(region); setPresent(next);
    drag = { mode: "draw", start: p, before, createdId: region.id };
  });

  canvas.addEventListener("pointermove", (event) => {
    if (editingLocked || !drag) return;
    const p = point(event); const next = cloneLayout(layout()); const index = next.regions.findIndex((region) => region.id === selectedId);
    if (index < 0) return;
    if (drag.mode === "move") next.regions[index].box = moveBox(drag.original, p.x - drag.start.x, p.y - drag.start.y);
    else if (drag.mode === "resize") next.regions[index].box = resizeBox(drag.before.regions.find((region) => region.id === selectedId).box, drag.handle, p.x - drag.start.x, p.y - drag.start.y);
    else {
      next.regions[index].box = {
        x: Math.min(drag.start.x, p.x), y: Math.min(drag.start.y, p.y),
        width: Math.max(0.01, Math.abs(p.x - drag.start.x)), height: Math.max(0.01, Math.abs(p.y - drag.start.y)),
      };
    }
    history = { ...history, present: normalizeLayout(next) }; save(); renderCanvas();
  });

  canvas.addEventListener("pointerup", () => {
    if (editingLocked || !drag) return;
    const current = cloneLayout(layout());
    history = { past: [...history.past, drag.before].slice(-100), present: current, future: [] };
    drag = null; save(); render();
  });

  add.addEventListener("click", () => commit((next) => { const region = makeRegion(next.regions.length + 1); next.regions.push(region); selectedId = region.id; }));
  duplicate.addEventListener("click", () => { const region = selected(); if (!region) return; commit((next) => { const copy = cloneLayout({ regions: [region] }).regions[0]; copy.id = makeRegion(next.regions.length + 1).id; copy.name = `${region.name} copy`; copy.box = moveBox(copy.box, 0.025, 0.025); next.regions.splice(selectedIndex() + 1, 0, copy); selectedId = copy.id; }); });
  remove.addEventListener("click", () => { const index = selectedIndex(); if (index < 0) return; commit((next) => { next.regions.splice(index, 1); selectedId = next.regions[Math.min(index, next.regions.length - 1)]?.id || null; }); });
  undo.addEventListener("click", () => { history = undoHistory(history); selectedId = history.present.regions.find((r) => r.id === selectedId)?.id || history.present.regions[0]?.id || null; save(); render(); });
  redo.addEventListener("click", () => { history = redoHistory(history); selectedId = history.present.regions.find((r) => r.id === selectedId)?.id || history.present.regions[0]?.id || null; save(); render(); });
  up.addEventListener("click", () => { const index = selectedIndex(); if (index <= 0) return; commit((next) => next.regions.splice(index - 1, 0, next.regions.splice(index, 1)[0])); });
  down.addEventListener("click", () => { const index = selectedIndex(); if (index < 0 || index >= layout().regions.length - 1) return; commit((next) => next.regions.splice(index + 1, 0, next.regions.splice(index, 1)[0])); });
  selection.addEventListener("change", () => { selectedId = selection.value; save(); render(); });
  const editStarts = new WeakMap();
  function bindLiveEdit(control, update) {
    control.addEventListener("focus", () => editStarts.set(control, cloneLayout(layout())));
    control.addEventListener("input", () => {
      if (editingLocked) return;
      const region = selected();
      if (!region) return;
      update(region, control.value);
      save(); renderCanvas();
    });
    control.addEventListener("change", () => {
      if (editingLocked) { editStarts.delete(control); return; }
      const before = editStarts.get(control);
      const current = cloneLayout(layout());
      if (before && serializeLayout(before) !== serializeLayout(current)) {
        history = { past: [...history.past, before].slice(-100), present: current, future: [] };
      }
      editStarts.delete(control);
      render();
    });
  }
  bindLiveEdit(name, (region, value) => { region.name = value; });
  bindLiveEdit(prompt, (region, value) => { region.prompt = value; });
  bindLiveEdit(strength, (region, value) => { region.strength = clamp(value, 0, 2); });
  let strengthOffsetStart = null;
  strengthOffset.addEventListener("pointerdown", () => { strengthOffsetStart = cloneLayout(layout()); });
  strengthOffset.addEventListener("input", () => {
    if (editingLocked) return;
    if (!strengthOffsetStart) strengthOffsetStart = cloneLayout(layout());
    history = { ...history, present: normalizeLayout({ ...layout(), strength_offset: clamp(strengthOffset.value, 0, 2) }) };
    save(); render();
  });
  strengthOffset.addEventListener("change", () => {
    if (editingLocked) { strengthOffsetStart = null; return; }
    const before = strengthOffsetStart;
    const current = cloneLayout(layout());
    if (before && serializeLayout(before) !== serializeLayout(current)) {
      history = { past: [...history.past, before].slice(-100), present: current, future: [] };
    }
    strengthOffsetStart = null;
    render();
  });
  for (const key of ["x", "y", "width", "height", "end"]) detailInputs[key].addEventListener("change", () => commit((next) => {
    const region = next.regions[selectedIndex()];
    if (key === "end") region.end = clamp(detailInputs[key].value);
    else region.box = normalizeLayout({ regions: [{ ...region, box: { ...region.box, [key]: Number(detailInputs[key].value) } }] }).regions[0].box;
  }));

  const file = document.createElement("input"); file.type = "file"; file.accept = "image/*"; file.hidden = true; root.append(file);
  reference.addEventListener("click", () => file.click());
  reference.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    if (editingLocked) return;
    delete node.properties.cmkRegionalReferenceImage;
    loadBackground();
  });
  file.addEventListener("change", () => {
    const chosen = file.files?.[0]; if (!chosen) return;
    const reader = new FileReader(); reader.onload = () => {
      const source = new Image(); source.onload = () => {
        if (editingLocked) return;
        const scale = Math.min(1, 1200 / Math.max(source.naturalWidth, source.naturalHeight));
        const compact = document.createElement("canvas"); compact.width = Math.max(1, Math.round(source.naturalWidth * scale)); compact.height = Math.max(1, Math.round(source.naturalHeight * scale));
        compact.getContext("2d").drawImage(source, 0, 0, compact.width, compact.height);
        node.properties.cmkRegionalReferenceImage = compact.toDataURL("image/jpeg", 0.82); loadBackground();
      }; source.src = String(reader.result);
    }; reader.readAsDataURL(chosen);
  });

  function loadBackground() {
    const data = node.properties.cmkRegionalReferenceImage;
    reference.title = data ? "Choose reference background · right-click to clear" : "Choose reference background";
    if (!data) { background = null; renderCanvas(); return; }
    background = new Image(); background.onload = renderCanvas; background.src = data;
  }

  backgroundSource.addEventListener("change", () => {
    if (editingLocked) return;
    node.properties.cmkRegionalBackgroundSource = normalizeBackgroundSource(backgroundSource.value);
    automaticSignature = "";
    automaticRequests.invalidate();
    if (node.properties.cmkRegionalBackgroundSource === "automatic") syncAutomaticBackground(true);
    else setBackgroundHint();
    node.setDirtyCanvas?.(true, true);
    renderCanvas();
  });

  function unavailableAutomaticBackground(signature, message) {
    if (automaticSignature !== signature) automaticRequests.invalidate();
    automaticSignature = signature;
    automaticBackground = null;
    automaticGeometry = null;
    setBackgroundHint(message);
    renderCanvas();
  }

  function syncAutomaticBackground(force = false) {
    if (normalizeBackgroundSource(node.properties.cmkRegionalBackgroundSource) !== "automatic") return;
    const source = syncAvailability();
    if (source?.unavailable) return;
    if (source.error) {
      unavailableAutomaticBackground(`unavailable:${source.error}`, source.error);
      return;
    }
    const start = source.start;
    const processMode = widgetValue(start, "process_mode", "Custom");
    const resizeMode = widgetValue(start, "resize_mode", "Fit");
    const cropPosition = widgetValue(start, "crop_position", "Center");
    const signature = automaticPreviewSignature({
      graphId: node.graph?.rootGraph?.id ?? node.graph?.id,
      startId: start.id,
      sourceId: source.node.id,
      imageReference: source.reference,
      width: layout().width,
      height: layout().height,
      processMode,
      resizeMode,
      cropPosition,
      modelFamily: source.family,
      sourceKind: source.kind,
      controlNodeId: source.controlNode?.id,
      imageRevision: sourceRevision(source),
    });
    if (!force && signature === automaticSignature) return;
    automaticSignature = signature;
    automaticBackground = null;
    automaticGeometry = null;
    const url = automaticImageUrl(source.reference, sourceRevision(source));
    if (!url) {
      unavailableAutomaticBackground(signature, "Automatic background unavailable · image reference cannot be resolved");
      return;
    }
    const request = automaticRequests.begin();
    setBackgroundHint("Loading automatic background …", true);
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      if (!automaticRequests.isCurrent(request) || signature !== automaticSignature) return;
      automaticGeometry = prepareImageGeometry(
        image.naturalWidth, image.naturalHeight, layout().width, layout().height,
        source.kind === "controlnet-reference" ? "Crop" : resizeMode,
        source.kind === "controlnet-reference" ? "Center" : cropPosition,
        source.kind === "controlnet-reference" ? "Custom" : processMode,
      );
      automaticBackground = image;
      setBackgroundHint();
      renderCanvas();
    };
    image.onerror = () => {
      if (!automaticRequests.isCurrent(request) || signature !== automaticSignature) return;
      automaticBackground = null;
      automaticGeometry = null;
      setBackgroundHint("Automatic background unavailable · image preview could not be loaded");
      renderCanvas();
    };
    image.src = url;
  }

  function syncProcessDimensions() {
    labelProcessInput(node);
    const start = upstreamStartNode(node);
    if (!start) {
      processSize.textContent = `PROCESS SIZE · ${layout().width} × ${layout().height} · 01 START HERE not connected`;
      return;
    }
    const dimensions = processDimensionsFromWidgets(
      widgetValue(start, "resolution", ""),
      Boolean(widgetValue(start, "swap_dimensions", false)),
      layout().width,
      layout().height,
    );
    processSize.textContent = `PROCESS SIZE · ${dimensions.width} × ${dimensions.height} · from 01 START HERE`;
    widthWidget.value = dimensions.width;
    heightWidget.value = dimensions.height;
    if (dimensions.width === layout().width && dimensions.height === layout().height) return;
    history.present.width = dimensions.width;
    history.present.height = dimensions.height;
    save(); render();
  }
  const originalSerialize = node.onSerialize;
  node.onSerialize = function(data) {
    save(); const result = originalSerialize?.apply(this, arguments);
    if (this.serialize_widgets) data.widgets_values = canonical.filter((widget) => widget?.serialize !== false).map((widget) => widget?.value ?? null);
    return result;
  };
  node.widgets = [...canonical.filter((widget) => widget !== dataWidget), panel, dataWidget];
  node._cmkRegionalLayoutReload = () => {
    history = createHistory(normalizeLayout(dataWidget.value, widthWidget.value, heightWidget.value));
    selectedId = history.present.regions.some((region) => region.id === node.properties.cmkRegionalSelectedId)
      ? node.properties.cmkRegionalSelectedId
      : history.present.regions[0]?.id || null;
    backgroundSource.value = normalizeBackgroundSource(node.properties.cmkRegionalBackgroundSource);
    automaticSignature = "";
    automaticRequests.invalidate();
    loadBackground(); syncAvailability(); syncProcessDimensions(); syncAutomaticBackground(true); render();
  };
  node._cmkRegionalLayoutSyncAvailability = syncAvailability;
  node._cmkRegionalLayoutSyncDimensions = syncProcessDimensions;
  node._cmkRegionalLayoutSyncBackground = syncAutomaticBackground;
  node._cmkRegionalLayoutSyncCanvas = resizeCanvas;
  node.setSize?.([Math.max(600, node.size?.[0] || 0), Math.max(912, node.size?.[1] || 0)]);
  // Observe the external size authority. Observing `stage` caused a 1px
  // self-lock after ComfyUI temporarily unmounted the widget during navigation.
  new ResizeObserver(resizeCanvas).observe(root);
  loadBackground(); syncAvailability(); syncProcessDimensions(); syncAutomaticBackground(true); render();
}

function schedule(node) {
  for (const delay of [0, 60, 180]) setTimeout(() => {
    try { if (target(node)) editor(node); }
    catch (error) { console.error("CMK Regional Layout UI:", error); }
  }, delay);
}

function scanGraph() {
  for (const node of app.graph?._nodes || []) if (target(node)) {
    editor(node);
    node._cmkRegionalLayoutSyncAvailability?.();
    node._cmkRegionalLayoutSyncDimensions?.();
    node._cmkRegionalLayoutSyncBackground?.();
    node._cmkRegionalLayoutSyncCanvas?.();
  }
}

function installGraphScan() {
  if (window.__cmkRegionalLayoutGraphScanInstalled) return;
  window.__cmkRegionalLayoutGraphScanInstalled = true;
  scanGraph();
  setInterval(scanGraph, 500);
}

if (!window.__cmkRegionalLayoutExtensionRegistered) {
window.__cmkRegionalLayoutExtensionRegistered = true;
app.registerExtension({
  name: "cmk.regional.layout.v2",
  setup() { installGraphScan(); },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_CLASS) return;
    for (const hook of ["onNodeCreated", "onConfigure", "onAdded"]) {
      const original = nodeType.prototype[hook];
      nodeType.prototype[hook] = function() {
        const result = original?.apply(this, arguments);
        if (hook === "onConfigure" && this._cmkRegionalLayoutReload) this._cmkRegionalLayoutReload();
        else schedule(this);
        return result;
      };
    }
  },
  nodeCreated(node) { if (target(node)) schedule(node); },
  loadedGraphNode(node) { if (target(node)) schedule(node); },
});
installGraphScan();
}
