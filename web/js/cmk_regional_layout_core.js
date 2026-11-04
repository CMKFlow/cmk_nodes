export const MIN_EXTENT = 0.01;

export function clamp(value, minimum = 0, maximum = 1) {
  return Math.min(maximum, Math.max(minimum, Number(value) || 0));
}

export function makeRegion(index = 1, box = {}) {
  return {
    id: `region-${Date.now().toString(36)}-${index}`,
    name: `Region ${index}`,
    prompt: "",
    strength: 1,
    end: 1,
    box: normalizeBox({ x: 0.1, y: 0.1, width: 0.35, height: 0.35, ...box }),
  };
}

export function normalizeBox(box = {}) {
  const width = clamp(box.width ?? 0.5, MIN_EXTENT, 1);
  const height = clamp(box.height ?? 0.5, MIN_EXTENT, 1);
  return {
    x: clamp(box.x ?? 0.25, 0, 1 - width),
    y: clamp(box.y ?? 0.25, 0, 1 - height),
    width,
    height,
  };
}

export function normalizeLayout(value, fallbackWidth = 1024, fallbackHeight = 1024) {
  let source = value;
  if (typeof source === "string") {
    try { source = JSON.parse(source); } catch { source = {}; }
  }
  if (!source || typeof source !== "object" || Array.isArray(source)) source = {};
  const used = new Set();
  const regions = (Array.isArray(source.regions) ? source.regions : []).filter((item) => item && typeof item === "object").map((item, offset) => {
    let id = String(item.id || `region-${offset + 1}`);
    if (used.has(id)) id = `${id}-${offset + 1}`;
    used.add(id);
    return {
      id,
      name: String(item.name || `Region ${offset + 1}`),
      prompt: String(item.prompt || ""),
      strength: clamp(item.strength ?? 1, 0, 2),
      end: clamp(item.end ?? 1),
      box: normalizeBox(item.box || item),
    };
  });
  return {
    version: 1,
    width: Math.max(64, Math.round(Number(source.width) || Number(fallbackWidth) || 1024)),
    height: Math.max(64, Math.round(Number(source.height) || Number(fallbackHeight) || 1024)),
    strength_offset: clamp(source.strength_offset ?? 0, 0, 2),
    regions,
  };
}

export function serializeLayout(layout) {
  return JSON.stringify(normalizeLayout(layout));
}

export function processDimensionsFromWidgets(resolution, swapDimensions = false, fallbackWidth = 1024, fallbackHeight = 1024) {
  const match = String(resolution || "").match(/(\d+)\s*[x×]\s*(\d+)/i);
  let width = match ? Number(match[1]) : Number(fallbackWidth) || 1024;
  let height = match ? Number(match[2]) : Number(fallbackHeight) || 1024;
  width = Math.max(64, Math.round(width));
  height = Math.max(64, Math.round(height));
  return swapDimensions ? { width: height, height: width } : { width, height };
}

function pythonRound(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function positionOffset(space, cropPosition, axis) {
  const position = String(cropPosition || "Center").trim().toLowerCase();
  if (axis === "x" && position === "left") return 0;
  if (axis === "x" && position === "right") return space;
  if (axis === "y" && position === "top") return 0;
  if (axis === "y" && position === "bottom") return space;
  return Math.floor(space / 2);
}

export function prepareImageGeometry(
  sourceWidth, sourceHeight, targetWidth, targetHeight,
  resizeMode = "Fit", cropPosition = "Center", processMode = "Custom",
) {
  const sw = Math.max(1, Number(sourceWidth) || 1);
  const sh = Math.max(1, Number(sourceHeight) || 1);
  const tw = Math.max(1, Math.round(Number(targetWidth) || 1));
  const th = Math.max(1, Math.round(Number(targetHeight) || 1));
  const guidedMode = String(processMode || "Custom").trim().toLowerCase();
  const mode = guidedMode.includes("extend") || guidedMode.includes("expand")
    ? "fit"
    : String(resizeMode || "Fit").trim().toLowerCase();

  if (mode === "stretch") {
    return { mode: "stretch", x: 0, y: 0, width: tw, height: th, targetWidth: tw, targetHeight: th };
  }

  const scale = mode === "crop" ? Math.max(tw / sw, th / sh) : Math.min(tw / sw, th / sh);
  const width = Math.max(1, pythonRound(sw * scale));
  const height = Math.max(1, pythonRound(sh * scale));
  if (mode === "crop") {
    const x = -positionOffset(width - tw, cropPosition, "x");
    const y = -positionOffset(height - th, cropPosition, "y");
    return {
      mode: "crop",
      x: x || 0,
      y: y || 0,
      width, height, targetWidth: tw, targetHeight: th,
    };
  }
  return {
    mode: "fit",
    x: positionOffset(tw - width, cropPosition, "x"),
    y: positionOffset(th - height, cropPosition, "y"),
    width, height, targetWidth: tw, targetHeight: th,
  };
}

export function normalizeBackgroundSource(value) {
  return String(value || "").trim().toLowerCase() === "automatic" ? "automatic" : "manual";
}

export function regionalCanvasGeometry(containerWidth, imageWidth, imageHeight) {
  const width = Number(containerWidth);
  if (!Number.isFinite(width) || width <= 16) return null;
  const ratio = Math.max(64, Number(imageWidth) || 1024) / Math.max(64, Number(imageHeight) || 1024);
  const stageWidth = Math.min(width - 16, 390 * ratio);
  if (!Number.isFinite(stageWidth) || stageWidth <= 0) return null;
  return { width: stageWidth, height: stageWidth / ratio };
}

export function packagedReferenceAssetPath(value) {
  const match = String(value || "").trim().match(/^CMK Package ·\s*(.+)$/);
  const filename = String(match?.[1] || "").trim();
  if (!filename || /[/\\]/.test(filename)) return null;
  return `/cmk/reference-assets/${encodeURIComponent(filename)}`;
}

export function firstImageReference(value) {
  const first = Array.isArray(value) ? value[0] : value;
  if (typeof first === "string") return first.trim() || null;
  if (!first || typeof first !== "object" || typeof first.filename !== "string") return null;
  return {
    filename: first.filename,
    subfolder: typeof first.subfolder === "string" ? first.subfolder : "",
    type: typeof first.type === "string" ? first.type : "input",
  };
}

export function regionalLayoutFamily(value) {
  const family = String(value || "SDXL").trim().toLowerCase().replaceAll("_", " ");
  if (family === "zit" || family.includes("z-image") || family.includes("z image")) return "zit";
  if (family === "hybrid") return "hybrid";
  return "sdxl";
}

export function regionalLayoutAvailable(value) {
  return regionalLayoutFamily(value) !== "zit";
}

export function graphNodeClass(node) {
  return node?.comfyClass || node?.type || node?.constructor?.comfyClass || node?.constructor?.nodeData?.name || "";
}

function graphLink(node, linkId) {
  if (!node || linkId == null) return null;
  const graph = node.graph || node._graph;
  return graph?.links?.get?.(linkId) || graph?.links?.[linkId] || null;
}

export function graphInputOrigin(node, inputName = null) {
  const inputs = node?.inputs || [];
  const input = inputName == null
    ? inputs.find((slot) => slot?.link != null)
    : inputs.find((slot) => slot?.name === inputName || slot?.localized_name === inputName);
  const link = graphLink(node, input?.link);
  const originId = link?.origin_id ?? link?.originId;
  const graph = node?.graph || node?._graph;
  return originId == null ? null : graph?.getNodeById?.(originId) || null;
}

function isReroute(node) {
  return /reroute/i.test(`${graphNodeClass(node)} ${node?.title || ""}`);
}

function processInputOrigin(node) {
  if (!node) return null;
  if (isReroute(node)) return graphInputOrigin(node);
  const inputs = node.inputs || [];
  const input = inputs.find((slot) => slot?.link != null && (
    slot?.name === "PROCESS SDXL" || slot?.localized_name === "PROCESS SDXL"
  )) || inputs.find((slot) => slot?.link != null && slot?.type === "CMK_PROCESS_SDXL")
    || inputs.find((slot) => slot?.link != null && (
      slot?.name === "PROCESS" || slot?.localized_name === "PROCESS"
    ));
  if (!input) return null;
  const link = graphLink(node, input.link);
  const originId = link?.origin_id ?? link?.originId;
  const graph = node.graph || node._graph;
  return originId == null ? null : graph?.getNodeById?.(originId) || null;
}

export function sdxlProcessPath(node) {
  const path = [];
  const visited = new Set();
  let current = processInputOrigin(node);
  while (current && !visited.has(current)) {
    visited.add(current);
    path.push(current);
    if (graphNodeClass(current) === "CMKPipeCreateImage") break;
    current = processInputOrigin(current);
  }
  return path;
}

export function upstreamStartNode(node) {
  return sdxlProcessPath(node).find((candidate) => graphNodeClass(candidate) === "CMKPipeCreateImage") || null;
}

function controlNetKind(node) {
  const type = graphNodeClass(node);
  if (type === "CMKCombinedControlNetPreparePipe") return "combined";
  if (type === "CMKControlNetPreparePipe") return "sdxl";
  return null;
}

export function upstreamSDXLControlNets(node) {
  const controls = [];
  for (const host of sdxlProcessPath(node)) {
    const directKind = controlNetKind(host);
    if (directKind) controls.push({ host, node: host, kind: directKind });
    const innerNodes = host?.subgraph?._nodes || host?.subgraph?.nodes || [];
    for (const inner of innerNodes) {
      const kind = controlNetKind(inner);
      if (kind) controls.push({ host, node: inner, kind });
    }
  }
  return controls;
}

export function upstreamSDXLControlNet(node) {
  return upstreamSDXLControlNets(node)[0] || null;
}

function widgetValue(node, name, fallback = null) {
  return node?.widgets?.find((widget) => widget?.name === name)?.value ?? fallback;
}

function imageSourceFromInput(node, inputName) {
  const input = node?.inputs?.find((slot) => slot?.name === inputName || slot?.localized_name === inputName);
  if (!input || input.link == null) return { connected: false };
  let source = graphInputOrigin(node, inputName);
  const visited = new Set();
  while (source && !visited.has(source)) {
    visited.add(source);
    const imageWidget = source.widgets?.find((widget) => widget?.name === "image");
    if (imageWidget) {
      return {
        connected: true,
        node: source,
        widget: imageWidget,
        reference: firstImageReference(imageWidget.value),
      };
    }
    if (!isReroute(source)) break;
    source = graphInputOrigin(source);
  }
  return { connected: true, node: source || null, reference: null };
}

export function startImageSource(start) {
  const source = imageSourceFromInput(start, "IMAGE");
  if (source.reference) return source;
  return source.connected
    ? { error: "Automatic background unavailable · IMAGE source is not a resolvable image loader" }
    : { error: "Automatic background unavailable · 01 START HERE has no IMAGE source" };
}

function linkedControlNetEnabled(controlNode, start) {
  let origin = graphInputOrigin(controlNode, "ENABLE");
  const visited = new Set();
  while (origin && isReroute(origin) && !visited.has(origin)) {
    visited.add(origin);
    origin = graphInputOrigin(origin);
  }
  if (origin) {
    const type = graphNodeClass(origin);
    const flag = widgetValue(origin, "flag", "");
    if (type === "CMKProcessEnableFlag" && flag === "controlnet_enabled") {
      return Boolean(widgetValue(start, "controlnet_enabled", false));
    }
    for (const name of ["value", "ENABLE", "enable"]) {
      const value = widgetValue(origin, name, null);
      if (typeof value === "boolean") return value;
    }
  }
  const local = widgetValue(controlNode, "ENABLE", widgetValue(controlNode, "enable", null));
  if (local != null) return Boolean(local);
  return Boolean(widgetValue(start, "controlnet_enabled", false));
}

export function resolveAutomaticPreviewSource(node) {
  const start = upstreamStartNode(node);
  if (!start) return { error: "Automatic background unavailable · 01 START HERE not connected" };
  const family = regionalLayoutFamily(widgetValue(start, "model_family", "SDXL"));
  if (family === "zit") return { unavailable: true, message: "not available", start, family };

  const fallback = () => {
    const source = startImageSource(start);
    return source.error ? source : { ...source, kind: "start", start, family };
  };
  for (const control of upstreamSDXLControlNets(node)) {
    if (!linkedControlNetEnabled(control.node, start)) continue;
    const imageSource = String(widgetValue(control.node, "IMAGE SOURCE", "Reference Image"));
    if (imageSource === "Base Image") {
      const source = startImageSource(start);
      return source.error ? source : {
        ...source, kind: "controlnet-base", start, family,
        controlNode: control.node, controlHost: control.host,
      };
    }

    const external = imageSourceFromInput(control.node, "REFERENCE IMAGE INPUT");
    if (external.connected) {
      if (external.reference) {
        return {
          ...external, kind: "controlnet-reference", start, family,
          controlNode: control.node, controlHost: control.host,
        };
      }
      return {
        error: "Automatic background unavailable · ControlNet reference source is not a resolvable image loader",
        start, family, controlNode: control.node, controlHost: control.host,
      };
    }
    const referenceWidget = control.node?.widgets?.find((widget) => widget?.name === "REFERENCE IMAGE");
    const reference = firstImageReference(referenceWidget?.value);
    if (reference) {
      return {
        node: control.node,
        widget: referenceWidget,
        reference,
        kind: "controlnet-reference",
        start,
        family,
        controlNode: control.node,
        controlHost: control.host,
      };
    }
  }
  return fallback();
}

export function automaticPreviewSignature(values = {}) {
  return JSON.stringify([
    values.graphId ?? "", values.startId ?? "", values.sourceId ?? "",
    values.imageReference ?? "", values.width ?? "", values.height ?? "",
    values.processMode ?? "", values.resizeMode ?? "", values.cropPosition ?? "",
    values.modelFamily ?? "", values.sourceKind ?? "", values.controlNodeId ?? "",
    values.imageRevision ?? "",
  ]);
}

export function createLatestRequestGuard() {
  let active = 0;
  return {
    begin() { active += 1; return active; },
    invalidate() { active += 1; },
    isCurrent(request) { return request === active; },
  };
}

export function cloneLayout(layout) {
  return normalizeLayout(JSON.parse(JSON.stringify(layout)));
}

export function createHistory(layout) {
  return { past: [], present: cloneLayout(layout), future: [] };
}

export function commitHistory(history, layout) {
  const next = cloneLayout(layout);
  if (serializeLayout(history.present) === serializeLayout(next)) return history;
  return { past: [...history.past, cloneLayout(history.present)].slice(-100), present: next, future: [] };
}

export function undoHistory(history) {
  if (!history.past.length) return history;
  const present = history.past[history.past.length - 1];
  return {
    past: history.past.slice(0, -1),
    present: cloneLayout(present),
    future: [cloneLayout(history.present), ...history.future],
  };
}

export function redoHistory(history) {
  if (!history.future.length) return history;
  const present = history.future[0];
  return {
    past: [...history.past, cloneLayout(history.present)].slice(-100),
    present: cloneLayout(present),
    future: history.future.slice(1),
  };
}

export function hitRegionIndices(regions, x, y) {
  const hits = [];
  for (let index = regions.length - 1; index >= 0; index--) {
    const box = regions[index].box;
    if (x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height) hits.push(index);
  }
  return hits;
}

export function moveBox(box, dx, dy) {
  return normalizeBox({ ...box, x: box.x + dx, y: box.y + dy });
}

export function resizeBox(box, handle, dx, dy) {
  let left = box.x;
  let top = box.y;
  let right = box.x + box.width;
  let bottom = box.y + box.height;
  if (handle.includes("w")) left = clamp(left + dx, 0, right - MIN_EXTENT);
  if (handle.includes("e")) right = clamp(right + dx, left + MIN_EXTENT, 1);
  if (handle.includes("n")) top = clamp(top + dy, 0, bottom - MIN_EXTENT);
  if (handle.includes("s")) bottom = clamp(bottom + dy, top + MIN_EXTENT, 1);
  return normalizeBox({ x: left, y: top, width: right - left, height: bottom - top });
}
