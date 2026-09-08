import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const VISUALIZER = "CMKVisualizer";
const stateByNode = new WeakMap();
const graphHookInstalled = Symbol("cmkVisualGraphHookInstalled");
const providerWidgetHookInstalled = Symbol("cmkVisualProviderWidgetHookInstalled");
const providerWidgetNames = new Set(["label", "sequence", "branch", "stage_key", "live_node_type"]);
let executionId = null;
let providerRefreshTimer = null;
let lastMetadataPreviewBlob = null;

function eventExecutionId(event) {
  return event?.detail?.prompt_id || event?.detail?.promptId || event?.detail?.jobId || null;
}

function imageUrl(info) {
  const query = new URLSearchParams(info || {});
  query.set("preview", query.get("preview") || "");
  return api.apiURL(`/view?${query}`);
}

function parseVisual(message) {
  const value = message?.cmk_visual?.[0];
  if (!value) return null;
  try { return JSON.parse(value); } catch { return null; }
}

function parseVisualStage(output) {
  const value = output?.cmk_visual_stage?.[0];
  if (!value) return null;
  try { return JSON.parse(value); } catch { return null; }
}

function nodeKind(node) {
  return node?.comfyClass || node?.type || node?.constructor?.comfyClass || "";
}

function widgetValue(node, name, fallback) {
  return node?.widgets?.find((widget) => widget.name === name)?.value ?? fallback;
}

function upstreamNode(node, wanted, visited = new Set()) {
  if (!node || visited.has(node.id)) return null;
  visited.add(node.id);
  if (nodeKind(node) === wanted) return node;
  for (const input of node.inputs || []) {
    const link = app.graph?.links?.[input.link];
    const origin = link && app.graph?.getNodeById?.(link.origin_id);
    const found = upstreamNode(origin, wanted, visited);
    if (found) return found;
  }
  return null;
}

function remappedDeclaredProvider(outerNode, item) {
  const innerNodes = outerNode?.subgraph?.nodes || outerNode?.subgraph?._nodes || [];
  if (!innerNodes.length) return item;
  const visualProviders = innerNodes.filter((inner) => nodeKind(inner) === "CMKVisualProvider");
  const visualProvider = visualProviders.find((inner) =>
    item.stage_key && String(widgetValue(inner, "stage_key", "")) === String(item.stage_key)
  ) || visualProviders.find((inner) =>
    Number(widgetValue(inner, "sequence", 0)) === Number(item.sequence || 0)
  ) || (visualProviders.length === 1 ? visualProviders[0] : null);
  const fallbackLiveTypes = {
    controlnet: "CMKControlNetPreparePipe",
    "first-pass": "CMKKSamplerPipe",
    identity: "CMKInstantIDSamplerSDXLPipe",
    refiner: "CMKRefinerPipe",
    detailer: "CMK_SmartDetailerPipe",
  };
  const liveType = String(
    widgetValue(visualProvider, "live_node_type", "")
    || fallbackLiveTypes[item.key]
    || "",
  );
  const liveTypeAliases = {
    FaceProcess: new Set(["CMKFaceProcessPipe", "CMK_FaceProcess", "CMKFaceProcess"]),
  };
  const matchesLiveType = (inner) => {
    const actual = nodeKind(inner);
    return actual === liveType || liveTypeAliases[liveType]?.has(actual);
  };
  const liveNodes = liveType
    ? innerNodes.filter(matchesLiveType)
    : [];
  const liveNode = liveNodes[0] || null;
  const scopedNodeId = (inner) => inner ? `${outerNode.id}:${inner.id}` : null;
  return {
    ...item,
    label: visualProvider
      ? String(widgetValue(visualProvider, "label", item.label || item.key || "Result"))
      : item.label,
    sequence: visualProvider
      ? Number(widgetValue(visualProvider, "sequence", item.sequence || 0))
      : item.sequence,
    branch: visualProvider
      ? String(widgetValue(visualProvider, "branch", item.branch || ""))
      : item.branch,
    stage_key: visualProvider
      ? String(widgetValue(visualProvider, "stage_key", item.stage_key || ""))
      : item.stage_key,
    provider_id: visualProvider
      ? `cmk-CMKVisualProvider-${outerNode.id}-${visualProvider.id}`
      : item.provider_id,
    provider_node_id: scopedNodeId(visualProvider) || item.provider_node_id,
    live_node_id: scopedNodeId(liveNode) || item.live_node_id,
    live_node_ids: liveNodes.length
      ? liveNodes.map(scopedNodeId)
      : (item.live_node_ids || (item.live_node_id ? [item.live_node_id] : [])),
    live_node_resolved: liveNodes.length > 0,
  };
}

function graphProviders() {
  const direct = (app.graph?._nodes || [])
    .filter((node) => nodeKind(node) === "CMKVisualProvider")
    .map((node) => {
      const sampler = upstreamNode(node, "CMKKSamplerPipe");
      const channelNames = (node.inputs || [])
        .filter((input) => ["IMAGE", "SOURCE", "BEFORE", "AFTER"].includes(input.name) && input.link != null)
        .map((input) => input.name.toLowerCase());
      return {
        provider_id: `cmk-CMKVisualProvider-${node.id}`,
        provider_node_id: String(node.id),
        module_instance_id: String(node.id),
        module_type: "CMKVisualProvider",
        module_label: String(widgetValue(node, "label", "Result")),
        sequence: Number(widgetValue(node, "sequence", 10)),
        branch: String(widgetValue(node, "branch", "")),
        stage_key: String(widgetValue(node, "stage_key", "")),
        status: "waiting",
        live_node_id: sampler ? String(sampler.id) : null,
        capabilities: { preview: true, multi_source: channelNames.length > 1, compare: channelNames.length > 1, live: Boolean(sampler) },
        channels: channelNames.map((name) => ({ name, image_index: -1 })),
      };
  });
  const declared = (app.graph?._nodes || []).flatMap((node) =>
    (node.properties?.cmkVisualProviders || [])
      .filter((declaredItem) => {
        // Provider declarations may name the public module switch that owns
        // their live signal. Legacy declarations derive the canonical name
        // from their key so already placed subgraphs obey the same contract.
        const enableWidget = String(
          declaredItem.enable_widget
          || (declaredItem.key ? `${declaredItem.key}_global_enable` : ""),
        );
        if (!enableWidget) return true;
        const widget = node?.widgets?.find((item) => item.name === enableWidget);
        return !widget || Boolean(widget.value);
      })
      .map((declaredItem) => {
      const item = remappedDeclaredProvider(node, declaredItem);
      return ({
      provider_id: item.provider_id || `cmk-${node.id}-${item.key}`,
      module_instance_id: String(node.id),
      module_type: nodeKind(node),
      module_label: String(item.label || item.key || "Result"),
      sequence: Number(item.sequence || 0),
      branch: String(item.branch || ""),
      stage_key: String(item.stage_key || ""),
      status: "waiting",
      live_node_id: item.live_node_id == null ? null : String(item.live_node_id),
      live_node_ids: (item.live_node_ids || []).map(String),
      live_node_resolved: Boolean(item.live_node_resolved),
      capabilities: { preview: true, multi_source: false, compare: false, live: false, ...(item.capabilities || {}) },
      channels: [],
    });}),
  );
  const providers = new Map();
  for (const provider of declared) {
    const current = providers.get(provider.provider_id);
    if (!current || (!current.live_node_resolved && provider.live_node_resolved)) {
      providers.set(provider.provider_id, provider);
    }
  }
  for (const provider of direct) providers.set(provider.provider_id, provider);
  return [...providers.values()].sort((a, b) => a.sequence - b.sequence);
}

function selectedProvider(state) {
  return state.providers.find((provider) => provider.provider_id === state.providerId) || state.providers[0];
}

function providerSemanticKey(provider) {
  if (provider?.branch && provider?.stage_key) {
    return `${provider.branch}\u0000${provider.stage_key}`;
  }
  return `${Number(provider?.sequence || 0)}\u0000${String(provider?.module_label || provider?.label || "")}`;
}

function completedDeclarations(declarations, completedProviders) {
  const grouped = new Map();
  for (const declaration of declarations) {
    const key = providerSemanticKey(declaration);
    const group = grouped.get(key) || [];
    group.push(declaration);
    grouped.set(key, group);
  }
  const consumed = new Set();
  return completedProviders.map((provider) => {
    const candidates = grouped.get(providerSemanticKey(provider)) || [];
    const exact = candidates.find((candidate) =>
      !consumed.has(candidate.provider_id)
      && nodeIdMatches(candidate.provider_node_id, provider.module_instance_id)
    );
    const declaration = exact || candidates.find((candidate) => !consumed.has(candidate.provider_id));
    if (declaration) consumed.add(declaration.provider_id);
    return [provider, declaration];
  });
}

function providerDisplayLabels(providers) {
  const counts = new Map();
  return providers.map((provider) => {
    const base = String(provider.module_label || provider.label || "Result");
    const count = (counts.get(base) || 0) + 1;
    counts.set(base, count);
    return count === 1 ? base : `${base} ${count}`;
  });
}

function nodeIdMatches(expected, candidate) {
  if (expected == null || candidate == null) return false;
  const left = String(expected);
  const right = String(candidate);
  return left === right || left.endsWith(`:${right}`) || right.endsWith(`:${left}`);
}

function providerForNode(state, ...nodeIds) {
  const catalog = state.catalog || [];
  const liveMatch = catalog.find((provider) =>
    (provider.live_node_ids || [provider.live_node_id])
      .filter((nodeId) => nodeId != null)
      .some((expected) => nodeIds.some((nodeId) => nodeIdMatches(expected, nodeId))),
  );
  if (liveMatch) return liveMatch;
  const moduleMatches = catalog.filter((provider) =>
    provider.capabilities?.live
    && !provider.live_node_id
    && nodeIds.some((nodeId) => nodeIdMatches(provider.module_instance_id, nodeId)),
  );
  return moduleMatches.length === 1 ? moduleMatches[0] : null;
}

function eventNodeIds(detail = {}) {
  if (["string", "number"].includes(typeof detail)) return [detail];
  const metadata = detail.metadata || detail.preview_metadata || {};
  return [
    detail.nodeId, detail.node_id, detail.node,
    detail.realNodeId, detail.real_node_id,
    detail.parentNodeId, detail.parent_node_id,
    detail.displayNodeId, detail.display_node_id,
    metadata.nodeId, metadata.node_id, metadata.node,
    metadata.realNodeId, metadata.real_node_id,
    metadata.parentNodeId, metadata.parent_node_id,
    metadata.displayNodeId, metadata.display_node_id,
  ].filter((value) => value != null);
}

function exposeProvider(state, provider) {
  if (!state || !provider) return null;
  let visible = state.providers.find((item) => item.provider_id === provider.provider_id);
  if (!visible) {
    visible = { ...provider, channels: [...(provider.channels || [])] };
    state.providers.push(visible);
  }
  return visible;
}

function activateProvider(state, provider) {
  const visible = exposeProvider(state, provider);
  if (!visible) return null;
  for (const item of state.providers) {
    if (item.provider_id !== visible.provider_id && item.status === "active") {
      item.status = "available";
    }
  }
  visible.status = "active";
  if (state.autoFollow) state.providerId = visible.provider_id;
  return visible;
}

function acceptLivePreview(state, provider, blob) {
  if (!state || !provider || !blob) return;
  const visible = activateProvider(state, provider);
  if (!visible) return;
  if (visible.liveUrl) URL.revokeObjectURL(visible.liveUrl);
  visible.liveUrl = URL.createObjectURL(blob);
  state.live = {
    provider_id: visible.provider_id,
    step: state.live?.provider_id === visible.provider_id ? state.live.step : 0,
    total: state.live?.provider_id === visible.provider_id ? state.live.total : 0,
  };
}

function acceptExecutedImage(state, provider, descriptor) {
  if (!state || !provider || !descriptor) return;
  const visible = activateProvider(state, provider);
  if (!visible) return;
  if (visible.liveUrl) URL.revokeObjectURL(visible.liveUrl);
  visible.liveUrl = null;
  visible.executedUrl = imageUrl(descriptor);
  state.live = null;
}

function loadWaitingProviders(node) {
  const state = stateByNode.get(node);
  if (!state || state.runActive) return;
  state.catalog = graphProviders();
  render(node);
}

function scheduleProviderRefresh() {
  if (providerRefreshTimer != null) clearTimeout(providerRefreshTimer);
  providerRefreshTimer = setTimeout(() => {
    providerRefreshTimer = null;
    for (const node of app.graph?._nodes || []) {
      if (stateByNode.has(node)) loadWaitingProviders(node);
    }
  }, 0);
}

function installGraphHooks(nodeType) {
  if (nodeType.prototype[graphHookInstalled]) return;
  nodeType.prototype[graphHookInstalled] = true;
  for (const hook of ["onAdded", "onRemoved", "onConnectionsChange"]) {
    const original = nodeType.prototype[hook];
    nodeType.prototype[hook] = function () {
      const result = original?.apply(this, arguments);
      scheduleProviderRefresh();
      return result;
    };
  }
}

function installProviderWidgetRefresh(nodeType) {
  if (nodeType.prototype[providerWidgetHookInstalled]) return;
  nodeType.prototype[providerWidgetHookInstalled] = true;
  const created = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = created?.apply(this, arguments);
    setTimeout(() => {
      for (const widget of this.widgets || []) {
        if (!providerWidgetNames.has(widget.name) || widget.__cmkVisualRefreshWrapped) continue;
        widget.__cmkVisualRefreshWrapped = true;
        const callback = widget.callback;
        widget.callback = function () {
          const callbackResult = callback?.apply(this, arguments);
          scheduleProviderRefresh();
          return callbackResult;
        };
      }
    }, 0);
    return result;
  };
  const changed = nodeType.prototype.onWidgetChanged;
  nodeType.prototype.onWidgetChanged = function (name) {
    const result = changed?.apply(this, arguments);
    if (providerWidgetNames.has(name)) scheduleProviderRefresh();
    return result;
  };
}

function channelUrl(state, channel) {
  return channel && state.images[channel.image_index] ? imageUrl(state.images[channel.image_index]) : "";
}

function comparePair(channels) {
  const named = (name) => channels.find((item) => item.name === name);
  return [named("before") || named("source"), named("after") || named("result")];
}

function renderImage(node, state, provider, channel) {
  if (provider.liveUrl) {
    const image = document.createElement("img");
    image.className = "cmk-visual-image";
    image.draggable = false;
    image.src = provider.liveUrl;
    return image;
  }
  if (provider.executedUrl) {
    const image = document.createElement("img");
    image.className = "cmk-visual-image";
    image.draggable = false;
    image.src = provider.executedUrl;
    return image;
  }
  const [left, right] = comparePair(provider.channels || []);
  if (provider.capabilities?.compare && left && right) {
    const compare = document.createElement("div");
    compare.className = "cmk-visual-click-compare";
    compare.title = "Hold mouse button to show before";
    const image = document.createElement("img");
    image.draggable = false;
    const beforeUrl = channelUrl(state, left);
    const afterUrl = channelUrl(state, right);
    const label = document.createElement("span");
    const show = (before) => {
      image.src = before ? beforeUrl : afterUrl;
      label.textContent = before ? "BEFORE" : "AFTER";
    };
    compare.onpointerdown = (event) => {
      if (event.button !== 0) return;
      compare.setPointerCapture?.(event.pointerId);
      show(true);
      event.preventDefault();
    };
    const showAfter = () => show(false);
    compare.onpointerup = showAfter;
    compare.onpointercancel = showAfter;
    compare.onlostpointercapture = showAfter;
    showAfter();
    compare.append(image, label);
    return compare;
  }
  const image = document.createElement("img");
  image.className = "cmk-visual-image";
  image.draggable = false;
  image.src = channelUrl(state, channel);
  return image;
}

function render(node) {
  const state = stateByNode.get(node);
  if (!state?.root) return;
  const provider = selectedProvider(state);
  state.root.replaceChildren();

  const providerNav = document.createElement("div");
  providerNav.className = "cmk-visual-nav";
  const displayLabels = providerDisplayLabels(state.providers);
  for (const [index, item] of state.providers.entries()) {
    const button = document.createElement("button");
    button.textContent = displayLabels[index];
    button.className = item.provider_id === provider?.provider_id ? "active" : "";
    button.onclick = (event) => {
      event.stopPropagation();
      state.providerId = item.provider_id;
      state.channel = null;
      render(node);
    };
    providerNav.append(button);
  }
  state.root.append(providerNav);
  if (!provider) return;

  const channels = provider.channels || [];
  const channel = channels.find((item) => item.name === state.channel) || channels.at(-1);
  if (channels.length > 1 && !provider.capabilities?.compare) {
    const channelNav = document.createElement("div");
    channelNav.className = "cmk-visual-nav channels";
    for (const item of channels) {
      const button = document.createElement("button");
      button.textContent = item.name;
      button.className = item.name === channel?.name ? "active" : "";
      button.onclick = (event) => {
        event.stopPropagation();
        state.channel = item.name;
        render(node);
      };
      channelNav.append(button);
    }
    state.root.append(channelNav);
  }

  state.root.append(renderImage(node, state, provider, channel));
}

api.addEventListener("execution_start", (event) => {
  executionId = eventExecutionId(event);
  for (const node of app.graph?._nodes || []) {
    const state = stateByNode.get(node);
    if (!state) continue;
    state.executionId = executionId;
    state.runActive = true;
    state.autoFollow = true;
    for (const provider of state.providers) {
      if (provider.liveUrl) URL.revokeObjectURL(provider.liveUrl);
    }
    state.catalog = graphProviders();
    state.providers = [];
    state.providerId = null;
    state.images = [];
    state.live = null;
    state.completedProviderIds = null;
    node.progress = null;
    render(node);
  }
});

api.addEventListener("b_preview_with_metadata", (event) => {
  const eventId = eventExecutionId(event);
  if (eventId && eventId !== executionId) return;
  const detail = event.detail || {};
  const blob = detail.blob;
  if (!blob) return;
  let accepted = false;
  for (const node of app.graph?._nodes || []) {
    const state = stateByNode.get(node);
    if (!state || !state.runActive || state.executionId !== executionId) continue;
    const provider = providerForNode(state, ...eventNodeIds(detail));
    if (!provider) continue;
    acceptLivePreview(state, provider, blob);
    accepted = true;
    render(node);
  }
  lastMetadataPreviewBlob = accepted ? blob : null;
});

api.addEventListener("b_preview", (event) => {
  const blob = event.detail instanceof Blob ? event.detail : event.detail?.blob;
  if (!blob || blob === lastMetadataPreviewBlob) {
    lastMetadataPreviewBlob = null;
    return;
  }
  const eventId = eventExecutionId(event);
  if (eventId && eventId !== executionId) return;
  const sourceNodeIds = eventNodeIds(event.detail || {});
  if (!sourceNodeIds.length) return;
  for (const node of app.graph?._nodes || []) {
    const state = stateByNode.get(node);
    if (!state || !state.runActive || state.executionId !== executionId) continue;
    const provider = providerForNode(state, ...sourceNodeIds);
    if (!provider) continue;
    acceptLivePreview(state, provider, blob);
    render(node);
  }
});

api.addEventListener("executed", (event) => {
  const eventId = eventExecutionId(event);
  if (eventId && eventId !== executionId) return;
  const detail = event.detail || {};
  const stage = parseVisualStage(detail.output);
  const stageDescriptor = detail.output?.cmk_visual_stage_images?.[0];
  if (stage && stageDescriptor) {
    const sourceNodeIds = eventNodeIds(detail);
    for (const node of app.graph?._nodes || []) {
      const state = stateByNode.get(node);
      if (!state || !state.runActive || state.executionId !== executionId) continue;
      const startNode = upstreamNode(node, "CMKPipeCreateImage");
      if (!startNode || !sourceNodeIds.some((id) => nodeIdMatches(startNode.id, id))) continue;
      acceptExecutedImage(state, stage, stageDescriptor);
      render(node);
    }
  }
  const descriptor = detail.output?.images?.[0];
  if (!descriptor) return;
  const sourceNodeIds = eventNodeIds(detail);
  if (!sourceNodeIds.length) return;
  for (const node of app.graph?._nodes || []) {
    const state = stateByNode.get(node);
    if (!state || !state.runActive || state.executionId !== executionId) continue;
    const provider = providerForNode(state, ...sourceNodeIds);
    if (!provider) continue;
    acceptExecutedImage(state, provider, descriptor);
    render(node);
  }
});

function finishExecution(event) {
  const eventId = eventExecutionId(event);
  if (eventId && eventId !== executionId) return;
  for (const node of app.graph?._nodes || []) {
    const state = stateByNode.get(node);
    if (!state) continue;
    if (state.completedProviderIds instanceof Set) {
      for (const provider of state.providers) {
        if (!state.completedProviderIds.has(provider.provider_id) && provider.liveUrl) {
          URL.revokeObjectURL(provider.liveUrl);
        }
      }
      state.providers = state.providers.filter((provider) =>
        state.completedProviderIds.has(provider.provider_id)
      );
      if (!state.providerId || !state.completedProviderIds.has(state.providerId)) {
        state.providerId = state.providers.at(-1)?.provider_id || null;
      }
    }
    state.runActive = false;
    state.autoFollow = false;
    state.live = null;
    node.progress = null;
    node.setDirtyCanvas?.(true, false);
    render(node);
  }
}

api.addEventListener("execution_success", finishExecution);
api.addEventListener("execution_error", finishExecution);

app.registerExtension({
  name: "cmk.visualizer",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    installGraphHooks(nodeType);
    if (nodeData.name === "CMKVisualProvider") installProviderWidgetRefresh(nodeType);
    if (nodeData.name !== VISUALIZER) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      const root = document.createElement("div");
      root.className = "cmk-visualizer";
      for (const eventName of ["pointerdown", "pointerup", "mousedown", "mouseup", "click", "dblclick", "contextmenu", "dragstart"]) {
        root.addEventListener(eventName, (event) => event.stopPropagation());
      }
      this.addDOMWidget("cmk_visualizer", "visual", root, { serialize: false });
      stateByNode.set(this, { root, catalog: [], providers: [], images: [], providerId: null, channel: null, live: null, completedProviderIds: null, executionId, runActive: false, autoFollow: false });
      setTimeout(() => loadWaitingProviders(this), 0);
      return result;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = configured?.apply(this, arguments);
      setTimeout(() => loadWaitingProviders(this), 0);
      return result;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = executed?.apply(this, arguments);
      const visual = parseVisual(message);
      const state = stateByNode.get(this);
      if (state && visual) {
        const existing = new Map(state.providers.map((item) => [item.provider_id, item]));
        const declarations = state.catalog?.length ? state.catalog : graphProviders();
        // The completed backend payload is authoritative for this run. Cached
        // modules publish their own providers; a provider omitted here was
        // bypassed and must not survive merely because it existed previously.
        const completed = new Map();
        const incomingImages = message.cmk_visual_images || [];
        let lastCompletedId = null;
        for (const [completedProvider, declaration] of completedDeclarations(
          declarations,
          visual.providers || [],
        )) {
          const providerId = declaration?.provider_id || completedProvider.provider_id;
          const provider = {
            ...completedProvider,
            provider_id: providerId,
            channels: (completedProvider.channels || []).map((channel) => ({
              ...channel,
              image_index: Number(channel.image_index),
            })),
          };
          const previous = existing.get(providerId) || existing.get(completedProvider.provider_id);
          if (previous?.liveUrl) URL.revokeObjectURL(previous.liveUrl);
          completed.set(providerId, provider);
          lastCompletedId = providerId;
        }
        state.providers = [...completed.values()];
        state.images = incomingImages;
        state.completedProviderIds = new Set(completed.keys());
        if (state.autoFollow && lastCompletedId) {
          state.providerId = lastCompletedId;
        } else if (!state.providerId || !state.providers.some((item) => item.provider_id === state.providerId)) {
          state.providerId = state.providers.at(-1)?.provider_id || null;
        }
        state.live = null;
        render(this);
      }
      return result;
    };
  },
});

const style = document.createElement("style");
style.textContent = `
.cmk-visualizer{width:100%;height:100%;min-width:0;min-height:0;overflow:hidden;display:flex;flex-direction:column;gap:7px;background:#111;padding:8px;box-sizing:border-box}
.cmk-visual-nav{display:flex;gap:5px;flex-wrap:wrap;flex:0 1 auto;min-height:0;overflow:auto}.cmk-visual-nav button{background:#282828;color:#bbb;border:1px solid #444;border-radius:4px;padding:4px 8px}.cmk-visual-nav button.active{background:#555;color:#fff}
.cmk-visual-nav.channels button{font-size:11px}.cmk-visual-image{display:block;width:100%;height:100%;min-width:0;min-height:0;flex:1 1 0;object-fit:contain;background:#000}
.cmk-visual-click-compare{position:relative;min-width:0;min-height:0;flex:1 1 0;overflow:hidden;background:#000;cursor:pointer}.cmk-visual-click-compare img{display:block;width:100%;height:100%;object-fit:contain}.cmk-visual-click-compare span{position:absolute;right:8px;bottom:8px;padding:3px 6px;border-radius:3px;background:#000a;color:#ddd;font-size:10px;pointer-events:none}
`;
document.head.append(style);
