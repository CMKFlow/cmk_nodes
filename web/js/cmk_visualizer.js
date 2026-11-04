import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const VISUALIZER = "CMKVisualizer";
const VISUAL_COMPARE = "CMKVisualCompare";
const LEGACY_VISUALIZER_INPUTS = ["MODEL (opt)", "PROCESS", "IMAGE", "LOG", "VISUAL"];
const VISUALIZER_COMMON_OFFSET = 10;
const stateByNode = new WeakMap();
const graphHookInstalled = Symbol("cmkVisualGraphHookInstalled");
const providerWidgetHookInstalled = Symbol("cmkVisualProviderWidgetHookInstalled");
const providerWidgetNames = new Set([
  "label", "sequence", "branch", "stage_key", "live_node_type",
  "controlnet_enabled",
]);
const visualDisplayNodes = new Set();
let executionId = null;
let executingNodeIds = [];
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

function applyVisualizerInputLabels(node) {
  if (nodeKind(node) !== VISUALIZER || !Array.isArray(node.inputs)) return;
  const model = node.inputs.find((input) => input?.name === "MODEL (opt)");
  if (model) model.localized_name = "MODEL";
  node.setDirtyCanvas?.(true, true);
}

function migrateLegacyVisualizerGraph(graphData) {
  if (!Array.isArray(graphData?.nodes) || !Array.isArray(graphData?.links)) return;
  for (const node of graphData.nodes) {
    if (node?.type !== VISUALIZER || !Array.isArray(node.inputs)) continue;
    const sockets = node.inputs.filter((input) => !input?.widget);
    if (
      sockets.length !== LEGACY_VISUALIZER_INPUTS.length
      || sockets.some((input, index) => input?.name !== LEGACY_VISUALIZER_INPUTS[index])
    ) continue;
    const legacyLinkIds = new Set(
      sockets
        .map((input) => input?.link)
        .filter((linkId) => linkId != null)
        .map(String),
    );
    for (const link of graphData.links) {
      if (Array.isArray(link)) {
        if (
          String(link[3]) === String(node.id)
          && Number(link[4]) >= 0
          && Number(link[4]) < LEGACY_VISUALIZER_INPUTS.length
          && legacyLinkIds.has(String(link[0]))
        ) link[4] = Number(link[4]) + VISUALIZER_COMMON_OFFSET;
        continue;
      }
      if (!link || typeof link !== "object") continue;
      const targetId = link.target_id ?? link.targetId;
      const targetSlotKey = "target_slot" in link ? "target_slot" : "targetSlot";
      const targetSlot = Number(link[targetSlotKey]);
      const linkId = link.id ?? link.link_id ?? link.linkId;
      if (
        String(targetId) === String(node.id)
        && targetSlot >= 0
        && targetSlot < LEGACY_VISUALIZER_INPUTS.length
        && legacyLinkIds.has(String(linkId))
      ) link[targetSlotKey] = targetSlot + VISUALIZER_COMMON_OFFSET;
    }
  }
}

function migrateLegacyVisualizerInputs(workflowData) {
  migrateLegacyVisualizerGraph(workflowData);
  for (const subgraph of workflowData?.definitions?.subgraphs || []) {
    migrateLegacyVisualizerGraph(subgraph);
  }
}

function runtimeProviderId(moduleType, moduleInstanceId) {
  const safe = (value) => String(value || "")
    .split("")
    .map((character) => /[A-Za-z0-9]/.test(character) ? character : "-")
    .join("")
    .replace(/^-+|-+$/g, "");
  return `cmk-${safe(moduleType)}-${safe(moduleInstanceId)}`;
}

function activeVisualDisplayNodes() {
  return [...new Set([...(app.graph?._nodes || []), ...visualDisplayNodes])]
    .filter((node) => stateByNode.has(node) && displayNodeEnabled(node));
}

function owningSubgraphNode(innerNode) {
  return (app.graph?._nodes || []).find((outerNode) => {
    const innerNodes = outerNode?.subgraph?.nodes || outerNode?.subgraph?._nodes || [];
    return innerNodes.includes(innerNode);
  }) || null;
}

function graphLink(graph, linkId) {
  return graph?.links?.get?.(linkId) || graph?.links?.[linkId] || null;
}

function displayNodeEnabled(node) {
  if (nodeKind(node) !== VISUAL_COMPARE) return true;
  const input = (node.inputs || []).find((item) => item.name === "enable");
  if (!input || input.link == null) return true;
  const graph = node.graph || node._graph;
  const link = graphLink(graph, input.link);
  if (!link) return true;
  const originId = link.origin_id ?? link.originId;
  const originSlot = Number(link.origin_slot ?? link.originSlot ?? 0);
  if (Number(originId) === -10) {
    const outerNode = owningSubgraphNode(node);
    const outerInput = outerNode?.inputs?.[originSlot];
    const widgetName = outerInput?.widget?.name || outerInput?.name;
    return widgetName ? Boolean(widgetValue(outerNode, widgetName, true)) : true;
  }
  const originNode = graph?.getNodeById?.(originId);
  const output = originNode?.outputs?.[originSlot];
  const widgetName = output?.widget?.name || output?.name;
  return widgetName ? Boolean(widgetValue(originNode, widgetName, true)) : true;
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
  ) || (visualProviders.length === 1 ? visualProviders[0] : null);
  const fallbackLiveTypes = {
    controlnet: "CMKControlNetPreparePipe",
    "first-pass": "CMKKSamplerPipe",
    sampling: "CMKKSamplerPipe",
    identity: "CMKInstantIDSamplerSDXLPipe",
    refiner: "CMKRefinerPipe",
    detailer: "CMK_SmartDetailerPipe",
    "mask-detailer": "CMKMaskDetailerProcess",
    faceprocess: "FaceProcess",
    facerebuild: "CMKInstantIDFaceRebuildSDXL",
    faceswap: "CMKFaceSwapImagePipe",
  };
  const liveType = String(
    widgetValue(visualProvider, "live_node_type", "")
    || fallbackLiveTypes[item.key]
    || "",
  );
  const liveTypeAliases = {
    CMKControlNetPreparePipe: new Set([
      "CMKControlNetPreparePipe",
      "CMKZITControlNetPreparePipe",
      "CMKCombinedControlNetPreparePipe",
    ]),
    FaceProcess: new Set(["CMKFaceProcessPipe", "CMK_FaceProcess", "CMKFaceProcess"]),
    CMKInstantIDSamplerSDXLPipe: new Set([
      "CMKInstantIDSamplerSDXLPipe",
      "CMKInstantIDRegionalAdvancedSDXLPipe",
    ]),
    CMKInstantIDFaceRebuildSDXL: new Set([
      "CMKInstantIDFaceRebuildSDXL",
      "CMKInstantIDFaceRebuildAdvancedSDXL",
    ]),
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
      ? runtimeProviderId("CMKVisualProvider", scopedNodeId(visualProvider))
      : (liveNode
          ? runtimeProviderId(nodeKind(liveNode), scopedNodeId(liveNode))
          : item.provider_id),
    provider_node_id: scopedNodeId(visualProvider) || item.provider_node_id,
    live_node_id: scopedNodeId(liveNode) || item.live_node_id,
    live_node_ids: liveNodes.length
      ? liveNodes.map(scopedNodeId)
      : (item.live_node_ids || (item.live_node_id ? [item.live_node_id] : [])),
    live_node_resolved: liveNodes.length > 0,
  };
}

function visualChainNodeIds(displayNode) {
  const graph = displayNode?.graph || displayNode?._graph || app.graph;
  const upstreamIds = [];
  const visited = new Set();
  let current = displayNode;
  while (current) {
    const visualInput = (current.inputs || []).find((input) => input.name === "VISUAL");
    const link = visualInput?.link == null ? null : graphLink(graph, visualInput.link);
    if (!link) break;
    const originId = link.origin_id ?? link.originId;
    const identity = String(originId);
    if (visited.has(identity)) break;
    visited.add(identity);
    upstreamIds.unshift(identity);
    current = graph?.getNodeById?.(originId);
  }
  return upstreamIds;
}

function graphProviders(displayNode) {
  const maskDetailers = (app.graph?._nodes || [])
    .filter((node) => nodeKind(node) === "CMKMaskDetailerProcess")
    .map((node) => ({
      provider_id: `cmk-CMKMaskDetailer-${node.id}`,
      module_instance_id: String(node.id),
      module_type: "CMKMaskDetailer",
      module_label: "Mask Detailer",
      key: "mask-detailer",
      sequence: 50,
      branch: "mask-detailer",
      stage_key: "mask-detailer.result",
      status: "waiting",
      live_node_id: String(node.id),
      live_node_ids: [String(node.id)],
      live_node_resolved: true,
      capabilities: { preview: true, multi_source: true, compare: true, live: true },
      channels: [
        { name: "before", image_index: -1 },
        { name: "after", image_index: -1 },
      ],
    }));
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
        key: String(widgetValue(node, "stage_key", "") || node.id),
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
        const widget = node?.widgets?.find((item) => item.name === enableWidget)
          || (app.graph?._nodes || [])
            .filter((item) => nodeKind(item) === "CMKPipeCreateImage")
            .flatMap((item) => item.widgets || [])
            .find((item) => item.name === enableWidget);
        return !widget || Boolean(widget.value);
      })
      .map((declaredItem) => {
      const item = remappedDeclaredProvider(node, declaredItem);
      return ({
      provider_id: item.provider_id || `cmk-${node.id}-${item.key}`,
      module_instance_id: String(node.id),
      module_type: nodeKind(node),
      module_label: String(item.label || item.key || "Result"),
      key: String(item.key || ""),
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
  for (const provider of [...direct, ...maskDetailers]) providers.set(provider.provider_id, provider);
  const values = [...providers.values()];
  const remaining = new Set(values);
  const ordered = [];
  for (const nodeId of visualChainNodeIds(displayNode)) {
    for (const provider of values) {
      if (remaining.has(provider) && String(provider.module_instance_id) === nodeId) {
        ordered.push(provider);
        remaining.delete(provider);
      }
    }
  }
  return [...ordered, ...values.filter((provider) => remaining.has(provider))];
}

function selectedProvider(state) {
  return state.providers.find((provider) => provider.provider_id === state.providerId) || state.providers[0];
}

function providerSemanticKey(provider) {
  if (provider?.branch && provider?.stage_key) {
    return `${provider.branch}\u0000${provider.stage_key}`;
  }
  if (provider?.key) return `key\u0000${provider.key}`;
  if (provider?.provider_id) return `provider\u0000${provider.provider_id}`;
  return `instance\u0000${provider?.module_type || ""}\u0000${provider?.module_instance_id || ""}`;
}

function sameProviderStage(left, right) {
  if (!left || !right) return false;
  if (left.provider_id && left.provider_id === right.provider_id) return true;
  if (providerSemanticKey(left) === providerSemanticKey(right)) return true;
  if (!left.stage_key || String(left.stage_key) !== String(right.stage_key || "")) return false;
  return !left.branch || !right.branch || String(left.branch) === String(right.branch);
}

function snapshotProviderIndex(state, provider) {
  const providers = state?.cachedSnapshot?.providers
    || state?.completedSnapshot?.providers
    || [];
  return providers.findIndex((item) => sameProviderStage(item, provider));
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

function cachedProviderSnapshot(state) {
  if (!(state?.completedProviderIds instanceof Set) || !state.providers.length) return null;
  return {
    providers: state.providers.map((provider) => ({
      ...provider,
      liveUrl: null,
      channels: (provider.channels || []).map((channel) => ({ ...channel })),
    })),
    images: [...(state.images || [])],
    providerId: state.providerId,
    channel: state.channel,
  };
}

function providerWasCached(provider, cachedNodeIds) {
  const candidates = [
    ...(provider.live_node_ids || []),
    provider.live_node_id,
    provider.provider_node_id,
    provider.module_instance_id,
  ].filter((value) => value != null);
  return candidates.some((candidate) =>
    cachedNodeIds.some((nodeId) => nodeIdMatches(candidate, nodeId)),
  );
}

function restoreCachedProviders(node, cachedNodeIds) {
  const state = stateByNode.get(node);
  const snapshot = state?.cachedSnapshot;
  if (!state || !snapshot || !state.runActive) return;
  const declarationsById = new Map(
    (state.catalog || []).map((provider) => [provider.provider_id, provider]),
  );
  const declarationsBySemanticKey = new Map(
    (state.catalog || []).map((provider) => [providerSemanticKey(provider), provider]),
  );
  const restored = [];
  // The completed backend VISUAL pipe is authoritative for module order.
  // Keep that order and only filter out providers whose computation is no
  // longer cached; the static graph catalog may differ for nested subgraphs.
  for (const previous of snapshot.providers) {
    const stageMatches = previous.stage_key
      ? (state.catalog || []).filter((provider) =>
          String(provider.stage_key || "") === String(previous.stage_key)
          && (!previous.branch || String(provider.branch || "") === String(previous.branch)),
        )
      : [];
    const declaration = declarationsById.get(previous.provider_id)
      || declarationsBySemanticKey.get(providerSemanticKey(previous))
      || (stageMatches.length === 1 ? stageMatches[0] : null);
    if (!declaration) continue;
    if (!providerWasCached(declaration, cachedNodeIds)) continue;
    restored.push({
      ...previous,
      ...declaration,
      channels: (previous.channels || []).map((channel) => ({ ...channel })),
      status: "available",
      liveUrl: null,
    });
  }
  if (!restored.length) return;
  state.providers = restored;
  state.images = [...snapshot.images];
  state.providerId = restored.some((provider) => provider.provider_id === snapshot.providerId)
    ? snapshot.providerId
    : restored.at(-1)?.provider_id || null;
  state.channel = snapshot.channel;
  render(node);
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
    const desiredIndex = snapshotProviderIndex(state, visible);
    if (desiredIndex < 0) {
      state.providers.push(visible);
    } else {
      const insertionIndex = state.providers.findIndex((item) => {
        const itemIndex = snapshotProviderIndex(state, item);
        return itemIndex >= 0 && itemIndex > desiredIndex;
      });
      state.providers.splice(insertionIndex < 0 ? state.providers.length : insertionIndex, 0, visible);
    }
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
  if (!displayNodeEnabled(node)) {
    state.catalog = [];
    state.providers = [];
    state.images = [];
    render(node);
    return;
  }
  state.catalog = graphProviders(node);
  render(node);
}

function scheduleProviderRefresh() {
  if (providerRefreshTimer != null) clearTimeout(providerRefreshTimer);
  providerRefreshTimer = setTimeout(() => {
    providerRefreshTimer = null;
    for (const node of activeVisualDisplayNodes()) loadWaitingProviders(node);
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
  state.root.replaceChildren();
  if (!displayNodeEnabled(node)) return;
  const provider = selectedProvider(state);

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
  executingNodeIds = [];
  for (const node of [...new Set([...(app.graph?._nodes || []), ...visualDisplayNodes])]) {
    const state = stateByNode.get(node);
    if (!state) continue;
    state.cachedSnapshot = state.completedSnapshot;
    state.executionId = executionId;
    state.runActive = true;
    state.autoFollow = true;
    for (const provider of state.providers) {
      if (provider.liveUrl) URL.revokeObjectURL(provider.liveUrl);
    }
    state.catalog = graphProviders(node);
    state.providers = [];
    state.providerId = null;
    state.images = [];
    state.live = null;
    state.completedProviderIds = null;
    node.progress = null;
    render(node);
  }
});

api.addEventListener("execution_cached", (event) => {
  const eventId = eventExecutionId(event);
  if (eventId && eventId !== executionId) return;
  const cachedNodeIds = (event.detail?.nodes || []).map(String);
  if (!cachedNodeIds.length) return;
  for (const node of activeVisualDisplayNodes()) {
    restoreCachedProviders(node, cachedNodeIds);
  }
});

api.addEventListener("executing", (event) => {
  const eventId = eventExecutionId(event);
  if (eventId && eventId !== executionId) return;
  const nodeIds = eventNodeIds(event.detail);
  if (nodeIds.length) executingNodeIds = nodeIds;
});

api.addEventListener("b_preview_with_metadata", (event) => {
  const eventId = eventExecutionId(event);
  if (eventId && eventId !== executionId) return;
  const detail = event.detail || {};
  const blob = detail.blob;
  if (!blob) return;
  const sourceNodeIds = eventNodeIds(detail);
  const previewNodeIds = sourceNodeIds.length ? sourceNodeIds : executingNodeIds;
  if (!previewNodeIds.length) return;
  let accepted = false;
  for (const node of activeVisualDisplayNodes()) {
    const state = stateByNode.get(node);
    if (!state || !state.runActive || state.executionId !== executionId) continue;
    const provider = providerForNode(state, ...previewNodeIds);
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
  const previewNodeIds = sourceNodeIds.length ? sourceNodeIds : executingNodeIds;
  if (!previewNodeIds.length) return;
  for (const node of activeVisualDisplayNodes()) {
    const state = stateByNode.get(node);
    if (!state || !state.runActive || state.executionId !== executionId) continue;
    const provider = providerForNode(state, ...previewNodeIds);
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
    for (const node of activeVisualDisplayNodes()) {
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
  for (const node of activeVisualDisplayNodes()) {
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
  executingNodeIds = [];
  for (const node of activeVisualDisplayNodes()) {
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
  beforeConfigureGraph(workflowData) {
    migrateLegacyVisualizerInputs(workflowData);
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    installGraphHooks(nodeType);
    if (["CMKVisualProvider", "CMKPipeCreateImage"].includes(nodeData.name)) {
      installProviderWidgetRefresh(nodeType);
    }
    if (nodeData.name !== VISUALIZER && nodeData.name !== VISUAL_COMPARE) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      applyVisualizerInputLabels(this);
      const compareOnly = nodeData.name === VISUAL_COMPARE;
      const root = document.createElement("div");
      root.className = compareOnly ? "cmk-visualizer cmk-preview-compare" : "cmk-visualizer";
      for (const eventName of ["pointerdown", "pointerup", "mousedown", "mouseup", "click", "dblclick", "contextmenu", "dragstart"]) {
        root.addEventListener(eventName, (event) => event.stopPropagation());
      }
      this.addDOMWidget("cmk_visualizer", "visual", root, { serialize: false });
      stateByNode.set(this, { root, catalog: [], providers: [], images: [], providerId: null, channel: null, live: null, completedProviderIds: null, cachedSnapshot: null, completedSnapshot: null, executionId, runActive: false, autoFollow: false, compareOnly });
      visualDisplayNodes.add(this);
      setTimeout(() => loadWaitingProviders(this), 0);
      return result;
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      visualDisplayNodes.delete(this);
      return removed?.apply(this, arguments);
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = configured?.apply(this, arguments);
      applyVisualizerInputLabels(this);
      setTimeout(() => loadWaitingProviders(this), 0);
      return result;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = executed?.apply(this, arguments);
      if (message?.cmk_visual_enabled?.[0] === false) {
        const state = stateByNode.get(this);
        if (state) {
          state.providers = [];
          state.images = [];
          state.completedProviderIds = new Set();
          state.cachedSnapshot = null;
          state.completedSnapshot = null;
          render(this);
        }
        return result;
      }
      const visual = parseVisual(message);
      const state = stateByNode.get(this);
      if (state && visual) {
        const existing = new Map(state.providers.map((item) => [item.provider_id, item]));
        const declarations = state.catalog?.length ? state.catalog : graphProviders(this);
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
        state.completedSnapshot = cachedProviderSnapshot(state);
        state.cachedSnapshot = null;
        render(this);
      }
      return result;
    };
  },
});

const style = document.createElement("style");
style.textContent = `
.cmk-visualizer{width:100%;height:100%;min-width:0;min-height:0;overflow:hidden;display:flex;flex-direction:column;gap:7px;background:#111;padding:8px;box-sizing:border-box}
.cmk-preview-compare>.cmk-visual-nav{display:none}
.cmk-visual-nav{display:flex;gap:5px;flex-wrap:wrap;flex:0 1 auto;min-height:0;overflow:auto}.cmk-visual-nav button{background:#282828;color:#bbb;border:1px solid #444;border-radius:4px;padding:4px 8px}.cmk-visual-nav button.active{background:#555;color:#fff}
.cmk-visual-nav.channels button{font-size:11px}.cmk-visual-image{display:block;width:100%;height:100%;min-width:0;min-height:0;flex:1 1 0;object-fit:contain;background:#000}
.cmk-visual-click-compare{position:relative;min-width:0;min-height:0;flex:1 1 0;overflow:hidden;background:#000;cursor:pointer}.cmk-visual-click-compare img{display:block;width:100%;height:100%;object-fit:contain}.cmk-visual-click-compare span{position:absolute;right:8px;bottom:8px;padding:3px 6px;border-radius:3px;background:#000a;color:#ddd;font-size:10px;pointer-events:none}
`;
document.head.append(style);
