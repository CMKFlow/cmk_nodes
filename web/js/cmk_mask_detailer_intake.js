import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS = "CMKMaskDetailerIntake";
const UNLOCKED_IMAGE = "None";

function findWidget(node, name) {
  return node?.widgets?.find((widget) => widget?.name === name);
}

function hideWidget(widget) {
  if (!widget || widget.__cmkHidden) return;
  widget.__cmkHidden = true;
  widget.type = "converted-widget";
  widget.computeSize = () => [0, -4];
  widget.hidden = true;
}

function setButtonLocked(button, locked) {
  if (!button) return;
  button.disabled = Boolean(locked);
  button.options = { ...(button.options || {}), disabled: Boolean(locked) };
}

function hasSnapshot(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return normalized !== "" && normalized !== "none";
}

function isPendingSnapshot(value) {
  return /(^|\/)incoming-[^/]+\.png$/i.test(String(value || ""));
}

function snapshotCandidates(value) {
  const normalized = String(value || "")
    .trim()
    .replace(/\s+\[(?:input|output|temp)\]$/i, "");
  if (!normalized) return [];
  const slash = normalized.lastIndexOf("/");
  const folder = slash >= 0 ? normalized.slice(0, slash) : "";
  const filename = slash >= 0 ? normalized.slice(slash + 1) : normalized;
  const prefix = "clipspace-painted-masked-";
  const candidates = [];
  if (filename.startsWith(prefix)) {
    candidates.push(`${folder ? `${folder}/` : ""}clipspace-mask-${filename.slice(prefix.length)}`);
  }
  candidates.push(normalized);
  return [...new Set(candidates)];
}

function viewPath(value) {
  const slash = value.lastIndexOf("/");
  const params = new URLSearchParams({
    filename: slash >= 0 ? value.slice(slash + 1) : value,
    type: "input",
  });
  if (slash >= 0) params.set("subfolder", value.slice(0, slash));
  return `/view?${params.toString()}`;
}

async function snapshotValidity(value) {
  const candidates = snapshotCandidates(value);
  if (!candidates.length) return "invalid";
  for (const candidate of candidates) {
    let response;
    try {
      response = await api.fetchApi(viewPath(candidate), {
        method: "HEAD",
        cache: "no-store",
      });
    } catch {
      return "unknown";
    }
    if (response.ok) return "valid";
    if (response.status !== 404) return "unknown";
  }
  return "invalid";
}

function ensureImageOption(node, value) {
  const image = findWidget(node, "image");
  const selected = String(value || "").trim();
  if (!image || !hasSnapshot(selected)) return;
  const values = image.options?.values;
  if (Array.isArray(values) && !values.includes(selected)) values.push(selected);
  image.value = selected;
  node.properties ||= {};
  node.properties.image = selected;
}

function configuredImageValue(node, configuration = null) {
  return configuration?.widgets_values_named?.image
    || configuration?.properties?.image
    || configuration?.widgets_values?.[1]
    || node?.properties?.image
    || findWidget(node, "image")?.value;
}

function unlockInvalidSnapshot(node, expectedValue) {
  const image = findWidget(node, "image");
  if (!image || String(image.value || "") !== expectedValue) return;
  image.value = UNLOCKED_IMAGE;
  node.properties ||= {};
  node.properties.image = UNLOCKED_IMAGE;
  image.callback?.(UNLOCKED_IMAGE);
  const state = node.__cmkMaskIntakeState;
  if (state) {
    state.capture.value = false;
    state.status.value = "UNLOCKED";
    setButtonLocked(state.button, false);
  }
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function validateConfiguredSnapshot(node, value) {
  const expectedValue = String(value || "").trim();
  if (!hasSnapshot(expectedValue) || isPendingSnapshot(expectedValue)) return;
  if (node.__cmkMaskValidationValue === expectedValue) return;
  node.__cmkMaskValidationValue = expectedValue;
  const generation = (node.__cmkMaskValidationGeneration || 0) + 1;
  node.__cmkMaskValidationGeneration = generation;
  void snapshotValidity(expectedValue).then((validity) => {
    if (validity !== "invalid") return;
    if (node.__cmkMaskValidationGeneration !== generation) return;
    unlockInvalidSnapshot(node, expectedValue);
  });
}


function setup(node) {
  if (!node || node.__cmkMaskIntake) return;
  const capture = findWidget(node, "capture_current");
  const image = findWidget(node, "image");
  if (!capture || !image) return;
  ensureImageOption(node, configuredImageValue(node));
  node.__cmkMaskIntake = true;
  hideWidget(capture);
  hideWidget(image);

  const initiallyPending = isPendingSnapshot(image.value);
  const initiallyLocked = hasSnapshot(image.value) && !initiallyPending;
  const status = node.addWidget(
    "text",
    "status",
    initiallyLocked ? "LOCKED" : (initiallyPending ? "LIVE · CAPTURE CURRENT" : "NO SNAPSHOT"),
    () => {},
  );
  status.serialize = false;
  status.disabled = true;
  status.readOnly = true;

  const button = node.addWidget("button", "CAPTURE CURRENT IMAGE", null, async () => {
    if (button.disabled) return;
    capture.value = true;
    status.value = "CAPTURE REQUESTED · STARTING …";
    node.setDirtyCanvas?.(true, true);
    try {
      if (typeof app.queuePrompt === "function") await app.queuePrompt(0, 1);
      else status.value = "CAPTURE REQUESTED · START WORKFLOW";
    } finally {
      capture.value = false;
    }
  });
  button.serialize = false;
  setButtonLocked(button, initiallyLocked);
  node.__cmkMaskIntakeState = { capture, image, status, button };
  validateConfiguredSnapshot(node, image.value);

  const originalExecuted = node.onExecuted;
  node.onExecuted = function(message) {
    const result = originalExecuted?.apply(this, arguments);
    const filename = message?.cmk_mask_snapshot?.[0] || message?.cmk_mask_preview?.[0];
    const state = message?.cmk_mask_status?.[0];
    const lock = message?.cmk_mask_lock?.[0];
    if (filename) {
      this.__cmkMaskValidationGeneration = (this.__cmkMaskValidationGeneration || 0) + 1;
      this.__cmkMaskValidationValue = null;
      ensureImageOption(this, filename);
      image.callback?.(filename);
    }
    capture.value = false;
    status.value = state || (filename ? "LOCKED" : "NO SNAPSHOT");
    setButtonLocked(button, lock === "locked");
    this.setDirtyCanvas?.(true, true);
    return result;
  };
}

app.registerExtension({
  name: "cmk.mask-detailer.terminal-intake.v1",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
      const result = created?.apply(this, arguments);
      setTimeout(() => setup(this), 0);
      return result;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function(configuration) {
      const result = configured?.apply(this, arguments);
      const configuredValue = configuredImageValue(this, configuration);
      ensureImageOption(this, configuredValue);
      setTimeout(() => {
        const restoredValue = configuredImageValue(this, configuration);
        ensureImageOption(this, restoredValue);
        setup(this);
        validateConfiguredSnapshot(this, restoredValue);
      }, 0);
      return result;
    };
  },
  nodeCreated(node) {
    if (node?.comfyClass === NODE_CLASS || node?.type === NODE_CLASS) setTimeout(() => setup(node), 0);
  },
  loadedGraphNode(node) {
    if (node?.comfyClass !== NODE_CLASS && node?.type !== NODE_CLASS) return;
    for (const delay of [0, 50, 250]) {
      setTimeout(() => {
        const configuredValue = configuredImageValue(node);
        ensureImageOption(node, configuredValue);
        setup(node);
        validateConfiguredSnapshot(node, configuredValue);
      }, delay);
    }
  },
});
