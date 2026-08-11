import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const TARGET_TITLES = new Set([
  "CMK Flow · 20 Refiner SDXL",
  "CMK Flow · 10 KSampler Z-Image Turbo",
]);

function refreshFinalPreviewLayout() {
  const graph = app.rootGraph ?? app.graph;
  for (const node of graph?._nodes ?? []) {
    if (!TARGET_TITLES.has(node?.title)) continue;
    // Re-apply only this node's existing bounds. This refreshes the Vue
    // preview layout without resizing the global canvas or browser viewport.
    node.setSize?.([node.size[0], node.size[1]]);
    node.onResize?.([node.size[0], node.size[1]]);
    node.setDirtyCanvas?.(true, true);
  }
  app.canvas?.setDirty?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

api.addEventListener("execution_success", () => {
  requestAnimationFrame(() => {
    refreshFinalPreviewLayout();
    requestAnimationFrame(refreshFinalPreviewLayout);
  });
});

app.registerExtension({
  name: "cmk.flow.final_preview_refresh",
});
