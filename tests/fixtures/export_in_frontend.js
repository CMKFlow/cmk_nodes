// S1 automated fixture export — run in the browser console (or the agent's
// browser pane) of the target ComfyUI instance.  Uses only the original
// frontend: app.loadGraphData() and app.graphToPrompt().  Nothing is queued.
//
// Prerequisite: build_workflows.py wrote <name>.workflow.json into
// ComfyUI/temp/cmk_s1_export/.  Results are written through the instance's
// own /api/userdata endpoint to user/default/cmk_s1_export/<name>.{workflow,api}.json
// and are then copied into tests/fixtures/{workflows,api_prompts}/<name>.json.
const names = ["detailer_adv", "faceprocess_std", "faceprocess_adv", "facerebuild_std",
  "facerebuild_adv", "faceswap_std", "faceswap_adv", "maskdetailer", "upscale_save",
  "chain_postprocess", "upscale_save_legacy_fullflow"];
const view = (f) => fetch(`/api/view?filename=${encodeURIComponent(f)}&type=temp&subfolder=cmk_s1_export&t=${Date.now()}`).then(r => r.json());
const put = (f, obj) => fetch(`/api/userdata/${encodeURIComponent("cmk_s1_export/" + f)}?overwrite=true`,
  { method: "POST", body: JSON.stringify(obj, null, 2) + "\n" }).then(r => r.status);
const summary = [];
for (const name of names) {
  await app.loadGraphData(await view(`${name}.workflow.json`), true, true, `cmk_s1_${name}`);
  await new Promise(r => setTimeout(r, 1500));
  const prompt = await app.graphToPrompt();
  summary.push({ name, upload: [await put(`${name}.workflow.json`, prompt.workflow), await put(`${name}.api.json`, prompt.output)] });
}
summary;
