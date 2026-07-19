#!/usr/bin/env node

import fs from "node:fs";

const workflowPaths = [
  "workflows/examples/CMK Swap Video.json",
  "workflows/showcase/CMK FaceSwap Video.json",
];

for (const workflowPath of workflowPaths) {
  const workflow = JSON.parse(fs.readFileSync(workflowPath, "utf8"));
  const soundNodeIds = new Set(
    workflow.nodes
      .filter((node) => node.type === "PlaySoundKJ")
      .map((node) => node.id),
  );

  const soundAudioLinkIds = new Set(
    workflow.links
      .filter((link) => soundNodeIds.has(link[3]) && link[4] === 1)
      .map((link) => link[0]),
  );
  const audioLoaderIds = new Set(
    workflow.links
      .filter((link) => soundAudioLinkIds.has(link[0]))
      .map((link) => link[1])
      .filter((nodeId) => {
        const node = workflow.nodes.find((candidate) => candidate.id === nodeId);
        if (node?.type !== "LoadAudio") return false;
        return workflow.links
          .filter((link) => link[1] === nodeId)
          .every((link) => soundNodeIds.has(link[3]));
      }),
  );

  const removedNodeIds = new Set([...soundNodeIds, ...audioLoaderIds]);
  if (removedNodeIds.size === 0) continue;

  workflow.nodes = workflow.nodes.filter((node) => !removedNodeIds.has(node.id));
  workflow.links = workflow.links.filter(
    (link) => !removedNodeIds.has(link[1]) && !removedNodeIds.has(link[3]),
  );

  for (const node of workflow.nodes) {
    for (const input of node.inputs ?? []) {
      if (
        input.link != null &&
        !workflow.links.some((link) => link[0] === input.link)
      ) {
        input.link = null;
      }
    }
    for (const output of node.outputs ?? []) {
      if (Array.isArray(output.links)) {
        output.links = output.links.filter((linkId) =>
          workflow.links.some((link) => link[0] === linkId),
        );
      }
    }
  }

  fs.writeFileSync(workflowPath, `${JSON.stringify(workflow)}\n`);
  console.log(`${workflowPath}: removed ${[...removedNodeIds].join(", ")}`);
}
