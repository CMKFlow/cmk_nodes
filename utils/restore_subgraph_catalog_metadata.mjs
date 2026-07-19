#!/usr/bin/env node

import fs from "node:fs";
import { execFileSync } from "node:child_process";

const files = [
  "subgraphs/CMK Flow · 02 LoRA Stack.json",
  "subgraphs/CMK Flow · 10 KSampler 1st Pass.json",
  "subgraphs/CMK Flow · 20 Refiner.json",
  "subgraphs/CMK Flow · 30 Detailer.json",
  "subgraphs/CMK Flow · 40 FaceSwap.json",
  "subgraphs/CMK Flow · 50 FaceProcess.json",
  "subgraphs/CMK Flow · 90 Upscale & Save.json",
];

for (const file of files) {
  const current = JSON.parse(fs.readFileSync(file, "utf8"));
  const published = JSON.parse(
    execFileSync("git", ["show", `HEAD:${file}`], { encoding: "utf8" }),
  );
  const catalogMetadata = published.extra?.CMKFlow;
  if (!catalogMetadata?.published) {
    throw new Error(`${file}: HEAD has no published extra.CMKFlow metadata`);
  }

  current.extra = { ...(current.extra ?? {}), CMKFlow: catalogMetadata };
  fs.writeFileSync(file, `${JSON.stringify(current)}\n`);
  console.log(`${file}: restored extra.CMKFlow`);
}
