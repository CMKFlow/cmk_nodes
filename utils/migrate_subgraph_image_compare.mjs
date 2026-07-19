import fs from "node:fs";
import path from "node:path";

const contentDirs = [
    path.resolve("subgraphs"),
    path.resolve("workflows/examples"),
    path.resolve("workflows/reference"),
    path.resolve("workflows/showcase"),
];

for (const contentDir of contentDirs) {
  for (const filename of fs.readdirSync(contentDir)) {
    if (!filename.endsWith(".json")) continue;
    const file = path.join(contentDir, filename);
    const document = JSON.parse(fs.readFileSync(file, "utf8"));
    let changed = false;

    for (const definition of document.definitions?.subgraphs || []) {
        for (const node of definition.nodes || []) {
            if (node.type !== "CMKImageCompare" && node.type !== "Image Comparer [Eclipse]") continue;
            node.type = "ImageCompare";
            node.outputs = [];
            node.properties ||= {};
            node.properties["Node name for S&R"] = "ImageCompare";
            if (!node.inputs.some((input) => input.name === "compare_view")) {
                node.inputs.push({
                    localized_name: "compare_view",
                    name: "compare_view",
                    type: "IMAGECOMPARE",
                    widget: { name: "compare_view" },
                    link: null,
                });
            }

            const outer = document.nodes?.find((candidate) => candidate.type === definition.id);
            if (outer) {
                outer.properties ||= {};
                outer.properties.proxyWidgets ||= [];
                outer.properties.proxyWidgets = outer.properties.proxyWidgets.filter(
                    ([nodeId, name]) => !(String(nodeId) === String(node.id) && name === "eclipse_comparer"),
                );
                if (!outer.properties.proxyWidgets.some(([nodeId, name]) => String(nodeId) === String(node.id) && name === "compare_view")) {
                    outer.properties.proxyWidgets.push([String(node.id), "compare_view"]);
                }
            }
            changed = true;
        }
    }

    if (changed) fs.writeFileSync(file, JSON.stringify(document));
  }
}
