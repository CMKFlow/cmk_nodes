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

        const migrate = (node) => {
            if (node.type !== "SEGSPreview" && node.type !== "ImpactSEGSPreview") return;
            node.type = "CMKSEGSPreview";
            node.properties ||= {};
            node.properties["Node name for S&R"] = "CMKSEGSPreview";
            changed = true;
        };

        for (const node of document.nodes || []) migrate(node);
        for (const definition of document.definitions?.subgraphs || []) {
            for (const node of definition.nodes || []) migrate(node);
        }

        if (changed) fs.writeFileSync(file, JSON.stringify(document));
    }
}
