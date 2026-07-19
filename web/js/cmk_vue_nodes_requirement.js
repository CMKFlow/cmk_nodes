import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const SETTING_ID = "Comfy.VueNodes.Enabled";
const BANNER_ID = "cmk-vue-nodes-required";
const CHECK_INTERVAL_MS = 750;

async function vueNodesEnabled() {
    const response = await api.fetchApi(`/settings/${encodeURIComponent(SETTING_ID)}`);
    return response.ok && await response.json() === true;
}

function showRequirementBanner() {
    if (document.getElementById(BANNER_ID)) return;

    const banner = document.createElement("aside");
    banner.id = BANNER_ID;
    banner.setAttribute("role", "alert");
    Object.assign(banner.style, {
        position: "fixed",
        zIndex: "100000",
        right: "18px",
        top: "72px",
        width: "min(430px, calc(100vw - 36px))",
        padding: "16px 44px 16px 18px",
        border: "1px solid #d39a32",
        borderRadius: "10px",
        background: "#211b12",
        color: "#f4ead7",
        boxShadow: "0 12px 32px rgba(0, 0, 0, .42)",
        font: "13px/1.45 system-ui, sans-serif",
    });
    banner.innerHTML = `
        <strong style="display:block;margin-bottom:5px;color:#ffc75f;font-size:14px">
            CMK Flow benötigt Vue Nodes / Nodes 2.0
        </strong>
        <span>
            Aktiviere in den ComfyUI-Einstellungen <b>Vue Nodes</b> beziehungsweise
            <b>Nodes 2.0</b>. Die Oberfläche stellt sich unmittelbar um. Andernfalls
            fehlen CMKs Advanced-Umschaltung, Dropdowns und dynamische Node-Layouts.
        </span>
        <button type="button" aria-label="Hinweis schließen"
            style="position:absolute;right:10px;top:9px;border:0;background:transparent;color:#d8c9ac;font-size:22px;cursor:pointer">×</button>
    `;
    banner.querySelector("button")?.addEventListener("click", () => banner.remove());
    document.body.append(banner);

    const watcher = window.setInterval(async () => {
        if (!document.getElementById(BANNER_ID)) {
            window.clearInterval(watcher);
            return;
        }
        try {
            if (await vueNodesEnabled()) {
                banner.remove();
                window.clearInterval(watcher);
            }
        } catch (_) {}
    }, CHECK_INTERVAL_MS);
}

app.registerExtension({
    name: "cmk.vue_nodes.requirement.v1",
    async setup() {
        try {
            if (!await vueNodesEnabled()) showRequirementBanner();
        } catch (error) {
            console.warn("[CMK Flow] Could not verify Vue Nodes setting.", error);
        }
    },
});
