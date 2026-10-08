import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const SETTING_ID = "Comfy.Execution.PreviewMethod";
const BANNER_ID = "cmk-live-preview-required";
const CHECK_INTERVAL_MS = 750;

async function previewMethod() {
    const response = await api.fetchApi(`/settings/${encodeURIComponent(SETTING_ID)}`);
    if (!response.ok) return "default";
    const value = await response.json();
    return typeof value === "string" ? value.toLowerCase() : "default";
}

function previewEnabled(method) {
    return method === "auto" || method === "latent2rgb" || method === "taesd";
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
            CMK Flow: Live-Preview ist deaktiviert
        </strong>
        <span>
            Die ComfyUI-Einstellung <b>Live preview method</b> steht auf
            <b>default</b> oder <b>none</b>. Damit zeigen CMK KSampler und
            Visualizer keine Vorschau während des Samplings. Wähle in den
            ComfyUI-Einstellungen <b>auto</b> oder <b>latent2rgb</b>.
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
            if (previewEnabled(await previewMethod())) {
                banner.remove();
                window.clearInterval(watcher);
            }
        } catch (_) {}
    }, CHECK_INTERVAL_MS);
}

app.registerExtension({
    name: "cmk.live_preview.requirement.v1",
    async setup() {
        try {
            if (!previewEnabled(await previewMethod())) showRequirementBanner();
        } catch (error) {
            console.warn("[CMK Flow] Could not verify live-preview setting.", error);
        }
    },
});
