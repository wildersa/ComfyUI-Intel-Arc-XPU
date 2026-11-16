import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NOTICE_GUARD_KEY = "__omnixpu_notice_shown__";

function showNotice(payload) {
    const toast = app?.extensionManager?.toast;
    if (!toast) {
        return;
    }

    const summary = payload.summary || "ComfyUI-Intel-Arc-XPU";
    const detail = payload.detail || "";
    const severity = payload.severity || "info";
    const life = severity === "info" ? 9000 : 18000;

    if (typeof toast.add === "function") {
        toast.add({ severity, summary, detail, life });
        return;
    }
    if (typeof toast.addAlert === "function") {
        toast.addAlert(`${summary}: ${detail}`);
        return;
    }
    if (typeof toast === "function") {
        toast(`${summary}: ${detail}`);
    }
}

app.registerExtension({
    name: "OmniXPU.SetupNotice",
    async setup() {
        if (window[NOTICE_GUARD_KEY]) {
            return;
        }
        window[NOTICE_GUARD_KEY] = true;

        try {
            const resp =
                api && typeof api.fetchApi === "function"
                    ? await api.fetchApi("/omnixpu/setup-status")
                    : await fetch("/omnixpu/setup-status");
            if (!resp.ok) {
                return;
            }
            const payload = await resp.json();
            if (!payload || payload.status === "ok") {
                return;
            }
            showNotice(payload);
        } catch (err) {
            console.warn("[OmniXPU] Could not query setup status:", err);
        }
    },
});
