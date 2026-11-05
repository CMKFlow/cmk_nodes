const APP_STUB = "data:text/javascript," + encodeURIComponent(
    "export const app = globalThis.__cmkTestApp;",
);
const API_STUB = "data:text/javascript," + encodeURIComponent(
    "export const api = globalThis.__cmkTestApi;",
);

export async function resolve(specifier, context, nextResolve) {
    if (specifier === "../../../scripts/app.js" || specifier === "/scripts/app.js") {
        return { url: APP_STUB, shortCircuit: true };
    }
    if (specifier === "/scripts/api.js") {
        return { url: API_STUB, shortCircuit: true };
    }
    return nextResolve(specifier, context);
}
