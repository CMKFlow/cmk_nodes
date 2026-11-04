const APP_STUB = "data:text/javascript," + encodeURIComponent(
    "export const app = globalThis.__cmkTestApp;",
);

export async function resolve(specifier, context, nextResolve) {
    if (specifier === "../../../scripts/app.js") {
        return { url: APP_STUB, shortCircuit: true };
    }
    return nextResolve(specifier, context);
}
