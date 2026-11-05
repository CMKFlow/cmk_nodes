export const ENGINE_SDXL = "SDXL";
export const ENGINE_ZIT = "Z-Image Turbo";
export const STRATEGY_STANDARD = "STANDARD";
export const STRATEGY_IDENTITY = "IDENTITY";
export const FINISH_REFINER = "SDXL Refiner";
export const FINISH_ZIT = "Z-Image Turbo";
export const FINISH_NONE = "None";

export function allowedFinishes(engine, strategy) {
    if (engine === ENGINE_ZIT) return [FINISH_NONE];
    if (strategy === STRATEGY_IDENTITY) return [FINISH_REFINER];
    return [FINISH_REFINER, FINISH_ZIT];
}

export function normalizeSelection(engine, strategy, finish) {
    const normalizedEngine = engine === ENGINE_ZIT ? ENGINE_ZIT : ENGINE_SDXL;
    const normalizedStrategy = normalizedEngine === ENGINE_ZIT
        ? STRATEGY_STANDARD
        : (strategy === STRATEGY_IDENTITY ? STRATEGY_IDENTITY : STRATEGY_STANDARD);
    const finishes = allowedFinishes(normalizedEngine, normalizedStrategy);
    return {
        engine: normalizedEngine,
        strategy: normalizedStrategy,
        finish: finishes.includes(finish) ? finish : finishes[0],
    };
}

export function modelFamilyFor(engine, strategy, finish) {
    const selection = normalizeSelection(engine, strategy, finish);
    if (selection.engine === ENGINE_ZIT) return "Z-Image Turbo";
    return selection.finish === FINISH_ZIT ? "Hybrid" : "SDXL";
}

export function stopSupported({ engine, strategy, finish, inpaint, instantidMode }) {
    const selection = normalizeSelection(engine, strategy, finish);
    if (
        selection.engine === ENGINE_SDXL
        && selection.strategy === STRATEGY_STANDARD
        && selection.finish === FINISH_ZIT
    ) return true;
    return Boolean(
        selection.engine === ENGINE_SDXL
        && selection.strategy === STRATEGY_IDENTITY
        && selection.finish === FINISH_REFINER
        // Inpaint always uses HYBRID INPAINT; the mode widget is hidden there.
        && (inpaint || String(instantidMode || "HYBRID").trim().toUpperCase() === "HYBRID")
    );
}

export function effectiveStop(requested, configuration) {
    return Boolean(requested && stopSupported(configuration));
}
