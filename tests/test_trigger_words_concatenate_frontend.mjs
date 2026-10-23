import assert from "node:assert/strict";
import test from "node:test";

import {
    collectToggleLoraNames,
    concatenatedToggles,
    createRevisionGuard,
} from "../web/js/cmk_trigger_words_concatenate_core.js";

function node(id, comfyClass, outputs = []) {
    return { id, comfyClass, mode: 0, inputs: [], outputs, widgets: [] };
}

function graph(nodes, links) {
    const byId = new Map(nodes.map((item) => [item.id, item]));
    const value = { _nodes: nodes, links, getNodeById: (id) => byId.get(id) };
    for (const item of nodes) item.graph = value;
    return value;
}

function connect(links, id, source, output, target, input) {
    links[id] = { origin_id: source.id, origin_slot: output, target_id: target.id };
    source.outputs[output].links ||= [];
    source.outputs[output].links.push(id);
    target.inputs[input].link = id;
}

test("nested concatenate nodes preserve input order and duplicate LoRAs", () => {
    const stackA = node(1, "Lora Stacker (LoraManager)", [
        { name: "LORA_STACK", type: "LORA_STACK" },
        { name: "trigger_words", type: "STRING" },
    ]);
    stackA.lorasWidget = { value: [
        { name: "a.safetensors", active: true },
        { name: "same.safetensors", active: true },
    ] };
    const stackB = node(2, "Lora Stacker (LoraManager)", stackA.outputs.map((item) => ({ ...item })));
    stackB.lorasWidget = { value: [
        { name: "same.safetensors", active: true },
        { name: "off.safetensors", active: false },
    ] };
    const stackC = node(3, "Lora Stacker (LoraManager)", stackA.outputs.map((item) => ({ ...item })));
    stackC.lorasWidget = { value: [{ name: "c.safetensors", active: true }] };
    const concatAB = node(4, "CMKTriggerWordsConcatenate", [{ name: "trigger_words" }]);
    concatAB.inputs = [{ name: "trigger_words_1" }, { name: "trigger_words_2" }];
    const concatABC = node(5, "CMKTriggerWordsConcatenate", [{ name: "trigger_words" }]);
    concatABC.inputs = [{ name: "trigger_words_1" }, { name: "trigger_words_2" }];
    const toggle = node(6, "TriggerWord Toggle (LoraManager)");
    toggle.inputs = [{ name: "trigger_words" }];
    const links = {};
    const currentGraph = graph([stackA, stackB, stackC, concatAB, concatABC, toggle], links);
    connect(links, 10, stackA, 1, concatAB, 0);
    connect(links, 11, stackB, 1, concatAB, 1);
    connect(links, 12, concatAB, 0, concatABC, 0);
    connect(links, 13, stackC, 1, concatABC, 1);
    connect(links, 14, concatABC, 0, toggle, 0);

    assert.deepEqual(collectToggleLoraNames(toggle), [
        "a.safetensors", "same.safetensors", "same.safetensors", "c.safetensors",
    ]);
    assert.deepEqual(concatenatedToggles(currentGraph), [toggle]);
});

test("LORA_STACK inputs precede local stacker entries", () => {
    const upstream = node(1, "Lora Stacker (LoraManager)", [
        { name: "LORA_STACK", type: "LORA_STACK" },
        { name: "trigger_words", type: "STRING" },
    ]);
    upstream.lorasWidget = { value: [{ name: "upstream", active: true }] };
    const downstream = node(2, "Lora Stacker (LoraManager)", upstream.outputs.map((item) => ({ ...item })));
    downstream.inputs = [{ name: "lora_stack", type: "LORA_STACK" }];
    downstream.lorasWidget = { value: [{ name: "local", active: true }] };
    const concat = node(3, "CMKTriggerWordsConcatenate", [{ name: "trigger_words" }]);
    concat.inputs = [{ name: "trigger_words_1" }, { name: "trigger_words_2" }];
    const toggle = node(4, "TriggerWord Toggle (LoraManager)");
    toggle.inputs = [{ name: "trigger_words" }];
    const links = {};
    graph([upstream, downstream, concat, toggle], links);
    connect(links, 10, upstream, 0, downstream, 0);
    connect(links, 11, downstream, 1, concat, 0);
    connect(links, 12, concat, 0, toggle, 0);

    assert.deepEqual(collectToggleLoraNames(toggle), ["upstream", "local"]);
});

test("cycles terminate without repeating entries", () => {
    const concatA = node(1, "CMKTriggerWordsConcatenate", [{ name: "trigger_words" }]);
    const concatB = node(2, "CMKTriggerWordsConcatenate", [{ name: "trigger_words" }]);
    concatA.inputs = [{ name: "trigger_words_1" }, { name: "trigger_words_2" }];
    concatB.inputs = [{ name: "trigger_words_1" }, { name: "trigger_words_2" }];
    const toggle = node(3, "TriggerWord Toggle (LoraManager)");
    toggle.inputs = [{ name: "trigger_words" }];
    const links = {};
    graph([concatA, concatB, toggle], links);
    connect(links, 10, concatB, 0, concatA, 0);
    connect(links, 11, concatA, 0, concatB, 0);
    connect(links, 12, concatA, 0, toggle, 0);

    assert.deepEqual(collectToggleLoraNames(toggle), []);
});

test("only the latest asynchronous update remains current", () => {
    const guard = createRevisionGuard();
    const toggle = {};
    const first = guard.next(toggle);
    const second = guard.next(toggle);

    assert.equal(guard.isCurrent(toggle, first), false);
    assert.equal(guard.isCurrent(toggle, second), true);
});
