# CMK Flow Compatibility

## Frontend requirement

CMK Flow requires ComfyUI **Vue Nodes / Nodes 2.0** to be enabled in the active
user profile. The legacy LiteGraph renderer does not provide the dynamic and
advanced widget behavior used by CMK Flow.

For sampler and refiner previews, set **Comfy → Execution → Live preview
method** to **auto** in the active user profile.

**Specification:** Draft 0.2  
**Purpose:** Integration contract for third-party nodes and modules  
**Authority:** `ARCHITECTURE.md` remains authoritative for CMK's own architecture

CMK Flow Compatibility describes how an external ComfyUI extension can integrate
with CMK Flow without weakening its guided, predictable user experience.

This document is intentionally small. It is not a certification programme, a
marketplace policy, or a promise that compatible extensions will automatically
appear in the CMK Flow Browser. It defines a stable direction developers can
already design against.

## 1. Compatibility is a user promise

A technically connectable node is not automatically CMK Flow compatible.

A compatible integration must make the following promise to the user:

- its purpose is clear from its name and visible controls;
- its position in a workflow is understandable;
- its inputs and outputs have one unambiguous responsibility;
- disabling it does not produce unexpected work or side effects;
- it cannot silently replace current image or process data with stale data;
- errors remain local to the feature that caused them;
- normal use does not require knowledge of the internal implementation.

The aim is not to expose every possibility. The aim is to provide a dependable
solution that can be freely combined with other CMK Flow modules.

## 2. Two integration levels

### 2.1 CMK Toolbox Integration

A Toolbox integration is an individual node for experienced users. It may use
native ComfyUI types, expose technical controls, and allow alternative wiring.

It must still provide:

- a clear and maintained purpose;
- accurate input and output names;
- sensible defaults;
- a short functional description;
- explicit optional dependencies;
- actionable runtime errors instead of breaking unrelated nodes.

Toolbox integration does **not** imply CMK Flow module compatibility.

### 2.2 CMK Flow Module Compatibility

A Flow-compatible integration is a guided module intended for the curated Flow
area. In addition to the Toolbox requirements, it must comply with all rules in
the following sections.

The preferred visible outer workflow remains:

```text
MODEL | PROCESS | IMAGE | LOG
```

Specialised protected transitions such as `SAMPLED`, `SAMPLER`, `REFINER`,
`DETAILER`, and `FACE` are permitted only for a clearly defined producer and
consumer pair.

Model-family compatibility is declared per integration and is not implied by
CMK Flow compatibility itself. An integration may target SDXL, another model
family, several families, or a model-independent processing step.

## 3. Transport responsibilities

| Role | Responsibility | Required behaviour |
|---|---|---|
| `MODEL` | Shared model resources | Read-only; never polluted with local patches or module state |
| `PROCESS` | Non-pixel process context | Copy-on-write; no authoritative image hidden inside it |
| `IMAGE` | Current authoritative pixels | Every visible image change travels through this connection |
| `LOG` | Structured documentation | Passive; never controls execution or changes results |

An integration must not use one role as an undocumented substitute for another.

### 3.1 No accidental passthrough

A compute node should only expose values that it produces or intentionally
owns. It must not pass unrelated inputs through merely to make a workflow look
linear.

If an outer module must preserve `MODEL`, `PROCESS`, `IMAGE`, or `LOG`, that
responsibility belongs to an explicit module boundary. Public results must come
from that boundary; no alternative output may bypass it.

### 3.2 Authoritative image

`IMAGE` is the current workflow image. A module may use internal working copies,
but it must never prefer a stale image stored in `PROCESS`, a cache, or an
internal branch over the image connected to its public input.

## 4. Enable and bypass behaviour

Optional modules distinguish between:

```text
GLOBAL ENABLE  → enables or skips the complete module
LOCAL ENABLE   → enables one instance inside a parallel module
MODE           → selects a real processing function
```

When a complete module is disabled:

- expensive models and processors should not be loaded unnecessarily;
- no hidden branch should execute merely to build a preview;
- the authoritative input image must leave the boundary unchanged;
- `PROCESS` and `MODEL` remain unmodified;
- `LOG` may document the bypass but must not trigger it.

## 5. Parallel branches and boundaries

Parallel execution is allowed when it provides a real functional benefit.

Each branch must:

- receive immutable shared preparation data;
- own its local parameters and results;
- produce a branch result suitable for an explicit merge;
- avoid modifying shared `MODEL` or `PROCESS` objects in place;
- be independently disableable when this is part of the module design.

After merging, one mandatory boundary materialises the module result. Previews,
comparers, following modules, and public outputs must all read from the same
post-boundary result.

If persistent caching is used, its key must include every value that can affect
the result. Cache formats need a revision marker and must fail safely after a
code or dependency change. Caches are accelerators, not portable workflow data.

## 6. Visible UI conventions

Standard controls express user intent. Technical tuning belongs under Advanced.

Recommended order:

```text
MODEL
PROCESS
IMAGE or protected transition
LOG
GLOBAL / LOCAL ENABLE
standard controls
advanced controls
```

Visible socket names use concise roles such as `MODEL`, `PROCESS`, `IMAGE`, and
`LOG`. Internal implementation names may be more technical, but they must not
leak into the guided surface without a user-facing reason.

A compatible module should include:

- a concise display name;
- a one-sentence description;
- short descriptions of its main functions;
- a recommended placement in the Flow;
- honest input and output information;
- a real screenshot or preview when visual distinction is useful;
- status `STABLE`, `BETA`, or `EXPERIMENTAL`.

## 7. Discovery metadata

Flow metadata belongs next to the published module or in an equivalent registry
owned by the providing extension. It must not be permanently hard-coded into UI
logic.

Proposed minimum registration record:

```json
{
  "schemaVersion": 1,
  "provider": "example-node-pack",
  "published": true,
  "displayName": "60 Example Module",
  "category": "Process",
  "description": "Performs one clearly defined step in a CMK Flow.",
  "status": "BETA",
  "order": 60,
  "compatibility": ["CMK Flow", "SDXL"],
  "features": [
    {
      "title": "Clear user-facing function",
      "description": "Explains the result rather than the implementation."
    }
  ],
  "placementNote": "Place after the first sampler and before the final output.",
  "recommendedPredecessors": ["10 KSampler 1st Pass"],
  "recommendedSuccessors": ["90 Upscale & Save"]
}
```

The relationship fields are always read from the perspective of the module
being registered:

- `recommendedPredecessors`: modules that should be placed before this module;
- `recommendedSuccessors`: modules that should be placed after this module.

For the example above, the new module belongs after `10 KSampler 1st Pass` and
before `90 Upscale & Save`.

The existing CMK Flow Browser internally uses the older field names
`recommendedBefore` and `recommendedAfter`. They carry the same predecessor and
successor meaning, but are not part of the public Draft 0.2 registration record
because their reading direction can be misunderstood.

The current Flow Browser only publishes entries explicitly curated by
`cmk_nodes`. External discovery is a planned extension point. Until that exists,
metadata demonstrates readiness but does not guarantee listing.

## 8. Dependency and failure rules

An optional dependency must not prevent unrelated nodes from registering.

If a required model, engine, or node pack is unavailable, the affected function
must provide a clear message that identifies:

- what is missing;
- which feature needs it;
- whether the rest of the pack remains usable.

No compatible integration may silently download dependencies, models, or
updates through the Flow Browser.

## 9. Compatibility checklist

Before describing an integration as CMK Flow compatible, verify:

- [ ] Its purpose and recommended placement are understandable without opening its internals.
- [ ] Every public input and output has one documented responsibility.
- [ ] `MODEL` remains read-only.
- [ ] `PROCESS` is copied before modification and contains no authoritative pixels.
- [ ] `IMAGE` always represents the current result.
- [ ] `LOG` is passive documentation.
- [ ] Disabled execution is a genuine lazy bypass.
- [ ] Parallel branches merge through one mandatory boundary.
- [ ] Public outputs and previews cannot bypass that boundary.
- [ ] Optional dependencies fail locally and clearly.
- [ ] Standard controls express user intent; technical controls are Advanced.
- [ ] Metadata, status, placement advice, and compatibility information are present.
- [ ] The integration has been tested standalone and inside a representative CMK Flow.

## 10. Naming and claims

Until a formal review process exists, third-party projects should use wording
such as:

> Designed for CMK Flow compatibility according to Draft 0.2.

They should not claim to be “official”, “certified”, or “approved by CMK” unless
that statement has been explicitly granted.

The specification will evolve from practical integrations. Backward-compatible
changes are preferred; unavoidable contract changes require a new specification
version and migration guidance.
