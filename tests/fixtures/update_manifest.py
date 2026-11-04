"""Record exported API-prompt fixtures in fixtures/manifest.json.

Run after exporting fixtures as described in tests/FIXTURES_EXPORT.md.  The
script only reads the exported files and the canonical subgraphs; it never
modifies a workflow.

    python -I tests/fixtures/update_manifest.py --frontend-version 1.49.6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent
TESTS_ROOT = FIXTURES.parent
sys.path.insert(0, str(TESTS_ROOT))

from validator.graph_model import load_definitions  # noqa: E402
from validator.report import PACKAGE_ROOT, canonical_definitions  # noqa: E402

COMFY_ROOT = PACKAGE_ROOT.parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comfy_version() -> str:
    namespace = {}
    exec((COMFY_ROOT / "comfyui_version.py").read_text(encoding="utf-8"), namespace)
    return namespace["__version__"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-version", required=True)
    args = parser.parse_args(argv)
    canon = canonical_definitions()
    entries = []
    for workflow in sorted((FIXTURES / "workflows").glob("*.json")):
        api = FIXTURES / "api_prompts" / workflow.name
        if not api.is_file():
            print(f"missing API prompt for {workflow.name}", file=sys.stderr)
            return 1
        definitions = []
        for graph in load_definitions(workflow):
            canonical = canon.get(graph.name)
            definitions.append({
                "name": graph.name,
                "signature": graph.signature(),
                "canonical_at_export": bool(canonical and canonical.signature() == graph.signature()),
            })
        entries.append({
            "name": workflow.stem,
            "workflow_file": f"workflows/{workflow.name}",
            "workflow_sha256": sha256(workflow),
            "api_prompt_file": f"api_prompts/{api.name}",
            "api_prompt_sha256": sha256(api),
            "definitions": definitions,
        })
    manifest = {
        "comfyui_version": comfy_version(),
        "frontend_version": args.frontend_version,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fixtures": entries,
    }
    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"recorded {len(entries)} fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
