"""Export the AI-owned stream contract as deterministic OpenAPI JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.contracts.stream_events_v2 import event_schema
from app.contracts.stream_requests import ChatStreamRequest, CoachAnalyzeRequest

CONTRACT_PATH = ROOT / "contracts" / "ai-stream-v2.openapi.json"


def _extract_schema(
    schema: dict[str, Any],
    components: dict[str, Any],
) -> dict[str, Any]:
    definitions = schema.pop("$defs", {})
    for name, definition in definitions.items():
        existing = components.get(name)
        if existing is not None and existing != definition:
            raise RuntimeError(f"Conflicting contract schema: {name}")
        components[name] = definition
    return schema


def build_contract_document() -> dict[str, Any]:
    """Build the standalone OpenAPI document from runtime Pydantic models."""

    schemas: dict[str, Any] = {}
    chat_request = ChatStreamRequest.model_json_schema(
        ref_template="#/components/schemas/{model}"
    )
    coach_request = CoachAnalyzeRequest.model_json_schema(
        ref_template="#/components/schemas/{model}"
    )
    event_union = event_schema()

    schemas["ChatStreamRequest"] = _extract_schema(chat_request, schemas)
    schemas["CoachAnalyzeRequest"] = _extract_schema(coach_request, schemas)
    schemas["AiStreamV2Event"] = _extract_schema(event_union, schemas)

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Bega AI Stream Contract",
            "version": "2.0.0",
        },
        "paths": {},
        "components": {"schemas": schemas},
    }


def render_contract_json(document: dict[str, Any]) -> str:
    """Render stable UTF-8 JSON with one trailing newline."""

    return json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        Path(temp_name).replace(path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_contract_json(build_contract_document())

    if args.check:
        if not CONTRACT_PATH.exists() or CONTRACT_PATH.read_text(
            encoding="utf-8"
        ) != rendered:
            print(
                "AI stream contract artifact is out of date. "
                "Run scripts/export_ai_stream_contract.py."
            )
            return 1
        print("AI stream contract artifact is current.")
        return 0

    _write_atomic(CONTRACT_PATH, rendered)
    print(f"Wrote {CONTRACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
