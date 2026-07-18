#!/usr/bin/env python3
"""Build a P0-only operator input packet from handoff CSVs.

This script copies only P0 operator-data queue/field rows into a focused input
packet. It never fills operator values, crawls baseball data, or repairs data.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_INPUT = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv"
)
DEFAULT_FIELDS_INPUT = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_operator_packet"
    / "post_db_fast_path_docker_kbo500"
)
P0_DOMAIN_ORDER = ("season_meta", "schedule_window", "game_day_lineup", "roster_news")
P0_DOMAINS = set(P0_DOMAIN_ORDER)

QUEUE_FIELDNAMES = [
    "queue_id",
    "priority",
    "priority_reason",
    "domain",
    "contract_code",
    "question",
    "required_fields",
    "endpoint_count",
    "endpoints",
    "sample_answer",
    "operator_status",
    "operator_owner",
    "operator_notes",
]
FIELDS_FIELDNAMES = [
    "queue_id",
    "domain",
    "contract_code",
    "question",
    "field_name",
    "field_description",
    "required",
    "operator_value",
    "operator_notes",
]


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _validate_headers(
    *,
    queue_fields: Sequence[str],
    field_fields: Sequence[str],
) -> None:
    if list(queue_fields) != QUEUE_FIELDNAMES:
        raise ValueError(
            f"queue CSV header must match {QUEUE_FIELDNAMES}; got {list(queue_fields)}"
        )
    if list(field_fields) != FIELDS_FIELDNAMES:
        raise ValueError(
            f"fields CSV header must match {FIELDS_FIELDNAMES}; got {list(field_fields)}"
        )


def _select_p0_rows(
    queue_rows: Sequence[Mapping[str, str]],
    field_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    p0_queue_rows = [
        dict(row)
        for row in queue_rows
        if _normalize_text(row.get("priority")) == "P0"
        and _normalize_text(row.get("domain")) in P0_DOMAINS
    ]
    p0_queue_ids = {_normalize_text(row.get("queue_id")) for row in p0_queue_rows}
    p0_field_rows = [
        dict(row)
        for row in field_rows
        if _normalize_text(row.get("queue_id")) in p0_queue_ids
        and _normalize_text(row.get("domain")) in P0_DOMAINS
    ]
    return p0_queue_rows, p0_field_rows


def _build_summary(
    *,
    queue_path: Path,
    fields_path: Path,
    queue_rows: Sequence[Mapping[str, str]],
    field_rows: Sequence[Mapping[str, str]],
    generated_files: Mapping[str, str],
) -> dict[str, Any]:
    domain_counts = Counter(_normalize_text(row.get("domain")) for row in queue_rows)
    status_counts = Counter(
        _normalize_text(row.get("operator_status")) for row in queue_rows
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_queue_items": len(queue_rows),
        "total_field_rows": len(field_rows),
        "domain_counts": {
            domain: domain_counts.get(domain, 0) for domain in P0_DOMAIN_ORDER
        },
        "status_counts": dict(sorted(status_counts.items())),
        "source_queue_path": str(queue_path),
        "source_fields_path": str(fields_path),
        "generated_files": dict(generated_files),
    }


def _render_checklist(summary: Mapping[str, Any]) -> str:
    domain_counts = summary.get("domain_counts") or {}
    status_counts = summary.get("status_counts") or {}
    lines = [
        "# P0 Operator Data Input Checklist",
        "",
        "## Scope",
        "",
        f"- Queue rows: `{summary.get('total_queue_items', 0)}`",
        f"- Field rows: `{summary.get('total_field_rows', 0)}`",
        "- Domains: `season_meta`, `schedule_window`, `game_day_lineup`, `roster_news`",
        "- Do not add external baseball crawling, web search, or synthesized baseball data.",
        "- Keep unanswered, unverified, or inconsistent rows on the `MANUAL_BASEBALL_DATA_REQUIRED` path.",
        "",
        "## Counts",
        "",
    ]
    for domain in P0_DOMAIN_ORDER:
        lines.append(f"- `{domain}`: `{domain_counts.get(domain, 0)}`")
    lines.extend(["", "## Current Status", ""])
    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(
        [
            "",
            "## Operator Status Transition",
            "",
            "- Keep `operator_status=pending` until every required `operator_value` row is filled.",
            "- Change to `operator_status=ready_for_validation` only after source metadata is complete.",
            "- Use `validated` only after validation passes; use `rejected` when the service should keep the manual contract.",
            "",
            "## Required Source Metadata",
            "",
            "- `source_name`: internal or official source name checked by the operator.",
            "- `source_checked_at`: ISO date or datetime.",
            "- `is_verified`: `true` only after operator verification.",
            "- `confidence`: numeric value `0.70` or higher.",
            "",
            "## Domain Notes",
            "",
            "- `season_meta`: enter only confirmed events. If not confirmed, keep the manual contract.",
            "- `schedule_window`: `game_id`, `home_team`, and `away_team` must not conflict with the DB game row.",
            "- `game_day_lineup`: `player_name` must resolve to exactly one `player_basic` row.",
            "- `roster_news`: `season_year`, `team_code`, `player_name`, `roster_event_type`, and `effective_date` are required.",
            "",
            "## Next Commands",
            "",
            "```bash",
            "BEGA_SKIP_APP_INIT=1 .venv/bin/python scripts/run_operator_data_p0_filled_intake.py \\",
            "  --queue reports/operator_data_operator_packet/post_db_fast_path_docker_kbo500/p0_queue.csv \\",
            "  --fields reports/operator_data_operator_packet/post_db_fast_path_docker_kbo500/p0_fields.csv \\",
            "  --source-queue reports/operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv \\",
            "  --source-fields reports/operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv \\",
            '  --db-url "$POSTGRES_DB_URL" \\',
            "  --output-dir reports/operator_data_p0_filled_intake/post_db_fast_path_docker_kbo500",
            "```",
            "",
            "Recovery readiness requires DB checks to run. Do not use `--skip-db-checks` for the filled-intake gate.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_packet(
    *,
    queue_path: Path,
    fields_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    queue_fields, queue_rows = _read_csv(queue_path)
    field_fields, field_rows = _read_csv(fields_path)
    _validate_headers(queue_fields=queue_fields, field_fields=field_fields)
    p0_queue_rows, p0_field_rows = _select_p0_rows(queue_rows, field_rows)

    p0_queue_path = output_dir / "p0_queue.csv"
    p0_fields_path = output_dir / "p0_fields.csv"
    checklist_path = output_dir / "p0_input_checklist.md"
    summary_path = output_dir / "p0_input_summary.json"
    generated_files = {
        "p0_queue": str(p0_queue_path),
        "p0_fields": str(p0_fields_path),
        "checklist": str(checklist_path),
        "summary": str(summary_path),
    }
    summary = _build_summary(
        queue_path=queue_path,
        fields_path=fields_path,
        queue_rows=p0_queue_rows,
        field_rows=p0_field_rows,
        generated_files=generated_files,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(p0_queue_path, p0_queue_rows, QUEUE_FIELDNAMES)
    _write_csv(p0_fields_path, p0_field_rows, FIELDS_FIELDNAMES)
    checklist_path.write_text(_render_checklist(summary), encoding="utf-8")
    _write_json(summary_path, summary)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a P0 operator-data input packet."
    )
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE_INPUT))
    parser.add_argument("--fields", default=str(DEFAULT_FIELDS_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_packet(
        queue_path=Path(args.queue),
        fields_path=Path(args.fields),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
