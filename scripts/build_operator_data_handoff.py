#!/usr/bin/env python3
"""Build operator-facing CSV handoff queues from taxonomy output.

This script transforms existing ``operator_data_required`` taxonomy records into
operator fill-in artifacts. It does not collect, infer, or repair baseball data.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import analyze_operator_data_required as taxonomy_analyzer

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_required_taxonomy_post_db_fast_path_docker_kbo500.json"
)
DEFAULT_QUEUE_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_handoff_queue_post_db_fast_path_docker_kbo500.csv"
)
DEFAULT_FIELDS_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_handoff_fields_post_db_fast_path_docker_kbo500.csv"
)
DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_handoff_queue_post_db_fast_path_docker_kbo500_summary.json"
)

PRIORITY_BY_DOMAIN = {
    "season_meta": "P0",
    "schedule_window": "P0",
    "game_day_lineup": "P0",
    "roster_news": "P0",
    "venue_ticket": "P1",
    "broadcast_media": "P1",
    "fan_event": "P1",
    "unsupported_external": "P2",
    "db_fast_path_candidate": "P2",
    "subjective_prediction": "P3",
}

PRIORITY_REASONS = {
    "P0": "운영자가 제공할 구조화 데이터 필드가 명확해 우선 입력 대상입니다.",
    "P1": "서비스/구장/팬 경험 정보로 운영자 확인 후 입력할 수 있습니다.",
    "P2": "지원 여부 또는 fast-path 보강 여부를 운영자가 판정해야 합니다.",
    "P3": "주관 평가 기준 합의가 필요해 구조화 가능 항목 뒤에 처리합니다.",
}

PRIORITY_ORDER = ("P0", "P1", "P2", "P3")
DOMAIN_ORDER = tuple(PRIORITY_BY_DOMAIN.keys())
DOMAIN_SORT_INDEX = {domain: index for index, domain in enumerate(DOMAIN_ORDER)}
PRIORITY_SORT_INDEX = {priority: index for index, priority in enumerate(PRIORITY_ORDER)}

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


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"taxonomy payload must be an object: {path}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("taxonomy payload must contain a records list")
    return payload


def _required_fields(record: Mapping[str, Any]) -> List[str]:
    raw_fields = record.get("required_fields") or []
    if isinstance(raw_fields, str):
        return [field for field in raw_fields.split("|") if field]
    if isinstance(raw_fields, Sequence):
        return [str(field) for field in raw_fields if str(field)]
    return []


def _endpoints(record: Mapping[str, Any]) -> str:
    raw_endpoints = record.get("endpoints") or []
    if isinstance(raw_endpoints, str):
        return raw_endpoints
    if isinstance(raw_endpoints, Sequence):
        return "|".join(str(endpoint) for endpoint in raw_endpoints if str(endpoint))
    return ""


def _manual_records(taxonomy: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    records = taxonomy.get("records")
    if not isinstance(records, list):
        raise ValueError("taxonomy payload must contain a records list")
    return [
        record
        for record in records
        if isinstance(record, Mapping)
        and str(record.get("final_verdict") or "") == "operator_data_required"
    ]


def _record_sort_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    domain = str(record.get("domain") or "")
    priority = PRIORITY_BY_DOMAIN.get(domain, "P3")
    return (
        PRIORITY_SORT_INDEX.get(priority, len(PRIORITY_SORT_INDEX)),
        DOMAIN_SORT_INDEX.get(domain, len(DOMAIN_SORT_INDEX)),
        str(record.get("question") or ""),
    )


def build_handoff(
    taxonomy: Mapping[str, Any],
    *,
    source_taxonomy_path: str,
) -> Dict[str, Any]:
    manual_records = sorted(_manual_records(taxonomy), key=_record_sort_key)

    queue_rows: List[Dict[str, Any]] = []
    field_rows: List[Dict[str, Any]] = []
    for index, record in enumerate(manual_records, start=1):
        domain = str(record.get("domain") or "")
        if domain not in PRIORITY_BY_DOMAIN:
            raise ValueError(f"unsupported handoff domain: {domain}")

        priority = PRIORITY_BY_DOMAIN[domain]
        required_fields = _required_fields(record)
        queue_id = f"ODQ-{index:04d}"
        question = str(record.get("question") or "")
        contract_code = str(record.get("contract_code") or "")
        queue_rows.append(
            {
                "queue_id": queue_id,
                "priority": priority,
                "priority_reason": PRIORITY_REASONS[priority],
                "domain": domain,
                "contract_code": contract_code,
                "question": question,
                "required_fields": "|".join(required_fields),
                "endpoint_count": record.get("endpoint_count") or 0,
                "endpoints": _endpoints(record),
                "sample_answer": str(record.get("sample_answer") or ""),
                "operator_status": "pending",
                "operator_owner": "",
                "operator_notes": "",
            }
        )

        contract = taxonomy_analyzer.CONTRACTS.get(domain)
        field_descriptions = contract.field_descriptions if contract is not None else {}
        for field_name in required_fields:
            field_rows.append(
                {
                    "queue_id": queue_id,
                    "domain": domain,
                    "contract_code": contract_code,
                    "question": question,
                    "field_name": field_name,
                    "field_description": field_descriptions.get(field_name, ""),
                    "required": "true",
                    "operator_value": "",
                    "operator_notes": "",
                }
            )

    priority_counter = Counter(row["priority"] for row in queue_rows)
    domain_counter = Counter(row["domain"] for row in queue_rows)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_taxonomy_path": source_taxonomy_path,
        "total_queue_items": len(queue_rows),
        "total_field_rows": len(field_rows),
        "priority_counts": {
            priority: priority_counter.get(priority, 0) for priority in PRIORITY_ORDER
        },
        "domain_counts": {
            domain: domain_counter.get(domain, 0) for domain in DOMAIN_ORDER
        },
    }
    return {
        "summary": summary,
        "queue_rows": queue_rows,
        "field_rows": field_rows,
    }


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _summary_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build operator handoff CSVs from operator-data-required taxonomy output."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--queue-output", default=str(DEFAULT_QUEUE_OUTPUT))
    parser.add_argument("--fields-output", default=str(DEFAULT_FIELDS_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the generated handoff summary to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    handoff = build_handoff(
        _load_json(input_path),
        source_taxonomy_path=_summary_path(input_path),
    )

    write_csv(Path(args.queue_output), handoff["queue_rows"], QUEUE_FIELDNAMES)
    write_csv(Path(args.fields_output), handoff["field_rows"], FIELDS_FIELDNAMES)
    write_json(Path(args.summary_output), handoff["summary"])

    if args.print_summary:
        print(json.dumps(handoff["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
