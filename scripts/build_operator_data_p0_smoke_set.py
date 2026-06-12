#!/usr/bin/env python3
"""Build P0 operator-data smoke question lists from validation output.

The script reads normalized validation rows and writes question lists plus an
expectations manifest. It does not collect, infer, repair, or write baseball
data.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORMALIZED = (
    PROJECT_ROOT
    / "reports"
    / "operator_data_validation"
    / "post_db_fast_path_docker_kbo500"
    / "operator_data_normalized_rows.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "operator_data_smoke"
DEFAULT_RECOVERED_OUTPUT = (
    PROJECT_ROOT / "scripts" / "smoke_chatbot_operator_data_p0_recovered.txt"
)
DEFAULT_MANUAL_OUTPUT = (
    PROJECT_ROOT / "scripts" / "smoke_chatbot_operator_data_p0_manual_controls.txt"
)
DEFAULT_ALL_OUTPUT = PROJECT_ROOT / "scripts" / "smoke_chatbot_operator_data_p0_all.txt"
P0_DOMAIN_ORDER = ("season_meta", "schedule_window", "game_day_lineup", "roster_news")
P0_DOMAINS = set(P0_DOMAIN_ORDER)
DOMAIN_SORT = {domain: index for index, domain in enumerate(P0_DOMAIN_ORDER)}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"normalized row {line_number} must be an object")
            rows.append(dict(payload))
    return rows


def _sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return (
        DOMAIN_SORT.get(_normalize_text(row.get("domain")), len(DOMAIN_SORT)),
        _normalize_text(row.get("queue_id")),
    )


def _expectation_for_row(row: Mapping[str, Any], expectation: str) -> dict[str, Any]:
    domain = _normalize_text(row.get("domain"))
    base = {
        "queue_id": _normalize_text(row.get("queue_id")),
        "domain": domain,
        "question": _normalize_text(row.get("question")),
        "expectation": expectation,
        "operator_status": _normalize_text(row.get("operator_status")),
        "validation_status": _normalize_text(row.get("validation_status")),
        "apply_eligible": bool(row.get("apply_eligible")),
        "skip_reason": _normalize_text(row.get("skip_reason")),
    }
    if expectation == "recovered":
        base.update(
            {
                "expected_strategy": "operator_data_fast_path",
                "expected_source_tier": "operator_data",
                "expected_operator_data_domain": domain,
            }
        )
    else:
        base.update(
            {
                "expected_status": "operator_data_required_or_expected_non_answer",
                "manual_contract_allowed": True,
            }
        )
    return base


def build_smoke_set(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    p0_rows = [row for row in rows if _normalize_text(row.get("domain")) in P0_DOMAINS]
    recovered_rows = sorted(
        [row for row in p0_rows if bool(row.get("apply_eligible"))],
        key=_sort_key,
    )
    manual_rows = sorted(
        [row for row in p0_rows if not bool(row.get("apply_eligible"))],
        key=_sort_key,
    )
    expectations = [
        *[_expectation_for_row(row, "recovered") for row in recovered_rows],
        *[_expectation_for_row(row, "manual_control") for row in manual_rows],
    ]
    warnings: list[str] = []
    if not recovered_rows:
        warnings.append("no_apply_eligible_p0_rows_for_recovered_smoke")

    domain_counts = Counter(_normalize_text(row.get("domain")) for row in p0_rows)
    return {
        "summary": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_p0_rows": len(p0_rows),
            "recovered_count": len(recovered_rows),
            "manual_control_count": len(manual_rows),
            "domain_counts": {
                domain: domain_counts.get(domain, 0) for domain in P0_DOMAIN_ORDER
            },
            "warnings": warnings,
        },
        "expectations": expectations,
        "recovered_questions": [
            _normalize_text(row.get("question")) for row in recovered_rows
        ],
        "manual_control_questions": [
            _normalize_text(row.get("question")) for row in manual_rows
        ],
        "all_questions": [
            *[_normalize_text(row.get("question")) for row in recovered_rows],
            *[_normalize_text(row.get("question")) for row in manual_rows],
        ],
    }


def _write_questions(path: Path, questions: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{question}\n" for question in questions if _normalize_text(question)),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_files(
    *,
    normalized_path: Path,
    output_dir: Path,
    recovered_output: Path,
    manual_output: Path,
    all_output: Path,
    expectations_output: Optional[Path] = None,
) -> dict[str, Any]:
    result = build_smoke_set(_load_jsonl(normalized_path))
    expectations_path = (
        expectations_output or output_dir / "operator_data_p0_smoke_expectations.json"
    )
    payload = {
        "summary": {
            **result["summary"],
            "normalized_path": str(normalized_path),
            "recovered_question_file": str(recovered_output),
            "manual_control_question_file": str(manual_output),
            "all_question_file": str(all_output),
        },
        "expectations": result["expectations"],
    }
    _write_questions(recovered_output, result["recovered_questions"])
    _write_questions(manual_output, result["manual_control_questions"])
    _write_questions(all_output, result["all_questions"])
    _write_json(expectations_path, payload)
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P0 operator-data smoke sets.")
    parser.add_argument("--normalized", default=str(DEFAULT_NORMALIZED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--recovered-output", default=str(DEFAULT_RECOVERED_OUTPUT))
    parser.add_argument("--manual-output", default=str(DEFAULT_MANUAL_OUTPUT))
    parser.add_argument("--all-output", default=str(DEFAULT_ALL_OUTPUT))
    parser.add_argument("--expectations-output", default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = build_files(
        normalized_path=Path(args.normalized),
        output_dir=Path(args.output_dir),
        recovered_output=Path(args.recovered_output),
        manual_output=Path(args.manual_output),
        all_output=Path(args.all_output),
        expectations_output=(
            Path(args.expectations_output) if args.expectations_output else None
        ),
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
