#!/usr/bin/env python3
"""Verify P0 operator-data smoke reports against generated expectations."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "operator_data_smoke"
P0_DOMAINS = {"season_meta", "schedule_window", "game_day_lineup", "roster_news"}
EXPECTED_ENDPOINTS = ("/ai/chat/completion", "/ai/chat/stream")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be an object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _nested_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _metadata(item: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    sample = item.get("sample_response")
    if isinstance(sample, Mapping):
        merged.update(sample)
        nested = sample.get("metadata")
        if isinstance(nested, Mapping):
            merged.update(nested)
    meta = item.get("meta")
    if isinstance(meta, Mapping):
        merged.update(meta)
    return merged


def _answerability(item: Mapping[str, Any]) -> dict[str, Any]:
    return _nested_dict(item.get("answerability"))


def _result_index(
    results: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in results:
        question = _normalize_text(item.get("question"))
        endpoint = _normalize_text(item.get("endpoint"))
        if question and endpoint:
            index[(question, endpoint)] = item
    return index


def _failure(
    *,
    question: str,
    endpoint: str,
    expectation: str,
    code: str,
    message: str,
    queue_id: str = "",
    domain: str = "",
) -> dict[str, Any]:
    return {
        "question": question,
        "endpoint": endpoint,
        "expectation": expectation,
        "code": code,
        "message": message,
        "queue_id": queue_id,
        "domain": domain,
    }


def _verify_recovered(
    *,
    item: Mapping[str, Any],
    expectation: Mapping[str, Any],
    endpoint: str,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    question = _normalize_text(expectation.get("question"))
    queue_id = _normalize_text(expectation.get("queue_id"))
    expected_domain = _normalize_text(expectation.get("expected_operator_data_domain"))
    metadata = _metadata(item)
    answer = _normalize_text(item.get("answer"))
    strategy = _normalize_text(metadata.get("strategy"))
    source_tier = _normalize_text(metadata.get("source_tier"))
    domain = _normalize_text(metadata.get("operator_data_domain"))

    def add(code: str, message: str) -> None:
        failures.append(
            _failure(
                question=question,
                endpoint=endpoint,
                expectation="recovered",
                code=code,
                message=message,
                queue_id=queue_id,
                domain=expected_domain,
            )
        )

    if item.get("ok") is not True:
        add(
            "smoke_item_failed", "Smoke item did not pass quality/answerability checks."
        )
    if strategy != "operator_data_fast_path":
        add(
            "wrong_strategy",
            f"Expected operator_data_fast_path, got {strategy or 'missing'}.",
        )
    if source_tier != "operator_data":
        add(
            "wrong_source_tier",
            f"Expected source_tier=operator_data, got {source_tier or 'missing'}.",
        )
    if domain not in P0_DOMAINS or domain != expected_domain:
        add(
            "wrong_operator_data_domain",
            f"Expected domain={expected_domain}, got {domain or 'missing'}.",
        )
    if "MANUAL_BASEBALL_DATA_REQUIRED" in answer:
        add(
            "manual_contract_returned",
            "Recovered expectation returned manual-data contract.",
        )
    return failures


def _verify_manual_control(
    *,
    item: Mapping[str, Any],
    expectation: Mapping[str, Any],
    endpoint: str,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    question = _normalize_text(expectation.get("question"))
    queue_id = _normalize_text(expectation.get("queue_id"))
    domain = _normalize_text(expectation.get("domain"))
    metadata = _metadata(item)
    answer = _normalize_text(item.get("answer"))
    strategy = _normalize_text(metadata.get("strategy"))
    answerability = _answerability(item)
    status = _normalize_text(answerability.get("status"))
    expected_non_answer = answerability.get("expected_non_answer") is True
    manual_contract = "MANUAL_BASEBALL_DATA_REQUIRED" in answer
    allowed_status = status in {
        "operator_data_required",
        "clarification_required",
        "future_event_pending",
    }

    def add(code: str, message: str) -> None:
        failures.append(
            _failure(
                question=question,
                endpoint=endpoint,
                expectation="manual_control",
                code=code,
                message=message,
                queue_id=queue_id,
                domain=domain,
            )
        )

    if strategy == "operator_data_fast_path":
        add(
            "unexpected_operator_fast_path",
            "Manual-control expectation used operator fast-path.",
        )
    if not (manual_contract or expected_non_answer or allowed_status):
        add(
            "manual_contract_not_preserved",
            "Manual-control expectation did not preserve manual or expected non-answer behavior.",
        )
    return failures


def verify_report(
    *,
    smoke_report: Mapping[str, Any],
    expectations_payload: Mapping[str, Any],
) -> dict[str, Any]:
    raw_results = smoke_report.get("results") or []
    if not isinstance(raw_results, list):
        raise ValueError("smoke report must contain a results list")
    results = [item for item in raw_results if isinstance(item, Mapping)]
    result_questions = {_normalize_text(item.get("question")) for item in results}
    result_questions.discard("")
    result_by_key = _result_index(results)

    raw_expectations = expectations_payload.get("expectations") or []
    if not isinstance(raw_expectations, list):
        raise ValueError("expectations payload must contain an expectations list")
    expectations = [
        item
        for item in raw_expectations
        if isinstance(item, Mapping) and _normalize_text(item.get("question"))
    ]
    expected_questions = {
        _normalize_text(item.get("question")) for item in expectations
    }
    failures: list[dict[str, Any]] = []

    for question in sorted(result_questions - expected_questions):
        failures.append(
            _failure(
                question=question,
                endpoint="",
                expectation="unknown",
                code="unexpected_question",
                message="Smoke report contains a question absent from expectations.",
            )
        )

    verified_items = 0
    for expectation in expectations:
        question = _normalize_text(expectation.get("question"))
        kind = _normalize_text(expectation.get("expectation"))
        for endpoint in EXPECTED_ENDPOINTS:
            item = result_by_key.get((question, endpoint))
            if item is None:
                failures.append(
                    _failure(
                        question=question,
                        endpoint=endpoint,
                        expectation=kind,
                        code="endpoint_missing",
                        message="Expected endpoint result is missing from smoke report.",
                        queue_id=_normalize_text(expectation.get("queue_id")),
                        domain=_normalize_text(expectation.get("domain")),
                    )
                )
                continue
            verified_items += 1
            if kind == "recovered":
                failures.extend(
                    _verify_recovered(
                        item=item, expectation=expectation, endpoint=endpoint
                    )
                )
            elif kind == "manual_control":
                failures.extend(
                    _verify_manual_control(
                        item=item, expectation=expectation, endpoint=endpoint
                    )
                )
            else:
                failures.append(
                    _failure(
                        question=question,
                        endpoint=endpoint,
                        expectation=kind,
                        code="unknown_expectation",
                        message=f"Unsupported expectation kind: {kind}",
                        queue_id=_normalize_text(expectation.get("queue_id")),
                        domain=_normalize_text(expectation.get("domain")),
                    )
                )

    status = "pass" if not failures else "fail"
    return {
        "summary": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "report_result_count": len(results),
            "expected_question_count": len(expected_questions),
            "verified_item_count": verified_items,
            "failure_count": len(failures),
        },
        "failures": failures,
    }


def run_verification(
    *, report_path: Path, expectations_path: Path, output_dir: Path
) -> dict[str, Any]:
    report = verify_report(
        smoke_report=_read_json(report_path),
        expectations_payload=_read_json(expectations_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "operator_data_p0_smoke_verification_summary.json", report)
    _write_csv(
        output_dir / "operator_data_p0_smoke_verification_failures.csv",
        report["failures"],
        [
            "question",
            "endpoint",
            "expectation",
            "code",
            "message",
            "queue_id",
            "domain",
        ],
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify P0 operator-data smoke reports."
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--expectations", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        default=True,
        help="Exit 0 even when smoke verification fails.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_verification(
        report_path=Path(args.report),
        expectations_path=Path(args.expectations),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.strict and report["summary"]["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
