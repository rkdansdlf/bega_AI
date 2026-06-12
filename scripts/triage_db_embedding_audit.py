#!/usr/bin/env python3
"""Read-only triage for DB/embedding consistency audit reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUAL_CONTRACT = "MANUAL_BASEBALL_DATA_REQUIRED"

ACTION_FIELDNAMES = [
    "priority",
    "action_type",
    "source_report",
    "table",
    "year",
    "source_table",
    "source_row_id",
    "finding_type",
    "recommended_command",
    "manual_contract",
    "notes",
]

CRITICAL_ACTION_TYPES = {
    "manual_data_required",
    "reembed_missing_rows",
    "reembed_source_refresh",
    "reembed_missing_embedding",
}


@dataclass(frozen=True)
class TriageAction:
    priority: str
    action_type: str
    source_report: str
    table: str = ""
    year: str = ""
    source_table: str = ""
    source_row_id: str = ""
    finding_type: str = ""
    recommended_command: str = ""
    manual_contract: str = ""
    notes: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify DB/embedding audit report findings into read-only action buckets."
    )
    parser.add_argument("--coverage-report", default="")
    parser.add_argument("--source-drift-report", default="")
    parser.add_argument("--storage-report", default="")
    parser.add_argument("--embedding-256-report", default="")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument(
        "--fail-on",
        choices=["never", "critical", "any"],
        default="never",
    )
    return parser


def resolve_output_dir(raw_output_dir: str) -> Path:
    output_dir = Path(raw_output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def report_path(raw_path: str) -> Optional[Path]:
    if not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def timestamp(value: str = "") -> str:
    return value.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_report(path: Optional[Path], label: str, warnings: List[str]) -> Dict[str, Any]:
    if path is None:
        warnings.append(f"{label}: report path not provided")
        return {}
    if not path.exists():
        warnings.append(f"{label}: report not found: {path}")
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: malformed JSON in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{label}: JSON root must be an object: {path}")
    return loaded


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _sample(values: Sequence[Any], limit: int) -> List[Any]:
    if limit <= 0:
        return []
    return list(values[:limit])


def _season_arg(year: str) -> str:
    if year and year != "0":
        return f" --season-year {year}"
    return ""


def _table_arg(source_table: str) -> str:
    return f" --source-table {source_table}" if source_table else ""


def _reembed_command(table: str, year: str) -> str:
    return (
        ".venv/bin/python scripts/reembed_missing_rows.py "
        "--report-path <coverage-report> "
        f"--start-year {year or '<start-year>'} --end-year {year or '<end-year>'}"
    )


def _backfill_command(source_table: str, year: str) -> str:
    return (
        ".venv/bin/python scripts/backfill_rag_chunk_metadata.py --dry-run"
        f"{_table_arg(source_table)}{_season_arg(year)}"
    )


def _cleanup_command() -> str:
    return ".venv/bin/python scripts/cleanup_embedding_256_warnings.py --dry-run"


def _source_refresh_command(source_table: str, year: str) -> str:
    year_args = f" --start-year {year} --end-year {year}" if year and year != "0" else ""
    return (
        ".venv/bin/python scripts/audit_rag_chunk_source_drift.py "
        "--mode all --sample-limit 20"
        f"{year_args}"
        + (
            "  # then re-ingest/re-embed only after operator approval"
            if source_table
            else "  # inspect source drift before any re-ingest"
        )
    )


def action_from_source_drift_finding(
    finding: Mapping[str, Any],
) -> TriageAction:
    finding_type = _string(finding.get("type"))
    table = _string(finding.get("table"))
    year = _string(finding.get("year"))
    source_table = _string(finding.get("source_table"))
    source_row_id = _string(finding.get("source_row_id"))

    if finding_type == "missing_active_chunk":
        return TriageAction(
            priority="P1",
            action_type="reembed_missing_rows",
            source_report="source_drift",
            table=table,
            year=year,
            source_table=source_table,
            source_row_id=source_row_id,
            finding_type=finding_type,
            recommended_command=_reembed_command(table, year),
            notes="Expected source row has no active rag_chunks row.",
        )
    if finding_type in {"content_hash_mismatch", "chunk_hash_mismatch"}:
        return TriageAction(
            priority="P1",
            action_type="reembed_source_refresh",
            source_report="source_drift",
            table=table,
            year=year,
            source_table=source_table,
            source_row_id=source_row_id,
            finding_type=finding_type,
            recommended_command=_source_refresh_command(source_table, year),
            notes="Rendered source content no longer matches stored chunk metadata.",
        )
    if finding_type == "embedding_missing":
        return TriageAction(
            priority="P1",
            action_type="reembed_missing_embedding",
            source_report="source_drift",
            table=table,
            year=year,
            source_table=source_table,
            source_row_id=source_row_id,
            finding_type=finding_type,
            recommended_command=_reembed_command(table, year),
            notes="Stored chunk has no embedding vector.",
        )
    if finding_type == "metadata_lineage_missing":
        return TriageAction(
            priority="P2",
            action_type="metadata_backfill",
            source_report="source_drift",
            table=table,
            year=year,
            source_table=source_table,
            source_row_id=source_row_id,
            finding_type=finding_type,
            recommended_command=_backfill_command(source_table, year),
            notes="Stored chunk is missing lineage/hash/embedding metadata.",
        )
    if finding_type in {
        "embedding_model_mismatch",
        "embedding_dim_mismatch",
        "embedding_version_mismatch",
    }:
        return TriageAction(
            priority="P2",
            action_type="embedding_metadata_review",
            source_report="source_drift",
            table=table,
            year=year,
            source_table=source_table,
            source_row_id=source_row_id,
            finding_type=finding_type,
            recommended_command=_cleanup_command(),
            notes="Embedding metadata differs from current runtime target.",
        )
    if finding_type == "extra_active_chunk":
        return TriageAction(
            priority="P3",
            action_type="stale_chunk_review",
            source_report="source_drift",
            table=table,
            year=year,
            source_table=source_table,
            source_row_id=source_row_id,
            finding_type=finding_type,
            notes="Active chunk does not map to the current expected source set.",
        )
    if finding_type == "inactive_expected_chunk":
        return TriageAction(
            priority="P2",
            action_type="manual_data_required",
            source_report="source_drift",
            table=table,
            year=year,
            source_table=source_table,
            source_row_id=source_row_id,
            finding_type=finding_type,
            manual_contract=MANUAL_CONTRACT,
            notes="Expected row is present only as inactive; operator should verify source state before repair.",
        )
    return TriageAction(
        priority="P3",
        action_type="review",
        source_report="source_drift",
        table=table,
        year=year,
        source_table=source_table,
        source_row_id=source_row_id,
        finding_type=finding_type,
        notes="Unrecognized source drift finding requires manual review.",
    )


def triage_source_drift(
    report: Mapping[str, Any],
    *,
    sample_limit: int,
) -> List[TriageAction]:
    actions: List[TriageAction] = []
    for row in report.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        findings = row.get("findings") if isinstance(row.get("findings"), list) else []
        sampled_findings = _sample(findings, sample_limit)
        for finding in sampled_findings:
            if isinstance(finding, Mapping):
                actions.append(action_from_source_drift_finding(finding))
    return actions


def triage_coverage(
    report: Mapping[str, Any],
    *,
    sample_limit: int,
) -> List[TriageAction]:
    actions: List[TriageAction] = []
    for row in report.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        status = _string(row.get("status"))
        if status == "OK":
            continue
        table = _string(row.get("table"))
        year = _string(row.get("year"))
        source_table = _string(row.get("source_table") or table)
        missing_samples = [
            _string(value)
            for value in (row.get("missing_samples") or [])
            if _string(value)
        ]
        extra_samples = [
            _string(value)
            for value in (row.get("extra_samples") or [])
            if _string(value)
        ]
        if "MISSING" in status or _int(row.get("missing_count")) > 0:
            samples = _sample(missing_samples, sample_limit) or [""]
            for source_row_id in samples:
                actions.append(
                    TriageAction(
                        priority="P1",
                        action_type="reembed_missing_rows",
                        source_report="coverage",
                        table=table,
                        year=year,
                        source_table=source_table,
                        source_row_id=source_row_id,
                        finding_type="coverage_missing",
                        recommended_command=_reembed_command(table, year),
                        notes=(
                            f"Coverage expected={row.get('expected_count')} "
                            f"present={row.get('present_count')} missing={row.get('missing_count')}."
                        ),
                    )
                )
        if "EXTRA" in status or _int(row.get("extra_count")) > 0:
            samples = _sample(extra_samples, sample_limit) or [""]
            for source_row_id in samples:
                actions.append(
                    TriageAction(
                        priority="P3",
                        action_type="stale_chunk_review",
                        source_report="coverage",
                        table=table,
                        year=year,
                        source_table=source_table,
                        source_row_id=source_row_id,
                        finding_type="coverage_extra",
                        notes=(
                            f"Coverage actual={row.get('actual_count')} "
                            f"present={row.get('present_count')} extra={row.get('extra_count')}."
                        ),
                    )
                )
    return actions


def triage_storage(report: Mapping[str, Any]) -> List[TriageAction]:
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        return []
    metadata_fields = [
        "missing_topic_key",
        "missing_source_type",
        "missing_source_uri",
        "missing_quality_score",
        "missing_content_hash",
        "missing_chunk_hash",
        "missing_embedding_model",
        "missing_embedding_dim",
        "missing_embedding_version",
        "missing_chunking_version",
        "empty_metadata",
        "metadata_backfill_needed",
    ]
    missing_total = sum(_int(summary.get(field)) for field in metadata_fields)
    actions: List[TriageAction] = []
    if missing_total > 0:
        actions.append(
            TriageAction(
                priority="P2",
                action_type="metadata_backfill",
                source_report="storage",
                finding_type="storage_metadata_missing",
                recommended_command=".venv/bin/python scripts/backfill_rag_chunk_metadata.py --dry-run",
                notes=f"Storage audit found {missing_total} missing lineage/hash/metadata signals.",
            )
        )
    missing_embedding = _int(summary.get("missing_embedding"))
    if missing_embedding > 0:
        actions.append(
            TriageAction(
                priority="P1",
                action_type="reembed_missing_embedding",
                source_report="storage",
                finding_type="storage_missing_embedding",
                notes=f"Storage audit found {missing_embedding} chunks with NULL embeddings.",
            )
        )
    if _int(summary.get("active_expired")) > 0 or _int(summary.get("active_past_valid_to")) > 0:
        actions.append(
            TriageAction(
                priority="P3",
                action_type="stale_chunk_review",
                source_report="storage",
                finding_type="storage_active_expired",
                notes="Active rows are past valid_to/expires_at and should be reviewed before cleanup.",
            )
        )
    return actions


def triage_embedding_256(report: Mapping[str, Any]) -> List[TriageAction]:
    findings = report.get("findings")
    if not isinstance(findings, list):
        return []
    actions: List[TriageAction] = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        severity = _string(finding.get("severity"))
        code = _string(finding.get("code"))
        if severity not in {"warning", "fail"}:
            continue
        action_type = "embedding_metadata_review"
        command = _cleanup_command()
        priority = "P2" if severity == "warning" else "P1"
        if code in {"metadata_conflicts", "runtime_embedding_model", "runtime_embed_dim"}:
            action_type = "manual_data_required"
            command = ""
        actions.append(
            TriageAction(
                priority=priority,
                action_type=action_type,
                source_report="embedding_256",
                finding_type=code,
                recommended_command=command,
                manual_contract=MANUAL_CONTRACT if action_type == "manual_data_required" else "",
                notes=_string(finding.get("message")),
            )
        )
    return actions


def summarize_actions(
    actions: Sequence[TriageAction],
    *,
    warnings: Sequence[str],
) -> Dict[str, Any]:
    by_type = Counter(action.action_type for action in actions)
    by_priority = Counter(action.priority for action in actions)
    critical_count = sum(
        1 for action in actions if action.action_type in CRITICAL_ACTION_TYPES
    )
    return {
        "status": "needs_action" if actions else "clear",
        "action_count": len(actions),
        "critical_count": critical_count,
        "by_action_type": dict(sorted(by_type.items())),
        "by_priority": dict(sorted(by_priority.items())),
        "warning_count": len(warnings),
    }


def action_to_row(action: TriageAction) -> Dict[str, str]:
    return {key: _string(value) for key, value in asdict(action).items()}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, actions: Sequence[TriageAction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTION_FIELDNAMES)
        writer.writeheader()
        for action in actions:
            writer.writerow(action_to_row(action))


def render_markdown_handoff(
    *,
    generated_at: str,
    summary: Mapping[str, Any],
    actions: Sequence[TriageAction],
    warnings: Sequence[str],
    sample_limit: int,
) -> str:
    grouped: Dict[str, List[TriageAction]] = defaultdict(list)
    for action in actions:
        grouped[action.action_type].append(action)

    lines = [
        "# DB Embedding Audit Triage",
        "",
        f"- generated_at_utc: {generated_at}",
        f"- status: {summary.get('status')}",
        f"- action_count: {summary.get('action_count')}",
        f"- critical_count: {summary.get('critical_count')}",
        "",
        "## Action Summary",
        "",
    ]
    for action_type, count in (summary.get("by_action_type") or {}).items():
        lines.append(f"- {action_type}: {count}")
    if not actions:
        lines.append("- No action candidates found.")

    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    for action_type in sorted(grouped):
        lines.extend(["", f"## {action_type}", ""])
        for action in grouped[action_type][:sample_limit]:
            target = " ".join(
                part
                for part in [
                    action.table,
                    action.year,
                    action.source_table,
                    action.source_row_id,
                ]
                if part
            )
            target_text = target or action.finding_type or action.source_report
            lines.append(
                f"- [{action.priority}] {target_text}: {action.notes or action.finding_type}"
            )
            if action.manual_contract:
                lines.append(f"  manual_contract: `{action.manual_contract}`")
            if action.recommended_command:
                lines.append(f"  recommended_command: `{action.recommended_command}`")
    return "\n".join(lines) + "\n"


def exit_code_for_fail_on(fail_on: str, actions: Sequence[TriageAction]) -> int:
    if fail_on == "never":
        return 0
    if fail_on == "any":
        return 1 if actions else 0
    if fail_on == "critical":
        return (
            1
            if any(action.action_type in CRITICAL_ACTION_TYPES for action in actions)
            else 0
        )
    return 0


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.sample_limit < 0:
        raise ValueError("sample-limit must be >= 0")
    warnings: List[str] = []
    reports = {
        "coverage": load_report(
            report_path(args.coverage_report), "coverage", warnings
        ),
        "source_drift": load_report(
            report_path(args.source_drift_report), "source_drift", warnings
        ),
        "storage": load_report(report_path(args.storage_report), "storage", warnings),
        "embedding_256": load_report(
            report_path(args.embedding_256_report), "embedding_256", warnings
        ),
    }
    actions: List[TriageAction] = []
    actions.extend(
        triage_coverage(reports["coverage"], sample_limit=args.sample_limit)
    )
    actions.extend(
        triage_source_drift(
            reports["source_drift"],
            sample_limit=args.sample_limit,
        )
    )
    actions.extend(triage_storage(reports["storage"]))
    actions.extend(triage_embedding_256(reports["embedding_256"]))

    generated_at = datetime.now(timezone.utc).isoformat()
    summary = summarize_actions(actions, warnings=warnings)
    return {
        "generated_at_utc": generated_at,
        "triage": "db_embedding_audit",
        "input": {
            "coverage_report": args.coverage_report,
            "source_drift_report": args.source_drift_report,
            "storage_report": args.storage_report,
            "embedding_256_report": args.embedding_256_report,
            "sample_limit": args.sample_limit,
            "fail_on": args.fail_on,
        },
        "summary": summary,
        "warnings": list(warnings),
        "actions": [action_to_row(action) for action in actions],
        "_action_objects": actions,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    stamp = timestamp(args.timestamp)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"db_embedding_triage_{stamp}.json"
    csv_path = output_dir / f"db_embedding_triage_actions_{stamp}.csv"
    md_path = output_dir / f"db_embedding_triage_handoff_{stamp}.md"
    actions = list(result.pop("_action_objects"))

    write_json(json_path, result)
    write_csv(csv_path, actions)
    md = render_markdown_handoff(
        generated_at=_string(result.get("generated_at_utc")),
        summary=result["summary"],
        actions=actions,
        warnings=result["warnings"],
        sample_limit=args.sample_limit,
    )
    md_path.write_text(md, encoding="utf-8")

    print(f"triage json saved: {json_path}")
    print(f"triage csv saved: {csv_path}")
    print(f"triage handoff saved: {md_path}")
    print(f"status: {result['summary']['status']}")
    return exit_code_for_fail_on(args.fail_on, actions)


if __name__ == "__main__":
    raise SystemExit(main())
