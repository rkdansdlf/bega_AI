#!/usr/bin/env python3
"""Read-only audit for source DB to rag_chunks content and embedding drift."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("BEGA_SKIP_APP_INIT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psycopg
    from psycopg.rows import dict_row
except ModuleNotFoundError as exc:
    psycopg = None
    dict_row = None
    _PSYCOPG_IMPORT_ERROR = exc

from app.config import get_settings
from app.core.rag_storage import (
    build_chunk_storage_fields,
    chunk_hash as build_chunk_hash,
    infer_chunk_index,
    is_search_worthy_content,
    resolve_chunking_version,
    resolve_embedding_dim,
    resolve_embedding_model,
    resolve_embedding_version,
    scan_sensitive_content,
)
from scripts.ingest_from_kbo import (
    ChunkPayload,
    TABLE_PROFILES,
    _build_chunk_payload_dicts_for_row,
    build_select_query,
    build_static_profile_chunk_payloads,
    resolve_primary_key_columns,
)
from scripts.sync_rag_chunks import _load_settings_from_env_file
from scripts.verify_embedding_coverage import (
    CoverageTarget,
    _build_legacy_from_mapping,
    build_expected_source_row_id,
    build_targets,
    normalize_actual_source_row_id,
)

PGVECTOR_SEARCH_PATH = "public, extensions, security"

FINDING_TYPES = (
    "missing_active_chunk",
    "inactive_expected_chunk",
    "extra_active_chunk",
    "content_hash_mismatch",
    "chunk_hash_mismatch",
    "embedding_missing",
    "embedding_model_mismatch",
    "embedding_dim_mismatch",
    "embedding_version_mismatch",
    "metadata_lineage_missing",
)

LINEAGE_FIELDS = (
    "source_type",
    "source_uri",
    "topic_key",
    "content_hash",
    "chunk_hash",
    "embedding_model",
    "embedding_dim",
    "embedding_version",
    "chunking_version",
)


@dataclass(frozen=True)
class ExpectedChunk:
    table: str
    source_table: str
    source_row_id: str
    title: str
    content: str
    content_hash: str
    chunk_hash: str
    embedding_model: str
    embedding_dim: int
    embedding_version: int
    chunking_version: int
    source_type: str
    source_uri: str
    topic_key: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ActualChunk:
    source_table: str
    source_row_id: str
    normalized_source_row_id: str
    title: str
    content: str
    content_hash: Optional[str]
    chunk_hash: Optional[str]
    embedding_model: Optional[str]
    embedding_dim: Optional[int]
    embedding_version: Optional[int]
    chunking_version: Optional[int]
    source_type: Optional[str]
    source_uri: Optional[str]
    topic_key: Optional[str]
    metadata: Dict[str, Any]
    meta: Dict[str, Any]
    is_active: bool
    embedding_present: bool
    updated_at: Any = None


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required to run audit_rag_chunk_source_drift.py. "
            "Install dependencies and retry. "
            f"Detail: {_PSYCOPG_IMPORT_ERROR}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit that compares source DB rendered chunks with "
            "active rag_chunks rows."
        )
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--mode",
        choices=["all", "seasonal", "static"],
        default="all",
        help="Audit scope.",
    )
    parser.add_argument(
        "--source-env-file",
        default="",
        help="Env file used to resolve source DB config when --source-db-url is omitted.",
    )
    parser.add_argument(
        "--source-db-url",
        default="",
        help="Source PostgreSQL URL override.",
    )
    parser.add_argument(
        "--dest-env-file",
        default="",
        help="Env file used to resolve RAG DB config when --dest-db-url is omitted.",
    )
    parser.add_argument(
        "--dest-db-url",
        default="",
        help="Destination/RAG PostgreSQL URL override.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Max findings retained per target and finding type.",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=500,
        help="Source DB fetch batch size.",
    )
    parser.add_argument(
        "--today",
        default="",
        help="Renderer date override in YYYY-MM-DD. Defaults to current UTC date.",
    )
    parser.add_argument(
        "--use-legacy-renderer",
        action="store_true",
        help="Use ingest_from_kbo legacy label/value renderer.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--csv-output",
        default="",
        help="Optional CSV findings output path.",
    )
    return parser


def resolve_db_urls(args: argparse.Namespace) -> tuple[str, str]:
    settings = get_settings()
    source_db_url = args.source_db_url.strip()
    dest_db_url = args.dest_db_url.strip()

    if not source_db_url:
        if args.source_env_file.strip():
            source_db_url = _load_settings_from_env_file(
                args.source_env_file
            ).database_url
        else:
            source_db_url = settings.database_url

    if not dest_db_url:
        if args.dest_env_file.strip():
            dest_db_url = _load_settings_from_env_file(args.dest_env_file).database_url
        else:
            dest_db_url = settings.database_url

    return source_db_url, dest_db_url


def build_audit_targets(
    mode: str,
    start_year: int,
    end_year: int,
) -> List[CoverageTarget]:
    return build_targets(mode, start_year, end_year)


def resolve_report_path(raw_path: str, default_path: Path) -> Path:
    if not raw_path.strip():
        return default_path
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def runtime_snapshot(settings: Any, *, today_str: str) -> Dict[str, Any]:
    return {
        "today": today_str,
        "embed_provider": getattr(settings, "embed_provider", None),
        "resolved_embedding_model": resolve_embedding_model(settings),
        "embed_dim": resolve_embedding_dim(settings),
        "rag_embedding_version": resolve_embedding_version(settings),
        "rag_chunking_version": resolve_chunking_version(settings),
        "rag_quality_min_chars": int(
            getattr(settings, "rag_quality_min_chars", 50) or 50
        ),
    }


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _chunk_payload_to_expected(
    payload: ChunkPayload,
    *,
    settings: Any,
) -> tuple[Optional[ExpectedChunk], Optional[str]]:
    min_chars = int(getattr(settings, "rag_quality_min_chars", 50) or 50)
    findings = sorted(
        set(
            scan_sensitive_content(payload.content)
            + scan_sensitive_content(payload.meta)
        )
    )
    if findings:
        return None, "sensitive_content"
    if not is_search_worthy_content(payload.content, min_chars=min_chars):
        return None, "low_value_content"

    storage_fields = build_chunk_storage_fields(
        settings=settings,
        source_table=payload.table,
        source_row_id=payload.source_row_id,
        content=payload.content,
        meta=payload.meta,
        source_type=payload.source_type,
        source_uri=payload.source_uri,
        topic_key=payload.topic_key,
        valid_from=payload.valid_from,
        valid_to=payload.valid_to,
        expires_at=payload.expires_at,
        quality_score=payload.quality_score,
    )
    return (
        ExpectedChunk(
            table=payload.table,
            source_table=payload.table,
            source_row_id=payload.source_row_id,
            title=payload.title,
            content=payload.content,
            content_hash=str(storage_fields["content_hash"]),
            chunk_hash=str(storage_fields["chunk_hash"]),
            embedding_model=str(storage_fields["embedding_model"]),
            embedding_dim=int(storage_fields["embedding_dim"]),
            embedding_version=int(storage_fields["embedding_version"]),
            chunking_version=int(storage_fields["chunking_version"]),
            source_type=str(storage_fields["source_type"]),
            source_uri=str(storage_fields["source_uri"]),
            topic_key=str(storage_fields["topic_key"]),
            metadata=dict(storage_fields["metadata"]),
        ),
        None,
    )


def iter_static_expected_chunks(
    target: CoverageTarget,
    *,
    settings: Any,
) -> Iterable[tuple[Optional[ExpectedChunk], Optional[str]]]:
    profile = TABLE_PROFILES.get(target.table, {})
    for payload in build_static_profile_chunk_payloads(
        target.table,
        profile,
        settings=settings,
    ):
        yield _chunk_payload_to_expected(payload, settings=settings)


def iter_source_expected_chunks(
    source_conn: Any,
    target: CoverageTarget,
    *,
    settings: Any,
    read_batch_size: int,
    use_legacy_renderer: bool,
    today_str: str,
) -> Iterable[tuple[Optional[ExpectedChunk], Optional[str], Optional[str]]]:
    profile = TABLE_PROFILES.get(target.table, {})
    pk_columns = resolve_primary_key_columns(source_conn, target.table, profile)
    query, params = build_select_query(
        target.table,
        profile,
        pk_columns,
        limit=None,
        season_year=target.year if target.year != 0 else None,
        since=None,
    )
    pk_hint: Sequence[str] = profile.get("pk_hint", [])

    with source_conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        while True:
            rows = cur.fetchmany(read_batch_size)
            if not rows:
                break
            for raw_row in rows:
                row = dict(raw_row)
                source_row_id = build_expected_source_row_id(
                    row=row,
                    target=target,
                    pk_columns=pk_columns,
                    pk_hint=pk_hint,
                )
                legacy_row_id = _build_legacy_from_mapping(row, target.table)
                payloads = _build_chunk_payload_dicts_for_row(
                    table_name=target.table,
                    row=row,
                    source_row_id=source_row_id,
                    use_legacy_renderer=use_legacy_renderer,
                    today_str=today_str,
                )
                for item in payloads:
                    expected, skipped_reason = _chunk_payload_to_expected(
                        ChunkPayload(**item),
                        settings=settings,
                    )
                    yield expected, skipped_reason, legacy_row_id


def load_expected_chunks(
    source_conn: Any,
    target: CoverageTarget,
    *,
    settings: Any,
    read_batch_size: int,
    use_legacy_renderer: bool,
    today_str: str,
) -> tuple[Dict[str, ExpectedChunk], Dict[str, int], Dict[str, str]]:
    expected: Dict[str, ExpectedChunk] = {}
    skipped: Counter[str] = Counter()
    legacy_aliases: Dict[str, str] = {}

    profile = TABLE_PROFILES.get(target.table, {})
    if profile.get("source_file"):
        iterator = (
            (chunk, reason, None)
            for chunk, reason in iter_static_expected_chunks(
                target,
                settings=settings,
            )
        )
    else:
        iterator = iter_source_expected_chunks(
            source_conn,
            target,
            settings=settings,
            read_batch_size=read_batch_size,
            use_legacy_renderer=use_legacy_renderer,
            today_str=today_str,
        )

    for chunk, skipped_reason, legacy_row_id in iterator:
        if skipped_reason:
            skipped[skipped_reason] += 1
            continue
        if chunk is None:
            continue
        expected[chunk.source_row_id] = chunk
        if legacy_row_id and legacy_row_id != chunk.source_row_id:
            legacy_suffix = ""
            expected_suffix = ""
            if "#part" in chunk.source_row_id:
                expected_base, expected_part = chunk.source_row_id.split("#part", 1)
                expected_suffix = f"#part{expected_part}"
            else:
                expected_base = chunk.source_row_id
            legacy_aliases.setdefault(
                legacy_row_id + legacy_suffix,
                expected_base + expected_suffix,
            )

    return expected, dict(skipped), legacy_aliases


def _actual_rows_query(target: CoverageTarget) -> tuple[str, tuple[Any, ...]]:
    query = """
        SELECT
            source_table,
            source_row_id,
            title,
            content,
            content_hash,
            chunk_hash,
            embedding_model,
            embedding_dim,
            embedding_version,
            chunking_version,
            source_type,
            source_uri,
            topic_key,
            metadata,
            meta,
            COALESCE(is_active, true) AS is_active,
            embedding IS NOT NULL AS embedding_present,
            updated_at
        FROM rag_chunks
        WHERE source_table = %s
    """
    params: List[Any] = [target.source_table]
    profile = TABLE_PROFILES.get(target.table, {})
    if profile.get("source_file"):
        from scripts.ingest_from_kbo import build_static_source_row_prefix

        prefix = build_static_source_row_prefix(target.table, profile)
        query += " AND (source_row_id = %s OR source_row_id LIKE %s)"
        params.extend([prefix, f"{prefix}#part%"])
    elif target.year != 0:
        query += " AND season_year = %s"
        params.append(target.year)
    query += " ORDER BY source_row_id, updated_at DESC NULLS LAST"
    return query, tuple(params)


def normalize_actual_chunk_source_row_id(
    raw_source_row_id: str,
    table: str,
    meta: Optional[Dict[str, Any]] = None,
    legacy_aliases: Optional[Dict[str, str]] = None,
) -> str:
    if table in TABLE_PROFILES and TABLE_PROFILES.get(table, {}).get("source_file"):
        return raw_source_row_id
    base = raw_source_row_id
    suffix = ""
    if "#part" in raw_source_row_id:
        base, suffix_part = raw_source_row_id.split("#part", 1)
        suffix = f"#part{suffix_part}"
    normalized_base = normalize_actual_source_row_id(
        base,
        table,
        meta=meta,
        legacy_aliases=legacy_aliases,
    )
    normalized = normalized_base + suffix
    if legacy_aliases:
        return legacy_aliases.get(normalized, normalized)
    return normalized


def fetch_actual_chunks(
    dest_conn: Any,
    target: CoverageTarget,
    *,
    legacy_aliases: Dict[str, str],
) -> List[ActualChunk]:
    query, params = _actual_rows_query(target)
    rows: List[ActualChunk] = []
    with dest_conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        for row in cur.fetchall():
            raw = dict(row)
            meta = _as_dict(raw.get("meta"))
            rows.append(
                ActualChunk(
                    source_table=str(raw.get("source_table") or ""),
                    source_row_id=str(raw.get("source_row_id") or ""),
                    normalized_source_row_id=normalize_actual_chunk_source_row_id(
                        str(raw.get("source_row_id") or ""),
                        target.table,
                        meta=meta,
                        legacy_aliases=legacy_aliases,
                    ),
                    title=str(raw.get("title") or ""),
                    content=str(raw.get("content") or ""),
                    content_hash=raw.get("content_hash"),
                    chunk_hash=raw.get("chunk_hash"),
                    embedding_model=raw.get("embedding_model"),
                    embedding_dim=_as_int(raw.get("embedding_dim")),
                    embedding_version=_as_int(raw.get("embedding_version")),
                    chunking_version=_as_int(raw.get("chunking_version")),
                    source_type=raw.get("source_type"),
                    source_uri=raw.get("source_uri"),
                    topic_key=raw.get("topic_key"),
                    metadata=_as_dict(raw.get("metadata")),
                    meta=meta,
                    is_active=bool(raw.get("is_active")),
                    embedding_present=bool(raw.get("embedding_present")),
                    updated_at=raw.get("updated_at"),
                )
            )
    return rows


def _finding(
    target: CoverageTarget,
    finding_type: str,
    *,
    expected: Optional[ExpectedChunk] = None,
    actual: Optional[ActualChunk] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source_row_id = (
        expected.source_row_id
        if expected is not None
        else actual.normalized_source_row_id if actual is not None else ""
    )
    item: Dict[str, Any] = {
        "type": finding_type,
        "table": target.table,
        "year": target.year,
        "source_table": target.source_table,
        "source_row_id": source_row_id,
    }
    if actual is not None:
        item["actual_source_row_id"] = actual.source_row_id
        item["is_active"] = actual.is_active
    if details:
        item.update(details)
    return item


def _sample_allowed(
    sample_counts: Dict[str, int],
    finding_type: str,
    sample_limit: int,
) -> bool:
    if sample_limit <= 0:
        return False
    current = sample_counts.get(finding_type, 0)
    if current >= sample_limit:
        return False
    sample_counts[finding_type] = current + 1
    return True


def _choose_actual(rows: Sequence[ActualChunk]) -> Optional[ActualChunk]:
    if not rows:
        return None
    active = [row for row in rows if row.is_active]
    if active:
        return active[0]
    return rows[0]


def _missing_lineage_fields(actual: ActualChunk) -> List[str]:
    raw = {
        "source_type": actual.source_type,
        "source_uri": actual.source_uri,
        "topic_key": actual.topic_key,
        "content_hash": actual.content_hash,
        "chunk_hash": actual.chunk_hash,
        "embedding_model": actual.embedding_model,
        "embedding_dim": actual.embedding_dim,
        "embedding_version": actual.embedding_version,
        "chunking_version": actual.chunking_version,
    }
    missing = [
        field
        for field in LINEAGE_FIELDS
        if raw.get(field) is None or str(raw.get(field)).strip() == ""
    ]
    if not actual.metadata:
        missing.append("metadata")
    return missing


def _compatible_chunk_hashes(expected: ExpectedChunk, actual: ActualChunk) -> set[str]:
    hashes = {expected.chunk_hash}
    if actual.source_row_id and actual.source_row_id != expected.source_row_id:
        hashes.add(
            build_chunk_hash(
                source_table=actual.source_table,
                source_row_id=actual.source_row_id,
                chunk_index=infer_chunk_index(actual.source_row_id),
                content_hash_value=expected.content_hash,
            )
        )
    return hashes


def compare_target_chunks(
    target: CoverageTarget,
    *,
    expected_chunks: Dict[str, ExpectedChunk],
    actual_chunks: Sequence[ActualChunk],
    skipped_expected: Optional[Dict[str, int]] = None,
    sample_limit: int,
) -> Dict[str, Any]:
    finding_counts: Counter[str] = Counter()
    finding_samples: List[Dict[str, Any]] = []
    sample_counts: Dict[str, int] = {}
    actual_by_id: Dict[str, List[ActualChunk]] = defaultdict(list)
    for actual in actual_chunks:
        actual_by_id[actual.normalized_source_row_id].append(actual)

    def add(
        finding_type: str,
        *,
        expected: Optional[ExpectedChunk] = None,
        actual: Optional[ActualChunk] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        finding_counts[finding_type] += 1
        if _sample_allowed(sample_counts, finding_type, sample_limit):
            finding_samples.append(
                _finding(
                    target,
                    finding_type,
                    expected=expected,
                    actual=actual,
                    details=details,
                )
            )

    for expected_id, expected in expected_chunks.items():
        actual = _choose_actual(actual_by_id.get(expected_id, []))
        if actual is None:
            add("missing_active_chunk", expected=expected)
            continue
        if not actual.is_active:
            add("inactive_expected_chunk", expected=expected, actual=actual)
            continue

        if (
            actual.content_hash != expected.content_hash
            or actual.content != expected.content
        ):
            add(
                "content_hash_mismatch",
                expected=expected,
                actual=actual,
                details={
                    "expected_content_hash": expected.content_hash,
                    "actual_content_hash": actual.content_hash,
                    "content_text_mismatch": actual.content != expected.content,
                },
            )
        if actual.chunk_hash not in _compatible_chunk_hashes(expected, actual):
            add(
                "chunk_hash_mismatch",
                expected=expected,
                actual=actual,
                details={
                    "expected_chunk_hash": expected.chunk_hash,
                    "actual_chunk_hash": actual.chunk_hash,
                },
            )
        if not actual.embedding_present:
            add("embedding_missing", expected=expected, actual=actual)
        if actual.embedding_model != expected.embedding_model:
            add(
                "embedding_model_mismatch",
                expected=expected,
                actual=actual,
                details={
                    "expected_embedding_model": expected.embedding_model,
                    "actual_embedding_model": actual.embedding_model,
                },
            )
        if actual.embedding_dim != expected.embedding_dim:
            add(
                "embedding_dim_mismatch",
                expected=expected,
                actual=actual,
                details={
                    "expected_embedding_dim": expected.embedding_dim,
                    "actual_embedding_dim": actual.embedding_dim,
                },
            )
        if actual.embedding_version != expected.embedding_version:
            add(
                "embedding_version_mismatch",
                expected=expected,
                actual=actual,
                details={
                    "expected_embedding_version": expected.embedding_version,
                    "actual_embedding_version": actual.embedding_version,
                },
            )
        missing_fields = _missing_lineage_fields(actual)
        if missing_fields:
            add(
                "metadata_lineage_missing",
                expected=expected,
                actual=actual,
                details={"missing_fields": ",".join(missing_fields)},
            )

    for actual in actual_chunks:
        if actual.is_active and actual.normalized_source_row_id not in expected_chunks:
            add("extra_active_chunk", actual=actual)

    counts = {
        finding_type: int(finding_counts.get(finding_type, 0))
        for finding_type in FINDING_TYPES
    }
    return {
        "table": target.table,
        "year": target.year,
        "source_table": target.source_table,
        "expected_chunks": len(expected_chunks),
        "actual_chunks": len(actual_chunks),
        "active_actual_chunks": sum(1 for chunk in actual_chunks if chunk.is_active),
        "skipped_expected": skipped_expected or {},
        "finding_counts": counts,
        "status": "OK" if sum(counts.values()) == 0 else "DRIFT",
        "findings": finding_samples,
    }


def _configure_read_only_session(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on;")
        cur.execute(f"SET search_path TO {PGVECTOR_SEARCH_PATH};")
        cur.execute("SET statement_timeout TO 0;")


def audit_target(
    source_conn: Any,
    dest_conn: Any,
    target: CoverageTarget,
    *,
    settings: Any,
    read_batch_size: int,
    use_legacy_renderer: bool,
    today_str: str,
    sample_limit: int,
) -> Dict[str, Any]:
    expected_chunks, skipped_expected, legacy_aliases = load_expected_chunks(
        source_conn,
        target,
        settings=settings,
        read_batch_size=read_batch_size,
        use_legacy_renderer=use_legacy_renderer,
        today_str=today_str,
    )
    actual_chunks = fetch_actual_chunks(
        dest_conn,
        target,
        legacy_aliases=legacy_aliases,
    )
    return compare_target_chunks(
        target,
        expected_chunks=expected_chunks,
        actual_chunks=actual_chunks,
        skipped_expected=skipped_expected,
        sample_limit=sample_limit,
    )


def summarize_results(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for row in rows:
        totals["expected_chunks"] += int(row.get("expected_chunks") or 0)
        totals["actual_chunks"] += int(row.get("actual_chunks") or 0)
        totals["active_actual_chunks"] += int(row.get("active_actual_chunks") or 0)
        for key, value in (row.get("finding_counts") or {}).items():
            totals[key] += int(value or 0)
        for key, value in (row.get("skipped_expected") or {}).items():
            skipped[key] += int(value or 0)
    drift_total = sum(totals.get(finding_type, 0) for finding_type in FINDING_TYPES)
    return {
        "status": "pass" if drift_total == 0 else "fail",
        "targets": len(rows),
        "drift_total": drift_total,
        "counts": dict(totals),
        "skipped_expected": dict(skipped),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "type",
        "table",
        "year",
        "source_table",
        "source_row_id",
        "actual_source_row_id",
        "is_active",
        "expected_content_hash",
        "actual_content_hash",
        "content_text_mismatch",
        "expected_chunk_hash",
        "actual_chunk_hash",
        "expected_embedding_model",
        "actual_embedding_model",
        "expected_embedding_dim",
        "actual_embedding_dim",
        "expected_embedding_version",
        "actual_embedding_version",
        "missing_fields",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            for finding in row.get("findings", []):
                writer.writerow(finding)


def main() -> int:
    _require_psycopg()
    parser = build_parser()
    args = parser.parse_args()
    if args.start_year > args.end_year:
        print("start-year must be less than or equal to end-year", file=sys.stderr)
        return 1
    if args.sample_limit < 0:
        print("sample-limit must be >= 0", file=sys.stderr)
        return 1
    if args.read_batch_size <= 0:
        print("read-batch-size must be > 0", file=sys.stderr)
        return 1

    settings = get_settings()
    today_str = args.today.strip() or datetime.utcnow().strftime("%Y-%m-%d")
    source_db_url, dest_db_url = resolve_db_urls(args)
    targets = build_audit_targets(args.mode, args.start_year, args.end_year)

    rows: List[Dict[str, Any]] = []
    generated_at = datetime.now(timezone.utc).isoformat()
    default_output_path = (
        PROJECT_ROOT
        / "reports"
        / f"rag_source_drift_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path = resolve_report_path(args.output, default_output_path)
    csv_output_path = (
        resolve_report_path(args.csv_output, PROJECT_ROOT / args.csv_output)
        if args.csv_output.strip()
        else None
    )

    with (
        psycopg.connect(source_db_url) as source_conn,
        psycopg.connect(dest_db_url) as dest_conn,
    ):
        _configure_read_only_session(source_conn)
        _configure_read_only_session(dest_conn)
        for index, target in enumerate(targets, start=1):
            print(
                f"[{index}/{len(targets)}] auditing table={target.table} year={target.year}",
                flush=True,
            )
            row = audit_target(
                source_conn,
                dest_conn,
                target,
                settings=settings,
                read_batch_size=args.read_batch_size,
                use_legacy_renderer=args.use_legacy_renderer,
                today_str=today_str,
                sample_limit=args.sample_limit,
            )
            rows.append(row)
            source_conn.rollback()
            dest_conn.rollback()

    summary = summarize_results(rows)
    payload = {
        "generated_at_utc": generated_at,
        "audit": "rag_chunk_source_drift",
        "input": {
            "mode": args.mode,
            "start_year": args.start_year,
            "end_year": args.end_year,
            "sample_limit": args.sample_limit,
            "read_batch_size": args.read_batch_size,
            "use_legacy_renderer": args.use_legacy_renderer,
        },
        "runtime": runtime_snapshot(settings, today_str=today_str),
        "summary": summary,
        "rows": rows,
    }
    write_json(output_path, payload)
    print(f"summary saved: {output_path}")
    if csv_output_path:
        write_csv(csv_output_path, rows)
        print(f"findings csv saved: {csv_output_path}")
    print(f"status: {summary['status']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
