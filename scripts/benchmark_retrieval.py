#!/usr/bin/env python3
"""Offline A/B benchmark for document retrieval chunking quality."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.core.chunking import legacy_window_chunks, smart_chunks
from app.core.embeddings import async_embed_query, async_embed_texts
from scripts.ingest_from_kbo import (
    TABLE_PROFILES,
    build_static_source_row_id,
    read_static_profile_content,
)

DOCUMENT_SOURCE_TABLES = ("markdown_docs", "kbo_regulations", "kbo_definitions")


@dataclass(frozen=True)
class BenchmarkCase:
    category: str
    query: str
    filters: Dict[str, Any] = field(default_factory=dict)
    expected_top_table: Optional[str] = None
    expected_tool: Optional[str] = None
    expected_substrings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorpusChunk:
    source_table: str
    source_profile: str
    source_row_id: str
    title: str
    content: str
    embedding: Sequence[float]


@dataclass
class VariantMetrics:
    source_tables: List[str]
    top1_source_table_hit: bool
    answerable: bool
    zero_hit: bool
    retrieval_latency_ms: float
    result_count: int
    error: Optional[str] = None


DEFAULT_CASES: List[BenchmarkCase] = [
    BenchmarkCase(
        category="markdown_docs",
        query="플래툰 전략이 뭐야?",
        expected_top_table="markdown_docs",
        expected_substrings=("플래툰",),
    ),
    BenchmarkCase(
        category="markdown_docs",
        query="잠실 구장의 상징성과 응원 문화 알려줘",
        expected_top_table="markdown_docs",
        expected_substrings=("잠실", "응원"),
    ),
    BenchmarkCase(
        category="markdown_docs",
        query="BABIP와 FIP 차이 알려줘",
        expected_top_table="markdown_docs",
        expected_substrings=("BABIP", "FIP"),
    ),
    BenchmarkCase(
        category="kbo_regulations",
        query="정규시즌 타이브레이크 규정 알려줘",
        expected_top_table="kbo_regulations",
        expected_substrings=("타이브레이크",),
    ),
    BenchmarkCase(
        category="kbo_regulations",
        query="와일드카드 결정전 방식 알려줘",
        expected_top_table="kbo_regulations",
        expected_substrings=("와일드카드",),
    ),
    BenchmarkCase(
        category="kbo_regulations",
        query="웨이버 제도가 뭐야?",
        expected_top_table="kbo_regulations",
        expected_substrings=("웨이버",),
    ),
    BenchmarkCase(
        category="kbo_definitions",
        query="player_season_batting의 extra_stats 컬럼은 뭐야?",
        expected_top_table="kbo_definitions",
        expected_substrings=("extra_stats",),
    ),
    BenchmarkCase(
        category="kbo_definitions",
        query="ops 컬럼은 어떤 의미야?",
        expected_top_table="kbo_definitions",
        expected_substrings=("ops", "출루율"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs candidate retrieval quality for document corpora."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Top-k retrieval limit per query.",
    )
    parser.add_argument(
        "--variant",
        choices=("both", "baseline", "candidate"),
        default="both",
        help="Run both variants or a single variant.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def _build_variant_filters(
    case: BenchmarkCase,
    *,
    exclude_source_tables: tuple[str, ...] = (),
) -> Dict[str, Any]:
    filters = dict(case.filters)
    if exclude_source_tables:
        filters["_exclude_source_tables"] = list(exclude_source_tables)
    return filters


def _iter_document_profiles() -> Iterable[tuple[str, Dict[str, Any]]]:
    for profile_key, profile in TABLE_PROFILES.items():
        if not profile.get("source_file"):
            continue
        if str(profile.get("source_table", profile_key)) not in DOCUMENT_SOURCE_TABLES:
            continue
        yield profile_key, profile


def _chunk_static_profile_content(
    profile_key: str,
    profile: Dict[str, Any],
    *,
    variant: str,
    settings: Any,
) -> List[Dict[str, Any]]:
    content = read_static_profile_content(profile_key, profile)
    normalized = content.strip()
    if not normalized:
        return []

    if profile.get("single_chunk"):
        chunks = [normalized]
    elif variant == "baseline":
        chunks = legacy_window_chunks(content)
    else:
        chunks = smart_chunks(content, settings=settings)

    total_chunks = len(chunks)
    source_table = str(profile.get("source_table", profile_key))
    title = str(profile.get("title") or profile_key)
    rows: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        rows.append(
            {
                "source_table": source_table,
                "source_profile": profile_key,
                "source_row_id": build_static_source_row_id(
                    profile_key,
                    profile,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                ),
                "title": title,
                "content": chunk,
            }
        )
    return rows


async def _build_variant_corpus(
    *,
    variant: str,
    settings: Any,
) -> List[CorpusChunk]:
    rows: List[Dict[str, Any]] = []
    for profile_key, profile in _iter_document_profiles():
        rows.extend(
            _chunk_static_profile_content(
                profile_key,
                profile,
                variant=variant,
                settings=settings,
            )
        )

    if not rows:
        return []

    embeddings = await async_embed_texts(
        [row["content"] for row in rows],
        settings,
    )
    return [
        CorpusChunk(
            source_table=row["source_table"],
            source_profile=row["source_profile"],
            source_row_id=row["source_row_id"],
            title=row["title"],
            content=row["content"],
            embedding=embedding,
        )
        for row, embedding in zip(rows, embeddings)
    ]


def _cosine_similarity(
    query_embedding: Sequence[float],
    doc_embedding: Sequence[float],
) -> float:
    if not query_embedding or not doc_embedding:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(query_embedding, doc_embedding))
    query_norm = math.sqrt(sum(float(value) * float(value) for value in query_embedding))
    doc_norm = math.sqrt(sum(float(value) * float(value) for value in doc_embedding))
    if query_norm == 0.0 or doc_norm == 0.0:
        return 0.0
    return dot / (query_norm * doc_norm)


def _apply_filters(
    chunks: Sequence[CorpusChunk],
    filters: Dict[str, Any],
) -> List[CorpusChunk]:
    include_table = filters.get("source_table")
    exclude_tables = set(filters.get("_exclude_source_tables", []))
    filtered: List[CorpusChunk] = []
    for chunk in chunks:
        if include_table and chunk.source_table != include_table:
            continue
        if chunk.source_table in exclude_tables:
            continue
        filtered.append(chunk)
    return filtered


def _contains_expected_substrings(
    docs: Sequence[CorpusChunk],
    expected_substrings: tuple[str, ...],
) -> bool:
    if not expected_substrings:
        return True
    haystack = "\n".join(doc.content for doc in docs)
    return all(token in haystack for token in expected_substrings)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 2)
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return round(ordered[lower_index], 2)
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = position - lower_index
    return round(lower_value + (upper_value - lower_value) * weight, 2)


async def _run_variant(
    case: BenchmarkCase,
    *,
    limit: int,
    chunks: Sequence[CorpusChunk],
    settings: Any,
) -> VariantMetrics:
    filters = _build_variant_filters(case)
    try:
        start = time.perf_counter()
        query_embedding = await async_embed_query(case.query, settings)
        filtered_chunks = _apply_filters(chunks, filters)
        ranked = sorted(
            filtered_chunks,
            key=lambda chunk: _cosine_similarity(query_embedding, chunk.embedding),
            reverse=True,
        )[:limit]
        elapsed_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        return VariantMetrics(
            source_tables=[],
            top1_source_table_hit=False,
            answerable=False,
            zero_hit=True,
            retrieval_latency_ms=0.0,
            result_count=0,
            error=str(exc),
        )

    source_tables = [chunk.source_table for chunk in ranked]
    top1_hit = bool(source_tables) and (
        case.expected_top_table is not None
        and source_tables[0] == case.expected_top_table
    )
    return VariantMetrics(
        source_tables=source_tables,
        top1_source_table_hit=top1_hit,
        answerable=_contains_expected_substrings(ranked, case.expected_substrings),
        zero_hit=not ranked,
        retrieval_latency_ms=round(elapsed_ms, 2),
        result_count=len(ranked),
    )


def _aggregate_variant(rows: Sequence[Dict[str, Any]], variant_name: str) -> Dict[str, Any]:
    variant_rows = [row[variant_name] for row in rows]
    total = len(variant_rows)
    successful = [row for row in variant_rows if not row.get("error")]
    latencies = [float(row["retrieval_latency_ms"]) for row in successful]
    denominator = len(successful) if successful else total
    if denominator == 0:
        denominator = 1
    return {
        "cases": total,
        "error_count": sum(1 for row in variant_rows if row.get("error")),
        "zero_hit_rate": round(
            sum(1 for row in successful if row["zero_hit"]) / denominator,
            4,
        ),
        "top1_source_table_precision": round(
            sum(1 for row in successful if row["top1_source_table_hit"]) / denominator,
            4,
        ),
        "answerable_rate": round(
            sum(1 for row in successful if row["answerable"]) / denominator,
            4,
        ),
        "retrieval_p50_ms": _percentile(latencies, 0.5),
        "retrieval_p95_ms": _percentile(latencies, 0.95),
    }


def _build_acceptance(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_p95 = float(baseline["retrieval_p95_ms"])
    candidate_p95 = float(candidate["retrieval_p95_ms"])
    p95_limit = round(baseline_p95 * 1.15, 2) if baseline_p95 > 0 else candidate_p95
    checks = {
        "zero_hit_not_worse": candidate["zero_hit_rate"] <= baseline["zero_hit_rate"],
        "quality_improved": (
            candidate["top1_source_table_precision"]
            > baseline["top1_source_table_precision"]
            or candidate["answerable_rate"] > baseline["answerable_rate"]
        ),
        "p95_within_budget": baseline_p95 == 0 or candidate_p95 <= p95_limit,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_p95_limit_ms": p95_limit,
    }


def _aggregate_results(
    case_results: List[Dict[str, Any]],
    *,
    variants: Sequence[str],
) -> Dict[str, Any]:
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for result in case_results:
        by_category.setdefault(result["category"], []).append(result)

    summary: Dict[str, Any] = {
        "overall": {},
        "by_category": {},
    }
    for variant in variants:
        summary["overall"][variant] = _aggregate_variant(case_results, variant)

    if "baseline" in variants and "candidate" in variants:
        summary["overall"]["acceptance"] = _build_acceptance(
            summary["overall"]["baseline"],
            summary["overall"]["candidate"],
        )

    for category, rows in by_category.items():
        category_summary: Dict[str, Any] = {}
        for variant in variants:
            category_summary[variant] = _aggregate_variant(rows, variant)
        if "baseline" in variants and "candidate" in variants:
            category_summary["acceptance"] = _build_acceptance(
                category_summary["baseline"],
                category_summary["candidate"],
            )
        summary["by_category"][category] = category_summary

    return summary


async def run_benchmark(
    *,
    limit: int,
    variant: str = "both",
) -> Dict[str, Any]:
    settings = get_settings()
    variants = ["baseline", "candidate"] if variant == "both" else [variant]
    corpora = {
        variant_name: await _build_variant_corpus(
            variant=variant_name,
            settings=settings,
        )
        for variant_name in variants
    }

    report: Dict[str, Any] = {
        "input": {
            "limit": limit,
            "variant": variant,
        },
        "cases": [],
    }
    for case in DEFAULT_CASES:
        row: Dict[str, Any] = {
            "category": case.category,
            "query": case.query,
            "filters": case.filters,
            "expected_top_table": case.expected_top_table,
            "expected_substrings": list(case.expected_substrings),
        }
        for variant_name in variants:
            metrics = await _run_variant(
                case,
                limit=limit,
                chunks=corpora[variant_name],
                settings=settings,
            )
            row[variant_name] = asdict(metrics)
        report["cases"].append(row)

    report["summary"] = _aggregate_results(report["cases"], variants=variants)
    return report


def main() -> int:
    args = parse_args()
    report = asyncio.run(run_benchmark(limit=args.limit, variant=args.variant))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote report: {output_path}")

    variants = ["baseline", "candidate"] if args.variant == "both" else [args.variant]
    has_errors = any(
        case[variant_name].get("error")
        for case in report["cases"]
        for variant_name in variants
    )
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
