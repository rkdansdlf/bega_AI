#!/usr/bin/env python3
"""
RAG 파이프라인 단계별 레이턴시 벤치마크.

측정 대상:
  1. 엔티티 추출      (entity_extraction_ms)
  2. 쿼리 변환        (query_transform_ms)
  3. 임베딩 생성      (embedding_ms)
  4. 벡터 검색        (vector_search_ms)
  5. 컨텍스트 포맷팅  (context_format_ms)
  6. 전체 파이프라인  (total_ms, DB+LLM 제외 단계만)

사용법:
  # 오프라인 (DB/LLM 없이 — 임베딩은 local pseudo-vector 사용)
  EMBED_PROVIDER=local PYTHONPATH=. python scripts/benchmark_rag_pipeline.py

  # DB 포함 (벡터 검색 측정)
  EMBED_PROVIDER=local POSTGRES_DB_URL=... python scripts/benchmark_rag_pipeline.py --db

  # 반복 횟수 조정
  python scripts/benchmark_rag_pipeline.py --runs 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

# --- 레이턴시 목표 (SLA) ---
LATENCY_TARGETS_MS = {
    "entity_extraction_ms": 50,
    "query_transform_ms": 30,
    "embedding_ms": 200,
    "vector_search_ms": 500,
    "context_format_ms": 50,
    "total_ms": 800,  # DB 포함 전체 (LLM 제외)
}

# --- 벤치마크 쿼리 세트 ---
BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    {"query": "2024년 KIA 타이거즈 홈런왕은 누구야?", "intent": "stats_lookup"},
    {"query": "류현진 선수 커리어 통계", "intent": "player_profile"},
    {"query": "LG 트윈스 vs 두산 베어스 팀 비교", "intent": "comparison"},
    {"query": "골든글러브 외야수 수상 기준", "intent": "award_lookup"},
    {"query": "2023년 삼성 라이온즈 팀 분석", "intent": "team_analysis"},
    {"query": "ERA 방어율이란?", "intent": "knowledge_explanation"},
    {"query": "타율 1위 타자는", "intent": "stats_lookup"},
    {"query": "2024년 MVP 수상자", "intent": "award_lookup"},
    {"query": "KIA 1번 타자 출루율", "intent": "stats_lookup"},
    {"query": "오승환 통산 세이브 기록", "intent": "player_profile"},
]


@dataclass
class StageMeasurement:
    stage: str
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class RunResult:
    query: str
    intent: str
    stages: List[StageMeasurement] = field(default_factory=list)
    total_ms: float = 0.0
    error: Optional[str] = None

    def stage_latency(self, stage: str) -> Optional[float]:
        for s in self.stages:
            if s.stage == stage and s.success:
                return s.latency_ms
        return None


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return round(ordered[lo], 2)
    w = pos - lo
    return round(ordered[lo] * (1 - w) + ordered[hi] * w, 2)


def _measure_sync(fn, *args, **kwargs) -> tuple[Any, float]:
    """동기 함수 실행 시간 측정 (ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


async def _measure_async(coro) -> tuple[Any, float]:
    """비동기 함수 실행 시간 측정 (ms)."""
    t0 = time.perf_counter()
    result = await coro
    return result, (time.perf_counter() - t0) * 1000


async def _run_single(
    query: str,
    intent: str,
    *,
    settings: Any,
    with_db: bool,
) -> RunResult:
    from app.core.entity_extractor import extract_entities_from_query, EntityFilter
    from app.core.query_transformer import QueryTransformer
    from app.core.context_formatter import ContextFormatter
    from app.core.embeddings import async_embed_query

    run = RunResult(query=query, intent=intent)
    t_start = time.perf_counter()

    # 1. 엔티티 추출
    entity_filter: EntityFilter
    try:
        entity_filter, ms = _measure_sync(extract_entities_from_query, query)
        run.stages.append(StageMeasurement("entity_extraction_ms", ms, True))
    except Exception as exc:
        run.stages.append(StageMeasurement("entity_extraction_ms", 0, False, str(exc)))
        entity_filter = EntityFilter()

    # 2. 쿼리 변환
    try:
        transformer = QueryTransformer()
        expanded_variations, ms = _measure_sync(
            transformer.expand_query_with_rules, query, entity_filter
        )
        expanded = (
            [v.query for v in expanded_variations] if expanded_variations else [query]
        )
        run.stages.append(StageMeasurement("query_transform_ms", ms, True))
    except Exception as exc:
        run.stages.append(StageMeasurement("query_transform_ms", 0, False, str(exc)))
        expanded = [query]

    # 3. 임베딩 생성 (첫 번째 쿼리 변형)
    embed_query = expanded[0] if expanded else query
    try:
        embedding, ms = await _measure_async(async_embed_query(embed_query, settings))
        run.stages.append(StageMeasurement("embedding_ms", ms, True))
    except Exception as exc:
        run.stages.append(StageMeasurement("embedding_ms", 0, False, str(exc)))
        embedding = None

    # 4. 벡터 검색 (DB 필요)
    if with_db and embedding is not None:
        try:
            from app.core.retrieval import retrieve_chunks

            from app.core.entity_extractor import convert_to_db_filters

            filters = convert_to_db_filters(entity_filter)

            chunks, ms = await _measure_async(
                retrieve_chunks(
                    embed_query, top_k=5, filters=filters, settings=settings
                )
            )
            run.stages.append(StageMeasurement("vector_search_ms", ms, True))
        except Exception as exc:
            run.stages.append(StageMeasurement("vector_search_ms", 0, False, str(exc)))
            chunks = []
    else:
        chunks = []
        if with_db:
            run.stages.append(
                StageMeasurement("vector_search_ms", 0, False, "embedding failed")
            )

    # 5. 컨텍스트 포맷팅
    try:
        formatter = ContextFormatter()
        mock_processed = {
            "pitchers": [],
            "batters": [],
            "awards": [],
            "raw_docs": [c.get("content", "") for c in chunks] if chunks else [],
        }
        current_year = entity_filter.season_year or 2024
        _, ms = _measure_sync(
            formatter.format_context,
            mock_processed,
            intent,
            query,
            entity_filter,
            current_year,
        )
        run.stages.append(StageMeasurement("context_format_ms", ms, True))
    except Exception as exc:
        run.stages.append(StageMeasurement("context_format_ms", 0, False, str(exc)))

    run.total_ms = (time.perf_counter() - t_start) * 1000
    return run


async def run_benchmark(
    *,
    runs: int,
    with_db: bool,
) -> Dict[str, Any]:
    from app.config import get_settings

    settings = get_settings()

    all_results: List[RunResult] = []
    for q in BENCHMARK_QUERIES:
        for _ in range(runs):
            result = await _run_single(
                q["query"],
                q["intent"],
                settings=settings,
                with_db=with_db,
            )
            all_results.append(result)

    return _build_report(all_results, runs=runs, with_db=with_db)


def _build_report(
    results: List[RunResult], *, runs: int, with_db: bool
) -> Dict[str, Any]:
    stage_names = [
        "entity_extraction_ms",
        "query_transform_ms",
        "embedding_ms",
        "vector_search_ms",
        "context_format_ms",
    ]

    stage_latencies: Dict[str, List[float]] = {s: [] for s in stage_names}
    total_latencies: List[float] = []
    errors: List[str] = []

    for r in results:
        total_latencies.append(r.total_ms)
        for stage in stage_names:
            lat = r.stage_latency(stage)
            if lat is not None:
                stage_latencies[stage].append(lat)
        for s in r.stages:
            if not s.success and s.error:
                errors.append(f"{r.query[:30]} [{s.stage}]: {s.error}")

    summary: Dict[str, Any] = {
        "config": {
            "runs_per_query": runs,
            "total_runs": len(results),
            "with_db": with_db,
            "embed_provider": os.getenv("EMBED_PROVIDER", "unknown"),
        },
        "stages": {},
        "total": {
            "p50_ms": _percentile(total_latencies, 0.5),
            "p99_ms": _percentile(total_latencies, 0.99),
            "target_ms": LATENCY_TARGETS_MS["total_ms"],
            "pass": _percentile(total_latencies, 0.99)
            <= LATENCY_TARGETS_MS["total_ms"],
        },
        "errors": errors[:10],
    }

    for stage in stage_names:
        lats = stage_latencies[stage]
        target = LATENCY_TARGETS_MS.get(stage, 0)
        p99 = _percentile(lats, 0.99)
        summary["stages"][stage] = {
            "p50_ms": _percentile(lats, 0.5),
            "p99_ms": p99,
            "sample_count": len(lats),
            "target_ms": target,
            "pass": p99 <= target if lats else None,
        }

    return summary


def _print_report(report: Dict[str, Any]) -> None:
    cfg = report["config"]
    print(f"\n{'='*60}")
    print(f"  RAG Pipeline Benchmark")
    print(f"  runs/query={cfg['runs_per_query']}  total={cfg['total_runs']}")
    print(f"  db={cfg['with_db']}  embed={cfg['embed_provider']}")
    print(f"{'='*60}")

    print(f"\n{'Stage':<25} {'P50':>8} {'P99':>8} {'Target':>8} {'Pass':>6}")
    print("-" * 60)

    stages = report.get("stages", {})
    for stage, metrics in stages.items():
        p50 = f"{metrics['p50_ms']:.1f}ms"
        p99 = f"{metrics['p99_ms']:.1f}ms"
        tgt = f"{metrics['target_ms']}ms"
        passed = metrics.get("pass")
        mark = "OK" if passed else ("--" if passed is None else "FAIL")
        print(f"  {stage:<23} {p50:>8} {p99:>8} {tgt:>8} {mark:>6}")

    total = report.get("total", {})
    print("-" * 60)
    p50 = f"{total['p50_ms']:.1f}ms"
    p99 = f"{total['p99_ms']:.1f}ms"
    tgt = f"{total['target_ms']}ms"
    mark = "OK" if total.get("pass") else "FAIL"
    print(f"  {'total (no LLM)':<23} {p50:>8} {p99:>8} {tgt:>8} {mark:>6}")

    errors = report.get("errors", [])
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG 파이프라인 단계별 레이턴시 벤치마크"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="쿼리당 반복 횟수 (기본값: 5)",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="DB 연결 포함 (벡터 검색 측정, POSTGRES_DB_URL 필요)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="JSON 결과 저장 경로 (선택)",
    )
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    report = await run_benchmark(runs=args.runs, with_db=args.db)
    _print_report(report)

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote: {out}")

    # 에러 없이 모든 측정 완료 → 0, 실패 스테이지 있으면 1
    has_fail = any(
        m.get("pass") is False for m in report.get("stages", {}).values()
    ) or not report.get("total", {}).get("pass", True)
    return 1 if has_fail else 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
