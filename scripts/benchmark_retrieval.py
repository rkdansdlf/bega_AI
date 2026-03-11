#!/usr/bin/env python3
"""Offline A/B benchmark for the game flow summary rollout."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.baseball_agent import BaseballStatisticsAgent
from app.config import get_settings
from app.core.embeddings import async_embed_query
from app.core.entity_extractor import extract_entities_from_query
from app.core.retrieval import similarity_search
from app.deps import get_connection_pool
from app.ml.intent_router import predict_intent


@dataclass(frozen=True)
class BenchmarkCase:
    category: str
    query: str
    filters: Dict[str, Any] = field(default_factory=dict)
    expected_tool: Optional[str] = None
    expected_top_table: Optional[str] = None
    expected_substrings: tuple[str, ...] = ()


@dataclass
class VariantMetrics:
    source_tables: List[str]
    top_k_hit: bool
    avg_context_chars: float
    retrieval_latency_ms: float
    result_count: int
    error: Optional[str] = None


@dataclass
class BenchmarkRuntime:
    embedding_error: Optional[str] = None


DEFAULT_CASES: List[BenchmarkCase] = [
    BenchmarkCase(
        category="inning_box_score",
        query="2025년 5월 1일 경기 박스스코어 알려줘",
        filters={"season_year": 2025},
        expected_tool="get_game_box_score",
        expected_substrings=("box_score",),
    ),
    BenchmarkCase(
        category="inning_box_score",
        query="2025년 5월 1일 경기 이닝별 득점 알려줘",
        filters={"season_year": 2025},
        expected_tool="get_game_box_score",
        expected_substrings=("box_score",),
    ),
    BenchmarkCase(
        category="general_game_detail",
        query="2025년 5월 1일 경기 일정 알려줘",
        filters={"season_year": 2025},
        expected_tool="get_games_by_date",
        expected_top_table="game",
    ),
    BenchmarkCase(
        category="game_flow_narrative",
        query="2025년 5월 1일 경기 흐름 요약해줘",
        filters={"season_year": 2025},
        expected_tool=None,
        expected_top_table="game_flow_summary",
    ),
    BenchmarkCase(
        category="game_flow_narrative",
        query="2025년 5월 1일 승부가 언제 갈렸어?",
        filters={"season_year": 2025},
        expected_tool=None,
        expected_top_table="game_flow_summary",
    ),
    BenchmarkCase(
        category="game_flow_narrative",
        query="2025년 5월 1일 초중후반 득점 양상 알려줘",
        filters={"season_year": 2025},
        expected_tool=None,
        expected_top_table="game_flow_summary",
    ),
    BenchmarkCase(
        category="player_season_stats",
        query="김도영 선수 2024년 기록 알려줘",
        filters={"season_year": 2024},
        expected_tool="get_player_stats",
        expected_top_table="player_season_batting",
        expected_substrings=("김도영",),
    ),
    BenchmarkCase(
        category="player_season_stats",
        query="2024년 평균자책점 1위 누구야?",
        filters={"season_year": 2024},
        expected_tool="get_leaderboard",
        expected_top_table="player_season_pitching",
    ),
    BenchmarkCase(
        category="mixed_freeform",
        query="2024년 우승팀은?",
        filters={"season_year": 2024},
        expected_tool="get_korean_series_winner",
        expected_substrings=("우승",),
    ),
]


async def _noop_llm_generator(*_: Any, **__: Any):
    if False:
        yield ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs candidate retrieval for game-flow summary rollout."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Top-k retrieval limit per query.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def _serialize_tool_result(result: Any) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


def _contains_expected_substrings(
    payload: Any, expected_substrings: tuple[str, ...]
) -> bool:
    if not expected_substrings:
        return True
    haystack = _serialize_tool_result(payload)
    return all(needle in haystack for needle in expected_substrings)


def _plan_tools(agent: BaseballStatisticsAgent, case: BenchmarkCase) -> Dict[str, Any]:
    entity_filter = extract_entities_from_query(case.query)
    plan = agent._build_fast_path_plan(case.query, entity_filter, context={})
    if plan is not None:
        return plan
    return {
        "tool_calls": [],
        "planner_mode": "none",
        "intent": predict_intent(case.query),
    }


def _execute_first_tool(
    agent: BaseballStatisticsAgent, plan: Dict[str, Any]
) -> Dict[str, Any]:
    tool_calls = plan.get("tool_calls", [])
    if not tool_calls:
        return {
            "tool_name": None,
            "success": False,
            "message": "no_tool_call",
            "data": {},
            "expected_match": False,
        }

    first_tool = tool_calls[0]
    result = agent.tool_caller.execute_tool(first_tool)
    return {
        "tool_name": first_tool.tool_name,
        "success": result.success,
        "message": result.message,
        "data": result.data,
        "expected_match": True,
    }


def _build_variant_filters(
    case: BenchmarkCase,
    *,
    exclude_source_tables: tuple[str, ...],
) -> Dict[str, Any]:
    filters = dict(case.filters)
    if case.category == "game_flow_narrative":
        filters["source_table"] = "game_flow_summary"
    if exclude_source_tables:
        filters["_exclude_source_tables"] = list(exclude_source_tables)
    return filters


async def _run_variant(
    settings,
    connection,
    case: BenchmarkCase,
    *,
    limit: int,
    exclude_source_tables: tuple[str, ...],
    runtime: BenchmarkRuntime,
) -> VariantMetrics:
    if runtime.embedding_error:
        return VariantMetrics(
            source_tables=[],
            top_k_hit=False,
            avg_context_chars=0.0,
            retrieval_latency_ms=0.0,
            result_count=0,
            error=runtime.embedding_error,
        )

    filters = _build_variant_filters(
        case,
        exclude_source_tables=exclude_source_tables,
    )

    try:
        start_time = time.perf_counter()
        embedding = await async_embed_query(case.query, settings)
        docs = (
            similarity_search(
                connection,
                embedding,
                limit=limit,
                filters=filters,
                keyword=None,
            )
            if embedding
            else []
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "insufficient_quota" in lowered or "exceeded your current quota" in lowered:
            runtime.embedding_error = message
        return VariantMetrics(
            source_tables=[],
            top_k_hit=False,
            avg_context_chars=0.0,
            retrieval_latency_ms=0.0,
            result_count=0,
            error=message,
        )

    context_lengths = [
        len(doc.get("content", "")) for doc in docs if doc.get("content")
    ]
    source_tables = [
        str(doc.get("source_table", "")) for doc in docs if doc.get("source_table")
    ]
    avg_context_chars = (
        round(sum(context_lengths) / len(context_lengths), 2)
        if context_lengths
        else 0.0
    )
    top_k_hit = (
        bool(case.expected_top_table)
        and case.expected_top_table in source_tables[:limit]
    )

    return VariantMetrics(
        source_tables=source_tables,
        top_k_hit=bool(top_k_hit),
        avg_context_chars=avg_context_chars,
        retrieval_latency_ms=round(elapsed_ms, 2),
        result_count=len(docs),
    )


def _aggregate_results(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in case_results:
        by_category[result["category"]].append(result)

    summary: Dict[str, Any] = {}
    for category, rows in by_category.items():
        route_ok = sum(1 for row in rows if row["tool_choice_ok"])
        correctness_ok = sum(1 for row in rows if row["expected_facts_ok"])
        variants: Dict[str, Dict[str, float]] = {}
        for variant_name in ("baseline", "candidate"):
            variant_rows = [row[variant_name] for row in rows]
            successful_rows = [row for row in variant_rows if not row.get("error")]
            variants[variant_name] = {
                "avg_latency_ms": (
                    round(
                        sum(row["retrieval_latency_ms"] for row in successful_rows)
                        / len(successful_rows),
                        2,
                    )
                    if successful_rows
                    else 0.0
                ),
                "avg_context_chars": (
                    round(
                        sum(row["avg_context_chars"] for row in successful_rows)
                        / len(successful_rows),
                        2,
                    )
                    if successful_rows
                    else 0.0
                ),
                "top_k_hit_rate": (
                    round(
                        sum(1 for row in successful_rows if row["top_k_hit"])
                        / len(successful_rows),
                        4,
                    )
                    if successful_rows
                    else 0.0
                ),
                "error_count": sum(1 for row in variant_rows if row.get("error")),
            }
        summary[category] = {
            "cases": len(rows),
            "tool_choice_rate": round(route_ok / len(rows), 4),
            "expected_fact_rate": round(correctness_ok / len(rows), 4),
            "baseline": variants["baseline"],
            "candidate": variants["candidate"],
        }
    return summary


async def run_benchmark(limit: int) -> Dict[str, Any]:
    settings = get_settings()
    pool = get_connection_pool()
    runtime = BenchmarkRuntime()
    report: Dict[str, Any] = {
        "cases": [],
    }

    with pool.connection() as conn:
        agent = BaseballStatisticsAgent(conn, _noop_llm_generator)

        for case in DEFAULT_CASES:
            plan = _plan_tools(agent, case)
            planned_tool_names = [
                tool_call.tool_name for tool_call in plan.get("tool_calls", [])
            ]
            tool_result = _execute_first_tool(agent, plan)
            if case.expected_tool is None:
                tool_choice_ok = not planned_tool_names
            else:
                tool_choice_ok = bool(
                    planned_tool_names and planned_tool_names[0] == case.expected_tool
                )
            expected_facts_ok = _contains_expected_substrings(
                tool_result.get("data", {}),
                case.expected_substrings,
            )

            baseline_metrics = await _run_variant(
                settings,
                conn,
                case,
                limit=limit,
                exclude_source_tables=("game_flow_summary",),
                runtime=runtime,
            )
            candidate_metrics = await _run_variant(
                settings,
                conn,
                case,
                limit=limit,
                exclude_source_tables=(),
                runtime=runtime,
            )

            report["cases"].append(
                {
                    "category": case.category,
                    "query": case.query,
                    "filters": case.filters,
                    "predicted_intent": predict_intent(case.query),
                    "planner_mode": plan.get("planner_mode", "none"),
                    "planned_tools": planned_tool_names,
                    "expected_tool": case.expected_tool,
                    "tool_choice_ok": tool_choice_ok,
                    "tool_result": {
                        "tool_name": tool_result["tool_name"],
                        "success": tool_result["success"],
                        "message": tool_result["message"],
                    },
                    "expected_facts_ok": expected_facts_ok,
                    "baseline": asdict(baseline_metrics),
                    "candidate": asdict(candidate_metrics),
                }
            )

    report["summary"] = _aggregate_results(report["cases"])
    return report


def main() -> int:
    args = parse_args()
    report = asyncio.run(run_benchmark(limit=args.limit))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote report: {output_path}")

    has_errors = any(
        case[variant].get("error")
        for case in report["cases"]
        for variant in ("baseline", "candidate")
    )
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
