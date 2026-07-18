from __future__ import annotations

import asyncio
from typing import Any, Iterable, List

from scripts import benchmark_retrieval as benchmark


def test_candidate_source_priority_matches_query_domain() -> None:
    assert benchmark._candidate_source_priority("정규시즌 타이브레이크 규정 알려줘") == {
        "kbo_regulations": 0,
        "kbo_definitions": 1,
        "markdown_docs": 2,
    }
    assert benchmark._candidate_source_priority(
        "player_season_batting의 extra_stats 컬럼은 뭐야?"
    )["kbo_definitions"] == 0
    assert benchmark._candidate_source_priority("ops 컬럼은 어떤 의미야?")[
        "kbo_definitions"
    ] == 0
    assert benchmark._candidate_source_priority("웨이버 제도가 뭐야?")[
        "kbo_regulations"
    ] == 0
    assert benchmark._candidate_source_priority("플래툰 전략이 뭐야?")[
        "markdown_docs"
    ] == 0


def test_acceptance_uses_absolute_jitter_budget_for_fast_retrieval() -> None:
    baseline = {
        "zero_hit_rate": 0.0,
        "top1_source_table_precision": 0.5,
        "answerable_rate": 0.5,
        "retrieval_p95_ms": 2.0,
    }
    candidate = {
        "zero_hit_rate": 0.0,
        "top1_source_table_precision": 1.0,
        "answerable_rate": 0.5,
        "retrieval_p95_ms": 7.0,
    }

    acceptance = benchmark._build_acceptance(baseline, candidate)

    assert acceptance["candidate_p95_limit_ms"] == 12.0
    assert acceptance["checks"]["p95_within_budget"] is True
    assert acceptance["passed"] is True


def test_report_gate_fails_when_acceptance_fails_without_runtime_errors() -> None:
    report = {
        "cases": [
            {
                "baseline": {"error": None},
                "candidate": {"error": None},
            }
        ],
        "summary": {"overall": {"acceptance": {"passed": False}}},
    }

    assert benchmark._report_passes(report, variants=["baseline", "candidate"]) is False


def test_report_gate_fails_when_any_category_acceptance_fails() -> None:
    report = {
        "cases": [
            {
                "baseline": {"error": None},
                "candidate": {"error": None},
            }
        ],
        "summary": {
            "overall": {"acceptance": {"passed": True}},
            "by_category": {
                "regulation": {"acceptance": {"passed": False}},
            },
        },
    }

    assert benchmark._report_passes(report, variants=["baseline", "candidate"]) is False


def _vectorize(text: str) -> List[float]:
    lowered = text.lower()
    features = [
        "플래툰",
        "잠실",
        "babip",
        "fip",
        "타이브레이크",
        "와일드카드",
        "웨이버",
        "extra_stats",
        "ops",
        "출루율",
    ]
    return [float(lowered.count(token.lower())) for token in features]


async def _fake_embed_texts(texts: Iterable[str], settings: Any) -> List[List[float]]:
    del settings
    return [_vectorize(text) for text in texts]


async def _fake_embed_query(text: str, settings: Any) -> List[float]:
    del settings
    return _vectorize(text)


def test_run_benchmark_reports_document_quality_metrics(monkeypatch: Any) -> None:
    monkeypatch.setattr(benchmark, "async_embed_texts", _fake_embed_texts)
    monkeypatch.setattr(benchmark, "async_embed_query", _fake_embed_query)

    report = asyncio.run(benchmark.run_benchmark(limit=3, variant="both"))

    overall = report["summary"]["overall"]
    assert set(report["summary"]["by_category"]) == {
        "markdown_docs",
        "kbo_regulations",
        "kbo_definitions",
    }
    assert {
        "zero_hit_rate",
        "top1_source_table_precision",
        "answerable_rate",
        "retrieval_p50_ms",
        "retrieval_p95_ms",
    } <= set(overall["baseline"])
    assert {
        "zero_hit_not_worse",
        "quality_not_worse",
        "quality_or_latency_improved",
        "p95_within_budget",
    } <= set(overall["acceptance"]["checks"])
