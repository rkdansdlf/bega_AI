from __future__ import annotations

import asyncio
from typing import Any, Iterable, List

from scripts import benchmark_retrieval as benchmark


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
        "quality_improved",
        "p95_within_budget",
    } <= set(overall["acceptance"]["checks"])
