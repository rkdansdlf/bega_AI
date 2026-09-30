import asyncio
import hashlib
import math
import re

import pytest

from scripts import benchmark_chunk_params as bcp


def test_parse_config_validation():
    assert bcp.parse_config("650/900/80") == (650, 900, 80)
    for bad in ("900/650/80", "650/900/700", "a/b/c", "650/900"):
        with pytest.raises(ValueError):
            bcp.parse_config(bad)


def _bow_embed(texts):
    """Deterministic bag-of-words hashing vectors (meaningful for overlap)."""
    out = []
    for text in texts:
        vec = [0.0] * 256
        for tok in re.findall(r"[가-힣A-Za-z0-9]+", text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256
            vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


def test_score_config_rewards_chunk_that_contains_the_answer():
    chunks = [
        {
            "source_table": "kbo_regulations",
            "content": "타이브레이크 규정: 동률 시 순위 결정",
        },
        {"source_table": "markdown_docs", "content": "완전히 다른 이야기 플래툰 전략"},
    ]
    cases = [
        {
            "query": "타이브레이크 규정",
            "expected_substrings": ["타이브레이크"],
            "expected_top_table": "kbo_regulations",
        }
    ]
    emb = _bow_embed([c["content"] for c in chunks])
    q = _bow_embed([c["query"] for c in cases])
    m = bcp.score_config(chunks, emb, cases, q)
    assert m["recall@5"] == 1.0 and m["mrr"] == 1.0
    assert m["top1_source_precision"] == 1.0


def test_answer_split_across_chunks_scores_zero():
    chunks = [
        {"source_table": "t", "content": "타이브레이크"},
        {"source_table": "t", "content": "동률 규정"},
    ]
    cases = [{"query": "q", "expected_substrings": ["타이브레이크", "동률"]}]
    emb = _bow_embed([c["content"] for c in chunks])
    m = bcp.score_config(chunks, emb, cases, _bow_embed(["q"]))
    assert m["recall@5"] == 0.0 and m["mrr"] == 0.0


def test_ranking_prefers_recall_then_mrr_then_fewer_chunks():
    results = [
        {"config": "a", "metrics": {"recall@5": 0.5, "mrr": 0.9, "chunks": 10}},
        {"config": "b", "metrics": {"recall@5": 0.8, "mrr": 0.4, "chunks": 50}},
        {"config": "c", "metrics": {"recall@5": 0.8, "mrr": 0.4, "chunks": 20}},
    ]
    assert [r["config"] for r in bcp.rank_configs(results)] == ["c", "b", "a"]


def test_end_to_end_over_real_static_corpus_with_offline_embedder():
    async def embed(texts):
        return _bow_embed(texts)

    report = asyncio.run(bcp.run([(650, 900, 80), (400, 700, 60)], embed))
    assert report["best"] in {"650/900/80", "400/700/60"}
    assert {r["config"] for r in report["results"]} == {"650/900/80", "400/700/60"}
    for r in report["results"]:
        assert r["metrics"]["chunks"] > 0
        assert r["metrics"]["max_chunk_chars"] <= 900
