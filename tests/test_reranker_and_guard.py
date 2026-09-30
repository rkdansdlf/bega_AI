import asyncio
from types import SimpleNamespace

import httpx

from app.core.relevance_guard import apply_relevance_guard
from app.core.reranker import (
    HTTPCrossEncoderReranker,
    ScoreReranker,
    build_reranker,
)


def _run(coro):
    return asyncio.run(coro)


DOCS = [
    {"id": 1, "content": "a", "similarity": 0.9, "combined_score": 0.01},
    {"id": 2, "content": "b", "similarity": 0.8, "combined_score": 0.03},
    {"id": 3, "content": "c", "similarity": 0.7, "combined_score": 0.02},
]


def test_guard_drops_other_team_and_season_but_keeps_unknown():
    docs = [
        {"id": 1, "team_id": "LG", "season_year": 2025},
        {"id": 2, "team_id": "KIA", "season_year": 2025},
        {"id": 3, "team_id": "LG", "season_year": 2024},
        {"id": 4},  # no metadata: unknown is not a conflict
        {"id": 5, "meta": {"team_id": "lg", "season_year": "2025"}},
    ]
    res = apply_relevance_guard(docs, {"team_id": "LG", "season_year": 2025})
    assert [d["id"] for d in res.kept] == [1, 4, 5]
    assert {r["dimension"] for r in res.reasons} == {"team_id", "season_year"}


def test_guard_all_dropped_flag_and_no_constraints_noop():
    docs = [{"id": 1, "team_id": "KIA"}]
    assert apply_relevance_guard(docs, {"team_id": "LG"}).all_dropped
    assert apply_relevance_guard(docs, {}).kept == docs


def test_score_reranker_orders_by_combined_score():
    out = _run(ScoreReranker().rerank("q", DOCS, 2))
    assert [d["id"] for d in out.docs] == [2, 3]
    assert out.version == "score_v1"


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_http_reranker_reorders_by_model_scores():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.99},
                    {"index": 2, "relevance_score": 0.5},
                    {"index": 1, "relevance_score": 0.1},
                ]
            },
        )

    rr = HTTPCrossEncoderReranker(
        url="http://rerank.test/v1/rerank", model="m", client=_client(handler)
    )
    out = _run(rr.rerank("q", DOCS, 2))
    assert [d["id"] for d in out.docs] == [1, 3]
    assert out.docs[0]["rerank_score"] == 0.99
    assert not out.degraded


def test_http_reranker_fails_open_to_score_order():
    def handler(request):
        return httpx.Response(503)

    rr = HTTPCrossEncoderReranker(
        url="http://rerank.test/v1/rerank", model="m", client=_client(handler)
    )
    out = _run(rr.rerank("q", DOCS, 3))
    assert out.degraded
    assert [d["id"] for d in out.docs] == [2, 3, 1]
    assert out.version.startswith("http:m->")


def test_http_reranker_rejects_out_of_range_index():
    def handler(request):
        return httpx.Response(
            200, json={"results": [{"index": 9, "relevance_score": 1}]}
        )

    rr = HTTPCrossEncoderReranker(
        url="http://rerank.test/x", model="m", client=_client(handler)
    )
    assert _run(rr.rerank("q", DOCS, 2)).degraded


def test_build_reranker_selection():
    off = SimpleNamespace(rag_rerank_enabled=False)
    assert build_reranker(off) is None
    score = SimpleNamespace(rag_rerank_enabled=True, rag_reranker_provider="score")
    assert isinstance(build_reranker(score), ScoreReranker)
    incomplete = SimpleNamespace(rag_rerank_enabled=True, rag_reranker_provider="http")
    assert isinstance(build_reranker(incomplete), ScoreReranker)
    http = SimpleNamespace(
        rag_rerank_enabled=True,
        rag_reranker_provider="http",
        rag_reranker_url="http://x",
        rag_reranker_model="m",
    )
    assert isinstance(build_reranker(http), HTTPCrossEncoderReranker)
