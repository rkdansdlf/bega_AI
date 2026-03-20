from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import psycopg
import pytest

from app.config import get_settings
from app.core.rag import RAGPipeline


def _make_pipeline(**settings_updates) -> RAGPipeline:
    settings = get_settings().model_copy(update=settings_updates)
    connection = MagicMock(spec=psycopg.Connection)
    return RAGPipeline(settings=settings, connection=connection)


def _search_strategy(search_limit: int, *, db_filters=None):
    return {
        "entity_filter": SimpleNamespace(
            game_date=None,
            player_name=None,
            position_type=None,
            season_year=None,
            team_id=None,
            stat_type=None,
        ),
        "db_filters": db_filters or {},
        "is_ranking_query": False,
        "ranking_count": None,
        "search_limit": search_limit,
    }


@pytest.mark.asyncio
async def test_run_passes_search_strategy_limit_to_multi_query(monkeypatch) -> None:
    pipeline = _make_pipeline()

    monkeypatch.setattr(
        "app.core.rag.enhance_search_strategy", lambda _query: _search_strategy(13)
    )
    monkeypatch.setattr(
        pipeline, "_is_statistical_query", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        pipeline, "_is_general_conversation", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        pipeline, "_is_regulation_query", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(pipeline, "_is_game_query", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        pipeline,
        "_is_game_flow_narrative_query",
        lambda *_args, **_kwargs: False,
    )

    async def _fake_retrieve_with_multi_query(
        query,
        entity_filter,
        *,
        filters=None,
        use_llm_expansion=False,
        limit=None,
    ):
        del query, entity_filter, filters, use_llm_expansion
        assert limit == 13
        raise RuntimeError("limit_asserted")

    monkeypatch.setattr(
        pipeline, "retrieve_with_multi_query", _fake_retrieve_with_multi_query
    )

    with pytest.raises(RuntimeError, match="limit_asserted"):
        await pipeline.run("일반 검색 질문", intent="stats_lookup")


@pytest.mark.asyncio
async def test_run_uses_configured_fallback_limit(monkeypatch) -> None:
    pipeline = _make_pipeline(retrieval_fallback_limit_relaxed=14)

    monkeypatch.setattr(
        "app.core.rag.enhance_search_strategy",
        lambda _query: _search_strategy(
            9, db_filters={"source_table": "markdown_docs"}
        ),
    )
    monkeypatch.setattr(
        pipeline, "_is_statistical_query", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        pipeline, "_is_general_conversation", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        pipeline, "_is_regulation_query", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(pipeline, "_is_game_query", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        pipeline,
        "_is_game_flow_narrative_query",
        lambda *_args, **_kwargs: False,
    )

    async def _fake_retrieve_with_multi_query(*_args, **_kwargs):
        return []

    async def _fake_retrieve(
        query,
        *,
        limit=None,
        filters=None,
        entity_filter=None,
        use_hyde=True,
    ):
        del query, filters, entity_filter, use_hyde
        assert limit == 14
        raise RuntimeError("fallback_limit_asserted")

    monkeypatch.setattr(
        pipeline, "retrieve_with_multi_query", _fake_retrieve_with_multi_query
    )
    monkeypatch.setattr(pipeline, "retrieve", _fake_retrieve)

    with pytest.raises(RuntimeError, match="fallback_limit_asserted"):
        await pipeline.run("폴백 검색 질문", intent="stats_lookup")
