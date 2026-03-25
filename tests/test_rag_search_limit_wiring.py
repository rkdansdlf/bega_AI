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


def _search_strategy(
    search_limit: int,
    *,
    db_filters=None,
    player_name=None,
    team_id=None,
):
    return {
        "entity_filter": SimpleNamespace(
            game_date=None,
            player_name=player_name,
            position_type=None,
            season_year=None,
            team_id=team_id,
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
    pipeline = _make_pipeline(
        retrieval_fallback_limit_relaxed=14,
        retrieval_single_query_for_strict_entity=False,
    )

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


@pytest.mark.asyncio
async def test_run_uses_single_query_for_strict_player_entity(monkeypatch) -> None:
    pipeline = _make_pipeline(retrieval_single_query_for_strict_entity=True)

    monkeypatch.setattr(
        "app.core.rag.enhance_search_strategy",
        lambda _query: _search_strategy(11, player_name="김현수"),
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

    async def _unexpected_multi_query(*_args, **_kwargs):
        raise AssertionError("strict player query should not use multi-query retrieval")

    async def _fake_retrieve(
        query,
        *,
        limit=None,
        filters=None,
        entity_filter=None,
        use_hyde=True,
    ):
        del query, entity_filter, use_hyde
        assert limit == 11
        assert "source_table" not in (filters or {})
        raise RuntimeError("single_query_asserted")

    monkeypatch.setattr(pipeline, "retrieve_with_multi_query", _unexpected_multi_query)
    monkeypatch.setattr(pipeline, "retrieve", _fake_retrieve)

    with pytest.raises(RuntimeError, match="single_query_asserted"):
        await pipeline.run("김현수 시즌 성적 알려줘", intent="stats_lookup")


@pytest.mark.asyncio
async def test_run_uses_single_query_for_source_table_constrained_docs(
    monkeypatch,
) -> None:
    pipeline = _make_pipeline(retrieval_single_query_for_strict_entity=True)

    monkeypatch.setattr(
        "app.core.rag.enhance_search_strategy",
        lambda _query: _search_strategy(
            7,
            db_filters={"source_table": "markdown_docs"},
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

    async def _unexpected_multi_query(*_args, **_kwargs):
        raise AssertionError("source_table constrained query should use single-query")

    async def _fake_retrieve(
        query,
        *,
        limit=None,
        filters=None,
        entity_filter=None,
        use_hyde=True,
    ):
        del query, entity_filter, use_hyde
        assert limit == 7
        assert filters == {"source_table": "markdown_docs"}
        raise RuntimeError("source_table_single_query_asserted")

    monkeypatch.setattr(pipeline, "retrieve_with_multi_query", _unexpected_multi_query)
    monkeypatch.setattr(pipeline, "retrieve", _fake_retrieve)

    with pytest.raises(RuntimeError, match="source_table_single_query_asserted"):
        await pipeline.run("용어집 문서 검색", intent="stats_lookup")
