from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.core.context_formatter import ContextFormatter
from app.core.rag import RAGPipeline
from scripts.benchmark_retrieval import (
    DEFAULT_CASES,
    BenchmarkCase,
    _build_variant_filters,
)


def test_context_formatter_prefers_game_flow_summary() -> None:
    formatter = ContextFormatter()
    processed_data = {
        "pitchers": [],
        "batters": [],
        "awards": [],
        "movements": [],
        "warnings": [],
        "context": None,
        "raw_docs": [],
        "games": [
            {
                "source_table": "game_flow_summary",
                "game_id": "20250501LGHH0",
                "game_date": "2025-05-01",
                "home_team": "LG",
                "away_team": "HH",
                "home_team_name": "LG 트윈스",
                "away_team_name": "한화 이글스",
                "home_score": 4,
                "away_score": 1,
                "content": "\n".join(
                    [
                        "[TL;DR] 2025-05-01 한화 이글스 1-4 LG 트윈스 LG 트윈스가 5회말부터 리드를 지켰습니다.",
                        "[핵심 문장] LG 트윈스는 5회말 이후 리드를 잃지 않았습니다.",
                        "[상세]",
                        "- 득점 이닝: 한화 이글스 1회 1점 / LG 트윈스 5회 4점",
                        '[META] {"kind": "game_flow_summary"}',
                        "[출처] KBO 경기 흐름 요약 (game_flow_summary#game_id=20250501LGHH0) / 기준일 2025-05-01",
                    ]
                ),
            },
            {
                "source_table": "game_metadata",
                "game_id": "20250501LGHH0",
                "stadium_name": "잠실",
                "attendance": 21500,
                "game_duration": 188,
            },
        ],
    }
    entity_filter = SimpleNamespace(
        award_type=None,
        movement_type=None,
        stat_type=None,
        player_name=None,
        team_id=None,
        game_date="2025-05-01",
    )

    context = formatter.format_context(
        processed_data,
        "game_lookup",
        "2025년 5월 1일 경기 흐름 요약해줘",
        entity_filter,
        2025,
    )

    assert "[TL;DR]" in context
    assert "잠실" in context
    assert "[META]" not in context
    assert "[출처]" not in context


def test_process_and_enrich_docs_treats_game_flow_summary_as_game_doc() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)

    processed = asyncio.run(
        pipeline._process_and_enrich_docs(
            [
                {
                    "source_table": "game_flow_summary",
                    "game_id": "20250501LGHH0",
                    "title": "flow",
                    "content": "flow content",
                    "meta": {"game_id": "20250501LGHH0"},
                }
            ],
            2025,
        )
    )

    assert processed["games"][0]["source_table"] == "game_flow_summary"


def test_rag_run_skips_agent_first_for_game_flow_and_drops_team_filter(
    monkeypatch,
) -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = SimpleNamespace(default_search_limit=5)
    pipeline.context_formatter = SimpleNamespace(
        format_context=lambda *_args, **_kwargs: "ctx"
    )

    async def _fake_generate(_messages):
        return "answer"

    async def _unexpected_agent(*_args, **_kwargs):
        raise AssertionError("agent-first should be skipped for flow narrative queries")

    captured = {}

    async def _fake_retrieve_with_multi_query(
        _query,
        _entity_filter,
        *,
        filters=None,
        use_llm_expansion=False,
        limit=None,
    ):
        del use_llm_expansion, limit
        captured["filters"] = dict(filters or {})
        return [
            {
                "id": 1,
                "title": "game flow",
                "content": "[TL;DR] flow",
                "source_table": "game_flow_summary",
                "game_id": "20250501LGHH0",
                "meta": {"game_id": "20250501LGHH0"},
            }
        ]

    async def _fake_process(_docs, _year):
        return {
            "pitchers": [],
            "batters": [],
            "games": [],
            "awards": [],
            "movements": [],
            "raw_docs": [],
            "warnings": [],
            "context": None,
        }

    pipeline._generate = _fake_generate
    pipeline._try_agent_first = _unexpected_agent
    pipeline.retrieve_with_multi_query = _fake_retrieve_with_multi_query
    pipeline._process_and_enrich_docs = _fake_process
    pipeline._is_statistical_query = lambda _query, _entity_filter: False
    pipeline._is_general_conversation = lambda _query: False
    pipeline._is_regulation_query = lambda _query: False
    pipeline._is_game_query = lambda _query: True
    pipeline._is_game_flow_narrative_query = lambda _query: True

    monkeypatch.setattr(
        "app.core.rag.enhance_search_strategy",
        lambda _query: {
            "entity_filter": SimpleNamespace(
                season_year=2025,
                team_id="LG",
                player_name=None,
                stat_type=None,
                position_type=None,
                league_type=None,
                award_type=None,
                movement_type=None,
                game_date="2025-05-01",
            ),
            "db_filters": {
                "season_year": 2025,
                "team_id": "LG",
                "meta.game_date": "2025-05-01",
            },
            "is_ranking_query": False,
            "ranking_count": None,
            "search_limit": 15,
        },
    )

    result = asyncio.run(
        pipeline.run(
            "2025년 5월 1일 경기 흐름 요약해줘",
            intent="game_lookup",
        )
    )

    assert captured["filters"]["source_table"] == "game_flow_summary"
    assert "team_id" not in captured["filters"]
    assert result["answer"] == "answer"


def test_benchmark_cases_include_document_quality_buckets() -> None:
    categories = {case.category for case in DEFAULT_CASES}

    assert categories == {
        "markdown_docs",
        "kbo_regulations",
        "kbo_definitions",
    }
    assert any(case.expected_top_table == "kbo_definitions" for case in DEFAULT_CASES)


def test_benchmark_variant_filters_keep_exclusion_list() -> None:
    filters = _build_variant_filters(
        BenchmarkCase(
            category="markdown_docs",
            query="플래툰 전략이 뭐야?",
        ),
        exclude_source_tables=("kbo_regulations",),
    )

    assert filters["_exclude_source_tables"] == ["kbo_regulations"]
