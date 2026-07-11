"""Tests for AI optimization operation scripts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import chat_model_routing_experiment
from scripts.cleanup_chat_semantic_cache import CleanupScope, build_where_clause


def test_chat_model_routing_loads_plain_text_questions(tmp_path: Path) -> None:
    samples = tmp_path / "samples.txt"
    samples.write_text("# comment\nKIA 전력 알려줘\n\nLG 불펜 분석\n", encoding="utf-8")

    assert chat_model_routing_experiment._load_questions(samples) == [
        "KIA 전력 알려줘",
        "LG 불펜 분석",
    ]


def test_chat_model_routing_loads_jsonl_questions(tmp_path: Path) -> None:
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        "\n".join(
            [
                json.dumps({"question": "두산 선발 분석"}, ensure_ascii=False),
                json.dumps("SSG 최근 흐름", ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    assert chat_model_routing_experiment._load_questions(samples) == [
        "두산 선발 분석",
        "SSG 최근 흐름",
    ]


def test_chat_model_routing_percentile_bounds() -> None:
    assert chat_model_routing_experiment._percentile([], 0.95) == 0.0
    assert chat_model_routing_experiment._percentile([10, 20, 30], 0.95) == 30


def test_semantic_cache_cleanup_where_clause_uses_filters() -> None:
    scope = CleanupScope(
        days=7,
        source_tier="database",
        embedding_signature="openrouter:small:256",
        max_hit_count=0,
        limit=50,
        dry_run=True,
        allow_global=False,
    )

    where_clause, params = build_where_clause(scope)

    assert "expires_at < now() - make_interval(days => %s)" in where_clause
    assert "source_tier = %s" in where_clause
    assert "embedding_signature = %s" in where_clause
    assert "hit_count <= %s" in where_clause
    assert params == [7, "database", "openrouter:small:256", 0]


def test_semantic_cache_cleanup_blocks_immediate_global_purge() -> None:
    scope = CleanupScope(
        days=0,
        source_tier=None,
        embedding_signature=None,
        max_hit_count=None,
        limit=50,
        dry_run=True,
        allow_global=False,
    )

    try:
        build_where_clause(scope)
    except ValueError as exc:
        assert "global immediate purge" in str(exc)
    else:
        raise AssertionError("expected ValueError")
