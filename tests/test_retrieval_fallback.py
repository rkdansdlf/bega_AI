"""
similarity_search_with_fallback() 단위 테스트.

DB 연결 없이 similarity_search를 mock으로 교체하여 4단계 필터 완화 로직,
내부 필터 보존, min_results 경계값, 반환 형식을 검증한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.retrieval import (
    _FALLBACK_FILTER_KEYS,
    _INTERNAL_FILTER_EXCLUDE_SOURCE_TABLES,
    _INTERNAL_FILTER_INCLUDE_INNING_SCORES,
    similarity_search_with_fallback,
)

# ── 공통 헬퍼 ──────────────────────────────────────────────────────────────────

DUMMY_EMBEDDING = [0.1] * 8
MOCK_DOC = {"content": "test doc", "score": 0.9}
CONN = MagicMock()  # psycopg.Connection mock (실제 DB 호출 없음)

_PATCH_TARGET = "app.core.retrieval.similarity_search"


def _full_filters() -> Dict[str, Any]:
    return {"season_year": 2024, "team_id": "KIA", "source_table": "batting_season"}


# ── TestFallbackLevelProgression ──────────────────────────────────────────────


class TestFallbackLevelProgression:
    @pytest.mark.asyncio
    async def test_level1_success_no_fallback(self):
        """Level 1에서 결과가 있으면 즉시 반환, 추가 호출 없음."""
        with patch(_PATCH_TARGET, return_value=[MOCK_DOC]) as mock_ss:
            results, level = await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=_full_filters()
            )
        assert results == [MOCK_DOC]
        assert level == "level_1"
        mock_ss.assert_called_once()

    @pytest.mark.asyncio
    async def test_level1_fail_level2_success(self):
        """Level 1 0건 → Level 2(source_table 제거) 에서 결과 반환."""
        side_effects = [[], [MOCK_DOC]]
        with patch(_PATCH_TARGET, side_effect=side_effects) as mock_ss:
            results, level = await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=_full_filters()
            )
        assert results == [MOCK_DOC]
        assert level == "level_2"
        assert mock_ss.call_count == 2

        # Level 2 호출 시 source_table이 filters에 없어야 함
        second_call_filters = mock_ss.call_args_list[1][1]["filters"]
        assert "source_table" not in second_call_filters
        assert second_call_filters["season_year"] == 2024
        assert second_call_filters["team_id"] == "KIA"

    @pytest.mark.asyncio
    async def test_level2_fail_level3_success(self):
        """Level 1·2 실패 → Level 3(team_id 추가 제거) 에서 결과 반환."""
        side_effects = [[], [], [MOCK_DOC]]
        with patch(_PATCH_TARGET, side_effect=side_effects) as mock_ss:
            results, level = await similarity_search_with_fallback(
                CONN,
                DUMMY_EMBEDDING,
                limit=5,
                filters=_full_filters(),
                intent="knowledge_explanation",
            )
        assert level == "level_3"
        assert mock_ss.call_count == 3

        third_call_filters = mock_ss.call_args_list[2][1]["filters"]
        assert "source_table" not in third_call_filters
        assert "team_id" not in third_call_filters
        assert third_call_filters["season_year"] == 2024

    @pytest.mark.asyncio
    async def test_level3_fail_level4_success(self):
        """Level 1·2·3 실패 → Level 4(season_year 제거, 벡터만) 에서 결과 반환."""
        side_effects = [[], [], [], [MOCK_DOC]]
        with patch(_PATCH_TARGET, side_effect=side_effects) as mock_ss:
            results, level = await similarity_search_with_fallback(
                CONN,
                DUMMY_EMBEDDING,
                limit=5,
                filters=_full_filters(),
                intent="knowledge_explanation",
            )
        assert level == "level_4"
        assert mock_ss.call_count == 4

        fourth_call_filters = mock_ss.call_args_list[3][1]["filters"]
        # 모든 사용자 필터가 제거되어 None 또는 빈 dict
        if fourth_call_filters is not None:
            assert "source_table" not in fourth_call_filters
            assert "team_id" not in fourth_call_filters
            assert "season_year" not in fourth_call_filters

    @pytest.mark.asyncio
    async def test_all_levels_exhausted_returns_empty(self):
        """모든 레벨 소진 시 빈 리스트와 exhausted 레벨 문자열 반환."""
        with patch(_PATCH_TARGET, return_value=[]) as mock_ss:
            results, level = await similarity_search_with_fallback(
                CONN,
                DUMMY_EMBEDDING,
                limit=5,
                filters=_full_filters(),
                intent="knowledge_explanation",
            )
        assert results == []
        assert "exhausted" in level
        # 4번(level 1~4) 호출
        assert mock_ss.call_count == 4

    @pytest.mark.asyncio
    async def test_fallback_order_matches_filter_keys(self):
        """_FALLBACK_FILTER_KEYS 순서대로 필터가 제거됨을 검증."""
        assert _FALLBACK_FILTER_KEYS == ("source_table", "team_id", "season_year")


# ── TestFallbackFilterIsolation ───────────────────────────────────────────────


class TestFallbackFilterIsolation:
    @pytest.mark.asyncio
    async def test_internal_inning_filter_preserved_across_levels(self):
        """내부 필터(_include_game_inning_scores)는 fallback에도 항상 유지."""
        filters = {
            "season_year": 2024,
            "team_id": "KIA",
            _INTERNAL_FILTER_INCLUDE_INNING_SCORES: True,
        }
        side_effects = [[], [MOCK_DOC]]  # Level 2에서 성공
        with patch(_PATCH_TARGET, side_effect=side_effects) as mock_ss:
            await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=filters
            )

        for call_args in mock_ss.call_args_list:
            f = call_args[1]["filters"] or {}
            assert f.get(_INTERNAL_FILTER_INCLUDE_INNING_SCORES) is True

    @pytest.mark.asyncio
    async def test_internal_exclude_source_filter_preserved(self):
        """내부 필터(_exclude_source_tables)는 fallback에도 항상 유지."""
        filters = {
            "season_year": 2024,
            _INTERNAL_FILTER_EXCLUDE_SOURCE_TABLES: ["game_logs"],
        }
        side_effects = [[], [MOCK_DOC]]
        with patch(_PATCH_TARGET, side_effect=side_effects) as mock_ss:
            await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=filters
            )

        for call_args in mock_ss.call_args_list:
            f = call_args[1]["filters"] or {}
            assert _INTERNAL_FILTER_EXCLUDE_SOURCE_TABLES in f

    @pytest.mark.asyncio
    async def test_none_filters_treated_as_empty(self):
        """filters=None 입력 시 빈 필터로 안전하게 처리됨."""
        with patch(_PATCH_TARGET, return_value=[MOCK_DOC]):
            results, level = await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=None
            )
        assert results == [MOCK_DOC]
        assert level == "level_1"

    @pytest.mark.asyncio
    async def test_empty_filters_dict(self):
        """filters={} 입력 시 level 1에서 즉시 반환."""
        with patch(_PATCH_TARGET, return_value=[MOCK_DOC]):
            results, level = await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters={}
            )
        assert level == "level_1"

    @pytest.mark.asyncio
    async def test_original_filters_not_mutated(self):
        """함수 호출 후 원본 filters dict가 수정되지 않음."""
        original = _full_filters()
        snapshot = dict(original)
        with patch(_PATCH_TARGET, return_value=[]):
            await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=original
            )
        assert original == snapshot


# ── TestFallbackMinResults ────────────────────────────────────────────────────


class TestFallbackMinResults:
    @pytest.mark.asyncio
    async def test_min_results_default_is_1(self):
        """기본 min_results=1: 결과가 1개 이상이면 반환."""
        with patch(_PATCH_TARGET, return_value=[MOCK_DOC]):
            results, _ = await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=_full_filters()
            )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_min_results_0_returns_immediately(self):
        """min_results=0: 빈 결과도 level 1에서 즉시 반환."""
        with patch(_PATCH_TARGET, return_value=[]) as mock_ss:
            results, level = await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=_full_filters(), min_results=0
            )
        assert results == []
        assert level == "level_1"
        mock_ss.assert_called_once()

    @pytest.mark.asyncio
    async def test_min_results_3_requires_enough_docs(self):
        """min_results=3: 결과가 3개 미만이면 계속 fallback."""
        two_docs = [MOCK_DOC, MOCK_DOC]
        three_docs = [MOCK_DOC, MOCK_DOC, MOCK_DOC]
        side_effects = [two_docs, three_docs]  # Level 1: 2개, Level 2: 3개
        with patch(_PATCH_TARGET, side_effect=side_effects) as mock_ss:
            results, level = await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=_full_filters(), min_results=3
            )
        assert len(results) == 3
        assert level == "level_2"
        assert mock_ss.call_count == 2


# ── TestFallbackReturnFormat ──────────────────────────────────────────────────


class TestFallbackReturnFormat:
    @pytest.mark.asyncio
    async def test_return_type_is_tuple(self):
        with patch(_PATCH_TARGET, return_value=[MOCK_DOC]):
            result = await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=_full_filters()
            )
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_first_element_is_list(self):
        with patch(_PATCH_TARGET, return_value=[MOCK_DOC]):
            results, _ = await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=_full_filters()
            )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_second_element_is_string(self):
        with patch(_PATCH_TARGET, return_value=[MOCK_DOC]):
            _, level = await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=_full_filters()
            )
        assert isinstance(level, str)

    @pytest.mark.parametrize(
        "num_empty,expected_level",
        [
            (0, "level_1"),
            (1, "level_2"),
            (2, "level_3"),
            (3, "level_4"),
        ],
    )
    @pytest.mark.asyncio
    async def test_level_string_format(self, num_empty, expected_level):
        """각 fallback 레벨이 올바른 문자열 형식으로 반환됨."""
        side_effects = [[]] * num_empty + [[MOCK_DOC]]
        with patch(_PATCH_TARGET, side_effect=side_effects):
            _, level = await similarity_search_with_fallback(
                CONN,
                DUMMY_EMBEDDING,
                limit=5,
                filters=_full_filters(),
                intent="knowledge_explanation",
            )
        assert level == expected_level

    @pytest.mark.asyncio
    async def test_exhausted_level_contains_exhausted(self):
        """모든 레벨 소진 시 반환 문자열에 'exhausted' 포함."""
        with patch(_PATCH_TARGET, return_value=[]):
            _, level = await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters=_full_filters()
            )
        assert "exhausted" in level

    @pytest.mark.asyncio
    async def test_limit_passed_through(self):
        """limit 파라미터가 similarity_search에 그대로 전달됨."""
        with patch(_PATCH_TARGET, return_value=[MOCK_DOC]) as mock_ss:
            await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=10, filters={}
            )
        assert mock_ss.call_args[1]["limit"] == 10

    @pytest.mark.asyncio
    async def test_keyword_passed_through(self):
        """keyword 파라미터가 similarity_search에 전달됨."""
        with patch(_PATCH_TARGET, return_value=[MOCK_DOC]) as mock_ss:
            await similarity_search_with_fallback(
                CONN, DUMMY_EMBEDDING, limit=5, filters={}, keyword="홈런"
            )
        assert mock_ss.call_args[1]["keyword"] == "홈런"


class TestFactualQueriesKeepCoreEntities:
    @pytest.mark.asyncio
    async def test_factual_fallback_stops_after_source_table(self):
        with patch(_PATCH_TARGET, return_value=[]) as mock_ss:
            results, level = await similarity_search_with_fallback(
                CONN,
                DUMMY_EMBEDDING,
                limit=5,
                filters=_full_filters(),
                intent="stats_lookup",
            )
        assert results == []
        assert "exhausted" in level
        assert mock_ss.call_count == 2
        for call in mock_ss.call_args_list:
            assert call[1]["filters"]["team_id"] == _full_filters()["team_id"]
            assert call[1]["filters"]["season_year"] == _full_filters()["season_year"]
