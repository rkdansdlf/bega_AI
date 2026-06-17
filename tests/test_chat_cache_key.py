"""chat_cache_key.py 단위 테스트.

모든 공개 함수와 내부 정규화 헬퍼를 외부 의존성 없이 검증한다.
캐시 키 충돌 시 다른 사용자에게 잘못된 응답이 노출될 수 있으므로
해시 결정론·필터 키 순서 독립성 검증이 핵심이다.
"""
from __future__ import annotations

from datetime import date

from app.core.chat_cache_key import (
    CHAT_CACHE_SCHEMA_VERSION,
    DEFAULT_TTL_SECONDS,
    INTENT_TTL_SECONDS,
    _normalize_filters,
    _normalize_question,
    build_chat_cache_key,
    get_ttl_seconds,
    has_temporal_keyword,
)


# ── TestGetTTLSeconds ─────────────────────────────────────────────────────────

class TestGetTTLSeconds:
    def test_stats_lookup_returns_correct_ttl(self):
        assert get_ttl_seconds("stats_lookup") == INTENT_TTL_SECONDS["stats_lookup"]

    def test_player_profile_returns_correct_ttl(self):
        assert get_ttl_seconds("player_profile") == INTENT_TTL_SECONDS["player_profile"]

    def test_comparison_returns_correct_ttl(self):
        assert get_ttl_seconds("comparison") == INTENT_TTL_SECONDS["comparison"]

    def test_knowledge_explanation_returns_correct_ttl(self):
        assert get_ttl_seconds("knowledge_explanation") == INTENT_TTL_SECONDS["knowledge_explanation"]

    def test_unknown_intent_returns_default(self):
        assert get_ttl_seconds("nonexistent_intent") == DEFAULT_TTL_SECONDS

    def test_none_intent_returns_default(self):
        assert get_ttl_seconds(None) == DEFAULT_TTL_SECONDS

    def test_empty_string_intent_returns_default(self):
        assert get_ttl_seconds("") == DEFAULT_TTL_SECONDS

    def test_player_profile_longer_than_stats_lookup(self):
        assert get_ttl_seconds("player_profile") > get_ttl_seconds("stats_lookup")

    def test_comparison_longer_than_recent_form(self):
        assert get_ttl_seconds("comparison") > get_ttl_seconds("recent_form")


# ── TestHasTemporalKeyword ────────────────────────────────────────────────────

class TestHasTemporalKeyword:
    def test_오늘_returns_true(self):
        assert has_temporal_keyword("오늘 경기 결과") is True

    def test_지금_returns_true(self):
        assert has_temporal_keyword("지금 1위 팀이 어디야?") is True

    def test_현재_returns_true(self):
        assert has_temporal_keyword("현재 KIA 순위") is True

    def test_최근_returns_true(self):
        assert has_temporal_keyword("최근 KIA 성적") is True

    def test_올시즌_returns_true(self):
        assert has_temporal_keyword("올시즌 홈런 1위") is True

    def test_실시간_returns_true(self):
        assert has_temporal_keyword("실시간 순위") is True

    def test_전망_returns_true(self):
        assert has_temporal_keyword("KIA 전망 어때") is True

    def test_month_day_pattern_returns_true(self):
        assert has_temporal_keyword("5월 15일 경기 결과") is True

    def test_date_hyphen_pattern_returns_true(self):
        assert has_temporal_keyword("2024-05-10 경기") is True

    def test_non_temporal_returns_false(self):
        assert has_temporal_keyword("KIA 홈런 순위") is False

    def test_year_only_no_date_returns_false(self):
        assert has_temporal_keyword("2024 시즌 ERA 순위") is False

    def test_current_year_leaderboard_returns_true(self):
        assert has_temporal_keyword(f"{date.today().year}년 홈런왕은 누구야?") is True

    def test_empty_string_returns_false(self):
        assert has_temporal_keyword("") is False

    def test_pure_historical_returns_false(self):
        assert has_temporal_keyword("류현진 통산 ERA") is False

    def test_series_without_temporal_returns_false(self):
        assert has_temporal_keyword("한국시리즈 역대 MVP 알려줘") is False

    def test_series_with_오늘_returns_true(self):
        assert has_temporal_keyword("오늘 시리즈 현황 알려줘") is True

    def test_series_with_실시간_returns_true(self):
        assert has_temporal_keyword("실시간 시리즈 중계 알려줘") is True


# ── TestBuildChatCacheKey ─────────────────────────────────────────────────────

class TestBuildChatCacheKey:
    def test_schema_version_bumped_for_256_embedding_rollout(self):
        assert CHAT_CACHE_SCHEMA_VERSION == "v12"

    def test_returns_tuple_of_two(self):
        result = build_chat_cache_key(question="KIA 홈런")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_key_is_64_char_hex(self):
        key, _ = build_chat_cache_key(question="KIA 홈런")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_deterministic_same_input_same_key(self):
        k1, _ = build_chat_cache_key(question="KIA 홈런", filters={"season_year": 2024})
        k2, _ = build_chat_cache_key(question="KIA 홈런", filters={"season_year": 2024})
        assert k1 == k2

    def test_different_questions_different_keys(self):
        k1, _ = build_chat_cache_key(question="KIA 홈런")
        k2, _ = build_chat_cache_key(question="삼성 ERA")
        assert k1 != k2

    def test_different_filters_different_keys(self):
        k1, _ = build_chat_cache_key(question="q", filters={"season_year": 2024})
        k2, _ = build_chat_cache_key(question="q", filters={"season_year": 2023})
        assert k1 != k2

    def test_filter_key_order_invariant(self):
        k_a, _ = build_chat_cache_key(question="q", filters={"a": 1, "b": 2})
        k_b, _ = build_chat_cache_key(question="q", filters={"b": 2, "a": 1})
        assert k_a == k_b

    def test_none_filters_equals_empty_filters(self):
        k_none, _ = build_chat_cache_key(question="q", filters=None)
        k_empty, _ = build_chat_cache_key(question="q", filters={})
        assert k_none == k_empty

    def test_case_insensitive(self):
        k_upper, _ = build_chat_cache_key(question="KIA 홈런")
        k_lower, _ = build_chat_cache_key(question="kia 홈런")
        assert k_upper == k_lower

    def test_whitespace_normalized(self):
        k_single, _ = build_chat_cache_key(question="KIA 홈런")
        k_double, _ = build_chat_cache_key(question="KIA  홈런")
        assert k_single == k_double

    def test_payload_contains_normalized_question(self):
        _, payload = build_chat_cache_key(question="KIA  홈런")
        assert payload["question"] == "kia 홈런"

    def test_payload_contains_schema_version(self):
        _, payload = build_chat_cache_key(question="q")
        assert payload["schema"] == CHAT_CACHE_SCHEMA_VERSION

    def test_payload_none_filter_value_removed(self):
        _, payload = build_chat_cache_key(
            question="q", filters={"season_year": 2024, "team_id": None}
        )
        assert "team_id" not in payload["filters"]
        assert payload["filters"]["season_year"] == 2024

    def test_payload_has_filters_key(self):
        _, payload = build_chat_cache_key(question="q")
        assert "filters" in payload

    def test_custom_schema_version_reflected_in_payload(self):
        _, payload = build_chat_cache_key(question="q", schema_version="v99")
        assert payload["schema"] == "v99"

    def test_custom_schema_version_changes_key(self):
        k_default, _ = build_chat_cache_key(question="q")
        k_custom, _ = build_chat_cache_key(question="q", schema_version="v99")
        assert k_default != k_custom


# ── TestNormalizeFilters ──────────────────────────────────────────────────────

class TestNormalizeFilters:
    def test_none_value_removed(self):
        result = _normalize_filters({"a": 1, "b": None})
        assert "b" not in result
        assert result["a"] == 1

    def test_empty_string_removed(self):
        result = _normalize_filters({"a": "hello", "b": ""})
        assert "b" not in result
        assert result["a"] == "hello"

    def test_empty_list_removed(self):
        result = _normalize_filters({"a": [1, 2], "b": []})
        assert "b" not in result
        assert result["a"] == [1, 2]

    def test_keys_sorted(self):
        result = _normalize_filters({"z": 3, "a": 1, "m": 2})
        assert list(result.keys()) == ["a", "m", "z"]

    def test_valid_values_preserved(self):
        result = _normalize_filters({"season_year": 2024, "team_id": "KIA"})
        assert result["season_year"] == 2024
        assert result["team_id"] == "KIA"

    def test_none_input_returns_empty_dict(self):
        assert _normalize_filters(None) == {}

    def test_empty_dict_returns_empty_dict(self):
        assert _normalize_filters({}) == {}


# ── TestNormalizeQuestion ─────────────────────────────────────────────────────

class TestNormalizeQuestion:
    def test_uppercased_lowercased(self):
        result = _normalize_question("KIA ERA")
        assert result == result.lower()

    def test_consecutive_spaces_collapsed(self):
        result = _normalize_question("KIA  홈런")
        assert "  " not in result

    def test_leading_trailing_stripped(self):
        result = _normalize_question("  KIA  ")
        assert result == result.strip()

    def test_empty_returns_empty(self):
        assert _normalize_question("") == ""

    def test_mixed_case_and_spaces_normalized(self):
        assert _normalize_question("KIA  홈런") == "kia 홈런"
