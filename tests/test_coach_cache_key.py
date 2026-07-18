"""coach_cache_key.py 단위 테스트.

Coach 전용 캐시 키 생성 모듈의 모든 공개 함수를 외부 의존성 없이 검증한다.
캐시 키 충돌 = 다른 사용자에게 잘못된 Coach 분석이 노출되는 보안 이슈이므로
해시 결정론·focus 정렬·lineup 순서 독립성 검증이 핵심이다.
"""
from __future__ import annotations

from app.core.coach_cache_key import (
    FOCUS_ORDER,
    FOCUS_SET,
    build_coach_cache_key,
    build_focus_signature,
    build_lineup_signature,
    build_question_signature,
    build_starter_signature,
    normalize_focus,
    normalize_question_override,
)
from app.core.coach_cache_contract import COACH_CACHE_SCHEMA_VERSION


# ── TestNormalizeFocus ────────────────────────────────────────────────────────

class TestNormalizeFocus:
    def test_valid_focus_returned(self):
        result = normalize_focus(["bullpen"])
        assert result == ["bullpen"]

    def test_invalid_focus_filtered_out(self):
        result = normalize_focus(["invalid_focus", "bullpen"])
        assert result == ["bullpen"]

    def test_none_returns_empty_list(self):
        assert normalize_focus(None) == []

    def test_empty_sequence_returns_empty_list(self):
        assert normalize_focus([]) == []

    def test_duplicates_removed(self):
        result = normalize_focus(["bullpen", "bullpen", "starter"])
        assert result.count("bullpen") == 1

    def test_uppercase_normalized(self):
        result = normalize_focus(["BULLPEN"])
        assert "bullpen" in result

    def test_canonical_order_preserved(self):
        result = normalize_focus(["batting", "recent_form", "bullpen"])
        expected_order = [f for f in FOCUS_ORDER if f in {"batting", "recent_form", "bullpen"}]
        assert result == expected_order

    def test_all_valid_focus_accepted(self):
        all_focus = list(FOCUS_SET)
        result = normalize_focus(all_focus)
        assert set(result) == FOCUS_SET


# ── TestNormalizeQuestionOverride ─────────────────────────────────────────────

class TestNormalizeQuestionOverride:
    def test_none_returns_none(self):
        assert normalize_question_override(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_question_override("") is None

    def test_whitespace_only_returns_none(self):
        assert normalize_question_override("   ") is None

    def test_consecutive_spaces_collapsed(self):
        result = normalize_question_override("KIA  분석")
        assert result == "KIA 분석"

    def test_normal_text_preserved(self):
        result = normalize_question_override("양현종 선발 분석")
        assert result == "양현종 선발 분석"


# ── TestBuildFocusSignature ───────────────────────────────────────────────────

class TestBuildFocusSignature:
    def test_none_returns_all(self):
        assert build_focus_signature(None) == "all"

    def test_empty_list_returns_all(self):
        assert build_focus_signature([]) == "all"

    def test_single_focus_returned(self):
        result = build_focus_signature(["bullpen"])
        assert result == "bullpen"

    def test_multiple_focus_joined_with_plus(self):
        result = build_focus_signature(["bullpen", "recent_form"])
        assert "+" in result
        assert "bullpen" in result
        assert "recent_form" in result

    def test_invalid_focus_produces_all(self):
        assert build_focus_signature(["nonexistent"]) == "all"


# ── TestBuildQuestionSignature ────────────────────────────────────────────────

class TestBuildQuestionSignature:
    def test_none_returns_auto(self):
        assert build_question_signature(None) == "auto"

    def test_empty_string_returns_auto(self):
        assert build_question_signature("") == "auto"

    def test_text_returns_q_prefix_hash(self):
        result = build_question_signature("양현종 분석")
        assert result.startswith("q:")
        assert len(result) == len("q:") + 16

    def test_deterministic(self):
        r1 = build_question_signature("테스트 질문")
        r2 = build_question_signature("테스트 질문")
        assert r1 == r2

    def test_different_texts_different_signatures(self):
        r1 = build_question_signature("질문 A")
        r2 = build_question_signature("질문 B")
        assert r1 != r2


# ── TestBuildStarterSignature ─────────────────────────────────────────────────

class TestBuildStarterSignature:
    def test_both_none_returns_digest_with_pending_prefix(self):
        # None|None → "|" → not empty → produces "starter_pending:{hash}"
        result = build_starter_signature(None, None)
        assert result.startswith("starter_pending:")

    def test_both_provided_returns_digest(self):
        result = build_starter_signature("양현종", "원태인")
        assert result.startswith("starter_pending:")
        assert len(result) == len("starter_pending:") + 12

    def test_deterministic(self):
        r1 = build_starter_signature("양현종", "원태인")
        r2 = build_starter_signature("양현종", "원태인")
        assert r1 == r2

    def test_different_pitchers_different_signatures(self):
        r1 = build_starter_signature("양현종", "원태인")
        r2 = build_starter_signature("김광현", "원태인")
        assert r1 != r2


# ── TestBuildLineupSignature ──────────────────────────────────────────────────

class TestBuildLineupSignature:
    def test_none_returns_pending(self):
        assert build_lineup_signature(None) == "lineup_pending"

    def test_empty_list_returns_pending(self):
        assert build_lineup_signature([]) == "lineup_pending"

    def test_players_returns_digest(self):
        result = build_lineup_signature(["김도영", "나성범"])
        assert result.startswith("lineup:")
        assert len(result) == len("lineup:") + 12

    def test_order_invariant(self):
        r1 = build_lineup_signature(["김도영", "나성범"])
        r2 = build_lineup_signature(["나성범", "김도영"])
        assert r1 == r2

    def test_different_players_different_signatures(self):
        r1 = build_lineup_signature(["김도영"])
        r2 = build_lineup_signature(["나성범"])
        assert r1 != r2


# ── TestBuildCoachCacheKey ────────────────────────────────────────────────────

def _default_key(**overrides):
    kwargs = dict(
        schema_version="v1",
        prompt_version="v2",
        home_team_code="KIA",
        away_team_code="LG",
        year=2025,
        game_type="REG",
        focus=["bullpen"],
        question_override=None,
    )
    kwargs.update(overrides)
    return build_coach_cache_key(**kwargs)


class TestBuildCoachCacheKey:
    def test_cache_contract_uses_readable_analysis_schema_version(self):
        assert COACH_CACHE_SCHEMA_VERSION == "coach_analysis_v2"

    def test_returns_tuple_of_two(self):
        result = _default_key()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_key_is_64_char_hex(self):
        key, _ = _default_key()
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_deterministic(self):
        k1, _ = _default_key()
        k2, _ = _default_key()
        assert k1 == k2

    def test_different_focus_different_key(self):
        k1, _ = _default_key(focus=["bullpen"])
        k2, _ = _default_key(focus=["starter"])
        assert k1 != k2

    def test_different_team_different_key(self):
        k1, _ = _default_key(home_team_code="KIA")
        k2, _ = _default_key(home_team_code="LG")
        assert k1 != k2

    def test_different_analysis_type_different_key(self):
        preview_key, preview_payload = _default_key(
            request_mode="auto_brief",
            analysis_type="game_preview",
        )
        review_key, review_payload = _default_key(
            request_mode="auto_brief",
            analysis_type="game_review",
        )

        assert preview_payload["analysis_type"] == "game_preview"
        assert review_payload["analysis_type"] == "game_review"
        assert preview_key != review_key

    def test_payload_contains_schema(self):
        _, payload = _default_key()
        assert payload["schema"] == "v1"

    def test_payload_contains_focus_signature(self):
        _, payload = _default_key(focus=["bullpen"])
        assert "focus_signature" in payload
        assert payload["focus_signature"] == "bullpen"

    def test_team_code_uppercased_in_payload(self):
        _, payload = _default_key(home_team_code="kia")
        assert payload["team_code_canonical"] == "KIA"
