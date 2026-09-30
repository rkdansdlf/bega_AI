from app.core.semantic_cache_equivalence import (
    check_equivalence,
    extract_teams,
    extract_years,
)
from scripts.semantic_cache_shadow_eval import build_report


def test_extractors():
    assert extract_years("2025년 LG와 2024시즌") == {2025, 2024}
    assert extract_teams("LG 트윈스와 기아타이거즈") == {"LG", "KIA"}


def test_same_scope_is_equivalent():
    assert (
        check_equivalence(
            request_question="2025 LG 성적",
            cached_question="LG 2025 시즌 성적 알려줘",
            cached_answer="2025 LG 는 85승",
            fresh_answer="2025 LG 85승",
        )
        == []
    )


def test_different_season_same_numbers_is_rejected():
    reasons = check_equivalence(
        request_question="2025 LG 성적",
        cached_question="2024 LG 성적",
        cached_answer="2024 LG 는 85승",
        fresh_answer="2025 LG 는 85승",
    )
    assert "season_scope_mismatch" in reasons
    assert "answer_season_mismatch" in reasons


def test_different_team_is_rejected():
    reasons = check_equivalence(
        request_question="2025 LG 성적",
        cached_question="2025 KIA 성적",
        cached_answer="2025 KIA 는 85승",
        fresh_answer="2025 LG 는 85승",
    )
    assert "team_scope_mismatch" in reasons
    assert "answer_team_mismatch" in reasons


def test_answerability_source_and_freshness_mismatch():
    reasons = check_equivalence(
        cached_answer="MANUAL_BASEBALL_DATA_REQUIRED",
        fresh_answer="LG는 85승입니다",
        cached_provenance={
            "data_sources": [{"id": "a"}],
            "as_of_date": "2025-09-01",
        },
        fresh_provenance={
            "data_sources": [{"id": "b"}],
            "as_of_date": "2025-09-20",
        },
    )
    assert {
        "answerability_mismatch",
        "source_provenance_mismatch",
        "freshness_mismatch",
    } <= set(reasons)


def test_missing_provenance_is_not_a_mismatch():
    assert check_equivalence(cached_answer="LG 85승", fresh_answer="LG 85승") == []


def test_shadow_report_fails_sample_on_scope_mismatch():
    sample_ok = {
        "question": "2025 LG 성적",
        "cached_question": "LG 2025 성적",
        "cached_answer": "2025 LG 는 85승 57패 입니다",
        "fresh_answer": "2025 LG 는 85승 57패 입니다",
    }
    sample_bad = dict(sample_ok, cached_question="2024 LG 성적")
    report = build_report([sample_ok, sample_bad], min_token_jaccard=0.5)
    assert [d["status"] for d in report["details"]] == ["passed", "failed"]
    assert "season_scope_mismatch" in report["details"][1]["failure_reasons"]
