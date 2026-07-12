from __future__ import annotations

import json

from scripts import semantic_cache_shadow_eval as shadow_eval


def test_compare_answers_reports_token_overlap() -> None:
    comparison = shadow_eval.compare_answers(
        "LG 타선 흐름은 안정적입니다.",
        "LG 타선 흐름은 안정적입니다.",
    )

    assert comparison["token_jaccard"] == 1.0
    assert comparison["cached_chars"] > 0
    assert comparison["fresh_chars"] > 0


def test_build_report_flags_low_similarity_candidate() -> None:
    report = shadow_eval.build_report(
        [
            {
                "cache_key": "hit-1",
                "question": "LG 분위기",
                "cached_answer": "LG 타선 흐름은 안정적입니다.",
                "fresh_answer": "KIA 불펜 소모가 큽니다.",
            }
        ],
        min_token_jaccard=0.72,
    )

    assert report["summary"]["sample_count"] == 1
    assert report["summary"]["compared_count"] == 1
    assert report["summary"]["failed_count"] == 1
    assert report["summary"]["potential_false_positive_rate"] == 1.0
    assert report["details"][0]["status"] == "failed"


def test_build_report_rejects_stale_numeric_fact_despite_high_overlap() -> None:
    report = shadow_eval.build_report(
        [
            {
                "cache_key": "stale-number",
                "question": "LG 현재 승률 알려줘.",
                "cached_answer": "LG의 현재 승률은 0.600이며 최근 흐름은 안정적입니다.",
                "fresh_answer": "LG의 현재 승률은 0.500이며 최근 흐름은 안정적입니다.",
            }
        ],
        min_token_jaccard=0.50,
    )

    assert report["summary"]["failed_count"] == 1
    assert report["details"][0]["status"] == "failed"
    assert "numeric_fact_mismatch" in report["details"][0]["failure_reasons"]


def test_build_report_rejects_manual_contract_mismatch() -> None:
    report = shadow_eval.build_report(
        [
            {
                "cache_key": "manual-contract",
                "question": "선수의 숫자 밖 리더십을 알려줘.",
                "cached_answer": "경험이 많은 멘토 역할을 수행합니다.",
                "fresh_answer": "MANUAL_BASEBALL_DATA_REQUIRED: 내부 평가 자료가 필요합니다.",
            }
        ],
        min_token_jaccard=0.0,
    )

    assert report["summary"]["failed_count"] == 1
    assert "manual_contract_mismatch" in report["details"][0]["failure_reasons"]


def test_build_report_fails_coverage_when_comparison_answer_is_missing() -> None:
    report = shadow_eval.build_report(
        [
            {
                "cache_key": "missing-fresh",
                "question": "LG 흐름 알려줘.",
                "cached_answer": "LG 흐름은 안정적입니다.",
            }
        ],
        min_token_jaccard=0.72,
    )

    assert report["summary"]["skipped_count"] == 1
    assert report["summary"]["coverage_complete"] is False
    assert report["summary"]["gate_passed"] is False
    assert report["details"][0]["failure_reasons"] == ["missing_comparison_answer"]


def test_compare_answers_normalizes_decimal_and_percent_rates() -> None:
    comparison = shadow_eval.compare_answers(
        "LG의 승률은 0.600입니다.",
        "LG의 승률은 60%입니다.",
    )

    assert comparison["numeric_facts_match"] is True
    assert comparison["numeric_bindings_match"] is True


def test_build_report_rejects_numbers_swapped_between_players() -> None:
    report = shadow_eval.build_report(
        [
            {
                "question": "김도영과 문보경 타율 비교해줘.",
                "cached_answer": "김도영 타율 0.300, 문보경 타율 0.280입니다.",
                "fresh_answer": "김도영 타율 0.280, 문보경 타율 0.300입니다.",
            }
        ],
        min_token_jaccard=0.50,
    )

    assert report["summary"]["failed_count"] == 1
    assert "numeric_binding_mismatch" in report["details"][0]["failure_reasons"]


def test_load_samples_accepts_json_list(tmp_path) -> None:
    sample_path = tmp_path / "samples.json"
    sample_path.write_text(
        json.dumps([{"question": "LG 분위기", "cached_answer": "cached"}]),
        encoding="utf-8",
    )

    samples = shadow_eval.load_samples(sample_path)

    assert samples == [{"question": "LG 분위기", "cached_answer": "cached"}]


def test_load_samples_accepts_jsonl(tmp_path) -> None:
    sample_path = tmp_path / "samples.jsonl"
    sample_path.write_text(
        '{"question": "LG 분위기", "cached_answer": "cached"}\n',
        encoding="utf-8",
    )

    samples = shadow_eval.load_samples(sample_path)

    assert samples == [{"question": "LG 분위기", "cached_answer": "cached"}]
