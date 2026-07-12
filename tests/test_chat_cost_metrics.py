"""Tests for chat token and cost attribution helpers."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.core import chat_cost_metrics
from app.core.chat_model_usage import ModelUsageEstimate


class _CounterRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, str], float]] = []
        self._labels: dict[str, str] = {}

    def labels(self, **labels: str) -> "_CounterRecorder":
        self._labels = labels
        return self

    def inc(self, value: float = 1.0) -> None:
        self.calls.append((dict(self._labels), value))


def _usage_record(
    *,
    model: str = " vendor/planner ",
    outcome: str = "success",
    pricing_source: str = "model_catalog",
    total_cost_usd: Decimal | None = Decimal("0.000012"),
) -> ModelUsageEstimate:
    return ModelUsageEstimate(
        role="planner",
        provider=" openrouter ",
        model=model,
        outcome=outcome,
        pricing_source=pricing_source,
        input_chars=28,
        output_chars=7,
        input_tokens=8,
        output_tokens=2,
        input_cost_usd=Decimal("0.000008") if total_cost_usd is not None else None,
        output_cost_usd=Decimal("0.000004") if total_cost_usd is not None else None,
        total_cost_usd=total_cost_usd,
    )


def test_classify_chat_question_type_uses_stable_low_cardinality_buckets() -> None:
    assert (
        chat_cost_metrics.classify_chat_question_type(
            "김도영, 문보경, 노시환의 타석 접근을 서사적으로 풀어줘."
        )
        == "multi_player_narrative"
    )
    assert (
        chat_cost_metrics.classify_chat_question_type("FA 보상선수 규정 핵심이 뭐야?")
        == "regulation"
    )
    assert (
        chat_cost_metrics.classify_chat_question_type("LG와 KT를 비교해줘?")
        == "team_analysis"
    )
    assert (
        chat_cost_metrics.classify_chat_question_type("김도영의 2026년 기록 알려줘.")
        == "player_analysis"
    )
    assert (
        chat_cost_metrics.classify_chat_question_type("오늘 볼 만한 건 뭐야?")
        == "general"
    )


def test_record_chat_token_estimate_attributes_type_and_planner(monkeypatch) -> None:
    legacy_tokens = _CounterRecorder()
    legacy_cost = _CounterRecorder()
    typed_tokens = _CounterRecorder()
    typed_cost = _CounterRecorder()
    monkeypatch.setattr(
        chat_cost_metrics, "AI_CHAT_TOKEN_ESTIMATE_TOTAL", legacy_tokens
    )
    monkeypatch.setattr(
        chat_cost_metrics, "AI_CHAT_COST_ESTIMATE_USD_TOTAL", legacy_cost
    )
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_CHAT_TOKEN_ESTIMATE_BY_TYPE_TOTAL",
        typed_tokens,
        raising=False,
    )
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_CHAT_COST_ESTIMATE_BY_TYPE_USD_TOTAL",
        typed_cost,
        raising=False,
    )

    settings = type(
        "Settings",
        (),
        {
            "llm_provider": "openrouter",
            "chatbot_model_name": "test-model",
            "chat_cost_input_usd_per_1m_tokens": 1.0,
            "chat_cost_output_usd_per_1m_tokens": 2.0,
        },
    )()
    chat_cost_metrics.record_chat_token_estimate(
        settings,
        route="completion",
        cache_state="generated",
        question="FA 보상선수 규정 핵심이 뭐야?",
        answer="내부 규정 문서 기준 답변입니다.",
        question_type="regulation",
        planner_mode="fast_path",
    )

    assert len(typed_tokens.calls) == 2
    assert typed_tokens.calls[0][0] == {
        "route": "completion",
        "question_type": "regulation",
        "planner_mode": "fast_path",
        "token_type": "input",
        "cache_state": "generated",
    }
    assert typed_cost.calls[0][0] == {
        "route": "completion",
        "question_type": "regulation",
        "planner_mode": "fast_path",
        "provider": "openrouter",
        "model": "test-model",
    }


def test_record_model_usage_estimate_records_priced_attempt(monkeypatch) -> None:
    token_counter = _CounterRecorder()
    cost_counter = _CounterRecorder()
    outcome_counter = _CounterRecorder()
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_MODEL_USAGE_TOKEN_ESTIMATE_TOTAL",
        token_counter,
        raising=False,
    )
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_MODEL_USAGE_COST_ESTIMATE_USD_TOTAL",
        cost_counter,
        raising=False,
    )
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_MODEL_USAGE_OUTCOME_TOTAL",
        outcome_counter,
        raising=False,
    )

    record = _usage_record()
    chat_cost_metrics.record_model_usage_estimate(record)

    assert token_counter.calls[0] == (
        {
            "role": "planner",
            "provider": "openrouter",
            "model": "vendor/planner",
            "token_type": "input",
            "outcome": "success",
        },
        8,
    )
    assert token_counter.calls[1][0]["token_type"] == "output"
    assert token_counter.calls[1][1] == 2
    assert outcome_counter.calls == [
        (
            {
                "role": "planner",
                "provider": "openrouter",
                "model": "vendor/planner",
                "result": "priced",
            },
            1.0,
        )
    ]
    assert cost_counter.calls[0][1] == pytest.approx(float(record.total_cost_usd))


def test_record_model_usage_estimate_records_unpriced_attempt_without_cost(
    monkeypatch,
) -> None:
    token_counter = _CounterRecorder()
    cost_counter = _CounterRecorder()
    outcome_counter = _CounterRecorder()
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_MODEL_USAGE_TOKEN_ESTIMATE_TOTAL",
        token_counter,
        raising=False,
    )
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_MODEL_USAGE_COST_ESTIMATE_USD_TOTAL",
        cost_counter,
        raising=False,
    )
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_MODEL_USAGE_OUTCOME_TOTAL",
        outcome_counter,
        raising=False,
    )

    chat_cost_metrics.record_model_usage_estimate(
        _usage_record(pricing_source="unpriced", total_cost_usd=None)
    )

    assert len(token_counter.calls) == 2
    assert cost_counter.calls == []
    assert outcome_counter.calls[0][0]["result"] == "unpriced"


def test_record_model_usage_estimate_collapses_arbitrary_unpriced_models(
    monkeypatch,
) -> None:
    token_counter = _CounterRecorder()
    cost_counter = _CounterRecorder()
    outcome_counter = _CounterRecorder()
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_MODEL_USAGE_TOKEN_ESTIMATE_TOTAL",
        token_counter,
        raising=False,
    )
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_MODEL_USAGE_COST_ESTIMATE_USD_TOTAL",
        cost_counter,
        raising=False,
    )
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_MODEL_USAGE_OUTCOME_TOTAL",
        outcome_counter,
        raising=False,
    )

    for index in range(200):
        chat_cost_metrics.record_model_usage_estimate(
            _usage_record(
                model=f"unpriced/provider-model-{index}",
                pricing_source="unpriced",
                total_cost_usd=None,
            )
        )

    assert {labels["model"] for labels, _ in token_counter.calls} == {"unknown"}
    assert {labels["model"] for labels, _ in outcome_counter.calls} == {"unknown"}
    assert cost_counter.calls == []


def test_record_model_usage_estimate_keeps_failed_partial_priced_attempt(
    monkeypatch,
) -> None:
    token_counter = _CounterRecorder()
    cost_counter = _CounterRecorder()
    outcome_counter = _CounterRecorder()
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_MODEL_USAGE_TOKEN_ESTIMATE_TOTAL",
        token_counter,
        raising=False,
    )
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_MODEL_USAGE_COST_ESTIMATE_USD_TOTAL",
        cost_counter,
        raising=False,
    )
    monkeypatch.setattr(
        chat_cost_metrics,
        "AI_MODEL_USAGE_OUTCOME_TOTAL",
        outcome_counter,
        raising=False,
    )

    record = _usage_record(outcome="failed")
    chat_cost_metrics.record_model_usage_estimate(record)

    assert token_counter.calls[0][0]["outcome"] == "failed"
    assert outcome_counter.calls[0][0]["result"] == "failed"
    assert cost_counter.calls[0][1] == pytest.approx(float(record.total_cost_usd))
