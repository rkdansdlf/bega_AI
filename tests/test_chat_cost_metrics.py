"""Tests for chat token and cost attribution helpers."""

from __future__ import annotations

from app.core import chat_cost_metrics


class _CounterRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, str], float]] = []
        self._labels: dict[str, str] = {}

    def labels(self, **labels: str) -> "_CounterRecorder":
        self._labels = labels
        return self

    def inc(self, value: float = 1.0) -> None:
        self.calls.append((dict(self._labels), value))


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
