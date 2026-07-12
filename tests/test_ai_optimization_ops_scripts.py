"""Tests for AI optimization operation scripts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts import chat_model_routing_experiment
from scripts.cleanup_chat_semantic_cache import CleanupScope, build_where_clause


def test_model_routing_runbook_and_ci_contract() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    runbook = (
        repository_root / "docs" / "ai-optimization-rollout-runbook.md"
    ).read_text(encoding="utf-8")
    env_example = (repository_root / ".env.example").read_text(encoding="utf-8")
    workflow = (
        repository_root / ".github" / "workflows" / "ai-pr-gate.yml"
    ).read_text(encoding="utf-8")
    normalized_runbook = " ".join(runbook.lower().split())
    pricing_line = next(
        line
        for line in env_example.splitlines()
        if line.startswith("CHAT_MODEL_PRICING_JSON=")
    )
    pricing_catalog = json.loads(pricing_line.split("=", 1)[1])

    assert "CHAT_MODEL_PRICING_JSON" in env_example
    assert "CHAT_MODEL_PRICING_JSON" in runbook
    assert pricing_catalog == {
        "example-provider": {
            "example/planner-model": {
                "input_usd_per_1m_tokens": "0.10",
                "output_usd_per_1m_tokens": "0.20",
            },
            "example/answer-model": {
                "input_usd_per_1m_tokens": "1.00",
                "output_usd_per_1m_tokens": "2.00",
            },
        },
    }
    serialized_pricing_catalog = json.dumps(pricing_catalog).lower()
    assert "default" not in serialized_pricing_catalog
    assert "unknown" not in serialized_pricing_catalog
    assert "credential" not in serialized_pricing_catalog
    assert "api_key" not in serialized_pricing_catalog
    assert "Golden path" in runbook
    assert "scripts/chat_quality_golden_60.json" in runbook
    assert "AI_MODEL_ROUTING_OUTPUT=outputs/model-routing/baseline.json" in runbook
    assert "AI_MODEL_ROUTING_OUTPUT=outputs/model-routing/candidate.json" in runbook
    assert "--baseline-report outputs/model-routing/baseline.json" in runbook
    assert "--planner-model-label \"$CHAT_PLANNER_MODEL_NAME\"" in runbook
    assert "--answer-model-label \"$CHAT_ANSWER_MODEL_NAME\"" in runbook
    assert "planner reduction of at least 20%" in runbook
    assert "candidate total model cost must not increase" in runbook
    assert "answer model must remain fixed" in normalized_runbook
    assert "Cache bypass is required and defaults to enabled" in runbook
    assert "defaults to 60 cases" in runbook
    assert "SHA-256" in runbook
    assert "exact provider/model catalog entries" in runbook
    assert "0: valid evidence and all gates pass" in runbook
    assert "1: valid evidence but a quality or cost criterion fails" in runbook
    assert "2: invalid or incomplete evidence, configuration, input, or report" in runbook
    assert "separate live-call approval" in runbook
    assert "does not authorize deployment" in runbook
    assert "must not contain answers" in runbook
    assert (
        "cli planner/answer labels are evidence assertions only and do not configure "
        "the server."
        in normalized_runbook
    )
    assert (
        "before the baseline, configure the controlled ai service with "
        "`chat_planner_model_name`, `chat_answer_model_name`, and the matching "
        "`chat_model_pricing_json` catalog."
        in normalized_runbook
    )
    assert "restart or redeploy the controlled ai service" in normalized_runbook
    assert "verify its health and active configuration" in normalized_runbook
    assert (
        "before the candidate, change only `chat_planner_model_name` and its "
        "catalog entry as needed; keep `chat_answer_model_name` fixed."
        in normalized_runbook
    )
    assert (
        "absent usage is invalid only when the selected planner or answer path requires "
        "an llm call."
        in normalized_runbook
    )
    assert (
        "a deterministic mode may validly make no model call and therefore have empty "
        "usage."
        in normalized_runbook
    )
    assert "Actual deployment remains separately approved" in runbook

    for test_path in (
        "tests/test_chat_model_usage.py",
        "tests/test_chat_model_routing_experiment.py",
        "tests/test_chat_quality_golden_dataset.py",
    ):
        assert test_path in workflow
    assert "scripts/chat_model_routing_experiment.py" not in workflow


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


def test_chat_model_routing_cli_keeps_operator_contract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chat_model_routing_experiment.py",
            "--samples",
            "golden.json",
            "--baseline-report",
            "baseline.json",
            "--planner-model-label",
            "planner-v2",
            "--answer-model-label",
            "answer-v1",
        ],
    )

    args = chat_model_routing_experiment.parse_args()

    assert args.samples == "golden.json"
    assert args.baseline_report == "baseline.json"
    assert args.planner_model_label == "planner-v2"
    assert args.answer_model_label == "answer-v1"
    assert args.internal_api_key_env == "AI_INTERNAL_TOKEN"
    assert args.limit == 60
    assert args.cache_bypass is True


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
