from __future__ import annotations

import builtins
import json
import sys

import pytest

from scripts import smoke_chatbot_quality as smoke


def test_summarize_results_includes_latency_and_error_metrics() -> None:
    results = [
        {
            "endpoint": "/ai/chat/completion",
            "ok": True,
            "latency_ms": 100.0,
            "error": None,
            "status_code": 200,
            "quality": {
                "table_present": True,
                "section_present": True,
                "source_present": True,
            },
        },
        {
            "endpoint": "/ai/chat/completion",
            "ok": False,
            "latency_ms": 300.0,
            "error": "timeout while waiting response",
            "status_code": None,
            "quality": {
                "table_present": False,
                "section_present": False,
                "source_present": False,
            },
        },
        {
            "endpoint": "/ai/chat/stream",
            "ok": True,
            "latency_ms": 150.0,
            "error": None,
            "status_code": 200,
            "quality": {
                "table_present": True,
                "section_present": True,
                "source_present": True,
            },
        },
    ]

    summary = smoke._summarize_results(results, stream_fallback_ratio_max=0.40)

    assert "completion_metrics" in summary
    assert "stream_metrics" in summary
    assert summary["completion_metrics"]["latency_ms"]["p95"] is not None
    assert summary["completion_metrics"]["timeout_rate"] > 0.0
    assert summary["overall_error_rate"] > 0.0


def test_summarize_results_includes_planner_distribution_and_failed_cases() -> None:
    results = [
        {
            "endpoint": "/ai/chat/completion",
            "question": "질문 1",
            "ok": True,
            "latency_ms": 120.0,
            "error": None,
            "status_code": 200,
            "cached": False,
            "quality_pass": True,
            "quality": {},
            "sample_response": {
                "planner_mode": "default_llm_planner",
                "planner_cache_hit": True,
                "tool_execution_mode": "parallel",
                "perf": {
                    "planner_cache_hit": True,
                    "tool_execution_mode": "parallel",
                },
            },
        },
        {
            "endpoint": "/ai/chat/stream",
            "question": "질문 2",
            "ok": False,
            "latency_ms": 210.0,
            "error": "status=500",
            "status_code": 500,
            "cached": False,
            "quality_pass": False,
            "quality": {},
            "meta": {
                "planner_mode": "fast_path",
                "planner_cache_hit": False,
                "tool_execution_mode": "sequential",
                "fallback_reason": "temporary_generation_issue",
                "perf": {
                    "planner_cache_hit": False,
                    "tool_execution_mode": "sequential",
                },
            },
        },
    ]

    summary = smoke._summarize_results(results, stream_fallback_ratio_max=0.40)

    assert summary["planner_bucket_distribution"]["overall"]["llm"]["ratio"] == 0.5
    assert (
        summary["planner_bucket_distribution"]["overall"]["fast_path"]["ratio"] == 0.5
    )
    assert summary["planner_cache_metrics"]["overall"]["hit_count"] == 1
    assert (
        summary["tool_execution_mode_distribution"]["overall"]["parallel"]["count"]
        == 1
    )
    assert len(summary["failed_cases"]) == 1
    failed_case = summary["failed_cases"][0]
    assert failed_case["planner_mode"] == "fast_path"
    assert failed_case["planner_cache_hit"] is False
    assert failed_case["tool_execution_mode"] == "sequential"


def test_summarize_results_parses_stream_meta_from_sse_sample_response() -> None:
    stream_meta = {
        "planner_mode": "cache",
        "planner_cache_hit": False,
        "tool_execution_mode": "none",
        "fallback_reason": None,
        "perf": {
            "planner_mode": "cache",
            "planner_cache_hit": False,
            "tool_execution_mode": "none",
            "first_token_ms": 0.0,
            "total_ms": 0.0,
        },
    }
    results = [
        {
            "endpoint": "/ai/chat/stream",
            "question": "질문 3",
            "ok": True,
            "latency_ms": 180.0,
            "error": None,
            "status_code": 200,
            "cached": True,
            "quality_pass": True,
            "quality": {},
            "meta": {
                "cached": True,
                "perf": {
                    "first_token_ms": 0.0,
                    "total_ms": 0.0,
                },
            },
            "sample_response": "\\n".join(
                [
                    'event: status',
                    'data: {"message":"⚡"}',
                    'event: meta',
                    f"data: {json.dumps(stream_meta, ensure_ascii=False)}",
                    'event: done',
                    'data: [DONE]',
                ]
            ),
        }
    ]

    summary = smoke._summarize_results(results, stream_fallback_ratio_max=0.40)

    assert summary["planner_mode_distribution"]["stream"]["cache"]["count"] == 1
    assert summary["planner_cache_metrics"]["stream"]["miss_count"] == 1
    assert summary["tool_execution_mode_distribution"]["stream"]["none"]["count"] == 1
    assert summary["perf_metrics"]["stream"]["first_token_ms"]["count"] == 1
    assert summary["perf_metrics"]["stream"]["total_ms"]["max"] == 0.0


def test_check_stream_stops_reading_immediately_after_done_event() -> None:
    class _FakeResponse:
        status_code = 200

        def iter_lines(self):
            yield "event: status"
            yield 'data: {"message":"⏺️"}'
            yield "event: message"
            yield 'data: {"delta":"첫 청크"}'
            yield "event: meta"
            yield (
                'data: {"cached": false, "perf": {"first_token_ms": 1.0, "total_ms": 2.0}}'
            )
            yield "event: done"
            yield "data: [DONE]"
            raise AssertionError("iter_lines should stop after [DONE]")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeClient:
        def stream(self, *args, **kwargs):
            del args, kwargs
            return _FakeResponse()

    result = smoke._check_stream(
        _FakeClient(),
        "http://127.0.0.1:8001",
        "질문 4",
        history_payload=None,
        headers={"X-Internal-Api-Key": "token"},
        rate_limit_retries=0,
        rate_limit_base_delay=0.1,
        stream_style="markdown",
    )

    assert result["ok"] is True
    assert result["status_code"] == 200
    assert result["meta"]["perf"]["first_token_ms"] == 1.0
    assert result["answer"] == "첫 청크"


def test_official_question_lists_include_llm_canary_preset() -> None:
    assert smoke.OFFICIAL_QUESTION_LISTS["llm_canary_20"].name == (
        "smoke_chatbot_quality_llm_canary_20.txt"
    )


def test_categorize_planner_mode_treats_fast_path_bundle_as_fast_path() -> None:
    assert smoke._categorize_planner_mode("fast_path_bundle") == "fast_path"


def test_build_console_summary_omits_full_failed_case_list() -> None:
    summary = {
        "failed": 4,
        "failed_cases": [
            {"question": "질문 1"},
            {"question": "질문 2"},
            {"question": "질문 3"},
            {"question": "질문 4"},
        ],
    }

    console_summary = smoke._build_console_summary(summary)

    assert "failed_cases" not in console_summary["summary"]
    assert console_summary["summary"]["failed_case_count"] == 4
    assert console_summary["summary"]["failed_case_preview"] == [
        {"question": "질문 1"},
        {"question": "질문 2"},
        {"question": "질문 3"},
    ]


def test_main_writes_summary_output_before_console_summary_print(
    tmp_path, monkeypatch
) -> None:
    output_path = tmp_path / "report.json"
    summary_output = tmp_path / "summary.json"
    lock_path = tmp_path / "smoke.lock"

    def _fake_check(  # noqa: ANN001, ANN202
        client,
        base_url,
        question,
        **kwargs,
    ):
        return {
            "endpoint": kwargs.get("endpoint", "/ai/chat/completion"),
            "question": question,
            "ok": True,
            "latency_ms": 100.0,
            "error": None,
            "status_code": 200,
            "quality_pass": True,
            "quality": {
                "natural_chat": True,
                "no_table_markup": True,
                "no_briefing_headers": True,
                "no_source_line": True,
                "no_raw_chunk_marker": True,
                "no_low_data_fallback": True,
            },
            "sample_response": {
                "planner_mode": "cache",
                "planner_cache_hit": True,
                "tool_execution_mode": "none",
                "perf": {
                    "planner_mode": "cache",
                    "planner_cache_hit": True,
                    "tool_execution_mode": "none",
                    "first_token_ms": 5.0,
                    "total_ms": 10.0,
                },
            },
        }

    def _fake_completion(client, base_url, question, **kwargs):  # noqa: ANN001, ANN202
        return _fake_check(
            client,
            base_url,
            question,
            endpoint="/ai/chat/completion",
            **kwargs,
        )

    def _fake_stream(client, base_url, question, **kwargs):  # noqa: ANN001, ANN202
        return _fake_check(
            client,
            base_url,
            question,
            endpoint="/ai/chat/stream",
            **kwargs,
        )

    released: list[str] = []

    monkeypatch.setattr(smoke, "_acquire_execution_lock", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        smoke,
        "_release_execution_lock",
        lambda path: released.append(str(path)),
    )
    monkeypatch.setattr(smoke, "_load_questions", lambda *args, **kwargs: ["질문 1"])
    monkeypatch.setattr(smoke, "_check_completion", _fake_completion)
    monkeypatch.setattr(smoke, "_check_stream", _fake_stream)
    monkeypatch.setattr(smoke, "_print_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_chatbot_quality.py",
            "--base-url",
            "http://127.0.0.1:8001",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_output),
            "--lock-file",
            str(lock_path),
        ],
    )

    real_print = builtins.print

    def _raising_print(*args, **kwargs):  # noqa: ANN002, ANN003
        if args and isinstance(args[0], str) and args[0].lstrip().startswith(
            '{\n  "summary"'
        ):
            raise RuntimeError("stop-after-console-summary")
        return real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", _raising_print)

    with pytest.raises(RuntimeError, match="stop-after-console-summary"):
        smoke.main()

    assert output_path.exists()
    assert summary_output.exists()
    summary_payload = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary_payload["summary"]["failed"] == 0
    assert released == [str(lock_path)]
