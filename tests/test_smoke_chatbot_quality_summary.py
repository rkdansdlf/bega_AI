from __future__ import annotations

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
