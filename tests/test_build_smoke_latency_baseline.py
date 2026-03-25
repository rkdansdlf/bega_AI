from __future__ import annotations

import argparse
import json
import sys

from scripts import build_smoke_latency_baseline as baseline


def _summary_doc(
    completion_p95: float,
    stream_p95: float,
    *,
    stream_first_token_p95: float = 120.0,
) -> dict:
    return {
        "summary": {
            "overall_error_rate": 0.1,
            "overall_timeout_rate": 0.05,
            "completion_metrics": {
                "pass_rate": 0.9,
                "error_rate": 0.1,
                "timeout_rate": 0.05,
                "latency_ms": {
                    "p50": completion_p95 - 20,
                    "p95": completion_p95,
                    "p99": completion_p95 + 30,
                    "avg": completion_p95 - 10,
                },
            },
            "completion_fallback_metrics": {
                "fallback_ratio": 0.0,
            },
            "stream_metrics": {
                "pass_rate": 0.92,
                "error_rate": 0.08,
                "timeout_rate": 0.04,
                "latency_ms": {
                    "p50": stream_p95 - 20,
                    "p95": stream_p95,
                    "p99": stream_p95 + 30,
                    "avg": stream_p95 - 10,
                },
            },
            "stream_fallback_metrics": {
                "fallback_ratio": 0.2,
            },
            "perf_metrics": {
                "completion": {
                    "first_token_ms": {
                        "p50": 80.0,
                        "p95": 90.0,
                        "p99": 95.0,
                        "avg": 85.0,
                    }
                },
                "stream": {
                    "first_token_ms": {
                        "p50": stream_first_token_p95 - 20,
                        "p95": stream_first_token_p95,
                        "p99": stream_first_token_p95 + 20,
                        "avg": stream_first_token_p95 - 10,
                    }
                },
            },
        }
    }


def test_build_smoke_latency_baseline_main(tmp_path, monkeypatch) -> None:
    in1 = tmp_path / "s1.json"
    in2 = tmp_path / "s2.json"
    out = tmp_path / "baseline.json"
    in1.write_text(json.dumps(_summary_doc(200.0, 300.0)), encoding="utf-8")
    in2.write_text(json.dumps(_summary_doc(220.0, 320.0)), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_smoke_latency_baseline.py",
            "--inputs",
            str(in1),
            str(in2),
            "--output",
            str(out),
        ],
    )
    assert baseline.main() == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["run_count"] == 2
    assert payload["metrics"]["completion"]["latency_ms"]["p95"] == 210.0
    assert payload["metrics"]["completion"]["latency_ms"]["p95_min"] == 200.0
    assert payload["metrics"]["completion"]["latency_ms"]["p95_max"] == 220.0
    assert payload["metrics"]["stream"]["latency_ms"]["p95"] == 310.0
    assert payload["metrics"]["stream"]["latency_ms"]["p95_min"] == 300.0
    assert payload["metrics"]["stream"]["latency_ms"]["p95_max"] == 320.0
    assert payload["metrics"]["stream"]["first_token_ms"]["p95"] == 120.0
    assert payload["metrics"]["stream"]["fallback_ratio"] == 0.2


def test_llm_canary_preset_paths_are_registered() -> None:
    args = argparse.Namespace(inputs=None, preset="llm_canary_20", output=None)

    inputs = baseline._resolve_inputs(args)
    output = baseline._resolve_output(args)

    assert inputs == [
        str(
            baseline.REPORTS_DIR
            / "smoke_chatbot_quality_llm_canary_20_baseline_summary.json"
        ),
        str(
            baseline.REPORTS_DIR
            / "smoke_chatbot_quality_llm_canary_20_v1_summary.json"
        ),
        str(
            baseline.REPORTS_DIR
            / "smoke_chatbot_quality_llm_canary_20_v2_summary.json"
        ),
    ]
    assert output == (
        baseline.REPORTS_DIR / "smoke_latency_baseline_llm_canary_20.json"
    )
