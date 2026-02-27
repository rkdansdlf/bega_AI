from __future__ import annotations

import json
import sys

from scripts import build_smoke_latency_baseline as baseline


def _summary_doc(completion_p95: float, stream_p95: float) -> dict:
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
    assert payload["metrics"]["stream"]["latency_ms"]["p95"] == 310.0
