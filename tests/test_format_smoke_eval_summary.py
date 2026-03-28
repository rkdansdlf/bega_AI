from __future__ import annotations

from scripts import format_smoke_eval_summary as formatter


def test_render_lines_includes_stream_first_message_warning_detail() -> None:
    payload = {
        "status": "PASS_NO_REGRESSION",
        "failure_codes": [],
        "warnings": ["stream:stream_first_message_p95_regression"],
        "planner_checks": {"player_fast_path_ratio": 1.0},
        "memory": {"candidate_memory_mb": 512.4},
        "endpoints": {
            "stream": {
                "baseline": {
                    "p95_regression_anchor": 1000.0,
                    "stream_first_message_p95_regression_anchor": 400.0,
                },
                "candidate": {
                    "p95": 850.0,
                    "first_token_p95": 12.0,
                    "stream_first_message_p95": 460.0,
                },
                "delta": {
                    "stream_first_message_p95_ratio": 0.15,
                },
            }
        },
    }

    lines = formatter.render_lines("[ai-smoke]", "llm_canary_20", payload)

    assert lines[0].startswith("[ai-smoke] llm_canary_20: PASS_NO_REGRESSION")
    assert "player_fast_path_ratio=1.000" in lines[0]
    assert "stream_first_message_p95=460.00ms" in lines[0]
    assert "peak_memory=512.40MB" in lines[0]
    assert lines[1] == (
        "[ai-smoke] llm_canary_20: warnings=stream:stream_first_message_p95_regression"
    )
    assert lines[2] == (
        "[ai-smoke] llm_canary_20: warning_policy=advisory_only blocking=false"
    )
    assert lines[3] == (
        "[ai-smoke] llm_canary_20: stream_first_message_warning "
        "candidate=460.00ms anchor=400.00ms delta=+15.00%"
    )


def test_render_lines_skips_warning_detail_when_warning_absent() -> None:
    payload = {
        "status": "PASS_IMPROVED",
        "failure_codes": [],
        "warnings": [],
        "planner_checks": {},
        "memory": {},
        "endpoints": {
            "stream": {
                "baseline": {},
                "candidate": {"p95": 120.0, "first_token_p95": 5.0},
                "delta": {},
            }
        },
    }

    lines = formatter.render_lines(
        "[quality-regression] ai", "latest_eval=foo.json", payload
    )

    assert len(lines) == 1
    assert lines[0] == (
        "[quality-regression] ai latest_eval=foo.json: PASS_IMPROVED "
        "stream_p95=120.00ms stream_first_token_p95=5.00ms"
    )
