from __future__ import annotations

import builtins
import json
import sys

import pytest

from scripts import smoke_chatbot as smoke


def test_build_console_summary_wraps_summary_payload() -> None:
    summary = {"total": 5, "passed": 5, "failed": 0}

    console_summary = smoke._build_console_summary(summary)

    assert console_summary == {"summary": summary}
    assert console_summary["summary"] is not summary


def test_main_writes_summary_output_before_console_print(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "report.json"
    summary_output = tmp_path / "summary.json"

    def _ok(endpoint: str) -> dict[str, object]:
        return {
            "endpoint": endpoint,
            "status_code": 200,
            "ok": True,
            "latency_ms": 1.0,
            "error": None,
            "sample_response": {"status": "ok"},
        }

    monkeypatch.setattr(smoke, "check_health", lambda *args, **kwargs: _ok("/health"))
    monkeypatch.setattr(
        smoke,
        "check_completion",
        lambda *args, **kwargs: _ok("/ai/chat/completion"),
    )
    monkeypatch.setattr(
        smoke,
        "check_chat_stream",
        lambda *args, **kwargs: _ok("/ai/chat/stream"),
    )
    monkeypatch.setattr(
        smoke,
        "check_coach_stream",
        lambda *args, **kwargs: _ok("/coach/stream"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_chatbot.py",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_output),
            "--chat-batch-size",
            "0",
        ],
    )

    real_print = builtins.print

    def _raising_print(*args, **kwargs):  # noqa: ANN002, ANN003
        if args and isinstance(args[0], str) and args[0].startswith('{"summary"'):
            raise RuntimeError("stop-after-console-summary")
        return real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", _raising_print)

    with pytest.raises(RuntimeError, match="stop-after-console-summary"):
        smoke.main()

    assert output_path.exists()
    assert summary_output.exists()
    assert json.loads(summary_output.read_text(encoding="utf-8"))["failed"] == 0
