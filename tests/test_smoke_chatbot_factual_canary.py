from __future__ import annotations

import builtins
import json
import sys

import pytest

from scripts import smoke_chatbot_factual_canary as factual


def test_build_console_summary_wraps_summary_payload() -> None:
    summary = {"total": 2, "passed": 2, "failed": 0}

    console_summary = factual._build_console_summary(summary)

    assert console_summary == {"summary": summary}
    assert console_summary["summary"] is not summary


def test_main_writes_summary_output_before_console_print(tmp_path, monkeypatch) -> None:
    case_path = tmp_path / "cases.json"
    output_path = tmp_path / "report.json"
    summary_output = tmp_path / "summary.json"
    case_path.write_text(
        json.dumps(
            [
                {
                    "id": "case-1",
                    "question": "질문 1",
                    "required_all": [],
                    "required_any": [],
                    "forbidden": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        factual,
        "_check_completion",
        lambda *args, **kwargs: {
            "endpoint": "/ai/chat/completion",
            "status_code": 200,
            "latency_ms": 10.0,
            "answer": "정상 응답",
            "error": None,
        },
    )
    monkeypatch.setattr(
        factual,
        "_check_stream",
        lambda *args, **kwargs: {
            "endpoint": "/ai/chat/stream",
            "status_code": 200,
            "latency_ms": 5.0,
            "answer": "정상 응답",
            "error": None,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_chatbot_factual_canary.py",
            "--case-file",
            str(case_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_output),
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
        factual.main()

    assert output_path.exists()
    assert summary_output.exists()
    assert json.loads(summary_output.read_text(encoding="utf-8"))["failed"] == 0
