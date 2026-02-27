from __future__ import annotations

from datetime import datetime

from app.core import runtime_feature_audit as audit


def test_scan_code_usage_counts_expected_patterns(tmp_path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import concurrent.interpreters",
                "value = t\"hello {name}\"",
                "try:",
                "    pass",
                "except (TypeError, ValueError):",
                "    pass",
                "try:",
                "    pass",
                "except ValueError, TypeError:",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )
    usage = audit.scan_code_usage(tmp_path)
    assert usage["python_file_count"] == 1
    assert usage["future_annotations_count"] == 1
    assert usage["concurrent_interpreters_usage"] == 1
    assert usage["tstring_usage"] == 1
    assert usage["except_paren_multi"] == 1
    assert usage["except_no_paren_multi"] == 1


def test_run_feature_audit_has_fixed_top_level_schema() -> None:
    report = audit.run_feature_audit()
    assert sorted(report.keys()) == [
        "decision_flags",
        "generated_at",
        "runtime",
        "syntax_support",
        "usage_scan",
    ]

    datetime.fromisoformat(report["generated_at"])

    runtime = report["runtime"]
    for key in (
        "python_version",
        "python_executable",
        "gil_disabled",
        "tail_call_interp",
        "has_concurrent_interpreters",
    ):
        assert key in runtime

    syntax_support = report["syntax_support"]
    assert "pep758_except_without_parentheses" in syntax_support
    assert "pep750_t_strings" in syntax_support
    assert "deferred_annotations_default" in syntax_support


def test_collect_syntax_support_reports_probe_shape() -> None:
    result = audit.collect_syntax_support()
    pep750 = result["pep750_t_strings"]
    pep758 = result["pep758_except_without_parentheses"]
    deferred = result["deferred_annotations_default"]

    assert isinstance(pep750["supported"], bool)
    assert isinstance(pep758["supported"], bool)
    assert isinstance(deferred["definition_succeeded"], bool)
    assert isinstance(deferred["annotation_access_raises_name_error"], bool)
