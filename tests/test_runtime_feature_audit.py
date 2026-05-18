"""runtime_feature_audit.py 단위 테스트.

순수 함수(collect_runtime_info, collect_syntax_support, collect_decision_flags,
scan_code_usage, run_feature_audit)의 반환 구조와 scan_code_usage의 regex 동작을
tmp_path fixture를 활용해 외부 의존성 없이 검증한다.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.runtime_feature_audit import (
    collect_decision_flags,
    collect_runtime_info,
    collect_syntax_support,
    default_project_root,
    run_feature_audit,
    scan_code_usage,
)


# ── TestDefaultProjectRoot ────────────────────────────────────────────────────

class TestDefaultProjectRoot:
    def test_returns_path_instance(self):
        assert isinstance(default_project_root(), Path)

    def test_directory_exists(self):
        assert default_project_root().is_dir()


# ── TestCollectRuntimeInfo ────────────────────────────────────────────────────

class TestCollectRuntimeInfo:
    def test_returns_dict(self):
        assert isinstance(collect_runtime_info(), dict)

    def test_python_version_key_present(self):
        assert "python_version" in collect_runtime_info()

    def test_platform_key_present(self):
        assert "platform" in collect_runtime_info()

    def test_gc_threshold_is_list(self):
        result = collect_runtime_info()
        assert isinstance(result["gc_threshold"], list)

    def test_has_concurrent_interpreters_is_bool(self):
        result = collect_runtime_info()
        assert isinstance(result["has_concurrent_interpreters"], bool)


# ── TestCollectSyntaxSupport ──────────────────────────────────────────────────

class TestCollectSyntaxSupport:
    def test_returns_dict(self):
        assert isinstance(collect_syntax_support(), dict)

    def test_pep758_key_present(self):
        assert "pep758_except_without_parentheses" in collect_syntax_support()

    def test_pep750_key_present(self):
        assert "pep750_t_strings" in collect_syntax_support()

    def test_each_probe_has_supported_field(self):
        result = collect_syntax_support()
        for key in ("pep758_except_without_parentheses", "pep750_t_strings"):
            assert "supported" in result[key]

    def test_deferred_annotations_key_present(self):
        assert "deferred_annotations_default" in collect_syntax_support()


# ── TestCollectDecisionFlags ──────────────────────────────────────────────────

class TestCollectDecisionFlags:
    def test_returns_dict(self):
        assert isinstance(collect_decision_flags(), dict)

    def test_free_threaded_is_bool(self):
        assert isinstance(collect_decision_flags()["free_threaded_production_now"], bool)

    def test_rollout_strategy_is_str(self):
        assert isinstance(collect_decision_flags()["rollout_strategy"], str)

    def test_has_expected_keys(self):
        flags = collect_decision_flags()
        assert "python_support_policy" in flags
        assert "t_strings_adopt_now" in flags


# ── TestRunFeatureAudit ───────────────────────────────────────────────────────

class TestRunFeatureAudit:
    def test_returns_dict(self):
        assert isinstance(run_feature_audit(), dict)

    def test_generated_at_key_present(self):
        result = run_feature_audit()
        assert "generated_at" in result

    def test_generated_at_is_iso_format(self):
        generated_at = run_feature_audit()["generated_at"]
        assert "T" in generated_at
        assert "+" in generated_at or "Z" in generated_at

    def test_top_level_schema_keys(self):
        result = run_feature_audit()
        for key in ("runtime", "syntax_support", "decision_flags", "usage_scan"):
            assert key in result

    def test_accepts_none_project_root(self):
        result = run_feature_audit(project_root=None)
        assert isinstance(result, dict)


# ── TestScanCodeUsage ─────────────────────────────────────────────────────────

class TestScanCodeUsage:
    def test_empty_project_returns_zeros(self, tmp_path):
        result = scan_code_usage(tmp_path)
        assert result["python_file_count"] == 0
        assert result["future_annotations_count"] == 0

    def test_future_annotations_counted(self, tmp_path):
        (tmp_path / "mod.py").write_text("from __future__ import annotations\nx = 1\n")
        result = scan_code_usage(tmp_path)
        assert result["future_annotations_count"] == 1

    def test_two_files_with_future_annotations(self, tmp_path):
        for name in ("a.py", "b.py"):
            (tmp_path / name).write_text("from __future__ import annotations\n")
        result = scan_code_usage(tmp_path)
        assert result["future_annotations_count"] == 2

    def test_git_dir_excluded(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "hooks.py").write_text("from __future__ import annotations\n")
        result = scan_code_usage(tmp_path)
        assert result["future_annotations_count"] == 0

    def test_venv_dir_excluded(self, tmp_path):
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "site.py").write_text("from __future__ import annotations\n")
        result = scan_code_usage(tmp_path)
        assert result["future_annotations_count"] == 0

    def test_tests_dir_excluded(self, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_x.py").write_text("from __future__ import annotations\n")
        result = scan_code_usage(tmp_path)
        assert result["future_annotations_count"] == 0

    def test_python_file_count_accurate(self, tmp_path):
        for name in ("x.py", "y.py", "z.py"):
            (tmp_path / name).write_text("pass\n")
        result = scan_code_usage(tmp_path)
        assert result["python_file_count"] == 3

    def test_result_has_all_keys(self, tmp_path):
        result = scan_code_usage(tmp_path)
        for key in (
            "python_file_count",
            "future_annotations_count",
            "concurrent_interpreters_usage",
            "tstring_usage",
            "except_paren_multi",
            "except_no_paren_multi",
        ):
            assert key in result
