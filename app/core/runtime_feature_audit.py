"""Python 3.14 feature audit helpers for AI service runtime and code usage."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import importlib.util
from pathlib import Path
import platform
import re
import sys
import sysconfig
from typing import Any, Dict, Iterable

_FUTURE_ANNOTATIONS_PATTERN = re.compile(
    r"from\s+__future__\s+import\s+annotations", re.MULTILINE
)
_CONCURRENT_INTERPRETERS_PATTERN = re.compile(r"concurrent\.interpreters")
_TSTRING_PATTERN = re.compile(r"(?<![A-Za-z0-9_])t(?:\"|')")
_EXCEPT_PAREN_MULTI_PATTERN = re.compile(r"except\s*\(")
_EXCEPT_NO_PAREN_MULTI_PATTERN = re.compile(
    r"except\s+[A-Za-z_][A-Za-z0-9_\.]*\s*,\s*[A-Za-z_][A-Za-z0-9_\.]*\s*:"
)

_PEP758_PROBE = """try:
    pass
except ValueError, TypeError:
    pass
"""

_PEP750_PROBE = """name = "bega"
value = t"Hello {name}"
"""

_DEFERRED_ANNOTATION_PROBE = """def _probe(x: UnknownType):
    return x
"""

_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
_SKIP_SCAN_FILE_NAMES = {
    "runtime_feature_audit.py",
    "verify_py314_features.py",
}


def default_project_root() -> Path:
    """Return AI service root directory (bega_AI)."""
    return Path(__file__).resolve().parents[2]


def _iter_python_files(project_root: Path) -> Iterable[Path]:
    for path in project_root.rglob("*.py"):
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        rel_path = path.relative_to(project_root)
        if "tests" in rel_path.parts:
            continue
        if path.name in _SKIP_SCAN_FILE_NAMES:
            continue
        yield path


def _compile_probe(code: str) -> Dict[str, Any]:
    try:
        compile(code, "<feature_probe>", "exec")
        return {"supported": True, "error": None}
    except SyntaxError as exc:
        return {"supported": False, "error": f"{type(exc).__name__}: {exc.msg}"}


def _probe_deferred_annotations_default() -> Dict[str, Any]:
    namespace: Dict[str, Any] = {}
    try:
        exec(_DEFERRED_ANNOTATION_PROBE, namespace, namespace)
    except Exception as exc:  # noqa: BLE001
        return {
            "definition_succeeded": False,
            "annotation_access_raises_name_error": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    annotation_error: str | None = None
    try:
        _ = namespace["_probe"].__annotations__
    except Exception as exc:  # noqa: BLE001
        annotation_error = f"{type(exc).__name__}: {exc}"

    return {
        "definition_succeeded": True,
        "annotation_access_raises_name_error": bool(
            annotation_error and annotation_error.startswith("NameError:")
        ),
        "error": annotation_error,
    }


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def scan_code_usage(project_root: Path | None = None) -> Dict[str, Any]:
    """Scan python files to count specific syntax and feature usages."""
    root = project_root or default_project_root()
    python_files = list(_iter_python_files(root))

    future_annotations_count = 0
    concurrent_interpreters_usage = 0
    tstring_usage = 0
    except_paren_multi = 0
    except_no_paren_multi = 0

    for file_path in python_files:
        text = _safe_read_text(file_path)
        future_annotations_count += len(_FUTURE_ANNOTATIONS_PATTERN.findall(text))
        concurrent_interpreters_usage += len(
            _CONCURRENT_INTERPRETERS_PATTERN.findall(text)
        )
        tstring_usage += len(_TSTRING_PATTERN.findall(text))
        except_paren_multi += len(_EXCEPT_PAREN_MULTI_PATTERN.findall(text))
        except_no_paren_multi += len(_EXCEPT_NO_PAREN_MULTI_PATTERN.findall(text))

    return {
        "python_file_count": len(python_files),
        "future_annotations_count": future_annotations_count,
        "concurrent_interpreters_usage": concurrent_interpreters_usage,
        "tstring_usage": tstring_usage,
        "except_paren_multi": except_paren_multi,
        "except_no_paren_multi": except_no_paren_multi,
    }


def collect_runtime_info() -> Dict[str, Any]:
    """Collect runtime-level flags from current interpreter."""
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cache_tag": sys.implementation.cache_tag,
        "platform": platform.platform(),
        "gil_disabled": sysconfig.get_config_var("Py_GIL_DISABLED"),
        "tail_call_interp": sysconfig.get_config_var("Py_TAIL_CALL_INTERP"),
        "has_concurrent_interpreters": importlib.util.find_spec(
            "concurrent.interpreters"
        )
        is not None,
        "has_annotationlib": importlib.util.find_spec("annotationlib") is not None,
        "gc_threshold": list(gc.get_threshold()),
        "gc_generation_count": len(gc.get_stats()),
    }


def collect_syntax_support() -> Dict[str, Any]:
    """Run compile probes for selected Python 3.14 syntax features."""
    pep758 = _compile_probe(_PEP758_PROBE)
    pep750 = _compile_probe(_PEP750_PROBE)
    deferred = _probe_deferred_annotations_default()
    return {
        "pep758_except_without_parentheses": pep758,
        "pep750_t_strings": pep750,
        "deferred_annotations_default": deferred,
    }


def collect_decision_flags() -> Dict[str, Any]:
    """Return project policy flags for staged 3.14 adoption."""
    return {
        "python_support_policy": "3.14-only",
        "rollout_strategy": "gradual",
        "runtime_baseline_mode": "docker-and-local",
        "primary_metric": "api_latency",
        "free_threaded_production_now": False,
        "subinterpreters_production_now": False,
        "t_strings_adopt_now": False,
        "pep758_style_adopt_now": False,
    }


def run_feature_audit(project_root: Path | None = None) -> Dict[str, Any]:
    """Build full audit report with fixed top-level schema."""
    root = project_root or default_project_root()
    return {
        "runtime": collect_runtime_info(),
        "syntax_support": collect_syntax_support(),
        "usage_scan": scan_code_usage(root),
        "decision_flags": collect_decision_flags(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
