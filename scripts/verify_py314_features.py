#!/usr/bin/env python3
"""Verify Python 3.14 feature support and project usage status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from app.core.runtime_feature_audit import run_feature_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Python 3.14 feature audit report for AI service."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for JSON report.",
    )
    parser.add_argument(
        "--format",
        choices=("json",),
        default="json",
        help="Report output format. Currently only json is supported.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Optional project root override (default: bega_AI).",
    )
    return parser.parse_args()


def _write_report(path_str: str, report: Dict[str, Any]) -> Path:
    output_path = Path(path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else None
    report = run_feature_audit(project_root=project_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output:
        written = _write_report(args.output, report)
        print(f"report saved: {written}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
