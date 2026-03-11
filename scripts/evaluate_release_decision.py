#!/usr/bin/env python3
"""릴리스/롤아웃 결정 초안의 deterministic eval CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("BEGA_SKIP_APP_INIT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.release_decision_agent import (
    ReleaseDecisionDraft,
    ReleaseDecisionRunResult,
    SCENARIO_PRESETS,
    ResponsesReleaseDecisionAgent,
)
from app.agents.release_decision_eval import (
    evaluate_release_decision,
    load_eval_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = PROJECT_ROOT / "bega_AI" / "evals" / "release_decision_cases.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "release_decision_eval"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate release decision drafts against deterministic checks."
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES),
        help="Path to eval cases JSON.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing saved run JSON files keyed by case_id.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Generate fresh drafts via Responses API before scoring.",
    )
    parser.add_argument(
        "--workspace-root",
        default=str(PROJECT_ROOT),
        help="Workspace root path.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Responses API model name for --live mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for live runs and summary reports.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional output path for summary JSON.",
    )
    return parser


def _load_saved_run(path: Path) -> ReleaseDecisionRunResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ReleaseDecisionRunResult.model_validate(payload)


def _load_saved_draft(path: Path) -> ReleaseDecisionDraft:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "draft" in payload:
        return ReleaseDecisionRunResult.model_validate(payload).draft
    return ReleaseDecisionDraft.model_validate(payload)


def main() -> int:
    args = build_parser().parse_args()
    cases = load_eval_cases(Path(args.cases))
    workspace_root = Path(args.workspace_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    for case in cases:
        if args.live:
            preset = SCENARIO_PRESETS[case.scenario]
            agent = ResponsesReleaseDecisionAgent(
                workspace_root=workspace_root,
                allowed_roots=preset.allowed_roots,
                model=args.model,
            )
            run = agent.draft(
                scenario=case.scenario,
                task_prompt=case.task_prompt,
                seed_paths=case.seed_paths or list(preset.seed_paths),
            )
            run_path = output_dir / f"{case.case_id}.json"
            run_path.write_text(
                json.dumps(run.model_dump(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            draft = run.draft
        else:
            if not args.input_dir:
                raise ValueError("--input-dir is required when --live is not used")
            run_path = Path(args.input_dir) / f"{case.case_id}.json"
            draft = _load_saved_draft(run_path)

        evaluation = evaluate_release_decision(draft, case)
        reports.append(
            {
                "case_id": case.case_id,
                "status": evaluation.status,
                "decision_ok": evaluation.decision_ok,
                "missing_keywords": evaluation.missing_keywords,
                "missing_sources": evaluation.missing_sources,
            }
        )

    pass_count = sum(1 for item in reports if item["status"] == "PASS")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(reports),
        "pass_count": pass_count,
        "fail_count": len(reports) - pass_count,
        "cases": reports,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0 if pass_count == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
