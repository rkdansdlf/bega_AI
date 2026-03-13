#!/usr/bin/env python3
"""Responses API 기반 릴리스/롤아웃 결정 초안 생성 CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("BEGA_SKIP_APP_INIT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.release_decision_agent import (
    SCENARIO_PRESETS,
    ResponsesReleaseDecisionAgent,
    render_release_decision_markdown,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draft a release/rollout decision from repo documents using Responses API."
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_PRESETS.keys()),
        required=True,
        help="Built-in scenario preset.",
    )
    parser.add_argument(
        "--task-prompt",
        default=None,
        help="Optional override for the scenario task prompt.",
    )
    parser.add_argument(
        "--seed-path",
        action="append",
        default=None,
        help="Additional repo-relative seed document path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--allowed-root",
        action="append",
        default=None,
        help="Additional repo-relative allowed root. Can be passed multiple times.",
    )
    parser.add_argument(
        "--workspace-root",
        default=str(WORKSPACE_ROOT),
        help="Workspace root path.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Responses API model name. Defaults to OPENAI_RESPONSES_MODEL or gpt-4.1-mini.",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=6,
        help="Maximum tool loop rounds.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2200,
        help="Maximum final output tokens.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional markdown output path.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    preset = SCENARIO_PRESETS[args.scenario]
    workspace_root = Path(args.workspace_root).resolve()
    seed_paths = list(preset.seed_paths)
    if args.seed_path:
        seed_paths.extend(args.seed_path)
    allowed_roots = list(preset.allowed_roots)
    if args.allowed_root:
        allowed_roots.extend(args.allowed_root)

    agent = ResponsesReleaseDecisionAgent(
        workspace_root=workspace_root,
        allowed_roots=allowed_roots,
        model=args.model,
        max_tool_rounds=args.max_tool_rounds,
        max_output_tokens=args.max_output_tokens,
    )
    result = agent.draft(
        scenario=preset.name,
        task_prompt=args.task_prompt or preset.task_prompt,
        seed_paths=seed_paths,
    )
    payload = result.model_dump()
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.markdown_output:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_release_decision_markdown(result.draft),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
