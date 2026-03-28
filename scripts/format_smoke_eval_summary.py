#!/usr/bin/env python3
"""Render a compact human-readable smoke eval summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a compact summary for a smoke eval JSON file."
    )
    parser.add_argument("--eval", required=True, dest="eval_path")
    parser.add_argument("--label", required=True)
    parser.add_argument("--prefix", default="[ai-smoke]")
    return parser.parse_args()


def _format_warning_detail(
    prefix: str, label: str, payload: dict[str, Any]
) -> list[str]:
    warnings = payload.get("warnings") or []
    if "stream:stream_first_message_p95_regression" not in warnings:
        return []

    stream = (payload.get("endpoints") or {}).get("stream") or {}
    baseline = stream.get("baseline") or {}
    candidate = stream.get("candidate") or {}
    delta = stream.get("delta") or {}
    candidate_p95 = candidate.get("stream_first_message_p95")
    anchor_p95 = baseline.get("stream_first_message_p95_regression_anchor")
    delta_ratio = delta.get("stream_first_message_p95_ratio")

    if not isinstance(candidate_p95, (int, float)) or not isinstance(
        anchor_p95, (int, float)
    ):
        return []

    line = (
        f"{prefix} {label}: stream_first_message_warning "
        f"candidate={candidate_p95:.2f}ms anchor={anchor_p95:.2f}ms"
    )
    if isinstance(delta_ratio, (int, float)):
        line += f" delta={delta_ratio * 100:+.2f}%"
    return [line]


def render_lines(prefix: str, label: str, payload: dict[str, Any]) -> list[str]:
    status = payload.get("status", "UNKNOWN")
    failures = payload.get("failure_codes") or []
    warnings = payload.get("warnings") or []
    planner = payload.get("planner_checks") or {}
    llm_ratio = planner.get("llm_ratio")
    player_fast_path_ratio = planner.get("player_fast_path_ratio")
    stream = (payload.get("endpoints") or {}).get("stream") or {}
    stream_candidate = stream.get("candidate") or {}
    memory = payload.get("memory") or {}

    line = f"{prefix} {label}: {status}"
    if isinstance(player_fast_path_ratio, (int, float)):
        line += f" player_fast_path_ratio={player_fast_path_ratio:.3f}"
    if isinstance(llm_ratio, (int, float)):
        line += f" llm_ratio={llm_ratio:.3f}"

    stream_p95 = stream_candidate.get("p95")
    if isinstance(stream_p95, (int, float)):
        line += f" stream_p95={stream_p95:.2f}ms"

    stream_ttfe = stream_candidate.get("first_token_p95")
    if isinstance(stream_ttfe, (int, float)):
        line += f" stream_first_token_p95={stream_ttfe:.2f}ms"

    stream_first_message = stream_candidate.get("stream_first_message_p95")
    if isinstance(stream_first_message, (int, float)):
        line += f" stream_first_message_p95={stream_first_message:.2f}ms"

    candidate_memory_mb = memory.get("candidate_memory_mb")
    if isinstance(candidate_memory_mb, (int, float)):
        line += f" peak_memory={candidate_memory_mb:.2f}MB"

    lines = [line]
    if failures:
        lines.append(
            f"{prefix} {label}: failures={', '.join(str(item) for item in failures)}"
        )
    if warnings:
        lines.append(
            f"{prefix} {label}: warnings={', '.join(str(item) for item in warnings)}"
        )
        if not failures:
            lines.append(
                f"{prefix} {label}: warning_policy=advisory_only blocking=false"
            )
    lines.extend(_format_warning_detail(prefix, label, payload))
    return lines


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.eval_path).read_text(encoding="utf-8"))
    for line in render_lines(args.prefix, args.label, payload):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
