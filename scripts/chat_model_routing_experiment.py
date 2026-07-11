#!/usr/bin/env python3
"""Replay chat samples and summarize model-routing latency/cost signals.

The service reads planner/answer model choices from environment-level settings.
This harness records the labels used for a run so operators can compare reports
from separate deployments or restarts without adding request-level model
override fields to the public API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx


def _load_questions(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json", ".jsonl"}:
        questions: List[str] = []
        records: Iterable[Any]
        if path.suffix.lower() == ".json":
            parsed = json.loads(raw)
            records = (
                parsed.get("samples", parsed) if isinstance(parsed, dict) else parsed
            )
        else:
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        for item in records:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict) and item.get("question"):
                questions.append(str(item["question"]))
        return questions
    return [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return round(ordered[index], 2)


async def _call_completion(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    internal_api_key: str,
    question: str,
    cache_bypass: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    response = await client.post(
        f"{base_url.rstrip('/')}/ai/chat/completion",
        json={"question": question, "cache_bypass": cache_bypass},
        headers={"X-Internal-Api-Key": internal_api_key},
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    payload: Dict[str, Any]
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        payload = {"raw": response.text[:500]}
    return {
        "question": question,
        "status_code": response.status_code,
        "ok": response.is_success,
        "latency_ms": latency_ms,
        "cache_bypass": cache_bypass,
        "answer_chars": len(str(payload.get("answer") or "")),
        "meta": payload.get("meta") if isinstance(payload.get("meta"), dict) else {},
        "error": None if response.is_success else payload,
    }


async def async_main(args: argparse.Namespace) -> int:
    questions = _load_questions(Path(args.samples))[: args.limit]
    internal_api_key = args.internal_api_key or os.getenv(args.internal_api_key_env, "")
    if not internal_api_key:
        raise ValueError("internal API key is required")
    if not questions:
        raise ValueError("no questions loaded")

    timeout = httpx.Timeout(args.timeout)
    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for question in questions:
            results.append(
                await _call_completion(
                    client=client,
                    base_url=args.base_url,
                    internal_api_key=internal_api_key,
                    question=question,
                    cache_bypass=args.cache_bypass,
                )
            )

    latencies = [item["latency_ms"] for item in results if item["ok"]]
    failures = [item for item in results if not item["ok"]]
    report = {
        "summary": {
            "sample_count": len(results),
            "success_count": len(results) - len(failures),
            "failure_count": len(failures),
            "latency_avg_ms": (
                round(statistics.mean(latencies), 2) if latencies else 0.0
            ),
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "planner_model_label": args.planner_model_label,
            "answer_model_label": args.answer_model_label,
            "cache_bypass": args.cache_bypass,
        },
        "details": results,
    }

    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0 if not failures else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chat model-routing experiment samples."
    )
    parser.add_argument("--samples", default=os.getenv("AI_MODEL_ROUTING_SAMPLES", ""))
    parser.add_argument(
        "--base-url",
        default=os.getenv("AI_MODEL_ROUTING_BASE_URL", "http://localhost:8001"),
    )
    parser.add_argument("--output", default=os.getenv("AI_MODEL_ROUTING_OUTPUT", ""))
    parser.add_argument("--internal-api-key", default=None)
    parser.add_argument("--internal-api-key-env", default="AI_INTERNAL_TOKEN")
    parser.add_argument(
        "--planner-model-label", default=os.getenv("CHAT_PLANNER_MODEL_NAME", "default")
    )
    parser.add_argument(
        "--answer-model-label", default=os.getenv("CHAT_ANSWER_MODEL_NAME", "default")
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--cache-bypass", action="store_true")
    args = parser.parse_args()
    if not args.samples:
        parser.error("--samples or AI_MODEL_ROUTING_SAMPLES is required")
    return args


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
