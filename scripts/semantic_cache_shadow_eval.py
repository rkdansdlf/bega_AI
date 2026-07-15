#!/usr/bin/env python3
"""Build a semantic cache shadow evaluation report.

The script compares semantic-cache candidate answers with fresh answers. Samples
can be operator-provided JSON/JSONL records, or records can omit ``fresh_answer``
and let the script replay the question against the internal chat completion API
with ``cache_bypass=true``.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx


MANUAL_DATA_CONTRACT = "MANUAL_BASEBALL_DATA_REQUIRED"
NUMERIC_FACT_PATTERN = re.compile(r"(?<![0-9A-Za-z가-힣])[-+]?\d+(?:[.,]\d+)*(?:%)?")
DEFAULT_RELEASE_MIN_SAMPLES = 100
DEFAULT_RELEASE_MIN_OBSERVATION_DAYS = 7.0
DEFAULT_RELEASE_MAX_FALSE_POSITIVE_RATE = 0.005


def _tokenize(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[0-9A-Za-z가-힣]+", str(text or ""))
        if token.strip()
    }


def _normalize_numeric_fact(raw: str) -> str:
    value = str(raw or "").replace(",", "")
    is_percent = value.endswith("%")
    if is_percent:
        value = value[:-1]
    try:
        decimal_value = Decimal(value)
    except InvalidOperation:
        return value
    if is_percent:
        decimal_value /= Decimal(100)
    normalized = format(decimal_value.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def _numeric_facts(text: str) -> set[str]:
    return {
        _normalize_numeric_fact(match.group(0))
        for match in NUMERIC_FACT_PATTERN.finditer(str(text or ""))
    }


def _numeric_fact_bindings(text: str) -> set[tuple[tuple[str, ...], str]]:
    value = str(text or "")
    bindings: set[tuple[tuple[str, ...], str]] = set()
    for match in NUMERIC_FACT_PATTERN.finditer(value):
        prefix = value[max(0, match.start() - 40) : match.start()]
        prefix = re.split(r"[,;\n]", prefix)[-1]
        context_tokens = re.findall(r"[A-Za-z가-힣_]+", prefix.casefold())[-3:]
        bindings.add(
            (
                tuple(context_tokens),
                _normalize_numeric_fact(match.group(0)),
            )
        )
    return bindings


def _parse_observed_at(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def next_rollout_percent(current_percent: int, *, gate_passed: bool) -> int:
    if not gate_passed:
        return 0
    current = max(0, min(100, int(current_percent)))
    if current < 5:
        return 5
    if current < 25:
        return 25
    return 100


def compare_answers(cached_answer: str, fresh_answer: str) -> Dict[str, Any]:
    cached_tokens = _tokenize(cached_answer)
    fresh_tokens = _tokenize(fresh_answer)
    union = cached_tokens | fresh_tokens
    intersection = cached_tokens & fresh_tokens
    jaccard = round(len(intersection) / len(union), 4) if union else 0.0
    cached_len = len(str(cached_answer or ""))
    fresh_len = len(str(fresh_answer or ""))
    length_ratio = round(cached_len / max(fresh_len, 1), 4)
    manual_contract_match = (
        MANUAL_DATA_CONTRACT in str(cached_answer or "")
    ) == (MANUAL_DATA_CONTRACT in str(fresh_answer or ""))
    return {
        "token_jaccard": jaccard,
        "length_ratio": length_ratio,
        "cached_chars": cached_len,
        "fresh_chars": fresh_len,
        "numeric_facts_match": _numeric_facts(cached_answer)
        == _numeric_facts(fresh_answer),
        "numeric_bindings_match": _numeric_fact_bindings(cached_answer)
        == _numeric_fact_bindings(fresh_answer),
        "manual_contract_match": manual_contract_match,
    }


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return [item for item in payload["samples"] if isinstance(item, dict)]
    raise ValueError("JSON samples must be a list or an object with samples[]")


def load_samples(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return _load_json_records(path)

    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


async def fetch_fresh_answer(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    internal_api_key: str,
    question: str,
    filters: Optional[Dict[str, Any]],
) -> str:
    response = await client.post(
        f"{base_url.rstrip('/')}/ai/chat/completion",
        json={
            "question": question,
            "filters": filters,
            "cache_bypass": True,
        },
        headers={"X-Internal-Api-Key": internal_api_key},
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("answer") or "")


async def enrich_fresh_answers(
    samples: List[Dict[str, Any]],
    *,
    base_url: Optional[str],
    internal_api_key: Optional[str],
    timeout_seconds: float,
) -> List[Dict[str, Any]]:
    if not base_url:
        return samples
    if not internal_api_key:
        raise ValueError("internal_api_key is required when base_url is provided")

    enriched: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for sample in samples:
            item = dict(sample)
            if not item.get("fresh_answer"):
                item["fresh_answer"] = await fetch_fresh_answer(
                    client=client,
                    base_url=base_url,
                    internal_api_key=internal_api_key,
                    question=str(item.get("question") or ""),
                    filters=item.get("filters") if isinstance(item.get("filters"), dict) else None,
                )
            enriched.append(item)
    return enriched


def build_report(
    samples: Iterable[Dict[str, Any]],
    *,
    min_token_jaccard: float,
    release_min_samples: int = DEFAULT_RELEASE_MIN_SAMPLES,
    release_min_observation_days: float = DEFAULT_RELEASE_MIN_OBSERVATION_DAYS,
    release_max_false_positive_rate: float = DEFAULT_RELEASE_MAX_FALSE_POSITIVE_RATE,
    current_rollout_percent: int = 0,
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    compared_count = 0
    failed_count = 0
    skipped_count = 0
    observed_at_values: List[datetime] = []

    for index, sample in enumerate(samples, start=1):
        observed_at = _parse_observed_at(sample.get("observed_at"))
        if observed_at is not None:
            observed_at_values.append(observed_at)
        cached_answer = str(
            sample.get("cached_answer")
            or sample.get("semantic_cached_answer")
            or sample.get("response_text")
            or ""
        )
        fresh_answer = str(sample.get("fresh_answer") or "")
        status = "skipped"
        comparison: Dict[str, Any] = {}
        if cached_answer and fresh_answer:
            compared_count += 1
            comparison = compare_answers(cached_answer, fresh_answer)
            failure_reasons: List[str] = []
            if comparison["token_jaccard"] < min_token_jaccard:
                failure_reasons.append("low_token_overlap")
            if not comparison["numeric_facts_match"]:
                failure_reasons.append("numeric_fact_mismatch")
            if not comparison["numeric_bindings_match"]:
                failure_reasons.append("numeric_binding_mismatch")
            if not comparison["manual_contract_match"]:
                failure_reasons.append("manual_contract_mismatch")
            comparison["failure_reasons"] = failure_reasons
            status = "failed" if failure_reasons else "passed"
            if status == "failed":
                failed_count += 1
        else:
            skipped_count += 1
            comparison["failure_reasons"] = ["missing_comparison_answer"]

        details.append(
            {
                "index": index,
                "cache_key": sample.get("cache_key"),
                "question": sample.get("question"),
                "observed_at": sample.get("observed_at"),
                "status": status,
                **comparison,
            }
        )

    false_positive_rate = (
        round(failed_count / compared_count, 4) if compared_count else 0.0
    )
    coverage_complete = bool(details) and skipped_count == 0
    gate_passed = coverage_complete and failed_count == 0
    observation_window_days = 0.0
    if len(observed_at_values) >= 2:
        observation_window_days = round(
            (max(observed_at_values) - min(observed_at_values)).total_seconds()
            / 86400.0,
            4,
        )
    rollout_gate_checks = {
        "quality_gate_passed": gate_passed,
        "minimum_samples_met": compared_count >= max(1, int(release_min_samples)),
        "observation_window_met": observation_window_days
        >= max(0.0, float(release_min_observation_days)),
        "false_positive_rate_within_budget": false_positive_rate
        <= max(0.0, float(release_max_false_positive_rate)),
    }
    rollout_gate_passed = all(rollout_gate_checks.values())
    return {
        "summary": {
            "sample_count": len(details),
            "compared_count": compared_count,
            "passed_count": compared_count - failed_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "coverage_complete": coverage_complete,
            "gate_passed": gate_passed,
            "potential_false_positive_rate": false_positive_rate,
            "min_token_jaccard": min_token_jaccard,
            "observation_window_days": observation_window_days,
            "rollout_gate_passed": rollout_gate_passed,
            "rollout_gate_checks": rollout_gate_checks,
            "release_min_samples": release_min_samples,
            "release_min_observation_days": release_min_observation_days,
            "release_max_false_positive_rate": release_max_false_positive_rate,
            "current_rollout_percent": current_rollout_percent,
            "recommended_rollout_percent": next_rollout_percent(
                current_rollout_percent,
                gate_passed=rollout_gate_passed,
            ),
        },
        "details": details,
    }


async def async_main(args: argparse.Namespace) -> int:
    samples = load_samples(Path(args.samples))
    internal_api_key = args.internal_api_key or os.getenv(args.internal_api_key_env, "")
    enriched = await enrich_fresh_answers(
        samples,
        base_url=args.base_url,
        internal_api_key=internal_api_key or None,
        timeout_seconds=args.timeout,
    )
    report = build_report(
        enriched,
        min_token_jaccard=args.min_token_jaccard,
        release_min_samples=args.release_min_samples,
        release_min_observation_days=args.release_min_observation_days,
        release_max_false_positive_rate=args.release_max_false_positive_rate,
        current_rollout_percent=args.current_rollout_percent,
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    else:
        print(output)
    gate_name = "rollout_gate_passed" if args.release_gate else "gate_passed"
    return 0 if report["summary"][gate_name] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare semantic cache shadow candidates against fresh answers."
    )
    parser.add_argument("--samples", required=True, help="Input JSON or JSONL samples.")
    parser.add_argument("--output", help="Output report JSON path.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Internal AI service base URL. If omitted, only existing fresh_answer fields are evaluated.",
    )
    parser.add_argument("--internal-api-key", default=None)
    parser.add_argument("--internal-api-key-env", default="AI_INTERNAL_TOKEN")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--min-token-jaccard", type=float, default=0.72)
    parser.add_argument(
        "--release-gate",
        action="store_true",
        help="Require the rollout observation window, sample count, and false-positive budget.",
    )
    parser.add_argument(
        "--release-min-samples", type=int, default=DEFAULT_RELEASE_MIN_SAMPLES
    )
    parser.add_argument(
        "--release-min-observation-days",
        type=float,
        default=DEFAULT_RELEASE_MIN_OBSERVATION_DAYS,
    )
    parser.add_argument(
        "--release-max-false-positive-rate",
        type=float,
        default=DEFAULT_RELEASE_MAX_FALSE_POSITIVE_RATE,
    )
    parser.add_argument("--current-rollout-percent", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
