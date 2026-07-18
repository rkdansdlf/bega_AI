#!/usr/bin/env python3
"""Evaluate semantic-cache serving canaries from internal Prometheus metrics."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Optional
import urllib.parse
import urllib.request


DEFAULT_MIN_STAGE_HOURS = 24.0
DEFAULT_MIN_REQUESTS = 100
DEFAULT_MIN_LATENCY_IMPROVEMENT = 0.15
DEFAULT_MAX_5XX_RATIO = 0.03
DEFAULT_MAX_CACHE_ERROR_RATIO = 0.05
DEFAULT_MIN_SEMANTIC_HITS = 1


@dataclass(frozen=True)
class CanaryMetrics:
    baseline_p95_seconds: Optional[float]
    canary_p95_seconds: Optional[float]
    request_count: float
    error_5xx_count: float
    semantic_lookup_count: float
    semantic_lookup_error_count: float
    semantic_hit_count: float
    availability_min: Optional[float]
    active_safety_alerts: float


class PrometheusClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(0.1, float(timeout_seconds))

    def query(self, expression: str) -> Optional[float]:
        query = urllib.parse.urlencode({"query": expression})
        request = urllib.request.Request(
            f"{self.base_url}/api/v1/query?{query}",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.load(response)
        if payload.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {expression}")
        results = (payload.get("data") or {}).get("result") or []
        if not results:
            return None
        value = float(results[0]["value"][1])
        return value if math.isfinite(value) else None


def build_promql_queries(*, window_hours: int) -> dict[str, str]:
    hours = max(1, int(window_hours))
    window = f"{hours}h"
    route_filter = 'route=~"/ai/chat/(completion|stream)"'
    successful_filter = f'{route_filter},status_group="2xx"'
    latency_buckets = "ai_http_request_duration_seconds_bucket"
    request_total = "ai_http_requests_total"
    safety_alerts = (
        "AIHTTP5xxErrorRatioHigh|AIHTTPChatLatencyP95High|"
        "AISemanticCacheLookupErrorRateHigh|AISemanticCacheStoreErrorRateHigh"
    )
    return {
        "baseline_p95_seconds": (
            "histogram_quantile(0.95, sum(increase("
            f'{latency_buckets}{{job="kbo-ai-chatbot",{successful_filter}}}'
            f"[{window}] offset {window})) by (le))"
        ),
        "canary_p95_seconds": (
            "histogram_quantile(0.95, sum(increase("
            f'{latency_buckets}{{job="kbo-ai-chatbot",{successful_filter}}}'
            f"[{window}])) by (le))"
        ),
        "request_count": (
            f'sum(increase({request_total}{{job="kbo-ai-chatbot",{route_filter}}}'
            f"[{window}])) or vector(0)"
        ),
        "error_5xx_count": (
            f'sum(increase({request_total}{{job="kbo-ai-chatbot",{route_filter},'
            f'status_group="5xx"}}[{window}])) or vector(0)'
        ),
        "semantic_lookup_count": (
            'sum(increase(ai_semantic_response_cache_total{job="kbo-ai-chatbot",'
            f'operation="lookup"}}[{window}])) or vector(0)'
        ),
        "semantic_lookup_error_count": (
            'sum(increase(ai_semantic_response_cache_total{job="kbo-ai-chatbot",'
            f'operation="lookup",result="error"}}[{window}])) or vector(0)'
        ),
        "semantic_hit_count": (
            'sum(increase(ai_semantic_response_cache_total{job="kbo-ai-chatbot",'
            f'operation="lookup",result="hit"}}[{window}])) or vector(0)'
        ),
        "availability_min": (
            f'min_over_time(up{{job="kbo-ai-chatbot"}}[{window}])'
        ),
        "active_safety_alerts": (
            f'count(ALERTS{{alertstate="firing",alertname=~"{safety_alerts}"}}) '
            "or vector(0)"
        ),
    }


def collect_metrics(
    client: PrometheusClient,
    *,
    window_hours: int,
) -> tuple[CanaryMetrics, dict[str, str]]:
    queries = build_promql_queries(window_hours=window_hours)
    values = {name: client.query(expression) for name, expression in queries.items()}

    def count(name: str) -> float:
        return max(0.0, float(values[name] or 0.0))

    metrics = CanaryMetrics(
        baseline_p95_seconds=values["baseline_p95_seconds"],
        canary_p95_seconds=values["canary_p95_seconds"],
        request_count=count("request_count"),
        error_5xx_count=count("error_5xx_count"),
        semantic_lookup_count=count("semantic_lookup_count"),
        semantic_lookup_error_count=count("semantic_lookup_error_count"),
        semantic_hit_count=count("semantic_hit_count"),
        availability_min=values["availability_min"],
        active_safety_alerts=count("active_safety_alerts"),
    )
    return metrics, queries


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _latency_improvement(metrics: CanaryMetrics) -> Optional[float]:
    baseline = metrics.baseline_p95_seconds
    canary = metrics.canary_p95_seconds
    if baseline is None or canary is None or baseline <= 0:
        return None
    return round((baseline - canary) / baseline, 6)


def _next_rollout_percent(current_percent: int) -> int:
    current = max(0, min(100, int(current_percent)))
    if current < 25:
        return 25
    return 100


def build_report(
    *,
    metrics: CanaryMetrics,
    current_rollout_percent: int,
    stage_started_at: datetime,
    now: datetime,
    shadow_gate_passed: bool,
    min_stage_hours: float = DEFAULT_MIN_STAGE_HOURS,
    min_requests: int = DEFAULT_MIN_REQUESTS,
    min_latency_improvement: float = DEFAULT_MIN_LATENCY_IMPROVEMENT,
    max_5xx_ratio: float = DEFAULT_MAX_5XX_RATIO,
    max_cache_error_ratio: float = DEFAULT_MAX_CACHE_ERROR_RATIO,
    min_semantic_hits: int = DEFAULT_MIN_SEMANTIC_HITS,
) -> dict[str, Any]:
    if stage_started_at.tzinfo is None or now.tzinfo is None:
        raise ValueError("stage_started_at and now must include a timezone")

    elapsed_hours = max(
        0.0,
        (now - stage_started_at).total_seconds() / 3600.0,
    )
    latency_improvement = _latency_improvement(metrics)
    error_5xx_ratio = _ratio(metrics.error_5xx_count, metrics.request_count)
    cache_error_ratio = _ratio(
        metrics.semantic_lookup_error_count,
        metrics.semantic_lookup_count,
    )
    valid_stage = int(current_rollout_percent) in {5, 25, 100}
    checks = {
        "valid_serving_stage": valid_stage,
        "shadow_gate_passed": bool(shadow_gate_passed),
        "minimum_stage_duration_met": elapsed_hours >= max(0.0, min_stage_hours),
        "minimum_requests_met": metrics.request_count >= max(1, min_requests),
        "latency_improvement_met": (
            latency_improvement is not None
            and latency_improvement >= min_latency_improvement
        ),
        "http_5xx_within_budget": (
            error_5xx_ratio is not None and error_5xx_ratio <= max_5xx_ratio
        ),
        "cache_errors_within_budget": (
            cache_error_ratio is not None
            and cache_error_ratio <= max_cache_error_ratio
        ),
        "semantic_hits_observed": metrics.semantic_hit_count
        >= max(1, min_semantic_hits),
        "availability_continuous": (
            metrics.availability_min is not None
            and metrics.availability_min >= 1.0
        ),
        "no_active_safety_alerts": metrics.active_safety_alerts <= 0,
    }
    enough_http_traffic = metrics.request_count >= max(1, min_requests)
    enough_cache_traffic = metrics.semantic_lookup_count >= 20
    safety_stop = (
        not valid_stage
        or not shadow_gate_passed
        or not checks["availability_continuous"]
        or not checks["no_active_safety_alerts"]
        or (enough_http_traffic and not checks["http_5xx_within_budget"])
        or (enough_cache_traffic and not checks["cache_errors_within_budget"])
    )
    gate_passed = all(checks.values())
    if safety_stop:
        recommended_rollout = 0
        decision = "stop"
    elif gate_passed:
        recommended_rollout = _next_rollout_percent(current_rollout_percent)
        decision = "promote"
    else:
        recommended_rollout = int(current_rollout_percent)
        decision = "hold"

    return {
        "summary": {
            "decision": decision,
            "gate_passed": gate_passed,
            "safety_stop": safety_stop,
            "current_rollout_percent": int(current_rollout_percent),
            "recommended_rollout_percent": recommended_rollout,
            "stage_elapsed_hours": round(elapsed_hours, 4),
            "latency_improvement_ratio": latency_improvement,
            "error_5xx_ratio": error_5xx_ratio,
            "semantic_cache_error_ratio": cache_error_ratio,
            "checks": checks,
            "thresholds": {
                "min_stage_hours": min_stage_hours,
                "min_requests": min_requests,
                "min_latency_improvement": min_latency_improvement,
                "max_5xx_ratio": max_5xx_ratio,
                "max_cache_error_ratio": max_cache_error_ratio,
                "min_semantic_hits": min_semantic_hits,
            },
        },
        "metrics": asdict(metrics),
    }


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def _load_shadow_gate(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool((payload.get("summary") or {}).get("rollout_gate_passed"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a semantic cache serving canary from Prometheus metrics."
    )
    parser.add_argument("--prometheus-url", default="http://prometheus:9090")
    parser.add_argument("--shadow-gate-report", required=True)
    parser.add_argument("--stage-started-at", required=True)
    parser.add_argument("--current-rollout-percent", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--min-stage-hours", type=float, default=DEFAULT_MIN_STAGE_HOURS)
    parser.add_argument("--min-requests", type=int, default=DEFAULT_MIN_REQUESTS)
    parser.add_argument(
        "--min-latency-improvement",
        type=float,
        default=DEFAULT_MIN_LATENCY_IMPROVEMENT,
    )
    parser.add_argument("--max-5xx-ratio", type=float, default=DEFAULT_MAX_5XX_RATIO)
    parser.add_argument(
        "--max-cache-error-ratio",
        type=float,
        default=DEFAULT_MAX_CACHE_ERROR_RATIO,
    )
    parser.add_argument(
        "--min-semantic-hits",
        type=int,
        default=DEFAULT_MIN_SEMANTIC_HITS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = PrometheusClient(args.prometheus_url, timeout_seconds=args.timeout)
    metrics, queries = collect_metrics(client, window_hours=args.window_hours)
    report = build_report(
        metrics=metrics,
        current_rollout_percent=args.current_rollout_percent,
        stage_started_at=_parse_datetime(args.stage_started_at),
        now=datetime.now(timezone.utc),
        shadow_gate_passed=_load_shadow_gate(Path(args.shadow_gate_report)),
        min_stage_hours=args.min_stage_hours,
        min_requests=args.min_requests,
        min_latency_improvement=args.min_latency_improvement,
        max_5xx_ratio=args.max_5xx_ratio,
        max_cache_error_ratio=args.max_cache_error_ratio,
        min_semantic_hits=args.min_semantic_hits,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["promql"] = queries
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(
        "semantic cache canary "
        f"decision={summary['decision']} "
        f"current={summary['current_rollout_percent']} "
        f"recommended={summary['recommended_rollout_percent']}"
    )
    if summary["gate_passed"]:
        return 0
    return 2 if summary["safety_stop"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
