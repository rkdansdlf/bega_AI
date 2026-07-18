#!/usr/bin/env python3
"""
RAG 파이프라인 성능 대시보드.

Prometheus 메트릭 또는 DB 직접 쿼리로 다음 지표를 출력합니다:
  - 임베딩 캐시 히트율
  - 채팅 응답 캐시 히트율 (intent별)
  - Coach 캐시 상태 분포
  - 벡터 검색 레이턴시 (DB 있을 때)

사용법:
  PYTHONPATH=. python scripts/performance_dashboard.py
  PYTHONPATH=. python scripts/performance_dashboard.py --db            # DB 쿼리 포함
  PYTHONPATH=. python scripts/performance_dashboard.py --json          # JSON 출력
  PYTHONPATH=. python scripts/performance_dashboard.py --prometheus    # Prometheus 스크랩
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()


# ── Prometheus 메트릭 이름 ──────────────────────────────────────────────────
METRIC_EMBEDDING_CACHE_HIT = "ai_embedding_cache_hit_total"
METRIC_EMBEDDING_CACHE_MISS = "ai_embedding_cache_miss_total"
METRIC_RESPONSE_CACHE_HIT = "ai_response_cache_hit_total"
METRIC_RESPONSE_CACHE_MISS = "ai_response_cache_miss_total"
METRIC_VECTOR_SEARCH_LATENCY = "ai_vector_search_latency_seconds"


# ── DB 쿼리 ────────────────────────────────────────────────────────────────
_CHAT_CACHE_STATS_SQL = """
SELECT
    COALESCE(intent, 'unknown') AS intent,
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE status = 'COMPLETED') AS completed,
    COUNT(*) FILTER (WHERE expires_at > now()) AS active,
    COUNT(*) FILTER (WHERE expires_at <= now()) AS expired,
    ROUND(AVG(EXTRACT(EPOCH FROM (expires_at - created_at)) / 3600)::numeric, 1) AS avg_ttl_hours
FROM chat_response_cache
GROUP BY intent
ORDER BY total_rows DESC
"""

_COACH_CACHE_STATS_SQL = """
SELECT
    status,
    COUNT(*) AS count,
    COUNT(*) FILTER (WHERE updated_at > now() - INTERVAL '1 hour') AS updated_last_1h,
    COUNT(*) FILTER (WHERE updated_at > now() - INTERVAL '24 hours') AS updated_last_24h
FROM coach_analysis_cache
GROUP BY status
ORDER BY count DESC
"""

_RECENT_CACHE_ACTIVITY_SQL = """
SELECT
    DATE_TRUNC('hour', created_at) AS hour_bucket,
    COUNT(*) AS new_entries,
    COUNT(*) FILTER (WHERE status = 'COMPLETED') AS completed
FROM chat_response_cache
WHERE created_at > now() - INTERVAL '24 hours'
GROUP BY hour_bucket
ORDER BY hour_bucket DESC
LIMIT 12
"""


def _try_db_query(db_url: str, sql: str) -> Optional[List[Dict[str, Any]]]:
    """psycopg3로 DB 쿼리. 실패 시 None 반환."""
    try:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(db_url, row_factory=dict_row, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall()
    except Exception as exc:
        return None


def _fetch_prometheus_metric(base_url: str, metric: str) -> Optional[float]:
    """Prometheus API에서 메트릭 현재값 조회."""
    try:
        import urllib.request
        import urllib.parse

        query = urllib.parse.urlencode({"query": metric})
        url = f"{base_url}/api/v1/query?{query}"
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read())
            results = data.get("data", {}).get("result", [])
            if results:
                return float(results[0]["value"][1])
    except Exception:
        pass
    return None


def collect_embedding_cache_metrics(
    prometheus_url: Optional[str] = None,
) -> Dict[str, Any]:
    """임베딩 캐시 히트율 수집."""
    metrics: Dict[str, Any] = {"source": "unavailable"}

    if prometheus_url:
        hit = _fetch_prometheus_metric(prometheus_url, METRIC_EMBEDDING_CACHE_HIT)
        miss = _fetch_prometheus_metric(prometheus_url, METRIC_EMBEDDING_CACHE_MISS)
        if hit is not None and miss is not None:
            total = hit + miss
            metrics = {
                "source": "prometheus",
                "hit_total": hit,
                "miss_total": miss,
                "hit_rate_pct": round(hit / total * 100, 1) if total > 0 else 0.0,
            }

    return metrics


def collect_chat_cache_metrics(
    db_url: Optional[str], prometheus_url: Optional[str]
) -> Dict[str, Any]:
    """채팅 응답 캐시 지표 수집."""
    result: Dict[str, Any] = {}

    if prometheus_url:
        hit = _fetch_prometheus_metric(prometheus_url, METRIC_RESPONSE_CACHE_HIT)
        miss = _fetch_prometheus_metric(prometheus_url, METRIC_RESPONSE_CACHE_MISS)
        if hit is not None and miss is not None:
            total = hit + miss
            result["prometheus"] = {
                "hit_total": hit,
                "miss_total": miss,
                "hit_rate_pct": round(hit / total * 100, 1) if total > 0 else 0.0,
            }

    if db_url:
        rows = _try_db_query(db_url, _CHAT_CACHE_STATS_SQL)
        if rows is not None:
            result["by_intent"] = [dict(r) for r in rows]

        recent = _try_db_query(db_url, _RECENT_CACHE_ACTIVITY_SQL)
        if recent is not None:
            result["recent_24h"] = [dict(r) for r in recent]

    return result


def collect_coach_cache_metrics(db_url: Optional[str]) -> Dict[str, Any]:
    """Coach 캐시 상태 분포 수집."""
    if not db_url:
        return {}

    rows = _try_db_query(db_url, _COACH_CACHE_STATS_SQL)
    if rows is None:
        return {"error": "db_query_failed"}

    status_map = {r["status"]: dict(r) for r in rows}
    total = sum(r["count"] for r in rows)
    completed = status_map.get("COMPLETED", {}).get("count", 0)
    pending = status_map.get("PENDING", {}).get("count", 0)
    failed = status_map.get("FAILED", {}).get("count", 0)

    return {
        "total": total,
        "completed": completed,
        "pending": pending,
        "failed": failed,
        "completed_rate_pct": round(completed / total * 100, 1) if total > 0 else 0.0,
        "by_status": [dict(r) for r in rows],
    }


def build_dashboard(
    *,
    db_url: Optional[str],
    prometheus_url: Optional[str],
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()

    return {
        "collected_at": now,
        "embedding_cache": collect_embedding_cache_metrics(prometheus_url),
        "chat_cache": collect_chat_cache_metrics(db_url, prometheus_url),
        "coach_cache": collect_coach_cache_metrics(db_url),
    }


def _print_dashboard(data: Dict[str, Any]) -> None:
    print(f"\n{'='*62}")
    print(f"  RAG Performance Dashboard")
    print(f"  {data.get('collected_at', '')}")
    print(f"{'='*62}")

    # 임베딩 캐시
    emb = data.get("embedding_cache", {})
    print("\n[임베딩 캐시]")
    if emb.get("source") == "prometheus":
        print(f"  hit_rate : {emb.get('hit_rate_pct', 'N/A')}%")
        print(f"  hit      : {emb.get('hit_total', 'N/A')}")
        print(f"  miss     : {emb.get('miss_total', 'N/A')}")
    else:
        print(f"  데이터 없음 (--prometheus 옵션 또는 Prometheus 연결 필요)")

    # 채팅 캐시
    chat = data.get("chat_cache", {})
    print("\n[채팅 응답 캐시]")
    if "prometheus" in chat:
        p = chat["prometheus"]
        print(f"  hit_rate : {p.get('hit_rate_pct', 'N/A')}%")
    if "by_intent" in chat:
        print(
            f"  {'intent':<25} {'total':>6} {'completed':>10} {'active':>7} {'avg_ttl':>9}"
        )
        print(f"  {'-'*55}")
        for row in chat["by_intent"]:
            print(
                f"  {str(row.get('intent', '')):<25}"
                f" {row.get('total_rows', 0):>6}"
                f" {row.get('completed', 0):>10}"
                f" {row.get('active', 0):>7}"
                f" {str(row.get('avg_ttl_hours', 'N/A')):>8}h"
            )

    # Coach 캐시
    coach = data.get("coach_cache", {})
    print("\n[Coach 캐시]")
    if "error" in coach:
        print(f"  DB 오류: {coach['error']}")
    elif coach:
        print(f"  total     : {coach.get('total', 0)}")
        print(
            f"  completed : {coach.get('completed', 0)} ({coach.get('completed_rate_pct', 0)}%)"
        )
        print(f"  pending   : {coach.get('pending', 0)}")
        print(f"  failed    : {coach.get('failed', 0)}")
    else:
        print("  DB 연결 필요 (--db 옵션)")

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 파이프라인 성능 대시보드")
    parser.add_argument(
        "--db",
        action="store_true",
        help="PostgreSQL DB 쿼리 포함 (POSTGRES_DB_URL 환경변수 필요)",
    )
    parser.add_argument(
        "--prometheus",
        default="",
        metavar="URL",
        help="Prometheus base URL (예: http://localhost:9090)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="JSON 형식으로 출력",
    )
    parser.add_argument(
        "--output",
        default="",
        help="JSON 저장 경로 (선택)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    db_url: Optional[str] = None
    if args.db:
        db_url = os.getenv("POSTGRES_DB_URL", "")
        if not db_url:
            print(
                "ERROR: --db 옵션 사용 시 POSTGRES_DB_URL 환경변수가 필요합니다.",
                file=sys.stderr,
            )
            return 1

    prometheus_url = args.prometheus or None

    data = build_dashboard(db_url=db_url, prometheus_url=prometheus_url)

    if args.json_output:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    else:
        _print_dashboard(data)

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
