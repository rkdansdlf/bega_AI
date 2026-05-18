#!/usr/bin/env python3
"""
RAG 파이프라인 로그 분석 스크립트.

uvicorn/Python logging 출력을 파싱하여 다음을 리포트:
  1. 가장 느린 쿼리 TOP 10 (RAG stage duration 기준)
  2. 엔티티 추출 실패율 (entity_filter 필드 모두 None인 비율)
  3. 캐시 히트/미스 패턴
  4. Fallback 레벨 분포 (Level 1~4 사용 빈도)
  5. 오류율 및 에러 유형

사용법:
  python scripts/analyze_rag_logs.py --log-file logs/app.log
  python scripts/analyze_rag_logs.py --log-file logs/app.log --minutes 60
  python scripts/analyze_rag_logs.py --log-file logs/app.log --json
  cat logs/app.log | python scripts/analyze_rag_logs.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── 로그 패턴 ─────────────────────────────────────────────────────────────────

# 2026-05-10 18:05:57,178 - INFO - [EntityExtractor] Extracting entities from: ...
_LOG_PREFIX = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,\.]\d+)\s+-\s+(?P<level>\w+)\s+-\s+(?P<msg>.+)$"
)

# [RAG] Processing query: ...
_QUERY_START = re.compile(r"\[RAG\] Processing query: (?P<query>.+)")

# Hybrid RRF search took 123.45ms (results=5, hybrid=True)
_SEARCH_LATENCY = re.compile(
    r"\[Search\] Hybrid RRF search took (?P<ms>[\d.]+)ms \(results=(?P<results>\d+)"
)

# [EntityExtractor] Extracted entities: year=None, team=None, player=None, stat=None, ...
_ENTITY_EXTRACT = re.compile(r"\[EntityExtractor\] Extracted entities: (?P<fields>.+)")

# [Search] Fallback level_N returned K results (removed: [...])
_FALLBACK_RESULT = re.compile(
    r"\[Search\] Fallback (?P<level>level_\d+) returned (?P<count>\d+) results"
)

# [Search] Fallback: 0 results at level_N, retrying without 'key'
_FALLBACK_RETRY = re.compile(
    r"\[Search\] Fallback: 0 results at (?P<level>level_\d+), retrying without '(?P<key>\w+)'"
)

# AI_RESPONSE_CACHE_TOTAL ... result=hit or result=miss
_CACHE_HIT = re.compile(r"result=(?P<result>hit|miss)\b")

# Error patterns
_ERROR_PATTERN = re.compile(r"\[RAG\] .*(error|failed|exception)", re.IGNORECASE)


def _parse_ts(ts_str: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _all_none(fields_str: str) -> bool:
    """EntityFilter 출력에서 모든 필드가 None인지 확인."""
    # "year=None, team=None, player=None, ..." 형식 파싱
    values = re.findall(r"\w+=(\S+?)(?:,|$)", fields_str)
    return all(v in ("None", "null") for v in values) if values else False


# ── 파서 ──────────────────────────────────────────────────────────────────────

def parse_log_lines(
    lines,
    *,
    since: Optional[datetime] = None,
) -> Dict[str, Any]:
    """로그 라인 시퀀스를 파싱하여 분석 데이터를 수집한다."""
    data: Dict[str, Any] = {
        "queries": [],          # (query_text, latency_ms)
        "entity_failures": 0,  # 엔티티 추출 전부 None인 횟수
        "entity_total": 0,
        "fallback_levels": Counter(),    # level → count
        "fallback_retries": Counter(),   # removed_key → count
        "cache_results": Counter(),      # hit/miss
        "errors": [],
        "total_lines": 0,
    }

    current_query: Optional[str] = None
    current_ts: Optional[datetime] = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        data["total_lines"] += 1

        m = _LOG_PREFIX.match(line)
        if not m:
            continue

        ts = _parse_ts(m.group("ts"))
        if since and ts and ts < since:
            continue

        msg = m.group("msg")

        # 쿼리 시작 추적
        q_match = _QUERY_START.search(msg)
        if q_match:
            current_query = q_match.group("query").strip()
            current_ts = ts
            continue

        # 검색 레이턴시
        s_match = _SEARCH_LATENCY.search(msg)
        if s_match and current_query:
            data["queries"].append((
                current_query,
                float(s_match.group("ms")),
                int(s_match.group("results")),
            ))
            current_query = None
            continue

        # 엔티티 추출 결과
        e_match = _ENTITY_EXTRACT.search(msg)
        if e_match:
            data["entity_total"] += 1
            if _all_none(e_match.group("fields")):
                data["entity_failures"] += 1
            continue

        # Fallback 성공 레벨
        fb_match = _FALLBACK_RESULT.search(msg)
        if fb_match:
            data["fallback_levels"][fb_match.group("level")] += 1
            continue

        # Fallback 재시도
        fbr_match = _FALLBACK_RETRY.search(msg)
        if fbr_match:
            data["fallback_retries"][fbr_match.group("key")] += 1
            continue

        # 캐시 히트/미스
        c_match = _CACHE_HIT.search(msg)
        if c_match:
            data["cache_results"][c_match.group("result")] += 1
            continue

        # 에러
        if m.group("level") in ("ERROR", "CRITICAL"):
            data["errors"].append(msg[:200])

    return data


# ── 리포트 생성 ────────────────────────────────────────────────────────────────

def build_report(data: Dict[str, Any]) -> Dict[str, Any]:
    queries = data["queries"]
    queries_sorted = sorted(queries, key=lambda x: x[1], reverse=True)

    # 캐시 통계
    cache = data["cache_results"]
    cache_total = cache.get("hit", 0) + cache.get("miss", 0)
    cache_hit_rate = round(cache.get("hit", 0) / cache_total * 100, 1) if cache_total else None

    # 엔티티 추출 실패율
    entity_fail_rate = (
        round(data["entity_failures"] / data["entity_total"] * 100, 1)
        if data["entity_total"] > 0
        else None
    )

    # 평균 검색 레이턴시
    latencies = [q[1] for q in queries]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else None
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99) - 1] if len(latencies) >= 2 else latencies[0] if latencies else None

    return {
        "summary": {
            "total_log_lines": data["total_lines"],
            "total_queries_tracked": len(queries),
            "avg_search_latency_ms": avg_latency,
            "p99_search_latency_ms": p99_latency,
            "cache_hit_rate_pct": cache_hit_rate,
            "cache_hit_total": cache.get("hit", 0),
            "cache_miss_total": cache.get("miss", 0),
            "entity_failure_rate_pct": entity_fail_rate,
            "entity_failures": data["entity_failures"],
            "entity_total": data["entity_total"],
            "error_count": len(data["errors"]),
        },
        "slowest_queries": [
            {"query": q[:80], "latency_ms": round(ms, 1), "results": r}
            for q, ms, r in queries_sorted[:10]
        ],
        "fallback_level_distribution": dict(data["fallback_levels"]),
        "fallback_retried_keys": dict(data["fallback_retries"]),
        "recent_errors": data["errors"][:10],
    }


def _print_report(report: Dict[str, Any]) -> None:
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"  RAG Log Analysis Report")
    print(f"{'='*60}")
    print(f"\n[전체]")
    print(f"  분석된 로그 라인수 : {s['total_log_lines']}")
    print(f"  추적된 검색 횟수   : {s['total_queries_tracked']}")

    print(f"\n[검색 레이턴시]")
    print(f"  평균   : {s['avg_search_latency_ms']}ms" if s['avg_search_latency_ms'] else "  데이터 없음")
    print(f"  P99    : {s['p99_search_latency_ms']}ms" if s['p99_search_latency_ms'] else "")

    print(f"\n[캐시]")
    if s["cache_hit_rate_pct"] is not None:
        print(f"  히트율 : {s['cache_hit_rate_pct']}%  (hit={s['cache_hit_total']}, miss={s['cache_miss_total']})")
    else:
        print("  캐시 로그 없음")

    print(f"\n[엔티티 추출]")
    if s["entity_failure_rate_pct"] is not None:
        print(f"  실패율 : {s['entity_failure_rate_pct']}%  ({s['entity_failures']} / {s['entity_total']})")
    else:
        print("  엔티티 로그 없음")

    fbd = report["fallback_level_distribution"]
    if fbd:
        print(f"\n[Fallback 레벨 분포]")
        for level, cnt in sorted(fbd.items()):
            print(f"  {level} : {cnt}회")

    fbr = report["fallback_retried_keys"]
    if fbr:
        print(f"\n[Fallback 제거된 필터]")
        for key, cnt in sorted(fbr.items(), key=lambda x: -x[1]):
            print(f"  {key} : {cnt}회 제거")

    slow = report["slowest_queries"]
    if slow:
        print(f"\n[가장 느린 쿼리 TOP {len(slow)}]")
        for i, item in enumerate(slow, 1):
            print(f"  {i:2d}. [{item['latency_ms']}ms, results={item['results']}] {item['query']}")

    errors = report["recent_errors"]
    if errors:
        print(f"\n[최근 오류 ({len(errors)}건)]")
        for e in errors:
            print(f"  - {e}")

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 파이프라인 로그 분석")
    parser.add_argument(
        "--log-file",
        default="",
        help="로그 파일 경로 (미지정 시 stdin)",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=0,
        help="최근 N분 로그만 분석 (0=전체)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="JSON 형식 출력",
    )
    parser.add_argument(
        "--output",
        default="",
        help="JSON 저장 경로 (선택)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    since: Optional[datetime] = None
    if args.minutes > 0:
        since = datetime.now(timezone.utc) - timedelta(minutes=args.minutes)

    if args.log_file:
        log_path = Path(args.log_file).expanduser().resolve()
        if not log_path.exists():
            print(f"ERROR: 파일을 찾을 수 없습니다: {log_path}", file=sys.stderr)
            return 1
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            data = parse_log_lines(fh, since=since)
    else:
        data = parse_log_lines(sys.stdin, since=since)

    report = build_report(data)

    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
