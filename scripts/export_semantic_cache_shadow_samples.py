#!/usr/bin/env python3
"""Export completed semantic-cache shadow observations for release evaluation."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Optional

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This standalone exporter only needs validated settings, not FastAPI app startup.
os.environ.setdefault("BEGA_SKIP_APP_INIT", "1")

from app.config import get_settings


DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_QUERY_TIMEOUT_SECONDS = 30.0
DEFAULT_EXPORT_LIMIT = 1_000


def build_export_query(
    *, days: int, limit: int, route: Optional[str]
) -> tuple[str, tuple[Any, ...]]:
    clauses = [
        "fresh_answer IS NOT NULL",
        "observed_at >= now() - make_interval(days => %s)",
    ]
    params: list[Any] = [max(1, int(days))]
    normalized_route = str(route or "").strip().lower()
    if normalized_route:
        clauses.append("route = %s")
        params.append(normalized_route)
    params.append(max(1, min(int(limit), 100_000)))
    return (
        f"""
        SELECT id, request_cache_key, candidate_cache_key, route,
               question_text, filters_json, cached_answer, fresh_answer,
               similarity, observed_at
        FROM chat_semantic_cache_shadow_observation
        WHERE {' AND '.join(clauses)}
        ORDER BY observed_at ASC, id ASC
        LIMIT %s
        """,
        tuple(params),
    )


def row_to_sample(row: tuple[Any, ...]) -> dict[str, Any]:
    filters = row[5]
    if isinstance(filters, str):
        try:
            filters = json.loads(filters)
        except json.JSONDecodeError:
            filters = None
    observed_at = row[9]
    if hasattr(observed_at, "isoformat"):
        observed_at = observed_at.isoformat()
    return {
        "observation_id": int(row[0]),
        "cache_key": str(row[2]),
        "request_cache_key": str(row[1]),
        "route": str(row[3]),
        "question": str(row[4]),
        "filters": filters if isinstance(filters, dict) else None,
        "cached_answer": str(row[6]),
        "fresh_answer": str(row[7]),
        "similarity": float(row[8]),
        "observed_at": str(observed_at),
    }


async def export_samples(
    *,
    days: int,
    limit: int,
    route: Optional[str],
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    query_timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    sql, params = build_export_query(days=days, limit=limit, route=route)
    connection = await _connect_database(
        connect_timeout_seconds=connect_timeout_seconds
    )
    try:
        async with asyncio.timeout(max(0.1, float(query_timeout_seconds))):
            cursor = await connection.execute(sql, params)
            rows = await cursor.fetchall()
    finally:
        await connection.close()
    return [row_to_sample(tuple(row)) for row in rows]


async def _connect_database(
    *, connect_timeout_seconds: float
) -> psycopg.AsyncConnection[Any]:
    settings = get_settings()
    return await psycopg.AsyncConnection.connect(
        settings.database_url,
        autocommit=True,
        connect_timeout=max(1, int(connect_timeout_seconds)),
        target_session_attrs="read-write",
        options="-c search_path=public,security,extensions",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export completed semantic-cache shadow observations."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=DEFAULT_EXPORT_LIMIT)
    parser.add_argument("--route", choices=("completion", "stream"), default=None)
    parser.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--query-timeout-seconds",
        type=float,
        default=DEFAULT_QUERY_TIMEOUT_SECONDS,
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    samples = await export_samples(
        days=args.days,
        limit=args.limit,
        route=args.route,
        connect_timeout_seconds=args.connect_timeout_seconds,
        query_timeout_seconds=args.query_timeout_seconds,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "window_days": max(1, int(args.days)),
        "route": args.route,
        "samples": samples,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"semantic cache shadow samples: {output_path} ({len(samples)} rows)")
    return 0


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
