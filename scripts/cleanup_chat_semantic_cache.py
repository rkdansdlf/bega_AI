#!/usr/bin/env python3
"""Dry-run and purge utility for chat semantic response cache."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.deps import get_connection_pool


@dataclass
class CleanupScope:
    days: int
    source_tier: Optional[str]
    embedding_signature: Optional[str]
    max_hit_count: Optional[int]
    limit: int
    dry_run: bool
    allow_global: bool


def build_where_clause(scope: CleanupScope) -> tuple[str, List[Any]]:
    clauses = ["expires_at < now() - make_interval(days => %s)"]
    params: List[Any] = [max(0, int(scope.days))]

    if scope.source_tier:
        clauses.append("source_tier = %s")
        params.append(scope.source_tier)

    if scope.embedding_signature:
        clauses.append("embedding_signature = %s")
        params.append(scope.embedding_signature)

    if scope.max_hit_count is not None:
        clauses.append("hit_count <= %s")
        params.append(max(0, int(scope.max_hit_count)))

    if not scope.allow_global and len(clauses) == 1 and scope.days <= 0:
        raise ValueError("global immediate purge requires --allow-global")

    return " AND ".join(clauses), params


async def cleanup_chat_semantic_cache(scope: CleanupScope) -> dict[str, Any]:
    where_clause, params = build_where_clause(scope)
    bounded_limit = max(1, min(int(scope.limit), 10_000))
    pool = get_connection_pool()

    async with pool.connection() as conn:
        cur = await conn.execute(
            f"""
            SELECT cache_key, intent, source_tier, hit_count, expires_at
            FROM chat_semantic_response_cache
            WHERE {where_clause}
            ORDER BY expires_at ASC
            LIMIT %s
            """,
            (*params, bounded_limit),
        )
        rows = await cur.fetchall()
        cache_keys = [row[0] for row in rows]

        if not cache_keys or scope.dry_run:
            return {
                "deleted": 0,
                "would_delete": len(cache_keys),
                "sample": [
                    {
                        "cache_key": row[0],
                        "intent": row[1],
                        "source_tier": row[2],
                        "hit_count": row[3],
                        "expires_at": str(row[4]),
                    }
                    for row in rows[:20]
                ],
            }

        result = await conn.execute(
            "DELETE FROM chat_semantic_response_cache WHERE cache_key = ANY(%s)",
            (cache_keys,),
        )
        await conn.commit()
        return {
            "deleted": getattr(result, "rowcount", 0) or 0,
            "would_delete": len(cache_keys),
            "sample": [],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean up chat semantic response cache."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=0,
        help="Delete entries expired more than N days ago.",
    )
    parser.add_argument("--source-tier", default=None)
    parser.add_argument("--embedding-signature", default=None)
    parser.add_argument("--max-hit-count", type=int, default=None)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-global", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    scope = CleanupScope(
        days=args.days,
        source_tier=args.source_tier,
        embedding_signature=args.embedding_signature,
        max_hit_count=args.max_hit_count,
        limit=args.limit,
        dry_run=args.dry_run,
        allow_global=args.allow_global,
    )
    result = await cleanup_chat_semantic_cache(scope)
    payload = {
        "status": "PASS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": asdict(scope),
        "result": result,
    }
    output = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
