#!/usr/bin/env python3
"""Operate embedding generations (blue/green switch for retrieval).

    manage_embedding_generation.py list
    manage_embedding_generation.py register g2 --model pplx-embed --dim 1536 --version 3
    manage_embedding_generation.py audit g2
    manage_embedding_generation.py activate g2 [--min-coverage 0.995] [--force]
    manage_embedding_generation.py rollback

Rows for a new generation are written by the crawler, not here. ``activate``
refuses until the coverage audit passes, then flips the ACTIVE pointer in one
transaction. Requires migration 007 and RAG_GENERATION_GATE_ENABLED=true to
take effect at retrieval time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import embedding_generations as eg  # noqa: E402


async def _run(args: argparse.Namespace) -> int:
    from app.deps import get_rag_connection_pool

    pool = get_rag_connection_pool()
    await pool.open(wait=True, timeout=10.0)
    try:
        async with pool.connection() as conn:
            if args.cmd == "list":
                cur = await conn.execute(
                    "SELECT generation_id, embedding_model, embedding_dim, "
                    "embedding_version, status FROM rag_embedding_generations "
                    "ORDER BY created_at"
                )
                for row in await cur.fetchall():
                    print("\t".join(str(c) for c in row))
            elif args.cmd == "register":
                await eg.register_generation(
                    conn,
                    generation_id=args.generation_id,
                    embedding_model=args.model,
                    embedding_dim=args.dim,
                    embedding_version=args.version,
                    note=args.note,
                )
                print(f"registered {args.generation_id} (BUILDING)")
            elif args.cmd == "audit":
                gen = await eg.get_generation(conn, args.generation_id)
                if gen is None:
                    print(f"unknown generation: {args.generation_id}", file=sys.stderr)
                    return 2
                print(
                    json.dumps(await eg.audit_generation_coverage(conn, gen), indent=2)
                )
            elif args.cmd == "activate":
                out = await eg.activate_generation(
                    conn,
                    args.generation_id,
                    min_coverage=args.min_coverage,
                    force=args.force,
                )
                print(json.dumps(out, indent=2, default=str))
            elif args.cmd == "rollback":
                print(
                    json.dumps(
                        await eg.rollback_generation(conn), indent=2, default=str
                    )
                )
    except eg.GenerationError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    finally:
        await pool.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("rollback")
    reg = sub.add_parser("register")
    reg.add_argument("generation_id")
    reg.add_argument("--model", required=True)
    reg.add_argument("--dim", type=int, required=True)
    reg.add_argument("--version", type=int, required=True)
    reg.add_argument("--note")
    aud = sub.add_parser("audit")
    aud.add_argument("generation_id")
    act = sub.add_parser("activate")
    act.add_argument("generation_id")
    act.add_argument("--min-coverage", type=float, default=eg.DEFAULT_MIN_COVERAGE)
    act.add_argument("--force", action="store_true")
    return asyncio.run(_run(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
