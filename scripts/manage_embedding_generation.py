#!/usr/bin/env python3
"""Operate embedding generations (blue/green switch for retrieval).

    manage_embedding_generation.py list
    manage_embedding_generation.py register g1 --model pplx-embed --dim 1536 --version 2 --mirror-inline
    manage_embedding_generation.py backfill g1            # copy current inline vectors (no re-embed)
    manage_embedding_generation.py build-index g1         # partial HNSW index (CONCURRENTLY)
    manage_embedding_generation.py audit g1
    manage_embedding_generation.py activate g1 [--min-coverage 0.995] [--force]
    manage_embedding_generation.py rollback
    manage_embedding_generation.py drop g0                # reclaim space of a retired generation

Physical store (``--store generations``, the default when RAG_EMBEDDING_STORE is
``generations``; migration 009): generations coexist in ``rag_chunk_embeddings``.
Adopting it, once:

    register g1 --mirror-inline  →  backfill g1  →  build-index g1  →  activate g1
    then set RAG_EMBEDDING_STORE=generations and restart.

Rolling to a new model: ``register g2 --model … --dim …``, have the embedding
writer call ``embedding_generations.write_embeddings`` (or write
``rag_chunk_embeddings`` directly), ``build-index g2``, ``audit g2``, ``activate
g2``. ``rollback`` flips back to g1, whose rows were never touched.

Rows for a *new model* are written by the crawler / a re-embed job, not here.
``activate`` refuses until the ANN index exists and coverage passes, then flips
the ACTIVE pointer in one transaction. Pass ``--db-url`` to target a database
directly instead of the configured RAG pool.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import embedding_generations as eg  # noqa: E402


@contextlib.asynccontextmanager
async def _connect(db_url: str | None):
    """Autocommit connection (CREATE INDEX CONCURRENTLY cannot run in a txn)."""
    if db_url:
        import psycopg

        async with await psycopg.AsyncConnection.connect(
            db_url, autocommit=True
        ) as conn:
            yield conn
        return
    from app.deps import get_rag_connection_pool

    pool = get_rag_connection_pool()
    await pool.open(wait=True, timeout=10.0)
    try:
        async with pool.connection() as conn:
            yield conn
    finally:
        await pool.close()


def _default_store() -> str:
    from app.config import get_settings

    return str(getattr(get_settings(), "rag_embedding_store", eg.STORE_INLINE))


async def _list(conn, store: str) -> None:
    extended = store == eg.STORE_GENERATIONS
    cur = await conn.execute(
        "SELECT generation_id, embedding_model, embedding_dim, embedding_version, "
        "status"
        + (", mirror_inline, index_ready" if extended else "")
        + " FROM rag_embedding_generations ORDER BY created_at"
    )
    for row in await cur.fetchall():
        line = "\t".join(str(c) for c in row)
        if extended:
            count = await conn.execute(
                "SELECT count(*) FROM rag_chunk_embeddings WHERE generation_id = %s",
                (row[0],),
            )
            line += f"\trows={(await count.fetchone())[0]}"
        print(line)


async def _run(args: argparse.Namespace) -> int:
    store = args.store or _default_store()
    extended = store == eg.STORE_GENERATIONS
    try:
        async with _connect(args.db_url) as conn:
            if args.cmd == "list":
                await _list(conn, store)
            elif args.cmd == "register":
                await eg.register_generation(
                    conn,
                    generation_id=args.generation_id,
                    embedding_model=args.model,
                    embedding_dim=args.dim,
                    embedding_version=args.version,
                    note=args.note,
                    mirror_inline=args.mirror_inline,
                )
                print(f"registered {args.generation_id} (BUILDING)")
            elif args.cmd == "backfill":
                out = await eg.backfill_from_inline(
                    conn, args.generation_id, batch_size=args.batch_size
                )
                print(json.dumps(out, indent=2))
            elif args.cmd == "build-index":
                print(
                    json.dumps(
                        await eg.build_generation_index(conn, args.generation_id),
                        indent=2,
                    )
                )
            elif args.cmd == "audit":
                gen = await eg.get_generation(
                    conn, args.generation_id, extended=extended
                )
                if gen is None:
                    print(f"unknown generation: {args.generation_id}", file=sys.stderr)
                    return 2
                print(
                    json.dumps(
                        await eg.audit_generation_coverage(conn, gen, store=store),
                        indent=2,
                    )
                )
            elif args.cmd == "activate":
                out = await eg.activate_generation(
                    conn,
                    args.generation_id,
                    min_coverage=args.min_coverage,
                    force=args.force,
                    store=store,
                )
                print(json.dumps(out, indent=2, default=str))
            elif args.cmd == "rollback":
                print(
                    json.dumps(
                        await eg.rollback_generation(conn, store=store),
                        indent=2,
                        default=str,
                    )
                )
            elif args.cmd == "drop":
                print(
                    json.dumps(
                        await eg.drop_generation_data(conn, args.generation_id),
                        indent=2,
                    )
                )
    except eg.GenerationError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--db-url", help="connect directly (autocommit) instead of the RAG pool"
    )
    p.add_argument(
        "--store",
        choices=[eg.STORE_INLINE, eg.STORE_GENERATIONS],
        help="default: RAG_EMBEDDING_STORE",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("rollback")
    reg = sub.add_parser("register")
    reg.add_argument("generation_id")
    reg.add_argument("--model", required=True)
    reg.add_argument("--dim", type=int, required=True)
    reg.add_argument("--version", type=int, required=True)
    reg.add_argument("--note")
    reg.add_argument(
        "--mirror-inline",
        action="store_true",
        help="mirror inline rag_chunks.embedding writes with this signature "
        "into the generation (physical store)",
    )
    bf = sub.add_parser("backfill")
    bf.add_argument("generation_id")
    bf.add_argument("--batch-size", type=int, default=5000)
    for name in ("build-index", "audit", "drop"):
        sub.add_parser(name).add_argument("generation_id")
    act = sub.add_parser("activate")
    act.add_argument("generation_id")
    act.add_argument("--min-coverage", type=float, default=eg.DEFAULT_MIN_COVERAGE)
    act.add_argument("--force", action="store_true")
    return p


def main(argv=None) -> int:
    return asyncio.run(_run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
