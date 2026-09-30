#!/usr/bin/env python3
"""Mine rag_retrieval_events for golden-dataset candidates.

    python scripts/mine_retrieval_events.py --days 7 --out evals/candidates.jsonl
    python scripts/mine_retrieval_events.py --events-file events.json   # offline

Extracts real user questions that hit zero-hit / fallback / relaxed-constraint /
low-similarity / bad-source paths. Output rows are UNLABELED (status
``needs_label``); fill ``relevant_doc_keys`` from the internal DB before moving
a row into ``evals/rag_retrieval_v1.jsonl``. Reads the internal database only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.event_mining import build_candidates, merge_feedback  # noqa: E402

_QUERY = """
SELECT user_query, intent, metadata_filter, retrieved_chunk_ids,
       selected_chunk_ids, scores, success, error_type, created_at
FROM rag_retrieval_events
WHERE created_at >= now() - make_interval(days => %s)
ORDER BY created_at DESC
LIMIT %s
"""


_FEEDBACK_QUERY = """
SELECT question, rating, corrected_fact
FROM rag_answer_feedback
WHERE created_at >= now() - make_interval(days => %s)
"""


async def _fetch_feedback(days: int) -> List[Dict[str, Any]]:
    from app.deps import get_connection_pool

    pool = get_connection_pool()
    await pool.open(wait=True, timeout=10.0)
    try:
        async with pool.connection() as conn:
            cur = await conn.execute(_FEEDBACK_QUERY, (days,))
            return [
                dict(zip(("question", "rating", "corrected_fact"), row))
                for row in await cur.fetchall()
            ]
    finally:
        await pool.close()


async def _fetch(days: int, limit: int) -> List[Dict[str, Any]]:
    from app.deps import get_rag_connection_pool

    pool = get_rag_connection_pool()
    await pool.open(wait=True, timeout=10.0)
    try:
        async with pool.connection() as conn:
            cur = await conn.execute(_QUERY, (days, limit))
            cols = [
                "user_query",
                "intent",
                "metadata_filter",
                "retrieved_chunk_ids",
                "selected_chunk_ids",
                "scores",
                "success",
                "error_type",
                "created_at",
            ]
            return [dict(zip(cols, row)) for row in await cur.fetchall()]
    finally:
        await pool.close()


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--min-occurrences", type=int, default=1)
    p.add_argument("--feedback-file", help="JSON list of feedback rows (offline)")
    p.add_argument(
        "--with-feedback",
        action="store_true",
        help="also read rag_answer_feedback from the DB",
    )
    p.add_argument("--events-file", help="JSON list of events (skips the DB)")
    p.add_argument("--out", help="write candidates as JSONL")
    args = p.parse_args(argv)

    if args.events_file:
        events = json.loads(Path(args.events_file).read_text(encoding="utf-8"))
    else:
        events = asyncio.run(_fetch(args.days, args.limit))

    report = build_candidates(events, min_occurrences=args.min_occurrences)
    feedback = None
    if args.feedback_file:
        feedback = json.loads(Path(args.feedback_file).read_text(encoding="utf-8"))
    elif args.with_feedback:
        feedback = asyncio.run(_fetch_feedback(args.days))
    if feedback is not None:
        report = merge_feedback(report, feedback)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for row in report["candidates"]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(report['candidates'])} candidates -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
