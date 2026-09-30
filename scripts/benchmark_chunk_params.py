#!/usr/bin/env python3
"""Compare semantic-chunking parameters against the retrieval benchmark cases.

    python scripts/benchmark_chunk_params.py                       # default grid
    python scripts/benchmark_chunk_params.py --config 650/900/80 --config 400/700/60
    python scripts/benchmark_chunk_params.py --out evals/reports/chunk_params.json

Each config is ``target/max/overlap`` (chars). For every config the static
document corpus is re-chunked with ``smart_chunks``, embedded with the
configured embedding provider, and every ``DEFAULT_CASES`` query is ranked by
cosine similarity. A chunk is *relevant* when it contains all of the case's
``expected_substrings`` — so configs that split an answer across chunks are
penalised, which is exactly the failure mode chunk size controls.

This is the evidence for the default 650/900/80. NOTE: with
``EMBED_PROVIDER=local`` vectors are deterministic pseudo-vectors and ranks are
NOT meaningful; the report flags that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GRID: Tuple[Tuple[int, int, int], ...] = (
    (650, 900, 80),  # current default
    (400, 700, 60),
    (900, 1200, 100),
    (500, 800, 80),
    (650, 900, 0),  # overlap ablation
)
TOP_K = 5
EmbedFn = Callable[[Sequence[str]], Awaitable[List[List[float]]]]


def parse_config(raw: str) -> Tuple[int, int, int]:
    target, max_chars, overlap = (int(p) for p in raw.split("/"))
    if not (0 < target <= max_chars) or overlap < 0 or overlap >= target:
        raise ValueError(f"invalid chunk config: {raw}")
    return target, max_chars, overlap


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def score_config(
    chunks: Sequence[Dict[str, Any]],
    chunk_embeddings: Sequence[Sequence[float]],
    cases: Sequence[Dict[str, Any]],
    query_embeddings: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Recall@K / MRR / top-1 source precision for one chunking of the corpus."""
    hits = 0
    rr_sum = 0.0
    top1_ok = 0
    top1_total = 0
    scored = 0
    for case, q_emb in zip(cases, query_embeddings):
        expected = tuple(case.get("expected_substrings") or ())
        ranked = sorted(
            range(len(chunks)),
            key=lambda i: cosine(q_emb, chunk_embeddings[i]),
            reverse=True,
        )
        if case.get("expected_top_table") and ranked:
            top1_total += 1
            if chunks[ranked[0]]["source_table"] == case["expected_top_table"]:
                top1_ok += 1
        if not expected:
            continue
        scored += 1
        first_rank: Optional[int] = None
        for rank, idx in enumerate(ranked, start=1):
            if all(tok in chunks[idx]["content"] for tok in expected):
                first_rank = rank
                break
        if first_rank is not None and first_rank <= TOP_K:
            hits += 1
        if first_rank is not None:
            rr_sum += 1.0 / first_rank
    lengths = [len(c["content"]) for c in chunks] or [0]
    return {
        "chunks": len(chunks),
        "avg_chunk_chars": round(sum(lengths) / len(lengths), 1),
        "max_chunk_chars": max(lengths),
        "scored_cases": scored,
        f"recall@{TOP_K}": round(hits / scored, 4) if scored else None,
        "mrr": round(rr_sum / scored, 4) if scored else None,
        "top1_source_precision": round(top1_ok / top1_total, 4) if top1_total else None,
    }


def rank_configs(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Best first: recall, then MRR, then top-1 precision, then fewer chunks."""

    def key(r: Dict[str, Any]) -> Tuple:
        m = r["metrics"]
        return (
            -(m.get(f"recall@{TOP_K}") or 0.0),
            -(m.get("mrr") or 0.0),
            -(m.get("top1_source_precision") or 0.0),
            m["chunks"],
        )

    return sorted(results, key=key)


def chunk_corpus(config: Tuple[int, int, int]) -> List[Dict[str, Any]]:
    from scripts.benchmark_retrieval import (
        _iter_document_profiles,
        build_static_source_row_id,
        read_static_profile_content,
    )
    from app.core.chunking import smart_chunks

    target, max_chars, overlap = config
    rows: List[Dict[str, Any]] = []
    for profile_key, profile in _iter_document_profiles():
        content = read_static_profile_content(profile_key, profile).strip()
        if not content:
            continue
        chunks = (
            [content]
            if profile.get("single_chunk")
            else smart_chunks(
                content,
                target_chars=target,
                max_chars=max_chars,
                overlap_chars=overlap,
            )
        )
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            rows.append(
                {
                    "source_table": str(profile.get("source_table", profile_key)),
                    "source_row_id": build_static_source_row_id(
                        profile_key, profile, chunk_index=idx, total_chunks=total
                    ),
                    "content": chunk,
                }
            )
    return rows


async def run(
    configs: Sequence[Tuple[int, int, int]], embed: EmbedFn
) -> Dict[str, Any]:
    from scripts.benchmark_retrieval import DEFAULT_CASES

    cases = [
        {
            "query": c.query,
            "expected_substrings": list(c.expected_substrings),
            "expected_top_table": c.expected_top_table,
        }
        for c in DEFAULT_CASES
    ]
    query_embeddings = await embed([c["query"] for c in cases])
    results = []
    for config in configs:
        chunks = chunk_corpus(config)
        embeddings = await embed([c["content"] for c in chunks]) if chunks else []
        results.append(
            {
                "config": "/".join(str(v) for v in config),
                "metrics": score_config(chunks, embeddings, cases, query_embeddings),
            }
        )
    ranked = rank_configs(results)
    return {
        "top_k": TOP_K,
        "cases": len(cases),
        "ranking": [r["config"] for r in ranked],
        "results": results,
        "best": ranked[0]["config"] if ranked else None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", action="append", help="target/max/overlap")
    p.add_argument("--out")
    args = p.parse_args(argv)
    configs = (
        [parse_config(c) for c in args.config] if args.config else list(DEFAULT_GRID)
    )

    from app.config import get_settings
    from app.core.embeddings import async_embed_texts

    settings = get_settings()

    async def embed(texts: Sequence[str]) -> List[List[float]]:
        return await async_embed_texts(list(texts), settings)

    report = asyncio.run(run(configs, embed))
    provider = str(getattr(settings, "embed_provider", "unknown"))
    report["embed_provider"] = provider
    report["meaningful"] = provider != "local"
    if not report["meaningful"]:
        print(
            "WARNING: EMBED_PROVIDER=local uses pseudo-vectors; ranking below is "
            "not evidence.",
            file=sys.stderr,
        )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
