#!/usr/bin/env python3
"""Golden RAG evaluation: retrieval and generation, scored separately.

    # offline gate (replays recorded results; used in CI)
    python scripts/eval_rag_golden.py retrieval --baseline evals/baselines/retrieval_v1.json
    python scripts/eval_rag_golden.py generation --baseline evals/baselines/generation_v1.json

    # refresh from the real internal DB (never touches external sources)
    python scripts/eval_rag_golden.py retrieval --live --write-recorded

Exit code 1 on any regression vs. the baseline or on a failed case, so the
script can block a PR. ``--update-baseline`` rewrites the baseline file.
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

from app.eval import grounding, retrieval_metrics as rm  # noqa: E402

EVALS = ROOT / "evals"

GEN_LOWER_IS_BETTER = {
    "unsupported_claim_rate",
    "numeric_hallucination_rate",
    "entity_hallucination_rate",
}
GEN_HIGHER_IS_BETTER = {"pass_rate", "citation_precision", "citation_recall"}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def _live_retrieval(cases: List[rm.RetrievalCase]) -> Dict[str, list]:
    from app.config import get_settings
    from app.core.embeddings import async_embed_query
    from app.core.retrieval import similarity_search_with_fallback
    from app.deps import get_rag_connection_pool

    settings = get_settings()
    pool = get_rag_connection_pool()
    await pool.open(wait=True, timeout=10.0)
    out: Dict[str, list] = {}
    try:
        for case in cases:
            embedding = await async_embed_query(case.question, settings)
            async with pool.connection() as conn:
                docs, _level = await similarity_search_with_fallback(
                    conn,
                    embedding,
                    limit=10,
                    filters=case.filters or None,
                    settings=settings,
                    intent="stats_lookup",
                )
            out[case.id] = [
                {
                    "source_table": d.get("source_table"),
                    "source_row_id": d.get("source_row_id"),
                }
                for d in docs
            ]
    finally:
        await pool.close()
    return out


def run_retrieval(args: argparse.Namespace) -> Dict[str, Any]:
    cases = [rm.RetrievalCase.from_dict(r) for r in _read_jsonl(Path(args.dataset))]
    recorded_path = Path(args.recorded)
    if args.live:
        recorded = asyncio.run(_live_retrieval(cases))
        if args.write_recorded:
            recorded_path.write_text(
                json.dumps(recorded, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    else:
        recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    rows = [rm.evaluate_case(c, recorded.get(c.id, [])) for c in cases]
    return {"summary": rm.aggregate(rows, cases), "rows": rows}


def run_generation(args: argparse.Namespace) -> Dict[str, Any]:
    cases = _read_jsonl(Path(args.dataset))
    recorded = json.loads(Path(args.recorded).read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        rec = recorded.get(case["id"]) or {"answer": "", "cited_sources": []}
        rows.append(
            grounding.evaluate_generation(case, rec["answer"], rec["cited_sources"])
        )
    return {"summary": grounding.aggregate_generation(rows), "rows": rows}


def compare_generation(
    cur: Dict[str, Any], base: Dict[str, Any], tol: float
) -> List[str]:
    problems = []
    for k in sorted(GEN_HIGHER_IS_BETTER):
        if (
            cur.get(k) is not None
            and base.get(k) is not None
            and cur[k] < base[k] - tol
        ):
            problems.append(f"{k} regressed: {cur[k]:.3f} < {base[k]:.3f}")
    for k in sorted(GEN_LOWER_IS_BETTER):
        if (
            cur.get(k) is not None
            and base.get(k) is not None
            and cur[k] > base[k] + tol
        ):
            problems.append(f"{k} regressed: {cur[k]:.3f} > {base[k]:.3f}")
    return problems


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["retrieval", "generation"])
    p.add_argument("--dataset")
    p.add_argument("--recorded")
    p.add_argument("--baseline")
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("--live", action="store_true")
    p.add_argument("--write-recorded", action="store_true")
    p.add_argument("--update-baseline", action="store_true")
    p.add_argument("--json", action="store_true", help="print full report as JSON")
    args = p.parse_args(argv)
    args.dataset = args.dataset or str(EVALS / f"rag_{args.kind}_v1.jsonl")
    args.recorded = args.recorded or str(EVALS / f"rag_{args.kind}_v1.recorded.json")

    report = run_retrieval(args) if args.kind == "retrieval" else run_generation(args)
    summary = report["summary"]
    try:
        from app.config import get_settings
        from app.core.fingerprint import build_response_fingerprint

        report["fingerprint"] = build_response_fingerprint(get_settings())
    except Exception as exc:  # noqa: BLE001 - fingerprint is best-effort in eval
        report["fingerprint"] = {"error": type(exc).__name__}
    printable = (
        report if args.json else {**summary, "fingerprint": report["fingerprint"]}
    )
    print(json.dumps(printable, ensure_ascii=False, indent=2))

    problems: List[str] = []
    if args.kind == "generation":
        problems += [
            f"case failed: {r['id']}" for r in report["rows"] if not r["passed"]
        ]
    else:
        problems += [
            f"no-answer violated: {r['id']}"
            for r in report["rows"]
            if r["no_answer_violation"]
        ]

    if args.baseline:
        bpath = Path(args.baseline)
        if args.update_baseline:
            bpath.parent.mkdir(parents=True, exist_ok=True)
            bpath.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"baseline written: {bpath}")
        elif bpath.exists():
            baseline = json.loads(bpath.read_text(encoding="utf-8"))
            problems += (
                rm.compare_to_baseline(summary, baseline, tolerance=args.tolerance)
                if args.kind == "retrieval"
                else compare_generation(summary, baseline, args.tolerance)
            )
        else:
            problems.append(f"baseline missing: {bpath}")

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
