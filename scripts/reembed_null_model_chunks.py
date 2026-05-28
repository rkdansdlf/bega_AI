#!/usr/bin/env python3
"""
rag_chunks 임베딩 스크립트 [v2 async-native]

주요 기능:
  1. NULL 청크 재임베딩: embedding_model IS NULL 인 청크를 지정 모델로 임베딩
  2. 월말 업그레이드:  --model-filter local 로 local:* 청크를 openrouter로 업그레이드
  3. 날짜 필터:        --max-game-date / --min-game-date 로 game_date 기준 분할 임베딩
  4. 시즌 필터:        --season-year 로 특정 시즌만 처리

성능 특성:
  - 단일 이벤트 루프에서 httpx 클라이언트를 재사용 → 배치마다 TLS 핸드셰이크 없음
  - UNNEST 배치 UPDATE → DB 왕복 1회/배치
  - 처리량: ~90 청크/s

사용 예:
  # NULL 청크 전체 재임베딩 (openrouter, 기본값)
  BEGA_SKIP_APP_INIT=1 PYTHONPATH=. .venv/bin/python scripts/reembed_null_model_chunks.py \\
      --batch-size 256 --commit-interval 2000 --max-concurrency 8

  # 2026년 3~4월 + non-game 청크만 openrouter 임베딩
  BEGA_SKIP_APP_INIT=1 PYTHONPATH=. .venv/bin/python scripts/reembed_null_model_chunks.py \\
      --season-year 2026 --max-game-date 2026-04-30 \\
      --batch-size 256 --commit-interval 2000 --max-concurrency 8

  # 2026년 5월 game 청크만 local 임베딩
  BEGA_SKIP_APP_INIT=1 EMBED_PROVIDER=local PYTHONPATH=. .venv/bin/python scripts/reembed_null_model_chunks.py \\
      --season-year 2026 --min-game-date 2026-05-01 \\
      --batch-size 256 --commit-interval 2000 --max-concurrency 8

  # 월말: 5월 local 청크를 openrouter로 업그레이드
  BEGA_SKIP_APP_INIT=1 PYTHONPATH=. .venv/bin/python scripts/reembed_null_model_chunks.py \\
      --model-filter local --season-year 2026 --max-game-date 2026-05-31 \\
      --batch-size 256 --commit-interval 2000 --max-concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("BEGA_SKIP_APP_INIT", "1")

try:
    import psycopg
except ModuleNotFoundError as exc:
    print(f"ERROR: psycopg not found: {exc}", file=sys.stderr)
    sys.exit(1)

from app.config import get_settings
from app.core.embeddings import async_embed_texts

# ── 재시도 래퍼 (async) ──────────────────────────────────────────────────────


async def _embed_with_retry(
    texts: List[str],
    settings,
    max_concurrency: int,
    max_attempts: int = 3,
) -> List[List[float]]:
    """async_embed_texts 를 재시도 래퍼로 감싼다."""
    for attempt in range(max_attempts):
        try:
            return await async_embed_texts(
                texts, settings, max_concurrency=max_concurrency
            )
        except Exception as exc:
            if attempt == max_attempts - 1:
                raise
            wait = 2**attempt + random.uniform(0, 1)
            print(
                f"  [WARN] embed failed (attempt {attempt + 1}/{max_attempts}), "
                f"retrying in {wait:.1f}s: {exc}",
                flush=True,
            )
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


# ── 메인 로직 (async) ────────────────────────────────────────────────────────


def _build_where_clause(
    model_filter: str,
    source_table_filter: Optional[str],
    season_year: Optional[int],
    max_game_date: Optional[str],
    min_game_date: Optional[str],
) -> str:
    """
    필터 인자들로부터 WHERE 절 문자열을 조합한다 (id > %s 와 LIMIT %s 는 호출측에서 추가).

    model_filter:
      'null'  → embedding_model IS NULL
      'local' → embedding_model LIKE 'local:%'

    max_game_date: game_date 없는 청크(season stats 등)는 포함, game_date 있는 청크는 ≤ 날짜만 포함.
    min_game_date: game_date 없는 청크는 제외, game_date 있는 청크는 ≥ 날짜만 포함.
    """
    parts: List[str] = []

    # ── embedding_model 필터 ────────────────────────────────────────────────
    if model_filter == "local":
        parts.append("embedding_model LIKE 'local:%'")
    else:
        parts.append("embedding_model IS NULL")

    # ── source_table 필터 ───────────────────────────────────────────────────
    if source_table_filter:
        parts.append(f"source_table = '{source_table_filter}'")

    # ── season_year 필터 ────────────────────────────────────────────────────
    if season_year is not None:
        parts.append(f"season_year = {season_year}")

    # ── game_date 상한 (non-game 청크는 포함) ────────────────────────────────
    if max_game_date:
        parts.append(
            f"(meta->>'game_date' IS NULL"
            f" OR (meta->>'game_date')::date <= '{max_game_date}')"
        )

    # ── game_date 하한 (non-game 청크는 제외) ────────────────────────────────
    if min_game_date:
        parts.append(f"(meta->>'game_date')::date >= '{min_game_date}'")

    return "WHERE " + "\n  AND ".join(parts)


async def reembed_null_model_chunks(
    *,
    dest_db_url: str,
    batch_size: int = 256,
    commit_interval: int = 2000,
    max_concurrency: int = 8,
    source_table_filter: Optional[str] = None,
    season_year: Optional[int] = None,
    max_game_date: Optional[str] = None,
    min_game_date: Optional[str] = None,
    model_filter: str = "null",
    dry_run: bool = False,
) -> int:
    """
    rag_chunks 중 지정 필터에 해당하는 청크를 임베딩 후 UPDATE.

    model_filter='null'  → embedding_model IS NULL 인 청크 (기본)
    model_filter='local' → embedding_model LIKE 'local:%' 인 청크 (월말 업그레이드용)

    max_game_date: 'YYYY-MM-DD' — game_date ≤ 날짜인 청크만. non-game 청크는 항상 포함.
    min_game_date: 'YYYY-MM-DD' — game_date ≥ 날짜인 청크만. non-game 청크는 제외.
    season_year:   특정 시즌만 처리.

    Returns: 총 처리된 청크 수
    """
    settings = get_settings()
    target_model = (
        f"{settings.embed_provider}:"
        f"{settings.openrouter_embed_model or settings.embed_model}"
    )
    print(f"Target embedding model: {target_model}", flush=True)

    where = _build_where_clause(
        model_filter=model_filter,
        source_table_filter=source_table_filter,
        season_year=season_year,
        max_game_date=max_game_date,
        min_game_date=min_game_date,
    )

    # 처리할 총 건수 확인
    with psycopg.connect(dest_db_url, connect_timeout=30) as count_conn:
        with count_conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM rag_chunks {where}")
            total = cur.fetchone()[0]

    label = "local-model" if model_filter == "local" else "NULL-model"
    print(f"{label} chunks to process: {total:,}", flush=True)
    if total == 0:
        print("Nothing to do.", flush=True)
        return 0

    processed = 0
    committed = 0
    since_commit = 0
    errors = 0
    start_time = time.time()

    with psycopg.connect(dest_db_url, connect_timeout=30) as conn:
        conn.autocommit = False

        # id 기반 페이지네이션
        last_id = 0
        while True:
            # ── 다음 배치 조회 ────────────────────────────────────────────────
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, content
                    FROM rag_chunks
                    {where}
                      AND id > %s
                    ORDER BY id
                    LIMIT %s
                    """,
                    (last_id, batch_size),
                )
                rows: List[Tuple[int, str]] = cur.fetchall()

            if not rows:
                break

            ids = [r[0] for r in rows]
            texts = [r[1] or "" for r in rows]
            last_id = ids[-1]

            # 빈 내용 필터
            non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
            if not non_empty_indices:
                processed += len(rows)
                continue

            embed_texts_list = [texts[i] for i in non_empty_indices]
            embed_ids = [ids[i] for i in non_empty_indices]

            if dry_run:
                # dry-run: DB 쓰기 없음
                processed += len(rows)
                since_commit += len(embed_ids)
                if since_commit >= commit_interval:
                    print(
                        f"  [DRY-RUN] [{processed:,}/{total:,}] "
                        f"would_commit={since_commit} (no actual DB write)",
                        flush=True,
                    )
                    since_commit = 0
                continue

            # ── 임베딩 (단일 루프 내 httpx 클라이언트 재사용) ─────────────────
            t0 = time.time()
            try:
                vectors = await _embed_with_retry(
                    embed_texts_list, settings, max_concurrency=max_concurrency
                )
            except Exception as exc:
                print(
                    f"  [ERROR] embed failed for batch ending at id={last_id}: {exc}",
                    flush=True,
                )
                errors += len(embed_texts_list)
                processed += len(rows)
                continue
            t_embed = time.time() - t0

            embed_dim = len(vectors[0]) if vectors else 1536

            # ── UNNEST 배치 UPDATE (한 번의 쿼리로 N개 행 업데이트) ────────────
            t1 = time.time()
            vector_strs = [
                "[" + ",".join(f"{v:.8f}" for v in vec) + "]" for vec in vectors
            ]
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE rag_chunks AS r
                    SET embedding       = d.emb::vector,
                        embedding_model = %s,
                        embedding_dim   = %s,
                        updated_at      = NOW()
                    FROM (
                        SELECT unnest(%s::int[]) AS id,
                               unnest(%s::text[]) AS emb
                    ) AS d
                    WHERE r.id = d.id
                      AND r.embedding_model IS NULL
                    """,
                    (target_model, embed_dim, embed_ids, vector_strs),
                )
            t_db = time.time() - t1

            processed += len(rows)
            since_commit += len(embed_ids)

            if since_commit >= commit_interval:
                conn.commit()
                committed += since_commit
                since_commit = 0
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta_sec = (total - processed) / rate if rate > 0 else 0
                print(
                    f"  [{processed:,}/{total:,}] committed={committed:,} "
                    f"errors={errors} rate={rate:.1f}/s "
                    f"ETA={eta_sec/60:.1f}min "
                    f"[embed={t_embed:.1f}s db={t_db:.2f}s]",
                    flush=True,
                )

        # 마지막 커밋
        if since_commit > 0:
            conn.commit()
            committed += since_commit

    elapsed = time.time() - start_time
    print(
        f"\nDone: processed={processed:,} committed={committed:,} errors={errors} "
        f"elapsed={elapsed/60:.1f}min",
        flush=True,
    )
    return processed


async def _main_async() -> int:
    parser = argparse.ArgumentParser(
        description="rag_chunks 를 임베딩 후 UPDATE [v2 async] — NULL 청크 재임베딩 및 월별 local→openrouter 업그레이드 지원"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="임베딩 배치 크기 (기본: 256)",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=2000,
        help="커밋 간격 — 청크 수 (기본: 2000)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="임베딩 API 동시 요청 수 (기본: 8)",
    )
    parser.add_argument(
        "--source-table",
        default="",
        help="특정 source_table 만 처리 (기본: 전체)",
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=None,
        help="특정 시즌만 처리 (예: 2026). 미지정 시 전체.",
    )
    parser.add_argument(
        "--max-game-date",
        default="",
        help="game_date 상한 YYYY-MM-DD (예: 2026-04-30). "
        "game_date 없는 청크(season stats 등)는 이 필터와 무관하게 포함.",
    )
    parser.add_argument(
        "--min-game-date",
        default="",
        help="game_date 하한 YYYY-MM-DD (예: 2026-05-01). "
        "game_date 없는 청크는 이 필터로 제외됨.",
    )
    parser.add_argument(
        "--model-filter",
        choices=("null", "local"),
        default="null",
        help="처리할 청크 필터: null=embedding_model IS NULL (기본), "
        "local=embedding_model LIKE 'local:%%' (월말 업그레이드용)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="임베딩 API 호출 없이 시뮬레이션 (DB 쓰기 없음)",
    )
    parser.add_argument(
        "--db-url",
        default="",
        help="대상 DB URL (기본: POSTGRES_DB_URL 환경변수)",
    )
    args = parser.parse_args()

    # ── 날짜 형식 검증 ───────────────────────────────────────────────────────
    from datetime import datetime as _dt

    for flag, val in [
        ("--max-game-date", args.max_game_date),
        ("--min-game-date", args.min_game_date),
    ]:
        if val:
            try:
                _dt.strptime(val, "%Y-%m-%d")
            except ValueError:
                print(
                    f"ERROR: {flag} 는 YYYY-MM-DD 형식이어야 합니다: {val!r}",
                    file=sys.stderr,
                )
                return 1

    settings = get_settings()
    db_url = (
        args.db_url.strip()
        or os.environ.get("POSTGRES_DB_URL", "")
        or settings.database_url
    )
    if not db_url:
        print(
            "ERROR: DB URL not configured (set POSTGRES_DB_URL or --db-url)",
            file=sys.stderr,
        )
        return 1

    print(f"DB: {db_url.split('@')[-1]}", flush=True)
    print(
        f"batch_size={args.batch_size} commit_interval={args.commit_interval} "
        f"max_concurrency={args.max_concurrency} model_filter={args.model_filter} "
        f"season_year={args.season_year} max_game_date={args.max_game_date or '-'} "
        f"min_game_date={args.min_game_date or '-'} dry_run={args.dry_run}",
        flush=True,
    )

    await reembed_null_model_chunks(
        dest_db_url=db_url,
        batch_size=args.batch_size,
        commit_interval=args.commit_interval,
        max_concurrency=args.max_concurrency,
        source_table_filter=args.source_table or None,
        season_year=args.season_year,
        max_game_date=args.max_game_date or None,
        min_game_date=args.min_game_date or None,
        model_filter=args.model_filter,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main_async()))
