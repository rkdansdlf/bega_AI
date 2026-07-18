"""HNSW 벡터 인덱스 생성/마이그레이션 스크립트.

이 스크립트는 rag_chunks 테이블에 HNSW 인덱스를 생성하고,
선택적으로 구형 IVFFlat 인덱스를 제거합니다.

주요 기능:
  - pgvector 버전 확인 (HNSW는 pgvector 0.5.0+)
  - CREATE INDEX CONCURRENTLY 사용 (운영 중 테이블 잠금 없음)
  - --dry-run: 실제 변경 없이 상태만 출력
  - --drop-ivfflat: HNSW 생성 후 구형 IVFFlat 인덱스 제거
  - --drop-vector-hnsw: halfvec 전환 후 중복 vector HNSW 인덱스 제거
  - EXPLAIN ANALYZE로 인덱스 사용 여부 검증

운영 전환 흐름:
  1. 이 스크립트를 --dry-run으로 실행하여 상태 확인
  2. 스크립트를 --drop-ivfflat 없이 실행하여 HNSW 생성
  3. AI 서비스에 AI_VECTOR_INDEX=hnsw 배포
     (halfvec 인덱스 생성 시 AI_VECTOR_QUANTIZATION=halfvec도 함께 배포)
  4. scripts/audit_embedding_256_migration.py 로 검색 SQL이 halfvec(256)인지 확인
  5. 정상 확인 후 --drop-ivfflat 옵션으로 IVFFlat 제거
  6. halfvec 전환 완료 후 --dry-run --drop-vector-hnsw 확인,
     이후 --drop-vector-hnsw 로 중복 vector HNSW 제거

사용 예:
  python scripts/create_vector_index.py --dry-run
  python scripts/create_vector_index.py
  python scripts/create_vector_index.py --drop-ivfflat
  python scripts/create_vector_index.py --dry-run --drop-vector-hnsw
  python scripts/create_vector_index.py --drop-vector-hnsw
  python scripts/create_vector_index.py --m 16 --ef-construction 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 기본값
# ---------------------------------------------------------------------------
HNSW_INDEX_NAME = "idx_rag_chunks_embedding_hnsw"
HALFVEC_HNSW_INDEX_NAME = "idx_rag_chunks_embedding_halfvec_hnsw"
IVFFLAT_INDEX_NAME = "idx_rag_chunks_embedding"
TABLE_NAME = "rag_chunks"

_PGVECTOR_MIN_VERSION_HNSW = (0, 5, 0)  # HNSW가 도입된 버전
_PGVECTOR_MIN_VERSION_HALFVEC = (0, 7, 0)  # halfvec 타입/연산자 지원 버전


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _parse_version(version_str: str) -> Tuple[int, ...]:
    """'0.7.4' → (0, 7, 4)."""
    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def _version_at_least(actual: Tuple[int, ...], minimum: Tuple[int, ...]) -> bool:
    max_len = max(len(actual), len(minimum))
    padded_actual = actual + (0,) * (max_len - len(actual))
    padded_minimum = minimum + (0,) * (max_len - len(minimum))
    return padded_actual >= padded_minimum


def _required_pgvector_version(quantization: str = "none") -> Tuple[int, ...]:
    if (quantization or "none").lower().strip() == "halfvec":
        return _PGVECTOR_MIN_VERSION_HALFVEC
    return _PGVECTOR_MIN_VERSION_HNSW


def _build_hnsw_create_sql(
    *,
    quantization: str = "none",
    embed_dim: int,
    m: int,
    ef_construction: int,
) -> str:
    if (quantization or "none").lower().strip() == "halfvec":
        return (
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {HALFVEC_HNSW_INDEX_NAME} "
            f"ON {TABLE_NAME} "
            f"USING hnsw ((embedding::halfvec({embed_dim})) halfvec_cosine_ops) "
            f"WITH (m = {m}, ef_construction = {ef_construction}) "
            "WHERE embedding IS NOT NULL"
        )
    return (
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {HNSW_INDEX_NAME} "
        f"ON {TABLE_NAME} "
        f"USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = {m}, ef_construction = {ef_construction}) "
        "WHERE embedding IS NOT NULL"
    )


def _build_vector_hnsw_drop_sql() -> str:
    return f"DROP INDEX CONCURRENTLY IF EXISTS {HNSW_INDEX_NAME}"


def _vector_hnsw_drop_error(
    *,
    quantization: str,
    embed_dim: int,
    distance_sql: str,
    halfvec_index_exists: bool,
) -> Optional[str]:
    if (quantization or "none").lower().strip() != "halfvec":
        return "AI_VECTOR_QUANTIZATION=halfvec is required to drop vector HNSW."
    if int(embed_dim) != 256:
        return "EMBED_DIM=256 is required to drop vector HNSW."
    if not halfvec_index_exists:
        return f"Required halfvec HNSW index is missing: {HALFVEC_HNSW_INDEX_NAME}"
    if "halfvec(256)" not in distance_sql:
        return "Retrieval distance SQL is not using halfvec(256)."
    return None


def _get_pgvector_version(cur: psycopg.Cursor) -> Optional[str]:
    try:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _index_exists(cur: psycopg.Cursor, index_name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE tablename = %s AND indexname = %s",
        (TABLE_NAME, index_name),
    )
    return cur.fetchone() is not None


def _get_table_row_count(cur: psycopg.Cursor) -> int:
    """EXPLAIN을 통한 빠른 행 수 추정 (reltuples)."""
    cur.execute(
        "SELECT reltuples::bigint FROM pg_class WHERE relname = %s", (TABLE_NAME,)
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


def _print_current_indexes(cur: psycopg.Cursor) -> None:
    cur.execute(
        """
        SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) AS size, indexdef
        FROM pg_indexes
        WHERE tablename = %s AND indexname LIKE '%%embedding%%'
        ORDER BY indexname
        """,
        (TABLE_NAME,),
    )
    rows = cur.fetchall()
    if rows:
        print(f"\n현재 {TABLE_NAME} embedding 인덱스:")
        for name, size, definition in rows:
            print(f"  [{size}] {name}")
            print(f"         {definition}")
    else:
        print(f"\n{TABLE_NAME}에 embedding 인덱스가 없습니다.")


def _verify_hnsw_usage(
    cur: psycopg.Cursor,
    embed_dim: int,
    ef_search: int,
    *,
    quantization: str = "none",
) -> bool:
    """EXPLAIN ANALYZE로 HNSW 인덱스 사용 여부를 검증합니다."""
    vector_str = "[" + ",".join("0.0" for _ in range(embed_dim)) + "]"
    distance_expr = "embedding <=> %s::vector"
    if (quantization or "none").lower().strip() == "halfvec":
        distance_expr = f"embedding::halfvec({embed_dim}) <=> %s::halfvec({embed_dim})"
    cur.execute(f"SET hnsw.ef_search = {ef_search}")
    cur.execute(
        "EXPLAIN (FORMAT TEXT) "
        "SELECT id FROM rag_chunks "
        "WHERE embedding IS NOT NULL "
        f"ORDER BY {distance_expr} LIMIT 10",
        (vector_str,),
    )
    plan_rows = cur.fetchall()
    plan = "\n".join(row[0] for row in plan_rows)
    uses_hnsw = "Index Scan" in plan and (
        HNSW_INDEX_NAME in plan or HALFVEC_HNSW_INDEX_NAME in plan
    )
    print("\nEXPLAIN (처음 5줄):")
    for line in plan.splitlines()[:5]:
        print(f"  {line}")
    return uses_hnsw


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def run(
    *,
    dry_run: bool,
    drop_ivfflat: bool,
    drop_vector_hnsw: bool,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> int:
    """
    Returns:
        0 = 성공, 1 = 오류.
    """
    from app.config import get_settings

    settings = get_settings()
    dsn = settings.database_url
    embed_dim: int = getattr(settings, "embed_dim", 256)
    quantization = getattr(settings, "ai_vector_quantization", "none")
    from app.core.retrieval import _embedding_distance_sql

    distance_sql = _embedding_distance_sql(settings)
    hnsw_index_name = (
        HALFVEC_HNSW_INDEX_NAME
        if str(quantization).lower().strip() == "halfvec"
        else HNSW_INDEX_NAME
    )

    print(f"데이터베이스 연결 중...")
    try:
        conn = psycopg.connect(dsn, connect_timeout=30, autocommit=True)
    except Exception as exc:
        print(f"[ERROR] DB 연결 실패: {exc}")
        return 1

    try:
        with conn.cursor() as cur:
            cur.execute("SET search_path TO public, extensions")

            # 1. pgvector 버전 확인
            pgvec_ver_str = _get_pgvector_version(cur)
            if pgvec_ver_str is None:
                print("[ERROR] pgvector 확장이 설치되어 있지 않습니다.")
                return 1
            pgvec_ver = _parse_version(pgvec_ver_str)
            required_pgvector = _required_pgvector_version(quantization)
            if not _version_at_least(pgvec_ver, required_pgvector):
                print(
                    f"[ERROR] pgvector {pgvec_ver_str}는 HNSW를 지원하지 않습니다. "
                    f"pgvector >= {'.'.join(str(v) for v in required_pgvector)} 필요."
                )
                return 1
            print(f"pgvector 버전: {pgvec_ver_str} ✓")

            # 2. 행 수 추정
            row_count = _get_table_row_count(cur)
            print(f"{TABLE_NAME} 추정 행 수: {row_count:,}")

            # 3. 현재 인덱스 상태 출력
            _print_current_indexes(cur)

            hnsw_exists = _index_exists(cur, hnsw_index_name)
            halfvec_hnsw_exists = _index_exists(cur, HALFVEC_HNSW_INDEX_NAME)
            vector_hnsw_exists = _index_exists(cur, HNSW_INDEX_NAME)
            ivfflat_exists = _index_exists(cur, IVFFLAT_INDEX_NAME)

            # 4. HNSW 인덱스 생성
            if hnsw_exists:
                print(f"\n[SKIP] HNSW 인덱스 '{hnsw_index_name}'가 이미 존재합니다.")
            else:
                hnsw_sql = _build_hnsw_create_sql(
                    quantization=quantization,
                    embed_dim=embed_dim,
                    m=m,
                    ef_construction=ef_construction,
                )
                if dry_run:
                    print(f"\n[DRY-RUN] 실행 예정 SQL:\n  {hnsw_sql}")
                else:
                    print(
                        f"\nHNSW 인덱스 생성 중 (m={m}, ef_construction={ef_construction})..."
                    )
                    print(
                        f"행 수 {row_count:,}개 기준 예상 소요 시간: "
                        f"{max(1, row_count // 50_000)}~{max(2, row_count // 20_000)}분"
                    )
                    print("(CREATE INDEX CONCURRENTLY: 운영 트래픽에 영향 없음)")
                    start = time.time()
                    cur.execute(hnsw_sql)
                    elapsed = time.time() - start
                    print(f"HNSW 인덱스 생성 완료! ({elapsed:.1f}초)")
                    hnsw_exists = True
                    if hnsw_index_name == HALFVEC_HNSW_INDEX_NAME:
                        halfvec_hnsw_exists = True

            # 5. IVFFlat 제거 (선택)
            if drop_ivfflat and ivfflat_exists:
                drop_sql = f"DROP INDEX CONCURRENTLY IF EXISTS {IVFFLAT_INDEX_NAME}"
                if dry_run:
                    print(f"\n[DRY-RUN] IVFFlat 제거 예정 SQL:\n  {drop_sql}")
                else:
                    if not hnsw_exists:
                        print(
                            "[WARNING] HNSW 인덱스가 없는 상태에서 IVFFlat을 제거하면 "
                            "검색 성능이 크게 저하됩니다. 작업을 건너뜁니다."
                        )
                    else:
                        print(f"\nIVFFlat 인덱스 '{IVFFLAT_INDEX_NAME}' 제거 중...")
                        cur.execute(drop_sql)
                        print("IVFFlat 인덱스 제거 완료!")
                        ivfflat_exists = False
            elif drop_ivfflat and not ivfflat_exists:
                print(
                    f"\n[SKIP] IVFFlat 인덱스 '{IVFFLAT_INDEX_NAME}'가 이미 없습니다."
                )

            if drop_vector_hnsw:
                drop_error = _vector_hnsw_drop_error(
                    quantization=quantization,
                    embed_dim=embed_dim,
                    distance_sql=distance_sql,
                    halfvec_index_exists=halfvec_hnsw_exists,
                )
                if drop_error:
                    print(f"\n[ERROR] vector HNSW 제거 조건 불충족: {drop_error}")
                    return 1
                drop_sql = _build_vector_hnsw_drop_sql()
                if not vector_hnsw_exists:
                    print(
                        f"\n[SKIP] vector HNSW 인덱스 '{HNSW_INDEX_NAME}'가 이미 없습니다."
                    )
                elif dry_run:
                    print(f"\n[DRY-RUN] vector HNSW 제거 예정 SQL:\n  {drop_sql}")
                else:
                    print(f"\nvector HNSW 인덱스 '{HNSW_INDEX_NAME}' 제거 중...")
                    cur.execute(drop_sql)
                    print("vector HNSW 인덱스 제거 완료!")
                    vector_hnsw_exists = False

            # 6. 최종 인덱스 상태
            print("\n--- 최종 인덱스 상태 ---")
            _print_current_indexes(cur)

            # 7. HNSW 사용 검증
            if hnsw_exists and not dry_run:
                print("\nHNSW 인덱스 사용 여부 검증 중...")
                uses_hnsw = _verify_hnsw_usage(
                    cur,
                    embed_dim,
                    ef_search,
                    quantization=quantization,
                )
                if uses_hnsw:
                    print("✓ HNSW 인덱스가 쿼리 플랜에서 사용됩니다.")
                else:
                    print(
                        "[WARNING] EXPLAIN 플랜에서 HNSW 인덱스 사용이 확인되지 않았습니다. "
                        "행 수가 너무 적거나 플래너 비용 추정 문제일 수 있습니다. "
                        "`SET enable_seqscan = off`로 강제 검증을 시도할 수 있습니다."
                    )

            # 8. 환경변수 안내
            if not dry_run:
                quantization_step = ""
                if str(quantization).lower().strip() == "halfvec":
                    quantization_step = "     AI_VECTOR_QUANTIZATION=halfvec\n"
                print(
                    "\n--- 다음 단계 ---\n"
                    "1. AI 서비스 환경변수 설정:\n"
                    "     AI_VECTOR_INDEX=hnsw\n"
                    f"{quantization_step}"
                    "2. 서비스 재시작 또는 재배포\n"
                    "3. 256-d 감사로 런타임/DB/검색 SQL 확인:\n"
                    "     python scripts/audit_embedding_256_migration.py\n"
                    "4. 기존 IVFFlat 제거 (아직 제거하지 않은 경우):\n"
                    f"     python scripts/create_vector_index.py --drop-ivfflat\n"
                    "5. halfvec 정상화 후 중복 vector HNSW 제거:\n"
                    "     python scripts/create_vector_index.py --dry-run --drop-vector-hnsw\n"
                    "     python scripts/create_vector_index.py --drop-vector-hnsw\n"
                    "6. ef_search 튜닝 (기본 100, 정확도↑ = 값↑, 속도↑ = 값↓):\n"
                    f"     RETRIEVAL_HNSW_EF_SEARCH={ef_search}  # 현재 기본값"
                )
    finally:
        conn.close()

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="rag_chunks에 HNSW 벡터 인덱스를 생성/마이그레이션합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 변경 없이 실행 예정 작업만 출력합니다.",
    )
    parser.add_argument(
        "--drop-ivfflat",
        action="store_true",
        help="HNSW 생성 후 구형 IVFFlat 인덱스(idx_rag_chunks_embedding)를 제거합니다.",
    )
    parser.add_argument(
        "--drop-vector-hnsw",
        action="store_true",
        help=(
            "AI_VECTOR_QUANTIZATION=halfvec 전환 후 중복 vector HNSW "
            "인덱스(idx_rag_chunks_embedding_hnsw)를 제거합니다."
        ),
    )
    parser.add_argument(
        "--m",
        type=int,
        default=16,
        help="HNSW 레이어당 최대 연결 수 (기본: 16, 정확도↑=값↑, 빌드 시간↑)",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=64,
        help="HNSW 빌드 시 동적 후보 리스트 크기 (기본: 64, 정확도↑=값↑, 빌드 시간↑)",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=100,
        help="검색 시 후보 리스트 크기 (기본: 100, RETRIEVAL_HNSW_EF_SEARCH 환경변수와 동일)",
    )
    args = parser.parse_args()

    sys.exit(
        run(
            dry_run=args.dry_run,
            drop_ivfflat=args.drop_ivfflat,
            drop_vector_hnsw=args.drop_vector_hnsw,
            m=args.m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )
    )


if __name__ == "__main__":
    main()
