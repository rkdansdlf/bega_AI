"""
PostgreSQL pgvector 성능 벤치마크

사용법:
    python scripts/benchmark_search.py
"""

import argparse
import os
import sys
import time
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

SearchFn = Callable[[Any, Sequence[float], int], List[Dict[str, Any]]]

# 테스트 쿼리 세트
TEST_QUERIES = [
    "2024년 KIA 타이거즈 홈런왕은 누구야?",
    "타율 1위 선수는?",
    "OPS가 뭐야?",
    "골든글러브 선정 기준은?",
    "두산 베어스의 2024년 성적은?",
    "2023년 MVP는 누구?",
    "삼진왕 투수",
    "포스트시즌 최다 홈런",
    "신인왕 후보는?",
    "역대 최고 타율 기록",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PostgreSQL pgvector benchmark.")
    parser.add_argument(
        "--source-db-url",
        default="",
        help="Source PostgreSQL 연결 URL (기본값: 환경변수 POSTGRES_DB_URL)",
    )
    parser.add_argument(
        "--supabase-url",
        default="",
        help="[Deprecated] --source-db-url 사용 권장",
    )
    return parser


def resolve_source_db_url(source_db_url: str, supabase_url: str) -> str:
    """환경 변수/인자를 우선순위에 따라 Source PostgreSQL URL로 변환한다."""
    if source_db_url:
        return source_db_url
    if supabase_url:
        print("[WARN] --supabase-url is deprecated. Use --source-db-url instead.")
        return supabase_url

    env_url = os.getenv("POSTGRES_DB_URL", "").strip()
    if env_url:
        return env_url

    legacy_url = os.getenv("SUPABASE_DB_URL", "").strip()
    if legacy_url:
        print("[WARN] SUPABASE_DB_URL is deprecated. Use POSTGRES_DB_URL instead.")
        return legacy_url

    return ""


def benchmark_postgres(
    embeddings: List[List[float]],
    postgres_url: str,
    search_fn: SearchFn,
    limit: int = 10,
) -> Dict[str, Any]:
    """PostgreSQL pgvector 성능 측정"""
    try:
        import psycopg
    except ModuleNotFoundError:
        print("❌ psycopg가 설치되어 있지 않아 벤치마크를 실행할 수 없습니다.")
        print("   pip install psycopg[binary] 후 재실행하세요.")
        return {}

    print("\n🔵 PostgreSQL pgvector 벤치마크")
    print("=" * 60)

    if not postgres_url:
        print("❌ POSTGRES_DB_URL 환경변수가 설정되지 않았습니다.")
        return {}

    conn = psycopg.connect(postgres_url)

    times = []
    results_count = []

    for i, embedding in enumerate(embeddings, 1):
        start = time.time()
        results = search_fn(conn, embedding, limit=limit)
        elapsed = time.time() - start

        times.append(elapsed)
        results_count.append(len(results))

        print(f"  쿼리 {i:2d}: {elapsed * 1000:6.1f}ms | {len(results):2d}개 결과")

    conn.close()

    return {
        "평균 시간": statistics.mean(times) * 1000,
        "중앙값": statistics.median(times) * 1000,
        "최소 시간": min(times) * 1000,
        "최대 시간": max(times) * 1000,
        "표준편차": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        "평균 결과 수": statistics.mean(results_count),
    }


def main():
    """벤치마크 실행"""
    parser = build_parser()
    args = parser.parse_args()
    source_db_url = resolve_source_db_url(
        source_db_url=args.source_db_url.strip(),
        supabase_url=args.supabase_url.strip(),
    )
    try:
        from app.config import Settings
        from app.core.embeddings import embed_texts
        from app.core.retrieval import similarity_search as postgres_search
    except ModuleNotFoundError as exc:
        print(f"❌ 벤치마크에 필요한 모듈이 없습니다: {exc.name}")
        print("   필요한 의존성을 설치한 뒤 재실행하세요.")
        return
    print("\n" + "=" * 60)
    print("PostgreSQL pgvector 성능 벤치마크")
    print("=" * 60)
    print(f"테스트 쿼리 수: {len(TEST_QUERIES)}")
    print(f"반환 결과 수: 10개")
    if source_db_url:
        print(f"연결 URL: {source_db_url}")
    print("=" * 60)

    # 임베딩 생성 (공통)
    print("\n임베딩 생성 중...")
    settings = Settings()
    start = time.time()
    embeddings = embed_texts(TEST_QUERIES, settings)
    embedding_time = time.time() - start

    if not embeddings or len(embeddings) != len(TEST_QUERIES):
        print("❌ 임베딩 생성 실패")
        return

    print(f"✓ 임베딩 생성 완료 ({embedding_time:.2f}초)")
    print(f"  - 차원: {len(embeddings[0])}")
    print(f"  - 개수: {len(embeddings)}")

    # PostgreSQL pgvector 벤치마크
    if not source_db_url:
        print("❌ POSTGRES_DB_URL 환경변수가 설정되지 않았습니다.")
        return

    def _search(
        connection: Any, embedding: Sequence[float], limit: int
    ) -> List[Dict[str, Any]]:
        return postgres_search(
            connection,
            embedding,
            limit=limit,
        )

    postgres_stats = benchmark_postgres(
        embeddings=embeddings,
        postgres_url=source_db_url,
        search_fn=_search,
        limit=10,
    )

    print("\n" + "=" * 60)
    print("📊 PostgreSQL pgvector 벤치마크 결과")
    print("=" * 60)

    if not postgres_stats:
        print("⚠️  벤치마크 데이터 부족")
        print("=" * 60 + "\n")
        return

    print(f"\n{'지표':<20} {'PostgreSQL pgvector':>18}")
    print("-" * 40)
    print(f"{'평균 시간':<20} {postgres_stats['평균 시간']:>16.1f}ms")
    print(f"{'중앙값':<20} {postgres_stats['중앙값']:>16.1f}ms")
    print(f"{'최소 시간':<20} {postgres_stats['최소 시간']:>16.1f}ms")
    print(f"{'최대 시간':<20} {postgres_stats['최대 시간']:>16.1f}ms")
    print(f"{'표준편차':<20} {postgres_stats['표준편차']:>16.1f}ms")
    print(f"{'평균 결과 수':<20} {postgres_stats['평균 결과 수']:>12.1f}개")

    print("=" * 60 + "\n")

    # 추가 분석
    print("💡 분석 및 권장사항:")
    print("-" * 60)

    print(f"\n• PostgreSQL 평균 응답 시간: {postgres_stats['평균 시간']:.1f}ms")
    print(f"• PostgreSQL 중앙값 응답 시간: {postgres_stats['중앙값']:.1f}ms")
    print("-" * 60)
    print("• PostgreSQL pgvector 기준 검색 성능으로 동작하도록 구성된 벤치마크입니다.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
