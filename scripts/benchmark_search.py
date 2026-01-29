"""
Supabase pgvector vs Firestore Vector Search 성능 벤치마크

사용법:
    python scripts/benchmark_search.py
"""

import os
import sys
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

import psycopg
from app.core.embeddings import embed_texts
from app.core.retrieval import similarity_search as supabase_search
# from app.core.retrieval_firestore import similarity_search_firestore
from app.config import Settings

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


def benchmark_supabase(
    embeddings: List[List[float]], limit: int = 10
) -> Dict[str, Any]:
    """Supabase pgvector 성능 측정"""
    print("\n🔵 Supabase pgvector 벤치마크")
    print("=" * 60)

    supabase_url = os.getenv("OCI_DB_URL")
    if not supabase_url:
        print("❌ OCI_DB_URL 환경변수가 설정되지 않았습니다.")
        return {}

    conn = psycopg.connect(supabase_url)

    times = []
    results_count = []

    for i, embedding in enumerate(embeddings, 1):
        start = time.time()
        results = supabase_search(conn, embedding, limit=limit)
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


# def benchmark_firestore(
#     embeddings: List[List[float]], limit: int = 10
# ) -> Dict[str, Any]:
#     """Firestore Vector Search 성능 측정"""
#     print("\n🟠 Firestore Vector Search 벤치마크")
#     print("=" * 60)
#
#     # Firebase 초기화 (환경 변수 설정)
#     os.environ["USE_FIRESTORE_SEARCH"] = "true"
#     os.environ["FIREBASE_SERVICE_ACCOUNT_KEY"] = str(
#         project_root / "bega-186a7-firebase-adminsdk-fbsvc-bb50c006a7.json"
#     )
#     os.environ["FIRESTORE_DATABASE_ID"] = "begachatbot"
#
#     times = []
#     results_count = []
#
#     for i, embedding in enumerate(embeddings, 1):
#         start = time.time()
#         # results = similarity_search_firestore(embedding, limit=limit)
#         # elapsed = time.time() - start
#
#         # times.append(elapsed)
#         # results_count.append(len(results))
#
#         # print(f"  쿼리 {i:2d}: {elapsed * 1000:6.1f}ms | {len(results):2d}개 결과")
#         pass
#
#     return {
#         "평균 시간": 0, # statistics.mean(times) * 1000,
#         "중앙값": 0, # statistics.median(times) * 1000,
#         "최소 시간": 0, # min(times) * 1000,
#         "최대 시간": 0, # max(times) * 1000,
#         "표준편차": 0, # statistics.stdev(times) * 1000 if len(times) > 1 else 0,
#         "평균 결과 수": 0, # statistics.mean(results_count),
#     }


def main():
    """벤치마크 실행"""
    print("\n" + "=" * 60)
    print("Supabase vs Firestore 성능 벤치마크")
    print("=" * 60)
    print(f"테스트 쿼리 수: {len(TEST_QUERIES)}")
    print(f"반환 결과 수: 10개")
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

    # Supabase 벤치마크
    supabase_stats = benchmark_supabase(embeddings, limit=10)

    # Firestore 벤치마크 (제거됨)
    firestore_stats = None

    # 결과 비교
    print("\n" + "=" * 60)
    print("📊 결과 비교")
    print("=" * 60)
    print(f"\n{'지표':<20} {'Supabase':>15} {'Firestore':>15} {'비율':>10}")
    print("-" * 65)

    if supabase_stats and firestore_stats:
        for key in ["평균 시간", "중앙값", "최소 시간", "최대 시간", "표준편차"]:
            supabase_val = supabase_stats[key]
            firestore_val = firestore_stats[key]
            ratio = firestore_val / supabase_val if supabase_val > 0 else 0

            print(
                f"{key:<20} {supabase_val:>12.1f}ms {firestore_val:>12.1f}ms {ratio:>9.2f}x"
            )

        print("-" * 65)
        print(
            f"{'평균 결과 수':<20} {supabase_stats['평균 결과 수']:>12.1f}개 {firestore_stats['평균 결과 수']:>12.1f}개"
        )

        print("\n" + "=" * 60)
        avg_ratio = firestore_stats["평균 시간"] / supabase_stats["평균 시간"]

        if avg_ratio < 0.8:
            winner = "Firestore"
            faster = (1 - avg_ratio) * 100
            print(f"🏆 승자: {winner} (약 {faster:.0f}% 빠름)")
        elif avg_ratio > 1.2:
            winner = "Supabase"
            faster = (avg_ratio - 1) * 100
            print(f"🏆 승자: {winner} (약 {faster:.0f}% 빠름)")
        else:
            print(f"🤝 비슷한 성능 (차이 {abs(avg_ratio - 1) * 100:.0f}%)")

    else:
        print("⚠️  벤치마크 데이터 부족")

    print("=" * 60 + "\n")

    # 추가 분석
    print("💡 분석 및 권장사항:")
    print("-" * 60)

    if supabase_stats and firestore_stats:
        supabase_avg = supabase_stats["평균 시간"]
        firestore_avg = firestore_stats["평균 시간"]

        print(f"• Supabase 평균 응답 시간: {supabase_avg:.1f}ms")
        print(f"• Firestore 평균 응답 시간: {firestore_avg:.1f}ms")

        if firestore_avg < 100:
            print("\n✅ Firestore 성능 우수 (100ms 이하)")
            print("   → Firestore 사용 권장")
        elif firestore_avg < supabase_avg:
            print(f"\n✅ Firestore가 {supabase_avg / firestore_avg:.1f}배 빠름")
            print("   → Firestore 사용 권장")
        else:
            print(f"\n⚠️  Supabase가 {firestore_avg / supabase_avg:.1f}배 빠름")
            print("   → 추가 최적화 필요:")
            print("     1. Firestore 벡터 인덱스 확인")
            print("     2. 네트워크 레이턴시 확인")
            print("     3. 인스턴스 위치 (리전) 확인")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
