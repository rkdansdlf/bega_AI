"""
Firestore 벡터 검색 디버깅 스크립트

검색 결과가 0건인 원인을 분석합니다:
1. 실제 데이터 샘플 확인
2. 샘플 데이터의 임베딩으로 직접 검색
3. 유사도 임계값 확인
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.retrieval_firestore import similarity_search_firestore, _init_firebase
from app.core.embeddings import embed_texts
from app.config import Settings


def main():
    print("\n" + "="*80)
    print("Firestore 벡터 검색 디버깅")
    print("="*80 + "\n")

    # Firebase 초기화
    db = _init_firebase()
    settings = Settings()

    # 1. 실제 데이터 샘플 확인
    print("[1단계] 실제 데이터 샘플 확인")
    print("-" * 80)

    docs = db.collection('rag_chunks').limit(5).get()

    print(f"샘플 문서 수: {len(docs)}\n")

    sample_embeddings = []
    sample_titles = []

    for i, doc in enumerate(docs, 1):
        data = doc.to_dict()
        title = data.get('title', 'N/A')
        content = data.get('content', 'N/A')[:100]
        embedding = data.get('embedding')

        print(f"문서 {i}:")
        print(f"  ID: {doc.id}")
        print(f"  제목: {title}")
        print(f"  내용: {content}...")
        print(f"  임베딩 차원: {len(embedding) if embedding else 0}")
        print(f"  seasonYear: {data.get('seasonYear', 'N/A')}")
        print(f"  teamId: {data.get('teamId', 'N/A')}")
        print(f"  sourceTable: {data.get('sourceTable', 'N/A')}")
        print()

        if embedding:
            sample_embeddings.append(embedding)
            sample_titles.append(title)

    if not sample_embeddings:
        print("❌ 임베딩이 있는 문서를 찾을 수 없습니다!")
        return

    # 2. 샘플 데이터의 임베딩으로 직접 검색
    print("[2단계] 샘플 임베딩으로 직접 검색 (자기 자신 찾기)")
    print("-" * 80)

    for i, (embedding, title) in enumerate(zip(sample_embeddings[:2], sample_titles[:2]), 1):
        print(f"\n테스트 {i}: '{title}' 의 임베딩으로 검색")

        results = similarity_search_firestore(
            embedding=embedding,
            limit=5
        )

        print(f"  결과 수: {len(results)}")

        if results:
            for j, result in enumerate(results, 1):
                print(f"    {j}. {result.get('title', 'N/A')}")
                print(f"       유사도: {result.get('similarity', 0):.4f}")
        else:
            print("  ⚠️ 결과 없음 - 자기 자신도 찾지 못함!")

    # 3. 새로운 쿼리로 검색
    print("\n[3단계] 새로운 쿼리로 검색")
    print("-" * 80)

    test_queries = [
        "타율",
        "선수",
        "baseball",
        "KBO",
    ]

    for query in test_queries:
        print(f"\n쿼리: '{query}'")

        # 임베딩 생성
        embedding = embed_texts([query], settings)[0]

        # 검색
        results = similarity_search_firestore(
            embedding=embedding,
            limit=3
        )

        print(f"  결과 수: {len(results)}")

        if results:
            for i, result in enumerate(results, 1):
                print(f"    {i}. {result.get('title', 'N/A')}")
                print(f"       유사도: {result.get('similarity', 0):.4f}")
        else:
            print("  ⚠️ 결과 없음")

    # 4. PostgreSQL과 비교
    print("\n[4단계] PostgreSQL과 비교 (기존 시스템)")
    print("-" * 80)

    try:
        from app.core.retrieval import similarity_search as similarity_search_pg

        query = "타율"
        embedding = embed_texts([query], settings)[0]

        pg_results = similarity_search_pg(embedding=embedding, limit=5)

        print(f"PostgreSQL 검색 결과: {len(pg_results)}개")

        if pg_results:
            print("\nTop 3 결과:")
            for i, result in enumerate(pg_results[:3], 1):
                print(f"  {i}. {result.get('title', 'N/A')}")
                print(f"     유사도: {result.get('similarity', 0):.4f}")

    except Exception as e:
        print(f"PostgreSQL 검색 실패: {e}")

    print("\n" + "="*80)
    print("디버깅 완료")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
