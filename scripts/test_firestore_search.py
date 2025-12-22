"""
Firestore Vector Search 테스트 스크립트

마이그레이션이 완료되고 벡터 인덱스가 생성된 후에 실행하세요.

사용법:
    python scripts/test_firestore_search.py
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 설정 (테스트용)
os.environ["USE_FIRESTORE_SEARCH"] = "true"
os.environ["FIREBASE_SERVICE_ACCOUNT_KEY"] = str(
    project_root / "bega-186a7-firebase-adminsdk-fbsvc-bb50c006a7.json"
)
os.environ["FIRESTORE_DATABASE_ID"] = "chatbot"

from dotenv import load_dotenv
load_dotenv()

from app.core.retrieval_firestore import similarity_search_firestore
from app.core.embeddings import embed_texts
from app.config import Settings


def test_basic_search():
    """기본 검색 테스트"""
    print("\n" + "="*60)
    print("1. 기본 검색 테스트")
    print("="*60)

    query = "2024년 KIA 타이거즈의 홈런왕은 누구야?"
    print(f"질문: {query}\n")

    # 임베딩 생성
    settings = Settings()
    embeddings = embed_texts([query], settings)

    if not embeddings:
        print("❌ 임베딩 생성 실패")
        return False

    print(f"✓ 임베딩 생성 완료 (차원: {len(embeddings[0])})")

    # Firestore 검색
    print("\nFirestore Vector Search 실행 중...")
    results = similarity_search_firestore(
        embeddings[0],
        limit=5,
    )

    if not results:
        print("⚠️  검색 결과 없음 (마이그레이션이 완료되지 않았거나 인덱스가 생성되지 않았을 수 있습니다)")
        return False

    print(f"\n✓ {len(results)}개 결과 반환\n")

    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.get('title', 'N/A')}")
        print(f"   유사도: {doc.get('similarity', 0):.4f}")
        print(f"   출처: {doc.get('source_table', 'N/A')}")
        print(f"   내용 (일부): {doc.get('content', '')[:100]}...")
        print()

    return True


def test_filtered_search():
    """필터링 검색 테스트"""
    print("\n" + "="*60)
    print("2. 필터링 검색 테스트 (2024년, KIA)")
    print("="*60)

    query = "홈런왕"
    print(f"질문: {query}")
    print(f"필터: seasonYear=2024, teamId=KIA\n")

    # 임베딩 생성
    settings = Settings()
    embeddings = embed_texts([query], settings)

    if not embeddings:
        print("❌ 임베딩 생성 실패")
        return False

    # Firestore 검색 (필터 적용)
    print("Firestore Vector Search 실행 중...")
    results = similarity_search_firestore(
        embeddings[0],
        limit=5,
        filters={
            "seasonYear": 2024,
            "teamId": "KIA"
        }
    )

    if not results:
        print("⚠️  검색 결과 없음")
        return False

    print(f"\n✓ {len(results)}개 결과 반환\n")

    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.get('title', 'N/A')}")
        print(f"   유사도: {doc.get('similarity', 0):.4f}")
        print(f"   출처: {doc.get('source_table', 'N/A')}")
        print(f"   메타데이터: seasonYear={doc.get('meta', {}).get('seasonYear')}, teamId={doc.get('meta', {}).get('teamId')}")
        print()

    return True


def test_document_count():
    """Firestore 문서 개수 확인"""
    print("\n" + "="*60)
    print("3. Firestore 문서 개수 확인")
    print("="*60)

    try:
        from firebase_admin import firestore
        from app.core.retrieval_firestore import _init_firebase

        db = _init_firebase()
        collection_ref = db.collection('rag_chunks')

        # 전체 문서 개수 확인 (샘플링)
        # 주의: Firestore는 전체 개수를 직접 세는 것이 비효율적이므로 샘플링 사용
        docs = collection_ref.limit(10).stream()
        sample_docs = list(docs)

        print(f"\n✓ 샘플 문서 확인: {len(sample_docs)}개")

        if sample_docs:
            print("\n첫 번째 문서 예시:")
            first_doc = sample_docs[0].to_dict()
            print(f"  - ID: {first_doc.get('id')}")
            print(f"  - Title: {first_doc.get('title')}")
            print(f"  - Source: {first_doc.get('sourceTable')}")
            print(f"  - Embedding 차원: {len(first_doc.get('embedding', []))}")

        return True

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_comparison():
    """Supabase vs Firestore 검색 비교 (옵션)"""
    print("\n" + "="*60)
    print("4. Supabase vs Firestore 검색 비교")
    print("="*60)

    query = "타율 1위"
    print(f"질문: {query}\n")

    # 임베딩 생성
    settings = Settings()
    embeddings = embed_texts([query], settings)

    if not embeddings:
        print("❌ 임베딩 생성 실패")
        return False

    # Firestore 검색
    print("1) Firestore Vector Search:")
    firestore_results = similarity_search_firestore(
        embeddings[0],
        limit=3,
    )

    if firestore_results:
        for i, doc in enumerate(firestore_results, 1):
            print(f"   {i}. {doc.get('title', 'N/A')} (유사도: {doc.get('similarity', 0):.4f})")
    else:
        print("   검색 결과 없음")

    # Supabase 검색 (비교용, 연결 가능한 경우만)
    try:
        import psycopg2
        from app.core.retrieval import similarity_search

        supabase_url = os.getenv("SUPABASE_DB_URL")
        if supabase_url:
            os.environ["USE_FIRESTORE_SEARCH"] = "false"  # 임시로 Supabase 모드로 전환

            print("\n2) Supabase pgvector:")
            conn = psycopg2.connect(supabase_url)

            supabase_results = similarity_search(
                conn,
                embeddings[0],
                limit=3,
            )

            if supabase_results:
                for i, doc in enumerate(supabase_results, 1):
                    print(f"   {i}. {doc.get('title', 'N/A')} (유사도: {doc.get('similarity', 0):.4f})")
            else:
                print("   검색 결과 없음")

            conn.close()
            os.environ["USE_FIRESTORE_SEARCH"] = "true"  # 다시 Firestore 모드로

    except Exception as e:
        print(f"\n2) Supabase pgvector: 비교 불가 ({e})")

    return True


def main():
    """테스트 실행"""
    print("\n" + "="*60)
    print("Firestore Vector Search 테스트")
    print("="*60)
    print("\n⚠️  주의 사항:")
    print("1. 마이그레이션이 완료되어야 합니다.")
    print("2. 벡터 인덱스가 생성되어야 합니다 (10-30분 소요).")
    print("3. 환경 변수가 올바르게 설정되어야 합니다.")
    print()

    input("계속하려면 Enter를 누르세요...")

    results = []

    # 1. 기본 검색
    results.append(("기본 검색", test_basic_search()))

    # 2. 필터링 검색
    results.append(("필터링 검색", test_filtered_search()))

    # 3. 문서 개수 확인
    results.append(("문서 개수", test_document_count()))

    # 4. 비교 테스트 (옵션)
    if os.getenv("SUPABASE_DB_URL"):
        results.append(("Supabase 비교", test_comparison()))

    # 결과 요약
    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)

    for name, success in results:
        status = "✓ 성공" if success else "❌ 실패"
        print(f"{name}: {status}")

    print("\n" + "="*60)
    all_success = all(success for _, success in results)

    if all_success:
        print("✓ 모든 테스트 통과!")
        print("\n다음 단계:")
        print("1. .env 파일에서 USE_FIRESTORE_SEARCH=true 설정")
        print("2. FastAPI 서버 재시작")
        print("3. 실제 챗봇으로 테스트")
        print("4. Supabase 데이터 삭제 (검증 완료 후)")
    else:
        print("⚠️  일부 테스트 실패")
        print("\n문제 해결:")
        print("1. 마이그레이션 상태 확인")
        print("2. 벡터 인덱스 상태 확인 (Firebase Console)")
        print("3. 환경 변수 확인 (.env)")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
