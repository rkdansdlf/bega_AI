"""
Firestore 임베딩 필드를 Vector 타입으로 변환하는 스크립트

기존 마이그레이션에서 임베딩을 일반 배열로 저장했지만,
Firestore 벡터 검색을 위해서는 Vector 타입이 필요합니다.

이 스크립트는 모든 문서의 임베딩 필드를 Vector 타입으로 업데이트합니다.

사용법:
    python scripts/fix_embedding_vector_type.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.vector import Vector
from tqdm import tqdm


def main():
    print("\n" + "="*80)
    print("Firestore 임베딩 필드를 Vector 타입으로 변환")
    print("="*80 + "\n")

    # Firebase 초기화
    service_account_key_path = os.getenv(
        "FIREBASE_SERVICE_ACCOUNT_KEY",
        "/Users/mac/project/KBO_platform/AI/bega-186a7-firebase-adminsdk-fbsvc-bb50c006a7.json"
    )

    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_key_path)
        firebase_admin.initialize_app(cred)

    db = firestore.client(database_id='begachatbot')

    print("📊 전체 문서 수 확인 중...")

    # 전체 문서 수 확인 (샘플링)
    batch_size = 1000
    total_count = 0
    last_doc = None

    # 빠른 카운트 (처음 10,000개만 확인하여 전체 추정)
    for _ in range(10):
        query = db.collection('rag_chunks').limit(batch_size)
        if last_doc:
            query = query.start_after(last_doc)

        docs = query.get()
        count = len(docs)
        total_count += count

        if count < batch_size:
            break
        last_doc = docs[-1]

    if total_count >= 10000:
        # 전체 예상 (이전 마이그레이션에서 224,565개로 확인됨)
        estimated_total = 224565
        print(f"✓ 예상 문서 수: ~{estimated_total:,}개")
    else:
        estimated_total = total_count
        print(f"✓ 전체 문서 수: {total_count:,}개")

    print(f"\n⚠️  주의: 이 작업은 모든 문서의 embedding 필드를 업데이트합니다.")
    print(f"         예상 소요 시간: 10-20분")
    print()

    response = input("계속하시겠습니까? (y/N): ")
    if response.lower() != 'y':
        print("취소되었습니다.")
        return

    print("\n" + "="*80)
    print("임베딩 필드 업데이트 시작")
    print("="*80 + "\n")

    # 배치 업데이트
    updated_count = 0
    error_count = 0
    last_doc = None

    # Firestore 배치 크기 제한: 페이로드 10MB 이하
    # 1536차원 벡터 × 100개 ≈ 1.1MB (안전)
    update_batch_size = 100

    with tqdm(total=estimated_total, desc="업데이트 진행") as pbar:
        while True:
            # 배치 가져오기
            query = db.collection('rag_chunks').limit(batch_size)
            if last_doc:
                query = query.start_after(last_doc)

            docs = query.get()

            if not docs:
                break

            # 100개씩 나눠서 배치 업데이트
            for i in range(0, len(docs), update_batch_size):
                chunk_docs = docs[i:i + update_batch_size]
                batch = db.batch()
                batch_count = 0

                for doc in chunk_docs:
                    try:
                        data = doc.to_dict()
                        embedding = data.get('embedding')

                        if embedding and isinstance(embedding, list):
                            # 일반 배열을 Vector 타입으로 변환
                            doc_ref = db.collection('rag_chunks').document(doc.id)
                            batch.update(doc_ref, {'embedding': Vector(embedding)})
                            batch_count += 1

                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:  # 처음 5개 에러만 출력
                            print(f"\n오류 (doc_id={doc.id}): {e}")

                # 배치 커밋
                if batch_count > 0:
                    try:
                        batch.commit()
                        updated_count += batch_count
                        pbar.update(len(chunk_docs))
                    except Exception as e:
                        print(f"\n배치 커밋 실패: {e}")
                        error_count += len(chunk_docs)
                else:
                    pbar.update(len(chunk_docs))

            last_doc = docs[-1]

            # 배치 크기보다 적게 가져왔으면 종료
            if len(docs) < batch_size:
                break

    print("\n" + "="*80)
    print("업데이트 완료!")
    print("="*80)
    print(f"\n총 업데이트: {updated_count:,}개")

    if error_count > 0:
        print(f"⚠️  오류 발생: {error_count}개")
    else:
        print("✓ 모든 문서가 성공적으로 업데이트되었습니다!")

    print(f"\n이제 벡터 검색이 정상적으로 작동합니다.")
    print(f"테스트: python scripts/quick_test_firestore.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
