"""Firestore Vector Search 기반의 유사도 검색 기능을 제공하는 모듈입니다.

이 모듈은 Firebase Firestore의 벡터 검색 기능을 사용하여
벡터 임베딩 간의 코사인 유사도를 계산하고, 관련성 높은 문서를 검색하는 기능을 구현합니다.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

logger = logging.getLogger(__name__)

# Firebase 초기화 (싱글톤 패턴)
_firebase_initialized = False
_db_firestore = None


def _init_firebase():
    """Firebase Admin SDK를 초기화합니다 (한 번만 실행)."""
    global _firebase_initialized, _db_firestore

    if _firebase_initialized:
        return _db_firestore

    try:
        # 환경 변수에서 설정 읽기
        service_account_key_path = os.getenv(
            "FIREBASE_SERVICE_ACCOUNT_KEY",
            "/Users/mac/project/KBO_platform/AI/bega-186a7-firebase-adminsdk-fbsvc-bb50c006a7.json"
        )
        database_id = os.getenv("FIRESTORE_DATABASE_ID", "chatbot")

        if not os.path.exists(service_account_key_path):
            raise FileNotFoundError(
                f"Firebase 서비스 계정 키 파일을 찾을 수 없습니다: {service_account_key_path}\n"
                f"환경 변수 FIREBASE_SERVICE_ACCOUNT_KEY를 설정하거나 파일을 생성하세요."
            )

        # Firebase Admin SDK 초기화
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_key_path)

            # 프로젝트 ID 읽기
            with open(service_account_key_path, 'r') as f:
                service_account_data = json.load(f)
                project_id = service_account_data.get('project_id')
                storage_bucket = f"{project_id}.firebasestorage.app"

            firebase_admin.initialize_app(cred, {
                'storageBucket': storage_bucket
            })

            logger.info(f"Firebase 초기화 완료: project_id={project_id}, database={database_id}")

        # Firestore 클라이언트 생성
        _db_firestore = firestore.client(database_id=database_id)
        _firebase_initialized = True

        return _db_firestore

    except Exception as e:
        logger.error(f"Firebase 초기화 실패: {e}")
        raise


def similarity_search_firestore(
    embedding: Sequence[float],
    *,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    keyword: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """주어진 임베딩과 유사한 문서를 Firestore에서 검색합니다.

    Args:
        embedding: 검색의 기준이 될 벡터 임베딩 (1536 차원).
        limit: 반환할 최대 문서 수.
        filters: 검색 결과를 필터링할 조건 (예: {'seasonYear': 2024, 'teamId': 'KIA'}).
        keyword: 텍스트 검색에 사용할 키워드 (현재 미지원, 향후 구현 가능).

    Returns:
        유사도 순으로 정렬된 문서 리스트. 각 문서는 사전(dict) 형태로 반환됩니다.

    Example:
        >>> from app.core.retrieval_firestore import similarity_search_firestore
        >>> embedding = [0.1, 0.2, ..., 0.5]  # 1536 차원 벡터
        >>> results = similarity_search_firestore(
        ...     embedding,
        ...     limit=5,
        ...     filters={'seasonYear': 2024, 'teamId': 'KIA'}
        ... )
        >>> for doc in results:
        ...     print(doc['title'], doc['similarity'])
    """
    try:
        # Firestore 클라이언트 가져오기
        db = _init_firebase()

        # 컬렉션 참조
        collection_ref = db.collection('rag_chunks')

        # 필터 조건 적용
        query = collection_ref
        if filters:
            for key, value in filters.items():
                if value is None:
                    continue

                # JSON 필드 필터링 지원 (예: "meta.league")
                if "." in key:
                    # Firestore에서는 중첩된 필드를 직접 쿼리 가능
                    query = query.where(key, "==", value)
                else:
                    query = query.where(key, "==", value)

        # 벡터 검색 실행
        # Firestore Vector Search는 find_nearest() 메서드 사용
        vector_query = query.find_nearest(
            vector_field="embedding",
            query_vector=Vector(embedding),
            distance_measure=DistanceMeasure.COSINE,
            limit=limit,
        )

        # 결과 가져오기
        docs = vector_query.stream()

        # 결과를 dict 리스트로 변환
        results = []
        for doc in docs:
            data = doc.to_dict()

            # pgvector 호환성을 위해 필드 이름 변환 (camelCase → snake_case)
            result = {
                'id': data.get('id'),
                'title': data.get('title'),
                'content': data.get('content'),  # 또는 Storage에서 로드
                'source_table': data.get('sourceTable'),
                'source_row_id': data.get('sourceRowId'),
                'meta': data.get('meta', {}),
                'similarity': 1.0 - doc.get('distance', 0),  # 코사인 거리 → 유사도
            }

            # content가 없고 storagePath가 있는 경우 (향후 구현 가능)
            if not result['content'] and data.get('storagePath'):
                # Firebase Storage에서 텍스트 로드
                # result['content'] = _load_from_storage(data['storagePath'])
                pass

            results.append(result)

        logger.info(f"Firestore 벡터 검색 완료: {len(results)}개 문서 반환 (limit={limit})")
        return results

    except Exception as e:
        logger.error(f"Firestore 벡터 검색 실패: {e}")
        # 오류 발생 시 빈 리스트 반환 (graceful degradation)
        return []


def _load_from_storage(storage_path: str) -> str:
    """Firebase Storage에서 텍스트 컨텐츠를 로드합니다 (옵션).

    Args:
        storage_path: Storage blob 경로 (예: "rag_chunks/player_stats/123.txt")

    Returns:
        텍스트 컨텐츠
    """
    try:
        from firebase_admin import storage

        bucket = storage.bucket()
        blob = bucket.blob(storage_path)
        content = blob.download_as_text(encoding='utf-8')

        return content

    except Exception as e:
        logger.warning(f"Storage에서 컨텐츠 로드 실패: {storage_path}, {e}")
        return ""


# 기존 코드와의 호환성을 위한 별칭
similarity_search = similarity_search_firestore
