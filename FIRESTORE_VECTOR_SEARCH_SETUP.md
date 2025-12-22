# Firestore Vector Search 설정 가이드

## 개요
Supabase의 pgvector에서 **Firestore Vector Search**로 마이그레이션하여 RAG 시스템을 구축하는 가이드입니다.

## 1단계: Firestore Vector Search 활성화

### 1.1 Google Cloud Console에서 API 활성화
```bash
# Firebase Console (https://console.firebase.google.com/)
1. 프로젝트 선택 (bega-186a7)
2. 왼쪽 메뉴에서 "Firestore Database" 클릭
3. 이미 "chatbot" 데이터베이스가 생성되어 있음 확인
```

### 1.2 Vertex AI API 활성화 (벡터 검색용)
```bash
# Google Cloud Console (https://console.cloud.google.com/)
1. 프로젝트 선택 (bega-186a7)
2. "APIs & Services" > "Enable APIs and Services" 클릭
3. "Vertex AI API" 검색 후 활성화
```

## 2단계: 벡터 인덱스 생성

Firestore에서 벡터 검색을 사용하려면 **벡터 인덱스**를 생성해야 합니다.

### 2.1 gcloud CLI로 인덱스 생성

```bash
# gcloud CLI 설치 (아직 안 했다면)
# https://cloud.google.com/sdk/docs/install

# 프로젝트 설정
gcloud config set project bega-186a7

# Firestore 벡터 인덱스 생성
gcloud firestore indexes composite create \
  --database=chatbot \
  --collection-group=rag_chunks \
  --field-config=field-path=embedding,vector-config='{"dimension":"1536","flat": {}}' \
  --field-config=field-path=seasonYear,order=ASCENDING \
  --field-config=field-path=teamId,order=ASCENDING
```

### 2.2 Firebase Console에서 인덱스 생성 (대안)

```
1. Firebase Console > Firestore Database > Indexes
2. "Create Index" 클릭
3. 설정:
   - Collection: rag_chunks
   - Fields to index:
     * embedding (Vector, dimension: 1536)
     * seasonYear (Ascending)
     * teamId (Ascending)
4. "Create" 클릭 후 인덱스 빌드 대기 (10-30분 소요)
```

### 2.3 인덱스 상태 확인

```bash
# CLI로 확인
gcloud firestore indexes composite list --database=chatbot

# 또는 Firebase Console에서 확인
# Firestore Database > Indexes 탭에서 상태 확인
# 상태가 "Building" → "Enabled"가 되면 사용 가능
```

## 3단계: Python 환경 설정

### 3.1 필수 패키지 설치

```bash
cd /Users/mac/project/KBO_platform/AI
source .venv/bin/activate

# Firebase Admin SDK 설치
pip install firebase-admin google-cloud-firestore

# requirements.txt 업데이트
pip freeze > requirements.txt
```

### 3.2 환경 변수 설정

`.env` 파일에 다음 추가:

```bash
# Firebase 설정
FIREBASE_SERVICE_ACCOUNT_KEY=/Users/mac/project/KBO_platform/AI/bega-186a7-firebase-adminsdk-fbsvc-bb50c006a7.json
FIREBASE_PROJECT_ID=bega-186a7
FIRESTORE_DATABASE_ID=chatbot

# 기존 Supabase 설정은 마이그레이션 완료 후 제거
# SUPABASE_DB_URL=...
```

## 4단계: 코드 구현

### 4.1 Firestore Vector Search 모듈 (`app/core/retrieval_firestore.py`)

새로운 검색 모듈이 생성되었습니다:
- `similarity_search_firestore()`: Firestore 벡터 검색 함수
- 메타데이터 필터링 지원 (seasonYear, teamId 등)
- 코사인 유사도 기반 검색

### 4.2 Agent 코드 수정

`app/agents/baseball_agent.py`에서 retrieval 모듈 import를 변경:

```python
# 기존
from ..core.retrieval import similarity_search

# 변경
from ..core.retrieval_firestore import similarity_search_firestore as similarity_search
```

또는 점진적 전환을 위해:

```python
# 환경 변수로 선택
import os
USE_FIRESTORE = os.getenv("USE_FIRESTORE_SEARCH", "false").lower() == "true"

if USE_FIRESTORE:
    from ..core.retrieval_firestore import similarity_search_firestore as similarity_search
else:
    from ..core.retrieval import similarity_search
```

## 5단계: 테스트

### 5.1 기본 검색 테스트

```bash
# FastAPI 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# 테스트 요청
curl -X POST "http://localhost:8001/chat/completion" \
  -H "Content-Type: application/json" \
  -d '{"question": "2024년 KIA 타이거즈의 홈런왕은 누구야?"}'
```

### 5.2 성능 비교

마이그레이션 완료 후:
1. Supabase pgvector 검색 (기존)
2. Firestore Vector Search (신규)

두 방식의 응답 시간 및 정확도를 비교하여 검증합니다.

## 6단계: Supabase 데이터 정리 (마이그레이션 완료 후)

### 6.1 검증 완료 후 Supabase 데이터 삭제

```bash
# 마이그레이션 스크립트의 cleanup 기능 사용
python scripts/migrate_to_firebase.py \
  --service-account-key bega-186a7-firebase-adminsdk-fbsvc-bb50c006a7.json \
  --cleanup

# 또는 직접 SQL 실행
psql $SUPABASE_DB_URL -c "
  UPDATE rag_chunks
  SET content = NULL, embedding = NULL
  WHERE meta ? 'firebase_doc_id'
"

# 공간 회수
psql $SUPABASE_DB_URL -c "VACUUM FULL rag_chunks;"
```

## 비용 예상

### Firestore Vector Search 비용 (2025년 1월 기준)

- **문서 저장**: $0.18/GB/월
  - 224,650 청크 × ~2KB = ~450MB → **$0.08/월**

- **읽기 작업**: $0.06 / 100,000 reads
  - 월 10,000 쿼리 × 10개 결과 = 100,000 reads → **$0.06/월**

- **벡터 검색**: $0.20 / 1,000 queries
  - 월 10,000 쿼리 → **$2.00/월**

**총 예상 비용: ~$2.14/월** (Supabase 무료 플랜 500MB 제한 대비 훨씬 저렴)

## Vertex AI Vector Search vs Firestore Vector Search 비교

| 항목 | Firestore Vector Search | Vertex AI Vector Search |
|------|------------------------|------------------------|
| **설정 난이도** | 쉬움 (Firestore 내장) | 복잡 (별도 서비스) |
| **인덱스 관리** | 자동 관리 | 수동 관리 필요 |
| **쿼리 방식** | Firestore SDK | gRPC API |
| **비용** | 저렴 (~$2/월) | 비쌈 (~$20+/월) |
| **성능** | 중간 (10-100ms) | 빠름 (1-10ms) |
| **확장성** | ~100만 벡터 | 수억 벡터 |
| **권장 사용** | 중소규모 RAG | 대규모 프로덕션 |

**현재 프로젝트 (224,650 청크)**: Firestore Vector Search가 충분히 적합합니다.

## 트러블슈팅

### 인덱스가 생성되지 않는 경우

```bash
# 인덱스 상태 확인
gcloud firestore indexes composite list --database=chatbot

# 인덱스 삭제 후 재생성
gcloud firestore indexes composite delete INDEX_ID --database=chatbot
# 위의 생성 명령어 다시 실행
```

### "Vector dimension mismatch" 오류

- embedding 차원이 1536인지 확인
- 마이그레이션 스크립트에서 임베딩이 올바르게 저장되었는지 확인

```python
# Firestore 문서 확인
doc = db.collection('rag_chunks').document('1').get()
embedding = doc.get('embedding')
print(f"Embedding dimension: {len(embedding)}")  # 1536이어야 함
```

### 검색 성능이 느린 경우

1. 인덱스가 "Enabled" 상태인지 확인
2. 필터 조건을 추가하여 검색 범위 축소
3. limit 값을 줄여서 반환 결과 수 제한

## 다음 단계

1. ✅ Firestore Vector Search 활성화
2. ✅ 벡터 인덱스 생성
3. ⏳ AI 서비스 코드를 Firestore로 변경
4. ⏳ 벡터 검색 테스트 및 검증
5. ⏳ Supabase 데이터 삭제 (검증 완료 후)
