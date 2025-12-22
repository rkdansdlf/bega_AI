#!/bin/bash
# Firestore 벡터 인덱스 생성 스크립트
#
# 사용법:
#   chmod +x scripts/create_firestore_index.sh
#   ./scripts/create_firestore_index.sh

set -e

PROJECT_ID="bega-186a7"
DATABASE_ID="chatbot"
COLLECTION="rag_chunks"

echo "============================================"
echo "Firestore Vector Index 생성"
echo "============================================"
echo "프로젝트: $PROJECT_ID"
echo "데이터베이스: $DATABASE_ID"
echo "컬렉션: $COLLECTION"
echo "============================================"
echo ""

# 1. gcloud CLI 설치 확인
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI가 설치되지 않았습니다."
    echo "다음 링크에서 설치하세요: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✓ gcloud CLI 설치 확인"

# 2. 프로젝트 설정
echo ""
echo "프로젝트 설정 중..."
gcloud config set project $PROJECT_ID

# 3. 현재 인증 확인
echo ""
echo "현재 인증 상태:"
gcloud auth list

# 4. Firestore API 활성화 확인
echo ""
echo "Firestore API 활성화 확인 중..."
if gcloud services list --enabled | grep -q firestore.googleapis.com; then
    echo "✓ Firestore API가 이미 활성화되어 있습니다."
else
    echo "Firestore API를 활성화합니다..."
    gcloud services enable firestore.googleapis.com
fi

# 5. 기존 인덱스 확인
echo ""
echo "============================================"
echo "기존 인덱스 확인 중..."
echo "============================================"
gcloud firestore indexes composite list --database=$DATABASE_ID || echo "인덱스가 없습니다."

# 6. 벡터 인덱스 생성
echo ""
echo "============================================"
echo "벡터 인덱스 생성 중..."
echo "============================================"
echo ""
echo "인덱스 구성:"
echo "  - embedding (Vector, dimension: 1536, COSINE)"
echo "  - seasonYear (ASCENDING)"
echo "  - teamId (ASCENDING)"
echo ""

# 주의: Firestore의 벡터 인덱스는 아직 gcloud CLI에서 완전히 지원되지 않을 수 있습니다.
# 이 경우 Firebase Console에서 수동으로 생성해야 합니다.

echo "⚠️  주의: gcloud CLI로 벡터 인덱스 생성이 실패할 수 있습니다."
echo "실패하는 경우 다음 방법으로 수동 생성하세요:"
echo ""
echo "1. Firebase Console 접속: https://console.firebase.google.com/"
echo "2. 프로젝트 선택: $PROJECT_ID"
echo "3. Firestore Database > Indexes 탭"
echo "4. 'Create Index' 클릭"
echo "5. 설정:"
echo "   - Collection: $COLLECTION"
echo "   - Fields:"
echo "     * embedding (Vector, dimension: 1536, distance: COSINE)"
echo "     * seasonYear (Ascending)"
echo "     * teamId (Ascending)"
echo ""
read -p "계속 진행하시겠습니까? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "취소되었습니다."
    exit 1
fi

# 벡터 인덱스 생성 시도
# 참고: Firestore Vector Search는 최신 기능이므로 CLI 지원이 제한적일 수 있습니다.
echo ""
echo "gcloud CLI로 인덱스 생성을 시도합니다..."

gcloud firestore indexes composite create \
  --database=$DATABASE_ID \
  --collection-group=$COLLECTION \
  --field-config=field-path=embedding,vector-config='{"dimension":"1536","flat": {}}' \
  --field-config=field-path=seasonYear,order=ASCENDING \
  --field-config=field-path=teamId,order=ASCENDING \
  2>&1 || {
    echo ""
    echo "============================================"
    echo "❌ gcloud CLI로 인덱스 생성 실패"
    echo "============================================"
    echo ""
    echo "다음 방법 중 하나를 선택하세요:"
    echo ""
    echo "방법 1: Firebase Console에서 수동 생성"
    echo "  1. https://console.firebase.google.com/project/$PROJECT_ID/firestore/indexes"
    echo "  2. 'Create Index' 클릭 후 위의 설정 입력"
    echo ""
    echo "방법 2: 간단한 단일 필드 인덱스만 생성 (권장)"
    echo "  gcloud firestore indexes fields create embedding \\"
    echo "    --database=$DATABASE_ID \\"
    echo "    --collection-group=$COLLECTION \\"
    echo "    --vector-config=dimension=1536,flat"
    echo ""
    exit 1
}

echo ""
echo "============================================"
echo "인덱스 생성이 시작되었습니다!"
echo "============================================"
echo ""
echo "인덱스 빌드는 10-30분 정도 소요될 수 있습니다."
echo "상태 확인:"
echo "  gcloud firestore indexes composite list --database=$DATABASE_ID"
echo ""
echo "또는 Firebase Console에서 확인:"
echo "  https://console.firebase.google.com/project/$PROJECT_ID/firestore/indexes"
echo ""
echo "인덱스 상태가 'Building' → 'Enabled'가 되면 사용 가능합니다."
echo "============================================"
