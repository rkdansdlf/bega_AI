# ⚾ KBO Chatbot API

한국 야구(KBO) 데이터 기반 AI 챗봇 API

## 📁 프로젝트 구조

```
kbo-chatbot/
├── main.py              # FastAPI 앱 및 라우터
├── config.py            # 설정 관리
├── database.py          # MySQL 연결 및 쿼리
├── chatbot.py           # 챗봇 로직 (Gemini API)
├── requirements.txt     # 필요 패키지
├── .env.example         # 환경변수 템플릿
├── .env                 # 환경변수 (직접 생성)
└── README.md            # 이 파일
```

## 🚀 빠른 시작

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
# .env.example을 .env로 복사
cp .env.example .env

# .env 파일 수정
# - GEMINI_BASE_URL
# - GEMINI_API_KEY
# - MySQL 정보
```

### 3. 데이터베이스 연결 테스트

```bash
python database.py
```

출력 예시:
```
==================================================
데이터베이스 연결 테스트
==================================================
[DB] 연결 테스트 성공
✅ 데이터베이스 연결 성공

📊 game 테이블 정보:
레코드 수: 1234
컬럼 수: 11
==================================================
```

### 4. 챗봇 로직 테스트

```bash
python chatbot.py
```

### 5. 서버 실행

```bash
# 방법 1: Python으로 직접 실행
python main.py

# 방법 2: uvicorn 명령어 사용
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

서버가 실행되면:
- API: http://localhost:8001
- Swagger 문서: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## 📡 API 엔드포인트

### 기본 정보

#### `GET /`
서비스 정보 및 엔드포인트 목록

**응답:**
```json
{
  "service": "KBO Chatbot API",
  "status": "running",
  "version": "1.0.0",
  "endpoints": { ... }
}
```

#### `GET /health`
서버 및 연결 상태 확인

**응답:**
```json
{
  "status": "healthy",
  "database": "connected",
  "api": "connected",
  "timestamp": "2025-10-15T..."
}
```

### 챗봇

#### `POST /api/chatbot`
상세 정보 포함 응답

**요청:**
```json
{
  "question": "이번 시즌 가장 큰 점수차로 이긴 경기는?"
}
```

**응답:**
```json
{
  "answer": "이번 시즌 가장 큰 점수차로 이긴 경기는...",
  "query_executed": "SELECT * FROM game WHERE...",
  "execution_time": 1.23
}
```

#### `POST /api/chatbot/simple`
답변만 반환 (프론트엔드용)

**요청:**
```json
{
  "question": "LG 트윈스 최근 경기는?"
}
```

**응답:**
```json
{
  "answer": "LG 트윈스의 최근 경기는..."
}
```

### 데이터베이스

#### `GET /api/db/info`
데이터베이스 테이블 정보

**응답:**
```json
{
  "table": "game",
  "columns": [...],
  "record_count": 1234
}
```

## 🧪 테스트

### cURL로 테스트

```bash
# 헬스 체크
curl http://localhost:8001/health

# 챗봇 질문
curl -X POST "http://localhost:8001/api/chatbot/simple" \
  -H "Content-Type: application/json" \
  -d '{"question": "오늘 경기 일정"}'
```

### Python으로 테스트

```python
import requests

# 챗봇 질문
response = requests.post(
    "http://localhost:8001/api/chatbot/simple",
    json={"question": "팀 순위 알려줘"}
)
print(response.json())
```

## 📝 파일별 역할

### `config.py`
- 환경변수 관리
- 설정값 중앙화
- Pydantic Settings 사용

### `database.py`
- MySQL 연결 관리
- SQL 쿼리 실행
- 연결 테스트 기능
- 데이터베이스 싱글톤 패턴

### `chatbot.py`
- Gemini API 호출
- Function Calling 처리
- 시스템 프롬프트 관리
- 챗봇 로직 캡슐화

### `main.py`
- FastAPI 앱 생성
- API 라우팅
- CORS 설정
- 에러 핸들링
- API 문서 자동 생성

## 🔧 환경변수 설명

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `MODEL_NAME` | Gemini 모델 이름 | gemini-2.5-flash |
| `GEMINI_BASE_URL` | Gemini API URL | - |
| `GEMINI_API_KEY` | Gemini API Key | - |
| `MYSQL_HOST` | MySQL 호스트 | localhost |
| `MYSQL_USER` | MySQL 사용자 | user1 |
| `MYSQL_PASSWORD` | MySQL 비밀번호 | 1234 |
| `MYSQL_DATABASE` | MySQL 데이터베이스 | api_test_data |
| `MYSQL_PORT` | MySQL 포트 | 3306 |
| `API_HOST` | API 서버 호스트 | 0.0.0.0 |
| `API_PORT` | API 서버 포트 | 8001 |
| `API_RELOAD` | 자동 재시작 | True |

## 💡 사용 예시

### 질문 예시

- "이번 시즌 가장 큰 점수차로 이긴 경기는?"
- "LG 트윈스의 최근 5경기 결과는?"
- "잠실야구장에서 열린 경기 수는?"
- "2025년 3월 경기 일정 알려줘"
- "KIA 타이거즈 홈경기 승률은?"

## 🐛 트러블슈팅

### 1. 데이터베이스 연결 실패

```bash
# MySQL 서버 확인
mysql -u user1 -p

# 권한 확인
SHOW GRANTS FOR 'user1'@'localhost';
```

### 2. API 키 오류

- `.env` 파일에 `GEMINI_API_KEY` 확인
- API 키 유효성 확인

### 3. 포트 충돌

```bash
# 8000번 포트 사용 확인
lsof -i :8001

# 다른 포트로 실행
API_PORT=8080 python main.py
```

## 📦 의존성

주요 패키지:
- `fastapi`: 웹 프레임워크
- `uvicorn`: ASGI 서버
- `openai`: Gemini API 클라이언트
- `mysql-connector-python`: MySQL 드라이버
- `pydantic`: 데이터 검증

