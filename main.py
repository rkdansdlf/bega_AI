"""
FastAPI 메인 파일
API 엔드포인트와 라우팅을 담당합니다.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime
from typing import Optional
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool # 이미 임포트되어 있음

from config import settings
from database import db_manager
from chatbot import chatbot


# ========================================
# FastAPI 앱 생성
# ========================================
app = FastAPI(
    title="KBO Chatbot API",
    description="한국 야구(KBO) 전문 챗봇 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Pydantic 모델 정의
# ========================================
class ChatRequest(BaseModel):
    """챗봇 요청 모델"""
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "이번 시즌 가장 큰 점수차로 이긴 경기 알려줘"
            }
        }


class ChatResponse(BaseModel):
    """챗봇 응답 모델 (상세)"""
    answer: str
    query_executed: Optional[str] = None
    execution_time: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "이번 시즌 가장 큰 점수차로 이긴 경기는...",
                "query_executed": "SELECT * FROM game...",
                "execution_time": 1.23
            }
        }


class SimpleChatResponse(BaseModel):
    """챗봇 응답 모델 (간단)"""
    answer: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "이번 시즌 가장 큰 점수차로 이긴 경기는..."
            }
        }


class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str
    database: str
    api: str
    timestamp: str


# ========================================
# API 엔드포인트
# ========================================

@app.get("/", tags=["기본"])
async def root():
    """
    API 루트 - 서비스 정보 및 엔드포인트 목록
    """
    return {
        "service": "KBO Chatbot API",
        "status": "running",
        "version": "1.0.0",
        "description": "한국 야구(KBO) 데이터 기반 AI 챗봇",
        "endpoints": {
            "health": "/health",
            "chatbot_detailed": "/api/chatbot",
            "chatbot_simple": "/api/chatbot/simple",
            "swagger_docs": "/docs",
            "redoc": "/redoc"
        },
        "example_questions": [
            "이번 시즌 가장 큰 점수차로 이긴 경기는?",
            "LG 트윈스의 최근 경기 결과는?",
            "잠실야구장에서 열린 경기 수는?"
        ]
    }


@app.get("/health", response_model=HealthResponse, tags=["기본"])
async def health_check():
    """
    헬스 체크 - 서버 및 연결 상태 확인
    
    데이터베이스와 API 연결 상태를 확인합니다.
    """
    # 데이터베이스 연결 테스트
    db_status = "connected" if db_manager.test_connection() else "disconnected"
    
    # API 연결 테스트 (간단한 체크)
    api_status = "connected" if chatbot.client else "disconnected"
    
    overall_status = "healthy" if db_status == "connected" and api_status == "connected" else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        database=db_status,
        api=api_status,
        timestamp=datetime.datetime.now().isoformat()
    )


@app.post("/api/chatbot", response_model=ChatResponse, tags=["챗봇"])
async def chatbot_detailed(request: ChatRequest):
    """
    챗봇 질문 처리 (상세 정보 포함)
    
    사용자의 질문을 처리하고 답변, 실행된 SQL 쿼리, 실행 시간을 반환합니다.
    
    Args:
        request: 질문 내용을 포함한 요청 객체
    
    Returns:
        ChatResponse: 답변, 쿼리, 실행 시간
    
    Raises:
        HTTPException: 질문이 비어있거나 서버 오류 발생시
    """
    # 입력 검증
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400, 
            detail="질문이 비어있습니다. 질문을 입력해주세요."
        )
    
    # 질문 길이 제한
    if len(request.question) > 500:
        raise HTTPException(
            status_code=400,
            detail="질문이 너무 깁니다. 500자 이내로 입력해주세요."
        )
    
    try:
        print(f"\n[API] 새로운 요청: {request.question}")
        
        # 챗봇 처리: run_in_threadpool을 사용하여 동기 함수를 스레드풀에서 실행 (FIXED)
        result = await run_in_threadpool(chatbot.process_question, request.question)
        
        return ChatResponse(
            answer=result["answer"],
            query_executed=result.get("query_executed"),
            execution_time=result.get("execution_time")
        )
    
    except Exception as e:
        print(f"[ERROR] Chatbot endpoint 오류: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"서버 내부 오류가 발생했습니다: {str(e)}"
        )



@app.post("/api/chatbot/simple", response_model=SimpleChatResponse, tags=["챗봇"])
async def chatbot_simple(request: ChatRequest):
    """
    챗봇 질문 처리 (답변만 반환)
    
    프론트엔드에서 간단하게 답변만 받고 싶을 때 사용합니다.
    
    Args:
        request: 질문 내용을 포함한 요청 객체
    
    Returns:
        SimpleChatResponse: 답변만 포함
    
    Raises:
        HTTPException: 질문이 비어있거나 서버 오류 발생시
    """
    # 입력 검증
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400, 
            detail="질문이 비어있습니다. 질문을 입력해주세요."
        )
    
    # 질문 길이 제한
    if len(request.question) > 500:
        raise HTTPException(
            status_code=400,
            detail="질문이 너무 깁니다. 500자 이내로 입력해주세요."
        )
    
    try:
        print(f"\n[API] 새로운 요청 (Simple): {request.question}")
        
        # 챗봇 처리: run_in_threadpool을 사용하여 동기 함수를 스레드풀에서 실행 (FIXED)
        result = await run_in_threadpool(chatbot.process_question, request.question)
        
        return SimpleChatResponse(answer=result["answer"])
    
    except Exception as e:
        print(f"[ERROR] Chatbot simple endpoint 오류: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"서버 내부 오류가 발생했습니다: {str(e)}"
        )


@app.get("/api/db/info", tags=["데이터베이스"])
async def get_database_info():
    """
    데이터베이스 정보 조회
    
    game 테이블의 구조와 레코드 수를 반환합니다.
    """
    try:
        # 이 함수도 동기 DB 호출을 포함하므로 threadpool에서 실행해야 안전합니다.
        table_info = await run_in_threadpool(db_manager.get_table_info)
        
        if not table_info:
            raise HTTPException(
                status_code=500,
                detail="데이터베이스 정보를 가져올 수 없습니다."
            )
        
        return {
            "table": "game",
            "columns": table_info["columns"],
            "record_count": table_info["record_count"],
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"[ERROR] DB info endpoint 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류: {str(e)}"
        )


# ========================================
# 이벤트 핸들러
# ========================================

@app.on_event("startup")
async def startup_event():
    """서버 시작시 실행"""
    print("\n" + "=" * 60)
    print("🚀 KBO Chatbot API 서버 시작")
    print("=" * 60)
    print(f"Model: {settings.MODEL_NAME}")
    print(f"Database: {settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}")
    print(f"Server: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"Docs: http://localhost:{settings.API_PORT}/docs")
    print("=" * 60 + "\n")
    
    # 연결 테스트
    db_ok = db_manager.test_connection()
    print(f"{'✅' if db_ok else '❌'} 데이터베이스 연결: {'성공' if db_ok else '실패'}")
    
    api_ok = chatbot.test_connection()
    print(f"{'✅' if api_ok else '❌'} API 연결: {'성공' if api_ok else '실패'}\n")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료시 실행"""
    print("\n" + "=" * 60)
    print("🛑 KBO Chatbot API 서버 종료")
    print("=" * 60 + "\n")

# ========================================
# 서버 실행
# ========================================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info"
    )
