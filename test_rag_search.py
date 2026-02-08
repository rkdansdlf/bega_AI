#!/usr/bin/env python3
"""
RAG 검색 단독 테스트
실행: python test_rag_search.py "김도영 선수 성적"
"""

import sys
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_rag_search(question: str):
    """RAG 검색 테스트"""
    
    # 필요한 모듈 임포트
    sys.path.insert(0, os.path.dirname(__file__))
    
    from app.config import get_settings
    from app.core.rag import RAGPipeline
    from app.deps import get_db_connection
    
    settings = get_settings()
    
    # DB 연결
    conn = next(get_db_connection())
    
    # RAG 파이프라인 생성
    pipeline = RAGPipeline(settings=settings, connection=conn)
    
    print(f"\n🔍 질문: {question}\n")
    print("="*60)
    
    # 검색 실행
    try:
        docs = await pipeline.retrieve(question, limit=5)
        
        print(f"\n✅ 검색 결과: {len(docs)}개 문서 발견\n")
        
        for i, doc in enumerate(docs, 1):
            print(f"[{i}] {doc.get('title', 'N/A')}")
            print(f"    출처: {doc.get('source_table', 'N/A')}")
            print(f"    유사도: {doc.get('similarity', 0):.4f}")
            print(f"    내용 미리보기: {doc.get('content', '')[:100]}...")
            print()
        
    except Exception as e:
        print(f"❌ 검색 실패: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "김도영 선수 2025년 성적"
    
    asyncio.run(test_rag_search(question))