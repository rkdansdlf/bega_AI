import asyncio
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.rag import RAGPipeline
from app.config import get_settings
from app.deps import get_connection_pool

# Test queries
TEST_CASES = [
    {
        "query": "김도영 선수 2024년 기록 알려줘",
        "expected_entities": ["김도영", "2024"],
        "expected_table": "player_season_batting"
    },
    {
        "query": "KIA 타이거즈 2024년 우승했어?",
        "expected_entities": ["KIA", "2024", "우승"],
        "expected_table": "awards" # Or game/team stats
    },
    {
        "query": "2024년 평균자책점 1위 누구야?",
        "expected_entities": ["2024", "평균자책점"],
        "expected_table": "player_season_pitching"
    },
    {
        "query": "양현종 통산 기록",
        "expected_entities": ["양현종"],
        "expected_table": "player_season_pitching"
    },
    {
        "query": "LG 트윈스 최근 경기 결과",
        "expected_entities": ["LG"],
        "expected_table": "game"
    }
]

async def run_benchmark():
    settings = get_settings()
    pool = get_connection_pool()
    
    print(f"Running benchmark with {len(TEST_CASES)} queries...")
    print("-" * 60)

    with pool.connection() as conn:
        rag = RAGPipeline(settings=settings, connection=conn)
        
        total_time = 0
        
        for case in TEST_CASES:
            query = case["query"]
            print(f"Query: {query}")
            
            start_time = time.perf_counter()
            # Benchmark baseline path: raw query only.
            # Note: retrieve() also accepts optional filters/entity_filter.
            docs = await rag.retrieve(query, limit=5)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            total_time += elapsed
            
            print(f"  Time: {elapsed:.2f}ms")
            print(f"  Retrieved: {len(docs)} docs")
            
            if docs:
                top_doc = docs[0]
                print(f"  Top Result: {top_doc.get('title', 'No Title')} ({top_doc.get('similarity', 0):.4f})")
                print(f"  Source: {top_doc.get('source_table', 'Unknown')}")
            else:
                print("  Top Result: None")
                
            print("-" * 60)

        avg_time = total_time / len(TEST_CASES)
        print(f"Average Retrieval Time: {avg_time:.2f}ms")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
