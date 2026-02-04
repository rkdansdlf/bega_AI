
import sys
import os
import asyncio
import logging

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings
from app.core.rag import RAGPipeline
import psycopg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_retrieval(query: str):
    settings = get_settings()
    db_url = settings.database_url
    print(f"Connecting to DB for retrieval test...")

    try:
        # RAGPipeline expects a sync connection
        with psycopg.connect(db_url) as conn:
            pipeline = RAGPipeline(settings=settings, connection=conn)
            
            print(f"Querying: {query}")
            docs = await pipeline.retrieve(query, limit=5)
            
            print(f"Found {len(docs)} documents.")
            for i, doc in enumerate(docs):
                meta = doc.get('meta', {})
                print(f"[{i+1}] Table: {doc.get('source_table')} | Title: {doc.get('title')}")
                print(f"    Season: {doc.get('season_year')} | Content snippet: {doc.get('content')[:100]}...")
                
    except Exception as e:
        print(f"Error during retrieval: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "2025년 김도영의 타율은?"
    
    asyncio.run(verify_retrieval(query))
