"""
OCI PostgreSQL 데이터베이스에 HNSW 벡터 인덱스를 생성하는 스크립트입니다.
이 인덱스는 벡터 유사도 검색 속도를 크게 향상시킵니다.
"""
import os
import time
import psycopg2
from dotenv import load_dotenv

load_dotenv("AI/.env")

dsn = os.getenv("OCI_DB_URL")
print(f"Connecting to database...")

try:
    conn = psycopg2.connect(dsn, connect_timeout=30)
    conn.autocommit = True
    
    with conn.cursor() as cur:
        # 1. Check if index already exists
        cur.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'rag_chunks' AND indexname = 'idx_rag_chunks_embedding_hnsw'
        """)
        existing = cur.fetchone()
        
        if existing:
            print("HNSW index already exists. Skipping creation.")
        else:
            print("Creating HNSW index on embedding column...")
            print("This may take several minutes depending on data size...")
            
            start = time.time()
            # Create HNSW index for cosine distance (<=>)
            # m=16: max connections per layer (default)
            # ef_construction=64: size of dynamic candidate list during construction
            cur.execute("""
                CREATE INDEX idx_rag_chunks_embedding_hnsw 
                ON rag_chunks 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
            elapsed = time.time() - start
            print(f"HNSW index created successfully in {elapsed:.2f}s")
        
        # 2. Verify index
        cur.execute("""
            SELECT indexname, indexdef FROM pg_indexes 
            WHERE tablename = 'rag_chunks' AND indexname LIKE '%embedding%'
        """)
        indexes = cur.fetchall()
        print(f"\nCurrent embedding indexes:")
        for idx in indexes:
            print(f"  - {idx[0]}")
        
        # 3. Test query performance
        print("\nTesting vector search performance...")
        vector_str = "[" + ",".join("0.0" for _ in range(1536)) + "]"
        
        cur.execute("EXPLAIN ANALYZE SELECT id, (1 - (embedding <=> %s::vector)) as similarity FROM rag_chunks LIMIT 5", (vector_str,))
        plan = cur.fetchall()
        print("Query plan:")
        for row in plan:
            print(f"  {row[0]}")
    
    conn.close()
    print("\nDatabase optimization complete!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
