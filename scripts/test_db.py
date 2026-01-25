import os
import time
import psycopg2
from dotenv import load_dotenv

load_dotenv("AI/.env")

dsn = os.getenv("OCI_DB_URL")
print(f"Connecting to: {dsn[:15]}...")

try:
    conn = psycopg2.connect(dsn, connect_timeout=10)
    conn.autocommit = True

    # 1536 dimension zero vector
    vector_str = "[" + ",".join("0.0" for _ in range(1536)) + "]"

    sql = f"""
    SELECT id, (1 - (embedding <=> '{vector_str}'::vector)) as similarity
    FROM rag_chunks
    LIMIT 1
    """

    print("Executing vector search...")
    start_q = time.time()
    with conn.cursor() as cur:
        cur.execute(sql)
        res = cur.fetchall()
    print(f"Query result count: {len(res)}, Time: {time.time() - start_q:.2f}s")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
