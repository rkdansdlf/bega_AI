import os
import psycopg2
from dotenv import load_dotenv

load_dotenv(dotenv_path="AI/.env")
DB_URL = os.getenv("SUPABASE_DB_URL")

try:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    print(f"Connected to: {DB_URL.split('@')[1]}")
    
    cur.execute("SHOW search_path")
    print(f"Search path: {cur.fetchone()[0]}")
    
    print("Searching for rag_chunks in information_schema...")
    cur.execute("SELECT table_schema, table_name FROM information_schema.tables WHERE table_name = 'rag_chunks'")
    found = cur.fetchall()
    for row in found:
        print(f"Found: {row[0]}.{row[1]}")
        
    if not found:
        print("rag_chunks NOT found in information_schema.")
        
    cur.close()
    conn.close()
except Exception as e:
    print(f"Error: {e}")
