import os
import psycopg2
from dotenv import load_dotenv

load_dotenv(dotenv_path="AI/.env")
DB_URL = os.getenv("SUPABASE_DB_URL")

try:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT table_schema, table_name 
        FROM information_schema.tables 
        WHERE table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name;
    """)
    print("--- Tables ---")
    for row in cur.fetchall():
        print(f"{row[0]}.{row[1]}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
