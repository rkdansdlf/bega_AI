import os
import psycopg2
from dotenv import load_dotenv

load_dotenv(dotenv_path="AI/.env")

DB_URL = os.getenv("SUPABASE_DB_URL")

try:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    print("--- player_basic columns ---")
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'player_basic'")
    for row in cur.fetchall():
        print(row[0])
        
    print("\n--- teams columns ---")
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'teams'")
    for row in cur.fetchall():
        print(row[0])

    cur.close()
    conn.close()
except Exception as e:
    print(f"Error: {e}")
