import os
import psycopg2
from dotenv import load_dotenv

load_dotenv(dotenv_path="AI/.env")

DB_URL = os.getenv("SUPABASE_DB_URL")

dummy_regulation = {
    "season_year": 2025,
    "team_id": None,
    "source_table": "kbo_regulations",
    "source_row_id": "reg_dummy_001",
    "title": "KBO 타이브레이크 규정 (예시)",
    "content": "정규시즌 1위가 2개 팀일 경우 무승부 없이 타이브레이크 경기를 거행하여 정규시즌 우승팀을 가린다. 경기는 1위 결정전으로 불리며, 승리 팀이 한국시리즈에 직행한다.",
    "meta": {
        "category": "basic",
        "regulation_code": "01-1",
        "document_type": "regulation"
    }
}

try:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # Check if dummy exists
    cur.execute("SELECT id FROM rag_chunks WHERE source_row_id = %s", (dummy_regulation["source_row_id"],))
    if cur.fetchone():
        print("Dummy regulation already exists.")
    else:
        print("Inserting dummy regulation...")
        import json
        cur.execute("""
            INSERT INTO rag_chunks (season_year, source_table, source_row_id, title, content, meta)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            dummy_regulation["season_year"],
            dummy_regulation["source_table"],
            dummy_regulation["source_row_id"],
            dummy_regulation["title"],
            dummy_regulation["content"],
            json.dumps(dummy_regulation["meta"])
        ))
        conn.commit()
        print("Inserted successfully.")

    cur.close()
    conn.close()
except Exception as e:
    print(f"Error: {e}")
