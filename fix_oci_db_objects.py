import os
import psycopg2
import json
from dotenv import load_dotenv

load_dotenv(dotenv_path="AI/.env")

# Use OCI DB URL as used by the application
DB_URL = os.getenv("OCI_DB_URL")

create_view_sql = """
CREATE OR REPLACE VIEW v_team_rank_all AS
WITH team_stats AS (
    SELECT 
        ks.season_year,
        team,
        SUM(CASE WHEN winning_team = team THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN winning_team IS NOT NULL AND winning_team != team THEN 1 ELSE 0 END) as losses,
        SUM(CASE WHEN winning_team IS NULL AND home_score = away_score THEN 1 ELSE 0 END) as draws
    FROM (
        SELECT season_id, home_team as team, winning_team, home_score, away_score 
        FROM game 
        WHERE game_status = 'COMPLETED'
        
        UNION ALL
        
        SELECT season_id, away_team as team, winning_team, home_score, away_score 
        FROM game 
        WHERE game_status = 'COMPLETED'
    ) all_games
    JOIN kbo_seasons ks ON all_games.season_id = ks.season_id
    WHERE ks.league_type_code = '0' -- 정규시즌만
    GROUP BY ks.season_year, team
),
ranked_stats AS (
    SELECT 
        season_year,
        team as team_id,
        team as team_name,
        wins,
        losses,
        draws,
        RANK() OVER (PARTITION BY season_year ORDER BY (wins::float / NULLIF(wins + losses, 0)) DESC) as season_rank
    FROM team_stats
)
SELECT 
    season_year,
    team_id,
    team_name,
    season_rank,
    wins,
    losses,
    draws
FROM ranked_stats;
"""

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
    print(f"Connecting to OCI DB: {DB_URL.split('@')[1]}...")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # 1. Create View
    print("Creating v_team_rank_all view...")
    cur.execute(create_view_sql)
    print("View created.")
    
    # 2. Insert Dummy Regulation
    print("Checking for dummy regulation...")
    cur.execute("SELECT id FROM rag_chunks WHERE source_row_id = %s", (dummy_regulation["source_row_id"],))
    if cur.fetchone():
        print("Dummy regulation already exists.")
    else:
        print("Inserting dummy regulation...")
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
        print("Dummy regulation inserted.")

    conn.commit()
    cur.close()
    conn.close()
    print("All OCI DB fixes applied successfully.")

except Exception as e:
    print(f"Error: {e}")
