
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def verify_advanced_logic():
    conn_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not conn_url:
        print("Error: DATABASE_URL not found")
        return

    try:
        conn = psycopg2.connect(conn_url)
        cur = conn.cursor()
        
        year = 2025
        
        # Test Query (Exact logic from database_query.py)
        query = """
            WITH team_pitching_raw AS (
                SELECT 
                    team_code,
                    -- 선발 요건: GS > 0 이거나 QS > 0 이거나 경기당 3이닝 이상 투구
                    SUM(CASE WHEN (COALESCE(games_started, 0) > 0 OR COALESCE(quality_starts, 0) > 0 OR (innings_pitched / NULLIF(games, 0)) >= 3) THEN innings_pitched ELSE 0 END) as starter_ip,
                    SUM(CASE WHEN NOT (COALESCE(games_started, 0) > 0 OR COALESCE(quality_starts, 0) > 0 OR (innings_pitched / NULLIF(games, 0)) >= 3) THEN innings_pitched ELSE 0 END) as bullpen_ip,
                    SUM(innings_pitched) as total_ip,
                    SUM(quality_starts) as total_qs,
                    SUM(CASE WHEN (COALESCE(games_started, 0) > 0 OR COALESCE(quality_starts, 0) > 0 OR (innings_pitched / NULLIF(games, 0)) >= 3) THEN 1 ELSE 0 END) as total_gs,
                    ROUND(AVG(era)::numeric, 2) as avg_era
                FROM player_season_pitching
                WHERE season = %s
                GROUP BY team_code
            ),
            fatigue_calc AS (
                SELECT
                    *,
                    ROUND((bullpen_ip / NULLIF(total_ip, 0) * 100)::numeric, 1) as bullpen_share,
                    ROUND(((total_qs::numeric / NULLIF(total_gs, 0)) * 100)::numeric, 1) as qs_rate
                FROM team_pitching_raw
            ),
            ranked_pitching AS (
                SELECT 
                    *,
                    RANK() OVER (ORDER BY avg_era ASC) as era_rank,
                    RANK() OVER (ORDER BY bullpen_share DESC) as load_rank
                FROM fatigue_calc
            )
            SELECT * FROM ranked_pitching ORDER BY load_rank ASC;
        """
        
        cur.execute(query, (year,))
        rows = cur.fetchall()
        
        print(f"--- 2025 Season Bullpen Share (Advanced Fallback Logic) ---")
        found_target = False
        
        # Column indices based on SELECT *:
        # 0: team_code
        # 1: starter_ip
        # 2: bullpen_ip
        # 3: total_ip
        # 4: total_qs
        # 5: total_gs
        # 6: avg_era
        # 7: bullpen_share
        # 8: qs_rate
        # 9: era_rank
        # 10: load_rank
        
        for row in rows:
            is_target = "LG" in row[0]
            prefix = ">> " if is_target else "   "
            starter_ip = row[1]
            bullpen_share = row[7]
            print(f"{prefix}Team: {row[0]:<5} | Starter IP: {starter_ip:>7.1f} | Share: {bullpen_share}%")
            if is_target: found_target = True
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_advanced_logic()
