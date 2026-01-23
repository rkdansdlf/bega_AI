
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def verify_qs_logic():
    conn_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not conn_url:
        print("Error: DATABASE_URL not found")
        return

    try:
        conn = psycopg2.connect(conn_url)
        cur = conn.cursor()
        
        year = 2025
        team_code = 'LG' # Name to Code handled in app, usually LG is LG
        
        print(f"--- Testing QS Rate Logic for {team_code} {year} ---")

        # 1. Reproduce Error (Current Logic)
        # Counts rows (players) as starts -> Denominator too small -> % > 100%
        bad_query = """
            WITH team_pitching_raw AS (
                SELECT 
                    team_code,
                    SUM(quality_starts) as total_qs,
                    -- BUG: Counts players (rows) instead of games
                    SUM(CASE WHEN (COALESCE(games_started, 0) > 0 OR COALESCE(quality_starts, 0) > 0 OR (innings_pitched / NULLIF(games, 0)) >= 3) THEN 1 ELSE 0 END) as total_gs_bad_count
                FROM player_season_pitching
                WHERE season = %s AND team_code = %s
                GROUP BY team_code
            )
            SELECT 
                total_qs, 
                total_gs_bad_count,
                ROUND(((total_qs::numeric / NULLIF(total_gs_bad_count, 0)) * 100)::numeric, 1) as bad_qs_rate
            FROM team_pitching_raw;
        """
        
        cur.execute(bad_query, (year, team_code))
        bad_row = cur.fetchone()
        if bad_row:
            print(f"[Current Code] Total QS: {bad_row[0]} | Denom (Players): {bad_row[1]} | Rate: {bad_row[2]}%")
        
        # 2. Proposed Fix
        # Sums 'games_started' or 'games' if GS is missing
        fix_query = """
            WITH team_pitching_raw AS (
                SELECT 
                    team_code,
                    SUM(quality_starts) as total_qs,
                    -- FIX: Sums actual games
                    SUM(
                        CASE 
                            WHEN COALESCE(games_started, 0) > 0 THEN games_started
                            WHEN (COALESCE(quality_starts, 0) > 0 OR (innings_pitched / NULLIF(games, 0)) >= 3) THEN games 
                            ELSE 0 
                        END
                    ) as total_gs_fixed
                FROM player_season_pitching
                WHERE season = %s AND team_code = %s
                GROUP BY team_code
            )
            SELECT 
                total_qs, 
                total_gs_fixed,
                ROUND(((total_qs::numeric / NULLIF(total_gs_fixed, 0)) * 100)::numeric, 1) as fixed_qs_rate
            FROM team_pitching_raw;
        """
        
        cur.execute(fix_query, (year, team_code))
        fix_row = cur.fetchone()
        if fix_row:
            print(f"[Proposed Fix] Total QS: {fix_row[0]} | Denom (Games): {fix_row[1]} | Rate: {fix_row[2]}%")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_qs_logic()
