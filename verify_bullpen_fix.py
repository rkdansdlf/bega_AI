import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()


def verify_bullpen_fix():
    conn_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not conn_url:
        print("Error: DATABASE_URL not found")
        return

    try:
        conn = psycopg2.connect(conn_url)
        cur = conn.cursor()

        year = 2025
        team_name = "LG"  # LG 트윈스

        # Test Query (same logic as in database_query.py)
        query = """
            SELECT 
                team_code,
                SUM(CASE WHEN COALESCE(games_started, 0) > 0 THEN innings_pitched ELSE 0 END) as starter_ip,
                SUM(CASE WHEN COALESCE(games_started, 0) = 0 THEN innings_pitched ELSE 0 END) as bullpen_ip,
                SUM(innings_pitched) as total_ip,
                ROUND((SUM(CASE WHEN COALESCE(games_started, 0) = 0 THEN innings_pitched ELSE 0 END) / NULLIF(SUM(innings_pitched), 0) * 100)::numeric, 1) as bullpen_share
            FROM player_season_pitching
            WHERE season = %s
            GROUP BY team_code
            ORDER BY bullpen_share DESC;
        """

        cur.execute(query, (year,))
        rows = cur.fetchall()

        print(f"--- 2025 Season Bullpen Share (Post-Fix) ---")
        found_target = False
        for row in rows:
            is_target = "LG" in row[0]
            prefix = ">> " if is_target else "   "
            print(
                f"{prefix}Team: {row[0]:<10} | Starter IP: {row[1]:>7.1f} | Bullpen IP: {row[2]:>7.1f} | Share: {row[4]}%"
            )
            if is_target:
                found_target = True

        if not found_target:
            print(f"\nWarning: LG not found in 2025 data.")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    verify_bullpen_fix()
