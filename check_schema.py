import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()


def list_columns():
    conn_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    try:
        conn = psycopg2.connect(conn_url)
        cur = conn.cursor()

        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'player_season_pitching'"
        )
        columns = cur.fetchall()
        print("Columns in player_season_pitching:")
        for col in columns:
            print(f"- {col[0]} ({col[1]})")

        cur.execute(
            "SELECT quality_starts, COUNT(*) FROM player_season_pitching WHERE season = 2025 GROUP BY quality_starts"
        )
        qs_stats = cur.fetchall()
        print("\nQuality Starts distribution for 2025:")
        for s in qs_stats:
            print(f"- QS: {s[0]} | Count: {s[1]}")

        cur.execute("""
            SELECT 
                CASE WHEN (innings_pitched / NULLIF(games, 0)) >= 3 THEN 'Potential Starter' ELSE 'Reliever' END as type,
                COUNT(*)
            FROM player_season_pitching 
            WHERE season = 2025 
            GROUP BY 1
        """)
        type_stats = cur.fetchall()
        print("\nInnings-per-game distribution (IP/G >= 3):")
        for s in type_stats:
            print(f"- Type: {s[0]} | Count: {s[1]}")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    list_columns()
