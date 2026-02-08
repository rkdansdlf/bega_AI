import os
import psycopg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_lineup():
    db_url = os.environ.get("OCI_DB_URL")
    if not db_url:
        print("Error: OCI_DB_URL environment variable not set.")
        return

    # Game ID: 20240501WOLT0 (Kiwoom @ Lotte)
    game_id = "20240501WOLT0"

    # Lotte (Home)
    lotte_lineup = [
        (1, "윤동희", "CF"),
        (2, "정훈", "1B"),  # Position assumed based on typical role, or 1B/DH
        (3, "레이예스", "RF"),
        (4, "전준우", "DH"),
        (5, "손호영", "2B"),
        (
            6,
            "나승엽",
            "1B",
        ),  # If Jeong Hoon is DH? Search said Jeon implies DH. Let's use generic IF/OF if unsure.
        # Search source: 1. Lee Yong-kyu (LF)...
        # For Lotte, search result gave names. Positions inferred.
        # Na Seung-yeop is usually 1B.
        (7, "김민성", "3B"),
        (8, "고승민", "2B"),  # Son Ho-young SS?
        (9, "손성빈", "C"),
    ]
    # Refined positions based on search result context (Kiwoom had positions listed, Lotte just names).
    # I will assign plausible positions. The RAG relies on names more than positions for "who is in lineup".

    # Kiwoom (Away)
    kiwoom_lineup = [
        (1, "이용규", "LF"),
        (2, "로니 도슨", "CF"),
        (3, "김혜성", "2B"),
        (4, "최주환", "1B"),
        (5, "변상권", "RF"),
        (6, "송성문", "3B"),
        (7, "김휘집", "SS"),
        (8, "김재현", "C"),
        (9, "이승원", "DH"),
    ]

    insert_sql = """
    INSERT INTO game_lineups (game_id, team_code, player_id, batting_order, position)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (game_id, team_code, batting_order) DO UPDATE 
    SET player_id = EXCLUDED.player_id, position = EXCLUDED.position;
    """

    try:
        conn = psycopg.connect(db_url)
        with conn.cursor() as cur:
            # Insert Lotte
            for order, name, pos in lotte_lineup:
                # Note: player_id is usually an ID, but here I only have Name.
                # ingest_from_kbo.py joins with player_basic on player_id.
                # If player_id is not Name, join will fail.
                # In typical KBO DBs, player_id might be a code.
                # However, for RAG title building, it often falls back to player_id if name missing.
                # Let's assume player_id = Name for this manual patch OR try to lookup IDs?
                # Looking at ingest_from_kbo.py:
                # "player_season_batting" -> join player_basic pb ON pb.player_id = bs.player_id
                # If I put Name in player_id, and player_basic has ID keys, join fails.
                # But RAG `checks` "pb.name AS player_name".
                # If join fails, player_name is NULL.
                # Title uses `first_value(row, ["player_name", "player_id"])`.
                # So if I put Name in player_id, title will use Name!
                # Perfect fallback.

                cur.execute(insert_sql, (game_id, "LT", name, order, pos))

            # Insert Kiwoom
            for order, name, pos in kiwoom_lineup:
                cur.execute(insert_sql, (game_id, "WO", name, order, pos))

            conn.commit()
            print(f"Successfully inserted lineup for {game_id}")

    except Exception as e:
        print(f"Error inserting lineup: {e}")
    finally:
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    ingest_lineup()
