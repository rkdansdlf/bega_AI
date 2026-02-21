import os
import psycopg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_table():
    db_url = os.environ.get("POSTGRES_DB_URL")
    if not db_url:
        print("Error: POSTGRES_DB_URL environment variable not set.")
        return

    create_sql = """
    CREATE TABLE IF NOT EXISTS game_lineups (
        game_id VARCHAR(20) NOT NULL,
        team_code VARCHAR(10) NOT NULL,
        player_id VARCHAR(20) NOT NULL,
        batting_order INT NOT NULL,
        position VARCHAR(10),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (game_id, team_code, batting_order)
    );
    
    CREATE INDEX IF NOT EXISTS idx_game_lineups_game_id ON game_lineups(game_id);
    CREATE INDEX IF NOT EXISTS idx_game_lineups_team_code ON game_lineups(team_code);
    """

    try:
        conn = psycopg.connect(db_url)
        with conn.cursor() as cur:
            cur.execute(create_sql)
            conn.commit()
            print("Table 'game_lineups' created successfully.")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    create_table()
