import os
import argparse
from dotenv import load_dotenv

load_dotenv(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
)

POSTGRES_DB_URL = os.getenv("POSTGRES_DB_URL")
LEGACY_SOURCE_DB_URL = os.getenv("SUPABASE_DB_URL")
DB_URL = POSTGRES_DB_URL or LEGACY_SOURCE_DB_URL


def setup_user_providers():
    if not DB_URL:
        raise RuntimeError(
            "POSTGRES_DB_URL is not configured. (SUPABASE_DB_URL fallback is deprecated)"
        )
    if not POSTGRES_DB_URL and LEGACY_SOURCE_DB_URL:
        print("[WARN] SUPABASE_DB_URL is deprecated. Use POSTGRES_DB_URL instead.")

    try:
        import psycopg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "psycopg is required to run setup_user_providers. "
            "Install dependencies (e.g. pip install -r requirements.txt) and retry."
        ) from exc

    print("Connecting to DB...")
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            # 1. Create table
            print("Creating 'security.user_providers' table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS security.user_providers (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES security.users(id) ON DELETE CASCADE,
                    provider VARCHAR(20) NOT NULL,
                    providerid VARCHAR(255) NOT NULL,
                    connected_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(provider, providerid),
                    UNIQUE(user_id, provider)  -- One provider account per user (e.g. one Google account per user)
                );
            """)

            # 2. Migrate existing data
            print("Migrating existing provider data...")
            cur.execute("""
                INSERT INTO security.user_providers (user_id, provider, providerid)
                SELECT id, provider, providerid
                FROM security.users
                WHERE provider IS NOT NULL 
                  AND provider != 'LOCAL'
                  AND provider != '' 
                  AND providerid IS NOT NULL
                ON CONFLICT (provider, providerid) DO NOTHING;
            """)

            inserted = cur.rowcount
            print(f"Migrated {inserted} records.")

            conn.commit()
            print("Setup successful!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Legacy provider columns migration helper for user providers."
    )
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    setup_user_providers()


if __name__ == "__main__":
    main()
