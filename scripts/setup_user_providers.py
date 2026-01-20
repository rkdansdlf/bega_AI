
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

DB_URL = os.getenv('SUPABASE_DB_URL')

def setup_user_providers():
    print("Connecting to DB...")
    with psycopg2.connect(DB_URL) as conn:
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

if __name__ == "__main__":
    setup_user_providers()
