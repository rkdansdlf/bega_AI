import os
import psycopg2
from dotenv import load_dotenv

# Load env from root
load_dotenv(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
)

DB_URL = os.getenv("SUPABASE_DB_URL")


def migrate_schema():
    print(f"Connecting to DB...")
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        # Check if table exists in public
        cur.execute("SELECT to_regclass('public.password_reset_tokens')")
        public_table = cur.fetchone()[0]

        # Check if table exists in security
        cur.execute("SELECT to_regclass('security.password_reset_tokens')")
        security_table = cur.fetchone()[0]

        if public_table:
            if security_table:
                print(
                    "Table 'password_reset_tokens' exists in BOTH 'public' and 'security' schemas."
                )
                print("Skipping automatic migration to avoid data loss.")
            else:
                print("Moving 'public.password_reset_tokens' to 'security' schema...")
                cur.execute(
                    "ALTER TABLE public.password_reset_tokens SET SCHEMA security"
                )
                conn.commit()
                print("Migration successful!")
        else:
            if security_table:
                print(
                    "Table 'password_reset_tokens' already exists in 'security' schema. No action needed."
                )
            else:
                print("Table 'password_reset_tokens' found in NEITHER schema.")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    migrate_schema()
