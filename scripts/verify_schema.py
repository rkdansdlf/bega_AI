
import os
import psycopg2
from dotenv import load_dotenv

# Load env from root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

DB_URL = os.getenv('SUPABASE_DB_URL')

def check_schema():
    print(f"Connecting to DB...")
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        # Check security schema tables
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'security'")
        security_tables = [row[0] for row in cur.fetchall()]
        print(f"Tables in 'security' schema: {security_tables}")
        
        # Check public schema tables (just relevant ones if any remain)
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'password_reset_tokens'")
        public_tokens = cur.fetchone()
        
        if 'password_reset_tokens' in security_tables:
            print("CONFIRMED: 'password_reset_tokens' is in 'security' schema.")
        else:
            print("WARNING: 'password_reset_tokens' NOT found in 'security' schema.")

        if public_tokens:
             print("WARNING: 'password_reset_tokens' STILL exists in 'public' schema.")
        else:
             print("CONFIRMED: 'password_reset_tokens' is NO LONGER in 'public' schema.")

        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_schema()
