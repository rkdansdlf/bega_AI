import psycopg2
import json
import os
from dotenv import load_dotenv

def verify_data():
    load_dotenv()
    url = os.getenv("OCI_DB_URL")
    if not url:
        print("OCI_DB_URL not found in .env")
        return

    try:
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        
        # Check if table exists
        cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'rag_chunks')")
        table_exists = cur.fetchone()[0]
        print(f"Table 'rag_chunks' exists: {table_exists}")
        
        if table_exists:
            # Check row count
            cur.execute("SELECT COUNT(*) FROM rag_chunks")
            count = cur.fetchone()[0]
            print(f"Row count in 'rag_chunks': {count}")
            
            if count > 0:
                # Get a sample row to check metadata
                cur.execute("SELECT id, source_table, title, meta FROM rag_chunks LIMIT 5")
                rows = cur.fetchall()
                print("\nSample Data (first 5 rows):")
                for row in rows:
                    rid, source, title, meta = row
                    print(f"ID: {rid}, Source: {source}, Title: {title}")
                    print(f"Metadata: {json.dumps(meta, indent=2, ensure_ascii=False)}")
                    print("-" * 20)
            else:
                print("No data found in 'rag_chunks'.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error connecting to OCI DB: {e}")

if __name__ == "__main__":
    verify_data()
