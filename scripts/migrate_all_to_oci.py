import psycopg2
from psycopg2.extras import execute_values
from app.config import get_settings
import sys
import json
import traceback

def get_all_tables(cur):
    """Fetch all base tables in the public schema."""
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    return [row[0] for row in cur.fetchall()]

def migrate_tables():
    settings = get_settings()
    
    print(f"Source: {settings.supabase_db_url.split('@')[-1] if settings.supabase_db_url else 'None'}")
    print(f"Destination: {settings.oci_db_url.split('@')[-1]}")
    
    try:
        source_conn = psycopg2.connect(settings.supabase_db_url)
        dest_conn = psycopg2.connect(settings.oci_db_url)
        
        source_cur = source_conn.cursor()
        dest_cur = dest_conn.cursor()
        
        # 1. Fetch all tables and exclude rag_chunks
        all_tables = get_all_tables(source_cur)
        # Exclude rag_chunks as per user request
        tables_to_migrate = [t for t in all_tables if t != 'rag_chunks']
        
        print(f"\nTotal tables discovered: {len(all_tables)}")
        print(f"Total tables to migrate (excluding rag_chunks): {len(tables_to_migrate)}")
        
        for table in tables_to_migrate:
            print(f"\n>>> Migrating table: {table}")
            
            try:
                # Get column info from source
                source_cur.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                    AND table_schema = 'public'
                    ORDER BY ordinal_position
                """)
                columns_info = source_cur.fetchall()
                if not columns_info:
                    print(f"  - No column info found for {table}. Skipping.")
                    continue
                    
                colnames = [col[0] for col in columns_info]
                
                create_columns = []
                for col_name, data_type, is_nullable in columns_info:
                    # Basic type mapping
                    if data_type == "USER-DEFINED":
                        type_str = "TEXT"
                    elif data_type == "ARRAY":
                        # Attempt to find the underlying type? 
                        # For simplicity, fallback to TEXT[] if we can't be sure
                        type_str = "TEXT[]"
                    else:
                        type_str = data_type
                    
                    null_str = "NULL" if is_nullable == "YES" else "NOT NULL"
                    create_columns.append(f'"{col_name}" {type_str} {null_str}')
                
                # Destination: Drop and recreate (Basic approach)
                print(f"  - Recreating table structure in destination...")
                dest_cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
                create_sql = f'CREATE TABLE "{table}" ({", ".join(create_columns)});'
                dest_cur.execute(create_sql)
                dest_conn.commit()
                
                # Copy data from source to destination
                print(f"  - Fetching and inserting data...")
                source_cur.execute(f'SELECT * FROM "{table}"')
                
                batch_size = 1000
                total_rows = 0
                
                while True:
                    batch = source_cur.fetchmany(batch_size)
                    if not batch:
                        break
                    
                    processed_batch = []
                    for row in batch:
                        new_row = []
                        for val in row:
                            # Convert dict to JSON string (for json/jsonb columns)
                            # Do NOT use json.dumps for lists, as psycopg2 handles them as Postgres arrays
                            if isinstance(val, dict):
                                new_row.append(json.dumps(val))
                            else:
                                new_row.append(val)
                        processed_batch.append(tuple(new_row))
                    
                    if processed_batch:
                        col_str = ", ".join([f'"{c}"' for c in colnames])
                        execute_values(
                            dest_cur,
                            f'INSERT INTO "{table}" ({col_str}) VALUES %s',
                            processed_batch,
                        )
                        dest_conn.commit()
                        total_rows += len(processed_batch)
                        print(f"    - Migrated {total_rows} rows so far...")

                print(f"  - Migration for {table} complete. Total rows: {total_rows}")
            
            except Exception as e:
                print(f"  - Error migrating table {table}: {e}")
                dest_conn.rollback()
                # Continue with next table
                continue

        print("\n==========================================")
        print("Data migration finished successfully (excluding rag_chunks).")
        print("==========================================")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Migration failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'source_conn' in locals():
            source_conn.close()
        if 'dest_conn' in locals():
            dest_conn.close()

if __name__ == "__main__":
    migrate_tables()
