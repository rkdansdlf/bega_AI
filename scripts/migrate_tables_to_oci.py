
import psycopg2
from psycopg2.extras import execute_values
from app.config import get_settings
import sys

def migrate_tables():
    settings = get_settings()
    
    # Tables to migrate in dependency order
    tables = [
        "kbo_seasons",
        "teams",
        "player_basic",
        "player_season_batting",
        "player_season_pitching",
        "game", 
        "game_batting_stats",
        "game_pitching_stats"
    ]
    
    try:
        source_conn = psycopg2.connect(settings.supabase_db_url)
        dest_conn = psycopg2.connect(settings.database_url)
        
        source_cur = source_conn.cursor()
        dest_cur = dest_conn.cursor()
        
        for table in tables:
            print(f"Migrating table: {table}")
            
            # 1. Get Schema from Source
            source_cur.execute(f"SELECT * FROM {table} LIMIT 0")
            colnames = [desc[0] for desc in source_cur.description]
            coltypes = []
            
            # Simple check to verify table exists in source
            print(f"  - Columns: {', '.join(colnames)}")
            
            # 2. Create Table in Destination (IF NOT EXISTS)
            # We'll use a simplified approach: fetch create statement or just rely on basic types
            # Ideally we want the exact schema. For now, let's try to infer or check if we can dump/restore schema easily.
            # Since we can't easily get full DDL via python without external tools like pg_dump, 
            # we will attempt to CREATE TABLE based on selected columns assuming standard types, 
            # OR better, since this is a one-off, we hope the destination table might NOT exist and we need to create it.
            
            # Actually, to generate DDL dynamically in Python is complex. 
            # A safer, more robust way for this specific environment (headless):
            # Read column types from information_schema
            
            source_cur.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
            """)
            columns_info = source_cur.fetchall()
            
            create_columns = []
            for col_name, data_type, is_nullable in columns_info:
                # Map simple types if needed, but Postgres to Postgres usually matches
                # Array types in information_schema like 'ARRAY' need handling
                if data_type == 'ARRAY':
                     # Need to find element type, but for now let's try text[] as fallback or just TEXT[]
                     # Fetching real type via pg_typeof might be better but let's assume standard arrays
                     # A quick hack for arrays:
                     type_str = "TEXT[]" # Generic fallback
                     # Let's verify specific known array columns if needed, or refine this.
                     # But for standard tables like teams, players, stats, arrays are rare except specific fields.
                     pass
                else:
                    type_str = data_type
                
                # Handle specific type conversions if known issues arise (e.g. user-defined types)
                if data_type == 'USER-DEFINED':
                     type_str = 'TEXT' # Fallback for enums if type doesn't exist in dest
                
                null_str = "NULL" if is_nullable == 'YES' else "NOT NULL"
                create_columns.append(f'"{col_name}" {type_str} {null_str}')
            
            # Since we don't have the exact DDL, and recreating constraints is hard, 
            # we will create a basic schema.
            # WARNING: This might fail on complex types or constraints.
            
            # Better Approach for this specific task:
            # Drop table if exists to ensure clean state (or TRUNCATE).
            dest_cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            
            # Construct CREATE TABLE
            # Note: This is a BEST EFFORT schema copy. 
            create_sql = f"CREATE TABLE {table} ({', '.join(create_columns)});"
            
            # Fix for known issue: vector type or specific extensions? 
            # These tables are standard relational data.
            
            # Let's assume schema copy might be tricky for all columns generic logic.
            # Instead, let's just fetch all data and use `pandas` or `csv` logic? 
            # No, let's stick to cursor copy.
            
            # Let's try to infer schema from the cursor description after a SELECT * LIMIT 0
             
            # Actually, the simplest way is to use `pg_dump -s -t table source | psql dest` via shell,
            # but we can't easily pipe passwords in shell command here safely or handle network.
            
            # Retry Schema:
            # We will construct a basic CREATE TABLE.
            # If it fails, we will print error.
            
            try:
                # We need to handle primary keys? For RAG lookup, maybe not strictly enforced, 
                # but good for performance. we will skip PK for now to ensure data copy success.
                print(f"  - Creating table {table} in destination...")
                # Mapping common postgres types
                
                dest_cur.execute(create_sql)
                dest_conn.commit()
            except Exception as e:
                print(f"  - Error creating table: {e}")
                dest_conn.rollback()
                continue

            # 3. Data Copy
            print(f"  - Copying data...")
            source_cur.execute(f"SELECT * FROM {table}")
            
            rows = []
            batch_size = 1000
            
            import json
            
            while True:
                batch = source_cur.fetchmany(batch_size)
                if not batch:
                    break
                
                # Pre-process batch to convert dict/lists to json string if needed
                processed_batch = []
                for row in batch:
                    new_row = []
                    for val in row:
                        if isinstance(val, dict):
                            new_row.append(json.dumps(val))
                        else:
                            new_row.append(val)
                    processed_batch.append(tuple(new_row))
                
                rows.extend(processed_batch)
                
                if len(rows) >= 5000:
                    execute_values(dest_cur, 
                        f"INSERT INTO {table} ({', '.join(colnames)}) VALUES %s", 
                        rows)
                    dest_conn.commit()
                    print(f"    - Inserted {len(rows)} rows...")
                    rows = []
            
            if rows:
                 execute_values(dest_cur, 
                        f"INSERT INTO {table} ({', '.join(colnames)}) VALUES %s", 
                        rows)
                 dest_conn.commit()
                 print(f"    - Inserted {len(rows)} rows...")
            
            print(f"  - Table {table} migration complete.")

        source_cur.close()
        dest_cur.close()
        source_conn.close()
        dest_conn.close()
        print("Migration finished successfully.")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    migrate_tables()
