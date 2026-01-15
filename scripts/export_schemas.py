import psycopg2
import csv
import os
from app.config import get_settings

def export_schemas_to_csv():
    settings = get_settings()
    
    # List of tables identified in migrate_tables_to_oci.py
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
    
    output_dir = "../docs"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Connect to Source (Supabase) to get the original schema
        conn = psycopg2.connect(settings.supabase_db_url)
        cur = conn.cursor()
        
        for table in tables:
            print(f"Exporting schema for: {table}")
            
            cur.execute(f"""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
            """)
            
            columns = cur.fetchall()
            
            if not columns:
                print(f"Warning: No columns found for table {table}")
                continue
                
            csv_path = os.path.join(output_dir, f"{table}_schema.csv")
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Header
                writer.writerow(['Column Name', 'Data Type', 'Nullable', 'Default'])
                
                for col in columns:
                    writer.writerow(col)
                    
            print(f"  - Saved to {csv_path}")
            
        cur.close()
        conn.close()
        print("Schema export complete.")
        
    except Exception as e:
        print(f"Error exporting schemas: {e}")

if __name__ == "__main__":
    export_schemas_to_csv()
