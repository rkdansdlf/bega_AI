# check_embedding.py (수정 버전)
import subprocess
import sys

script = """
import psycopg2
import os

conn = psycopg2.connect(os.getenv('SUPABASE_DB_URL'))
cur = conn.cursor()

# 테이블별 임베딩 생성률 확인
cur.execute('''
    SELECT 
        source_table,
        COUNT(*) as total,
        COUNT(embedding) as with_embedding,
        ROUND(COUNT(embedding)::decimal / COUNT(*) * 100, 1) as percentage,
        MIN(created_at) as first_created,
        MAX(created_at) as last_created
    FROM rag_chunks
    GROUP BY source_table
    ORDER BY percentage ASC
''')

print("\\n" + "="*80)
print("Table Embedding Status")  # 한글 대신 영어로
print("="*80)
print(f"{'Table':<25} {'Total':<8} {'Embedded':<8} {'Rate':<8} {'First Created':<20}")
print("-"*80)

for row in cur.fetchall():
    table, total, with_emb, pct, first, last = row
    print(f"{table:<25} {total:<8,} {with_emb:<8,} {pct:>6}% {str(first)[:19]:<20}")

print("="*80)

# 날짜별 수집 현황
cur.execute('''
    SELECT 
        DATE(created_at) as date,
        COUNT(*) as count,
        COUNT(embedding) as with_embedding
    FROM rag_chunks
    GROUP BY DATE(created_at)
    ORDER BY date DESC
    LIMIT 10
''')

print("\\nDaily Collection Status (Last 10 days)")
print("-"*60)
for date, count, with_emb in cur.fetchall():
    print(f"{date}: {count:,} collected, {with_emb:,} embedded")

cur.close()
conn.close()
"""

# UTF-8 인코딩 명시
result = subprocess.run(
    ["docker", "exec", "bega-ai-chatbot-1", "python3", "-c", script],
    capture_output=True,
    text=True,
    encoding='utf-8',  # 인코딩 명시
    errors='replace'   # 에러 무시
)

print(result.stdout)
if result.stderr:
    print("Warnings:", result.stderr)