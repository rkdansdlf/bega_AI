"""AI/docs 디렉토리의 마크다운 파일들을 읽어 데이터베이스에 인덱싱하는 스크립트.

이 스크립트는 지정된 디렉토리 내의 모든 .md 파일을 순회하며,
각 파일의 내용을 청크로 분할하고, 임베딩을 생성한 후,
PostgreSQL 데이터베이스의 rag_chunks 테이블에 저장합니다.
"""
import asyncio
import os
from typing import Generator

import psycopg2
from psycopg2.extensions import connection as PgConnection

# 프로젝트의 루트 디렉토리를 sys.path에 추가하여 모듈을 임포트할 수 있도록 함
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import get_settings
from app.core.chunking import smart_chunks
from app.core.embeddings import async_embed_texts

# --- 설정 ---
# 인덱싱할 최상위 디렉토리 목록
TARGET_DIRS = ["AI/docs/kbo_rulebook", "AI/docs/kbo_knowledge", "AI/docs"]
# 인덱싱할 파일 확장자
FILE_EXTENSION = ".md"
# 데이터베이스에 저장될 때 사용할 출처(source) 테이블 이름
SOURCE_TABLE_NAME = "markdown_docs"

def get_db_connection() -> Generator[PgConnection, None, None]:
    """데이터베이스 연결을 생성하는 제너레이터 함수."""
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url)
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()

def find_markdown_files(directories: list[str]) -> list[str]:
    """지정된 디렉토리에서 모든 마크다운 파일의 경로를 찾습니다."""
    markdown_files = []
    for directory in directories:
        if os.path.isfile(directory) and directory.endswith(FILE_EXTENSION):
            markdown_files.append(directory)
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(FILE_EXTENSION):
                    markdown_files.append(os.path.join(root, file))
    print(f"총 {len(markdown_files)}개의 마크다운 파일을 찾았습니다.")
    return markdown_files

import re
from typing import Optional

def infer_season_year(path: str) -> int:
    """Infer the season year from the file path. Looks for years between 1982 and 2029."""
    match = re.search(r'(19[8-9]\d|20[0-2]\d)', path) # Covers 1980-2029
    if match:
        year = int(match.group(1))
        if 1982 <= year <= 2029: # Plausible KBO year range
            return year
    return 0 # Default to 0 if no year is found to satisfy NOT NULL constraint

async def main():
    """메인 인덱싱 실행 함수."""
    settings = get_settings()
    files_to_ingest = find_markdown_files(TARGET_DIRS)
    
    conn_generator = get_db_connection()
    conn = next(conn_generator)

    total_chunks_ingested = 0

    try:
        with conn.cursor() as cur:
            for file_path in files_to_ingest:
                print(f"'{file_path}' 파일 처리 중...")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    print(f"  - 파일 읽기 오류: {e}")
                    continue

                # 파일 내용이 비어있으면 건너뜀
                if not content.strip():
                    print("  - 내용이 비어있어 건너뜁니다.")
                    continue
                
                # Infer metadata from path
                season_year = infer_season_year(file_path)
                league_type_code = 0 # Default to 0 to satisfy NOT NULL constraint

                # 청크 분할 및 임베딩 생성
                chunks = smart_chunks(content)
                embeddings = await async_embed_texts(chunks, settings)
                
                # 데이터베이스에 저장
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    vector_literal = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
                    # 고유 ID로 파일 경로와 청크 인덱스 사용
                    source_row_id = f"{file_path}_{i}"
                    
                    cur.execute(
                        """
                        INSERT INTO rag_chunks (season_year, league_type_code, source_table, source_row_id, title, content, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (source_table, source_row_id)
                        DO UPDATE SET 
                            content = EXCLUDED.content, 
                            embedding = EXCLUDED.embedding, 
                            season_year = EXCLUDED.season_year,
                            league_type_code = EXCLUDED.league_type_code,
                            updated_at = now()
                        """,
                        (
                            season_year,
                            league_type_code,
                            SOURCE_TABLE_NAME,
                            source_row_id,
                            os.path.basename(file_path), # 제목으로 파일명 사용
                            chunk,
                            vector_literal,
                        ),
                    )
                print(f"  - {len(chunks)}개의 청크를 데이터베이스에 저장했습니다. (Year: {season_year})")
                total_chunks_ingested += len(chunks)

    except Exception as e:
        print(f"데이터베이스 작업 중 오류 발생: {e}")
    finally:
        conn.close()
        print(f"\n총 {total_chunks_ingested}개의 청크를 성공적으로 인덱싱했습니다.")
        print("인덱싱 작업 완료.")

if __name__ == "__main__":
    # Python 3.7+ 에서는 asyncio.run()을 사용하여 비동기 함수를 실행할 수 있습니다.
    asyncio.run(main())
