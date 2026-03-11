"""AI/docs 디렉토리의 마크다운 파일들을 읽어 데이터베이스에 인덱싱하는 스크립트.

이 스크립트는 지정된 디렉토리 내의 모든 .md 파일을 순회하며,
각 파일의 내용을 청크로 분할하고, 임베딩을 생성한 후,
PostgreSQL 데이터베이스의 rag_chunks 테이블에 저장합니다.
"""

import asyncio
import json
import os
from typing import Generator
from datetime import datetime
from pathlib import Path

import psycopg

# 프로젝트의 루트 디렉토리를 sys.path에 추가하여 모듈을 임포트할 수 있도록 함
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.core.chunking import smart_chunks
from app.core.embeddings import async_embed_texts

# --- 설정 ---
# 인덱싱할 최상위 디렉토리 목록
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [
    str(PROJECT_ROOT / "docs" / "kbo_rulebook"),
    str(PROJECT_ROOT / "docs" / "kbo_knowledge"),
    str(PROJECT_ROOT / "docs"),
]
# 인덱싱할 파일 확장자
FILE_EXTENSION = ".md"
# 데이터베이스에 저장될 때 사용할 출처(source) 테이블 이름
SOURCE_TABLE_NAME = "markdown_docs"
PGVECTOR_SEARCH_PATH = "public, extensions, security"


def get_db_connection() -> Generator[psycopg.Connection, None, None]:
    """데이터베이스 연결을 생성하는 제너레이터 함수."""
    settings = get_settings()
    conn = psycopg.connect(settings.database_url, autocommit=True)
    with conn.cursor() as cur:
        cur.execute(f"SET search_path TO {PGVECTOR_SEARCH_PATH}")
    try:
        yield conn
    finally:
        conn.close()


def find_markdown_files(directories: list[str]) -> list[str]:
    """지정된 디렉토리에서 모든 마크다운 파일의 경로를 찾습니다."""
    markdown_files = []
    seen_paths: set[str] = set()

    def add_markdown_file(path: str) -> None:
        normalized_path = os.path.realpath(path)
        if normalized_path in seen_paths:
            return
        seen_paths.add(normalized_path)
        markdown_files.append(normalized_path)

    for directory in directories:
        if os.path.isfile(directory) and directory.endswith(FILE_EXTENSION):
            add_markdown_file(directory)
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(FILE_EXTENSION):
                    add_markdown_file(os.path.join(root, file))
    print(f"총 {len(markdown_files)}개의 마크다운 파일을 찾았습니다.")
    return markdown_files


import re


def infer_season_year(path: str) -> int:
    """Infer the season year from the file path. Looks for years between 1982 and 2029."""
    match = re.search(r"(19[8-9]\d|20[0-2]\d)", path)  # Covers 1980-2029
    if match:
        year = int(match.group(1))
        if 1982 <= year <= 2029:  # Plausible KBO year range
            return year
    return 0  # Default to 0 if no year is found to satisfy NOT NULL constraint


def infer_doc_metadata(relative_path: str) -> dict[str, str]:
    """문서 경로를 기반으로 검색용 메타데이터를 추정합니다."""
    normalized = relative_path.lower()
    current_year = datetime.now().year
    inferred_year = infer_season_year(relative_path)

    if "kbo_rulebook" in normalized or any(
        token in normalized for token in ("glossary", "rules_terms", "rulebook")
    ):
        knowledge_type = "rules_terms"
    elif any(
        token in normalized
        for token in ("strategy", "metric", "metrics", "game_types")
    ):
        knowledge_type = "strategy_metrics"
    elif any(
        token in normalized
        for token in ("culture", "history", "historical", "tradition")
    ):
        knowledge_type = "culture_history"
    elif any(
        token in normalized
        for token in ("season", "storyline", "review", "draft", "market")
    ):
        knowledge_type = "season_narrative"
    else:
        knowledge_type = "reference"

    league_scope = (
        "baseball_general"
        if knowledge_type in {"rules_terms", "strategy_metrics"}
        else "kbo"
    )
    freshness = (
        "seasonal"
        if inferred_year and inferred_year >= current_year - 1
        else "evergreen"
    )

    return {
        "league_scope": league_scope,
        "knowledge_type": knowledge_type,
        "freshness": freshness,
        "path": relative_path,
    }


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
                    with open(file_path, "r", encoding="utf-8") as f:
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
                league_type_code = 0  # Default to 0 to satisfy NOT NULL constraint
                relative_path = Path(file_path).resolve().relative_to(PROJECT_ROOT).as_posix()
                metadata = infer_doc_metadata(relative_path)

                # 청크 분할 및 임베딩 생성
                chunks = smart_chunks(content)
                embeddings = await async_embed_texts(chunks, settings)

                cur.execute(
                    """
                    DELETE FROM rag_chunks
                    WHERE source_table = %s
                      AND left(source_row_id, %s) = %s
                    """,
                    (
                        SOURCE_TABLE_NAME,
                        len(relative_path) + 1,
                        f"{relative_path}_",
                    ),
                )

                # 데이터베이스에 저장
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    vector_literal = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
                    # 고유 ID로 파일 경로와 청크 인덱스 사용
                    source_row_id = f"{relative_path}_{i}"

                    cur.execute(
                        """
                        INSERT INTO rag_chunks (season_year, league_type_code, source_table, source_row_id, title, content, meta, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (source_table, source_row_id)
                        DO UPDATE SET 
                            content = EXCLUDED.content, 
                            meta = EXCLUDED.meta,
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
                            Path(relative_path).name,
                            chunk,
                            json.dumps(metadata, ensure_ascii=False),
                            vector_literal,
                        ),
                    )
                print(
                    f"  - {len(chunks)}개의 청크를 데이터베이스에 저장했습니다. (Year: {season_year})"
                )
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
