"""
Supabase(Postgres)에서 KBO 관련 테이블을 읽어 사람이 읽기 쉬운 텍스트로 변환→(선택) 임베딩 생성→rag_chunks 테이블에 UPSERT 하는 배치 인젝션 파이프라인입니다.
 - 테이블별 선택/제목/하이라이트/렌더링 규칙을 프로필로 정의해 공통 로직으로 처리할 수 있습니다.

    source .venv/bin/activate
    예시) python ingest_from_kbo.py --tables player_season_batting player_season_pitching --season-year 2025 --read-batch-size 500 --embed-batch-size 32 --max-concurrency 2

    --read-batch-size 500
     - DB에서 한 번에 끌어올(row fetch) 레코드 묶음 크기.
    --embed-batch-size 32
     - 임베딩 API 한 요청에 넣는 텍스트 개수.
     - 32는 보통 임베딩 엔드포인트에서 안전하게 처리되는 중간값이어서 처리량/안정성 균형이 좋음.
    --max-concurrency 2
     - embed-batch를 몇개를 쌓아서 보낼지 정하는 명령어
     - 동시에 날리는 임베딩 요청 수.

    외부 api 사용 명령어 표준
    python ingest_from_kbo.py \
        --tables player_season_batting player_season_pitching \
        --season-year 2025 \
        --read-batch-size 500 \
        --embed-batch-size 24 \
        --max-concurrency 2 \
        --commit-interval 1000

    --commit-interval 1000 = 업서트한 레코드를 1000개 처리할 때마다 DB 트랜잭션을 커밋하라는 뜻.

     
Docker/compose 환경에서는 `working_dir=/app` 상태에서 동일한 명령을 실행한다.

위 명령을 실행하면 Supabase에서 데이터를 읽어와 벡터 임베딩과 함께 `rag_chunks`
테이블에 업서트한다.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

# get_settings().database_url로 Postgres 연결을 열고 쿼리 타임아웃을 막기 위해 SET statement_timeout TO 0; 적용. 각 테이블을 순서대로 처리.
from app.config import get_settings
from datetime import datetime

from app.core.chunking import smart_chunks
from app.core.embeddings import embed_texts
from app.core.renderers.baseball import (
    render_batting_season,
    render_pitching_season,
)


@dataclass
class ChunkPayload:
    table: str
    source_row_id: str
    title: str
    content: str
    season_year: Optional[int]
    season_id: Optional[int]
    league_type_code: Optional[int]
    team_id: Optional[str]
    player_id: Optional[str]
    meta: Optional[Dict[str, Any]]

# TABLE_PROFILES에 테이블별 메타가 있음: 설명, select_sql, 제목 구성용 필드(title_fields), 본문 하이라이트(highlights), 기본키 힌트(pk_hint), 전용 렌더러(renderer).
TABLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "kbo_metrics_explained": {
        "description": "KBO 야구 기록 지표 설명",
        "source_file": Path(__file__).parent.parent / "docs" / "kbo_metrics_explained.md",
        "source_table": "kbo_definitions", # A new source_table name for these definitions
        "title": "KBO 야구 기록 지표 설명", # Fixed title for the chunks
        "pk_hint": ["title"], # A simple PK hint, though not strictly needed for a single file
    },
    "kbo_regulations_basic": {
        "description": "KBO 기본 규정 (리그 구성, 경기 시간, 타이브레이크 등)",
        "source_file": Path(__file__).parent.parent / "docs" / "kbo_regulations" / "01_기본규정.md",
        "source_table": "kbo_regulations",
        "title": "KBO 기본 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_player": {
        "description": "KBO 선수 규정 (등록, FA, 외국인선수, 드래프트 등)",
        "source_file": Path(__file__).parent.parent / "docs" / "kbo_regulations" / "02_선수규정.md",
        "source_table": "kbo_regulations",
        "title": "KBO 선수 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_game": {
        "description": "KBO 경기 규정 (경기 진행, 방해, 보크, 홈런 등)",
        "source_file": Path(__file__).parent.parent / "docs" / "kbo_regulations" / "03_경기규정.md",
        "source_table": "kbo_regulations",
        "title": "KBO 경기 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_technical": {
        "description": "KBO 기술 규정 (기록, 통계, 심판, 용품 등)",
        "source_file": Path(__file__).parent.parent / "docs" / "kbo_regulations" / "04_기술규정.md",
        "source_table": "kbo_regulations",
        "title": "KBO 기술 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_discipline": {
        "description": "KBO 징계 규정 (폭력, 도박, 약물, 처벌 기준 등)",
        "source_file": Path(__file__).parent.parent / "docs" / "kbo_regulations" / "05_징계규정.md",
        "source_table": "kbo_regulations",
        "title": "KBO 징계 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_postseason": {
        "description": "KBO 포스트시즌 규정 (플레이오프, 와일드카드, 한국시리즈 등)",
        "source_file": Path(__file__).parent.parent / "docs" / "kbo_regulations" / "06_포스트시즌.md",
        "source_table": "kbo_regulations",
        "title": "KBO 포스트시즌 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_special": {
        "description": "KBO 특별 규정 (코로나19, 기상이변, 비상상황 등)",
        "source_file": Path(__file__).parent.parent / "docs" / "kbo_regulations" / "07_특별규정.md",
        "source_table": "kbo_regulations",
        "title": "KBO 특별 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_terms": {
        "description": "KBO 야구 용어 정의 (기본 용어, 통계 지표, 포지션 등)",
        "source_file": Path(__file__).parent.parent / "docs" / "kbo_regulations" / "08_용어정의.md",
        "source_table": "kbo_regulations",
        "title": "KBO 야구 용어 정의",
        "pk_hint": ["title"],
    },
    "player_season_batting": {
        "description": "KBO 타자 시즌 기록 요약",
        "kind": "batting_season",
        "title_fields": [
            ["season", "season_year"],
            ["team_name", "team_code"],
            ["player_name", "player_id"],
        ],
        "select_sql": """
            SELECT
                bs.*,
                ks.season_year,
                COALESCE(bs.season_id, ks.season_id) AS season_lookup_id,
                ks.league_type_code,
                pb.name AS player_name,
                t.team_name
            FROM player_season_batting bs
            LEFT JOIN kbo_seasons ks
              ON ks.season_year = bs.season
             AND ks.league_type_code = CASE WHEN bs.league = 'PLAYOFF' THEN 4 ELSE 0 END
            LEFT JOIN player_basic pb
              ON pb.player_id = bs.player_id
            LEFT JOIN teams t
              ON t.team_id = bs.team_code
            ORDER BY bs.season DESC, bs.team_code, bs.player_id
        """,
        "highlights": [
            ("시즌", ["season_year", "season"]),
            ("팀", ["team_name", "team_code"]),
            ("선수", ["player_name", "player_id"]),
            ("AVG", ["avg", "batting_avg"]),
            ("OPS", ["ops"]),
            ("OBP", ["obp"]),
            ("SLG", ["slg"]),
            ("HR", ["hr", "home_runs"]),
            ("RBI", ["rbi"]),
            ("경기", ["games", "g"]),
            ("타석", ["plate_appearances", "pa"]),
            ("도루", ["stolen_bases"]),
        ],
        "pk_hint": ["player_id", "season_id", "season", "league", "level"],
        "renderer": render_batting_season,
        "season_filter_column": "ks.season_year",
    },
    "player_season_pitching": {
        "description": "KBO 투수 시즌 기록 요약",
        "kind": "pitching_season",
        "title_fields": [
            ["season", "season_year"],
            ["team_name", "team_code"],
            ["player_name", "player_id"],
        ],
        "select_sql": """
            SELECT
                ps.*,
                ks.season_year,
                COALESCE(ps.season_id, ks.season_id) AS season_lookup_id,
                ks.league_type_code,
                pb.name AS player_name,
                t.team_name
            FROM player_season_pitching ps
            LEFT JOIN kbo_seasons ks
              ON ks.season_year = ps.season
             AND ks.league_type_code = CASE WHEN ps.league = 'PLAYOFF' THEN 4 ELSE 0 END
            LEFT JOIN player_basic pb
              ON pb.player_id = ps.player_id
            LEFT JOIN teams t
              ON t.team_id = ps.team_code
            ORDER BY ps.season DESC, ps.team_code, ps.player_id
        """,
        "highlights": [
            ("시즌", ["season_year", "season"]),
            ("팀", ["team_name", "team_code"]),
            ("선수", ["player_name", "player_id"]),
            ("ERA", ["era"]),
            ("승", ["wins", "win"]),
            ("패", ["losses", "loss"]),
            ("세이브", ["saves", "save"]),
            ("홀드", ["holds", "hold"]),
            ("이닝", ["innings_pitched", "ip"]),
            ("탈삼진", ["strikeouts", "so"]),
            ("WHIP", ["whip"]),
            ("FIP", ["fip"]),
            ("K/9", ["k_per_nine"]),
            ("BB/9", ["bb_per_nine"]),
        ],
        "pk_hint": ["player_id", "season_id", "season", "league", "level"],
        "renderer": render_pitching_season,
        "season_filter_column": "ks.season_year",
    },
    "game": {
        "description": "KBO 경기 기본 정보",
        "title_fields": [
            ["game_date", "date"],
            ["home_team_name", "home_team"],
            ["away_team_name", "away_team"],
        ],
        "select_sql": """
            SELECT
                g.*,
                ks.season_year,
                ks.league_type_code,
                s.stadium_name,
                ht.team_name AS home_team_name,
                at.team_name AS away_team_name
            FROM game g
            LEFT JOIN kbo_seasons ks
              ON ks.season_id = g.season_id
            LEFT JOIN stadiums s
              ON s.stadium_id = g.stadium_id
            LEFT JOIN teams ht
              ON ht.team_id = g.home_team
            LEFT JOIN teams at
              ON at.team_id = g.away_team
            ORDER BY g.game_date DESC, g.game_id
        """,
        "highlights": [
            ("경기일", ["game_date", "date"]),
            ("경기 ID", ["game_id", "id"]),
            ("구장", ["stadium", "stadium_name", "stadium_id"]),
            ("홈팀", ["home_team_name", "home_team"]),
            ("원정팀", ["away_team_name", "away_team"]),
            ("홈팀 점수", ["home_score"]),
            ("원정팀 점수", ["away_score"]),
            ("승리팀", ["winning_team"]),
            ("경기 상태", ["game_status"]),
            ("시작 시간", ["start_time"]),
        ],
        "pk_hint": ["id", "game_id"],
        "season_filter_column": "ks.season_year",
    },
    "box_score": {
        "description": "경기별 팀 박스 스코어 요약",
        "title_fields": [
            ["game_date"],
            ["game_id"],
            ["stadium", "stadium_name"],
        ],
        "select_sql": """
            SELECT
                bs.*,
                g.game_date,
                g.home_team,
                g.away_team,
                g.home_score,
                g.away_score,
                g.season_id,
                ks.season_year,
                ks.league_type_code,
                s.stadium_name
            FROM box_score bs
            LEFT JOIN game g
              ON g.game_id = bs.game_id
            LEFT JOIN kbo_seasons ks
              ON ks.season_id = g.season_id
            LEFT JOIN stadiums s
              ON s.stadium_id = bs.stadium_id
            ORDER BY g.game_date DESC NULLS LAST, bs.game_id
        """,
        "highlights": [
            ("경기", ["game_id"]),
            ("경기일", ["game_date"]),
            ("구장", ["stadium", "stadium_name", "stadium_id"]),
            ("관중", ["crowd"]),
            ("경기 시간", ["game_time"]),
            ("홈팀 성적", ["home_record"]),
            ("원정팀 성적", ["away_record"]),
            ("홈팀 득점", ["home_r"]),
            ("원정팀 득점", ["away_r"]),
            ("홈팀 안타", ["home_h"]),
            ("원정팀 안타", ["away_h"]),
            ("홈팀 실책", ["home_e"]),
            ("원정팀 실책", ["away_e"]),
        ],
        "pk_hint": ["id", "game_id"],
        "season_filter_column": "ks.season_year",
    },
    "game_summary": {
        "description": "경기 텍스트 요약 및 주요 이슈",
        "title_fields": [
            ["game_id"],
            ["summary_type"],
            ["player_name"],
        ],
        "select_sql": """
            SELECT
                gs.*,
                g.game_date,
                g.home_team,
                g.away_team,
                ht.team_name AS home_team_name,
                at.team_name AS away_team_name,
                ks.season_year,
                ks.league_type_code,
                g.season_id
            FROM game_summary gs
            LEFT JOIN game g
              ON g.game_id = gs.game_id
            LEFT JOIN teams ht
              ON ht.team_id = g.home_team
            LEFT JOIN teams at
              ON at.team_id = g.away_team
            LEFT JOIN kbo_seasons ks
              ON ks.season_id = g.season_id
            ORDER BY g.game_date DESC NULLS LAST, gs.id
        """,
        "highlights": [
            ("경기", ["game_id"]),
            ("경기일", ["game_date"]),
            ("요약 종류", ["summary_type"]),
            ("관련 선수", ["player_name"]),
            ("요약", ["detail_text"]),
        ],
        "pk_hint": ["game_id", "id"],
        "season_filter_column": "ks.season_year",
    },
    "kbo_seasons": {
        "description": "연도별 KBO 시즌 정보",
        "title_fields": [
            ["season_year"],
            ["league_type_name"],
        ],
        "highlights": [
            ("시즌", ["season_year"]),
            ("리그 코드", ["league_type_code"]),
            ("리그 명", ["league_type_name"]),
            ("시작일", ["start_date"]),
            ("종료일", ["end_date"]),
        ],
        "pk_hint": ["season_year", "season_id"],
    },
    "stadiums": {
        "description": "KBO 구장 정보",
        "title_fields": [
            ["stadium_name"],
            ["city"],
        ],
        "highlights": [
            ("구장", ["stadium_name", "name"]),
            ("도시", ["city"]),
            ("팀", ["team"]),
            ("수용 인원", ["capacity", "seating_capacity"]),
            ("개장 연도", ["open_year"]),
        ],
        "pk_hint": ["stadium_id"],
        "season_filter_column": None,
    },
    "teams": {
        "description": "KBO 구단 기본 정보",
        "title_fields": [
            ["team_name"],
            ["team_id"],
        ],
        "highlights": [
            ("구단", ["team_name"]),
            ("코드", ["team_id"]),
            ("약칭", ["team_short_name"]),
            ("연고지", ["city"]),
            ("구장", ["stadium_name"]),
            ("창단 연도", ["founded_year"]),
            ("팀 색상", ["color"]),
        ],
        "pk_hint": ["team_id"],
        "season_filter_column": None,
    },
    "team_history": {
        "description": "KBO 구단 변천사",
        "title_fields": [
            ["team_name", "team_code"],
            ["start_season"],
        ],
        "select_sql": """
            SELECT
                th.*,
                th.start_season AS season_year,
                s.stadium_name,
                t.team_name AS current_team_name
            FROM team_history th
            LEFT JOIN stadiums s
              ON s.stadium_id = th.stadium_id
            LEFT JOIN teams t
              ON t.team_id = th.team_code
            ORDER BY th.team_code, th.start_season
        """,
        "highlights": [
            ("구단", ["team_name", "team_code"]),
            ("시작 시즌", ["start_season"]),
            ("종료 시즌", ["end_season"]),
            ("도시", ["city"]),
            ("주경기장", ["stadium_name"]),
            ("현재 여부", ["is_current"]),
        ],
        "pk_hint": ["team_code", "start_season"],
        "season_filter_column": "th.start_season",
    },
    "hitter_record": {
        "description": "경기별 타자 기록",
        "title_fields": [
            ["game_id"],
            ["team_id"],
            ["player_name", "player_id"],
        ],
        "select_sql": """
            SELECT
                hr.*,
                ks.season_year,
                ks.league_type_code,
                t.team_name
            FROM hitter_record hr
            LEFT JOIN kbo_seasons ks
              ON ks.season_id = hr.season_id
            LEFT JOIN teams t
              ON t.team_id = hr.team_id
            ORDER BY hr.game_id DESC, hr.team_id, hr.batting_order
        """,
        "highlights": [
            ("경기", ["game_id"]),
            ("팀", ["team_name", "team_id"]),
            ("선수", ["player_name"]),
            ("타순", ["batting_order"]),
            ("포지션", ["position"]),
            ("타수", ["at_bats"]),
            ("안타", ["hits"]),
            ("득점", ["runs"]),
            ("타점", ["rbis"]),
            ("타율", ["batting_average"]),
        ],
        "pk_hint": ["game_id", "team_id", "player_name"],
        "season_filter_column": "ks.season_year",
    },
    "pitcher_record": {
        "description": "경기별 투수 기록",
        "title_fields": [
            ["game_id"],
            ["team_id"],
            ["player_name"],
        ],
        "select_sql": """
            SELECT
                pr.*,
                ks.season_year,
                ks.league_type_code,
                t.team_name
            FROM pitcher_record pr
            LEFT JOIN kbo_seasons ks
              ON ks.season_id = pr.season_id
            LEFT JOIN teams t
              ON t.team_id = pr.team_id
            ORDER BY pr.game_id DESC, pr.team_id, pr.id
        """,
        "highlights": [
            ("경기", ["game_id"]),
            ("팀", ["team_name", "team_id"]),
            ("선수", ["player_name"]),
            ("등판", ["appearance"]),
            ("결과", ["result"]),
            ("승", ["wins"]),
            ("패", ["losses"]),
            ("세이브", ["saves"]),
            ("이닝", ["innings", "innings_display"]),
            ("탈삼진", ["strikeouts"]),
            ("실점", ["runs_allowed"]),
        ],
        "pk_hint": ["game_id", "team_id", "player_name"],
        "season_filter_column": "ks.season_year",
    },
    "player_basic": {
        "description": "선수 기본 정보",
        "title_fields": [
            ["name"],
            ["player_id"],
        ],
        "select_sql": """
            SELECT
                pb.*,
                t.team_name
            FROM player_basic pb
            LEFT JOIN teams t
              ON t.team_id = pb.team_id
            ORDER BY pb.player_id
        """,
        "highlights": [
            ("선수", ["name"]),
            ("등번호", ["uniform_no"]),
            ("포지션", ["position"]),
            ("생년월일", ["birth_date"]),
            ("신체", ["height_cm", "weight_kg"]),
            ("소속팀", ["team_name", "team_id"]),
        ],
        "pk_hint": ["player_id"],
        "season_filter_column": None,
    },
    "team_name_mapping": {
        "description": "풀네임-코드 매핑",
        "title_fields": [
            ["full_name"],
            ["team_id"],
        ],
        "select_sql": """
            SELECT * FROM team_name_mapping ORDER BY full_name
        """,
        "highlights": [
            ("풀네임", ["full_name"]),
            ("팀 코드", ["team_id"]),
        ],
        "pk_hint": ["full_name"],
        "season_filter_column": None,
    },
    "team_profiles": {
        "description": "구단 프로필",
        "title_fields": [
            ["team_id"],
        ],
        "select_sql": """
            SELECT
                tp.*,
                t.team_name
            FROM team_profiles tp
            LEFT JOIN teams t
              ON t.team_id = tp.team_id
            ORDER BY tp.team_id
        """,
        "highlights": [
            ("팀", ["team_name", "team_id"]),
            ("프로필", ["profile"]),
        ],
        "pk_hint": ["team_id", "id"],
        "season_filter_column": None,
    },
}

# Tables the caller can choose. `rag_chunks` intentionally 제외.
DEFAULT_TABLES = [
    "player_season_batting",
    "player_season_pitching",
    "hitter_record",
    "pitcher_record",
    "game",
    "box_score",
    "game_summary",
    "kbo_seasons",
    "stadiums",
    "teams",
    "team_history",
    "player_basic",
    "team_profiles",
    "team_name_mapping",
    "kbo_metrics_explained",
    "kbo_regulations_basic",
    "kbo_regulations_player",
    "kbo_regulations_game",
    "kbo_regulations_technical",
    "kbo_regulations_discipline",
    "kbo_regulations_postseason",
    "kbo_regulations_special",
    "kbo_regulations_terms",
]

TARGET_RPM = 10
MIN_DELAY_SECONDS = 60 / TARGET_RPM


UPSERT_SQL = """
INSERT INTO rag_chunks (
    meta,
    season_year,
    season_id,
    league_type_code,
    team_id,
    player_id,
    source_table,
    source_row_id,
    title,
    content,
    embedding
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
ON CONFLICT (source_table, source_row_id)
DO UPDATE SET
    meta = EXCLUDED.meta,
    content = EXCLUDED.content,
    embedding = COALESCE(EXCLUDED.embedding, rag_chunks.embedding),
    season_year = COALESCE(EXCLUDED.season_year, rag_chunks.season_year),
    season_id = COALESCE(EXCLUDED.season_id, rag_chunks.season_id),
    league_type_code = COALESCE(EXCLUDED.league_type_code, rag_chunks.league_type_code),
    team_id = COALESCE(EXCLUDED.team_id, rag_chunks.team_id),
    player_id = COALESCE(EXCLUDED.player_id, rag_chunks.player_id),
    title = EXCLUDED.title,
    updated_at = now();
"""


def first_value(row: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_primary_key_columns(conn, table: str) -> List[str]:
    query = """
        SELECT a.attname
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
        WHERE c.relname = %s AND n.nspname = %s AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
    """
    with conn.cursor() as cur:
        cur.execute(query, (table, "public"))
        return [row[0] for row in cur.fetchall()]


def build_source_row_id(
    row: Dict[str, Any],
    table: str,
    pk_columns: Sequence[str],
    pk_hint: Sequence[str],
) -> str:
    candidates: List[str] = []
    for column in pk_columns:
        if column in row and row[column] is not None:
            candidates.append(f"{column}={row[column]}")
    if not candidates:
        for column in pk_hint:
            if column in row and row[column] is not None:
                candidates.append(f"{column}={row[column]}")
    if not candidates and "id" in row and row["id"] is not None:
        candidates.append(f"id={row['id']}")
    if candidates:
        return "|".join(candidates)

    digest = hashlib.md5(
        json.dumps(row, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return f"{table}_hash={digest}"


def build_title(
    row: Dict[str, Any], table: str, source_row_id: str, profile: Dict[str, Any]
) -> str:
    kind = profile.get("kind")
    if kind == "pitching_season":
        season = first_value(row, ["season_year", "season"])
        team = first_value(row, ["team_name", "team_code"])
        name = first_value(row, ["player_name"])
        base_parts = []
        if season:
            base_parts.append(f"{season}년")
        if team:
            base_parts.append(str(team))
        if name:
            base_parts.append(str(name))
        else:
            base_parts.append(f"선수 {row.get('player_id', '')}".strip())
        base = " ".join(filter(None, base_parts))

        def _fmt(keys: Sequence[str], digits: int) -> Optional[str]:
            value = first_value(row, keys)
            if value in (None, ""):
                return None
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return None

        details: List[str] = []
        era = _fmt(["era"], 2)
        if era:
            details.append(f"ERA {era}")
        whip = _fmt(["whip"], 2)
        if whip:
            details.append(f"WHIP {whip}")
        wins = first_value(row, ["win", "wins"])
        losses = first_value(row, ["loss", "losses"])
        saves = first_value(row, ["save", "saves"])
        record_parts = []
        if wins not in (None, ""):
            record_parts.append(f"{int(float(wins))}승")
        if losses not in (None, ""):
            record_parts.append(f"{int(float(losses))}패")
        if saves not in (None, ""):
            record_parts.append(f"{int(float(saves))}세")
        if record_parts:
            details.append(" ".join(record_parts))
        suffix = f"요약({', '.join(details)})" if details else "요약"
        return f"{base} – {suffix}"

    if kind == "batting_season":
        season = first_value(row, ["season_year", "season"])
        team = first_value(row, ["team_name", "team_code"])
        name = first_value(row, ["player_name"])
        base_parts = []
        if season:
            base_parts.append(f"{season}년")
        if team:
            base_parts.append(str(team))
        if name:
            base_parts.append(str(name))
        else:
            base_parts.append(f"선수 {row.get('player_id', '')}".strip())
        base = " ".join(filter(None, base_parts))

        def _fmt(keys: Sequence[str], digits: int) -> Optional[str]:
            value = first_value(row, keys)
            if value in (None, ""):
                return None
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return None

        details: List[str] = []
        avg = _fmt(["avg", "batting_avg"], 3)
        if avg:
            details.append(f"타율 {avg}")
        ops = _fmt(["ops"], 3)
        if ops:
            details.append(f"OPS {ops}")
        home_runs = first_value(row, ["hr", "home_runs"])
        if home_runs not in (None, ""):
            try:
                details.append(f"홈런 {int(float(home_runs))}")
            except (TypeError, ValueError):
                pass
        rbi = first_value(row, ["rbi"])
        if rbi not in (None, ""):
            try:
                details.append(f"타점 {int(float(rbi))}")
            except (TypeError, ValueError):
                pass
        suffix = f"요약({', '.join(details)})" if details else "요약"
        return f"{base} – {suffix}"

    parts: List[str] = []
    for group in profile.get("title_fields", []):
        keys = group if isinstance(group, list) else [group]
        value = first_value(row, keys)
        if value is not None:
            parts.append(str(value))
    if not parts:
        fallback = first_value(
            row,
            [
                "title",
                "headline",
                "player_name",
                "team_name",
                "stadium_name",
                "game_id",
            ],
        )
        if fallback is not None:
            parts.append(str(fallback))
    if not parts:
        parts.append(f"{table} {source_row_id}")
    return " ".join(parts)


def build_content(
    row: Dict[str, Any],
    table: str,
    source_row_id: str,
    profile: Dict[str, Any],
) -> str:
    lines: List[str] = []
    used_keys: set[str] = set()

    description = profile.get("description")
    if description:
        lines.append(description)

    for label, keys in profile.get("highlights", []):
        value = first_value(row, keys)
        if value is None:
            continue
        lines.append(f"{label}: {value}")
        for key in keys:
            if key in row:
                used_keys.add(key)

    for key, value in row.items():
        if key in used_keys or value in (None, ""):
            continue
        lines.append(f"{key}: {value}")

    lines.append(f"출처: {table}#{source_row_id}")
    return "\n".join(str(line) for line in lines)

# build_select_query가 프로필의 select_sql이 있으면 그 SQL에 season_year 등 필터를 주입하고 ORDER BY/ LIMIT를 붙임. 커스텀 SQL이 없으면 SELECT * FROM <table> + PK 순 정렬.
def build_select_query(
    table: str,
    profile: Dict[str, Any],
    pk_columns: Sequence[str],
    limit: Optional[int],
    season_year: Optional[int],
    since: Optional[datetime],
):
    custom_sql = profile.get("select_sql")
    season_filter_column = profile.get("season_filter_column", "season_year")
    params: List[Any] = []
    if custom_sql:
        stripped = custom_sql.strip()
        upper = stripped.upper()
        order_clause = ""
        order_idx = upper.find(" ORDER BY ")
        if order_idx != -1:
            base_sql = stripped[:order_idx]
            order_clause = stripped[order_idx:]
        else:
            base_sql = stripped

        where_clauses = []
        if season_year is not None and season_filter_column:
            where_clauses.append(f"{season_filter_column} = %s")
            params.append(season_year)
        if where_clauses:
            upper_base = base_sql.upper()
            has_where = " WHERE " in upper_base or upper_base.endswith("WHERE")
            if has_where:
                base_sql = f"{base_sql}\n   AND {' AND '.join(where_clauses)}"
            else:
                base_sql = f"{base_sql}\nWHERE {' AND '.join(where_clauses)}"

        query = base_sql
        if order_clause:
            query = f"{query}\n{order_clause}"
        if limit is not None:
            query = f"{query} LIMIT %s"
            params.append(limit)
        return query, tuple(params)

    query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table))
    where_parts: List[sql.SQL] = []
    if season_year is not None and season_filter_column:
        where_parts.append(
            sql.SQL("{} = %s").format(sql.Identifier(season_filter_column))
        )
        params.append(season_year)
    if since is not None:
        where_parts.append(sql.SQL("updated_at >= %s"))
        params.append(since)
    if where_parts:
        query = query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_parts)
    if pk_columns:
        order_cols = sql.SQL(", ").join(sql.Identifier(col) for col in pk_columns)
        query = query + sql.SQL(" ORDER BY {}").format(order_cols)
    if limit is not None:
        query = query + sql.SQL(" LIMIT %s")
        params.append(limit)
    return query, tuple(params)


def batched(iterable: Sequence[ChunkPayload], size: int) -> Iterable[List[ChunkPayload]]:
    total = len(iterable)
    for idx in range(0, total, size):
        yield list(iterable[idx : idx + size])


def flush_chunks(
    cur,
    settings,
    buffer: List[ChunkPayload],
    *,
    max_concurrency: int,
    commit_interval: int,
    stats: Dict[str, Any],
    skip_embedding: bool,
) -> int:
    if not buffer:
        return 0

    stats["batches"] = stats.get("batches", 0) + 1
    vector_literals: List[Optional[str]]
    embeddings: Optional[List[List[float]]] = None

    if skip_embedding:
        vector_literals = [None] * len(buffer)
    else:
        stats["embedding_calls"] = stats.get("embedding_calls", 0) + 1
        start = time.perf_counter()
        embeddings = embed_texts(
            [item.content for item in buffer],
            settings,
            max_concurrency=max_concurrency,
        )
        elapsed = time.perf_counter() - start
        stats["sleep_seconds"] = stats.get("sleep_seconds", 0.0) + elapsed
        vector_literals = [
            "[" + ",".join(f"{v:.8f}" for v in embedding) + "]" for embedding in embeddings
        ]

    for item, vector_literal in zip(buffer, vector_literals):
        cur.execute(
            UPSERT_SQL,
            (
                json.dumps(item.meta, default=str) if item.meta else None,
                item.season_year,
                item.season_id,
                item.league_type_code,
                item.team_id,
                item.player_id,
                item.table,
                item.source_row_id,
                item.title,
                item.content,
                vector_literal,
            ),
        )

    flushed = len(buffer)
    buffer.clear()
    stats["since_commit"] = stats.get("since_commit", 0) + flushed
    if commit_interval and stats["since_commit"] >= commit_interval:
        cur.connection.commit()
        stats["since_commit"] = 0

    if embeddings is not None:
        del embeddings
    gc.collect()
    return flushed


def ingest_table(
    conn,
    table: str,
    *,
    limit: Optional[int],
    embed_batch_size: int,
    read_batch_size: int,
    season_year: Optional[int],
    since: Optional[datetime],
    use_legacy_renderer: bool,
    skip_embedding: bool,
    max_concurrency: int,
    commit_interval: int,
    stats: Dict[str, Any],
) -> int:
    if table == "rag_chunks":
        print("경고: rag_chunks 테이블은 처리 대상에서 제외됩니다.")
        return 0

    profile = TABLE_PROFILES.get(table, {})
    total_chunks = 0
    buffer: List[ChunkPayload] = []
    settings = get_settings()
    processed_chunks = 0
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    # upsert 작업이 오래 걸려 타임아웃이 발생하지 않도록 statement_timeout을 방지
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout TO 0;")

    with conn.cursor(cursor_factory=RealDictCursor) as read_cur, conn.cursor() as write_cur:
        write_cur.execute("SET statement_timeout TO 0;")

        # --- NEW LOGIC FOR STATIC FILE ---
        if "source_file" in profile:
            print(f"      정적 파일 '{profile['source_file']}'을(를) 수집 중입니다...")
            try:
                with open(profile["source_file"], "r", encoding="utf-8") as f:
                    content = f.read()
            except FileNotFoundError:
                print(f"오류: '{profile['source_file']}' 파일을 찾을 수 없습니다.")
                return 0

            chunks = smart_chunks(content)
            if not chunks:
                print(f"오류: '{profile['source_file']}' 파일 내용에서 청크를 생성할 수 없습니다.")
                return 0

            for idx, chunk in enumerate(chunks, start=1):
                buffer.append(
                    ChunkPayload(
                        table=profile["source_table"],
                        source_row_id=f"{profile['source_table']}_part_{idx}",
                        title=profile["title"],
                        content=chunk,
                        season_year=0,
                        season_id=None,
                        league_type_code=0,
                        team_id=None,
                        player_id=None,
                        meta={"source_file": str(profile["source_file"]), "chunk_index": idx},
                    )
                )
            flushed = flush_chunks(
                write_cur,
                settings,
                buffer,
                max_concurrency=max_concurrency,
                commit_interval=commit_interval,
                stats=stats,
                skip_embedding=skip_embedding,
            )
            total_chunks += flushed
            conn.commit() # Commit after static file ingestion
            if flushed > 0:
                print(f"      총 {flushed}개 청크를 처리했습니다.", flush=True)
            return total_chunks
        # --- END NEW LOGIC ---

        pk_columns = get_primary_key_columns(conn, table)
        query, params = build_select_query(
            table,
            profile,
            pk_columns,
            limit,
            season_year,
            since,
        )

        fetched_rows = 0
        read_cur.itersize = read_batch_size
        read_cur.execute(query, params)

        while True:
            rows = read_cur.fetchmany(read_batch_size)
            if not rows:
                break
            fetched_rows += len(rows)
            print(
                f"      테이블 '{table}'에서 {fetched_rows}개 행을 가져왔습니다...",
                flush=True,
            )
            for raw_row in rows:
                row = dict(raw_row)
                source_row_id = build_source_row_id(
                    row, table, pk_columns, profile.get("pk_hint", [])
                )
                title = build_title(row, table, source_row_id, profile)
                renderer = profile.get("renderer")
                if renderer and not use_legacy_renderer:
                    enriched_row = dict(row)
                    enriched_row["source_table"] = table
                    enriched_row["source_row_id"] = source_row_id
                    content = renderer(
                        enriched_row,
                        league_avg=None,
                        percentiles=None,
                        today_str=today_str,
                    )
                else:
                    content = build_content(row, table, source_row_id, profile)

                season_year = coerce_int(
                    first_value(row, ["season_year", "season", "year"])
                )
                if season_year is None:
                    season_year = 0

                season_id = coerce_int(
                    first_value(row, ["season_id", "season_lookup_id"])
                )
                league_type_code = coerce_int(
                    first_value(row, ["league_type_code", "league_type", "league"])
                )
                if league_type_code is None:
                    league_type_code = 0

                team_id = first_value(
                    row,
                    [
                        "team_id",
                        "home_team_id",
                        "away_team_id",
                        "team",
                        "team_code",
                    ],
                )
                player_id = first_value(row, ["player_id"])

                # 긴 본문은 smart_chunks로 분할. 분할되면 #part{n} 접미어와 “(분할 n)”를 제목에 추가.
                chunks = smart_chunks(content)
                if not chunks:
                    continue

                if len(chunks) == 1:
                    buffer.append(
                        ChunkPayload(
                            table=table,
                            source_row_id=source_row_id,
                            title=title,
                            content=chunks[0],
                            season_year=season_year,
                            season_id=season_id,
                            league_type_code=league_type_code,
                            team_id=str(team_id) if team_id is not None else None,
                            player_id=str(player_id) if player_id is not None else None,
                            meta=row,
                        )
                    )
                else:
                    for idx, chunk in enumerate(chunks, start=1):
                        buffer.append(
                            ChunkPayload(
                                table=table,
                                source_row_id=f"{source_row_id}#part{idx}",
                                title=f"{title} (분할 {idx})",
                                content=chunk,
                                season_year=season_year,
                                season_id=season_id,
                                league_type_code=league_type_code,
                                team_id=str(team_id) if team_id is not None else None,
                                player_id=str(player_id)
                                if player_id is not None
                                else None,
                                meta=row,
                            )
                        )

                if len(buffer) >= embed_batch_size:
                    flushed = flush_chunks(
                        write_cur,
                        settings,
                        buffer,
                        max_concurrency=max_concurrency,
                        commit_interval=commit_interval,
                        stats=stats,
                        skip_embedding=skip_embedding,
                    )
                    total_chunks += flushed
                    processed_chunks += flushed
                    print(
                        f"      현재까지 {processed_chunks}개 청크를 처리했습니다...",
                        flush=True,
                    )

        flushed = flush_chunks(
            write_cur,
            settings,
            buffer,
            max_concurrency=max_concurrency,
            commit_interval=commit_interval,
            stats=stats,
            skip_embedding=skip_embedding,
        )
        total_chunks += flushed
        processed_chunks += flushed
        if flushed:
            print(
                f"      현재까지 {processed_chunks}개 청크를 처리했습니다...",
                flush=True,
            )
        conn.commit()

    if processed_chunks:
        print(f"      총 {processed_chunks}개 청크를 처리했습니다.", flush=True)

    return total_chunks


def ingest(
    tables: Sequence[str],
    *,
    limit: Optional[int],
    embed_batch_size: int,
    read_batch_size: int,
    season_year: Optional[int],
    use_legacy_renderer: bool,
    since: Optional[datetime],
    skip_embedding: bool,
    max_concurrency: int,
    commit_interval: int,
) -> None:
    settings = get_settings()
    conn = psycopg2.connect(settings.database_url)
    original_autocommit = conn.autocommit
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout TO 0;")
    conn.autocommit = original_autocommit
    ingested_total = 0
    try:
        for table in tables:
            print(f" 테이블 '{table}'을(를) 수집 중입니다 ...")
            stats = {
                "embedding_calls": 0,
                "sleep_seconds": 0.0,
                "batches": 0,
                "since_commit": 0,
            }
            chunks = ingest_table(
                conn,
                table,
                limit=limit,
                embed_batch_size=embed_batch_size,
                read_batch_size=read_batch_size,
                season_year=season_year,
                use_legacy_renderer=use_legacy_renderer,
                since=since,
                skip_embedding=skip_embedding,
                max_concurrency=max_concurrency,
                commit_interval=commit_interval,
                stats=stats,
            )
            ingested_total += chunks
            print(
                f"   -> 테이블 '{table}'에서 {chunks}개 청크를 작성했습니다 "
                f"(배치={stats['batches']}, 임베딩 호출={stats['embedding_calls']}, 대기 시간={stats['sleep_seconds']:.2f}초)"
            )
    finally:
        conn.close()
    print(f"총 {ingested_total}개 청크 수집을 완료했습니다.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supabase KBO 데이터를 rag_chunks로 임베딩합니다."
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=DEFAULT_TABLES,
        help="인덱싱할 테이블 리스트 (기본: 주요 14개 테이블).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="테이블당 최대 처리 행. 지정하지 않으면 전체 행을 사용합니다.",
    )
    # 배치 버퍼가 --embed-batch-size를 채우면 embed_texts 호출(동시성 --max-concurrency) → 벡터 리터럴 문자열로 변환 → 아래 UPSERT 실행.
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=32,
        help="임베딩 API 호출당 청크 수 (기본 32).",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=500,
        help="DB에서 한 번에 읽어올 레코드 수 (기본 500).",
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=None,
        help="특정 시즌 연도만 인덱싱하고 싶을 때 지정 (예: 2025).",
    )
    parser.add_argument(
        "--use-legacy-renderer",
        action="store_true",
        help="기존 레이블:값 기반 포맷을 사용합니다.",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="임베딩을 호출하지 않고 content만 구성합니다.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="임베딩 API 호출 동시성 (기본 1).",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="updated_at 기준 ISO8601 타임스탬프 이후 변경분만 처리합니다 (예: 2025-05-01T00:00:00).",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=500,
        help="이 수만큼 청크를 쓰면 커밋을 수행합니다.",
    )
    return parser.parse_args()

# 우선순위로 안전 변환(coerce_int/first_value).

if __name__ == "__main__":
    args = parse_args()
    ingest(
        tables=[t for t in args.tables if t != "rag_chunks"],
        limit=args.limit,
        embed_batch_size=max(1, args.embed_batch_size),
        read_batch_size=max(1, args.read_batch_size),
        season_year=args.season_year,
        use_legacy_renderer=args.use_legacy_renderer,
        since=datetime.fromisoformat(args.since) if args.since else None,
        skip_embedding=args.no_embed,
        max_concurrency=max(1, args.max_concurrency),
        commit_interval=max(1, args.commit_interval),
    )
