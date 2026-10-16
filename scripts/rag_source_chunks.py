"""Read-side helpers that rebuild expected rag_chunks content from source tables.

Write paths (upsert, checkpointed ingest, sync) were retired: rag_chunks is owned
by the crawler (KBO_playwright). The audit and verification scripts import the
pure functions in this module; nothing here writes to a database.
"""

from __future__ import annotations

import hashlib
import json
import logging as _logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
except ModuleNotFoundError as exc:
    psycopg = None
    sql = None
    dict_row = None
    _PSYCOPG_IMPORT_ERROR = exc

# get_settings().database_url로 Postgres 연결을 열고 쿼리 타임아웃을 막기 위해 SET statement_timeout TO 0; 적용. 각 테이블을 순서대로 처리.
from app.config import Settings, get_settings
from datetime import datetime, timezone

from app.core.chunking import smart_chunks
from app.core.ingest_sources import TRUSTED_INGEST_SOURCE_TABLES
from app.core.ingest_checkpoints import (
    CheckpointCursor,
    CheckpointOrder,
    CheckpointOrderField,
    CursorScalarType,
    IngestCheckpointCursorTypeError,
    IngestCheckpointCursorUnavailableError,
    IngestCheckpointIncompatibleError,
)

_ingest_logger = _logging.getLogger(__name__)












from app.core.rag_storage import (
    base_source_row_id,
    infer_quality_score,
)
from app.core.renderers.baseball import (
    render_batting_season,
    render_game_flow_summary,
    render_pitching_season,
    render_hitter_game,
    render_pitcher_game,
    render_team_batting_season,
    render_team_pitching_season,
    render_stat_ranking,
)


def _load_settings_from_env_file(env_file: str) -> Settings:
    env_path = Path(env_file).expanduser()
    if not env_path.is_absolute():
        env_path = (PROJECT_ROOT / env_path).resolve()
    if not env_path.exists():
        raise FileNotFoundError(f"Env file not found: {env_path}")

    values: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized = value.strip()
        if (
            len(normalized) >= 2
            and normalized[0] == normalized[-1]
            and normalized[0] in {"'", '"'}
        ):
            normalized = normalized[1:-1]
        values[key.strip()] = normalized

    return Settings(_env_file=None, **values)


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required to run ingest_from_kbo.py. "
            "Install dependencies (e.g. pip install psycopg[binary]) and retry. "
            f"Detail: {_PSYCOPG_IMPORT_ERROR}"
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
    source_type: Optional[str] = None
    source_uri: Optional[str] = None
    topic_key: Optional[str] = None
    valid_from: Any = None
    valid_to: Any = None
    expires_at: Any = None
    quality_score: Optional[float] = None




class ManualBaseballDataRequiredError(RuntimeError):
    """Raised when trusted internal source data needs operator repair."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        self.contract = dict(contract)
        super().__init__(str(self.contract.get("operator_message") or self.contract["code"]))


REQUIRED_SOURCE_COLUMNS: Mapping[str, frozenset[str]] = {
    "game": frozenset({"game_id", "game_date"}),
    "game_metadata": frozenset({"game_id"}),
    "game_summary": frozenset({"game_id", "game_date"}),
}


def validate_required_source_columns(
    table_name: str,
    actual_columns: Iterable[str],
    required_columns: Iterable[str],
) -> None:
    """Enforce configured source fields without treating zero rows as missing data."""

    actual = {str(column).strip().lower() for column in actual_columns}
    required = {str(column).strip().lower() for column in required_columns}
    missing = sorted(required - actual)
    if not missing:
        return
    _raise_manual_source_schema(table_name, missing)


def _raise_manual_source_schema(
    table_name: str,
    missing_fields: Iterable[str],
) -> None:
    missing = sorted(
        {str(field).strip() for field in missing_fields if str(field).strip()}
    )
    contract = {
        "code": "MANUAL_BASEBALL_DATA_REQUIRED",
        "scope": "ai_ingest_source_schema",
        "entity": table_name,
        "range": None,
        "missing_fields": missing,
        "import_source": "operator_manual_data",
        "blocking": True,
        "operator_message": (
            f"Internal source '{table_name}' is missing required fields: "
            + ", ".join(missing)
            + ". Provide operator-verified data or repair the trusted internal sync."
        ),
    }
    raise ManualBaseballDataRequiredError(contract)






IngestLeaseGuard = Callable[[bool], None]




def load_static_profile_payloads(
    table_name: str,
    profile: Mapping[str, Any],
    *,
    settings: Any,
) -> List[ChunkPayload]:
    """Load a trusted static source or require operator-provided repair."""

    try:
        payloads = build_static_profile_chunk_payloads(
            table_name,
            profile,
            settings=settings,
        )
    except RuntimeError:
        _raise_manual_source_schema(table_name, ("source_file",))
    if not payloads:
        _raise_manual_source_schema(table_name, ("source_content",))
    return payloads


def _cursor_column_names(cursor: Any) -> Set[str]:
    names: Set[str] = set()
    for column in cursor.description or ():
        name = getattr(column, "name", None)
        if name is None and isinstance(column, Sequence) and column:
            name = column[0]
        if name:
            names.add(str(name))
    return names


def _row_updated_at(row: Mapping[str, Any], profile: Mapping[str, Any]) -> datetime | None:
    if "watermark_fields" in profile:
        configured = profile.get("watermark_fields")
        if isinstance(configured, str):
            fields = (configured,)
        else:
            fields = tuple(configured or ())
    else:
        fields = (
            "updated_at",
            "game_updated_at",
            "latest_updated_at",
        )
    for field in fields:
        value = row.get(str(field))
        if value is None:
            continue
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _read_source_updated_before(cursor: Any) -> datetime:
    cursor.execute("SELECT clock_timestamp() AS source_updated_before")
    row = cursor.fetchone()
    if isinstance(row, Mapping):
        value = row.get("source_updated_before")
    elif isinstance(row, Sequence) and row:
        value = row[0]
    else:
        value = None
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise IngestCheckpointIncompatibleError(
            "source database clock did not return an aware timestamp"
        )
    return value


# TABLE_PROFILES에 테이블별 메타가 있음: 설명, select_sql, 제목 구성용 필드(title_fields), 본문 하이라이트(highlights), 기본키 힌트(pk_hint), 전용 렌더러(renderer).
TABLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "kbo_metrics_explained": {
        "description": "KBO 야구 기록 지표 설명",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_metrics_explained.md",
        "source_table": "kbo_definitions",  # A new source_table name for these definitions
        "title": "KBO 야구 기록 지표 설명",  # Fixed title for the chunks
        "pk_hint": [
            "title"
        ],  # A simple PK hint, though not strictly needed for a single file
    },
    "markdown_docs_rules_terms": {
        "description": "야구 규정과 용어 해설 문서",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_knowledge"
        / "rules_terms.md",
        "source_table": "markdown_docs",
        "title": "야구 규정과 용어 해설",
        "pk_hint": ["title"],
        "league_scope": "kbo+baseball",
        "knowledge_type": "rules_terms",
        "freshness": "evergreen",
    },
    "markdown_docs_strategy_metrics": {
        "description": "야구 전술과 지표 해설 문서",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_knowledge"
        / "strategy_metrics.md",
        "source_table": "markdown_docs",
        "title": "야구 전술과 지표 해설",
        "pk_hint": ["title"],
        "league_scope": "kbo+baseball",
        "knowledge_type": "strategy_metrics",
        "freshness": "evergreen",
    },
    "markdown_docs_culture_history": {
        "description": "구단/구장/팬문화/역사 해설 문서",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_knowledge"
        / "culture_history.md",
        "source_table": "markdown_docs",
        "title": "구단과 구장, 팬문화, 역사 해설",
        "pk_hint": ["title"],
        "league_scope": "kbo",
        "knowledge_type": "culture_history",
        "freshness": "evergreen",
    },
    "markdown_docs_2025_storylines": {
        "description": "2025 시즌 스토리라인 해설 문서",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_knowledge"
        / "season_2025_storylines.md",
        "source_table": "markdown_docs",
        "title": "2025 KBO 시즌 스토리라인",
        "pk_hint": ["title"],
        "season_year": 2025,
        "league_scope": "kbo",
        "knowledge_type": "season_storylines",
        "freshness": "seasonal",
    },
    "markdown_docs_chatbot_kb_v2": {
        "description": "KBO 챗봇 심화 지식 베이스 V2",
        "source_file": Path(__file__).parent.parent / "docs" / "KBO_CHATBOT_KB_V2.md",
        "source_table": "markdown_docs",
        "title": "KBO 챗봇 심화 지식 베이스 V2",
        "pk_hint": ["title"],
        "season_year": 2025,
        "league_scope": "kbo",
        "knowledge_type": "season_storylines",
        "freshness": "seasonal",
    },
    "kbo_regulations_basic": {
        "description": "KBO 기본 규정 (리그 구성, 경기 시간, 타이브레이크 등)",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_rulebook"
        / "league_regulations"
        / "01_regular_season.md",
        "source_table": "kbo_regulations",
        "title": "KBO 기본 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_player": {
        "description": "KBO 선수 규정 (등록, FA, 외국인선수, 드래프트 등)",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_rulebook"
        / "player_regulations"
        / "README.md",
        "source_table": "kbo_regulations",
        "title": "KBO 선수 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_game": {
        "description": "KBO 경기 규정 (경기 진행, 방해, 보크, 홈런 등)",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_rulebook"
        / "baseball_rules"
        / "README.md",
        "source_table": "kbo_regulations",
        "title": "KBO 경기 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_technical": {
        "description": "KBO 기술 규정 (기록, 통계, 심판, 용품 등)",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_rulebook"
        / "scoring_rules"
        / "README.md",
        "source_table": "kbo_regulations",
        "title": "KBO 기술 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_discipline": {
        "description": "KBO 징계 규정 (폭력, 도박, 약물, 처벌 기준 등)",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_rulebook"
        / "disciplinary_regulations"
        / "README.md",
        "source_table": "kbo_regulations",
        "title": "KBO 징계 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_postseason": {
        "description": "KBO 포스트시즌 규정 (플레이오프, 와일드카드, 한국시리즈 등)",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_rulebook"
        / "league_regulations"
        / "02_postseason.md",
        "source_table": "kbo_regulations",
        "title": "KBO 포스트시즌 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_special": {
        "description": "KBO 특별 규정 (코로나19, 기상이변, 비상상황 등)",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_rulebook"
        / "league_regulations"
        / "03_special_regulations.md",
        "source_table": "kbo_regulations",
        "title": "KBO 특별 규정",
        "pk_hint": ["title"],
    },
    "kbo_regulations_terms": {
        "description": "KBO 야구 용어 정의 (기본 용어, 통계 지표, 포지션 등)",
        "source_file": Path(__file__).parent.parent
        / "docs"
        / "kbo_rulebook"
        / "glossary.md",
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
        "since_filter_column": "bs.updated_at",
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
        "since_filter_column": "ps.updated_at",
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
                gm.updated_at AS game_updated_at,
                ht.team_name AS home_team_name,
                at.team_name AS away_team_name
            FROM game g
            LEFT JOIN kbo_seasons ks
              ON ks.season_id = g.season_id
            LEFT JOIN game_metadata gm
              ON gm.game_id = g.game_id
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
        "since_filter_column": "gm.updated_at",
        "watermark_fields": ("game_updated_at",),
    },
    "game_flow_summary": {
        "description": "KBO 경기 흐름 요약",
        "source_table": "game_flow_summary",
        "title_fields": [
            ["game_date"],
            ["away_team_name", "away_team"],
            ["home_team_name", "home_team"],
        ],
        "select_sql": """
            SELECT *
            FROM (
                WITH complete_games AS (
                    SELECT
                        gis.game_id
                    FROM game_inning_scores gis
                    GROUP BY gis.game_id
                    HAVING COUNT(*) > 0
                       AND COUNT(*) = COUNT(DISTINCT gis.inning::text || ':' || gis.team_side)
                       AND COUNT(*) = 2 * COUNT(DISTINCT gis.inning)
                ),
                inning_pairs AS (
                    SELECT
                        gis.game_id,
                        gis.inning,
                        MAX(CASE WHEN gis.team_side = 'home' THEN gis.runs END) AS home_runs,
                        MAX(CASE WHEN gis.team_side = 'away' THEN gis.runs END) AS away_runs,
                        BOOL_OR(gis.is_extra IS TRUE) AS is_extra,
                        MAX(gis.updated_at) AS updated_at
                    FROM game_inning_scores gis
                    JOIN complete_games cg ON cg.game_id = gis.game_id
                    GROUP BY gis.game_id, gis.inning
                )
                SELECT
                    g.game_id,
                    g.game_date,
                    ks.season_year,
                    g.season_id,
                    ks.league_type_code,
                    g.home_team,
                    g.away_team,
                    ht.team_name AS home_team_name,
                    at.team_name AS away_team_name,
                    g.home_score,
                    g.away_score,
                    g.winning_team,
                    json_agg(
                        json_build_object(
                            'inning', ip.inning,
                            'home_runs', COALESCE(ip.home_runs, 0),
                            'away_runs', COALESCE(ip.away_runs, 0),
                            'is_extra', COALESCE(ip.is_extra, false)
                        )
                        ORDER BY ip.inning
                    ) AS inning_lines_json,
                    COALESCE(
                        MAX(ip.updated_at),
                        TIMESTAMP '1970-01-01'
                    ) AS latest_updated_at
                FROM game g
                JOIN complete_games cg ON cg.game_id = g.game_id
                JOIN inning_pairs ip ON ip.game_id = g.game_id
                LEFT JOIN kbo_seasons ks ON ks.season_id = g.season_id
                LEFT JOIN teams ht ON ht.team_id = g.home_team
                LEFT JOIN teams at ON at.team_id = g.away_team
                WHERE g.home_score IS NOT NULL
                  AND g.away_score IS NOT NULL
                GROUP BY
                    g.game_id,
                    g.game_date,
                    ks.season_year,
                    g.season_id,
                    ks.league_type_code,
                    g.home_team,
                    g.away_team,
                    ht.team_name,
                    at.team_name,
                    g.home_score,
                    g.away_score,
                    g.winning_team
            ) game_flow_rows
            ORDER BY game_date DESC, game_id
        """,
        "highlights": [
            ("경기일", ["game_date"]),
            ("경기", ["game_id"]),
            ("원정팀", ["away_team_name", "away_team"]),
            ("홈팀", ["home_team_name", "home_team"]),
            ("원정팀 점수", ["away_score"]),
            ("홈팀 점수", ["home_score"]),
            ("승리팀", ["winning_team"]),
        ],
        "pk_columns_override": ["game_id"],
        "pk_hint": ["game_id"],
        "single_chunk": True,
        "renderer": render_game_flow_summary,
        "season_filter_column": "season_year",
        "since_filter_column": "latest_updated_at",
    },
    "game_batting_stats": {
        "description": "경기별 타자 기록",
        "kind": "batting_game",
        "title_fields": [
            ["game_date"],
            ["team_name", "team_code"],
            ["player_name", "player_id"],
        ],
        "select_sql": """
            SELECT
                gbs.*,
                g.game_date,
                ks.season_year,
                ks.league_type_code,
                pb.name AS player_name,
                t.team_name
            FROM game_batting_stats gbs
            LEFT JOIN game g ON g.game_id = gbs.game_id
            LEFT JOIN kbo_seasons ks ON ks.season_id = g.season_id
            LEFT JOIN player_basic pb ON pb.player_id = gbs.player_id
            LEFT JOIN teams t ON t.team_id = gbs.team_code
            ORDER BY g.game_date DESC, gbs.team_code, gbs.batting_order
        """,
        "highlights": [
            ("경기", ["game_id"]),
            ("경기일", ["game_date"]),
            ("팀", ["team_name", "team_code"]),
            ("선수", ["player_name", "player_id"]),
            ("타순", ["batting_order"]),
            ("포지션", ["position"]),
            ("타수", ["at_bats", "ab"]),
            ("안타", ["hits", "h"]),
            ("득점", ["runs", "r"]),
            ("타점", ["rbi"]),
            ("홈런", ["home_runs", "hr"]),
            ("도루", ["stolen_bases", "sb"]),
            ("볼넷", ["walks", "bb"]),
            ("삼진", ["strikeouts", "so"]),
        ],
        "pk_hint": ["game_id", "player_id", "team_code"],
        "season_filter_column": "ks.season_year",
        "since_filter_column": "gbs.updated_at",
        "renderer": render_hitter_game,
    },
    "game_pitching_stats": {
        "description": "경기별 투수 기록",
        "kind": "pitching_game",
        "title_fields": [
            ["game_date"],
            ["team_name", "team_code"],
            ["player_name", "player_id"],
        ],
        "select_sql": """
            SELECT
                gps.*,
                g.game_date,
                ks.season_year,
                ks.league_type_code,
                pb.name AS player_name,
                t.team_name
            FROM game_pitching_stats gps
            LEFT JOIN game g ON g.game_id = gps.game_id
            LEFT JOIN kbo_seasons ks ON ks.season_id = g.season_id
            LEFT JOIN player_basic pb ON pb.player_id = gps.player_id
            LEFT JOIN teams t ON t.team_id = gps.team_code
            ORDER BY g.game_date DESC, gps.team_code, gps.appearance_seq
        """,
        "highlights": [
            ("경기", ["game_id"]),
            ("경기일", ["game_date"]),
            ("팀", ["team_name", "team_code"]),
            ("선수", ["player_name", "player_id"]),
            ("등판 순서", ["appearance_seq"]),
            ("결과", ["result"]),
            ("이닝", ["innings_pitched", "ip"]),
            ("실점", ["earned_runs", "er"]),
            ("피안타", ["hits_allowed", "h"]),
            ("볼넷", ["walks", "bb"]),
            ("탈삼진", ["strikeouts", "so"]),
        ],
        "pk_hint": ["game_id", "player_id", "team_code"],
        "season_filter_column": "ks.season_year",
        "since_filter_column": "gps.updated_at",
        "renderer": render_pitcher_game,
    },
    "game_inning_scores": {
        "description": "이닝별 점수 기록",
        "title_fields": [
            ["game_date"],
            ["home_team_name", "home_team"],
            ["away_team_name", "away_team"],
        ],
        "select_sql": """
            SELECT
                gis.*,
                g.game_date,
                g.home_team,
                g.away_team,
                ks.season_year,
                ks.league_type_code,
                ht.team_name AS home_team_name,
                at.team_name AS away_team_name,
                g.home_score,
                g.away_score
            FROM game_inning_scores gis
            LEFT JOIN game g ON g.game_id = gis.game_id
            LEFT JOIN kbo_seasons ks ON ks.season_id = g.season_id
            LEFT JOIN teams ht ON ht.team_id = g.home_team
            LEFT JOIN teams at ON at.team_id = g.away_team
            ORDER BY g.game_date DESC, gis.game_id, gis.inning
        """,
        "highlights": [
            ("경기", ["game_id"]),
            ("경기일", ["game_date"]),
            ("이닝", ["inning"]),
            ("홈팀", ["home_team_name", "home_team"]),
            ("홈팀 점수", ["home_score"]),
            ("원정팀", ["away_team_name", "away_team"]),
            ("원정팀 점수", ["away_score"]),
        ],
        "pk_hint": ["game_id", "inning"],
        "season_filter_column": "ks.season_year",
        "since_filter_column": "gis.updated_at",
    },
    "game_lineups": {
        "description": "경기 라인업 정보",
        "title_fields": [
            ["game_date"],
            ["team_name", "team_code"],
        ],
        "select_sql": """
            SELECT
                gl.*,
                g.game_date,
                ks.season_year,
                ks.league_type_code,
                pb.name AS player_name,
                t.team_name
            FROM game_lineups gl
            LEFT JOIN game g ON g.game_id = gl.game_id
            LEFT JOIN kbo_seasons ks ON ks.season_id = g.season_id
            LEFT JOIN player_basic pb ON pb.player_id = gl.player_id
            LEFT JOIN teams t ON t.team_id = gl.team_code
            ORDER BY g.game_date DESC, gl.game_id, gl.team_code, gl.batting_order
        """,
        "highlights": [
            ("경기", ["game_id"]),
            ("경기일", ["game_date"]),
            ("팀", ["team_name", "team_code"]),
            ("선수", ["player_name", "player_id"]),
            ("타순", ["batting_order"]),
            ("포지션", ["position"]),
        ],
        "pk_hint": ["game_id", "team_code", "batting_order"],
        "season_filter_column": "ks.season_year",
        "since_filter_column": "gl.updated_at",
    },
    "game_metadata": {
        "description": "경기 메타데이터 (심판, 날씨, 관중)",
        "title_fields": [
            ["game_date"],
            ["home_team_name", "home_team"],
            ["away_team_name", "away_team"],
        ],
        "select_sql": """
            SELECT
                gm.*,
                gm.game_time_minutes AS game_duration,
                g.game_date,
                g.home_team,
                g.away_team,
                ks.season_year,
                ks.league_type_code,
                ht.team_name AS home_team_name,
                at.team_name AS away_team_name
            FROM game_metadata gm
            LEFT JOIN game g ON g.game_id = gm.game_id
            LEFT JOIN kbo_seasons ks ON ks.season_id = g.season_id
            LEFT JOIN teams ht ON ht.team_id = g.home_team
            LEFT JOIN teams at ON at.team_id = g.away_team
            ORDER BY g.game_date DESC
        """,
        "highlights": [
            ("경기", ["game_id"]),
            ("경기일", ["game_date"]),
            ("홈팀", ["home_team_name", "home_team"]),
            ("원정팀", ["away_team_name", "away_team"]),
            ("관중", ["attendance", "crowd"]),
            ("경기 시간", ["game_duration", "game_time"]),
            ("날씨", ["weather"]),
        ],
        "pk_hint": ["game_id"],
        "season_filter_column": "ks.season_year",
        "since_filter_column": "gm.updated_at",
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
        "since_filter_column": "updated_at",
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
        "since_filter_column": "updated_at",
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
                th.season AS season_year,
                th.stadium AS stadium_name,
                t.team_name AS current_team_name
            FROM team_history th
            LEFT JOIN teams t
              ON t.team_id = th.team_code
            ORDER BY th.team_code, th.season
        """,
        "highlights": [
            ("구단", ["team_name", "team_code"]),
            ("시즌", ["season", "season_year"]),
            ("순위", ["ranking"]),
            ("도시", ["city"]),
            ("주경기장", ["stadium", "stadium_name"]),
        ],
        "pk_hint": ["team_code", "season"],
        "season_filter_column": "th.season",
        "since_filter_column": "th.updated_at",
    },
    "awards": {
        "description": "KBO 수상 기록 (MVP, 신인왕, 골든글러브)",
        "title_fields": [
            ["award_year"],
            ["award_type"],
            ["player_name"],
        ],
        "select_sql": """
            SELECT
                a.*,
                a.award_year AS season_year,
                a.team_name
            FROM awards a
            ORDER BY a.award_year DESC, a.award_type, a.player_name
        """,
        "highlights": [
            ("시즌", ["award_year", "season_year"]),
            ("수상 종류", ["award_type"]),
            ("선수", ["player_name"]),
            ("팀", ["team_name", "team"]),
            ("포지션", ["position"]),
        ],
        "pk_hint": ["id", "award_year", "award_type", "player_name"],
        "season_filter_column": "a.award_year",
        "since_filter_column": "a.updated_at",
    },
    "player_movements": {
        "description": "선수 이동 기록 (FA, 트레이드, 드래프트)",
        "title_fields": [
            ["date"],
            ["section"],
            ["player_name"],
        ],
        "select_sql": """
            SELECT
                pm.*,
                EXTRACT(YEAR FROM pm.date) AS season_year,
                t.team_name
            FROM player_movements pm
            LEFT JOIN teams t ON t.team_id = pm.team_code
            ORDER BY pm.date DESC, pm.player_name
        """,
        "highlights": [
            ("날짜", ["date"]),
            ("이동 유형", ["section"]),
            ("선수", ["player_name"]),
            ("팀", ["team_name", "team_code"]),
            ("비고", ["remarks"]),
        ],
        "pk_hint": ["id", "date", "player_name"],
        "season_filter_column": "EXTRACT(YEAR FROM pm.date)",
        "since_filter_column": "pm.updated_at",
    },
    "team_franchises": {
        "description": "KBO 프랜차이즈 그룹 정보",
        "title_fields": [
            ["name"],
            ["current_code"],
        ],
        "select_sql": """
            SELECT
                tf.*,
                (SELECT STRING_AGG(t.team_name, ', ' ORDER BY t.team_id)
                 FROM teams t WHERE t.franchise_id = tf.id) AS member_teams
            FROM team_franchises tf
            ORDER BY tf.id
        """,
        "highlights": [
            ("프랜차이즈", ["name"]),
            ("원래 코드", ["original_code"]),
            ("현재 코드", ["current_code"]),
            ("소속 팀", ["member_teams"]),
            ("공식 사이트", ["web_url"]),
        ],
        "pk_hint": ["id"],
        "season_filter_column": None,
        "since_filter_column": "tf.updated_at",
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
              ON t.team_name = pb.team
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
        "since_filter_column": None,
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
        "since_filter_column": None,
    },
    "team_season_batting": {
        "description": "KBO 팀 시즌 타격 기록",
        "kind": "team_batting_season",
        "title_fields": [
            ["season"],
            ["team_name", "team_id"],
        ],
        "select_sql": """
            SELECT
                tsb.*,
                t.team_name AS display_team_name
            FROM team_season_batting tsb
            LEFT JOIN teams t ON t.team_id = tsb.team_id
            ORDER BY tsb.season DESC, tsb.team_id
        """,
        "highlights": [
            ("시즌", ["season"]),
            ("팀", ["team_name", "team_id"]),
            ("경기", ["games"]),
            ("타율", ["avg"]),
            ("OPS", ["ops"]),
            ("홈런", ["home_runs"]),
            ("타점", ["rbi"]),
            ("도루", ["stolen_bases"]),
        ],
        "pk_hint": ["id", "team_id", "season", "league"],
        "renderer": render_team_batting_season,
        "season_filter_column": "tsb.season",
        "since_filter_column": "tsb.updated_at",
    },
    "team_season_pitching": {
        "description": "KBO 팀 시즌 투수 기록",
        "kind": "team_pitching_season",
        "title_fields": [
            ["season"],
            ["team_name", "team_id"],
        ],
        "select_sql": """
            SELECT
                tsp.*,
                t.team_name AS display_team_name
            FROM team_season_pitching tsp
            LEFT JOIN teams t ON t.team_id = tsp.team_id
            ORDER BY tsp.season DESC, tsp.team_id
        """,
        "highlights": [
            ("시즌", ["season"]),
            ("팀", ["team_name", "team_id"]),
            ("경기", ["games"]),
            ("승", ["wins"]),
            ("패", ["losses"]),
            ("ERA", ["era"]),
            ("WHIP", ["whip"]),
            ("탈삼진", ["strikeouts"]),
        ],
        "pk_hint": ["id", "team_id", "season", "league"],
        "renderer": render_team_pitching_season,
        "season_filter_column": "tsp.season",
        "since_filter_column": "tsp.updated_at",
    },
    "stat_rankings": {
        "description": "KBO 시즌 부문별 리더 순위",
        "title_fields": [
            ["season"],
            ["metric"],
            ["entity_label"],
        ],
        "select_sql": """
            SELECT
                sr.*,
                sr.season AS season_year,
                t.team_name
            FROM stat_rankings sr
            LEFT JOIN teams t ON t.team_id = sr.team_id
            ORDER BY sr.season DESC, sr.metric, sr.rank
        """,
        "highlights": [
            ("시즌", ["season"]),
            ("부문", ["metric"]),
            ("선수", ["entity_label", "entity_id"]),
            ("팀", ["team_name", "team_id"]),
            ("기록", ["value"]),
            ("순위", ["rank"]),
        ],
        "pk_hint": ["id", "season", "metric", "entity_id"],
        "renderer": render_stat_ranking,
        "season_filter_column": "sr.season",
        "since_filter_column": "sr.updated_at",
    },
    "game_summary": {
        "description": "경기 요약 정보 (승리 타점, 홈런 등 주요 기록 설명)",
        "title_fields": [
            ["game_id"],
            ["summary_type"],
            ["player_name"],
        ],
        "select_sql": """
            SELECT 
                gs.*,
                g.game_date,
                ks.season_year,
                ks.league_type_code,
                t.team_name
            FROM game_summary gs
            LEFT JOIN game g ON g.game_id = gs.game_id
            LEFT JOIN kbo_seasons ks ON ks.season_id = g.season_id
            LEFT JOIN teams t ON (t.team_id = g.home_team OR t.team_id = g.away_team)
            ORDER BY g.game_date DESC, gs.game_id, gs.id
        """,
        "checkpoint_select_sql": """
            SELECT
                gs.*,
                g.game_date,
                ks.season_year,
                ks.league_type_code,
                checkpoint_team.team_name
            FROM game_summary gs
            LEFT JOIN game g ON g.game_id = gs.game_id
            LEFT JOIN kbo_seasons ks ON ks.season_id = g.season_id
            LEFT JOIN LATERAL (
                SELECT t.team_name
                FROM teams t
                WHERE t.team_id = g.home_team OR t.team_id = g.away_team
                ORDER BY
                    CASE WHEN t.team_id = g.home_team THEN 0 ELSE 1 END,
                    t.team_id ASC
                LIMIT 1
            ) checkpoint_team ON TRUE
            ORDER BY g.game_date DESC, gs.game_id, gs.id
        """,
        "highlights": [
            ("경기 ID", ["game_id"]),
            ("구분", ["summary_type"]),
            ("선수", ["player_name", "player_id"]),
            ("내용", ["detail_text"]),
        ],
        "pk_hint": ["id"],
        "season_filter_column": "ks.season_year",
        "since_filter_column": "gs.updated_at",
        "date_to_exclusive_filter_column": "g.game_date",
    },
}

DATABASE_WATERMARK_FIELDS = {
    "player_season_batting": ("updated_at",),
    "player_season_pitching": ("updated_at",),
    "game": ("game_updated_at",),
    "game_flow_summary": ("latest_updated_at",),
    "game_batting_stats": ("updated_at",),
    "game_pitching_stats": ("updated_at",),
    "game_inning_scores": ("updated_at",),
    "game_lineups": ("updated_at",),
    "game_metadata": ("updated_at",),
    "kbo_seasons": (),
    "stadiums": ("updated_at",),
    "teams": ("updated_at",),
    "team_history": ("updated_at",),
    "team_name_mapping": (),
    "awards": ("updated_at",),
    "player_movements": ("updated_at",),
    "team_franchises": ("updated_at",),
    "player_basic": (),
    "team_profiles": (),
    "team_season_batting": ("updated_at",),
    "team_season_pitching": ("updated_at",),
    "stat_rankings": ("updated_at",),
    "game_summary": ("updated_at",),
}

for source_table, watermark_fields in DATABASE_WATERMARK_FIELDS.items():
    TABLE_PROFILES[source_table]["watermark_fields"] = watermark_fields

CUSTOM_CHECKPOINT_ORDERS = {
    "player_season_batting": (("id", "integer"),),
    "player_season_pitching": (("id", "integer"),),
    "game": (("id", "integer"),),
    "game_flow_summary": (("game_id", "text"),),
    "game_batting_stats": (("id", "integer"),),
    "game_pitching_stats": (("id", "integer"),),
    "game_inning_scores": (("id", "integer"),),
    "game_lineups": (("id", "integer"),),
    "game_metadata": (("game_id", "text"),),
    "team_history": (("id", "integer"),),
    "awards": (("id", "integer"),),
    "player_movements": (("id", "integer"),),
    "team_franchises": (("id", "integer"),),
    "player_basic": (("player_id", "text"),),
    "team_name_mapping": (("full_name", "text"),),
    "team_profiles": (("id", "integer"),),
    "team_season_batting": (("id", "integer"),),
    "team_season_pitching": (("id", "integer"),),
    "stat_rankings": (("id", "integer"),),
    "game_summary": (("id", "integer"),),
}

for source_table, checkpoint_order in CUSTOM_CHECKPOINT_ORDERS.items():
    TABLE_PROFILES[source_table]["checkpoint_order"] = checkpoint_order
    TABLE_PROFILES[source_table]["checkpoint_query_version"] = "1"

# Tables the caller can choose. `rag_chunks` intentionally 제외.
# team_daily_roster는 데이터가 많아 스킵
DEFAULT_TABLES = list(TRUSTED_INGEST_SOURCE_TABLES)

TARGET_RPM = 10
MIN_DELAY_SECONDS = 60 / TARGET_RPM
PGVECTOR_SEARCH_PATH = "public, extensions, security"

STADIUM_ID_ALIAS_MAP: Dict[str, str] = {
    "CHANGWON": "NCPARK",
    "DAEGU": "LIONS2",
    "DAEJEON": "EAGLES2",
    "GWANGJU": "CHAMPIONS",
    "INCHEON": "SSGLANDERS",
    "SUWON": "KTWIZ",
}

# source_row_id를 안정적으로 고정하기 위한 canonical key 규칙.
# 값이 누락되면 기존 fallback 빌더를 사용한다.
CANONICAL_SOURCE_ROW_KEYS: Dict[str, Sequence[str]] = {
    "kbo_seasons": ("season_id",),
    "team_history": ("id",),
    "awards": ("id",),
    "player_season_batting": ("id",),
    "player_season_pitching": ("id",),
    "game": ("id",),
    "game_flow_summary": ("game_id",),
    "game_metadata": ("game_id",),
    "game_inning_scores": ("id",),
    "game_lineups": ("id",),
    "game_batting_stats": ("id",),
    "game_pitching_stats": ("id",),
    "game_summary": ("id",),
    "teams": ("team_id",),
    "team_franchises": ("id",),
    "stadiums": ("stadium_id",),
    "player_basic": ("player_id",),
    "player_movements": ("id",),
    "team_season_batting": ("id",),
    "team_season_pitching": ("id",),
    "stat_rankings": ("id",),
}




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


def coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
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


def resolve_primary_key_columns(
    conn,
    table: str,
    profile: Optional[Dict[str, Any]] = None,
) -> List[str]:
    override = list((profile or {}).get("pk_columns_override") or [])
    if override:
        return override
    return get_primary_key_columns(conn, table)


def _postgres_type_to_cursor_type(pg_type: str) -> CursorScalarType:
    normalized = pg_type.lower()
    if normalized in {"smallint", "integer", "bigint"}:
        return "integer"
    if normalized.startswith(("numeric", "decimal")):
        return "decimal"
    if normalized == "date":
        return "date"
    if re.fullmatch(r"timestamp(?:\([0-6]\))? with time zone", normalized):
        return "datetime"
    if re.fullmatch(r"timestamp(?:\([0-6]\))? without time zone", normalized):
        return "datetime_naive"
    if normalized == "uuid":
        return "uuid"
    if normalized in {"text", "character", "character varying"} or normalized.startswith(
        ("character(", "character varying(")
    ):
        return "text"
    if normalized == "boolean":
        return "boolean"
    raise IngestCheckpointCursorTypeError(f"unsupported primary-key type: {pg_type}")


def resolve_checkpoint_order(
    conn,
    table_name: str,
    profile: Dict[str, Any],
) -> CheckpointOrder:
    if profile.get("select_sql"):
        configured_order = profile.get("checkpoint_order")
        if configured_order:
            return CheckpointOrder(
                source_table=table_name,
                fields=tuple(
                    CheckpointOrderField(name, scalar_type)
                    for name, scalar_type in configured_order
                ),
                query_version=str(profile.get("checkpoint_query_version", "1")),
            )
        raise IngestCheckpointCursorUnavailableError(
            f"custom source query has no checkpoint order: {table_name}"
        )
    if conn is None:
        raise IngestCheckpointCursorUnavailableError(
            f"database connection is required to inspect primary key: {table_name}"
        )

    query = """
        SELECT
            a.attname,
            format_type(a.atttypid, a.atttypmod),
            a.attnotnull
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
        WHERE c.relname = %s AND n.nspname = %s AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
    """
    with conn.cursor() as cur:
        cur.execute(query, (table_name, "public"))
        rows = cur.fetchall()

    if not rows:
        raise IngestCheckpointCursorUnavailableError(
            f"source table has no primary key: {table_name}"
        )
    if any(not attnotnull for _, _, attnotnull in rows):
        raise IngestCheckpointCursorUnavailableError(
            f"primary-key cursor contains nullable fields: {table_name}"
        )
    return CheckpointOrder(
        source_table=table_name,
        fields=tuple(
            CheckpointOrderField(name, _postgres_type_to_cursor_type(pg_type))
            for name, pg_type, _ in rows
        ),
    )


def _normalize_row_id_value(table: str, key: str, value: Any) -> Optional[str]:
    if value in (None, ""):
        return None

    normalized: str
    if isinstance(value, str):
        normalized = value.strip()
    else:
        normalized = str(value)

    if table == "stadiums" and key == "stadium_id":
        normalized = STADIUM_ID_ALIAS_MAP.get(normalized, normalized)

    return normalized or None


def build_canonical_source_row_id(row: Dict[str, Any], table: str) -> Optional[str]:
    keys = CANONICAL_SOURCE_ROW_KEYS.get(table)
    if not keys:
        return None

    parts: List[str] = []
    for key in keys:
        if key not in row:
            return None
        normalized_value = _normalize_row_id_value(table, key, row.get(key))
        if normalized_value is None:
            return None
        parts.append(f"{key}={normalized_value}")
    return "|".join(parts)


def _relative_source_file_path(source_file: Path) -> str:
    resolved = Path(source_file).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _default_static_knowledge_type(source_table: str) -> str:
    if source_table == "kbo_regulations":
        return "rules_terms"
    if source_table == "kbo_definitions":
        return "strategy_metrics"
    return "document"


def build_static_source_row_prefix(profile_key: str, profile: Dict[str, Any]) -> str:
    source_table = str(profile.get("source_table", profile_key))
    relative_path = _relative_source_file_path(Path(profile["source_file"]))
    raw_slug = (
        relative_path[:-3] if relative_path.lower().endswith(".md") else relative_path
    )
    slug = re.sub(r"[^0-9a-z]+", "_", raw_slug.lower()).strip("_")
    if not slug:
        slug = hashlib.md5(relative_path.encode("utf-8")).hexdigest()[:12]
    return f"{source_table}:{slug}"


def build_static_source_row_id(
    profile_key: str,
    profile: Dict[str, Any],
    *,
    chunk_index: int,
    total_chunks: int,
) -> str:
    prefix = build_static_source_row_prefix(profile_key, profile)
    if total_chunks <= 1:
        return prefix
    return f"{prefix}#part{chunk_index}"


def build_static_chunk_meta(
    profile_key: str,
    profile: Dict[str, Any],
    *,
    chunk_index: int,
) -> Dict[str, Any]:
    source_table = str(profile.get("source_table", profile_key))
    relative_path = _relative_source_file_path(Path(profile["source_file"]))
    source_type = _static_source_type(source_table, relative_path, profile)
    source_uri = str(profile.get("source_uri") or f"file:{relative_path}")
    topic_key = str(
        profile.get("topic_key") or build_static_source_row_prefix(profile_key, profile)
    )
    return {
        "source_file": str(Path(profile["source_file"]).resolve()),
        "source_path": relative_path,
        "source_uri": source_uri,
        "source_type": source_type,
        "topic_key": topic_key,
        "source_profile": profile_key,
        "chunk_index": chunk_index,
        "league_scope": profile.get(
            "league_scope",
            "kbo+baseball" if source_table == "markdown_docs" else "kbo",
        ),
        "knowledge_type": profile.get(
            "knowledge_type",
            _default_static_knowledge_type(source_table),
        ),
        "freshness": profile.get("freshness", "evergreen"),
        "source_version": profile.get("source_version", "v1"),
        "visibility": profile.get("visibility", "public"),
        "valid_from": profile.get("valid_from"),
        "valid_to": profile.get("valid_to"),
        "expires_at": profile.get("expires_at"),
    }


def _static_source_type(
    source_table: str,
    relative_path: str,
    profile: Dict[str, Any],
) -> str:
    explicit = profile.get("source_type")
    if explicit:
        return str(explicit)
    normalized = relative_path.lower()
    if source_table in {"kbo_regulations", "kbo_definitions"}:
        return "official_rulebook"
    if "kbo_rulebook" in normalized or "rulebook" in normalized:
        return "official_rulebook"
    return "markdown_doc"


def _static_quality_score(source_type: str, profile: Dict[str, Any]) -> float:
    return infer_quality_score(
        source_type,
        str(profile.get("source_table") or ""),
        meta=profile,
        explicit=profile.get("quality_score"),
    )


def _kbo_topic_key(source_table: str, source_row_id: str) -> str:
    return f"{source_table}:{base_source_row_id(source_row_id)}"


def _manual_lineup_notes_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_notes = row.get("notes")
    if not raw_notes:
        return {}
    if isinstance(raw_notes, dict):
        parsed = raw_notes
    else:
        try:
            parsed = json.loads(str(raw_notes))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    if not isinstance(parsed, dict):
        return {}
    if parsed.get("source_type") != "manual_lineup":
        return {}
    return dict(parsed)


def _manual_lineup_quality_score(manual_meta: Dict[str, Any]) -> float:
    confidence = coerce_float(manual_meta.get("confidence"))
    if confidence is None:
        confidence = 0.50
    return max(0.0, min(confidence, 0.70))


def read_static_profile_content(profile_key: str, profile: Dict[str, Any]) -> str:
    source_file = Path(profile["source_file"]).resolve()
    try:
        return source_file.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Static source file not found for profile '{profile_key}': {source_file}"
        ) from exc


def build_static_profile_chunk_payloads(
    profile_key: str,
    profile: Dict[str, Any],
    *,
    settings: Any,
    content: Optional[str] = None,
) -> List[ChunkPayload]:
    raw_content = (
        content
        if content is not None
        else read_static_profile_content(profile_key, profile)
    )
    normalized_content = raw_content.strip()
    if not normalized_content:
        return []

    if profile.get("single_chunk"):
        chunks = [normalized_content]
    else:
        chunks = smart_chunks(raw_content, settings=settings)
    if not chunks:
        return []

    total_chunks = len(chunks)
    source_table = str(profile.get("source_table", profile_key))
    season_year = int(profile.get("season_year", 0) or 0)
    title = str(profile.get("title") or profile_key)
    first_meta = build_static_chunk_meta(profile_key, profile, chunk_index=1)
    source_type = str(first_meta["source_type"])
    source_uri = str(first_meta["source_uri"])
    topic_key = str(first_meta["topic_key"])
    quality_score = _static_quality_score(source_type, profile)

    return [
        ChunkPayload(
            table=source_table,
            source_row_id=build_static_source_row_id(
                profile_key,
                profile,
                chunk_index=idx,
                total_chunks=total_chunks,
            ),
            title=title,
            content=chunk,
            season_year=season_year,
            season_id=None,
            league_type_code=0,
            team_id=None,
            player_id=None,
            meta=build_static_chunk_meta(
                profile_key,
                profile,
                chunk_index=idx,
            ),
            source_type=source_type,
            source_uri=source_uri,
            topic_key=topic_key,
            valid_from=profile.get("valid_from"),
            valid_to=profile.get("valid_to"),
            expires_at=profile.get("expires_at"),
            quality_score=quality_score,
        )
        for idx, chunk in enumerate(chunks, start=1)
    ]


def build_source_row_id(
    row: Dict[str, Any],
    table: str,
    pk_columns: Sequence[str],
    pk_hint: Sequence[str],
) -> str:
    canonical_row_id = build_canonical_source_row_id(row, table)
    if canonical_row_id:
        return canonical_row_id

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


def _find_top_level_keyword_positions(sql_text: str, keyword: str) -> List[int]:
    positions: List[int] = []
    upper = sql_text.upper()
    keyword_upper = keyword.upper()
    keyword_len = len(keyword_upper)
    depth = 0
    in_single_quote = False
    single_quote_backslash_escapes = False
    in_double_quote = False
    in_line_comment = False
    block_comment_depth = 0
    dollar_quote_delimiter: Optional[str] = None
    idx = 0

    while idx < len(sql_text):
        char = sql_text[idx]

        if in_line_comment:
            if char in "\r\n":
                in_line_comment = False
            idx += 1
            continue

        if block_comment_depth:
            if sql_text.startswith("/*", idx):
                block_comment_depth += 1
                idx += 2
                continue
            if sql_text.startswith("*/", idx):
                block_comment_depth -= 1
                idx += 2
                continue
            idx += 1
            continue

        if dollar_quote_delimiter is not None:
            if sql_text.startswith(dollar_quote_delimiter, idx):
                idx += len(dollar_quote_delimiter)
                dollar_quote_delimiter = None
            else:
                idx += 1
            continue

        if in_single_quote:
            if (
                single_quote_backslash_escapes
                and char == "\\"
                and idx + 1 < len(sql_text)
            ):
                idx += 2
                continue
            if char == "'" and idx + 1 < len(sql_text) and sql_text[idx + 1] == "'":
                idx += 2
                continue
            if char == "'":
                in_single_quote = False
                single_quote_backslash_escapes = False
            idx += 1
            continue

        if in_double_quote:
            if char == '"' and idx + 1 < len(sql_text) and sql_text[idx + 1] == '"':
                idx += 2
                continue
            if char == '"':
                in_double_quote = False
            idx += 1
            continue

        if sql_text.startswith("--", idx):
            in_line_comment = True
            idx += 2
            continue

        if sql_text.startswith("/*", idx):
            block_comment_depth = 1
            idx += 2
            continue

        if char == "$" and (
            idx == 0
            or re.fullmatch(r"[A-Za-z0-9_$]", sql_text[idx - 1]) is None
        ):
            delimiter_match = re.match(
                r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$", sql_text[idx:]
            )
            if delimiter_match is not None:
                dollar_quote_delimiter = delimiter_match.group(0)
                idx += len(dollar_quote_delimiter)
                continue

        if char == "'":
            in_single_quote = True
            prefix_idx = idx - 1
            single_quote_backslash_escapes = (
                prefix_idx >= 0
                and sql_text[prefix_idx] in {"e", "E"}
                and (
                    prefix_idx == 0
                    or not (
                        sql_text[prefix_idx - 1].isalnum()
                        or sql_text[prefix_idx - 1] in {"_", "$"}
                    )
                )
            )
            idx += 1
            continue

        if char == '"':
            in_double_quote = True
            idx += 1
            continue

        if char == "(":
            depth += 1
            idx += 1
            continue

        if char == ")":
            depth = max(0, depth - 1)
            idx += 1
            continue

        if depth == 0 and upper.startswith(keyword_upper, idx):
            prev_char = upper[idx - 1] if idx > 0 else " "
            next_pos = idx + keyword_len
            next_char = upper[next_pos] if next_pos < len(upper) else " "
            if not (prev_char.isalnum() or prev_char == "_") and not (
                next_char.isalnum() or next_char == "_"
            ):
                positions.append(idx)
                idx += keyword_len
                continue

        idx += 1

    return positions


# build_select_query가 프로필의 select_sql이 있으면 그 SQL에 season_year 등 필터를 주입하고 ORDER BY/ LIMIT를 붙임. 커스텀 SQL이 없으면 SELECT * FROM <table> + PK 순 정렬.
def _quoted_checkpoint_fields(order: CheckpointOrder) -> str:
    for field in order.fields:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field.name) is None:
            raise IngestCheckpointCursorUnavailableError("unsafe cursor field")
    return ", ".join(f'"{field.name}"' for field in order.fields)


def resolve_update_filter_column(
    profile: Mapping[str, Any],
    *,
    since: Optional[datetime],
) -> Optional[str]:
    if since is not None:
        column = profile.get("since_filter_column", "updated_at")
    else:
        column = profile.get("since_filter_column")
    if column is None:
        return None
    normalized = str(column).strip()
    return normalized or None


def build_select_query(
    table: str,
    profile: Dict[str, Any],
    pk_columns: Sequence[str],
    limit: Optional[int],
    season_year: Optional[int],
    since: Optional[datetime],
    source_updated_before: Optional[datetime] = None,
    date_to_exclusive: Optional[Any] = None,
    checkpoint_order: Optional[CheckpointOrder] = None,
    resume_cursor: Optional[CheckpointCursor] = None,
):
    custom_sql = profile.get("select_sql")
    if checkpoint_order is not None:
        custom_sql = profile.get("checkpoint_select_sql") or custom_sql
    season_filter_column = profile.get("season_filter_column", "season_year")
    since_filter_column = resolve_update_filter_column(profile, since=since)
    date_to_exclusive_filter_column = profile.get("date_to_exclusive_filter_column")
    if (
        checkpoint_order is not None
        and since_filter_column
        and source_updated_before is None
    ):
        raise IngestCheckpointIncompatibleError(
            "checkpointed update-filter query requires a source update cutoff"
        )
    params: List[Any] = []
    if custom_sql:
        stripped = custom_sql.strip()
        order_clause = ""
        order_matches = _find_top_level_keyword_positions(stripped, "ORDER BY")
        if order_matches:
            order_idx = order_matches[-1]
            base_sql = stripped[:order_idx].rstrip()
            order_clause = stripped[order_idx:]
        else:
            base_sql = stripped

        where_clauses = []
        if season_year is not None and season_filter_column:
            where_clauses.append(f"{season_filter_column} = %s")
            params.append(season_year)
        if since is not None and since_filter_column:
            where_clauses.append(f"{since_filter_column} >= %s")
            params.append(since)
        if (
            checkpoint_order is not None
            and source_updated_before is not None
            and since_filter_column
        ):
            if since is None:
                where_clauses.append(
                    f"({since_filter_column} IS NULL OR "
                    f"{since_filter_column} <= %s)"
                )
            else:
                where_clauses.append(f"{since_filter_column} <= %s")
            params.append(source_updated_before)
        if date_to_exclusive is not None and date_to_exclusive_filter_column:
            where_clauses.append(f"{date_to_exclusive_filter_column} < %s")
            params.append(date_to_exclusive)

        if where_clauses:
            has_where = bool(_find_top_level_keyword_positions(base_sql, "WHERE"))
            if has_where:
                base_sql = f"{base_sql}\n   AND {' AND '.join(where_clauses)}"
            else:
                base_sql = f"{base_sql}\nWHERE {' AND '.join(where_clauses)}"

        if checkpoint_order is not None:
            quoted_fields = _quoted_checkpoint_fields(checkpoint_order)
            if resume_cursor is not None and len(resume_cursor.values) != len(
                checkpoint_order.fields
            ):
                raise IngestCheckpointIncompatibleError("cursor arity mismatch")
            query = f"WITH checkpoint_source AS (\n{base_sql}\n)\nSELECT * FROM checkpoint_source"
            if resume_cursor is not None:
                placeholders = ", ".join("%s" for _ in checkpoint_order.fields)
                query = (
                    f"{query}\nWHERE ROW({quoted_fields}) > ROW({placeholders})"
                )
                params.extend(resume_cursor.values)
            ascending_order = ", ".join(
                f'"{field.name}" ASC' for field in checkpoint_order.fields
            )
            query = f"{query}\nORDER BY {ascending_order}"
        else:
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
    if since is not None and since_filter_column:
        column_name = str(since_filter_column).split(".")[-1]
        where_parts.append(sql.SQL("{} >= %s").format(sql.Identifier(column_name)))
        params.append(since)
    if (
        checkpoint_order is not None
        and source_updated_before is not None
        and since_filter_column
    ):
        column_name = str(since_filter_column).split(".")[-1]
        if since is None:
            identifier = sql.Identifier(column_name)
            where_parts.append(
                sql.SQL("({} IS NULL OR {} <= %s)").format(
                    identifier,
                    identifier,
                )
            )
        else:
            where_parts.append(
                sql.SQL("{} <= %s").format(sql.Identifier(column_name))
            )
        params.append(source_updated_before)
    if date_to_exclusive is not None and date_to_exclusive_filter_column:
        column_name = str(date_to_exclusive_filter_column).split(".")[-1]
        where_parts.append(sql.SQL("{} < %s").format(sql.Identifier(column_name)))
        params.append(date_to_exclusive)
    if checkpoint_order is not None:
        _quoted_checkpoint_fields(checkpoint_order)
        if resume_cursor is not None and len(resume_cursor.values) != len(
            checkpoint_order.fields
        ):
            raise IngestCheckpointIncompatibleError("cursor arity mismatch")
        if resume_cursor is not None:
            checkpoint_columns = sql.SQL(", ").join(
                sql.Identifier(field.name) for field in checkpoint_order.fields
            )
            checkpoint_placeholders = sql.SQL(", ").join(
                sql.SQL("%s") for _ in checkpoint_order.fields
            )
            where_parts.append(
                sql.SQL("ROW({}) > ROW({})").format(
                    checkpoint_columns,
                    checkpoint_placeholders,
                )
            )
            params.extend(resume_cursor.values)
    if where_parts:
        query = query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_parts)
    if checkpoint_order is not None:
        order_cols = sql.SQL(", ").join(
            sql.SQL("{} ASC").format(sql.Identifier(field.name))
            for field in checkpoint_order.fields
        )
        query = query + sql.SQL(" ORDER BY {}").format(order_cols)
    elif pk_columns:
        order_cols = sql.SQL(", ").join(sql.Identifier(col) for col in pk_columns)
        query = query + sql.SQL(" ORDER BY {}").format(order_cols)
    if limit is not None:
        query = query + sql.SQL(" LIMIT %s")
        params.append(limit)
    return query, tuple(params)


def batched(
    iterable: Sequence[ChunkPayload], size: int
) -> Iterable[List[ChunkPayload]]:
    total = len(iterable)
    for idx in range(0, total, size):
        yield list(iterable[idx : idx + size])


def should_run_row_stale_cleanup(
    *,
    since: Optional[datetime],
    limit: Optional[int],
    row_stale_cleanup: str,
) -> bool:
    return row_stale_cleanup in {"dry-run", "apply"} and since is None and limit is None




def _build_chunk_payload_dicts_for_row(
    table_name: str,
    row: Dict[str, Any],
    source_row_id: str,
    *,
    use_legacy_renderer: bool,
    today_str: str,
) -> List[Dict[str, Any]]:
    profile = TABLE_PROFILES.get(table_name, {})
    settings = get_settings()
    target_source_table = str(profile.get("source_table", table_name))
    title = build_title(row, table_name, source_row_id, profile)
    renderer = profile.get("renderer")

    if renderer and not use_legacy_renderer:
        enriched_row = dict(row)
        enriched_row["source_table"] = target_source_table
        enriched_row["source_row_id"] = source_row_id
        content = renderer(
            enriched_row,
            league_avg=None,
            percentiles=None,
            today_str=today_str,
        )
    else:
        content = build_content(row, table_name, source_row_id, profile)

    season_year = coerce_int(first_value(row, ["season_year", "season", "year"]))
    if season_year is None:
        season_year = 0

    season_id = coerce_int(first_value(row, ["season_id", "season_lookup_id"]))
    league_type_code = coerce_int(
        first_value(row, ["league_type_code", "league_type", "league"])
    )
    if league_type_code is None:
        league_type_code = 0

    team_id = first_value(
        row,
        ["team_id", "home_team_id", "away_team_id", "team", "team_code"],
    )
    player_id = first_value(row, ["player_id"])
    if profile.get("single_chunk"):
        chunks = [content.strip()] if content and content.strip() else []
    else:
        chunks = smart_chunks(content, settings=settings)
    if not chunks:
        return []

    manual_meta = (
        _manual_lineup_notes_meta(row) if target_source_table == "game_lineups" else {}
    )
    source_type = str(profile.get("source_type") or "kbo_db_table")
    source_uri = str(
        profile.get("source_uri") or f"db:{target_source_table}:{source_row_id}"
    )
    if manual_meta:
        source_type = "manual_lineup"
        source_uri = str(manual_meta.get("source_url") or source_uri)
    topic_key = str(
        profile.get("topic_key") or _kbo_topic_key(target_source_table, source_row_id)
    )
    if manual_meta:
        quality_score = _manual_lineup_quality_score(manual_meta)
    else:
        quality_score = infer_quality_score(
            source_type,
            target_source_table,
            meta=profile,
            explicit=profile.get("quality_score"),
        )
    valid_from = profile.get("valid_from")
    valid_to = profile.get("valid_to")
    expires_at = profile.get("expires_at")
    meta = dict(row)
    meta.setdefault("source_type", source_type)
    meta.setdefault("source_uri", source_uri)
    meta.setdefault("topic_key", topic_key)
    meta.setdefault("source_version", str(profile.get("source_version", "v1")))
    meta.setdefault("visibility", str(profile.get("visibility", "public")))
    meta.setdefault("valid_from", valid_from)
    meta.setdefault("valid_to", valid_to)
    meta.setdefault("expires_at", expires_at)
    if manual_meta:
        meta.update(manual_meta)
        meta["source_type"] = source_type
        meta["source_uri"] = source_uri
        meta["quality_score"] = quality_score

    payloads: List[Dict[str, Any]] = []
    if len(chunks) == 1:
        payloads.append(
            {
                "table": target_source_table,
                "source_row_id": source_row_id,
                "title": title,
                "content": chunks[0],
                "season_year": season_year,
                "season_id": season_id,
                "league_type_code": league_type_code,
                "team_id": str(team_id) if team_id is not None else None,
                "player_id": str(player_id) if player_id is not None else None,
                "meta": meta,
                "source_type": source_type,
                "source_uri": source_uri,
                "topic_key": topic_key,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "expires_at": expires_at,
                "quality_score": quality_score,
            }
        )
    else:
        for idx, chunk in enumerate(chunks, start=1):
            payloads.append(
                {
                    "table": target_source_table,
                    "source_row_id": f"{source_row_id}#part{idx}",
                    "title": f"{title} (분할 {idx})",
                    "content": chunk,
                    "season_year": season_year,
                    "season_id": season_id,
                    "league_type_code": league_type_code,
                    "team_id": str(team_id) if team_id is not None else None,
                    "player_id": str(player_id) if player_id is not None else None,
                    "meta": meta,
                    "source_type": source_type,
                    "source_uri": source_uri,
                    "topic_key": topic_key,
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "expires_at": expires_at,
                    "quality_score": quality_score,
                }
            )
    return payloads
























# 우선순위로 안전 변환(coerce_int/first_value).

