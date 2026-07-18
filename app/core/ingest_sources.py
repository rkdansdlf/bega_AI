"""Closed set of trusted internal sources eligible for RAG ingestion."""

from __future__ import annotations


TRUSTED_INGEST_SOURCE_TABLES: tuple[str, ...] = (
    "teams",
    "team_franchises",
    "team_history",
    "stadiums",
    "kbo_seasons",
    "player_basic",
    "awards",
    "player_movements",
    "player_season_batting",
    "player_season_pitching",
    "team_season_batting",
    "team_season_pitching",
    "stat_rankings",
    "game",
    "game_metadata",
    "game_flow_summary",
    "game_lineups",
    "game_batting_stats",
    "game_pitching_stats",
    "game_summary",
    "kbo_metrics_explained",
    "markdown_docs_rules_terms",
    "markdown_docs_strategy_metrics",
    "markdown_docs_culture_history",
    "markdown_docs_2025_storylines",
    "markdown_docs_chatbot_kb_v2",
    "kbo_regulations_basic",
    "kbo_regulations_player",
    "kbo_regulations_game",
    "kbo_regulations_technical",
    "kbo_regulations_discipline",
    "kbo_regulations_postseason",
    "kbo_regulations_special",
    "kbo_regulations_terms",
)
TRUSTED_INGEST_SOURCE_SET = frozenset(TRUSTED_INGEST_SOURCE_TABLES)
MAX_INGEST_TABLE_NAME_LENGTH = 128
MAX_INGEST_TABLES_PER_RUN = len(TRUSTED_INGEST_SOURCE_TABLES)


def normalize_ingest_source_table(value: object) -> str:
    """Return a bounded metric label for trusted ingestion sources."""

    normalized = str(value or "").strip()
    return normalized if normalized in TRUSTED_INGEST_SOURCE_SET else "other"
