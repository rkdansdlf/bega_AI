from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.oracle_baseball import (
    OracleBaseballPool,
    QueryDialect,
    is_oracle_baseball_connection,
)
from app.core.oracle_rag import OracleRagConnection
from app.tools.game_query import GameQueryTool

# ---------------------------------------------------------------------------
# QueryDialect
# ---------------------------------------------------------------------------


def test_postgres_dialect_bind_returns_positional_placeholder() -> None:
    dialect = QueryDialect(is_oracle=False)

    token = dialect.bind("LG")

    assert token == "%s"
    assert dialect.execute_args() == ["LG"]


def test_oracle_dialect_bind_returns_named_placeholder() -> None:
    dialect = QueryDialect(is_oracle=True)

    token = dialect.bind("LG", hint="team")

    assert token == ":team1"
    assert dialect.execute_args() == {"team1": "LG"}


def test_postgres_in_any_uses_a_single_array_bind() -> None:
    dialect = QueryDialect(is_oracle=False)

    fragment = dialect.in_any("g.home_team", ["LG", "HH", "OB"])

    assert fragment == "g.home_team = ANY(%s)"
    assert dialect.execute_args() == [["LG", "HH", "OB"]]


def test_oracle_in_any_expands_to_one_bind_per_value() -> None:
    dialect = QueryDialect(is_oracle=True)

    fragment = dialect.in_any("g.home_team", ["LG", "HH"], hint="v")

    assert fragment == "g.home_team IN (:v1, :v2)"
    assert dialect.execute_args() == {"v1": "LG", "v2": "HH"}


def test_in_any_with_no_values_matches_nothing_on_both_dialects() -> None:
    assert QueryDialect(is_oracle=False).in_any("g.home_team", []) == "1 = 0"
    assert QueryDialect(is_oracle=True).in_any("g.home_team", []) == "1 = 0"


def test_postgres_date_eq_uses_date_cast() -> None:
    dialect = QueryDialect(is_oracle=False)

    fragment = dialect.date_eq("g.game_date", "2025-10-01")

    assert fragment == "DATE(g.game_date) = %s"
    assert dialect.execute_args() == ["2025-10-01"]


def test_oracle_date_eq_uses_trunc_and_to_date() -> None:
    dialect = QueryDialect(is_oracle=True)

    fragment = dialect.date_eq("g.game_date", "2025-10-01", hint="d")

    assert fragment == "TRUNC(g.game_date) = TO_DATE(:d1, 'YYYY-MM-DD')"
    assert dialect.execute_args() == {"d1": "2025-10-01"}


def test_postgres_bind_date_is_a_plain_bind() -> None:
    dialect = QueryDialect(is_oracle=False)

    assert dialect.bind_date("2025-10-01") == "%s"
    assert dialect.execute_args() == ["2025-10-01"]


def test_oracle_bind_date_wraps_to_date() -> None:
    """Oracle has no implicit string->DATE conversion for bind variables — a
    bare bind against a DATE column raises ORA-01861. This is exactly the
    gap `BETWEEN`/`<` comparisons had before they used ``bind_date`` — only
    caught by running against a live ADB, since a fake cursor never
    validates a bind against a real DATE column."""
    dialect = QueryDialect(is_oracle=True)

    token = dialect.bind_date("2025-10-01", hint="d")

    assert token == "TO_DATE(:d1, 'YYYY-MM-DD')"
    assert dialect.execute_args() == {"d1": "2025-10-01"}


def test_postgres_limit_is_a_bound_literal() -> None:
    dialect = QueryDialect(is_oracle=False)

    assert dialect.limit(5) == "LIMIT %s"
    assert dialect.execute_args() == [5]


def test_oracle_limit_uses_fetch_first_rows_only() -> None:
    dialect = QueryDialect(is_oracle=True)

    fragment = dialect.limit(5, hint="lim")

    assert fragment == "FETCH FIRST :lim1 ROWS ONLY"
    assert dialect.execute_args() == {"lim1": 5}


def test_bind_order_matches_sql_text_order_for_postgres() -> None:
    """psycopg's %s is positional — params must line up with SQL left-to-right,
    not with the order dialect methods happen to be called in Python."""
    dialect = QueryDialect(is_oracle=False)
    home = dialect.in_any("g.home_team", ["LG"])
    away = dialect.in_any("g.away_team", ["HH"])
    sql = f"WHERE {home} AND {away}"

    assert sql == "WHERE g.home_team = ANY(%s) AND g.away_team = ANY(%s)"
    assert dialect.execute_args() == [["LG"], ["HH"]]


# ---------------------------------------------------------------------------
# is_oracle_baseball_connection / OracleBaseballPool
# ---------------------------------------------------------------------------


def test_is_oracle_baseball_connection_detects_the_shared_oracle_marker() -> None:
    assert is_oracle_baseball_connection(OracleRagConnection(object())) is True
    assert is_oracle_baseball_connection(SimpleNamespace()) is False


def test_oracle_baseball_pool_reports_closed_stats_before_open() -> None:
    pool = OracleBaseballPool("oracle+oracledb://user:pw@adb_medium", max_size=4)

    assert pool.backend == "oracle"
    assert pool.get_stats() == {"pool_open": False}


# ---------------------------------------------------------------------------
# GameQueryTool against a fake Oracle connection
# ---------------------------------------------------------------------------


class _FakeOracleCursor:
    def __init__(self, rows: list[tuple], names: list[str]) -> None:
        self.rows = rows
        self.description = [(name,) for name in names]
        self.executed: list[tuple[str, object]] = []

    async def execute(self, sql: str, params=None) -> None:  # noqa: ANN001
        self.executed.append((sql, params))

    async def fetchall(self):
        return self.rows

    async def fetchone(self):
        return self.rows[0] if self.rows else None

    async def close(self) -> None:
        return None


class _FakeOracleRawConnection:
    def __init__(self, cursor: _FakeOracleCursor) -> None:
        self._cursor = cursor

    def cursor(self):
        # oracledb's real AsyncConnection.cursor() is a plain method, not a
        # coroutine (confirmed against a live ADB with oracledb 4.0.2) — an
        # async fake here would mask the exact bug that shipped once.
        return self._cursor


def _oracle_game_query_tool(
    rows: list[tuple], names: list[str]
) -> tuple[GameQueryTool, _FakeOracleCursor]:
    cursor = _FakeOracleCursor(rows, names)
    raw_connection = _FakeOracleRawConnection(cursor)
    tool = GameQueryTool(OracleRagConnection(raw_connection))
    tool._mapping_load_pending = False  # skip the team-mapping DB round trip
    return tool, cursor


@pytest.mark.asyncio
async def test_game_query_tool_detects_an_oracle_connection() -> None:
    tool, _ = _oracle_game_query_tool([], [])

    assert tool._is_oracle is True


@pytest.mark.asyncio
async def test_get_games_by_date_builds_oracle_sql_and_maps_rows_to_dicts() -> None:
    columns = [
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "game_status",
        "stadium",
        "winning_team",
        "home_pitcher",
        "away_pitcher",
    ]
    row = ("G1", "2025-10-01", "LG", "HH", 5, 2, "COMPLETED", "잠실", "LG", "P1", "P2")
    tool, cursor = _oracle_game_query_tool([row], columns)

    result = await tool.get_games_by_date("2025-10-01")

    assert result["found"] is True
    assert result["games"][0]["game_id"] == "G1"
    assert result["games"][0][
        "home_team_name"
    ]  # `_format_game_response` ran on a plain dict

    sql, params = cursor.executed[0]
    assert "TO_DATE(:d1, 'YYYY-MM-DD')" in sql
    assert "= ANY(" not in sql
    assert isinstance(params, dict)
    assert params == {"d1": "2025-10-01"}


@pytest.mark.asyncio
async def test_get_team_recent_games_expands_any_into_oracle_in_list_and_binds_limit() -> (
    None
):
    tool, cursor = _oracle_game_query_tool([], [])
    tool.get_team_variants = lambda *_args, **_kwargs: ["LG", "엘지"]  # noqa: ARG005

    await tool.get_team_recent_games("LG", limit=3)

    sql, params = cursor.executed[0]
    assert "IN (:v1, :v2)" in sql
    assert "IN (:v3, :v4)" in sql
    assert "FETCH FIRST :lim" in sql
    assert 3 in params.values()
    assert isinstance(params, dict)


@pytest.mark.asyncio
async def test_get_head_to_head_keeps_case_and_where_binds_in_sql_order() -> None:
    tool, cursor = _oracle_game_query_tool([], [])
    tool.get_team_variants = lambda team, *_a, **_k: [team]  # noqa: ARG005

    await tool.get_head_to_head("LG", "HH", year=2025, limit=5)

    sql, params = cursor.executed[0]
    # CASE renders before WHERE in the SELECT, so its binds must be registered
    # first — this is the exact ordering bug class positional `%s` binding
    # would hit if the two were built out of sequence.
    case_index = sql.index("CASE")
    where_index = sql.index("WHERE")
    assert case_index < where_index
    assert isinstance(params, dict)


@pytest.mark.asyncio
async def test_get_schedule_wraps_between_dates_in_to_date() -> None:
    """Regression test: BETWEEN used a bare bind for the two date literals
    before ``bind_date`` existed, which raises ORA-01861 against a real
    Oracle DATE column — confirmed against a live ADB, not reproducible with
    a fake cursor since nothing there validates bind types."""
    tool, cursor = _oracle_game_query_tool([], [])

    await tool.get_schedule("2026-07-20", "2026-07-21")

    sql, params = cursor.executed[0]
    assert "BETWEEN TO_DATE(:d1, 'YYYY-MM-DD') AND TO_DATE(:d2, 'YYYY-MM-DD')" in sql
    assert params == {"d1": "2026-07-20", "d2": "2026-07-21"}


@pytest.mark.asyncio
async def test_get_head_to_head_wraps_as_of_game_date_in_to_date() -> None:
    tool, cursor = _oracle_game_query_tool([], [])
    tool.get_team_variants = lambda team, *_a, **_k: [team]  # noqa: ARG005

    await tool.get_head_to_head("LG", "HH", limit=5, as_of_game_date="2026-07-01")

    sql, params = cursor.executed[0]
    assert "g.game_date < TO_DATE(" in sql
    assert "2026-07-01" in params.values()
    assert "FETCH FIRST" in sql
