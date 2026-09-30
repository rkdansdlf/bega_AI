"""The Oracle retrieval path must honour the same quality contract as PostgreSQL."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.core import rag_readiness
from app.core.oracle_rag import (
    OracleRagConnection,
    _bind_filter_clauses,
    oracle_capabilities,
    oracle_exact_document_search,
    oracle_rag_readiness,
    oracle_similarity_search,
    resolve_oracle_generation,
)
from app.core.retrieval_contract import (
    REQUIRED_RESULT_KEYS,
    RetrievalContractViolation,
    UnsupportedBackendFilter,
    enforce_result_contract,
    missing_required_capabilities,
    normalize_result,
    resolve_backend_filters,
)
from app.core.retrieval_policy import InvalidRetrievalFilter
from tests.test_oracle_rag import _Connection, _Cursor

NAMES = [
    "ID",
    "TITLE",
    "CONTENT",
    "SOURCE_TABLE",
    "SOURCE_ROW_ID",
    "META",
    "CONTENT_HASH",
    "INDEX_VERSION",
    "INDEX_STATUS",
    "INDEXED_AT",
    "UPDATED_AT",
    "SEASON_YEAR",
    "TEAM_ID",
    "PLAYER_ID",
    "DISTANCE",
]


def _dense_row(*, team="LG", season=2025, player=None, version="rag-v1", rid=7):
    return (
        rid,
        "제목",
        "내용",
        "team_summary",
        f"team_id={team}|season_year={season}",
        "{}",
        "hash",
        version,
        "ACTIVE",
        None,
        None,
        season,
        team,
        player,
        0.1,
    )


def _connection(rows):
    cursor = _Cursor(rows, NAMES)
    return OracleRagConnection(_Connection(cursor)), cursor


def _run(coro):
    return asyncio.run(coro)


# --- filters: never silently ignored -----------------------------------------
def test_unknown_filter_key_is_rejected_not_ignored():
    with pytest.raises(InvalidRetrievalFilter):
        _bind_filter_clauses({"team_id": "LG", "totally_unknown": 1})


@pytest.mark.parametrize("key", ["topic_key", "source_type", "season_id"])
def test_policy_allowed_but_oracle_unenforceable_filter_fails_closed(key):
    with pytest.raises(UnsupportedBackendFilter):
        _bind_filter_clauses({key: "x"})


@pytest.mark.parametrize(
    "key",
    ["meta.a') OR ('1'='1", "meta.", "meta.x y", "meta.x;DROP", "evil.key"],
)
def test_json_path_injection_is_impossible(key):
    with pytest.raises(InvalidRetrievalFilter):
        _bind_filter_clauses({key: "v"})


def test_valid_filters_still_render_parameterised_predicates():
    clauses, params = _bind_filter_clauses(
        {"season_year": 2025, "team_id": "LG", "meta.league": "KBO", "player_id": None}
    )
    sql = " AND ".join(clauses)
    assert "season_year = :filter_season_year" in sql
    assert "team_id = :filter_team_id" in sql
    assert "JSON_VALUE(meta, '$.league') = :meta_" in sql
    assert "player_id" not in sql  # None is dropped, like PostgreSQL
    assert params["filter_team_id"] == "LG"


def test_internal_and_oracle_only_keys_are_accepted():
    resolved = resolve_backend_filters(
        {
            "_exclude_source_tables": ["x"],
            "source_table_in": ["a"],
            "game_date": "2025-05-01",
        },
        backend="oracle",
        supported_columns={"source_table"},
        extra_keys={"game_date"},
    )
    assert set(resolved) == {"_exclude_source_tables", "source_table_in", "game_date"}


# --- result shape: identical to PostgreSQL ------------------------------------
def test_oracle_rows_carry_entity_scope_and_provenance_keys():
    connection, _ = _connection([_dense_row(player="P1")])
    rows = _run(
        oracle_similarity_search(
            connection, [0.1, 0.2], limit=3, active_index_version=None
        )
    )
    row = rows[0]
    for key in REQUIRED_RESULT_KEYS:
        assert key in row, key
    assert (row["season_year"], row["team_id"], row["player_id"]) == (2025, "LG", "P1")
    assert row["retrieval_backend"] == "oracle"
    assert row["index_generation"] == "oracle:rag-v1"


def test_postgres_and_oracle_rows_normalise_to_the_same_key_set():
    pg = normalize_result(
        {
            "id": 1,
            "source_table": "t",
            "source_row_id": "r",
            "similarity": 0.5,
            "season_year": 2025,
            "team_id": "LG",
            "player_id": None,
            "meta": {},
        },
        backend="postgres",
        index_generation="g1",
    )
    ora = normalize_result(
        {"id": 2, "source_table": "t", "source_row_id": "r"},
        backend="oracle",
        index_generation="oracle:rag-v1",
    )
    assert set(REQUIRED_RESULT_KEYS) <= set(pg) and set(REQUIRED_RESULT_KEYS) <= set(
        ora
    )
    assert {k: pg[k] for k in ("retrieval_backend",)} != {
        k: ora[k] for k in ("retrieval_backend",)
    }


def test_missing_contract_keys_are_a_violation():
    with pytest.raises(RetrievalContractViolation, match="missing contract keys"):
        enforce_result_contract([{"id": 1}], backend="oracle")


# --- entity constraints are re-verified on returned rows ---------------------
def test_backend_that_returns_the_wrong_team_fails_closed():
    connection, _ = _connection([_dense_row(team="KIA")])
    with pytest.raises(RetrievalContractViolation, match="team_id"):
        _run(
            oracle_similarity_search(
                connection,
                [0.1],
                limit=3,
                filters={"team_id": "LG"},
                active_index_version=None,
            )
        )


def test_matching_entity_rows_pass():
    connection, _ = _connection([_dense_row(team="LG", season=2025)])
    rows = _run(
        oracle_similarity_search(
            connection,
            [0.1],
            limit=3,
            filters={"team_id": "lg", "season_year": 2025},
            active_index_version=None,
        )
    )
    assert len(rows) == 1


# --- generation gate ---------------------------------------------------------
def test_generation_gate_adds_index_version_predicate_to_every_query_shape():
    clauses, params = _bind_filter_clauses({}, active_index_version="rag-v2")
    assert "index_version = :active_index_version" in " AND ".join(clauses)
    assert params["active_index_version"] == "rag-v2"

    connection, cursor = _connection([_dense_row(version="rag-v2")])
    rows = _run(
        oracle_similarity_search(
            connection, [0.1], limit=3, active_index_version="rag-v2"
        )
    )
    sql, params = cursor.executed[0]
    assert "index_version = :active_index_version" in sql
    assert params["active_index_version"] == "rag-v2"
    assert rows[0]["index_generation"] == "oracle:rag-v2"


def test_exact_document_search_is_gated_and_normalised():
    cursor = _Cursor(
        [
            (
                1,
                "t",
                "c",
                "kbo_regulations",
                "r1",
                "{}",
                "rag-v2",
                2025,
                None,
                None,
                1.0,
                1.0,
            )
        ],
        [
            "ID",
            "TITLE",
            "CONTENT",
            "SOURCE_TABLE",
            "SOURCE_ROW_ID",
            "META",
            "INDEX_VERSION",
            "SEASON_YEAR",
            "TEAM_ID",
            "PLAYER_ID",
            "SIMILARITY",
            "COMBINED_SCORE",
        ],
    )
    connection = OracleRagConnection(_Connection(cursor))
    rows = _run(
        oracle_exact_document_search(
            connection,
            ["타이브레이크"],
            limit=3,
            source_tables=("kbo_regulations",),
            active_index_version="rag-v2",
        )
    )
    sql, params = cursor.executed[0]
    assert "index_version = :active_index_version" in sql
    assert rows[0]["retrieval_backend"] == "oracle"
    assert rows[0]["index_generation"] == "oracle:rag-v2"


def test_explicit_index_version_filter_cannot_widen_past_the_active_generation():
    with pytest.raises(RetrievalContractViolation, match="active generation"):
        _bind_filter_clauses({"index_version": "rag-v1"}, active_index_version="rag-v2")
    clauses, _ = _bind_filter_clauses(
        {"index_version": "rag-v2"}, active_index_version="rag-v2"
    )
    assert " AND ".join(clauses).count("index_version = ") == 1


def test_gate_enabled_without_active_version_fails_closed():
    on_missing = SimpleNamespace(rag_generation_gate_enabled=True)
    with pytest.raises(RetrievalContractViolation, match="ACTIVE_INDEX_VERSION"):
        resolve_oracle_generation(on_missing)
    assert (
        resolve_oracle_generation(SimpleNamespace(rag_generation_gate_enabled=False))
        is None
    )
    ok = SimpleNamespace(
        rag_generation_gate_enabled=True, rag_oracle_active_index_version=" rag-v2 "
    )
    assert resolve_oracle_generation(ok) == "rag-v2"


def test_default_generation_comes_from_process_settings(monkeypatch):
    from app import config

    monkeypatch.setattr(
        config,
        "get_settings",
        lambda: SimpleNamespace(
            rag_generation_gate_enabled=True, rag_oracle_active_index_version="rag-v9"
        ),
    )
    connection, cursor = _connection([_dense_row(version="rag-v9")])
    _run(oracle_similarity_search(connection, [0.1], limit=3))  # no explicit arg
    assert cursor.executed[0][1]["active_index_version"] == "rag-v9"


# --- readiness fails closed ---------------------------------------------------
def test_oracle_declares_required_capabilities():
    caps = oracle_capabilities(None)
    assert missing_required_capabilities(caps) == []
    assert caps["temporal_filters"] is False  # declared, informational only
    assert missing_required_capabilities({"entity_scope": True}) == [
        "filter_allowlist",
        "generation_gate",
    ]


def _readiness_cursor(retrievable, in_generation):
    return _Cursor(
        [(10, 10, 0, 10, 1, retrievable, in_generation)],
        [
            "TOTAL_ROWS",
            "VECTOR_ROWS",
            "MISSING_ROWS",
            "MATCHING_DIM_ROWS",
            "VALID_INDEX_ROWS",
            "RETRIEVABLE_ROWS",
            "ACTIVE_GENERATION_ROWS",
        ],
    )


def test_readiness_requires_full_active_generation_coverage():
    ok = _run(
        oracle_rag_readiness(
            OracleRagConnection(_Connection(_readiness_cursor(10, 10))),
            active_index_version="rag-v2",
        )
    )
    assert ok["ready"] is True and ok["generation_ok"] is True

    mixed = _run(
        oracle_rag_readiness(
            OracleRagConnection(_Connection(_readiness_cursor(10, 7))),
            active_index_version="rag-v2",
        )
    )
    assert mixed["ready"] is False and mixed["generation_ok"] is False
    assert mixed["active_generation_rows"] == 7


def test_probe_is_down_when_gate_enabled_without_active_version():
    settings = SimpleNamespace(rag_generation_gate_enabled=True, embed_dim=1536)
    out = _run(rag_readiness._probe_oracle(object(), settings))
    assert out["rag_schema"]["status"] == "DOWN"
    assert out["rag_schema"]["code"] == "RETRIEVAL_CONTRACT_NOT_MET"


def test_probe_is_down_when_generation_coverage_is_partial(monkeypatch):
    class _Pool:
        def connection(self, timeout=None):
            class _Ctx:
                async def __aenter__(_s):
                    return OracleRagConnection(_Connection(_Cursor([(1,)], ["X"])))

                async def __aexit__(_s, *a):
                    return False

            return _Ctx()

    async def fake_readiness(conn, **kw):
        return {
            "vector_rows": 10,
            "matching_dim_rows": 10,
            "missing_rows": 0,
            "index_valid": True,
            "generation_ok": False,
            "contract": oracle_capabilities(kw.get("active_index_version")),
        }

    monkeypatch.setattr(rag_readiness, "oracle_rag_readiness", fake_readiness)
    settings = SimpleNamespace(
        rag_generation_gate_enabled=True,
        rag_oracle_active_index_version="rag-v2",
        embed_dim=1536,
    )
    out = _run(rag_readiness._probe_oracle(_Pool(), settings))
    assert out["rag_schema"]["status"] == "DOWN"
    assert out["rag_schema"]["generation_ok"] is False


def test_probe_is_up_when_contract_and_generation_hold(monkeypatch):
    class _Pool:
        def connection(self, timeout=None):
            class _Ctx:
                async def __aenter__(_s):
                    return OracleRagConnection(_Connection(_Cursor([(1,)], ["X"])))

                async def __aexit__(_s, *a):
                    return False

            return _Ctx()

    async def fake_readiness(conn, **kw):
        return {
            "vector_rows": 10,
            "matching_dim_rows": 10,
            "missing_rows": 0,
            "index_valid": True,
            "generation_ok": True,
            "contract": oracle_capabilities(kw.get("active_index_version")),
        }

    monkeypatch.setattr(rag_readiness, "oracle_rag_readiness", fake_readiness)
    settings = SimpleNamespace(rag_generation_gate_enabled=False, embed_dim=1536)
    out = _run(rag_readiness._probe_oracle(_Pool(), settings))
    assert all(c["status"] == "UP" for c in out.values())
