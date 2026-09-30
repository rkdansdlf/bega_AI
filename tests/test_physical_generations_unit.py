"""DB-free checks for the physical embedding store (the real-DB flow is covered
by test_physical_generations_pg.py when TEST_PGVECTOR_URL is set)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.core import embedding_generations as eg
from app.core import rag_readiness, retrieval
from app.core.retrieval_contract import RetrievalContractViolation
from scripts import manage_embedding_generation as cli


def _run(coro):
    return asyncio.run(coro)


def _gen(**over):
    base = dict(
        generation_id="g1",
        embedding_model="m",
        embedding_dim=256,
        embedding_version=1,
        status="ACTIVE",
        mirror_inline=True,
        index_ready=True,
    )
    base.update(over)
    return eg.Generation(**base)


# --- identifiers are the only thing interpolated into SQL --------------------
@pytest.mark.parametrize("ok", ["g1", "gen-2026-10", "a", "x_y-z", "0abc", "a" * 40])
def test_valid_generation_ids(ok):
    assert eg.validate_generation_id(ok) == ok
    assert eg.index_name_for(ok).startswith("idx_rag_emb_")
    assert "-" not in eg.index_name_for(ok)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "G1",
        "g 1",
        "g;drop table x",
        "-x",
        "_x",
        "a" * 41,
        "g'1",
        'g"1',
        "g.1",
        None,
    ],
)
def test_invalid_generation_ids_are_rejected(bad):
    with pytest.raises(eg.GenerationError):
        eg.validate_generation_id(bad)
    with pytest.raises(eg.GenerationError):
        eg.index_name_for(bad)


def test_index_name_fits_postgres_identifier_limit():
    assert len(eg.index_name_for("a" * 40)) <= 63


# --- retrieval SQL fragments -------------------------------------------------
def test_physical_sql_inlines_a_literal_generation_and_matching_expression():
    parts = retrieval._physical_sql_parts(_gen(generation_id="g-2", embedding_dim=256))
    assert "e.generation_id = 'g-2'" in parts["from"]  # literal => partial index usable
    assert "e.content_hash IS NOT DISTINCT FROM r.content_hash" in parts["from"]
    # Exactly the indexed expression: embedding::halfvec(dim) with cosine (<=>).
    assert parts["distance"] == "e.embedding::halfvec(256) <=> %s::halfvec(256)"
    assert "e2.generation_id = 'g-2'" in parts["keyword_guard"]
    assert "%s" not in parts["from"]  # no bind params in the join


# --- fail closed --------------------------------------------------------------
class _Cur:
    def __init__(self, row):
        self._row = row

    async def fetchone(self):
        return self._row


class _Conn:
    def __init__(self, row=None, error=None):
        self.row, self.error = row, error

    async def execute(self, sql, params=None):
        if self.error:
            raise self.error
        return _Cur(self.row)


@pytest.fixture(autouse=True)
def _fresh_cache():
    eg.reset_active_cache()
    yield
    eg.reset_active_cache()


def _settings(store):
    return SimpleNamespace(rag_embedding_store=store)


def test_inline_store_needs_no_registry():
    assert (
        _run(
            retrieval._resolve_physical_generation(
                _Conn(error=RuntimeError("x")), _settings("inline")
            )
        )
        is None
    )


def test_generations_store_without_active_generation_fails_closed():
    with pytest.raises(RetrievalContractViolation, match="no generation is ACTIVE"):
        _run(
            retrieval._resolve_physical_generation(
                _Conn(row=None), _settings("generations")
            )
        )


def test_generations_store_without_migrations_fails_closed():
    with pytest.raises(RetrievalContractViolation, match="migrations 007\\+009"):
        _run(
            retrieval._resolve_physical_generation(
                _Conn(error=RuntimeError("relation does not exist")),
                _settings("generations"),
            )
        )


def test_active_generation_is_resolved_with_index_state():
    row = ("g1", "m", 256, 1, "ACTIVE", True, True)
    out = _run(
        retrieval._resolve_physical_generation(_Conn(row=row), _settings("generations"))
    )
    assert (out.generation_id, out.embedding_dim, out.index_ready) == ("g1", 256, True)


def test_a_tampered_registry_id_never_reaches_sql():
    row = ("g1'; DROP TABLE rag_chunks; --", "m", 256, 1, "ACTIVE", True, True)
    with pytest.raises(RetrievalContractViolation, match="valid token"):
        _run(
            retrieval._resolve_physical_generation(
                _Conn(row=row), _settings("generations")
            )
        )


# --- settings -----------------------------------------------------------------
def test_store_setting_defaults_to_inline_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv("RAG_EMBEDDING_STORE", raising=False)
    assert Settings().rag_embedding_store == "inline"
    monkeypatch.setenv("RAG_EMBEDDING_STORE", "generations")
    assert Settings().rag_embedding_store == "generations"
    monkeypatch.setenv("RAG_EMBEDDING_STORE", "both")
    with pytest.raises(Exception):
        Settings()


# --- readiness ----------------------------------------------------------------
def test_physical_readiness_codes():
    comp = rag_readiness._physical_components
    ok = comp(
        True,
        True,
        {"generation_id": "g1", "dimension": 256, "index_live": True, "has_rows": True},
    )
    assert ok["rag_index"]["status"] == "UP" and ok["rag_vector"]["generation"] == "g1"
    none = comp(True, True, {"error": "no_active_generation"})
    assert none["rag_index"]["code"] == "RAG_GENERATION_NOT_ACTIVE"
    reg = comp(True, True, {"error": "registry_unavailable"})
    assert reg["rag_vector"]["code"] == "RAG_GENERATION_REGISTRY_UNAVAILABLE"
    no_index = comp(
        True,
        True,
        {
            "generation_id": "g1",
            "dimension": 256,
            "index_live": False,
            "has_rows": True,
        },
    )
    assert no_index["rag_index"]["code"] == "RAG_VECTOR_INDEX_NOT_READY"
    empty = comp(
        True,
        True,
        {
            "generation_id": "g1",
            "dimension": 256,
            "index_live": True,
            "has_rows": False,
        },
    )
    assert empty["rag_vector"]["status"] == "DOWN"


# --- registration guards -------------------------------------------------------
def test_register_rejects_oversized_dimension_and_bad_ids():
    class Boom:
        async def execute(self, *a, **k):  # pragma: no cover
            raise AssertionError("must not reach the database")

    for kwargs in (
        dict(generation_id="ok", embedding_dim=4001),
        dict(generation_id="ok", embedding_dim=0),
        dict(generation_id="Bad Id", embedding_dim=8),
    ):
        with pytest.raises(eg.GenerationError):
            _run(
                eg.register_generation(
                    Boom(), embedding_model="m", embedding_version=1, **kwargs
                )
            )


def test_write_embeddings_rejects_wrong_dimension_before_writing():
    class Conn:
        def __init__(self):
            self.writes = 0

        async def execute(self, sql, params=None):
            if "INSERT INTO rag_chunk_embeddings" in sql:
                self.writes += 1
            return _Cur(("g1", "m", 8, 1, "BUILDING", False, False))

    conn = Conn()
    with pytest.raises(eg.GenerationError, match="dims"):
        _run(eg.write_embeddings(conn, "g1", [(1, [0.0] * 4, "h")]))
    assert conn.writes == 0


# --- CLI ------------------------------------------------------------------------
def test_cli_parses_the_adoption_flow():
    p = cli.build_parser()
    a = p.parse_args(
        [
            "register",
            "g1",
            "--model",
            "m",
            "--dim",
            "1536",
            "--version",
            "2",
            "--mirror-inline",
        ]
    )
    assert a.cmd == "register" and a.mirror_inline is True
    assert p.parse_args(["backfill", "g1", "--batch-size", "100"]).batch_size == 100
    for cmd in ("build-index", "audit", "drop"):
        assert p.parse_args([cmd, "g1"]).generation_id == "g1"
    act = p.parse_args(["--store", "generations", "activate", "g1", "--force"])
    assert act.store == "generations" and act.force is True
    assert p.parse_args(["rollback"]).cmd == "rollback"
    with pytest.raises(SystemExit):
        p.parse_args(["--store", "both", "list"])
