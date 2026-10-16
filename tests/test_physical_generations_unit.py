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




# --- CLI ------------------------------------------------------------------------


# --- backfill CLI outcome -------------------------------------------------------


# ─── cache TTL, request pinning, propagation ──────────────────────────────────
from app.core import request_scope  # noqa: E402


class _CountingConn:
    def __init__(self, row):
        self.row, self.queries = row, 0

    async def execute(self, sql, params=None):
        self.queries += 1
        return _Cur(self.row)


G1_ROW = ("g1", "m", 256, 1, "ACTIVE")


def test_cache_ttl_is_honoured_and_zero_disables_caching(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(eg.time, "monotonic", lambda: clock["t"])

    conn = _CountingConn(G1_ROW)
    for _ in range(3):
        _run(eg.get_active_generation(conn, ttl=5))
    assert conn.queries == 1  # cached
    clock["t"] += 4.9
    _run(eg.get_active_generation(conn, ttl=5))
    assert conn.queries == 1
    clock["t"] += 0.2  # 5.1s > ttl
    _run(eg.get_active_generation(conn, ttl=5))
    assert conn.queries == 2

    eg.reset_active_cache()
    zero = _CountingConn(G1_ROW)
    for _ in range(3):
        _run(eg.get_active_generation(zero, ttl=0))
    assert zero.queries == 3  # ttl=0: look up every time


def test_activation_in_this_process_invalidates_the_cache_immediately():
    conn = _CountingConn(G1_ROW)
    _run(eg.get_active_generation(conn, ttl=300))
    eg.reset_active_cache()  # what activate/rollback/build-index/drop call
    _run(eg.get_active_generation(conn, ttl=300))
    assert conn.queries == 2


def test_propagation_window_is_reported_to_operators():
    assert eg._propagation(2.5) == {"max_propagation_seconds": 2.5}
    assert eg._propagation(None)["max_propagation_seconds"] == eg._ACTIVE_CACHE_TTL_S
    assert eg._propagation(-3) == {"max_propagation_seconds": 0.0}


def test_generation_cache_ttl_setting(monkeypatch):
    monkeypatch.delenv("RAG_GENERATION_CACHE_TTL_S", raising=False)
    assert Settings().rag_generation_cache_ttl_s == 5.0
    monkeypatch.setenv("RAG_GENERATION_CACHE_TTL_S", "0")
    assert Settings().rag_generation_cache_ttl_s == 0.0
    for bad in ("-1", "301"):
        monkeypatch.setenv("RAG_GENERATION_CACHE_TTL_S", bad)
        with pytest.raises(Exception):
            Settings()


def test_request_scope_pins_one_generation_for_every_task_of_a_request():
    calls = {"n": 0}

    async def loader():
        calls["n"] += 1
        await asyncio.sleep(0.02)  # the window in which an activation could land
        return f"gen-{calls['n']}"

    async def scenario():
        token = request_scope.begin_request_scope()
        try:
            # Parallel children (multi-query / HyDE) race to resolve first.
            got = await asyncio.gather(
                *(
                    retrieval._pinned_for_request("physical_generation", loader)
                    for _ in range(6)
                )
            )
            later = await retrieval._pinned_for_request("physical_generation", loader)
        finally:
            request_scope.end_request_scope(token)
        return got, later

    got, later = _run(scenario())
    assert set(got) == {"gen-1"} and later == "gen-1"
    assert calls["n"] == 1  # resolved once for the whole request


def test_without_a_request_scope_nothing_is_pinned():
    calls = {"n": 0}

    async def loader():
        calls["n"] += 1
        return calls["n"]

    async def scenario():
        return [
            await retrieval._pinned_for_request("physical_generation", loader)
            for _ in range(3)
        ]

    assert _run(scenario()) == [1, 2, 3]  # scripts/tests keep normal behaviour


def test_each_http_request_gets_its_own_scope():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.observability.tracing import RequestIdMiddleware

    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    seen = []

    @app.get("/x")
    async def x():
        scope = request_scope.current_scope()
        assert scope is not None and scope == {}
        scope["physical_generation"] = "pinned"
        seen.append(id(scope))
        return {"ok": True}

    client = TestClient(app)
    client.get("/x")
    client.get("/x")
    assert len(set(seen)) == 2  # no leakage between requests
    assert request_scope.current_scope() is None


def test_activate_cli_explains_how_long_other_instances_take(capsys):
    cli._note_propagation({"changed": True, "max_propagation_seconds": 5.0})
    err = capsys.readouterr().err
    assert "within 5s" in err and "/ready" in err
    cli._note_propagation({"changed": False, "max_propagation_seconds": 5.0})
    assert capsys.readouterr().err == ""  # nothing switched, nothing to say


@pytest.mark.parametrize("uptime_s", [0.5, 3.0, 100.0])
def test_a_never_populated_or_reset_cache_always_queries_even_on_a_freshly_booted_host(
    monkeypatch, uptime_s
):
    """time.monotonic() counts from boot. "Never looked up" must not be encoded as
    time zero, or on a host up for less than the TTL a reset/empty cache looks fresh
    and serves ``None`` ("no active generation") without touching the database."""
    monkeypatch.setattr(eg.time, "monotonic", lambda: uptime_s)
    eg.reset_active_cache()

    conn = _CountingConn(G1_ROW)
    first = _run(eg.get_active_generation(conn, ttl=300))
    assert conn.queries == 1 and first is not None and first.generation_id == "g1"

    eg.reset_active_cache()
    again = _run(eg.get_active_generation(conn, ttl=300))
    assert conn.queries == 2 and again is not None

    ext = _CountingConn(G1_ROW)
    _run(eg.get_active_generation(ext, ttl=300, extended=True))
    assert ext.queries == 1
