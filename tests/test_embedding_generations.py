import asyncio
from contextlib import asynccontextmanager

import pytest

from app.core import embedding_generations as eg


class FakeDB:
    """In-memory stand-in for the registry + rag_chunks coverage query."""

    def __init__(self):
        self.gens = {}  # id -> dict
        self.events = []
        self.chunks = []  # (model, dim, version, has_embedding)

    # --- connection API -------------------------------------------------
    @asynccontextmanager
    async def transaction(self):
        snapshot = ({k: dict(v) for k, v in self.gens.items()}, list(self.events))
        try:
            yield
        except BaseException:
            self.gens, self.events = snapshot
            raise

    async def execute(self, sql, params=()):
        sql = " ".join(sql.split())
        cur = _Cur()
        if sql.startswith("INSERT INTO rag_embedding_generations"):
            gid, model, dim, ver, _note = params
            self.gens[gid] = dict(
                id=gid, model=model, dim=dim, ver=ver, status="BUILDING"
            )
        elif sql.startswith("SELECT generation_id") and "status = 'ACTIVE'" in sql:
            cur.rows = [
                self._row(g) for g in self.gens.values() if g["status"] == "ACTIVE"
            ]
        elif sql.startswith("SELECT generation_id") and "generation_id = %s" in sql:
            g = self.gens.get(params[0])
            cur.rows = [self._row(g)] if g else []
        elif sql.startswith("SELECT count(*)"):
            model, dim, ver = params
            total = len(self.chunks)
            matching = sum(
                1
                for c in self.chunks
                if c[3] and (c[0], c[1], c[2]) == (model, dim, ver)
            )
            missing = sum(1 for c in self.chunks if not c[3])
            cur.rows = [(total, matching, missing)]
        elif "SET status = 'READY'" in sql:
            self.gens[params[0]]["status"] = "READY"
        elif "SET status = 'RETIRED'" in sql:
            self.gens[params[0]]["status"] = "RETIRED"
        elif "SET status = 'ACTIVE'" in sql:
            actives = [g for g in self.gens.values() if g["status"] == "ACTIVE"]
            if actives:
                raise RuntimeError("uq_single_active violated")
            self.gens[params[0]]["status"] = "ACTIVE"
        elif sql.startswith("INSERT INTO rag_embedding_generation_events"):
            gid, action, prev, _detail = params
            self.events.append((gid, action, prev))
        elif sql.startswith("SELECT previous_id"):
            acts = [e for e in self.events if e[1] == "ACTIVATE" and e[2]]
            cur.rows = [(acts[-1][2],)] if acts else []
        else:  # pragma: no cover
            raise AssertionError(sql)
        return cur

    @staticmethod
    def _row(g):
        return (g["id"], g["model"], g["dim"], g["ver"], g["status"])


class _Cur:
    rows = []

    async def fetchone(self):
        return self.rows[0] if self.rows else None


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_cache():
    eg.reset_active_cache()
    yield
    eg.reset_active_cache()


def _setup():
    db = FakeDB()
    _run(
        eg.register_generation(
            db,
            generation_id="g1",
            embedding_model="old",
            embedding_dim=1536,
            embedding_version=1,
        )
    )
    _run(
        eg.register_generation(
            db,
            generation_id="g2",
            embedding_model="new",
            embedding_dim=1536,
            embedding_version=2,
        )
    )
    return db


def test_activation_blocked_until_coverage_met():
    db = _setup()
    db.chunks = [("old", 1536, 1, True)] * 10
    with pytest.raises(eg.GenerationError, match="coverage"):
        _run(eg.activate_generation(db, "g2"))
    assert db.gens["g2"]["status"] == "BUILDING"


def test_activate_switches_pointer_and_keeps_previous_ready():
    db = _setup()
    db.chunks = [("old", 1536, 1, True)] * 10
    _run(eg.activate_generation(db, "g1"))
    db.chunks = [("new", 1536, 2, True)] * 10
    result = _run(eg.activate_generation(db, "g2"))
    assert result["previous_id"] == "g1"
    assert db.gens["g1"]["status"] == "READY"
    assert db.gens["g2"]["status"] == "ACTIVE"
    assert _run(eg.get_active_generation(db, use_cache=False)).generation_id == "g2"


def test_rollback_restores_previous_when_its_rows_exist():
    db = _setup()
    db.chunks = [("old", 1536, 1, True)] * 5
    _run(eg.activate_generation(db, "g1"))
    db.chunks = [("new", 1536, 2, True)] * 5 + [("old", 1536, 1, True)] * 5
    _run(eg.activate_generation(db, "g2", min_coverage=0.5))
    out = _run(eg.rollback_generation(db))
    assert out["generation_id"] == "g1"
    assert db.gens["g1"]["status"] == "ACTIVE"
    assert db.gens["g2"]["status"] == "RETIRED"


def test_rollback_refused_when_old_rows_were_overwritten():
    db = _setup()
    db.chunks = [("old", 1536, 1, True)] * 5
    _run(eg.activate_generation(db, "g1"))
    db.chunks = [("new", 1536, 2, True)] * 5  # overwritten in place
    _run(eg.activate_generation(db, "g2"))
    with pytest.raises(eg.GenerationError, match="overwritten"):
        _run(eg.rollback_generation(db))
    assert db.gens["g2"]["status"] == "ACTIVE"


def test_failed_activation_is_atomic():
    db = _setup()
    db.chunks = [("old", 1536, 1, True)] * 5
    _run(eg.activate_generation(db, "g1"))
    db.chunks = [("new", 1536, 2, True)] * 5
    real_execute = db.execute

    async def boom(sql, params=()):
        if "SET status = 'ACTIVE'" in " ".join(sql.split()) and params == ("g2",):
            raise RuntimeError("crash mid-switch")
        return await real_execute(sql, params)

    db.execute = boom
    with pytest.raises(RuntimeError):
        _run(eg.activate_generation(db, "g2"))
    assert db.gens["g1"]["status"] == "ACTIVE"  # pointer restored
    assert db.gens["g2"]["status"] == "BUILDING"


def test_filter_sql_only_with_active_generation():
    assert eg.generation_filter_sql(None) == ("", [])
    sql, params = eg.generation_filter_sql(eg.Generation("g", "m", 1536, 2, "ACTIVE"))
    assert "embedding_model = %s" in sql and params == ["m", 1536, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_retrieval_generation_gate_only_when_enabled(monkeypatch, enabled):
    from app.core import retrieval
    from app.config import get_settings
    from tests.test_retrieval import _DummyConnection

    async def _active(conn, **kw):
        return eg.Generation("g2", "new-model", 1536, 2, "ACTIVE")

    monkeypatch.setattr(retrieval, "get_active_generation", _active)
    settings = get_settings().model_copy(
        update={"rag_generation_gate_enabled": enabled}
    )
    conn = _DummyConnection([])
    await retrieval.similarity_search(
        conn, [0.1, 0.2, 0.3], limit=3, filters=None, settings=settings
    )
    sql, params = conn.last_cursor.executed[1]
    if enabled:
        assert "embedding_model = %s AND embedding_dim = %s" in sql
        assert ["new-model", 1536, 2] == params[1:4] or "new-model" in params
    else:
        assert "embedding_model = %s" not in sql
        assert "new-model" not in params
