import asyncio

from app.routers import coach as coach_router


class _Result:
    def __init__(self, *, fetchone_value=None, rowcount=0):
        self._fetchone_value = fetchone_value
        self.rowcount = rowcount

    def fetchone(self):
        return self._fetchone_value


class _TransactionCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ConnectionCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        return False


class _ScriptedConn:
    def __init__(self, steps):
        self._steps = list(steps)
        self.commits = 0

    def execute(self, sql, params):
        if not self._steps:
            raise AssertionError(f"Unexpected SQL: {sql}")
        step = self._steps.pop(0)
        expected = step.get("contains")
        if expected and expected not in sql:
            raise AssertionError(f"Expected SQL containing {expected!r}, got {sql!r}")
        return _Result(
            fetchone_value=step.get("fetchone"),
            rowcount=step.get("rowcount", 0),
        )

    def commit(self):
        self.commits += 1

    def transaction(self):
        return _TransactionCtx()


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def connection(self):
        return _ConnectionCtx(self._conn)


def test_claim_cache_generation_recreates_missing_row():
    conn = _ScriptedConn(
        [
            {"contains": "INSERT INTO coach_analysis_cache", "fetchone": None},
            {"contains": "SELECT status, response_json", "fetchone": None},
            {"contains": "INSERT INTO coach_analysis_cache", "fetchone": ("cache",)},
        ]
    )

    gate, cached, error_message, error_code, attempt_count = (
        coach_router._claim_cache_generation(
            pool=_Pool(conn),
            cache_key="cache",
            team_id="LG",
            year=2025,
            prompt_version="v11",
            model_name="openrouter/free",
            lease_owner="lease-owner",
            completed_ttl_seconds=None,
        )
    )

    assert gate == "ROW_RECREATED"
    assert cached is None
    assert error_message is None
    assert error_code is None
    assert attempt_count == 1


def test_store_completed_cache_reinserts_missing_row():
    conn = _ScriptedConn(
        [
            {"contains": "UPDATE coach_analysis_cache", "rowcount": 0},
            {"contains": "SELECT status, response_json, lease_owner", "fetchone": None},
            {"contains": "INSERT INTO coach_analysis_cache", "fetchone": ("cache",)},
        ]
    )

    result = coach_router._store_completed_cache(
        pool=_Pool(conn),
        cache_key="cache",
        lease_owner="lease-owner",
        team_id="LG",
        year=2025,
        prompt_version="v11",
        model_name="openrouter/free",
        response_payload={"headline": "x"},
        meta_defaults=coach_router._build_meta_payload_defaults(
            generation_mode="evidence_fallback",
            data_quality="grounded",
            used_evidence=["team_summary"],
            attempt_count=1,
        ),
    )

    assert result["outcome"] == "inserted_missing_row"
    assert conn.commits == 1


def test_store_failed_cache_reinserts_missing_row():
    conn = _ScriptedConn(
        [
            {"contains": "UPDATE coach_analysis_cache", "rowcount": 0},
            {"contains": "SELECT status, response_json, lease_owner", "fetchone": None},
            {"contains": "INSERT INTO coach_analysis_cache", "fetchone": ("cache",)},
        ]
    )

    result = coach_router._store_failed_cache(
        pool=_Pool(conn),
        cache_key="cache",
        lease_owner="lease-owner",
        team_id="LG",
        year=2025,
        prompt_version="v11",
        model_name="openrouter/free",
        attempt_count=2,
        error_code="empty_response",
        error_message="empty_response",
    )

    assert result["outcome"] == "inserted_missing_row"
    assert conn.commits == 1


def test_store_completed_cache_reports_finalize_conflict():
    conn = _ScriptedConn(
        [
            {"contains": "UPDATE coach_analysis_cache", "rowcount": 0},
            {
                "contains": "SELECT status, response_json, lease_owner",
                "fetchone": ("PENDING", None, "other-owner"),
            },
        ]
    )

    result = coach_router._store_completed_cache(
        pool=_Pool(conn),
        cache_key="cache",
        lease_owner="lease-owner",
        team_id="LG",
        year=2025,
        prompt_version="v11",
        model_name="openrouter/free",
        response_payload={"headline": "x"},
        meta_defaults=coach_router._build_meta_payload_defaults(
            generation_mode="evidence_fallback",
            data_quality="grounded",
            used_evidence=["team_summary"],
            attempt_count=1,
        ),
    )

    assert result["outcome"] == "finalize_conflict"
    assert result["lease_owner"] == "other-owner"


def test_wait_for_cache_terminal_state_returns_missing_row():
    conn = _ScriptedConn(
        [
            {"contains": "SELECT status, response_json", "fetchone": None},
        ]
    )

    result = asyncio.run(
        coach_router._wait_for_cache_terminal_state(
            _Pool(conn),
            "cache",
            timeout_seconds=0.01,
            poll_ms=1,
        )
    )

    assert result == {"status": "MISSING_ROW"}


def test_heartbeat_cache_lease_sets_event_when_row_missing(monkeypatch):
    conn = _ScriptedConn(
        [
            {"contains": "UPDATE coach_analysis_cache", "rowcount": 0},
        ]
    )
    lease_lost_event = asyncio.Event()
    monkeypatch.setattr(
        coach_router,
        "COACH_CACHE_HEARTBEAT_INTERVAL_SECONDS",
        0,
    )

    async def _run():
        await coach_router._heartbeat_cache_lease(
            pool=_Pool(conn),
            cache_key="cache",
            lease_owner="lease-owner",
            lease_lost_event=lease_lost_event,
        )

    asyncio.run(_run())

    assert lease_lost_event.is_set() is True
