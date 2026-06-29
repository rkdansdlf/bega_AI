import asyncio

from app.routers import coach as coach_router


class _Result:
    def __init__(self, *, fetchone_value=None, rowcount=0):
        self._fetchone_value = fetchone_value
        self.rowcount = rowcount

    async def fetchone(self):
        return self._fetchone_value


class _TransactionCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ConnectionCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ScriptedConn:
    def __init__(self, steps):
        self._steps = list(steps)
        self.commits = 0

    async def execute(self, sql, params):
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

    async def commit(self):
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
        asyncio.run(coach_router._claim_cache_generation(
            pool=_Pool(conn),
            cache_key="cache",
            team_id="LG",
            year=2025,
            prompt_version="v11",
            model_name="openrouter/free",
            lease_owner="lease-owner",
            completed_ttl_seconds=None,
        ))
    )

    assert gate == "ROW_RECREATED"
    assert cached is None
    assert error_message is None
    assert error_code is None
    assert attempt_count == 1


def test_read_completed_cache_if_fresh_returns_manual_cache_without_claiming():
    cached_payload = {
        "response": {
            "headline": "캐시된 AI 코치 분석",
            "detailed_markdown": "## 경기 복기\n- 기존 분석을 조회합니다.",
            "coach_note": "완료된 캐시를 재사용합니다.",
            "analysis": {
                "summary": "기존 분석입니다.",
                "verdict": "캐시 조회 결과입니다.",
                "strengths": [],
                "weaknesses": [],
                "risks": [],
                "why_it_matters": [],
                "swing_factors": [],
                "watch_points": [],
                "uncertainty": [],
            },
            "key_metrics": [],
        },
        "_meta": {
            "data_quality": "insufficient",
            "game_status_bucket": "COMPLETED",
            "used_evidence": ["game", "kbo_seasons"],
            "grounding_reasons": ["missing_summary"],
        },
    }
    conn = _ScriptedConn(
        [
            {
                "contains": "SELECT status, response_json",
                "fetchone": (
                    "COMPLETED",
                    cached_payload,
                    None,
                    None,
                    3,
                    None,
                    None,
                    None,
                ),
            },
        ]
    )

    gate, cached, error_message, error_code, attempt_count = (
        asyncio.run(coach_router._read_completed_cache_if_fresh(
            pool=_Pool(conn),
            cache_key="cache",
            completed_ttl_seconds=None,
            request_mode=coach_router.COACH_REQUEST_MODE_MANUAL,
            expected_data_quality="insufficient",
            expected_used_evidence=["game", "kbo_seasons"],
            expected_game_status_bucket="COMPLETED",
            current_root_causes=["missing_summary"],
        ))
    )

    assert gate == "HIT"
    assert cached == cached_payload
    assert error_message is None
    assert error_code is None
    assert attempt_count == 3


def test_store_completed_cache_reinserts_missing_row():
    conn = _ScriptedConn(
        [
            {"contains": "UPDATE coach_analysis_cache", "rowcount": 0},
            {"contains": "SELECT status, response_json, lease_owner", "fetchone": None},
            {"contains": "INSERT INTO coach_analysis_cache", "fetchone": ("cache",)},
        ]
    )

    result = asyncio.run(coach_router._store_completed_cache(
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
    ))

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

    result = asyncio.run(coach_router._store_failed_cache(
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
    ))

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

    result = asyncio.run(coach_router._store_completed_cache(
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
    ))

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
