from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Any
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.deps import get_agent
from app.main import app
from app.core.coach_cache_key import build_coach_cache_key


class _FakeAgent:
    async def process_query(
        self, question: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "answer": f"mock-answer:{question}",
            "tool_calls": [],
            "tool_results": [],
            "data_sources": ["mock"],
            "verified": True,
            "visualizations": [],
            "error": None,
        }

    def _convert_team_id_to_name(self, team_id: str) -> str:
        return "LG 트윈스" if team_id == "LG" else str(team_id)


class _DummyExecuteResult:
    def __init__(self, row: Any = None, rowcount: int = 1):
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        return self._row


class _DummyConnection:
    def __init__(
        self,
        cache_rows: dict[str, dict[str, Any]],
        counters: dict[str, int],
    ):
        self._cache_rows = cache_rows
        self._counters = counters

    def transaction(self):
        return _DummyConnectionContext(self)

    def execute(self, query: str, params: tuple[Any, ...] | None = None):
        normalized = " ".join(str(query).split()).lower()
        now_ref = datetime.now(timezone.utc)

        if "insert into coach_analysis_cache" in normalized:
            cache_key = str((params or ("",))[0])
            if cache_key in self._cache_rows:
                return _DummyExecuteResult(None)
            self._cache_rows[cache_key] = {
                "status": "PENDING",
                "response_json": None,
                "error_message": None,
                "updated_at": now_ref,
            }
            self._counters["insert"] += 1
            return _DummyExecuteResult((cache_key,))

        if (
            "select status, response_json, error_message, updated_at" in normalized
            and "from coach_analysis_cache" in normalized
        ):
            cache_key = str((params or ("",))[0])
            row = self._cache_rows.get(cache_key)
            if not row:
                return _DummyExecuteResult(None)
            return _DummyExecuteResult(
                (
                    row.get("status"),
                    row.get("response_json"),
                    row.get("error_message"),
                    row.get("updated_at"),
                )
            )

        if (
            "select status, response_json, error_message" in normalized
            and "from coach_analysis_cache" in normalized
        ):
            cache_key = str((params or ("",))[0])
            row = self._cache_rows.get(cache_key)
            if not row:
                return _DummyExecuteResult(None)
            return _DummyExecuteResult(
                (
                    row.get("status"),
                    row.get("response_json"),
                    row.get("error_message"),
                )
            )

        if "update coach_analysis_cache set status = 'completed'" in normalized:
            response_json, cache_key = params or (None, "")
            row = self._cache_rows.setdefault(str(cache_key), {})
            parsed_json = response_json
            if isinstance(response_json, str):
                try:
                    parsed_json = json.loads(response_json)
                except json.JSONDecodeError:
                    parsed_json = response_json
            row.update(
                {
                    "status": "COMPLETED",
                    "response_json": parsed_json,
                    "error_message": None,
                    "updated_at": now_ref,
                }
            )
            self._counters["completed_update"] += 1
            return _DummyExecuteResult(None, rowcount=1)

        if "update coach_analysis_cache set status = 'failed'" in normalized:
            error_message, cache_key = params or ("", "")
            row = self._cache_rows.setdefault(str(cache_key), {})
            row.update(
                {
                    "status": "FAILED",
                    "response_json": None,
                    "error_message": str(error_message),
                    "updated_at": now_ref,
                }
            )
            self._counters["failed_update"] += 1
            return _DummyExecuteResult(None, rowcount=1)

        if (
            "update coach_analysis_cache" in normalized
            and "set status = 'pending'" in normalized
        ):
            _, _, _, _, cache_key = params or ("", "", "", "", "")
            row = self._cache_rows.setdefault(str(cache_key), {})
            row.update(
                {
                    "status": "PENDING",
                    "error_message": None,
                    "updated_at": now_ref,
                }
            )
            self._counters["pending_update"] += 1
            return _DummyExecuteResult(None, rowcount=1)

        return _DummyExecuteResult()

    def commit(self) -> None:
        return None


class _DummyConnectionContext:
    def __init__(self, conn: _DummyConnection):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyPool:
    def __init__(self):
        self.cache_rows: dict[str, dict[str, Any]] = {}
        self.counters: dict[str, int] = {
            "insert": 0,
            "completed_update": 0,
            "failed_update": 0,
            "pending_update": 0,
        }
        self._conn = _DummyConnection(self.cache_rows, self.counters)

    def connection(self):
        return _DummyConnectionContext(self._conn)


class _ParsedCoachResponse:
    def model_dump(self) -> dict[str, Any]:
        return {
            "headline": "LG mock headline",
            "sentiment": "neutral",
            "key_metrics": [],
            "analysis": {"strengths": [], "weaknesses": [], "risks": []},
            "detailed_markdown": "## mock",
            "coach_note": "mock coach note for api smoke",
        }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    class _NeverExitEvent:
        async def wait(self):
            await asyncio.sleep(3600)

    dummy_pool = _DummyPool()
    llm_calls = {"count": 0}
    app.dependency_overrides[get_agent] = lambda: _FakeAgent()

    async def _fake_execute_coach_tools_parallel(*args, **kwargs):
        return {
            "home": {
                "summary": {"team_name": "LG 트윈스"},
                "advanced": {"metrics": {"batting": {"ops": 0.8}}},
            },
            "away": {},
        }

    def _fake_coach_llm_generator():
        llm_calls["count"] += 1

        async def _generator(messages, max_tokens=None):
            yield '{"headline":"mock","sentiment":"neutral"}'

        return _generator

    monkeypatch.setattr("app.deps.get_connection_pool", lambda: dummy_pool)
    monkeypatch.setattr("app.deps.close_connection_pool", lambda: None)
    monkeypatch.setattr("app.routers.coach.get_connection_pool", lambda: dummy_pool)
    monkeypatch.setattr(
        "app.routers.coach._execute_coach_tools_parallel",
        _fake_execute_coach_tools_parallel,
    )
    monkeypatch.setattr(
        "app.routers.coach.get_coach_llm_generator",
        _fake_coach_llm_generator,
    )
    monkeypatch.setattr(
        "app.routers.coach.parse_coach_response",
        lambda _: (_ParsedCoachResponse(), None),
    )
    monkeypatch.setattr(
        "sse_starlette.sse.AppStatus.should_exit_event",
        _NeverExitEvent(),
        raising=False,
    )
    monkeypatch.setattr(
        "sse_starlette.sse.AppStatus.should_exit",
        False,
        raising=False,
    )
    app.state._test_dummy_pool = dummy_pool
    app.state._test_llm_calls = llm_calls

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    for attr in ("_test_dummy_pool", "_test_llm_calls"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


def _event_index(payload: str, event_name: str) -> int:
    return payload.find(f"event: {event_name}")


def _assert_meta_fields(
    payload: str,
    expected_cache_state: str | None = None,
) -> None:
    assert '"request_mode":' in payload
    assert '"focus_signature":' in payload
    assert '"question_signature":' in payload
    assert '"cache_state":' in payload
    assert (
        '"cache_key_version": "v3"' in payload or '"cache_key_version":"v3"' in payload
    )

    if expected_cache_state is not None:
        assert (
            f'"cache_state": "{expected_cache_state}"' in payload
            or f'"cache_state":"{expected_cache_state}"' in payload
        )


def test_health_endpoint(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_completion_structure(client: TestClient):
    response = client.post(
        "/ai/chat/completion",
        json={"question": "LG 트윈스 2025 시즌 요약해줘"},
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.get("answer"), str)
    assert "verified" in body
    assert "tool_calls" in body
    assert "tool_results" in body


def test_chat_stream_event_order(client: TestClient):
    response = client.post(
        "/ai/chat/stream",
        json={"question": "LG 트윈스 2025 시즌 요약해줘", "style": "compact"},
    )

    assert response.status_code == 200
    text = response.text
    message_idx = _event_index(text, "message")
    meta_idx = _event_index(text, "meta")
    done_idx = _event_index(text, "done")

    assert message_idx != -1
    assert meta_idx != -1
    assert done_idx != -1
    assert message_idx < meta_idx < done_idx


def test_coach_analyze_event_order(client: TestClient):
    payload = {
        "home_team_id": "LG",
        "league_context": {"season_year": 2025},
        "focus": ["recent_form", "unknown"],
        "question_override": "LG 2025 전력 요약",
    }
    with client.stream("POST", "/coach/analyze", json=payload) as response:
        text = "".join(chunk for chunk in response.iter_text())

    assert response.status_code == 200
    message_idx = _event_index(text, "message")
    meta_idx = _event_index(text, "meta")
    done_idx = _event_index(text, "done")

    assert message_idx != -1
    assert meta_idx != -1
    assert done_idx != -1
    assert message_idx < meta_idx < done_idx

    assert '"validation_status":' in text
    assert '"cache_key_version": "v3"' in text
    assert '"cache_state":' in text
    assert '"request_mode":' in text
    assert '"focus_signature":' in text
    assert '"question_signature":' in text
    assert '"focus_signature": "recent_form"' in text


def test_coach_analyze_rejects_invalid_season_year(client: TestClient):
    payload = {
        "home_team_id": "LG",
        "league_context": {"season_year": 266},
    }
    response = client.post("/coach/analyze", json=payload)

    assert response.status_code == 400
    assert response.json().get("detail") == "invalid_season_year_for_analysis"


def test_coach_analyze_cache_split_by_focus(client: TestClient):
    base_payload = {
        "home_team_id": "LG",
        "league_context": {"season_year": 2025},
    }

    with client.stream(
        "POST", "/coach/analyze", json={**base_payload, "focus": ["recent_form"]}
    ) as response_recent_1:
        text_recent_1 = "".join(chunk for chunk in response_recent_1.iter_text())
    with client.stream(
        "POST", "/coach/analyze", json={**base_payload, "focus": ["bullpen"]}
    ) as response_bullpen:
        text_bullpen = "".join(chunk for chunk in response_bullpen.iter_text())
    with client.stream(
        "POST", "/coach/analyze", json={**base_payload, "focus": ["recent_form"]}
    ) as response_recent_2:
        text_recent_2 = "".join(chunk for chunk in response_recent_2.iter_text())

    assert response_recent_1.status_code == 200
    assert response_bullpen.status_code == 200
    assert response_recent_2.status_code == 200

    _assert_meta_fields(text_recent_1, "MISS_GENERATE")
    _assert_meta_fields(text_bullpen, "MISS_GENERATE")
    _assert_meta_fields(text_recent_2, "HIT")

    assert '"focus_signature": "recent_form"' in text_recent_1
    assert '"focus_signature": "bullpen"' in text_bullpen
    assert '"focus_signature": "recent_form"' in text_recent_2
    assert '"request_mode": "manual_detail"' in text_recent_1
    assert '"request_mode": "manual_detail"' in text_bullpen
    assert '"request_mode": "manual_detail"' in text_recent_2
    assert '"cached": true' in text_recent_2

    pool = app.state._test_dummy_pool
    llm_calls = app.state._test_llm_calls
    assert len(pool.cache_rows) == 2
    assert pool.counters["completed_update"] == 2
    assert llm_calls["count"] == 2


def test_coach_analyze_completed_cache_hit_for_stale_entry_does_not_regenerate(
    client: TestClient,
):
    payload = {
        "home_team_id": "LG",
        "away_team_id": "DB",
        "league_context": {
            "season_year": 2025,
            "league_type": "REGULAR",
        },
        "focus": ["recent_form"],
    }

    cache_key, _ = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="LG",
        away_team_code="DB",
        year=2025,
        game_type="REGULAR",
        focus=["recent_form"],
        question_override=None,
    )
    app.state._test_dummy_pool.cache_rows[cache_key] = {
        "status": "COMPLETED",
        "response_json": {
            "headline": "기존 캐시",
            "sentiment": "neutral",
            "key_metrics": [],
            "analysis": {"strengths": [], "weaknesses": [], "risks": []},
            "detailed_markdown": "## 테스트 캐시",
            "coach_note": "기존 캐시에서 반환",
        },
        "error_message": None,
        "updated_at": datetime.now(timezone.utc) - timedelta(days=14),
    }

    with client.stream("POST", "/coach/analyze", json=payload) as first_response:
        first_text = "".join(chunk for chunk in first_response.iter_text())

    with client.stream("POST", "/coach/analyze", json=payload) as second_response:
        second_text = "".join(chunk for chunk in second_response.iter_text())

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert '"cache_state": "HIT"' in first_text
    assert '"cache_state": "HIT"' in second_text
    assert '"cached": true' in first_text
    assert '"cached": true' in second_text
    _assert_meta_fields(first_text, "HIT")
    _assert_meta_fields(second_text, "HIT")

    llm_calls = app.state._test_llm_calls
    assert llm_calls["count"] == 0


def test_coach_analyze_cache_split_by_question_override(client: TestClient):
    base_payload = {
        "home_team_id": "LG",
        "league_context": {"season_year": 2025},
        "focus": ["recent_form"],
    }

    with client.stream(
        "POST",
        "/coach/analyze",
        json={**base_payload, "question_override": "LG 2025 최근 전력 분석"},
    ) as response_q1_1:
        text_q1_1 = "".join(chunk for chunk in response_q1_1.iter_text())
    with client.stream(
        "POST",
        "/coach/analyze",
        json={**base_payload, "question_override": "LG 2025 불펜 리스크 분석"},
    ) as response_q2:
        text_q2 = "".join(chunk for chunk in response_q2.iter_text())
    with client.stream(
        "POST",
        "/coach/analyze",
        json={**base_payload, "question_override": "LG 2025 최근 전력 분석"},
    ) as response_q1_2:
        text_q1_2 = "".join(chunk for chunk in response_q1_2.iter_text())

    assert response_q1_1.status_code == 200
    assert response_q2.status_code == 200
    assert response_q1_2.status_code == 200

    assert '"focus_signature": "recent_form"' in text_q1_1
    assert '"focus_signature": "recent_form"' in text_q2
    assert '"request_mode": "manual_detail"' in text_q1_1
    assert '"request_mode": "manual_detail"' in text_q2
    assert '"cached": true' in text_q1_2

    pool = app.state._test_dummy_pool
    llm_calls = app.state._test_llm_calls
    assert len(pool.cache_rows) == 2
    assert pool.counters["completed_update"] == 2
    assert llm_calls["count"] == 2


def test_coach_analyze_manual_meta_fields_and_question_signature(client: TestClient):
    payload = {
        "home_team_id": "LG",
        "league_context": {"season_year": 2025},
        "focus": ["recent_form", "bullpen"],
        "question_override": "LG 2025 상대 전력 비교 분석",
        "request_mode": "manual_detail",
    }

    with client.stream("POST", "/coach/analyze", json=payload) as response:
        text = "".join(chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert '"request_mode": "manual_detail"' in text
    assert '"focus_signature": "recent_form+bullpen"' in text
    assert '"question_signature": "q:' in text
    assert '"cache_state": "' in text
    assert '"validation_status": "success"' in text


def test_coach_analyze_auto_brief_key_stable_and_cached(client: TestClient):
    base_payload = {
        "home_team_id": "LG",
        "league_context": {"season_year": 2025},
        "focus": ["recent_form", "bullpen"],
        "request_mode": "auto_brief",
    }

    with client.stream(
        "POST",
        "/coach/analyze",
        json={
            **base_payload,
            "question_override": "LG 2025 최근 전력 중심 분석",
            "focus": ["recent_form"],
        },
    ) as response_auto_1:
        text_auto_1 = "".join(chunk for chunk in response_auto_1.iter_text())
    with client.stream(
        "POST",
        "/coach/analyze",
        json={
            **base_payload,
            "question_override": "LG 2025 불펜 리스크 분석",
            "focus": ["bullpen"],
            "request_mode": "auto_brief",
        },
    ) as response_auto_2:
        text_auto_2 = "".join(chunk for chunk in response_auto_2.iter_text())

    assert response_auto_1.status_code == 200
    assert response_auto_2.status_code == 200

    assert '"request_mode": "auto_brief"' in text_auto_1
    assert '"request_mode": "auto_brief"' in text_auto_2
    assert '"focus_signature": "recent_form"' in text_auto_1
    assert '"focus_signature": "recent_form"' in text_auto_2
    assert '"question_signature": "auto"' in text_auto_1
    assert '"question_signature": "auto"' in text_auto_2
    _assert_meta_fields(text_auto_1, "MISS_GENERATE")
    _assert_meta_fields(text_auto_2, "HIT")
    assert '"cached": true' in text_auto_2

    pool = app.state._test_dummy_pool
    llm_calls = app.state._test_llm_calls
    assert len(pool.cache_rows) == 1
    assert pool.counters["completed_update"] == 1
    assert llm_calls["count"] == 1


def test_coach_analyze_auto_vs_manual_use_different_max_tokens(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    captured_tokens: dict[str, int | None] = {}

    def _fake_coach_llm_generator():
        async def _generator(*_args: object, max_tokens=None, **_kwargs: object):
            captured_tokens["value"] = max_tokens
            yield '{"headline":"mock","sentiment":"neutral"}'

        return _generator

    dummy_settings = SimpleNamespace(
        openrouter_api_key="OPENROUTER_TEST_KEY",
        openrouter_base_url="https://openrouter.example/api/v1",
        openrouter_model="global-model",
        openrouter_referer=None,
        openrouter_app_title=None,
        coach_openrouter_model="coach-model-v1",
        coach_openrouter_fallback_models=[],
        coach_llm_read_timeout=10.0,
        coach_max_output_tokens=2000,
        coach_brief_max_output_tokens=8000,
    )
    monkeypatch.setattr("app.routers.coach.get_settings", lambda: dummy_settings)
    monkeypatch.setattr("app.deps.get_settings", lambda: dummy_settings)
    monkeypatch.setattr(
        "app.routers.coach.get_coach_llm_generator", _fake_coach_llm_generator
    )

    with client.stream(
        "POST",
        "/coach/analyze",
        json={
            "home_team_id": "LG",
            "league_context": {"season_year": 2025},
            "focus": ["recent_form"],
            "request_mode": "auto_brief",
        },
    ) as auto_response:
        auto_text = "".join(chunk for chunk in auto_response.iter_text())

    assert auto_response.status_code == 200
    assert '"request_mode": "auto_brief"' in auto_text
    assert captured_tokens.get("value") == 8000

    captured_tokens["value"] = None
    with client.stream(
        "POST",
        "/coach/analyze",
        json={
            "home_team_id": "LG",
            "league_context": {"season_year": 2025},
            "focus": ["recent_form"],
            "request_mode": "manual_detail",
            "question_override": "manual detail question variant",
        },
    ) as manual_response:
        manual_text = "".join(chunk for chunk in manual_response.iter_text())

    assert manual_response.status_code == 200
    assert '"request_mode": "manual_detail"' in manual_text
    assert captured_tokens.get("value") == 2000


def test_coach_analyze_pending_state_waits_and_skips_generation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("app.routers.coach.PENDING_WAIT_TIMEOUT_SECONDS", 0.05)

    payload = {
        "home_team_id": "LG",
        "away_team_id": "DB",
        "league_context": {"season_year": 2025, "league_type": "REGULAR"},
        "focus": ["recent_form"],
    }
    cache_key, _ = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="LG",
        away_team_code="DB",
        year=2025,
        game_type="REGULAR",
        focus=["recent_form"],
        question_override=None,
    )
    app.state._test_dummy_pool.cache_rows[cache_key] = {
        "status": "PENDING",
        "response_json": None,
        "error_message": "in progress",
        "updated_at": datetime.now(timezone.utc),
    }

    with client.stream("POST", "/coach/analyze", json=payload) as response:
        text = "".join(chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert (
        '"cache_state":"PENDING_WAIT"' in text
        or '"cache_state": "PENDING_WAIT"' in text
    )
    assert '"in_progress": true' in text
    assert '"request_mode": "manual_detail"' in text
    assert '"focus_signature": "recent_form"' in text
    assert '"question_signature":' in text
    _assert_meta_fields(text, "PENDING_WAIT")
    assert app.state._test_llm_calls["count"] == 0


def test_coach_analyze_stale_pending_triggers_regeneration(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    from app.routers.coach import PENDING_STALE_SECONDS

    monkeypatch.setattr("app.routers.coach.PENDING_WAIT_TIMEOUT_SECONDS", 0.05)

    payload = {
        "home_team_id": "LG",
        "away_team_id": "DB",
        "league_context": {"season_year": 2025, "league_type": "REGULAR"},
        "focus": ["recent_form"],
    }
    cache_key, _ = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="LG",
        away_team_code="DB",
        year=2025,
        game_type="REGULAR",
        focus=["recent_form"],
        question_override=None,
    )
    app.state._test_dummy_pool.cache_rows[cache_key] = {
        "status": "PENDING",
        "response_json": None,
        "error_message": "expired",
        "updated_at": datetime.now(timezone.utc)
        - timedelta(seconds=PENDING_STALE_SECONDS + 10),
    }

    with client.stream("POST", "/coach/analyze", json=payload) as response:
        text = "".join(chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert (
        '"cache_state": "PENDING_STALE_TAKEOVER"' in text
        or '"cache_state":"PENDING_STALE_TAKEOVER"' in text
    )
    assert '"request_mode": "manual_detail"' in text
    assert '"focus_signature": "recent_form"' in text
    assert '"question_signature":' in text
    _assert_meta_fields(text, "PENDING_STALE_TAKEOVER")
    assert app.state._test_llm_calls["count"] == 1


def test_coach_analyze_failed_lock_blocks_regeneration(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    payload = {
        "home_team_id": "LG",
        "away_team_id": "DB",
        "league_context": {"season_year": 2025, "league_type": "REGULAR"},
        "focus": ["recent_form"],
    }
    cache_key, _ = build_coach_cache_key(
        schema_version="v3",
        prompt_version="v5_focus",
        home_team_code="LG",
        away_team_code="DB",
        year=2025,
        game_type="REGULAR",
        focus=["recent_form"],
        question_override=None,
    )
    app.state._test_dummy_pool.cache_rows[cache_key] = {
        "status": "FAILED",
        "response_json": None,
        "error_message": "hard error",
        "updated_at": datetime.now(timezone.utc),
    }

    with client.stream("POST", "/coach/analyze", json=payload) as response:
        text = "".join(chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert (
        '"cache_state":"FAILED_LOCKED"' in text
        or '"cache_state": "FAILED_LOCKED"' in text
    )
    assert '"request_mode": "manual_detail"' in text
    assert '"focus_signature": "recent_form"' in text
    assert '"question_signature":' in text
    _assert_meta_fields(text, "FAILED_LOCKED")
    assert app.state._test_llm_calls["count"] == 0


@pytest.mark.asyncio
async def test_coach_llm_generator_prefers_coach_openrouter_settings(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.deps import get_coach_llm_generator

    captured: dict[str, Any] = {}

    dummy_settings = SimpleNamespace(
        openrouter_api_key="OPENROUTER_TEST_KEY",
        openrouter_base_url="https://openrouter.example/api/v1",
        openrouter_referer=None,
        openrouter_app_title=None,
        coach_openrouter_model="coach-model-v1",
        coach_openrouter_fallback_models=["fallback-model-v1"],
        coach_llm_read_timeout=10.0,
        coach_max_output_tokens=2000,
    )

    class _FakeRequestContext:
        def __init__(self, request_payload: dict[str, Any]):
            self._payload = request_payload
            self.status_code = 200

        async def __aenter__(self):
            captured["payload"] = self._payload
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"coach"}}]}'
            yield "data: [DONE]"

        async def aread(self):
            return b""

        def raise_for_status(self):
            return None

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(
            self, method: str, url: str, json: dict[str, Any], headers: dict[str, str]
        ):
            return _FakeRequestContext(json)

    class _FakeTimeout:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeHttpxModule:
        AsyncClient = _FakeAsyncClient
        Timeout = _FakeTimeout

    monkeypatch.setattr("app.deps.get_settings", lambda: dummy_settings)
    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpxModule)

    coach_llm = get_coach_llm_generator()
    chunks: list[str] = []
    async for chunk in coach_llm(
        [{"role": "user", "content": "hello"}], max_tokens=512
    ):
        chunks.append(chunk)

    assert chunks == ["coach"]
    assert captured["payload"]["model"] == "coach-model-v1"


@pytest.mark.asyncio
async def test_coach_llm_generator_falls_back_to_coach_fallback_models(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.deps import get_coach_llm_generator

    captured_payloads: list[dict[str, Any]] = []

    dummy_settings = SimpleNamespace(
        openrouter_api_key="OPENROUTER_TEST_KEY",
        openrouter_base_url="https://openrouter.example/api/v1",
        openrouter_referer=None,
        openrouter_app_title=None,
        coach_openrouter_model="coach-model-v1",
        coach_openrouter_fallback_models=["fallback-model-v1", "third-fallback-model"],
        coach_llm_read_timeout=10.0,
        coach_max_output_tokens=2000,
    )

    class _FakeRequestContext:
        def __init__(self, request_payload: dict[str, Any]):
            self._payload = request_payload
            self.status_code = 200

        async def __aenter__(self):
            captured_payloads.append(self._payload)
            if self._payload["model"] == "coach-model-v1":
                raise RuntimeError("primary model failed")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"coach fallback"}}]}'
            yield "data: [DONE]"

        async def aread(self):
            return b""

        def raise_for_status(self):
            return None

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(
            self,
            method: str,
            url: str,
            json: dict[str, Any],
            headers: dict[str, str],
        ):
            return _FakeRequestContext(json)

    class _FakeTimeout:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeHttpxModule:
        AsyncClient = _FakeAsyncClient
        Timeout = _FakeTimeout

    monkeypatch.setattr("app.deps.get_settings", lambda: dummy_settings)
    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpxModule)

    coach_llm = get_coach_llm_generator()
    chunks: list[str] = []
    async for chunk in coach_llm(
        [{"role": "user", "content": "hello"}], max_tokens=256
    ):
        chunks.append(chunk)

    assert chunks == ["coach fallback"]
    assert [payload["model"] for payload in captured_payloads] == [
        "coach-model-v1",
        "fallback-model-v1",
    ]
    assert captured_payloads[-1]["max_tokens"] == 256
