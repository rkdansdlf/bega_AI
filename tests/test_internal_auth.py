from types import SimpleNamespace

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.deps import require_ai_internal_token
from app.routers import chat_stream, coach, vision


class _DummyAgent:
    async def process_query(self, question, context=None):
        return {
            "answer": (
                "## 상세 내역\n"
                "| 항목 | 내용 |\n"
                "| --- | --- |\n"
                "| 상태 | 정상 |\n\n"
                "## 핵심 지표\n"
                "- 테스트 응답\n\n"
                "## 인사이트\n"
                "- 내부 토큰 인증 통과\n\n"
                "## 데이터 출처\n"
                "- 출처: 테스트"
            ),
            "intent": "test",
            "tool_calls": [],
            "tool_results": [],
            "data_sources": [],
            "verified": True,
            "visualizations": [],
        }

    def _convert_team_id_to_name(self, team_id):
        return team_id


def _build_client() -> TestClient:
    test_app = FastAPI()

    @test_app.get("/protected")
    def protected(_: None = Depends(require_ai_internal_token)):
        return {"ok": True}

    test_app.include_router(chat_stream.router)
    test_app.include_router(vision.router, prefix="/ai")
    test_app.include_router(coach.router, prefix="/ai")
    test_app.dependency_overrides[chat_stream.get_agent] = lambda: _DummyAgent()
    test_app.dependency_overrides[chat_stream.rate_limit_chat_dependency] = lambda: None
    test_app.dependency_overrides[vision.rate_limit_vision_dependency] = lambda: None
    test_app.dependency_overrides[coach.get_agent] = lambda: _DummyAgent()
    test_app.dependency_overrides[coach.rate_limit_coach_dependency] = lambda: None
    return TestClient(test_app)


def test_require_ai_internal_token_rejects_missing_token(monkeypatch):
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.deps.record_security_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )

    with _build_client() as client:
        response = client.get("/protected")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid internal API token"
    assert events == [
        (
            "AI_INTERNAL_AUTH_REJECT",
            {
                "endpoint": "/protected",
                "detail": "missing_or_invalid_token",
            },
        )
    ]


def test_require_ai_internal_token_accepts_bearer_authorization(monkeypatch):
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr(
        "app.deps.record_security_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )

    with _build_client() as client:
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer expected-token"},
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert events == []


def test_require_ai_internal_token_returns_503_when_misconfigured(monkeypatch):
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token=None),
    )
    monkeypatch.setattr(
        "app.deps.record_security_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )

    with _build_client() as client:
        response = client.get("/protected")

    assert response.status_code == 503
    assert response.json()["detail"] == "AI internal authentication is not configured"
    assert events == [
        (
            "AI_INTERNAL_AUTH_MISCONFIGURED",
            {
                "endpoint": "/protected",
                "detail": "missing_ai_internal_token",
            },
        )
    ]


def test_chat_completion_requires_internal_token(monkeypatch):
    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr("app.deps.record_security_event", lambda *args, **kwargs: None)

    payload = {
        "question": "테스트 질문",
        "history": [{"role": "user", "content": "직전 질문"}],
    }

    with _build_client() as client:
        response = client.post("/ai/chat/completion", json=payload)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid internal API token"


def test_chat_completion_accepts_internal_token(monkeypatch):
    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr("app.deps.record_security_event", lambda *args, **kwargs: None)

    payload = {
        "question": "테스트 질문",
        "history": [{"role": "user", "content": "직전 질문"}],
    }

    with _build_client() as client:
        response = client.post(
            "/ai/chat/completion",
            json=payload,
            headers={"X-Internal-Api-Key": "expected-token"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["cached"] is False
    assert body["intent"] == "test"
    assert "## 상세 내역" in body["answer"]


def test_ai_vision_ticket_requires_internal_token(monkeypatch):
    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr("app.deps.record_security_event", lambda *args, **kwargs: None)

    with _build_client() as client:
        response = client.post(
            "/ai/vision/ticket",
            files={"file": ("ticket.png", b"fake-image", "image/png")},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid internal API token"


def test_ai_vision_ticket_accepts_internal_token(monkeypatch):
    async def _sentinel_reader(*args, **kwargs):
        raise HTTPException(status_code=418, detail="vision-auth-passed")

    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr("app.deps.record_security_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "app.routers.vision._read_ticket_image_with_limit", _sentinel_reader
    )

    with _build_client() as client:
        response = client.post(
            "/ai/vision/ticket",
            files={"file": ("ticket.png", b"fake-image", "image/png")},
            headers={"X-Internal-Api-Key": "expected-token"},
        )

    assert response.status_code == 418
    assert response.json()["detail"] == "vision-auth-passed"


def test_ai_coach_analyze_requires_internal_token(monkeypatch):
    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr("app.deps.record_security_event", lambda *args, **kwargs: None)

    payload = {
        "home_team_id": "LG",
        "away_team_id": "KT",
        "request_mode": "auto_brief",
    }

    with _build_client() as client:
        response = client.post("/ai/coach/analyze", json=payload)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid internal API token"


def test_ai_coach_analyze_accepts_internal_token(monkeypatch):
    def _sentinel_pool():
        raise HTTPException(status_code=418, detail="coach-auth-passed")

    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token="expected-token"),
    )
    monkeypatch.setattr("app.deps.record_security_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.routers.coach.get_connection_pool", _sentinel_pool)

    payload = {
        "home_team_id": "LG",
        "away_team_id": "KT",
        "request_mode": "auto_brief",
    }

    with _build_client() as client:
        response = client.post(
            "/ai/coach/analyze",
            json=payload,
            headers={"X-Internal-Api-Key": "expected-token"},
        )

    assert response.status_code == 418
    assert response.json()["detail"] == "coach-auth-passed"
