from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.streaming.http_errors import install_ai_stream_http_error_handler
from app.streaming.versioned_sse import negotiate_event_version


def test_unsupported_version_returns_top_level_canonical_json() -> None:
    app = FastAPI()
    install_ai_stream_http_error_handler(app)

    @app.get("/stream")
    async def stream(version: str) -> dict[str, int]:
        return {"version": negotiate_event_version(version, endpoint="chat")}

    response = TestClient(app).get("/stream", params={"version": "3"})

    assert response.status_code == 406
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "code": "AI_EVENT_VERSION_UNSUPPORTED",
        "message": "지원하지 않는 AI 이벤트 버전입니다.",
        "detail": None,
        "retryable": False,
        "retry_after_seconds": None,
        "supported_versions": ["1", "2"],
    }
