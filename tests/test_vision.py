from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import vision

AI_INTERNAL_TEST_TOKEN = "local-test-token"


@pytest.fixture
def client(monkeypatch):
    test_app = FastAPI()
    test_app.include_router(vision.router)
    test_app.dependency_overrides[vision.rate_limit_vision_dependency] = lambda: None
    monkeypatch.setattr(
        "app.deps.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token=AI_INTERNAL_TEST_TOKEN),
    )
    monkeypatch.setattr("app.deps.record_security_event", lambda *args, **kwargs: None)
    with TestClient(
        test_app,
        headers={"X-Internal-Api-Key": AI_INTERNAL_TEST_TOKEN},
    ) as test_client:
        yield test_client


@pytest.fixture
def mock_settings():
    with patch("app.routers.vision.settings", autospec=True) as mock:
        yield mock


@pytest.fixture
def mock_genai():
    with patch("app.routers.vision.genai") as mock:
        yield mock


@pytest.fixture
def mock_image_open():
    with patch("app.routers.vision.Image.open") as mock:
        yield mock


def test_analyze_ticket_gemini_success(
    client, mock_settings, mock_genai, mock_image_open
):
    # Configure settings for Gemini
    mock_settings.llm_provider = "gemini"
    mock_settings.gemini_api_key = "test_key"
    mock_settings.vision_model = "gemini-2.0-flash"

    # Mock Gemini response
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"date": "2024-05-05", "stadium": "Jamsil", "homeTeam": "LG", "awayTeam": "Doosan"}'
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    # Mock Image.open
    mock_image = MagicMock()
    mock_image_open.return_value = mock_image

    # Create dummy image file
    files = {"file": ("ticket.jpg", b"fake_image_content", "image/jpeg")}

    response = client.post("/vision/ticket", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "2024-05-05"
    assert data["stadium"] == "Jamsil"
    assert data["homeTeam"] == "LG"

    # Verify Gemini was called
    mock_genai.configure.assert_called_with(api_key="test_key")
    mock_genai.GenerativeModel.assert_called_with("gemini-2.0-flash")
    mock_model.generate_content.assert_called_once()


def test_analyze_ticket_openrouter_success(client, mock_settings):
    # Configure settings for OpenRouter
    mock_settings.llm_provider = "openrouter"
    mock_settings.openrouter_api_key = "test_router_key"
    mock_settings.vision_model = "google/gemini-2.0-flash-001"

    # Mock shared OpenRouter client
    with patch("app.routers.vision.get_shared_httpx_client") as mock_get_client:
        mock_instance = AsyncMock()
        mock_get_client.return_value = mock_instance
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"date": "2024-05-05", "stadium": "Incheon", "homeTeam": "SSG", "awayTeam": "KT"}'
                    }
                }
            ]
        }
        mock_instance.post.return_value = mock_response

        files = {"file": ("ticket.jpg", b"fake_image_content", "image/jpeg")}
        response = client.post("/vision/ticket", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["stadium"] == "Incheon"
        assert data["homeTeam"] == "SSG"
