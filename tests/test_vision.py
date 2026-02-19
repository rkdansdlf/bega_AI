import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.config import get_settings

client = TestClient(app)


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


def test_analyze_ticket_gemini_success(mock_settings, mock_genai, mock_image_open):
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


def test_analyze_ticket_openrouter_success(mock_settings):
    # Configure settings for OpenRouter
    mock_settings.llm_provider = "openrouter"
    mock_settings.openrouter_api_key = "test_router_key"
    mock_settings.vision_model = "google/gemini-2.0-flash-001"

    # Mock httpx.AsyncClient
    with patch("app.routers.vision.httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_instance

        mock_response = MagicMock()
        mock_response.status_code = 200
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
