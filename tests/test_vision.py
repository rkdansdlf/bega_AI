from types import SimpleNamespace

import httpx
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import vision

AI_INTERNAL_TEST_TOKEN = "local-test-token"
GEMMA_VISION_MODEL = "google/gemma-4-31b-it:free"
MISTRAL_VISION_FALLBACK_MODEL = "mistralai/mistral-small-3.2-24b-instruct"

# 실제 이미지는 아니지만 형식 판별에 쓰는 헤더 구조는 갖춘 최소 바이트.
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"fake_image_content"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 13
WEBP_BYTES = b"RIFF" + b"\x1a\x00\x00\x00" + b"WEBP" + b"VP8 " + b"\x00" * 10


@pytest.fixture
def client(monkeypatch):
    test_app = FastAPI()
    test_app.include_router(vision.router)
    test_app.dependency_overrides[vision.rate_limit_vision_dependency] = lambda: None
    monkeypatch.setattr(
        "app.internal_auth.get_settings",
        lambda: SimpleNamespace(resolved_ai_internal_token=AI_INTERNAL_TEST_TOKEN),
    )
    monkeypatch.setattr(
        "app.internal_auth.record_security_event", lambda *args, **kwargs: None
    )
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
def mock_genai_types():
    with patch("app.routers.vision.genai_types", create=True) as mock:
        yield mock


def test_analyze_ticket_gemini_success(
    client, mock_settings, mock_genai, mock_genai_types
):
    # Configure settings for Gemini
    mock_settings.llm_provider = "gemini"
    mock_settings.gemini_api_key = "test_key"
    mock_settings.vision_model = "gemini-2.0-flash"

    # Mock Gemini response
    mock_sdk_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"date": "2024-05-05", "stadium": "Jamsil", "homeTeam": "LG", "awayTeam": "Doosan"}'
    mock_sdk_client.models.generate_content.return_value = mock_response
    mock_sdk_client.__enter__.return_value = mock_sdk_client
    mock_genai.Client.return_value = mock_sdk_client
    image_part = object()
    mock_genai_types.Part.from_bytes.return_value = image_part

    # Create dummy image file
    files = {"file": ("ticket.jpg", JPEG_BYTES, "image/jpeg")}

    response = client.post("/vision/ticket", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["date"] == "2024-05-05"
    assert data["stadium"] == "Jamsil"
    assert data["homeTeam"] == "LG"

    # Verify Gemini was called
    mock_genai.Client.assert_called_once_with(api_key="test_key")
    mock_genai_types.Part.from_bytes.assert_called_once_with(
        data=JPEG_BYTES,
        mime_type="image/jpeg",
    )
    generate_call = mock_sdk_client.models.generate_content.call_args.kwargs
    assert generate_call["model"] == "gemini-2.0-flash"
    assert generate_call["contents"][1] is image_part


def test_analyze_ticket_openrouter_success(client, mock_settings):
    # Configure settings for OpenRouter
    mock_settings.llm_provider = "openrouter"
    mock_settings.openrouter_api_key = "test_router_key"
    mock_settings.vision_model = GEMMA_VISION_MODEL
    mock_settings.vision_fallback_models = []
    mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
    mock_settings.openrouter_referer = None
    mock_settings.openrouter_app_title = None

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

        files = {"file": ("ticket.jpg", JPEG_BYTES, "image/jpeg")}
        response = client.post("/vision/ticket", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["stadium"] == "Incheon"
        assert data["homeTeam"] == "SSG"
        _, kwargs = mock_instance.post.call_args
        assert kwargs["json"]["model"] == GEMMA_VISION_MODEL
        content = kwargs["json"]["messages"][0]["content"]
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_analyze_ticket_openrouter_falls_back_to_mistral_on_primary_error(
    client, mock_settings
):
    mock_settings.llm_provider = "openrouter"
    mock_settings.openrouter_api_key = "test_router_key"
    mock_settings.vision_model = GEMMA_VISION_MODEL
    mock_settings.vision_fallback_models = [MISTRAL_VISION_FALLBACK_MODEL]
    mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
    mock_settings.openrouter_referer = None
    mock_settings.openrouter_app_title = None

    with patch("app.routers.vision.get_shared_httpx_client") as mock_get_client:
        mock_instance = AsyncMock()
        mock_get_client.return_value = mock_instance

        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        primary_response = MagicMock()
        primary_response.status_code = 429
        primary_response.text = "quota exceeded"
        primary_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "quota exceeded",
            request=request,
            response=primary_response,
        )

        fallback_response = MagicMock()
        fallback_response.status_code = 200
        fallback_response.raise_for_status.return_value = None
        fallback_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"date": "2026-05-07", "stadium": "Incheon", "homeTeam": "SSG", "awayTeam": "NC", "seat": 5, "price": "19,000"}'
                    }
                }
            ]
        }
        mock_instance.post.side_effect = [primary_response, fallback_response]

        files = {"file": ("ticket.jpg", JPEG_BYTES, "image/jpeg")}
        response = client.post("/vision/ticket", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["date"] == "2026-05-07"
        assert data["homeTeam"] == "SSG"
        assert data["seat"] == "5"
        assert data["price"] == 19000
        calls = mock_instance.post.call_args_list
        assert calls[0].kwargs["json"]["model"] == GEMMA_VISION_MODEL
        assert calls[1].kwargs["json"]["model"] == MISTRAL_VISION_FALLBACK_MODEL


# --- 업로드 이미지 형식 검증 ------------------------------------------------

TICKET_URL = "/vision/ticket"
SEAT_VIEW_URL = "/vision/seat-view-classify"
TICKET_JSON = '{"date": "2024-05-05", "stadium": "Incheon"}'
SEAT_VIEW_JSON = '{"label": "OTHER", "confidence": 0.5, "reason": "ok"}'
ENDPOINTS = [
    pytest.param(TICKET_URL, TICKET_JSON, id="ticket"),
    pytest.param(SEAT_VIEW_URL, SEAT_VIEW_JSON, id="seat-view"),
]


def _configure_openrouter(mock_settings, fallbacks=()):
    mock_settings.llm_provider = "openrouter"
    mock_settings.openrouter_api_key = "test_router_key"
    mock_settings.vision_model = GEMMA_VISION_MODEL
    mock_settings.vision_fallback_models = list(fallbacks)
    mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
    mock_settings.openrouter_referer = None
    mock_settings.openrouter_app_title = None


def _ok_response(content: str):
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


def _error_response(status: int, body: str):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = MagicMock()
    response.status_code = status
    response.text = body
    # 예외 메시지에도 본문이 들어 있어, 예외를 통째로 로그에 남기면 걸러진다.
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        body, request=request, response=response
    )
    return response


@pytest.mark.parametrize(
    "contents, expected_type",
    [
        (JPEG_BYTES, "image/jpeg"),
        (PNG_BYTES, "image/png"),
        (WEBP_BYTES, "image/webp"),
    ],
    ids=["jpeg", "png", "webp"],
)
def test_detect_image_type_recognizes_supported_formats(contents, expected_type):
    assert vision._detect_image_type(contents) == expected_type


@pytest.mark.parametrize(
    "contents",
    [
        b"",
        b"\xff\xd8",  # JPEG 시그니처보다 짧다
        b"<html><body>not an image</body></html>",
        b"%PDF-1.7\n",
        b"GIF89a\x01\x00\x01\x00",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,  # PNG 시그니처만 있고 IHDR 이 없다
        b"RIFF\x1a\x00\x00\x00WAVEfmt " + b"\x00" * 10,  # RIFF 지만 WebP 가 아니다
        b"RIFF\x1a\x00\x00\x00WEBPJUNK" + b"\x00" * 10,  # WebP 청크 종류가 아니다
    ],
    ids=[
        "empty",
        "short-jpeg",
        "html",
        "pdf",
        "gif",
        "png-no-ihdr",
        "wav",
        "webp-junk",
    ],
)
def test_detect_image_type_rejects_non_images(contents):
    assert vision._detect_image_type(contents) is None


@pytest.mark.parametrize("url, model_json", ENDPOINTS)
@pytest.mark.parametrize(
    "contents, declared, expected_type",
    [
        (PNG_BYTES, "image/png", "image/png"),
        (WEBP_BYTES, "image/webp", "image/webp"),
        # 선언된 형식과 실제 형식이 달라도 허용 형식이면 실제 형식으로 보낸다.
        (JPEG_BYTES, "image/png", "image/jpeg"),
    ],
    ids=["png", "webp", "declared-png-actually-jpeg"],
)
def test_upload_is_sent_with_the_detected_image_type(
    client, mock_settings, url, model_json, contents, declared, expected_type
):
    _configure_openrouter(mock_settings)

    with patch("app.routers.vision.get_shared_httpx_client") as mock_get_client:
        mock_instance = AsyncMock()
        mock_get_client.return_value = mock_instance
        mock_instance.post.return_value = _ok_response(model_json)

        response = client.post(url, files={"file": ("upload.bin", contents, declared)})

        assert response.status_code == 200
        content = mock_instance.post.call_args.kwargs["json"]["messages"][0]["content"]
        assert content[1]["image_url"]["url"].startswith(
            f"data:{expected_type};base64,"
        )


@pytest.mark.parametrize("url, model_json", ENDPOINTS)
@pytest.mark.parametrize(
    "contents",
    [
        b"<html><body>not an image</body></html>",
        b"%PDF-1.7\n" + b"x" * 64,
        b"GIF89a" + b"\x00" * 32,
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
    ],
    ids=["html", "pdf", "gif", "png-no-ihdr"],
)
def test_non_image_bytes_are_rejected_before_calling_the_model(
    client, mock_settings, mock_genai, url, model_json, contents
):
    _configure_openrouter(mock_settings)

    with patch("app.routers.vision.get_shared_httpx_client") as mock_get_client:
        mock_instance = AsyncMock()
        mock_get_client.return_value = mock_instance

        # 헤더는 허용 형식(image/jpeg)이라고 주장한다.
        response = client.post(
            url, files={"file": ("ticket.jpg", contents, "image/jpeg")}
        )

        assert response.status_code == 415
        mock_instance.post.assert_not_called()
        mock_genai.Client.assert_not_called()


@pytest.mark.parametrize("url, model_json", ENDPOINTS)
def test_unsupported_declared_type_is_still_rejected(
    client, mock_settings, url, model_json
):
    _configure_openrouter(mock_settings)

    response = client.post(url, files={"file": ("a.gif", JPEG_BYTES, "image/gif")})

    assert response.status_code == 415


@pytest.mark.parametrize("url, model_json", ENDPOINTS)
def test_empty_upload_is_still_a_bad_request(client, mock_settings, url, model_json):
    _configure_openrouter(mock_settings)

    response = client.post(url, files={"file": ("a.jpg", b"", "image/jpeg")})

    assert response.status_code == 400


# --- 외부 응답/모델 출력은 로그에 남기지 않는다 ------------------------------

PROVIDER_BODY_MARKER = "SENSITIVE_PROVIDER_BODY"
MODEL_OUTPUT_MARKER = "SENSITIVE_MODEL_TEXT"


@pytest.mark.parametrize("url, model_json", ENDPOINTS)
def test_provider_error_body_is_not_logged(
    client, mock_settings, caplog, url, model_json
):
    caplog.set_level("DEBUG")
    _configure_openrouter(mock_settings)

    with patch("app.routers.vision.get_shared_httpx_client") as mock_get_client:
        mock_instance = AsyncMock()
        mock_get_client.return_value = mock_instance
        mock_instance.post.return_value = _error_response(502, PROVIDER_BODY_MARKER)

        response = client.post(url, files={"file": ("a.jpg", JPEG_BYTES, "image/jpeg")})

    assert response.status_code == 500
    assert response.json()["detail"] == "OpenRouter API error: 502"
    assert PROVIDER_BODY_MARKER not in caplog.text
    # 원인 파악에 필요한 상태 코드는 남긴다.
    assert "status=502" in caplog.text


def test_fallback_warning_does_not_log_provider_error_body(
    client, mock_settings, caplog
):
    caplog.set_level("DEBUG")
    _configure_openrouter(mock_settings, fallbacks=[MISTRAL_VISION_FALLBACK_MODEL])

    with patch("app.routers.vision.get_shared_httpx_client") as mock_get_client:
        mock_instance = AsyncMock()
        mock_get_client.return_value = mock_instance
        mock_instance.post.side_effect = [
            _error_response(429, PROVIDER_BODY_MARKER),
            _ok_response(TICKET_JSON),
        ]

        response = client.post(
            TICKET_URL, files={"file": ("a.jpg", JPEG_BYTES, "image/jpeg")}
        )

    assert response.status_code == 200
    assert PROVIDER_BODY_MARKER not in caplog.text
    assert "trying fallback" in caplog.text
    assert "status=429" in caplog.text


@pytest.mark.parametrize(
    "url, bad_output",
    [
        # confidence 가 숫자가 아니라서 pydantic ValidationError 가 난다(입력값이 메시지에 포함됨).
        (
            SEAT_VIEW_URL,
            '{"label": "OTHER", "confidence": "%s", "reason": "x"}'
            % MODEL_OUTPUT_MARKER,
        ),
        (TICKET_URL, "%s is not json" % MODEL_OUTPUT_MARKER),
    ],
    ids=["seat-view-validation-error", "ticket-json-error"],
)
def test_unusable_model_output_is_not_logged(
    client, mock_settings, caplog, url, bad_output
):
    caplog.set_level("DEBUG")
    _configure_openrouter(mock_settings)

    with patch("app.routers.vision.get_shared_httpx_client") as mock_get_client:
        mock_instance = AsyncMock()
        mock_get_client.return_value = mock_instance
        mock_instance.post.return_value = _ok_response(bad_output)

        response = client.post(url, files={"file": ("a.jpg", JPEG_BYTES, "image/jpeg")})

    assert response.status_code == 500
    assert MODEL_OUTPUT_MARKER not in caplog.text
    assert "unusable response" in caplog.text
