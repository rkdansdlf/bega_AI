from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ValidationError, field_validator
from typing import Optional, Type, TypeVar
import logging

from ..config import get_settings
from ..core.http_clients import get_shared_httpx_client
from ..core.ratelimit import rate_limit_vision_dependency
from ..deps import require_ai_internal_token
import httpx
import base64
import json
from google import genai
from google.genai import types as genai_types
import tempfile

router = APIRouter(prefix="/vision", tags=["vision"])
settings = get_settings()
logger = logging.getLogger(__name__)
MAX_TICKET_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_TICKET_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
OPENROUTER_CHAT_COMPLETIONS_PATH = "/chat/completions"
T = TypeVar("T", bound=BaseModel)


def _normalize_content_type(content_type: Optional[str]) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def _detect_image_type(data: bytes) -> Optional[str]:
    """바이트 시그니처로 JPEG/PNG/WebP 여부를 판별한다(디코딩하지 않는다).

    업로드의 Content-Type 헤더는 클라이언트(백엔드는 파일 확장자)가 정하는 값이라
    실제 내용과 무관할 수 있다. 이 서비스는 이미지를 직접 파싱하지 않고 외부 모델에
    그대로 넘기므로, 신뢰할 수 없는 입력을 파싱하는 공격면을 늘리지 않도록 헤더 구조만
    확인한다. 어느 형식도 아니면 None.
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR":
        return "image/png"
    if (
        data[:4] == b"RIFF"
        and data[8:12] == b"WEBP"
        and data[12:16] in (b"VP8 ", b"VP8L", b"VP8X")
    ):
        return "image/webp"
    return None


def _describe_upstream_error(exc: BaseException) -> str:
    """로그용 요약. 예외 종류와 HTTP 상태만 남긴다.

    외부 제공자의 응답 본문이나 모델 출력(OCR 결과 등)은 예외 메시지에 섞여 들어올 수
    있으므로(예: pydantic ValidationError 의 input_value) 로그에 남기지 않는다.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{type(exc).__name__} status={exc.response.status_code}"
    return type(exc).__name__


def _build_data_url(contents: bytes, content_type: str) -> str:
    base64_image = base64.b64encode(contents).decode("utf-8")
    return f"data:{content_type};base64,{base64_image}"


def _clean_model_json_text(response_text: str) -> str:
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        return response_text[7:-3].strip()
    if response_text.startswith("```"):
        return response_text[3:-3].strip()
    return response_text


def _parse_model_json_response(response_text: str, response_model: Type[T]) -> T:
    data = json.loads(_clean_model_json_text(response_text))
    return response_model(**data)


def _request_gemini_vision_json(
    *,
    prompt: str,
    image_bytes: bytes,
    content_type: str,
    max_output_tokens: int,
    response_model: Type[T],
) -> T:
    image_part = genai_types.Part.from_bytes(
        data=image_bytes,
        mime_type=content_type,
    )
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        max_output_tokens=max_output_tokens,
    )
    with genai.Client(api_key=settings.gemini_api_key) as client:
        response = client.models.generate_content(
            model=settings.vision_model or "gemini-2.0-flash",
            contents=[prompt, image_part],
            config=config,
        )
    response_text = getattr(response, "text", None)
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError("Gemini vision response content is empty")
    return _parse_model_json_response(response_text, response_model)


def _resolve_vision_model_candidates() -> list[str]:
    configured_fallbacks = getattr(settings, "vision_fallback_models", [])
    if not isinstance(configured_fallbacks, (list, tuple)):
        configured_fallbacks = []

    candidates: list[str] = []
    for model in [settings.vision_model, *configured_fallbacks]:
        model_id = (model or "").strip()
        if model_id and model_id not in candidates:
            candidates.append(model_id)
    return candidates


async def _request_openrouter_vision_json(
    *,
    prompt: str,
    image_url: str,
    max_tokens: int,
    app_title: str,
    response_model: Type[T],
) -> T:
    if not settings.openrouter_api_key:
        raise RuntimeError("OpenRouter API key not configured")

    candidates = _resolve_vision_model_candidates()
    if not candidates:
        raise RuntimeError("Vision model is not configured")

    client = get_shared_httpx_client(
        "openrouter",
        timeout=httpx.Timeout(120.0, connect=10.0, read=60.0, pool=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    endpoint = (
        f"{settings.openrouter_base_url.rstrip('/')}{OPENROUTER_CHAT_COMPLETIONS_PATH}"
    )
    last_error: Exception | None = None

    for index, model_id in enumerate(candidates):
        try:
            response = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": settings.openrouter_referer
                    or "https://kbo-platform.com",
                    "X-Title": settings.openrouter_app_title or app_title,
                },
                json={
                    "model": model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url},
                                },
                            ],
                        }
                    ],
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            if not isinstance(response_text, str) or not response_text.strip():
                raise RuntimeError("OpenRouter vision response content is empty")
            parsed = _parse_model_json_response(response_text, response_model)
            if index > 0:
                logger.info("OpenRouter vision fallback succeeded: model=%s", model_id)
            return parsed
        except (
            httpx.HTTPError,
            json.JSONDecodeError,
            ValidationError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            last_error = exc
            if index >= len(candidates) - 1:
                break
            logger.warning(
                "OpenRouter vision model failed; trying fallback: model=%s next_model=%s error=%s",
                model_id,
                candidates[index + 1],
                _describe_upstream_error(exc),
            )

    raise last_error or RuntimeError("All OpenRouter vision models failed")


async def _read_ticket_image_with_limit(
    file: UploadFile,
    max_bytes: int,
    allowed_content_types: Optional[set[str]] = None,
) -> tuple[bytes, str]:
    content_type = _normalize_content_type(file.content_type)
    if allowed_content_types is not None and content_type not in allowed_content_types:
        raise HTTPException(
            status_code=415, detail="지원되지 않는 이미지 파일 타입입니다."
        )

    with tempfile.SpooledTemporaryFile(max_size=max_bytes, mode="w+b") as spool:
        total = 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413, detail="이미지 파일 크기가 너무 큽니다."
                )
            spool.write(chunk)

        if total == 0:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        spool.seek(0)
        contents = spool.read()

    if allowed_content_types is None:
        return contents, content_type or "application/octet-stream"

    # 헤더가 허용 형식이어도 실제 바이트가 이미지가 아니면 외부 모델로 보내지 않는다.
    # 이후 모델 호출에는 선언된 값이 아니라 판별된 형식을 쓴다.
    detected_type = _detect_image_type(contents)
    if detected_type is None or detected_type not in allowed_content_types:
        raise HTTPException(
            status_code=415, detail="지원되지 않는 이미지 파일 타입입니다."
        )
    return contents, detected_type


class TicketInfo(BaseModel):
    date: Optional[str] = None
    time: Optional[str] = None
    stadium: Optional[str] = None
    homeTeam: Optional[str] = None
    awayTeam: Optional[str] = None
    section: Optional[str] = None
    row: Optional[str] = None
    seat: Optional[str] = None
    peopleCount: Optional[int] = None
    price: Optional[int] = None
    reservationNumber: Optional[str] = None

    @field_validator(
        "date",
        "time",
        "stadium",
        "homeTeam",
        "awayTeam",
        "section",
        "row",
        "seat",
        "reservationNumber",
        mode="before",
    )
    @classmethod
    def _coerce_optional_string(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("peopleCount", "price", mode="before")
    @classmethod
    def _coerce_optional_int(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)

        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if not digits:
            return None
        return int(digits)


class SeatViewClassification(BaseModel):
    label: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None


@router.post("/ticket", response_model=TicketInfo)
async def analyze_ticket_image(
    file: UploadFile = File(...),
    _: None = Depends(rate_limit_vision_dependency),
    __: None = Depends(require_ai_internal_token),
):
    """
    Analyzes an uploaded ticket image using Gemini Vision (Native or OpenRouter) to extract details.
    """
    try:
        contents, content_type = await _read_ticket_image_with_limit(
            file,
            MAX_TICKET_IMAGE_BYTES,
            allowed_content_types=ALLOWED_TICKET_IMAGE_TYPES,
        )

        prompt = """
        Analyze this KBO(Korean Baseball Organization) ticket image and extract the following information in JSON format:
        - date (YYYY-MM-DD format)
        - time (HH:MM format)
        - stadium (Korean name, e.g., 잠실야구장, 사직야구장)
        - homeTeam (Korean name)
        - awayTeam (Korean name)
        - section (Block/Zone name)
        - row
        - seat (Seat number)
        - peopleCount (Number of people/tickets in this booking, e.g., '2매' -> 2)
        - price (Price per ticket/person, integer value)
        - reservationNumber (Booking/Reservation number, usually a long string of numbers)

        If any field is missing or illegible, set it to null.
        Return ONLY the JSON object, no markdown formatting.
        """

        if settings.llm_provider == "gemini":
            # Native Google Gemini Implementation
            if not settings.gemini_api_key:
                raise HTTPException(
                    status_code=500, detail="Gemini API Key not configured"
                )

            return await run_in_threadpool(
                _request_gemini_vision_json,
                prompt=prompt,
                image_bytes=contents,
                content_type=content_type,
                max_output_tokens=1000,
                response_model=TicketInfo,
            )

        else:
            # OpenRouter Implementation
            return await _request_openrouter_vision_json(
                prompt=prompt,
                image_url=_build_data_url(contents, content_type),
                max_tokens=1000,
                app_title="KBO Platform Ticket OCR",
                response_model=TicketInfo,
            )

    except HTTPException as exc:
        logger.warning(
            "Ticket image request rejected: status=%s detail=%s",
            exc.status_code,
            exc.detail,
        )
        raise
    except httpx.HTTPStatusError as e:
        logger.error("OpenRouter API error: %s", _describe_upstream_error(e))
        raise HTTPException(
            status_code=500, detail=f"OpenRouter API error: {e.response.status_code}"
        )
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(
            "Vision model returned an unusable response: %s",
            _describe_upstream_error(e),
        )
        raise HTTPException(status_code=500, detail="Failed to analyze ticket image")
    except Exception:
        logger.exception("Error processing ticket image")
        raise HTTPException(status_code=500, detail="Failed to analyze ticket image")


@router.post("/seat-view-classify", response_model=SeatViewClassification)
async def classify_seat_view_image(
    file: UploadFile = File(...),
    _: None = Depends(rate_limit_vision_dependency),
    __: None = Depends(require_ai_internal_token),
):
    """Classify an uploaded baseball-related image for seat-view moderation."""
    try:
        contents, content_type = await _read_ticket_image_with_limit(
            file,
            MAX_TICKET_IMAGE_BYTES,
            allowed_content_types=ALLOWED_TICKET_IMAGE_TYPES,
        )

        prompt = """
        Analyze this uploaded image for a KBO baseball app.
        Classify it into exactly one label:
        - SEAT_VIEW: a real in-stadium field/seat perspective photo from the audience area
        - TICKET: ticket, reservation, QR/barcode, receipt, or screenshot of ticket info
        - OTHER: selfie, food, mascot, concourse, unrelated baseball photo, or any non-seat-view image
        - INAPPROPRIATE: explicit, violent, hateful, unsafe, or clearly policy-violating content

        Return JSON only in this shape:
        {
          "label": "SEAT_VIEW|TICKET|OTHER|INAPPROPRIATE",
          "confidence": 0.0,
          "reason": "short Korean sentence"
        }

        Rules:
        - If the image is blurry, obstructed, or uncertain, prefer OTHER unless it is clearly a ticket.
        - Only use SEAT_VIEW when the camera viewpoint is plausibly from a spectator seat looking toward the field or stands.
        - confidence must be between 0 and 1.
        """

        if settings.llm_provider == "gemini":
            if not settings.gemini_api_key:
                raise HTTPException(
                    status_code=500, detail="Gemini API Key not configured"
                )

            return await run_in_threadpool(
                _request_gemini_vision_json,
                prompt=prompt,
                image_bytes=contents,
                content_type=content_type,
                max_output_tokens=500,
                response_model=SeatViewClassification,
            )
        else:
            return await _request_openrouter_vision_json(
                prompt=prompt,
                image_url=_build_data_url(contents, content_type),
                max_tokens=500,
                app_title="KBO Platform Seat View Classification",
                response_model=SeatViewClassification,
            )

    except HTTPException as exc:
        logger.warning(
            "Seat-view classification request rejected: status=%s detail=%s",
            exc.status_code,
            exc.detail,
        )
        raise
    except httpx.HTTPStatusError as e:
        logger.error("OpenRouter API error: %s", _describe_upstream_error(e))
        raise HTTPException(
            status_code=500, detail=f"OpenRouter API error: {e.response.status_code}"
        )
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(
            "Vision model returned an unusable response: %s",
            _describe_upstream_error(e),
        )
        raise HTTPException(
            status_code=500, detail="Failed to classify seat-view image"
        )
    except Exception:
        logger.exception("Error classifying seat-view image")
        raise HTTPException(
            status_code=500, detail="Failed to classify seat-view image"
        )
