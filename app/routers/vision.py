from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional
import logging

from ..config import get_settings
from ..core.ratelimit import rate_limit_vision_dependency
from ..deps import require_ai_internal_token
import httpx
import base64
import json
import google.generativeai as genai
from PIL import Image
import io
import tempfile


router = APIRouter(prefix="/vision", tags=["vision"])
settings = get_settings()
logger = logging.getLogger(__name__)
MAX_TICKET_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_TICKET_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _normalize_content_type(content_type: Optional[str]) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


async def _read_ticket_image_with_limit(
    file: UploadFile,
    max_bytes: int,
    allowed_content_types: Optional[set[str]] = None,
) -> tuple[bytes, str]:
    content_type = _normalize_content_type(file.content_type)
    if allowed_content_types is not None and content_type not in allowed_content_types:
        raise HTTPException(status_code=415, detail="지원되지 않는 이미지 파일 타입입니다.")

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
        return spool.read(), content_type or "application/octet-stream"


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

            genai.configure(api_key=settings.gemini_api_key)
            model = genai.GenerativeModel(settings.vision_model or "gemini-2.0-flash")

            def _load_image(image_bytes: bytes):
                with Image.open(io.BytesIO(image_bytes)) as pil_image:
                    return pil_image.copy()

            image = await run_in_threadpool(_load_image, contents)
            response = await run_in_threadpool(model.generate_content, [prompt, image])
            response_text = response.text.strip()

        else:
            # OpenRouter Implementation
            base64_image = base64.b64encode(contents).decode("utf-8")

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://kbo-platform.com",
                        "X-Title": "KBO Platform Ticket OCR",
                    },
                    json={
                        "model": settings.vision_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{content_type};base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        "max_tokens": 1000,
                    },
                )

                response.raise_for_status()
                result = response.json()
                response_text = result["choices"][0]["message"]["content"].strip()

        # Clean up response text (remove markdown code blocks if present)
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]

        data = json.loads(response_text)
        return TicketInfo(**data)

    except HTTPException as exc:
        logger.warning(
            "Ticket image request rejected: status=%s detail=%s",
            exc.status_code,
            exc.detail,
        )
        raise
    except httpx.HTTPStatusError as e:
        logger.error("OpenRouter API error: %s", e.response.text)
        raise HTTPException(
            status_code=500, detail=f"OpenRouter API error: {e.response.status_code}"
        )
    except Exception:
        logger.exception("Error processing ticket image")
        raise HTTPException(
            status_code=500, detail="Failed to analyze ticket image"
        )
