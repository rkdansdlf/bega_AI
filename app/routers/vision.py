from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import Optional
from app.config import get_settings
import httpx
import base64
import json
import google.generativeai as genai
from PIL import Image
import io

router = APIRouter(prefix="/vision", tags=["vision"])
settings = get_settings()


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
async def analyze_ticket_image(file: UploadFile = File(...)):
    """
    Analyzes an uploaded ticket image using Gemini Vision (Native or OpenRouter) to extract details.
    """
    try:
        # Read image content
        contents = await file.read()
        
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
                raise HTTPException(status_code=500, detail="Gemini API Key not configured")

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
            content_type = file.content_type or "image/jpeg"

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

    except httpx.HTTPStatusError as e:
        print(f"OpenRouter API error: {e.response.text}")
        raise HTTPException(
            status_code=500, detail=f"OpenRouter API error: {e.response.status_code}"
        )
    except Exception as e:
        print(f"Error processing ticket image: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze ticket image: {str(e)}"
        )
