from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.config import get_settings
import httpx
import base64
import json

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


@router.post("/ticket", response_model=TicketInfo)
async def analyze_ticket_image(file: UploadFile = File(...)):
    """
    Analyzes an uploaded ticket image using Gemini Vision via OpenRouter to extract details.
    """
    try:
        # Read image content and encode to base64
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        # Determine MIME type
        content_type = file.content_type or "image/jpeg"

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

        If any field is missing or illegible, set it to null.
        Return ONLY the JSON object, no markdown formatting.
        """

        # OpenRouter API 호출
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
                    "model": "google/gemini-3-flash-preview",
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

        # Extract response text
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
