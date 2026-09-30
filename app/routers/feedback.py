"""Answer feedback intake (internal, BFF → AI).

Authenticated with the shared internal token like every other router. The BFF
forwards a thumbs up/down plus the ``request_id`` / cache-key prefix / fingerprint
it received in the answer meta, so bad answers can be traced to a prompt,
retrieval, or source version and mined into golden-dataset candidates.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..deps import get_connection_pool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai/chat", tags=["chat"])


class FeedbackRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    rating: Literal["UP", "DOWN"]
    request_id: Optional[str] = Field(default=None, max_length=64)
    answer_key: Optional[str] = Field(default=None, max_length=64)
    reason: Optional[str] = Field(default=None, max_length=500)
    corrected_fact: Optional[str] = Field(default=None, max_length=1000)
    fingerprint: Optional[Dict[str, Any]] = None


class FeedbackResponse(BaseModel):
    feedback_id: int


async def store_feedback(pool: Any, payload: FeedbackRequest) -> int:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            INSERT INTO rag_answer_feedback
                (request_id, answer_key, question, rating, reason,
                 corrected_fact, fingerprint)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            RETURNING feedback_id
            """,
            (
                payload.request_id,
                payload.answer_key,
                payload.question,
                payload.rating,
                payload.reason,
                payload.corrected_fact,
                (
                    json.dumps(payload.fingerprint, ensure_ascii=False)
                    if payload.fingerprint
                    else None
                ),
            ),
        )
        row = await cur.fetchone()
    return int(row[0])


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    try:
        feedback_id = await store_feedback(get_connection_pool(), payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Feedback] store failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="feedback_unavailable") from exc
    return FeedbackResponse(feedback_id=feedback_id)
