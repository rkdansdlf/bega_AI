"""Chat streaming endpoints with SSE support."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..core.rag import RAGPipeline
from ..core.ratelimit import rate_limit_dependency
from ..deps import get_intent_router, get_rag_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


async def _render_answer(result: Dict[str, Any], style: str) -> str:
    if style == "json":
        return json.dumps(result, ensure_ascii=False)
    if style == "compact":
        answer = result.get("answer", "").replace("\n", " ").strip()
        return answer
    # default markdown/plain
    return result.get("answer", "")


async def _stream_response(
    request: Request,
    question: str,
    *,
    filters: Optional[Dict[str, Any]],
    style: str,
    pipeline: RAGPipeline,
    intent_router,
):
    if not question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    intent = intent_router(question)
    result: Optional[Dict[str, Any]] = None
    error_payload: Optional[Dict[str, Any]] = None
    try:
        result = await pipeline.run(question, intent=intent, filters=filters)
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat_stream encountered an error")
        error_payload = {"message": "internal_error", "detail": str(exc)}

    async def event_generator():
        yield {"event": "intent", "data": json.dumps({"intent": intent}, ensure_ascii=False)}
        if error_payload:
            yield {"event": "error", "data": json.dumps(error_payload, ensure_ascii=False)}
        elif result:
            rendered = await _render_answer(result, style)
            yield {"event": "message", "data": json.dumps({"delta": rendered}, ensure_ascii=False)}
            meta_payload = {
                "citations": result.get("citations", []),
                "style": style,
            }
            yield {"event": "meta", "data": json.dumps(meta_payload, ensure_ascii=False)}
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
        ping=15,
    )


class ChatPayload(Dict[str, Any]):
    question: str
    filters: Optional[Dict[str, Any]] = None
    style: Optional[str] = None


@router.post("/completion")
async def chat_completion(
    payload: Dict[str, Any] = Body(...),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    intent_router=Depends(get_intent_router),
    _: None = Depends(rate_limit_dependency),
):
    question = payload.get("question", "")
    if not question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    filters = payload.get("filters")
    intent = intent_router(question)
    result = await pipeline.run(question, intent=intent, filters=filters)
    return JSONResponse(result)


@router.post("/stream")
async def chat_stream_post(
    payload: Dict[str, Any] = Body(...),
    style: str = Query("markdown", regex="^(markdown|json|compact)$"),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    intent_router=Depends(get_intent_router),
    _: None = Depends(rate_limit_dependency),
    request: Request = None,
):
    question = payload.get("question", "")
    filters = payload.get("filters")
    style_override = payload.get("style")
    if style_override in {"markdown", "json", "compact"}:
        style = style_override

    return await _stream_response(
        request,
        question,
        filters=filters,
        style=style,
        pipeline=pipeline,
        intent_router=intent_router,
    )


@router.get("/stream")
async def chat_stream_get(
    q: str = Query("", description="질문 텍스트"),
    style: str = Query("markdown", regex="^(markdown|json|compact)$"),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    intent_router=Depends(get_intent_router),
    _: None = Depends(rate_limit_dependency),
    request: Request = None,
):
    return await _stream_response(
        request,
        q,
        filters=None,
        style=style,
        pipeline=pipeline,
        intent_router=intent_router,
    )
