"""SSE(Server-Sent Events)를 지원하는 채팅 스트리밍 엔드포인트를 정의합니다.

이 라우터는 사용자의 질문을 받아 RAG 파이프라인을 실행하고,
그 결과를 실시간으로 클라이언트에게 스트리밍하는 API를 제공합니다.
"""

from __future__ import annotations

import base64
import json
import logging
import openai
import os
from typing import Any, Dict, List, Optional

import re

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..core.rag import RAGPipeline
from ..core.ratelimit import rate_limit_dependency
from ..deps import get_intent_router, get_rag_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

MAX_HISTORY_MESSAGES = 8  # user/assistant 메시지 합산 기준


def _infer_filters_from_question(
    question: str,
    base_filters: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    질문에 포함된 연도 표현을 분석해 season_year 필터를 자동으로 지정합니다.
    base_filters에 season_year가 이미 있으면 그대로 반환합니다.
    """
    if base_filters and base_filters.get("season_year"):
        return base_filters

    match = re.search(r"(\d{2,4})\s*년", question)
    token: Optional[str] = None
    if match:
        token = match.group(1)
    else:
        match = re.search(r"(19|20)\d{2}", question)
        if match:
            token = match.group(0)

    year: Optional[int] = None
    if token:
        if len(token) == 4:
            year = int(token)
        elif len(token) == 2:
            short = int(token)
            year = 1900 + short if short >= 82 else 2000 + short

    if year and 1900 <= year <= 2100:
        merged = dict(base_filters) if base_filters else {}
        merged["season_year"] = year
        return merged

    return base_filters


def _decode_history_payload(payload: Any) -> Optional[List[Dict[str, str]]]:
    """클라이언트에서 전달된 history 데이터를 정규화합니다."""
    if not payload:
        return None

    items: Optional[List[Dict[str, Any]]] = None

    if isinstance(payload, list):
        items = payload  # 이미 파싱된 리스트
    elif isinstance(payload, str):
        try:
            decoded = base64.b64decode(payload).decode("utf-8")
            items = json.loads(decoded)
        except Exception:  # noqa: BLE001
            logger.warning("대화 history 디코딩에 실패했습니다.")
            return None

    if not items:
        return None

    normalized: List[Dict[str, str]] = []
    for item in items[-MAX_HISTORY_MESSAGES:]:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        text = content.strip()
        if not text:
            continue
        normalized.append({"role": role, "content": text})

    return normalized or None


async def _render_answer(result: Dict[str, Any], style: str) -> str:
    """RAG 파이프라인 결과를 지정된 스타일에 맞게 렌더링합니다."""
    if style == "json":
        return json.dumps(result, ensure_ascii=False)
    if style == "compact":
        answer = result.get("answer", "").replace("\n", " ").strip()
        return answer
    # 기본값은 markdown 또는 plain text 형식입니다.
    return result.get("answer", "")


async def _stream_response(
    request: Request,
    question: str,
    *,
    filters: Optional[Dict[str, Any]],
    style: str,
    history: Optional[List[Dict[str, str]]],
    pipeline: RAGPipeline,
    intent_router,
):
    """질문에 대한 답변을 생성하고 SSE 스트림으로 반환하는 핵심 로직입니다."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    # 사용자의 질문 의도를 파악합니다.
    intent = intent_router(question)
    result: Optional[Dict[str, Any]] = None
    error_payload: Optional[Dict[str, Any]] = None
    try:
        filters = _infer_filters_from_question(question, filters)
        # RAG 파이프라인을 실행하여 결과를 생성합니다.
        result = await pipeline.run(
            question,
            intent=intent,
            filters=filters,
            history=history,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat_stream에서 오류가 발생했습니다.")
        error_payload = {"message": "internal_error", "detail": str(exc)}

    async def event_generator():
        """SSE 이벤트 스트림을 생성하는 비동기 제너레이터입니다."""
        # 1. 의도(intent) 이벤트 전송
        yield {"event": "intent", "data": json.dumps({"intent": intent}, ensure_ascii=False)}
        
        # 2. 오류 발생 시 오류 이벤트 전송
        if error_payload:
            yield {"event": "error", "data": json.dumps(error_payload, ensure_ascii=False)}
        # 3. 성공 시 메시지와 메타데이터 이벤트 전송
        elif result:
            rendered = await _render_answer(result, style)
            # 답변의 일부(delta)를 message 이벤트로 전송
            yield {"event": "message", "data": json.dumps({"delta": rendered}, ensure_ascii=False)}
            
            # 인용(citations) 등 추가 정보를 meta 이벤트로 전송
            meta_payload = {
                "citations": result.get("citations", []),
                "style": style,
            }
            yield {"event": "meta", "data": json.dumps(meta_payload, ensure_ascii=False)}
            
        # 4. 스트림 종료를 알리는 done 이벤트 전송
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx 등 프록시에서 버퍼링 방지
        },
        ping=15,  # 15초마다 ping을 보내 연결 유지
    )


class ChatPayload(Dict[str, Any]):
    """채팅 요청 시 POST body의 스키마 정의."""
    question: str
    filters: Optional[Dict[str, Any]] = None
    style: Optional[str] = None


@router.post("/completion")
async def chat_completion(
    payload: Dict[str, Any] = Body(...),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    intent_router=Depends(get_intent_router),
    _: None = Depends(rate_limit_dependency),  # 요청 빈도 제한 적용
):
    """단일 JSON 응답으로 전체 채팅 답변을 반환하는 엔드포인트입니다."""
    question = payload.get("question", "")
    if not question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    filters = _infer_filters_from_question(question, payload.get("filters"))
    history = _decode_history_payload(payload.get("history"))
    intent = intent_router(question)
    result = await pipeline.run(
        question,
        intent=intent,
        filters=filters,
        history=history,
    )
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
    """POST 요청을 통해 질문을 받고, 답변을 SSE 스트림으로 반환합니다."""
    question = payload.get("question", "")
    filters = payload.get("filters")
    history = _decode_history_payload(payload.get("history"))
    
    # payload에 style이 지정된 경우, 쿼리 파라미터보다 우선 적용합니다.
    style_override = payload.get("style")
    if style_override in {"markdown", "json", "compact"}:
        style = style_override

    return await _stream_response(
        request,
        question,
        filters=filters,
        style=style,
        history=history,
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
    """GET 요청을 통해 질문을 받고, 답변을 SSE 스트림으로 반환합니다."""
    history_param = None
    if request is not None:
        history_param = request.query_params.get("history")

    history = _decode_history_payload(history_param)

    return await _stream_response(
        request,
        q,
        filters=None,
        style=style,
        history=history,
        pipeline=pipeline,
        intent_router=intent_router,
    )

openai.api_key = os.getenv("OPENAI_API_KEY")

@router.post("/voice")
async def transcribe_audio(
    file: UploadFile = File(...),
    _: None = Depends(rate_limit_dependency),
    ):
    try :
        contents = await file.read()
        import io
        audio_file = io.BytesIO(contents)
        audio_file.name = "audio.webm"

        response = await openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko"
        )
        return {"text": response.text}
    except Exception as e:
        logger.exception("음성 변환 중 오류 발생")
        raise HTTPException(status_code=500, detail=str(e))


