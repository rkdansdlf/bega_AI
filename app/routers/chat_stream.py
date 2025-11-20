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
from dotenv import load_dotenv

import re

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..deps import get_agent
from ..agents.baseball_agent import BaseballStatisticsAgent
from ..core.ratelimit import rate_limit_dependency

logger = logging.getLogger(__name__)
load_dotenv()
router = APIRouter(prefix="/chat", tags=["chat"])

MAX_HISTORY_MESSAGES = 8  # user/assistant 메시지 합산 기준


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
    """에이전트 결과를 지정된 스타일에 맞게 렌더링합니다."""
    if style == "json":
        return json.dumps(result, ensure_ascii=False, indent=2)
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
    agent: BaseballStatisticsAgent,
):
    """질문에 대한 답변을 생성하고 SSE 스트림으로 반환하는 핵심 로직입니다."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    result: Optional[Dict[str, Any]] = None
    error_payload: Optional[Dict[str, Any]] = None
    try:
        # 에이전트를 실행하여 결과를 생성합니다.
        result = await agent.process_query(
            question,
            context={"filters": filters, "history": history},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat_stream에서 오류가 발생했습니다.")
        error_payload = {"message": "internal_error", "detail": str(exc)}

    async def event_generator():
        """SSE 이벤트 스트림을 생성하는 비동기 제너레이터입니다."""
        # 1. 오류 발생 시 오류 이벤트 전송
        if error_payload:
            yield {"event": "error", "data": json.dumps(error_payload, ensure_ascii=False)}
        # 2. 성공 시 메시지와 메타데이터 이벤트 전송
        elif result:
            rendered = await _render_answer(result, style)
            # 답변의 일부(delta)를 message 이벤트로 전송
            yield {"event": "message", "data": json.dumps({"delta": rendered}, ensure_ascii=False)}
            
            def safe_serialize(obj):
                """JSON 직렬화 가능한 형태로 객체를 변환"""
                from datetime import datetime, date
                
                if obj is None:
                    return None
                elif isinstance(obj, (str, int, float, bool)):
                    return obj
                elif isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                elif hasattr(obj, 'to_dict'):
                    return safe_serialize(obj.to_dict())
                elif isinstance(obj, dict):
                    return {key: safe_serialize(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [safe_serialize(item) for item in obj]
                else:
                    # ToolResult 등의 객체
                    if hasattr(obj, '__dict__'):
                        return {key: safe_serialize(value) for key, value in obj.__dict__.items()}
                    else:
                        return str(obj)
            
            # 도구 호출 등 추가 정보를 meta 이벤트로 전송
            tool_results_raw = result.get("tool_results", [])
            tool_results_serialized = safe_serialize(tool_results_raw)
            
            meta_payload_raw = {
                "tool_calls": [tc.to_dict() for tc in result.get("tool_calls", [])],
                "tool_results": tool_results_serialized,
                "data_sources": result.get("data_sources", []),
                "verified": result.get("verified", False),
                "style": style,
            }
            # 전체 payload를 안전하게 직렬화
            meta_payload = safe_serialize(meta_payload_raw)
            yield {"event": "meta", "data": json.dumps(meta_payload, ensure_ascii=False)}
            
        # 3. 스트림 종료를 알리는 done 이벤트 전송
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
    agent: BaseballStatisticsAgent = Depends(get_agent),
    _: None = Depends(rate_limit_dependency),  # 요청 빈도 제한 적용
):
    """단일 JSON 응답으로 전체 채팅 답변을 반환하는 엔드포인트입니다."""
    question = payload.get("question", "")
    if not question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    filters = payload.get("filters")
    history = _decode_history_payload(payload.get("history"))

    result = await agent.process_query(
        question,
        context={"filters": filters, "history": history},
    )
    if isinstance(result, dict):
        return JSONResponse(result)
    else:
        # result가 객체라면 dict로 변환
        return JSONResponse({
            "answer": getattr(result, 'answer', str(result)),
            "citations": getattr(result, 'citations', []),
            "intent": intent
        })


@router.post("/stream")
async def chat_stream_post(
    payload: Dict[str, Any] = Body(...),
    style: str = Query("markdown", regex="^(markdown|json|compact)$"),
    agent: BaseballStatisticsAgent = Depends(get_agent),
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
        agent=agent,
    )


@router.get("/stream")
async def chat_stream_get(
    q: str = Query("", description="질문 텍스트"),
    style: str = Query("markdown", regex="^(markdown|json|compact)$"),
    agent: BaseballStatisticsAgent = Depends(get_agent),
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
        agent=agent,
    )

whisper_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY2"))

@router.post("/voice")
async def transcribe_audio(
    file: UploadFile = File(...),
    _: None = Depends(rate_limit_dependency),
):
    logger.info(f"===== 음성 변환 시작 =====")
    logger.info(f"파일명: {file.filename}, 타입: {file.content_type}")
    
    try:
        contents = await file.read()
        logger.info(f"파일 크기: {len(contents)} bytes")
        
        import io
        audio_file = io.BytesIO(contents)
        audio_file.name = "audio.webm"
        
        logger.info("OpenAI Whisper API 호출 중...")
        
        # 동기 호출인지 확인
        response = whisper_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko"
        )
        
        logger.info(f" 변환 성공! 텍스트 길이: {len(response.text)}")
        logger.info(f"변환된 텍스트: {response.text}")
        
        return {"text": response.text}
        
    except Exception as e:
        logger.exception(f" 음성 변환 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
