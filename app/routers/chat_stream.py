"""SSE(Server-Sent Events)를 지원하는 채팅 스트리밍 엔드포인트를 정의합니다.

이 라우터는 사용자의 질문을 받아 RAG 파이프라인을 실행하고,
그 결과를 실시간으로 클라이언트에게 스트리밍하는 API를 제공합니다.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import openai
import os
import secrets
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

import re

from fastapi import (
    APIRouter,
    Body,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    File,
)
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..config import get_settings
from ..deps import get_agent, get_connection_pool
from ..agents.baseball_agent import BaseballStatisticsAgent
from ..core.ratelimit import rate_limit_dependency
from ..core.chat_cache_key import build_chat_cache_key, has_temporal_keyword
from ..core.chat_cache import (
    get_cached_response,
    save_to_cache,
    update_hit_count,
    get_stats,
    delete_by_intent,
    delete_by_key,
)

logger = logging.getLogger(__name__)
load_dotenv()
router = APIRouter(prefix="/ai/chat", tags=["chat"])

MAX_HISTORY_MESSAGES = 8  # user/assistant 메시지 합산 기준

# 캐시 스키마 버전. 프롬프트 또는 정규화 방식 변경 시 올리면
# 기존 캐시가 자동으로 미스 처리됩니다.
CHAT_CACHE_SCHEMA_VERSION = "v1"


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


async def _async_update_hit_count(cache_key: str) -> None:
    """백그라운드에서 hit_count를 업데이트합니다 (응답 지연 없음)."""
    try:
        pool = get_connection_pool()
        with pool.connection() as conn:
            await update_hit_count(conn, cache_key)
    except Exception as exc:
        logger.warning("[ChatCache] hit_count background update failed: %s", exc)


def _make_cached_sse_response(cached: dict, style: str, cache_key: str) -> EventSourceResponse:
    """캐시된 응답을 SSE 형식으로 재스트리밍합니다.

    프론트엔드가 실제 스트리밍과 동일한 이벤트 시퀀스를 받을 수 있도록
    status → message(청크) → meta → done 순서로 이벤트를 생성합니다.
    """

    async def cached_generator():
        response_text = cached["response_text"]

        # status 이벤트: 캐시 히트 표시 (번개 이모지로 빠른 응답임을 암시)
        yield {
            "event": "status",
            "data": json.dumps({"message": "⚡"}, ensure_ascii=False),
        }

        # message 이벤트: 200자 청크로 나눠 전송 (프론트엔드 타이핑 효과 유지)
        chunk_size = 200
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i : i + chunk_size]
            yield {
                "event": "message",
                "data": json.dumps({"delta": chunk}, ensure_ascii=False),
            }

        # meta 이벤트 (cached: True 포함)
        yield {
            "event": "meta",
            "data": json.dumps(
                {
                    "tool_calls": [],
                    "tool_results": [],
                    "data_sources": [],
                    "verified": True,
                    "visualizations": [],
                    "style": style,
                    "cached": True,
                    "intent": cached.get("intent"),
                    "cache_key_prefix": cache_key[:8],
                },
                ensure_ascii=False,
            ),
        }
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(
        cached_generator(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Cache": "HIT",
        },
        ping=15,
    )


async def _stream_response(
    request: Request,
    question: str,
    *,
    filters: Optional[Dict[str, Any]],
    style: str,
    history: Optional[List[Dict[str, str]]],
    agent: BaseballStatisticsAgent,
    cache_key: Optional[str] = None,
):
    """질문에 대한 답변을 생성하고 SSE 스트림으로 반환하는 핵심 로직입니다.

    cache_key가 전달된 경우, 스트리밍 완료 후 응답 텍스트를 DB 캐시에 저장합니다.
    캐싱 조건(history-free & 실시간 키워드 없음)은 호출자(chat_stream_post)에서 판단합니다.
    """
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
            yield {
                "event": "error",
                "data": json.dumps(error_payload, ensure_ascii=False),
            }
        # 2. 성공 시 메시지와 메타데이터 이벤트 전송
        elif result:
            answer = result.get("answer")

            # answer가 비동기 제너레이터인 경우 (스트리밍)
            # 캐시 저장을 위해 전체 텍스트를 누적합니다.
            full_response_text = ""

            if hasattr(answer, "__aiter__"):
                async for delta in answer:
                    if delta:
                        full_response_text += delta
                    yield {
                        "event": "message",
                        "data": json.dumps({"delta": delta}, ensure_ascii=False),
                    }

            # answer가 일반 문자열인 경우 (비스트리밍/일상대화)
            else:
                rendered = await _render_answer(result, style)
                full_response_text = rendered
                yield {
                    "event": "message",
                    "data": json.dumps({"delta": rendered}, ensure_ascii=False),
                }

            def safe_serialize(obj):
                """JSON 직렬화 가능한 형태로 객체를 변환"""
                from datetime import datetime, date

                if obj is None:
                    return None
                elif isinstance(obj, (str, int, float, bool)):
                    return obj
                elif isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                elif hasattr(obj, "to_dict"):
                    return safe_serialize(obj.to_dict())
                elif isinstance(obj, dict):
                    return {key: safe_serialize(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [safe_serialize(item) for item in obj]
                else:
                    # ToolResult 등의 객체
                    if hasattr(obj, "__dict__"):
                        return {
                            key: safe_serialize(value)
                            for key, value in obj.__dict__.items()
                        }
                    else:
                        return str(obj)

            # 도구 호출 등 추가 정보를 meta 이벤트로 전송
            tool_results_raw = result.get("tool_results", [])
            tool_results_serialized = safe_serialize(tool_results_raw)

            intent = result.get("intent")

            meta_payload_raw = {
                "tool_calls": [tc.to_dict() for tc in result.get("tool_calls", [])],
                "tool_results": tool_results_serialized,
                "data_sources": result.get("data_sources", []),
                "verified": result.get("verified", False),
                "visualizations": result.get(
                    "visualizations", []
                ),  # 시각화 데이터 전달
                "style": style,
                "cached": False,
                "intent": intent,
            }
            # 전체 payload를 안전하게 직렬화
            meta_payload = safe_serialize(meta_payload_raw)
            yield {
                "event": "meta",
                "data": json.dumps(meta_payload, ensure_ascii=False),
            }

            # 캐시 저장: 에러 없이 완료되고 캐시 키가 있을 때만 저장
            if cache_key and full_response_text:
                from ..config import get_settings as _get_settings
                _settings = _get_settings()
                model_name = (
                    getattr(_settings, "coach_openrouter_model", None)
                    or getattr(_settings, "openrouter_model", None)
                    or getattr(_settings, "gemini_model", None)
                    or "unknown"
                )
                try:
                    pool = get_connection_pool()
                    with pool.connection() as conn:
                        await save_to_cache(
                            conn,
                            cache_key=cache_key,
                            question_text=question,
                            filters_json=filters,
                            intent=intent,
                            response_text=full_response_text,
                            model_name=model_name,
                        )
                    logger.info(
                        "[ChatCache] SAVED key=%s... intent=%s",
                        cache_key[:8],
                        intent,
                    )
                except Exception as exc:
                    logger.warning("[ChatCache] save failed: %s", exc)

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

    # If the answer is an async generator, consume it for non-streaming response.
    import inspect

    answer_obj = result.get("answer")
    if inspect.isasyncgen(answer_obj) or hasattr(answer_obj, "__aiter__"):
        full_answer = ""
        try:
            async for chunk in answer_obj:
                if chunk:
                    full_answer += chunk
            result["answer"] = full_answer
        except Exception as e:
            logger.error(f"Error consuming generator: {e}")
            result["answer"] = str(answer_obj)

    # ToolCall 등 커스텀 객체 직렬화 헬퍼
    def safe_serialize(obj):
        """JSON 직렬화 가능한 형태로 객체를 변환"""
        from datetime import datetime, date

        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif hasattr(obj, "to_dict"):
            return safe_serialize(obj.to_dict())
        elif isinstance(obj, dict):
            return {key: safe_serialize(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [safe_serialize(item) for item in obj]
        else:
            if hasattr(obj, "__dict__"):
                return {
                    key: safe_serialize(value) for key, value in obj.__dict__.items()
                }
            return str(obj)

    if isinstance(result, dict):
        return JSONResponse(safe_serialize(result))
    else:
        # result가 객체라면 dict로 변환
        return JSONResponse(
            safe_serialize(
                {
                    "answer": getattr(result, "answer", str(result)),
                    "citations": getattr(result, "citations", []),
                    "intent": getattr(result, "intent", "unknown"),
                }
            )
        )


@router.post("/stream")
async def chat_stream_post(
    payload: Dict[str, Any] = Body(...),
    style: str = Query("markdown", pattern="^(markdown|json|compact)$"),
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

    # 캐시 적용 조건: history-free 쿼리이고 실시간 키워드 없음
    # history가 있으면 대화 맥락이 있으므로 캐싱 불가
    # 실시간 키워드("오늘", "지금" 등)가 있으면 최신성이 중요하므로 캐싱 불가
    cacheable = (history is None) and (not has_temporal_keyword(question))
    cache_key: Optional[str] = None

    if cacheable:
        cache_key, _ = build_chat_cache_key(
            question=question,
            filters=filters,
            schema_version=CHAT_CACHE_SCHEMA_VERSION,
        )
        pool = get_connection_pool()
        with pool.connection() as conn:
            cached = await get_cached_response(conn, cache_key)

        if cached:
            logger.info(
                "[ChatCache] HIT key=%s... hit_count=%d",
                cache_key[:8],
                cached["hit_count"],
            )
            # hit_count는 background에서 업데이트 (응답 지연 없음)
            asyncio.create_task(_async_update_hit_count(cache_key))
            return _make_cached_sse_response(cached, style, cache_key)

    return await _stream_response(
        request,
        question,
        filters=filters,
        style=style,
        history=history,
        agent=agent,
        cache_key=cache_key,  # None이면 _stream_response 내에서 캐시 저장 건너뜀
    )


@router.get("/stream")
async def chat_stream_get(
    q: str = Query("", description="질문 텍스트"),
    style: str = Query("markdown", pattern="^(markdown|json|compact)$"),
    agent: BaseballStatisticsAgent = Depends(get_agent),
    _: None = Depends(rate_limit_dependency),
    request: Request = None,
):
    """GET 요청을 통해 질문을 받고, 답변을 SSE 스트림으로 반환합니다."""
    history_param = None
    if request is not None:
        history_param = request.query_params.get("history")

    history = _decode_history_payload(history_param)

    # GET 엔드포인트: filters 없음, cache_key 없음 (캐싱 미적용)
    # GET은 브라우저 테스트/디버깅 용도이므로 캐싱 복잡도를 추가하지 않음
    return await _stream_response(
        request,
        q,
        filters=None,
        style=style,
        history=history,
        agent=agent,
        cache_key=None,
    )


_whisper_client: openai.AsyncOpenAI | None = None


def _get_whisper_client() -> openai.AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY2") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503, detail="OPENAI_API_KEY가 설정되지 않았습니다."
        )
    global _whisper_client
    if _whisper_client is None:
        _whisper_client = openai.AsyncOpenAI(api_key=api_key)
    return _whisper_client


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

        whisper_client = _get_whisper_client()
        # 비동기 호출로 변경
        response = await whisper_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko",
            prompt="야구, KBO, 직관, 경기, 선수, 팀에 대한 질문입니다.",
        )

        logger.info(f" 변환 성공! 텍스트 길이: {len(response.text)}")
        logger.info(f"변환된 텍스트: {response.text}")

        return {"text": response.text}

    except Exception as e:
        logger.exception(f" 음성 변환 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── 캐시 관리 API ────────────────────────────────────────────────────────────
# 내부 관리용 엔드포인트입니다.
# 기본값은 비활성화(404)이며, 활성화 시 X-Cache-Admin-Token 헤더 검증이 필요합니다.


async def _require_chat_cache_admin(
    x_cache_admin_token: str = Header(default="", alias="X-Cache-Admin-Token"),
) -> None:
    """캐시 관리 API 접근 제어 dependency."""
    settings = get_settings()

    if not settings.chat_cache_admin_enabled:
        raise HTTPException(status_code=404, detail="Not Found")

    expected_token = (settings.chat_cache_admin_token or "").strip()
    if not expected_token:
        logger.error("[ChatCache] Admin API is enabled but token is not configured")
        raise HTTPException(status_code=503, detail="Chat cache admin misconfigured")

    if not secrets.compare_digest(x_cache_admin_token, expected_token):
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/cache/stats")
async def chat_cache_stats(_: None = Depends(_require_chat_cache_admin)):
    """캐시 현황 통계를 반환합니다."""
    pool = get_connection_pool()
    with pool.connection() as conn:
        stats = await get_stats(conn)
    return {"stats": stats}


@router.delete("/cache")
async def flush_cache_by_intent(
    intent: str = Query(..., description="삭제할 intent"),
    _: None = Depends(_require_chat_cache_admin),
):
    """특정 intent의 캐시 항목을 모두 삭제합니다."""
    pool = get_connection_pool()
    with pool.connection() as conn:
        deleted = await delete_by_intent(conn, intent)
    return {"deleted": deleted, "intent": intent}


@router.delete("/cache/{cache_key}")
async def invalidate_cache_entry(
    cache_key: str,
    _: None = Depends(_require_chat_cache_admin),
):
    """특정 캐시 키를 무효화합니다."""
    pool = get_connection_pool()
    with pool.connection() as conn:
        deleted = await delete_by_key(conn, cache_key)
    return {"deleted": deleted, "cache_key": cache_key}
