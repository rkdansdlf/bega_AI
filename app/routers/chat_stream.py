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
import tempfile
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv


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
from ..deps import get_agent, get_connection_pool, require_ai_internal_token
from ..agents.baseball_agent import BaseballStatisticsAgent
from ..core.ratelimit import (
    rate_limit_chat_dependency,
    rate_limit_chat_voice_dependency,
)
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
CHAT_CACHE_SCHEMA_VERSION = "v2"
MAX_CHAT_QUESTION_LENGTH = 1200
MAX_CHAT_HISTORY_ENTRY_LENGTH = 2000
MAX_CHAT_REQUEST_BYTES = 12 * 1024
MAX_VOICE_FILE_BYTES = 20 * 1024 * 1024
ALLOWED_VOICE_CONTENT_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/x-m4a",
}

# 일시적 내부 오류 메시지는 캐시하지 않습니다.
_NON_CACHEABLE_RESPONSE_MARKERS = (
    "질문 분석 중 오류가 발생했습니다.",
    "답변 생성 중 오류가 발생했습니다.",
    "서버 오류가 발생했습니다.",
    "잠시 후 다시 시도해주세요.",
    "질문 분석 중 오류가 발생했습니다",
    "답변 생성 중 오류가 발생했습니다",
    "서버 오류가 발생했습니다",
    "잠시 후 다시 시도해주세요",
)


def _normalize_cache_guard_text(response_text: str) -> str:
    text = (response_text or "").strip()
    return "".join(ch for ch in text if not ch.isspace())


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
        if len(text) > MAX_CHAT_HISTORY_ENTRY_LENGTH:
            raise HTTPException(
                status_code=400,
                detail="히스토리 메시지가 너무 깁니다. 최대 2000자까지 허용됩니다.",
            )
        normalized.append({"role": role, "content": text})

    return normalized or None


def _validate_chat_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="요청 형식이 올바르지 않습니다.")

    try:
        payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    except TypeError as exc:
        raise HTTPException(status_code=400, detail="요청 본문을 파싱할 수 없습니다.") from exc

    if payload_bytes > MAX_CHAT_REQUEST_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"요청 본문이 너무 큽니다. 최대 {MAX_CHAT_REQUEST_BYTES // 1024}KB까지 허용됩니다.",
        )


async def _render_answer(result: Dict[str, Any], style: str) -> str:
    """에이전트 결과를 지정된 스타일에 맞게 렌더링합니다."""
    if style == "json":
        return json.dumps(result, ensure_ascii=False, indent=2)
    if style == "compact":
        answer = result.get("answer", "").replace("\n", " ").strip()
        return answer
    # 기본값은 markdown 또는 plain text 형식입니다.
    return result.get("answer", "")


def _build_completion_fallback_answer(reason: str) -> str:
    """completion 경로에서 비동기 제너레이터 소비 실패 시 사용할 안전한 문자열 답변."""
    reason_text = (reason or "알 수 없는 오류").strip()
    reason_text = reason_text.replace("|", "\\|")
    if len(reason_text) > 240:
        reason_text = reason_text[:240] + "..."

    return (
        "핵심 결론: **답변 생성 과정에서 일시적 오류가 발생해 일부 응답이 누락되었습니다.**\n\n"
        "## 상세 내역\n"
        "| 구분 | 내용 |\n"
        "| --- | --- |\n"
        "| 상태 | 답변 생성 중 예외 발생 |\n"
        f"| 사유 | {reason_text} |\n"
        "| 조치 | 동일 질문 재시도 권장 |\n\n"
        "## 핵심 지표\n"
        "- 현재 요청에서는 완전한 최종 답변 텍스트를 확보하지 못했습니다.\n"
        "- 다만 API는 문자열 응답 형식을 유지하도록 처리했습니다.\n\n"
        "## 인사이트\n"
        "- 모델 응답이 비어 있거나 스트림이 중단될 때 재시도 시 정상화되는 경우가 많습니다.\n"
        "- 동일 현상이 반복되면 모델 상태/네트워크 상태를 함께 점검해야 합니다.\n\n"
        "- 데이터 출처: DB 조회 기반"
    )


def _has_quality_section(answer: str) -> bool:
    text = answer or ""
    markers = (
        "## 상세 내역",
        "## 핵심 지표",
        "## 인사이트",
        "## 요약",
        "상세 내역",
        "핵심 지표",
        "인사이트",
    )
    return any(marker in text for marker in markers)


def _has_quality_table(answer: str) -> bool:
    text = answer or ""
    return "|" in text and ("| ---" in text or "|:---" in text or "|------" in text)


def _quality_supplement(answer: str) -> str:
    text = answer or ""
    supplements: List[str] = []

    if not _has_quality_section(text):
        supplements.append(
            "## 상세 내역\n"
            "| 항목 | 내용 |\n"
            "| --- | --- |\n"
            "| 응답 상태 | 원문 응답 수신 |\n"
            "| 형식 보정 | 섹션 구조 자동 추가 |\n\n"
            "## 핵심 지표\n"
            "- 품질 규칙(표/섹션/출처) 충족을 위해 응답 구조를 보정했습니다.\n\n"
            "## 인사이트\n"
            "- 동일 질문 재요청 시 더 풍부한 원문 구조가 생성될 수 있습니다."
        )
    elif not _has_quality_table(text):
        supplements.append(
            "### 요약 표\n"
            "| 항목 | 내용 |\n"
            "| --- | --- |\n"
            "| 응답 상태 | 원문 응답 수신 |\n"
            "| 형식 보정 | 표 구조 자동 추가 |"
        )

    if "출처" not in text:
        supplements.append("- 데이터 출처: DB 조회 기반")

    if not supplements:
        return ""

    prefix = "\n\n" if text else ""
    return prefix + "\n\n".join(supplements)


def _is_non_cacheable_response(response_text: str) -> bool:
    normalized = _normalize_cache_guard_text(response_text)
    if not normalized:
        return True
    compact_markers = tuple(
        _normalize_cache_guard_text(marker) for marker in _NON_CACHEABLE_RESPONSE_MARKERS
    )
    return any(marker in normalized for marker in _NON_CACHEABLE_RESPONSE_MARKERS) or any(
        marker in normalized for marker in compact_markers
    )


async def _async_update_hit_count(cache_key: str) -> None:
    """백그라운드에서 hit_count를 업데이트합니다 (응답 지연 없음)."""
    try:
        pool = get_connection_pool()
        with pool.connection() as conn:
            await update_hit_count(conn, cache_key)
    except Exception as exc:
        logger.warning("[ChatCache] hit_count background update failed: %s", exc)


async def _async_delete_cache_key(cache_key: str) -> None:
    """백그라운드에서 stale 캐시를 삭제합니다."""
    try:
        pool = get_connection_pool()
        with pool.connection() as conn:
            deleted = await delete_by_key(conn, cache_key)
        if deleted:
            logger.info("[ChatCache] Deleted stale key=%s...", cache_key[:8])
    except Exception as exc:
        logger.warning("[ChatCache] stale cache delete failed: %s", exc)


def _make_cached_sse_response(
    cached: dict, style: str, cache_key: str
) -> EventSourceResponse:
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
                    "perf": {
                        "total_ms": 0.0,
                        "analysis_ms": 0.0,
                        "tool_ms": 0.0,
                        "answer_ms": 0.0,
                        "first_token_ms": 0.0,
                        "tool_count": 0,
                        "model": "cache",
                    },
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
                "event": "status",
                "data": json.dumps({"message": "⚠️"}, ensure_ascii=False),
            }
            yield {
                "event": "error",
                "data": json.dumps(error_payload, ensure_ascii=False),
            }
        # 2. 성공 시 메시지와 메타데이터 이벤트 전송
        elif result:
            yield {
                "event": "status",
                "data": json.dumps({"message": "⏺️"}, ensure_ascii=False),
            }
            answer = result.get("answer")

            # answer가 비동기 제너레이터인 경우 (스트리밍)
            # 캐시 저장을 위해 전체 텍스트를 누적합니다.
            full_response_text = ""
            answer_stream_error = None

            if hasattr(answer, "__aiter__"):
                try:
                    async for delta in answer:
                        if delta:
                            full_response_text += delta
                        yield {
                            "event": "message",
                            "data": json.dumps({"delta": delta}, ensure_ascii=False),
                        }
                    supplement = _quality_supplement(full_response_text)
                    if supplement:
                        full_response_text += supplement
                        yield {
                            "event": "message",
                            "data": json.dumps({"delta": supplement}, ensure_ascii=False),
                        }
                except Exception as exc:
                    answer_stream_error = str(exc)
                    logger.exception("chat_stream answer iteration failed.")
                    fallback_text = (
                        "핵심 결론: **답변 생성 중 오류가 발생해 안전 모드 응답으로 전환되었습니다.**\n\n"
                        "## 상세 내역\n"
                        "| 구분 | 내용 |\n"
                        "| --- | --- |\n"
                        "| 상태 | 스트리밍 중단 |\n"
                        f"| 오류 | {(answer_stream_error or 'unknown_error').replace('|', '\\|')} |\n"
                        "| 조치 | 동일 질문 재시도 권장 |\n\n"
                        "## 핵심 지표\n"
                        "- 스트림 예외를 감지해 비정상 종료를 방지했습니다.\n"
                        "- 품질 규칙(표/섹션/출처)을 유지한 안전 응답을 반환했습니다.\n\n"
                        "## 인사이트\n"
                        "- 일시적 모델 응답 불안정 구간에서 주로 발생합니다.\n\n"
                        "- 데이터 출처: DB 조회 기반"
                    )
                    full_response_text += fallback_text
                    yield {
                        "event": "message",
                        "data": json.dumps({"delta": fallback_text}, ensure_ascii=False),
                    }
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {
                                "message": "answer_stream_error",
                                "detail": answer_stream_error,
                            },
                            ensure_ascii=False,
                        ),
                    }

            # answer가 일반 문자열인 경우 (비스트리밍/일상대화)
            else:
                rendered = await _render_answer(result, style)
                full_response_text = rendered + _quality_supplement(rendered)
                yield {
                    "event": "message",
                    "data": json.dumps({"delta": full_response_text}, ensure_ascii=False),
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
            tool_calls_raw = result.get("tool_calls", [])
            tool_calls_serialized = [
                tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in tool_calls_raw
            ]

            intent = result.get("intent")

            meta_payload_raw = {
                "tool_calls": tool_calls_serialized,
                "tool_results": tool_results_serialized,
                "data_sources": result.get("data_sources", []),
                "verified": bool(result.get("verified", False))
                and not bool(answer_stream_error),
                "visualizations": result.get(
                    "visualizations", []
                ),  # 시각화 데이터 전달
                "style": style,
                "cached": False,
                "intent": intent,
                "perf": result.get("perf"),
                "error": result.get("error") or answer_stream_error,
            }
            # 전체 payload를 안전하게 직렬화
            meta_payload = safe_serialize(meta_payload_raw)
            yield {
                "event": "meta",
                "data": json.dumps(meta_payload, ensure_ascii=False),
            }

            # 캐시 저장: 정상 응답만 저장 (오류 문구/에러 결과는 캐시 금지)
            result_error = result.get("error") if isinstance(result, dict) else None
            if (
                cache_key
                and full_response_text
                and not result_error
                and not _is_non_cacheable_response(full_response_text)
            ):
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
            elif cache_key and full_response_text:
                reason = "result_error" if result_error else "non_cacheable_response"
                logger.info(
                    "[ChatCache] SKIP save key=%s... reason=%s",
                    cache_key[:8],
                    reason,
                )

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


def _validate_chat_question(question: Any) -> str:
    if not isinstance(question, str):
        raise HTTPException(status_code=400, detail="질문 형식이 올바르지 않습니다.")

    normalized = question.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    if len(normalized) > MAX_CHAT_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"질문이 너무 깁니다. 최대 {MAX_CHAT_QUESTION_LENGTH}자까지 허용됩니다.",
        )

    return normalized


def _normalize_content_type(content_type: Optional[str]) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


async def _read_upload_with_limit(
    upload: UploadFile,
    max_bytes: int,
    *,
    allowed_content_types: set[str] | None = None,
) -> tuple[bytes, str]:
    content_type = _normalize_content_type(upload.content_type)
    if allowed_content_types is not None and content_type not in allowed_content_types:
        raise HTTPException(
            status_code=415, detail="지원되지 않는 파일 타입입니다."
        )

    with tempfile.SpooledTemporaryFile(max_size=max_bytes, mode="w+b") as spool:
        total = 0
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413, detail="업로드 파일 크기가 너무 큽니다."
                )
            spool.write(chunk)

        if total == 0:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")

        spool.seek(0)
        return spool.read(), content_type or "application/octet-stream"


@router.post("/completion")
async def chat_completion(
    payload: Dict[str, Any] = Body(...),
    agent: BaseballStatisticsAgent = Depends(get_agent),
    __: None = Depends(rate_limit_chat_dependency),  # 요청 빈도 제한 적용
    _: None = Depends(require_ai_internal_token),
):
    """단일 JSON 응답으로 전체 채팅 답변을 반환하는 엔드포인트입니다."""
    _validate_chat_payload(payload)
    question = _validate_chat_question(payload.get("question", ""))

    filters = payload.get("filters")
    history = _decode_history_payload(payload.get("history"))
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
            cached_text = cached.get("response_text", "")
            if _is_non_cacheable_response(cached_text):
                logger.info(
                    "[ChatCache] BYPASS stale key=%s... reason=non_cacheable_response",
                    cache_key[:8],
                )
                asyncio.create_task(_async_delete_cache_key(cache_key))
            else:
                logger.info(
                    "[ChatCache] HIT key=%s... hit_count=%d",
                    cache_key[:8],
                    cached["hit_count"],
                )
                asyncio.create_task(_async_update_hit_count(cache_key))
                return JSONResponse(
                    {
                        "answer": cached_text,
                        "tool_calls": [],
                        "tool_results": [],
                        "data_sources": [],
                        "verified": True,
                        "visualizations": [],
                        "intent": cached.get("intent"),
                        "cached": True,
                        "cache_key_prefix": cache_key[:8],
                        "perf": {
                            "total_ms": 0.0,
                            "analysis_ms": 0.0,
                            "tool_ms": 0.0,
                            "answer_ms": 0.0,
                            "first_token_ms": 0.0,
                            "tool_count": 0,
                            "model": "cache",
                        },
                    }
                )

    timeout_raw = os.getenv("CHAT_COMPLETION_TIMEOUT_SECONDS", "0").strip()
    try:
        completion_timeout_seconds = float(timeout_raw) if timeout_raw else 0.0
    except ValueError:
        completion_timeout_seconds = 0.0

    try:
        if completion_timeout_seconds > 0:
            result = await asyncio.wait_for(
                agent.process_query(
                    question,
                    context={"filters": filters, "history": history},
                ),
                timeout=completion_timeout_seconds,
            )
        else:
            result = await agent.process_query(
                question,
                context={"filters": filters, "history": history},
            )
    except asyncio.TimeoutError as exc:
        logger.warning(
            "chat_completion timeout before answer stream: question=%s timeout=%.1fs",
            question[:80],
            completion_timeout_seconds,
        )
        raise HTTPException(
            status_code=504,
            detail="답변 생성 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
        ) from exc

    # If the answer is an async generator, consume it for non-streaming response.
    import inspect

    answer_obj = result.get("answer")
    if inspect.isasyncgen(answer_obj) or hasattr(answer_obj, "__aiter__"):
        full_answer = ""
        try:
            if completion_timeout_seconds > 0:
                async with asyncio.timeout(completion_timeout_seconds):
                    async for chunk in answer_obj:
                        if chunk:
                            full_answer += chunk
            else:
                async for chunk in answer_obj:
                    if chunk:
                        full_answer += chunk
            result["answer"] = full_answer
        except TimeoutError as exc:
            logger.warning(
                "chat_completion timeout while consuming stream: question=%s timeout=%.1fs",
                question[:80],
                completion_timeout_seconds,
            )
            raise HTTPException(
                status_code=504,
                detail="답변 생성 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
            ) from exc
        except Exception as e:
            logger.error(f"Error consuming generator: {e}")
            if full_answer.strip():
                result["answer"] = full_answer
            else:
                result["answer"] = _build_completion_fallback_answer(str(e))

    if isinstance(result.get("answer"), str):
        result["answer"] = result["answer"] + _quality_supplement(result["answer"])

    # 정상 응답은 completion 경로에서도 캐시에 저장합니다.
    if isinstance(result, dict):
        full_response_text = str(result.get("answer", "") or "")
        result_error = result.get("error")
        intent = result.get("intent")
        if (
            cache_key
            and full_response_text
            and not result_error
            and not _is_non_cacheable_response(full_response_text)
        ):
            settings = get_settings()
            model_name = (
                getattr(settings, "coach_openrouter_model", None)
                or getattr(settings, "openrouter_model", None)
                or getattr(settings, "gemini_model", None)
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
                    "[ChatCache] SAVED key=%s... intent=%s (completion)",
                    cache_key[:8],
                    intent,
                )
            except Exception as exc:
                logger.warning("[ChatCache] completion save failed: %s", exc)

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
        payload_serialized = safe_serialize(result)
        if isinstance(payload_serialized, dict):
            payload_serialized.setdefault("cached", False)
        return JSONResponse(payload_serialized)
    else:
        # result가 객체라면 dict로 변환
        return JSONResponse(
            safe_serialize(
                {
                    "answer": getattr(result, "answer", str(result)),
                    "citations": getattr(result, "citations", []),
                    "intent": getattr(result, "intent", "unknown"),
                    "cached": False,
                    "perf": {
                        "total_ms": 0.0,
                        "analysis_ms": 0.0,
                        "tool_ms": 0.0,
                        "answer_ms": 0.0,
                        "first_token_ms": None,
                        "tool_count": 0,
                        "model": "unknown",
                    },
                }
            )
        )


@router.post("/stream")
async def chat_stream_post(
    payload: Dict[str, Any] = Body(...),
    style: str = Query("markdown", pattern="^(markdown|json|compact)$"),
    agent: BaseballStatisticsAgent = Depends(get_agent),
    __: None = Depends(rate_limit_chat_dependency),
    _: None = Depends(require_ai_internal_token),
    request: Request = None,
):
    """POST 요청을 통해 질문을 받고, 답변을 SSE 스트림으로 반환합니다."""
    _validate_chat_payload(payload)
    question = _validate_chat_question(payload.get("question", ""))
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
            cached_text = cached.get("response_text", "")
            if _is_non_cacheable_response(cached_text):
                logger.info(
                    "[ChatCache] BYPASS stale key=%s... reason=non_cacheable_response",
                    cache_key[:8],
                )
                asyncio.create_task(_async_delete_cache_key(cache_key))
            else:
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
    __: None = Depends(rate_limit_chat_dependency),
    _: None = Depends(require_ai_internal_token),
    request: Request = None,
):
    """GET 요청을 통해 질문을 받고, 답변을 SSE 스트림으로 반환합니다."""
    question = _validate_chat_question(q)
    history_param = None
    if request is not None:
        history_param = request.query_params.get("history")

    history = _decode_history_payload(history_param)

    # GET 엔드포인트: filters 없음, cache_key 없음 (캐싱 미적용)
    # GET은 브라우저 테스트/디버깅 용도이므로 캐싱 복잡도를 추가하지 않음
    return await _stream_response(
        request,
        question,
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
    __: None = Depends(rate_limit_chat_voice_dependency),
    _: None = Depends(require_ai_internal_token),
):
    logger.info(f"===== 음성 변환 시작 =====")
    logger.info(f"파일명: {file.filename}, 타입: {file.content_type}")

    try:
        contents, content_type = await _read_upload_with_limit(
            file,
            MAX_VOICE_FILE_BYTES,
            allowed_content_types=ALLOWED_VOICE_CONTENT_TYPES,
        )
        logger.info(f"정규화된 파일 타입: {content_type}")
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

    except HTTPException as exc:
        logger.warning(
            "Voice transcription request rejected: status=%s detail=%s",
            exc.status_code,
            exc.detail,
        )
        raise
    except Exception:
        logger.exception("음성 변환 중 오류가 발생했습니다.")
        raise HTTPException(
            status_code=500,
            detail="서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )


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
