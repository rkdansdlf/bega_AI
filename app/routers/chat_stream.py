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
import re
import secrets
import tempfile
from datetime import date, datetime
from typing import Any, AsyncGenerator, Dict, List, Optional
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
CHAT_CACHE_SCHEMA_VERSION = "v9"
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
        raise HTTPException(
            status_code=400, detail="요청 본문을 파싱할 수 없습니다."
        ) from exc

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
    return (
        "방금 답변이 중간에 끊겼습니다. 같은 질문을 한 번 더 보내주시면 "
        "바로 다시 이어서 답하겠습니다."
    )


def _remaining_timeout_seconds(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        return 0.0
    return remaining


def _format_chatbot_table_row(cells: List[str]) -> Optional[str]:
    normalized = [str(cell).strip() for cell in cells if str(cell).strip()]
    if not normalized:
        return None
    label = normalized[0]
    particle = _select_topic_particle(label)
    if len(normalized) >= 3:
        return (
            f"{label}{particle} {normalized[1]}이고, "
            f"{normalized[2]} 정도로 보면 됩니다."
        )
    if len(normalized) == 2:
        return f"{label}{particle} {normalized[1]}입니다."
    return normalized[0]


def _extract_particle_target(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("**") and cleaned.endswith("**") and len(cleaned) > 4:
        cleaned = cleaned[2:-2].strip()
    cleaned = re.sub(r"[`*_~]+", "", cleaned)
    return cleaned.strip()


def _get_last_char(text: str) -> str:
    cleaned = _extract_particle_target(text)
    return cleaned[-1] if cleaned else ""


def _has_batchim(text: str) -> bool:
    last_char = _get_last_char(text)
    if not last_char:
        return False
    if "가" <= last_char <= "힣":
        return (ord(last_char) - ord("가")) % 28 != 0
    return False


def _select_topic_particle(text: str) -> str:
    return "은" if _has_batchim(text) else "는"


def _select_subject_particle(text: str) -> str:
    return "이" if _has_batchim(text) else "가"


def _select_direction_particle(text: str) -> str:
    last_char = _get_last_char(text)
    if not last_char:
        return "로"
    if "가" <= last_char <= "힣":
        final_consonant = (ord(last_char) - ord("가")) % 28
        if final_consonant == 0 or final_consonant == 8:
            return "로"
        return "으로"
    return "로"


def _postprocess_chatbot_answer_text(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return normalized

    normalized = re.sub(
        r"^(.*?는) 타선은 ([^,]+), 마운드는 ([^.]+) 기준으로 현재 전력의 방향성은 확인 가능합니다\.$",
        r"\1 \2의 타선과 \3의 마운드를 보면 지금 전력 흐름은 읽힙니다.",
        normalized,
        count=1,
        flags=re.MULTILINE,
    )

    def replace_bold_particle(match: re.Match[str]) -> str:
        token = match.group(1)
        particle = match.group(2)
        if particle in {"은", "는"}:
            fixed = _select_topic_particle(token)
        elif particle in {"이", "가"}:
            fixed = _select_subject_particle(token)
        else:
            fixed = _select_direction_particle(token)
        return f"{token}{fixed}"

    normalized = re.sub(
        r"(\*\*[^*]+\*\*)(은|는|이|가|로)(?=[\s,.)!?]|$)",
        replace_bold_particle,
        normalized,
    )
    return normalized


def _normalize_chatbot_answer_text(answer: str) -> str:
    text = answer or ""
    paragraphs: List[str] = []
    table_rows: List[List[str]] = []
    table_headers: List[str] = []

    def flush_table() -> None:
        nonlocal table_rows, table_headers
        if not table_rows:
            return
        table_sentences: List[str] = []
        for row in table_rows[:4]:
            sentence = _format_chatbot_table_row(row)
            if sentence:
                table_sentences.append(sentence)
        if table_sentences:
            paragraphs.append(" ".join(table_sentences))
        table_rows = []
        table_headers = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            flush_table()
            continue

        lowered = stripped.lower()
        if stripped.startswith("## ") or stripped.startswith("### "):
            flush_table()
            continue
        if lowered.startswith("출처:") or lowered.startswith("- 출처:"):
            flush_table()
            continue
        if lowered.startswith("source:") or lowered.startswith("- source:"):
            flush_table()
            continue

        if stripped.startswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            is_divider = all(cell and set(cell) <= {"-", ":", " "} for cell in cells)
            if is_divider:
                continue
            if not table_headers:
                table_headers = cells
            else:
                table_rows.append(cells)
            continue

        flush_table()

        if stripped.startswith("- ") or stripped.startswith("* "):
            stripped = stripped[2:].strip()
        if stripped:
            paragraphs.append(stripped)

    flush_table()
    normalized = "\n\n".join(part for part in paragraphs if part).strip()
    return _postprocess_chatbot_answer_text(normalized or text.strip())


def _ensure_quality_answer_text(answer: str) -> str:
    return _normalize_chatbot_answer_text(answer or "")


def _is_non_cacheable_response(response_text: str) -> bool:
    normalized = _normalize_cache_guard_text(response_text)
    if not normalized:
        return True
    compact_markers = tuple(
        _normalize_cache_guard_text(marker)
        for marker in _NON_CACHEABLE_RESPONSE_MARKERS
    )
    return any(
        marker in normalized for marker in _NON_CACHEABLE_RESPONSE_MARKERS
    ) or any(marker in normalized for marker in compact_markers)


def _safe_serialize(obj: Any) -> Any:
    """JSON 직렬화 가능한 형태로 객체를 변환합니다."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, "to_dict"):
        return _safe_serialize(obj.to_dict())
    if isinstance(obj, dict):
        return {key: _safe_serialize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if hasattr(obj, "__dict__"):
        return {key: _safe_serialize(value) for key, value in obj.__dict__.items()}
    return str(obj)


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
        response_text = _ensure_quality_answer_text(cached["response_text"])
        settings = get_settings()

        # status 이벤트: 캐시 히트 표시 (번개 이모지로 빠른 응답임을 암시)
        yield {
            "event": "status",
            "data": json.dumps({"message": "⚡"}, ensure_ascii=False),
        }

        # message 이벤트: 200자 청크로 나눠 전송 (프론트엔드 타이핑 효과 유지)
        chunk_size = max(1, int(settings.chat_cached_stream_chunk_size))
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
                    "grounding_mode": "cache",
                    "source_tier": "cache",
                    "answer_sources": [],
                    "as_of_date": None,
                    "fallback_reason": None,
                    "finish_reason": "completed",
                    "cancelled": False,
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
        ping=max(1, int(get_settings().chat_sse_ping_seconds)),
    )


async def _request_is_disconnected(request: Optional[Request]) -> bool:
    if request is None:
        return False

    try:
        return await request.is_disconnected()
    except RuntimeError:
        return False


def _log_cancelled_stream(question: str, source: str) -> None:
    logger.info("chat_stream cancelled source=%s question=%s", source, question[:120])


async def _chat_event_generator(
    *,
    request: Optional[Request],
    question: str,
    filters: Optional[Dict[str, Any]],
    style: str,
    result: Optional[Dict[str, Any]],
    error_payload: Optional[Dict[str, Any]],
    cache_key: Optional[str],
) -> AsyncGenerator[Dict[str, str], None]:
    """Build the SSE event stream while respecting downstream disconnects."""
    if error_payload:
        if await _request_is_disconnected(request):
            _log_cancelled_stream(question, "before-error-status")
            return
        yield {
            "event": "status",
            "data": json.dumps({"message": "⚠️"}, ensure_ascii=False),
        }
        if await _request_is_disconnected(request):
            _log_cancelled_stream(question, "after-error-status")
            return
        yield {
            "event": "error",
            "data": json.dumps(error_payload, ensure_ascii=False),
        }
        if await _request_is_disconnected(request):
            _log_cancelled_stream(question, "after-error")
            return
        yield {"event": "done", "data": "[DONE]"}
        return

    if not result:
        if not await _request_is_disconnected(request):
            yield {"event": "done", "data": "[DONE]"}
        return

    if await _request_is_disconnected(request):
        _log_cancelled_stream(question, "before-status")
        return

    yield {
        "event": "status",
        "data": json.dumps({"message": "⏺️"}, ensure_ascii=False),
    }

    answer = result.get("answer")
    full_response_chunks: List[str] = []
    answer_stream_error: Optional[str] = None
    stream_cancelled = False

    if hasattr(answer, "__aiter__"):
        try:
            async for delta in answer:
                if await _request_is_disconnected(request):
                    stream_cancelled = True
                    _log_cancelled_stream(question, "downstream-disconnect")
                    break
                if not delta:
                    continue
                full_response_chunks.append(delta)
                yield {
                    "event": "message",
                    "data": json.dumps({"delta": delta}, ensure_ascii=False),
                }
        except asyncio.CancelledError:
            stream_cancelled = True
            _log_cancelled_stream(question, "answer-iterator-cancelled")
        except Exception as exc:  # noqa: BLE001
            answer_stream_error = str(exc)
            logger.exception("chat_stream answer iteration failed.")
            fallback_text = (
                "지금 답변이 중간에 잠깐 끊겼습니다. "
                "같은 질문을 한 번 더 보내주시면 바로 다시 이어서 볼게요."
            )
            full_response_chunks.append(fallback_text)
            yield {
                "event": "message",
                "data": json.dumps({"delta": fallback_text}, ensure_ascii=False),
            }
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "message": "temporary_generation_issue",
                        "detail": "답변 생성이 잠깐 끊겨 재시도가 필요합니다.",
                    },
                    ensure_ascii=False,
                ),
            }
    else:
        if await _request_is_disconnected(request):
            stream_cancelled = True
            _log_cancelled_stream(question, "before-single-message")
        else:
            rendered = await _render_answer(result, style)
            full_response_chunks.append(_ensure_quality_answer_text(rendered))
            full_response_text = "".join(full_response_chunks)
            yield {
                "event": "message",
                "data": json.dumps({"delta": full_response_text}, ensure_ascii=False),
            }

    full_response_text = "".join(full_response_chunks)
    if not stream_cancelled and await _request_is_disconnected(request):
        stream_cancelled = True
        _log_cancelled_stream(question, "before-meta")

    tool_results_raw = result.get("tool_results", [])
    tool_results_serialized = _safe_serialize(tool_results_raw)
    tool_calls_raw = result.get("tool_calls", [])
    tool_calls_serialized = [
        tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in tool_calls_raw
    ]

    intent = result.get("intent")
    public_error = None
    internal_result_error = result.get("error")
    if internal_result_error:
        logger.warning(
            "chat_stream user_error_hidden result_error=%s",
            internal_result_error,
        )
        public_error = "temporary_generation_issue"
    if answer_stream_error:
        logger.warning(
            "chat_stream user_error_hidden stream_error=%s",
            answer_stream_error,
        )
        public_error = "temporary_generation_issue"

    finish_reason = "cancelled" if stream_cancelled else ("error" if public_error else "completed")
    meta_payload_raw = {
        "tool_calls": tool_calls_serialized,
        "tool_results": tool_results_serialized,
        "data_sources": result.get("data_sources", []),
        "verified": bool(result.get("verified", False)) and not bool(answer_stream_error),
        "visualizations": result.get("visualizations", []),
        "style": style,
        "cached": False,
        "intent": intent,
        "strategy": result.get("strategy"),
        "planner_mode": result.get("planner_mode"),
        "grounding_mode": result.get("grounding_mode"),
        "source_tier": result.get("source_tier"),
        "answer_sources": result.get("answer_sources", []),
        "as_of_date": result.get("as_of_date"),
        "fallback_reason": result.get("fallback_reason"),
        "fallback_answer_used": bool(result.get("fallback_answer_used", False))
        or bool(answer_stream_error),
        "perf": result.get("perf"),
        "error": public_error,
        "finish_reason": finish_reason,
        "cancelled": stream_cancelled,
    }

    if stream_cancelled:
        if cache_key and full_response_text:
            logger.info(
                "[ChatCache] SKIP save key=%s... reason=cancelled",
                cache_key[:8],
            )

        if not await _request_is_disconnected(request):
            meta_payload = _safe_serialize(meta_payload_raw)
            yield {
                "event": "meta",
                "data": json.dumps(meta_payload, ensure_ascii=False),
            }
            yield {"event": "done", "data": "[DONE]"}
        return
    meta_payload = _safe_serialize(meta_payload_raw)
    yield {
        "event": "meta",
        "data": json.dumps(meta_payload, ensure_ascii=False),
    }

    result_error = result.get("error") if isinstance(result, dict) else None
    if (
        cache_key
        and full_response_text
        and not result_error
        and not public_error
        and not bool(answer_stream_error)
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
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ChatCache] save failed: %s", exc)
    elif cache_key and full_response_text:
        if result_error:
            reason = "result_error"
        elif public_error or answer_stream_error:
            reason = "public_or_stream_error"
        else:
            reason = "non_cacheable_response"
        logger.info(
            "[ChatCache] SKIP save key=%s... reason=%s",
            cache_key[:8],
            reason,
        )

    yield {"event": "done", "data": "[DONE]"}


async def _chat_live_event_generator(
    *,
    request: Optional[Request],
    question: str,
    filters: Optional[Dict[str, Any]],
    style: str,
    cache_key: Optional[str],
    stream,
) -> AsyncGenerator[Dict[str, str], None]:
    settings = get_settings()
    full_response_chunks: List[str] = []
    buffered_meta: Dict[str, Any] = {}
    answer_stream_error: Optional[str] = None
    stream_cancelled = False

    if await _request_is_disconnected(request):
        _log_cancelled_stream(question, "before-live-status")
        return

    yield {
        "event": "status",
        "data": json.dumps({"message": "⏺️"}, ensure_ascii=False),
    }

    try:
        async for event in stream:
            if await _request_is_disconnected(request):
                stream_cancelled = True
                _log_cancelled_stream(question, "downstream-disconnect-live")
                break

            event_type = event.get("type")
            if event_type == "status":
                status_message = str(event.get("message", "")).strip()
                if status_message:
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {"message": status_message}, ensure_ascii=False
                        ),
                    }
            elif event_type == "tool_start":
                tool_name = str(event.get("tool", "")).strip()
                if tool_name:
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {"message": f"{tool_name} 조회 중"}, ensure_ascii=False
                        ),
                    }
            elif event_type == "tool_result":
                tool_message = str(event.get("message", "")).strip()
                if tool_message:
                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {"message": tool_message}, ensure_ascii=False
                        ),
                    }
            elif event_type == "answer_chunk":
                delta = event.get("content")
                if not delta:
                    continue
                full_response_chunks.append(delta)
                yield {
                    "event": "message",
                    "data": json.dumps({"delta": delta}, ensure_ascii=False),
                }
            elif event_type == "metadata":
                raw_meta = event.get("data")
                if isinstance(raw_meta, dict):
                    buffered_meta = raw_meta
    except asyncio.CancelledError:
        stream_cancelled = True
        _log_cancelled_stream(question, "live-stream-cancelled")
    except Exception as exc:  # noqa: BLE001
        answer_stream_error = str(exc)
        logger.exception("chat_stream process_query_stream iteration failed.")
        fallback_text = (
            "지금 답변이 중간에 잠깐 끊겼습니다. "
            "같은 질문을 한 번 더 보내주시면 바로 다시 이어서 볼게요."
        )
        full_response_chunks.append(fallback_text)
        yield {
            "event": "message",
            "data": json.dumps({"delta": fallback_text}, ensure_ascii=False),
        }
        yield {
            "event": "error",
            "data": json.dumps(
                {
                    "message": "temporary_generation_issue",
                    "detail": "답변 생성이 잠깐 끊겨 재시도가 필요합니다.",
                },
                ensure_ascii=False,
            ),
        }

    full_response_text = "".join(full_response_chunks)
    if not stream_cancelled and await _request_is_disconnected(request):
        stream_cancelled = True
        _log_cancelled_stream(question, "before-live-meta")

    tool_results_raw = buffered_meta.get("tool_results", [])
    tool_results_serialized = _safe_serialize(tool_results_raw)
    tool_calls_raw = buffered_meta.get("tool_calls", [])
    tool_calls_serialized = [
        tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in tool_calls_raw
    ]

    public_error = None
    internal_result_error = buffered_meta.get("error")
    if internal_result_error:
        logger.warning(
            "chat_stream user_error_hidden result_error=%s",
            internal_result_error,
        )
        public_error = "temporary_generation_issue"
    if answer_stream_error:
        logger.warning(
            "chat_stream user_error_hidden stream_error=%s",
            answer_stream_error,
        )
        public_error = "temporary_generation_issue"

    intent = buffered_meta.get("intent")
    finish_reason = "cancelled" if stream_cancelled else ("error" if public_error else "completed")
    meta_payload_raw = {
        "tool_calls": tool_calls_serialized,
        "tool_results": tool_results_serialized,
        "data_sources": buffered_meta.get("data_sources", []),
        "verified": bool(buffered_meta.get("verified", False))
        and not bool(answer_stream_error),
        "visualizations": buffered_meta.get("visualizations", []),
        "style": style,
        "cached": False,
        "intent": intent,
        "strategy": buffered_meta.get("strategy"),
        "planner_mode": buffered_meta.get("planner_mode"),
        "grounding_mode": buffered_meta.get("grounding_mode"),
        "source_tier": buffered_meta.get("source_tier"),
        "answer_sources": buffered_meta.get("answer_sources", []),
        "as_of_date": buffered_meta.get("as_of_date"),
        "fallback_reason": buffered_meta.get("fallback_reason"),
        "fallback_answer_used": bool(buffered_meta.get("fallback_answer_used", False))
        or bool(answer_stream_error),
        "perf": buffered_meta.get("perf"),
        "error": public_error,
        "finish_reason": finish_reason,
        "cancelled": stream_cancelled,
    }

    if stream_cancelled:
        if cache_key and full_response_text:
            logger.info(
                "[ChatCache] SKIP save key=%s... reason=cancelled",
                cache_key[:8],
            )

        if not await _request_is_disconnected(request):
            meta_payload = _safe_serialize(meta_payload_raw)
            yield {
                "event": "meta",
                "data": json.dumps(meta_payload, ensure_ascii=False),
            }
            yield {"event": "done", "data": "[DONE]"}
        return
    meta_payload = _safe_serialize(meta_payload_raw)
    yield {
        "event": "meta",
        "data": json.dumps(meta_payload, ensure_ascii=False),
    }

    result_error = buffered_meta.get("error")
    intent = buffered_meta.get("intent")
    if (
        cache_key
        and full_response_text
        and not result_error
        and not public_error
        and not bool(answer_stream_error)
        and not _is_non_cacheable_response(full_response_text)
    ):
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
                "[ChatCache] SAVED key=%s... intent=%s",
                cache_key[:8],
                intent,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ChatCache] save failed: %s", exc)
    elif cache_key and full_response_text:
        if result_error:
            reason = "result_error"
        elif public_error or answer_stream_error:
            reason = "public_or_stream_error"
        else:
            reason = "non_cacheable_response"
        logger.info(
            "[ChatCache] SKIP save key=%s... reason=%s",
            cache_key[:8],
            reason,
        )

    yield {"event": "done", "data": "[DONE]"}


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

    settings = get_settings()
    context_messages = history if history else [{"role": "user", "content": question}]
    agent_context = {
        "filters": filters,
        "history": history,
        "messages": context_messages,
        "request_mode": "stream",
        "persona": "chat",
    }

    if hasattr(agent, "process_query_stream"):
        try:
            live_stream = agent.process_query_stream(question, context=agent_context)
        except Exception:  # noqa: BLE001
            logger.exception("chat_stream live stream initialization failed.")
            live_stream = None
        else:
            return EventSourceResponse(
                _chat_live_event_generator(
                    request=request,
                    question=question,
                    filters=filters,
                    style=style,
                    cache_key=cache_key,
                    stream=live_stream,
                ),
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
                ping=max(1, int(settings.chat_sse_ping_seconds)),
            )

    result: Optional[Dict[str, Any]] = None
    error_payload: Optional[Dict[str, Any]] = None
    try:
        result = await agent.process_query(
            question,
            context=agent_context,
        )
    except asyncio.CancelledError:
        logger.info("chat_stream cancelled before response could start. question=%s", question[:120])
        raise
    except Exception:  # noqa: BLE001
        logger.exception("chat_stream에서 오류가 발생했습니다.")
        error_payload = {
            "message": "temporary_issue",
            "detail": "지금은 응답 템포가 잠깐 흔들리고 있어요. 같은 질문을 다시 보내주세요.",
        }

    return EventSourceResponse(
        _chat_event_generator(
            request=request,
            question=question,
            filters=filters,
            style=style,
            result=result,
            error_payload=error_payload,
            cache_key=cache_key,
        ),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx 등 프록시에서 버퍼링 방지
        },
        ping=max(1, int(settings.chat_sse_ping_seconds)),
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
        raise HTTPException(status_code=415, detail="지원되지 않는 파일 타입입니다.")

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
                        "answer": _ensure_quality_answer_text(cached_text),
                        "tool_calls": [],
                        "tool_results": [],
                        "data_sources": [],
                        "verified": True,
                        "visualizations": [],
                        "intent": cached.get("intent"),
                        "cached": True,
                        "grounding_mode": "cache",
                        "source_tier": "cache",
                        "answer_sources": [],
                        "as_of_date": None,
                        "fallback_reason": None,
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

    settings = get_settings()
    completion_timeout_seconds = max(
        0.0, float(settings.chat_completion_timeout_seconds)
    )
    timeout_deadline: Optional[float] = None
    if completion_timeout_seconds > 0:
        timeout_deadline = asyncio.get_running_loop().time() + completion_timeout_seconds

    try:
        context_messages = (
            history if history else [{"role": "user", "content": question}]
        )
        context = {
            "filters": filters,
            "history": history,
            "messages": context_messages,
            "request_mode": "completion",
            "persona": "chat",
        }
        if completion_timeout_seconds > 0:
            result = await asyncio.wait_for(
                agent.process_query(question, context=context),
                timeout=completion_timeout_seconds,
            )
        else:
            result = await agent.process_query(question, context=context)
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
        full_answer_chunks: List[str] = []
        public_error: Optional[str] = None
        try:
            if completion_timeout_seconds > 0:
                remaining_timeout = _remaining_timeout_seconds(timeout_deadline)
                if remaining_timeout is not None and remaining_timeout <= 0:
                    raise TimeoutError
                async with asyncio.timeout(remaining_timeout):
                    async for chunk in answer_obj:
                        if chunk:
                            full_answer_chunks.append(chunk)
            else:
                async for chunk in answer_obj:
                    if chunk:
                        full_answer_chunks.append(chunk)
            result["answer"] = "".join(full_answer_chunks)
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
            logger.error("Error consuming generator: %s", e)
            public_error = "temporary_generation_issue"
            if full_answer_chunks:
                result["answer"] = "".join(full_answer_chunks)
            else:
                result["answer"] = _build_completion_fallback_answer(str(e))
        if public_error:
            result["error"] = public_error

    if isinstance(result.get("answer"), str):
        result["answer"] = _ensure_quality_answer_text(result["answer"])

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

    if isinstance(result, dict):
        payload_serialized = _safe_serialize(result)
        if isinstance(payload_serialized, dict):
            answer_text = payload_serialized.get("answer")
            if isinstance(answer_text, str):
                payload_serialized["answer"] = _ensure_quality_answer_text(answer_text)
            payload_serialized.setdefault("cached", False)
            if payload_serialized.get("error"):
                logger.warning(
                    "chat_completion user_error_hidden error=%s",
                    payload_serialized.get("error"),
                )
                payload_serialized["error"] = "temporary_generation_issue"
        return JSONResponse(payload_serialized)
    else:
        # result가 객체라면 dict로 변환
        answer_text = getattr(result, "answer", str(result))
        if not isinstance(answer_text, str):
            answer_text = str(answer_text)
        answer_text = _ensure_quality_answer_text(answer_text)
        return JSONResponse(
            _safe_serialize(
                {
                    "answer": answer_text,
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
