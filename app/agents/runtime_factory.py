from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ..config import get_settings
from ..core.http_clients import get_shared_httpx_client
from ..core.shared_resources import get_shared_latest_baseball_tool
from .baseball_agent import BaseballAgentRuntime

logger = logging.getLogger(__name__)
_gemini_configured = False


def _ensure_gemini_configured(settings: Any) -> None:
    global _gemini_configured
    if not _gemini_configured and getattr(settings, "gemini_api_key", None):
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        _gemini_configured = True


def _extract_text_from_openrouter_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                nested_content = item.get("content")
                if isinstance(nested_content, str) and nested_content:
                    parts.append(nested_content)
        return "".join(parts)
    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
    return ""


def _parse_openrouter_stream_delta(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return "", "non_object_payload"

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return "", "missing_choices"
    if not choices:
        return "", "empty_choices"

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return "", "malformed_choice"

    delta_obj = first_choice.get("delta")
    if isinstance(delta_obj, dict):
        delta_text = _extract_text_from_openrouter_content(delta_obj.get("content"))
        if delta_text:
            return delta_text, "ok"

    message_obj = first_choice.get("message")
    if isinstance(message_obj, dict):
        message_text = _extract_text_from_openrouter_content(message_obj.get("content"))
        if message_text:
            return message_text, "ok"

    text_value = first_choice.get("text")
    if isinstance(text_value, str) and text_value:
        return text_value, "ok"

    if first_choice.get("finish_reason"):
        return "", "finished"
    return "", "empty_content"


def build_baseball_llm_generator(settings: Any):
    llm_logger = logging.getLogger("BaseballAgent")

    def is_server_error(exception: Exception) -> bool:
        return (
            isinstance(exception, httpx.HTTPStatusError)
            and exception.response.status_code >= 500
        )

    def _resolve_model_candidates(
        primary_model: str, fallback_models: list[str]
    ) -> list[str]:
        blocked = {"openrouter/auto"}
        candidates: list[str] = []
        for model in [primary_model] + list(fallback_models):
            if not model:
                continue
            if model in blocked:
                llm_logger.warning("[LLM] Skipping blocked model: %s", model)
                continue
            if model not in candidates:
                candidates.append(model)

        if candidates:
            return candidates

        if primary_model:
            llm_logger.warning(
                "[LLM] All configured models are blocked; fallback to primary: %s",
                primary_model,
            )
            return [primary_model]

        return [model for model in fallback_models if model]

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(is_server_error),
        before_sleep=before_sleep_log(llm_logger, logging.WARNING),
    )
    async def fetch_completion_stream(payload, headers):
        client = get_shared_httpx_client(
            "openrouter",
            timeout=120.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        async with client.stream(
            "POST",
            f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            if 400 <= response.status_code < 500:
                error_body = await response.aread()
                llm_logger.error(
                    "[OpenRouter 4xx] Status: %s, Body: %s",
                    response.status_code,
                    error_body.decode("utf-8", errors="replace"),
                )
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line

    async def openrouter_generator(messages, max_tokens=None):
        if not settings.openrouter_api_key:
            raise RuntimeError("OpenRouter API key is required.")

        effective_max_tokens = max_tokens or settings.max_output_tokens
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_referer or "",
            "X-Title": settings.openrouter_app_title or "",
        }
        models_to_try = _resolve_model_candidates(
            settings.openrouter_model,
            settings.openrouter_fallback_models,
        )
        llm_logger.info(
            "[LLM] Models to try (filtered): %s, max_tokens=%s",
            models_to_try,
            effective_max_tokens,
        )

        last_exception = None
        empty_chunk_retries = max(
            0, int(settings.chat_openrouter_empty_chunk_retries)
        )
        empty_chunk_backoff_ms = max(
            50, int(settings.chat_openrouter_empty_chunk_backoff_ms)
        )

        for index, model in enumerate(models_to_try):
            if index > 0:
                llm_logger.warning(
                    "[LLM Fallback] Trying model %d/%d: %s",
                    index + 1,
                    len(models_to_try),
                    model,
                )
            else:
                llm_logger.info(
                    "[LLM] Primary: %s, Fallbacks available: %s",
                    model,
                    settings.openrouter_fallback_models,
                )

            for retry_index in range(empty_chunk_retries + 1):
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.1,
                    "max_tokens": effective_max_tokens,
                }

                try:
                    chunk_count = 0
                    empty_choice_count = 0
                    malformed_chunk_count = 0

                    async for line in fetch_completion_stream(payload, headers):
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta, parse_reason = _parse_openrouter_stream_delta(
                                    data
                                )
                                if parse_reason in {"missing_choices", "empty_choices"}:
                                    empty_choice_count += 1
                                elif parse_reason in {
                                    "non_object_payload",
                                    "malformed_choice",
                                }:
                                    malformed_chunk_count += 1
                                if delta:
                                    chunk_count += 1
                                    yield delta
                            except json.JSONDecodeError:
                                llm_logger.warning("[LLM] JSON decode failed for chunk")
                                continue

                    should_retry_empty = (
                        chunk_count == 0
                        and retry_index < empty_chunk_retries
                        and empty_choice_count > 0
                    )
                    if should_retry_empty:
                        llm_logger.warning(
                            "[LLM] Empty completion chunk set detected for %s "
                            "(attempt %d/%d, empty_choice_count=%d malformed=%d)",
                            model,
                            retry_index + 1,
                            empty_chunk_retries + 1,
                            empty_choice_count,
                            malformed_chunk_count,
                        )
                        await asyncio.sleep(empty_chunk_backoff_ms / 1000.0)
                        continue

                    llm_logger.info("[LLM] Success: %d chunks from %s", chunk_count, model)
                    return

                except Exception as exc:  # noqa: BLE001
                    llm_logger.error("[LLM] Model %s failed: %s", model, exc)
                    last_exception = exc
                    break

        llm_logger.error(
            "[LLM] All %d models failed. Last error: %s",
            len(models_to_try),
            last_exception,
        )
        raise last_exception or RuntimeError("All models failed")

    async def gemini_generator(messages, max_tokens=None):
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig

        if not settings.gemini_api_key:
            raise RuntimeError("Gemini API key is required.")

        effective_max_tokens = max_tokens or settings.max_output_tokens
        _ensure_gemini_configured(settings)
        model = genai.GenerativeModel(settings.gemini_model)

        gemini_messages = []
        system_instruction = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = content
            else:
                gemini_role = "user" if role == "user" else "model"
                gemini_messages.append({"role": gemini_role, "parts": [content]})

        if system_instruction:
            model = genai.GenerativeModel(
                model_name=settings.gemini_model,
                system_instruction=system_instruction,
            )

        try:
            response = await model.generate_content_async(
                gemini_messages,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=effective_max_tokens,
                ),
                stream=True,
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:  # noqa: BLE001
            llm_logger.error("Gemini generation failed: %s", exc)
            raise

    if settings.llm_provider == "gemini":
        return gemini_generator
    return openrouter_generator


def create_baseball_agent_runtime(
    settings: Optional[Any] = None,
) -> BaseballAgentRuntime:
    resolved_settings = settings or get_settings()
    return BaseballAgentRuntime(
        llm_generator=build_baseball_llm_generator(resolved_settings),
        settings=resolved_settings,
        latest_baseball_tool=get_shared_latest_baseball_tool(),
        fast_path_enabled=resolved_settings.chat_fast_path_enabled,
        fast_path_scope=resolved_settings.chat_fast_path_scope,
        fast_path_min_messages=resolved_settings.chat_fast_path_min_messages,
        fast_path_tool_cap=resolved_settings.chat_fast_path_tool_cap,
        fast_path_fallback_on_empty=resolved_settings.chat_fast_path_fallback_on_empty,
        chat_dynamic_token_enabled=resolved_settings.chat_dynamic_token_enabled,
        chat_analysis_max_tokens=resolved_settings.chat_analysis_max_tokens,
        chat_answer_max_tokens_short=resolved_settings.chat_answer_max_tokens_short,
        chat_answer_max_tokens_long=resolved_settings.chat_answer_max_tokens_long,
        chat_answer_max_tokens_team=resolved_settings.chat_answer_max_tokens_team,
        chat_tool_result_max_chars=resolved_settings.chat_tool_result_max_chars,
        chat_tool_result_max_items=resolved_settings.chat_tool_result_max_items,
        chat_first_token_watchdog_seconds=resolved_settings.chat_first_token_watchdog_seconds,
        chat_first_token_retry_max_attempts=resolved_settings.chat_first_token_retry_max_attempts,
        chat_stream_first_token_watchdog_seconds=resolved_settings.chat_stream_first_token_watchdog_seconds,
        chat_stream_first_token_retry_max_attempts=resolved_settings.chat_stream_first_token_retry_max_attempts,
    )
