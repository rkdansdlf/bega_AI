from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import httpx
import logging

logger = logging.getLogger(__name__)

# LLM API 호출에 대한 표준 재시도 데코레이터
# 429 (Rate Limit) 및 5xx (Transient Server Error)에 대해 재시도 수행
llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=(
        retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)) |
        # Note: Add specific LLM provider exceptions if needed (e.g. OpenAI Error types)
        retry_if_exception_type(Exception) # fallback to general for unexpected transient issues
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
