import asyncio
import time
import os
from collections import defaultdict, deque
from typing import Deque, DefaultDict
from .security_metrics import record_security_event

from fastapi import HTTPException, Request


TRUST_X_FORWARDED_FOR_FOR_RATE_LIMIT = os.getenv(
    "AI_TRUST_X_FORWARDED_FOR", "false"
).strip().lower() in {"1", "true", "yes", "on"}


class InMemoryRateLimiter:
    """
    InMemoryRateLimiter는 API 서버의 과부하를 방지하기 위한 인메모리 속도 제한(Rate Limiter) 장치입니다.
    지정된 시간(window_seconds) 내에 특정 키(예: 클라이언트 IP)에 대한 요청 수를 제한합니다.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: DefaultDict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> None:
        """
        주어진 키(key)에 대해 속도 제한을 확인하고 적용합니다.
        요청이 허용된도를 초과하면 HTTPException(429 Too Many Requests)을 발생시킵니다.
        """
        now = time.time()
        async with self._lock:
            bucket = self._hits[key]
            while bucket and now - bucket[0] > self.window_seconds:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                retry_after = max(0, self.window_seconds - (now - bucket[0]))
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please wait before retrying.",
                    headers={"Retry-After": f"{int(retry_after)}"},
                )

            bucket.append(now)


CHAT_RATE_LIMITER = InMemoryRateLimiter(max_requests=60, window_seconds=60)
CHAT_VOICE_RATE_LIMITER = InMemoryRateLimiter(max_requests=20, window_seconds=60)
COACH_RATE_LIMITER = InMemoryRateLimiter(max_requests=25, window_seconds=60)
VISION_RATE_LIMITER = InMemoryRateLimiter(max_requests=15, window_seconds=60)
DEBUG_RATE_LIMITER = InMemoryRateLimiter(max_requests=30, window_seconds=60)


def _extract_client_key(request: Request, trust_x_forwarded_for: bool = False) -> str:
    """
    클라이언트 식별자 추출.

    운영 환경에서 직접 접근이 아닌 로드밸런서를 거칠 경우에는
    trust_x_forwarded_for=True로 설정하여 X-Forwarded-For 첫 번째 IP를 사용합니다.
    """
    if trust_x_forwarded_for:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            forwarded = xff.split(",")[0].strip()
            if forwarded:
                return forwarded

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def _dependency_factory(
    limiter: InMemoryRateLimiter, *, trust_x_forwarded_for: bool = False, name: str
):
    async def _dependency(request: Request) -> None:
        client_key = _extract_client_key(request, trust_x_forwarded_for=trust_x_forwarded_for)
        try:
            await limiter.check(client_key)
        except HTTPException as exc:
            if exc.status_code == 429:
                record_security_event(
                    "AI_RATE_LIMIT_EXCEEDED",
                    endpoint=request.url.path,
                    detail=name,
                )
            raise
    return _dependency


# 공용 엔드포인트(채팅/검색 상호작용): 공격 난이도 기준 기본값
rate_limit_chat_dependency = _dependency_factory(
    CHAT_RATE_LIMITER,
    trust_x_forwarded_for=TRUST_X_FORWARDED_FOR_FOR_RATE_LIMIT,
    name="chat"
)
rate_limit_chat_voice_dependency = _dependency_factory(
    CHAT_VOICE_RATE_LIMITER,
    trust_x_forwarded_for=TRUST_X_FORWARDED_FOR_FOR_RATE_LIMIT,
    name="chat_voice"
)

# The Coach/비전 API는 처리 비용이 크므로 stricter 규칙 적용
rate_limit_coach_dependency = _dependency_factory(
    COACH_RATE_LIMITER,
    trust_x_forwarded_for=TRUST_X_FORWARDED_FOR_FOR_RATE_LIMIT,
    name="coach"
)
rate_limit_vision_dependency = _dependency_factory(
    VISION_RATE_LIMITER,
    trust_x_forwarded_for=TRUST_X_FORWARDED_FOR_FOR_RATE_LIMIT,
    name="vision"
)

# 디버그/관리 성격 엔드포인트(검색/인제스트)는 별도 제한치
rate_limit_debug_dependency = _dependency_factory(
    DEBUG_RATE_LIMITER,
    trust_x_forwarded_for=TRUST_X_FORWARDED_FOR_FOR_RATE_LIMIT,
    name="debug"
)


# 기존 호환성: 기존 호출부가 남아 있을 때 동작 유지
rate_limit_dependency = rate_limit_chat_dependency
