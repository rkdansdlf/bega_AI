import asyncio
import time
from collections import defaultdict, deque
from typing import Deque, DefaultDict

from fastapi import HTTPException, Request


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


rate_limiter = InMemoryRateLimiter(max_requests=10, window_seconds=60)


async def rate_limit_dependency(request: Request) -> None:
    """
    FastAPI 의존성으로, 클라이언트 IP 주소별로 API 요청 속도 제한을 적용합니다.
    각 요청 시 호출되어 현재 클라이언트의 요청이 허용된 속도 제한을 초과하는지 검사합니다.
    """

    client_host = request.client.host if request.client else "unknown"
    await rate_limiter.check(client_host)
