import asyncio
import time
from collections import defaultdict, deque
from typing import Deque, DefaultDict

from fastapi import HTTPException, Request


class InMemoryRateLimiter:
    """
    Rate Limiter 간편한 인메모리 속도 제한 장치
    API 서버의 과부하를 막는 역할 
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: DefaultDict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> None:
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
    FastAPI dependency that enforces the rate limit per client IP.
    """

    client_host = request.client.host if request.client else "unknown"
    await rate_limiter.check(client_host)
