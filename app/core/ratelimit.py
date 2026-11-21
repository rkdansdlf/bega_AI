import asyncio
import ipaddress
import time
import os
from collections import defaultdict, deque
from typing import Deque, DefaultDict
from .security_metrics import record_security_event

from fastapi import HTTPException, Request

TRUST_X_FORWARDED_FOR_FOR_RATE_LIMIT = os.getenv(
    "AI_TRUST_X_FORWARDED_FOR", "false"
).strip().lower() in {"1", "true", "yes", "on"}


def _parse_trusted_proxy_count(raw: str | None) -> int:
    """AI_TRUSTED_PROXY_COUNT 를 1 이상의 정수로 읽는다(잘못된 값은 1)."""
    try:
        return max(1, int((raw or "").strip()))
    except ValueError:
        return 1


# X-Forwarded-For 를 신뢰할 때, 요청 경로에서 값을 덧붙이는 신뢰 프록시의 수.
# 클라이언트 IP 는 XFF 의 오른쪽에서 N번째 값이다.
TRUSTED_PROXY_COUNT = _parse_trusted_proxy_count(os.getenv("AI_TRUSTED_PROXY_COUNT"))


# 추적하는 서로 다른 키의 상한. 키가 클라이언트 제어 값(예: 위조된 X-Forwarded-For)이면
# 키 종류가 무한히 늘 수 있으므로 메모리를 묶어 두기 위한 안전장치다.
DEFAULT_MAX_TRACKED_KEYS = 10_000
# 상한을 넘은 새 키가 함께 쓰는 공유 버킷. 키를 계속 바꿔도 제한을 피할 수 없다.
OVERFLOW_KEY = "__rate_limit_overflow__"
# 상한 도달 시 긴급 정리를 시도하는 최소 간격(초). 폭주 중 요청마다 O(n) 순회를 막는다.
EMERGENCY_SWEEP_MIN_INTERVAL_SECONDS = 1.0


class InMemoryRateLimiter:
    """
    InMemoryRateLimiter는 API 서버의 과부하를 방지하기 위한 인메모리 속도 제한(Rate Limiter) 장치입니다.
    지정된 시간(window_seconds) 내에 특정 키(예: 클라이언트 IP)에 대한 요청 수를 제한합니다.

    메모리 보호:
    - 윈도우 안에 더 이상 요청이 없는 만료 키는 주기적으로 제거합니다.
    - 서로 다른 키가 max_keys 를 넘으면 새 키는 공유 overflow 버킷으로 합쳐져 같은
      제한을 받습니다(키를 바꿔 제한을 피하거나 메모리를 늘리는 것을 막음).
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        max_keys: int = DEFAULT_MAX_TRACKED_KEYS,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._hits: DefaultDict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._next_sweep_at = 0.0
        self._last_emergency_sweep_at = 0.0

    @property
    def tracked_keys(self) -> int:
        """현재 추적 중인 서로 다른 키의 수."""
        return len(self._hits)

    def _sweep_expired(self, now: float) -> None:
        """윈도우 안에 요청이 없는 키(가장 최근 요청도 만료)를 제거한다."""
        expired = [
            key
            for key, bucket in self._hits.items()
            if not bucket or now - bucket[-1] > self.window_seconds
        ]
        for key in expired:
            del self._hits[key]

    def _resolve_key(self, key: str, now: float) -> str:
        """정리와 키 수 상한을 적용해 실제로 사용할 키를 돌려준다."""
        if now >= self._next_sweep_at:
            self._sweep_expired(now)
            self._next_sweep_at = now + self.window_seconds

        if key in self._hits or len(self._hits) < self.max_keys:
            return key

        if now - self._last_emergency_sweep_at >= EMERGENCY_SWEEP_MIN_INTERVAL_SECONDS:
            self._sweep_expired(now)
            self._last_emergency_sweep_at = now
            if len(self._hits) < self.max_keys:
                return key

        return OVERFLOW_KEY

    async def check(self, key: str) -> None:
        """
        주어진 키(key)에 대해 속도 제한을 확인하고 적용합니다.
        요청이 허용된도를 초과하면 HTTPException(429 Too Many Requests)을 발생시킵니다.
        """
        now = time.time()
        async with self._lock:
            key = self._resolve_key(key, now)
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


CHAT_RATE_LIMITER = InMemoryRateLimiter(max_requests=1000, window_seconds=60)
CHAT_VOICE_RATE_LIMITER = InMemoryRateLimiter(max_requests=20, window_seconds=60)
COACH_RATE_LIMITER = InMemoryRateLimiter(max_requests=25, window_seconds=60)
VISION_RATE_LIMITER = InMemoryRateLimiter(max_requests=15, window_seconds=60)
DEBUG_RATE_LIMITER = InMemoryRateLimiter(max_requests=30, window_seconds=60)


def _client_ip_from_forwarded_for(request: Request, trusted_proxy_count: int) -> str | None:
    """X-Forwarded-For 에서 신뢰 프록시가 덧붙인 클라이언트 IP 를 고른다.

    프록시는 자신이 본 연결 주소를 XFF 의 오른쪽 끝에 덧붙이므로, 오른쪽에서
    trusted_proxy_count 번째 값만 신뢰할 수 있다. 그보다 왼쪽 값은 클라이언트가
    임의로 보낼 수 있어(위조) 사용하지 않는다.

    다음 경우에는 None 을 돌려주어 호출자가 연결 주소로 되돌아가게 한다.
    - 헤더가 없거나, 값이 신뢰 프록시 수보다 적어 예상한 프록시 체인이 아닌 경우
    - 고른 값이 올바른 IP 주소가 아닌 경우(키가 임의 문자열이 되는 것을 막음)
    """
    values = request.headers.getlist("x-forwarded-for")
    entries = [
        part.strip() for value in values for part in value.split(",") if part.strip()
    ]
    if len(entries) < trusted_proxy_count:
        return None
    try:
        return str(ipaddress.ip_address(entries[-trusted_proxy_count]))
    except ValueError:
        return None


def _extract_client_key(
    request: Request,
    trust_x_forwarded_for: bool = False,
    trusted_proxy_count: int = 1,
) -> str:
    """
    클라이언트 식별자 추출.

    기본값에서는 연결 주소(peer)를 사용하며 X-Forwarded-For 는 무시합니다.
    운영 환경에서 직접 접근이 아닌 프록시/로드밸런서를 거칠 경우에는
    trust_x_forwarded_for=True 로 설정하고 trusted_proxy_count(신뢰 프록시 수)를
    맞추면, XFF 를 오른쪽에서 trusted_proxy_count 번째 값으로 읽습니다.

    주의: 이 설정은 AI 서비스에 프록시를 통해서만 접근할 수 있다는 네트워크 구성을
    전제로 합니다. 프록시를 거치지 않는 직접 접속자의 XFF 는 검증하지 않습니다.
    """
    if trust_x_forwarded_for:
        forwarded = _client_ip_from_forwarded_for(request, trusted_proxy_count)
        if forwarded:
            return forwarded

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def _dependency_factory(
    limiter: InMemoryRateLimiter,
    *,
    trust_x_forwarded_for: bool = False,
    trusted_proxy_count: int = TRUSTED_PROXY_COUNT,
    name: str,
):
    async def _dependency(request: Request) -> None:
        client_key = _extract_client_key(
            request,
            trust_x_forwarded_for=trust_x_forwarded_for,
            trusted_proxy_count=trusted_proxy_count,
        )
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
    name="chat",
)
rate_limit_chat_voice_dependency = _dependency_factory(
    CHAT_VOICE_RATE_LIMITER,
    trust_x_forwarded_for=TRUST_X_FORWARDED_FOR_FOR_RATE_LIMIT,
    name="chat_voice",
)

# The Coach/비전 API는 처리 비용이 크므로 stricter 규칙 적용
rate_limit_coach_dependency = _dependency_factory(
    COACH_RATE_LIMITER,
    trust_x_forwarded_for=TRUST_X_FORWARDED_FOR_FOR_RATE_LIMIT,
    name="coach",
)
rate_limit_vision_dependency = _dependency_factory(
    VISION_RATE_LIMITER,
    trust_x_forwarded_for=TRUST_X_FORWARDED_FOR_FOR_RATE_LIMIT,
    name="vision",
)

# 디버그/관리 성격 엔드포인트(검색/인제스트)는 별도 제한치
rate_limit_debug_dependency = _dependency_factory(
    DEBUG_RATE_LIMITER,
    trust_x_forwarded_for=TRUST_X_FORWARDED_FOR_FOR_RATE_LIMIT,
    name="debug",
)


# 기존 호환성: 기존 호출부가 남아 있을 때 동작 유지
rate_limit_dependency = rate_limit_chat_dependency
