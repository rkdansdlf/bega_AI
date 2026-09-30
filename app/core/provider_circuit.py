"""Provider-wide circuit breaker (CLOSED → OPEN → HALF_OPEN).

Per-request retry/fallback already exists; this adds *outage memory*: once a
provider's recent failure rate crosses a threshold we stop sending it traffic
for a cooldown instead of failing the same way on every request.

Only availability failures count (timeouts, connection errors, HTTP 429/5xx).
Client errors such as 400/401/403/404 are request/config problems, and tripping
the breaker on them would hide a bad key behind a "provider unhealthy" state.
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum
from threading import Lock
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import httpx


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


def is_availability_failure(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, TimeoutError)):
        return True
    return False


class ProviderCircuit:
    def __init__(
        self,
        name: str,
        *,
        window_seconds: float = 60.0,
        min_requests: int = 5,
        failure_rate_threshold: float = 0.5,
        cooldown_seconds: float = 30.0,
        half_open_max_probes: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._window = float(window_seconds)
        self._min_requests = max(1, int(min_requests))
        self._threshold = float(failure_rate_threshold)
        self._cooldown = float(cooldown_seconds)
        self._max_probes = max(1, int(half_open_max_probes))
        self._clock = clock
        self._events: Deque[Tuple[float, bool]] = deque()  # (t, ok)
        self._state = CircuitState.CLOSED
        self._opened_at = 0.0
        self._probes = 0
        self._lock = Lock()

    # -- internals (lock held) -------------------------------------------
    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _failure_rate(self) -> Tuple[int, float]:
        total = len(self._events)
        if not total:
            return 0, 0.0
        failures = sum(1 for _, ok in self._events if not ok)
        return total, failures / total

    def _refresh_state(self, now: float) -> None:
        if self._state is CircuitState.OPEN and now - self._opened_at >= self._cooldown:
            self._state = CircuitState.HALF_OPEN
            self._probes = 0

    # -- public ----------------------------------------------------------
    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._refresh_state(self._clock())
            return self._state

    def allow_request(self) -> bool:
        with self._lock:
            now = self._clock()
            self._refresh_state(now)
            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.OPEN:
                return False
            if self._probes < self._max_probes:
                self._probes += 1
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            now = self._clock()
            if self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._events.clear()
            self._events.append((now, True))
            self._prune(now)

    def record_failure(self, exc: Optional[BaseException] = None) -> None:
        """Record a failure; non-availability errors (4xx) are ignored."""
        if exc is not None and not is_availability_failure(exc):
            return
        with self._lock:
            now = self._clock()
            if self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = now
                return
            self._events.append((now, False))
            self._prune(now)
            total, rate = self._failure_rate()
            if total >= self._min_requests and rate >= self._threshold:
                self._state = CircuitState.OPEN
                self._opened_at = now

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = self._clock()
            self._refresh_state(now)
            self._prune(now)
            total, rate = self._failure_rate()
            return {
                "provider": self.name,
                "state": self._state.value,
                "window_requests": total,
                "failure_rate": round(rate, 4),
            }
