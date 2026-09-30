"""Small process-local circuit breaker for synchronous model integrations."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic


@dataclass(frozen=True)
class CircuitSnapshot:
    ready: bool
    consecutive_failures: int


class ModelCircuitBreaker:
    def __init__(self, *, failure_threshold: int = 3, cooldown_seconds: float = 30.0):
        self._failure_threshold = max(1, int(failure_threshold))
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._lock = Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if monotonic() - self._opened_at < self._cooldown_seconds:
                return False
            self._opened_at = None
            self._consecutive_failures = self._failure_threshold - 1
            return True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._opened_at = monotonic()

    def snapshot(self) -> CircuitSnapshot:
        with self._lock:
            ready = self._opened_at is None or (
                monotonic() - self._opened_at >= self._cooldown_seconds
            )
            return CircuitSnapshot(
                ready=ready,
                consecutive_failures=self._consecutive_failures,
            )


release_decision_circuit = ModelCircuitBreaker()
