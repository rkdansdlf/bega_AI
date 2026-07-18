from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Deque, Optional

from ..observability.metrics import (
    AI_CHAT_QUEUE_DEPTH,
    AI_CHAT_QUEUE_ESTIMATED_WAIT_SECONDS,
    AI_CHAT_QUEUE_EVENTS_TOTAL,
)

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
DisconnectChecker = Callable[[], Awaitable[bool]]


class ChatQueueFull(Exception):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("Chat queue is full.")
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class ChatQueueStatus:
    state: str
    queue_position: int
    estimated_wait_time: int
    rpm_limit: int

    def to_payload(self) -> dict[str, int | str]:
        return {
            "state": self.state,
            "queuePosition": self.queue_position,
            "estimatedWaitTime": self.estimated_wait_time,
            "rpmLimit": self.rpm_limit,
        }


class ChatQueueReservation:
    def __init__(self, queue: "InMemoryChatQueue", reservation_id: int) -> None:
        self._queue = queue
        self.reservation_id = reservation_id

    @property
    def admitted(self) -> bool:
        return self._queue.is_admitted(self.reservation_id)

    def status(self) -> ChatQueueStatus:
        return self._queue.status_for(self.reservation_id)

    async def cancel(self) -> None:
        await self._queue.cancel(self.reservation_id)

    async def release(self) -> None:
        await self._queue.release(self.reservation_id)

    async def iter_statuses(
        self,
        disconnect_checker: Optional[DisconnectChecker] = None,
    ):
        async for status in self._queue.iter_statuses(
            self.reservation_id,
            disconnect_checker=disconnect_checker,
        ):
            yield status


class InMemoryChatQueue:
    def __init__(
        self,
        *,
        rpm_limit: int,
        window_seconds: int,
        max_size: int,
        max_wait_seconds: int,
        status_interval_seconds: int,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.rpm_limit = max(1, int(rpm_limit))
        self.window_seconds = max(1, int(window_seconds))
        self.max_size = max(0, int(max_size))
        self.max_wait_seconds = max(1, int(max_wait_seconds))
        self.status_interval_seconds = max(1, int(status_interval_seconds))
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._started_at: Deque[float] = deque()
        self._waiting: Deque[int] = deque()
        self._admitted: set[int] = set()
        self._sequence = 0

    async def reserve(self) -> ChatQueueReservation:
        async with self._lock:
            now = self._clock()
            self._prune_started_locked(now)
            self._admit_ready_locked(now)

            self._sequence += 1
            reservation_id = self._sequence

            if not self._waiting and len(self._started_at) < self.rpm_limit:
                self._started_at.append(now)
                self._admitted.add(reservation_id)
                self._record_event_locked("admitted")
                self._record_depth_locked()
                return ChatQueueReservation(self, reservation_id)

            if len(self._waiting) >= self.max_size:
                retry_after = self._estimate_wait_time_locked(1, now)
                self._record_overflow_locked(retry_after)
                raise ChatQueueFull(retry_after)

            wait_seconds = self._estimate_wait_time_locked(len(self._waiting) + 1, now)
            if wait_seconds > self.max_wait_seconds:
                self._record_overflow_locked(wait_seconds)
                raise ChatQueueFull(wait_seconds)

            self._waiting.append(reservation_id)
            self._record_event_locked("queued")
            AI_CHAT_QUEUE_ESTIMATED_WAIT_SECONDS.observe(wait_seconds)
            self._record_depth_locked()
            return ChatQueueReservation(self, reservation_id)

    def is_admitted(self, reservation_id: int) -> bool:
        return reservation_id in self._admitted

    def status_for(self, reservation_id: int) -> ChatQueueStatus:
        now = self._clock()
        self._prune_started_unlocked(now)
        self._admit_ready_locked(now)
        if reservation_id in self._admitted:
            return ChatQueueStatus("processing", 0, 0, self.rpm_limit)

        try:
            position = list(self._waiting).index(reservation_id) + 1
        except ValueError:
            return ChatQueueStatus("cancelled", 0, 0, self.rpm_limit)

        return ChatQueueStatus(
            "queued",
            position,
            self._estimate_wait_time_from_snapshot(position, now),
            self.rpm_limit,
        )

    async def cancel(self, reservation_id: int) -> None:
        async with self._lock:
            self._remove_waiting_locked(reservation_id)
            self._admitted.discard(reservation_id)
            self._record_event_locked("cancelled")
            self._record_depth_locked()

    async def release(self, reservation_id: int) -> None:
        async with self._lock:
            self._admitted.discard(reservation_id)
            self._record_event_locked("released")
            self._record_depth_locked()

    async def iter_statuses(
        self,
        reservation_id: int,
        *,
        disconnect_checker: Optional[DisconnectChecker] = None,
    ):
        while True:
            if disconnect_checker is not None and await disconnect_checker():
                await self.cancel(reservation_id)
                return

            async with self._lock:
                now = self._clock()
                self._prune_started_locked(now)
                self._admit_ready_locked(now)
                status = self._status_for_locked(reservation_id, now)

            yield status
            if status.state != "queued":
                return

            await self._sleep(self.status_interval_seconds)

    def _status_for_locked(self, reservation_id: int, now: float) -> ChatQueueStatus:
        if reservation_id in self._admitted:
            return ChatQueueStatus("processing", 0, 0, self.rpm_limit)

        try:
            position = list(self._waiting).index(reservation_id) + 1
        except ValueError:
            return ChatQueueStatus("cancelled", 0, 0, self.rpm_limit)

        return ChatQueueStatus(
            "queued",
            position,
            self._estimate_wait_time_locked(position, now),
            self.rpm_limit,
        )

    def _admit_ready_locked(self, now: float) -> None:
        admitted_count = 0
        while self._waiting and len(self._started_at) < self.rpm_limit:
            reservation_id = self._waiting.popleft()
            self._started_at.append(now)
            self._admitted.add(reservation_id)
            admitted_count += 1

        if admitted_count:
            self._record_event_locked("admitted_from_queue", admitted_count)
            self._record_depth_locked()

    def _remove_waiting_locked(self, reservation_id: int) -> None:
        self._waiting = deque(
            current for current in self._waiting if current != reservation_id
        )

    def _prune_started_unlocked(self, now: float) -> None:
        while self._started_at and now - self._started_at[0] >= self.window_seconds:
            self._started_at.popleft()

    def _prune_started_locked(self, now: float) -> None:
        self._prune_started_unlocked(now)

    def _estimate_wait_time_locked(self, position: int, now: float) -> int:
        return self._estimate_wait_time_from_snapshot(position, now)

    def _estimate_wait_time_from_snapshot(self, position: int, now: float) -> int:
        starts = deque(self._started_at)
        candidate_start = now

        for _ in range(max(1, position)):
            while starts and candidate_start - starts[0] >= self.window_seconds:
                starts.popleft()
            if len(starts) >= self.rpm_limit:
                candidate_start = starts[0] + self.window_seconds
                while starts and candidate_start - starts[0] >= self.window_seconds:
                    starts.popleft()
            starts.append(candidate_start)

        return max(0, int(math.ceil(candidate_start - now)))

    def _record_event_locked(self, event: str, amount: int = 1) -> None:
        AI_CHAT_QUEUE_EVENTS_TOTAL.labels(event=event).inc(amount)

    def _record_overflow_locked(self, retry_after_seconds: int) -> None:
        self._record_event_locked("overflow")
        AI_CHAT_QUEUE_ESTIMATED_WAIT_SECONDS.observe(retry_after_seconds)
        self._record_depth_locked()

    def _record_depth_locked(self) -> None:
        AI_CHAT_QUEUE_DEPTH.labels(state="waiting").set(len(self._waiting))
        AI_CHAT_QUEUE_DEPTH.labels(state="admitted").set(len(self._admitted))


_chat_queue: Optional[InMemoryChatQueue] = None
_chat_queue_signature: tuple[int, int, int, int, int] | None = None


def get_chat_queue(settings) -> InMemoryChatQueue:
    global _chat_queue, _chat_queue_signature

    signature = (
        int(getattr(settings, "chat_queue_rpm_limit", 18)),
        int(getattr(settings, "chat_queue_window_seconds", 60)),
        int(getattr(settings, "chat_queue_max_size", 60)),
        int(getattr(settings, "chat_queue_max_wait_seconds", 180)),
        int(getattr(settings, "chat_queue_status_interval_seconds", 1)),
    )

    if _chat_queue is None or _chat_queue_signature != signature:
        _chat_queue = InMemoryChatQueue(
            rpm_limit=signature[0],
            window_seconds=signature[1],
            max_size=signature[2],
            max_wait_seconds=signature[3],
            status_interval_seconds=signature[4],
        )
        _chat_queue_signature = signature

    return _chat_queue
