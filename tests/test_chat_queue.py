from __future__ import annotations

import asyncio

import pytest

from app.core.chat_queue import ChatQueueFull, InMemoryChatQueue


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


async def _next_status(reservation):
    async for status in reservation.iter_statuses():
        return status
    raise AssertionError("expected queue status")


@pytest.mark.asyncio
async def test_first_rpm_limit_requests_are_admitted_immediately() -> None:
    clock = FakeClock()
    queue = InMemoryChatQueue(
        rpm_limit=18,
        window_seconds=60,
        max_size=60,
        max_wait_seconds=180,
        status_interval_seconds=1,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )

    reservations = [await queue.reserve() for _ in range(18)]

    assert all(reservation.admitted for reservation in reservations)
    assert [reservation.status().state for reservation in reservations] == [
        "processing"
    ] * 18


@pytest.mark.asyncio
async def test_request_over_rpm_limit_waits_with_position_and_estimate() -> None:
    clock = FakeClock()
    queue = InMemoryChatQueue(
        rpm_limit=18,
        window_seconds=60,
        max_size=60,
        max_wait_seconds=180,
        status_interval_seconds=1,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    for _ in range(18):
        await queue.reserve()

    reservation = await queue.reserve()

    assert reservation.admitted is False
    assert reservation.status().to_payload() == {
        "state": "queued",
        "queuePosition": 1,
        "estimatedWaitTime": 60,
        "rpmLimit": 18,
    }


@pytest.mark.asyncio
async def test_queue_admits_waiters_in_fifo_order() -> None:
    clock = FakeClock()
    queue = InMemoryChatQueue(
        rpm_limit=1,
        window_seconds=60,
        max_size=10,
        max_wait_seconds=180,
        status_interval_seconds=60,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    first = await queue.reserve()
    second = await queue.reserve()
    third = await queue.reserve()

    assert first.admitted is True
    assert second.status().queue_position == 1
    assert third.status().queue_position == 2

    second_updates = second.iter_statuses()
    assert (await anext(second_updates)).state == "queued"
    assert (await anext(second_updates)).state == "processing"

    assert second.admitted is True
    assert third.admitted is False
    assert third.status().queue_position == 1


@pytest.mark.asyncio
async def test_cancelled_waiter_is_removed_from_queue() -> None:
    clock = FakeClock()
    queue = InMemoryChatQueue(
        rpm_limit=1,
        window_seconds=60,
        max_size=10,
        max_wait_seconds=180,
        status_interval_seconds=1,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    await queue.reserve()
    second = await queue.reserve()
    third = await queue.reserve()

    await second.cancel()

    assert third.status().queue_position == 1


@pytest.mark.asyncio
async def test_cancelled_waiter_reports_terminal_status() -> None:
    clock = FakeClock()
    queue = InMemoryChatQueue(
        rpm_limit=1,
        window_seconds=60,
        max_size=10,
        max_wait_seconds=180,
        status_interval_seconds=1,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    await queue.reserve()
    second = await queue.reserve()

    await second.cancel()

    assert second.status().to_payload() == {
        "state": "cancelled",
        "queuePosition": 0,
        "estimatedWaitTime": 0,
        "rpmLimit": 1,
    }

    updates = second.iter_statuses()
    assert (await anext(updates)).state == "cancelled"
    with pytest.raises(StopAsyncIteration):
        await anext(updates)


@pytest.mark.asyncio
async def test_status_poll_admits_ready_waiter_after_window() -> None:
    clock = FakeClock()
    queue = InMemoryChatQueue(
        rpm_limit=1,
        window_seconds=60,
        max_size=10,
        max_wait_seconds=180,
        status_interval_seconds=1,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    await queue.reserve()
    second = await queue.reserve()

    clock.now = 60

    assert second.status().state == "processing"
    assert second.admitted is True


@pytest.mark.asyncio
async def test_queue_max_size_overflow_raises_retryable_error() -> None:
    clock = FakeClock()
    queue = InMemoryChatQueue(
        rpm_limit=1,
        window_seconds=60,
        max_size=1,
        max_wait_seconds=180,
        status_interval_seconds=1,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    await queue.reserve()
    await queue.reserve()

    with pytest.raises(ChatQueueFull) as exc_info:
        await queue.reserve()

    assert exc_info.value.retry_after_seconds == 60


@pytest.mark.asyncio
async def test_concurrent_load_obeys_rpm_window_and_fifo_batches() -> None:
    clock = FakeClock()
    queue = InMemoryChatQueue(
        rpm_limit=18,
        window_seconds=60,
        max_size=60,
        max_wait_seconds=180,
        status_interval_seconds=1,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )

    reservations = await asyncio.gather(*(queue.reserve() for _ in range(40)))
    immediate = [reservation for reservation in reservations if reservation.admitted]
    queued = [reservation for reservation in reservations if not reservation.admitted]

    assert len(immediate) == 18
    assert len(queued) == 22
    assert [reservation.status().queue_position for reservation in queued] == list(
        range(1, 23)
    )
    assert [
        reservation.status().estimated_wait_time for reservation in queued[:18]
    ] == [60] * 18
    assert [
        reservation.status().estimated_wait_time for reservation in queued[18:]
    ] == [120] * 4

    clock.now = 60
    first_waiter_batch = [await _next_status(reservation) for reservation in queued]

    assert [status.state for status in first_waiter_batch[:18]] == ["processing"] * 18
    assert [status.queue_position for status in first_waiter_batch[18:]] == [1, 2, 3, 4]
    assert [status.estimated_wait_time for status in first_waiter_batch[18:]] == [
        60
    ] * 4

    clock.now = 120
    final_waiter_batch = [
        await _next_status(reservation) for reservation in queued[18:]
    ]

    assert [status.state for status in final_waiter_batch] == ["processing"] * 4
