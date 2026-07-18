"""Lease-based worker for durable AI ingestion runs."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from scripts.ingest_from_kbo import (
    DEFAULT_TABLES,
    IngestExecutionResult,
    ManualBaseballDataRequiredError,
    ingest,
)

from .ingest_runs import (
    IngestLeaseLostError,
    IngestRunMode,
    IngestRunRecord,
    IngestRunStatus,
    IngestTableResult,
    build_watermark_scope_key,
)
from ..observability.metrics import (
    AI_INGEST_ACTIVE_RUNS,
    AI_INGEST_HEARTBEATS_TOTAL,
    AI_INGEST_LEASE_RECOVERIES_TOTAL,
    AI_INGEST_QUEUED_RUNS,
    AI_INGEST_RUN_COMPLETIONS_TOTAL,
    AI_INGEST_RUN_DURATION_SECONDS,
    AI_INGEST_TABLE_DURATION_SECONDS,
    AI_INGEST_TABLE_SOURCE_ROWS_TOTAL,
    AI_INGEST_TABLE_WRITTEN_CHUNKS_TOTAL,
    AI_INGEST_WATERMARK_LAG_SECONDS,
    INGEST_TRIGGER_SOURCE_LABELS,
    normalize_ingest_terminal_status,
    normalize_ingest_trigger_source,
)


logger = logging.getLogger(__name__)


class IngestWorker:
    """Claim and execute one durable ingestion run at a time."""

    def __init__(
        self,
        *,
        store: Any,
        settings: Any,
        owner: str | None = None,
        ingest_function: Callable[..., IngestExecutionResult] = ingest,
    ) -> None:
        self.store = store
        self.settings = settings
        self.owner = owner or self._default_owner()
        self.ingest_function = ingest_function
        self.poll_seconds = max(
            0.05, float(getattr(settings, "ingest_worker_poll_seconds", 2.0))
        )
        self.lease_seconds = max(
            3, int(getattr(settings, "ingest_worker_lease_seconds", 120))
        )
        self.heartbeat_interval_seconds = max(1.0, self.lease_seconds / 3)
        self.heartbeat_safety_margin_seconds = min(
            5.0,
            max(0.25, self.lease_seconds / 6),
        )
        self.heartbeat_retry_initial_seconds = min(
            1.0,
            self.heartbeat_interval_seconds,
        )
        self.recovery_seconds = max(1.0, self.lease_seconds / 3)

    @staticmethod
    def _default_owner() -> str:
        return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:12]}"

    async def run_once(self) -> bool:
        """Process one queued run and return whether work was claimed."""

        await self._refresh_run_gauges()
        run = await self.store.claim_next(self.owner)
        if run is None:
            return False
        await self._refresh_run_gauges()

        trigger_source = normalize_ingest_trigger_source(run.request.trigger_source)
        started_at = perf_counter()
        terminal_status = None
        result = None
        lease_lost = asyncio.Event()
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(run, lease_lost))
        try:
            result = await self._execute(run, lease_lost)
            if lease_lost.is_set():
                raise IngestLeaseLostError
        except ManualBaseballDataRequiredError as exc:
            try:
                await self.store.finish_manual_data_required(
                    run.run_id,
                    self.owner,
                    exc.contract,
                )
            except IngestLeaseLostError:
                logger.error(
                    "Ingestion manual terminal skipped after lease loss run_id=%s",
                    run.run_id,
                )
            else:
                terminal_status = IngestRunStatus.MANUAL_BASEBALL_DATA_REQUIRED
        except asyncio.CancelledError:
            raise
        except IngestLeaseLostError:
            logger.error("Ingestion run stopped after lease loss run_id=%s", run.run_id)
        except Exception as exc:  # noqa: BLE001
            error_type = type(exc).__name__
            logger.error(
                "Ingestion run failed run_id=%s error_type=%s",
                run.run_id,
                error_type,
            )
            if lease_lost.is_set():
                logger.error(
                    "Ingestion failure terminal skipped after lease loss run_id=%s",
                    run.run_id,
                )
            else:
                try:
                    await self.store.finish_failed(
                        run.run_id,
                        self.owner,
                        error_code="INGEST_EXECUTION_FAILED",
                        error_message=error_type,
                    )
                except IngestLeaseLostError:
                    logger.error(
                        "Ingestion failure terminal rejected after lease loss run_id=%s",
                        run.run_id,
                    )
                else:
                    terminal_status = IngestRunStatus.FAILED
        else:
            try:
                await self.store.finish_success(
                    run.run_id,
                    self.owner,
                    result.tables,
                    result.watermarks,
                    build_watermark_scope_key(run.request),
                )
            except IngestLeaseLostError:
                logger.error(
                    "Ingestion success terminal skipped after lease loss run_id=%s",
                    run.run_id,
                )
            else:
                terminal_status = IngestRunStatus.SUCCEEDED
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            if terminal_status is not None:
                status_label = normalize_ingest_terminal_status(terminal_status)
                AI_INGEST_RUN_COMPLETIONS_TOTAL.labels(
                    status=status_label,
                    trigger_source=trigger_source,
                ).inc()
                AI_INGEST_RUN_DURATION_SECONDS.labels(
                    status=status_label,
                    trigger_source=trigger_source,
                ).observe(max(0.0, perf_counter() - started_at))
            await self._refresh_run_gauges()
        return True

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        """Poll the queue until shutdown is requested."""

        while not stop_event.is_set():
            try:
                processed = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Ingestion worker cycle failed error_type=%s",
                    type(exc).__name__,
                )
                processed = False
            if processed:
                continue
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass

    async def run_recovery_forever(self, stop_event: asyncio.Event) -> None:
        """Recover expired leases periodically, including after a warm restart."""

        while not stop_event.is_set():
            try:
                await self.recover_expired_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Ingestion lease recovery failed error_type=%s",
                    type(exc).__name__,
                )
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.recovery_seconds,
                )
            except TimeoutError:
                pass

    async def recover_expired_once(self) -> tuple[int, int]:
        """Recover expired leases and record startup or periodic outcomes once."""

        recovered, failed = await self.store.recover_expired()
        if recovered:
            AI_INGEST_LEASE_RECOVERIES_TOTAL.labels(result="requeued").inc(recovered)
        if failed:
            AI_INGEST_LEASE_RECOVERIES_TOTAL.labels(result="failed").inc(failed)
        if recovered or failed:
            logger.warning(
                "Expired ingestion leases processed recovered=%d failed=%d",
                recovered,
                failed,
            )
        await self._refresh_run_gauges()
        return recovered, failed

    async def _refresh_run_gauges(self) -> None:
        """Reconcile queue and watermark gauges from durable state."""

        try:
            counts = await self.store.count_active_by_status()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Ingestion run gauge reconciliation failed error_type=%s",
                type(exc).__name__,
            )
        else:
            queued_counts = {label: 0 for label in INGEST_TRIGGER_SOURCE_LABELS}
            active_counts = {label: 0 for label in INGEST_TRIGGER_SOURCE_LABELS}
            for (status, raw_trigger_source), count in counts.items():
                trigger_source = normalize_ingest_trigger_source(raw_trigger_source)
                if str(status).upper() == IngestRunStatus.QUEUED.value:
                    queued_counts[trigger_source] += max(0, int(count))
                elif str(status).upper() == IngestRunStatus.RUNNING.value:
                    active_counts[trigger_source] += max(0, int(count))

            for trigger_source in INGEST_TRIGGER_SOURCE_LABELS:
                AI_INGEST_QUEUED_RUNS.labels(trigger_source=trigger_source).set(
                    queued_counts[trigger_source]
                )
                AI_INGEST_ACTIVE_RUNS.labels(trigger_source=trigger_source).set(
                    active_counts[trigger_source]
                )

        try:
            watermarks = await self.store.get_latest_watermarks_by_table()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Ingestion watermark gauge reconciliation failed error_type=%s",
                type(exc).__name__,
            )
        else:
            latest_by_label: dict[str, datetime] = {}
            for source_table, watermark in watermarks.items():
                table_label = self._table_label(source_table)
                current = latest_by_label.get(table_label)
                if current is None or watermark > current:
                    latest_by_label[table_label] = watermark
            for table_label, watermark in latest_by_label.items():
                AI_INGEST_WATERMARK_LAG_SECONDS.labels(
                    source_table=table_label
                ).set(self._watermark_lag_seconds(watermark))

    async def _heartbeat_loop(
        self,
        run: IngestRunRecord,
        lease_lost: asyncio.Event,
    ) -> None:
        loop = asyncio.get_running_loop()
        last_confirmed = loop.time()
        interval = max(0.001, float(self.heartbeat_interval_seconds))
        retry_initial = max(0.001, float(self.heartbeat_retry_initial_seconds))
        safety_margin = max(
            0.0,
            min(float(self.heartbeat_safety_margin_seconds), self.lease_seconds / 2),
        )

        while True:
            await asyncio.sleep(interval)
            retry_delay = retry_initial
            attempt = 0
            while True:
                attempt += 1
                try:
                    lease_expires_at = await self.store.heartbeat(
                        run.run_id,
                        self.owner,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    remaining = (
                        last_confirmed
                        + float(self.lease_seconds)
                        - safety_margin
                        - loop.time()
                    )
                    if remaining <= 0:
                        AI_INGEST_HEARTBEATS_TOTAL.labels(result="exhausted").inc()
                        logger.error(
                            "Ingestion heartbeat retry budget exhausted "
                            "run_id=%s attempts=%d error_type=%s",
                            run.run_id,
                            attempt,
                            type(exc).__name__,
                        )
                        lease_lost.set()
                        return
                    AI_INGEST_HEARTBEATS_TOTAL.labels(result="retry").inc()
                    logger.warning(
                        "Ingestion heartbeat retry scheduled "
                        "run_id=%s attempt=%d remaining_seconds=%.3f error_type=%s",
                        run.run_id,
                        attempt,
                        remaining,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(min(retry_delay, remaining))
                    retry_delay = min(
                        retry_delay * 2,
                        max(retry_initial, interval),
                    )
                    continue

                if lease_expires_at is None:
                    AI_INGEST_HEARTBEATS_TOTAL.labels(result="rejected").inc()
                    logger.error(
                        "Ingestion run lease rejected run_id=%s",
                        run.run_id,
                    )
                    lease_lost.set()
                    return

                AI_INGEST_HEARTBEATS_TOTAL.labels(result="success").inc()
                last_confirmed = loop.time()
                break

    async def _execute(
        self,
        run: IngestRunRecord,
        lease_lost: asyncio.Event | None = None,
    ) -> IngestExecutionResult:
        table_results: dict[str, IngestTableResult] = {}
        explicit_since = self._parse_since(run.request.since)
        scope_key = build_watermark_scope_key(run.request)
        for source_table in run.request.tables:
            if lease_lost is not None and lease_lost.is_set():
                raise IngestLeaseLostError
            since = explicit_since
            if since is None and run.request.mode is IngestRunMode.INCREMENTAL:
                since = await self.store.get_watermark(source_table, scope_key)
            table_label = self._table_label(source_table)
            table_started_at = perf_counter()
            try:
                partial = await asyncio.to_thread(
                    self.ingest_function,
                    tables=[source_table],
                    source_db_url=self.settings.source_db_url,
                    limit=None,
                    embed_batch_size=max(1, int(self.settings.embed_batch_size)),
                    read_batch_size=500,
                    season_year=run.request.season_year,
                    use_legacy_renderer=False,
                    since=since,
                    skip_embedding=False,
                    max_concurrency=2,
                    commit_interval=500,
                    parallel_engine="thread",
                    workers=4,
                    row_stale_cleanup="off",
                    lease_run_id=run.run_id,
                    lease_owner=self.owner,
                )
            finally:
                AI_INGEST_TABLE_DURATION_SECONDS.labels(
                    source_table=table_label
                ).observe(max(0.0, perf_counter() - table_started_at))
            self._record_table_result_metrics(partial)
            if lease_lost is not None and lease_lost.is_set():
                raise IngestLeaseLostError
            table_results.update(partial.tables)
        return IngestExecutionResult(tables=table_results)

    @staticmethod
    def _parse_since(value: datetime | str | None) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    @staticmethod
    def _table_label(source_table: str) -> str:
        return source_table if source_table in set(DEFAULT_TABLES) else "other"

    def _record_table_result_metrics(self, result: IngestExecutionResult) -> None:
        for source_table, table_result in result.tables.items():
            table_label = self._table_label(source_table)
            AI_INGEST_TABLE_WRITTEN_CHUNKS_TOTAL.labels(
                source_table=table_label
            ).inc(table_result.written_chunks)
            AI_INGEST_TABLE_SOURCE_ROWS_TOTAL.labels(
                source_table=table_label
            ).inc(table_result.source_rows)

    @staticmethod
    def _watermark_lag_seconds(watermark: datetime) -> float:
        normalized = watermark
        if normalized.tzinfo is None:
            normalized = normalized.replace(tzinfo=UTC)
        return max(0.0, (datetime.now(UTC) - normalized.astimezone(UTC)).total_seconds())
