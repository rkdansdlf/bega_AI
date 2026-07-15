"""Lease-based worker for durable AI ingestion runs."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from datetime import datetime
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
    AI_INGEST_RUN_COMPLETIONS_TOTAL,
    AI_INGEST_RUN_DURATION_SECONDS,
    AI_INGEST_TABLE_WRITTEN_CHUNKS_TOTAL,
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
        self.recovery_seconds = max(1.0, self.lease_seconds / 3)

    @staticmethod
    def _default_owner() -> str:
        return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:12]}"

    async def run_once(self) -> bool:
        """Process one queued run and return whether work was claimed."""

        run = await self.store.claim_next(self.owner)
        if run is None:
            return False

        trigger_source = normalize_ingest_trigger_source(run.request.trigger_source)
        started_at = perf_counter()
        terminal_status = None
        result = None
        AI_INGEST_ACTIVE_RUNS.labels(trigger_source=trigger_source).inc()
        lease_lost = asyncio.Event()
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(run, lease_lost))
        try:
            result = await self._execute(run, lease_lost)
            if lease_lost.is_set():
                raise IngestLeaseLostError
        except ManualBaseballDataRequiredError as exc:
            await self.store.finish_manual_data_required(
                run.run_id,
                self.owner,
                exc.contract,
            )
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
            await self.store.finish_failed(
                run.run_id,
                self.owner,
                error_code="INGEST_EXECUTION_FAILED",
                error_message=error_type,
            )
            terminal_status = IngestRunStatus.FAILED
        else:
            await self.store.finish_success(
                run.run_id,
                self.owner,
                result.tables,
                result.watermarks,
                build_watermark_scope_key(run.request),
            )
            terminal_status = IngestRunStatus.SUCCEEDED
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            AI_INGEST_ACTIVE_RUNS.labels(trigger_source=trigger_source).dec()
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
            if result is not None:
                configured_tables = set(DEFAULT_TABLES)
                for source_table, table_result in result.tables.items():
                    table_label = (
                        source_table if source_table in configured_tables else "other"
                    )
                    AI_INGEST_TABLE_WRITTEN_CHUNKS_TOTAL.labels(
                        source_table=table_label
                    ).inc(table_result.written_chunks)
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
                recovered, failed = await self.store.recover_expired()
                if recovered or failed:
                    logger.warning(
                        "Expired ingestion leases processed recovered=%d failed=%d",
                        recovered,
                        failed,
                    )
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

    async def _heartbeat_loop(
        self,
        run: IngestRunRecord,
        lease_lost: asyncio.Event,
    ) -> None:
        interval = max(1.0, self.lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            try:
                owned = await self.store.heartbeat(run.run_id, self.owner)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Ingestion run heartbeat failed run_id=%s error_type=%s",
                    run.run_id,
                    type(exc).__name__,
                )
                lease_lost.set()
                return
            if not owned:
                logger.error("Ingestion run lease lost run_id=%s", run.run_id)
                lease_lost.set()
                return

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
            if lease_lost is not None and lease_lost.is_set():
                raise IngestLeaseLostError
            table_results.update(partial.tables)
        return IngestExecutionResult(tables=table_results)

    @staticmethod
    def _parse_since(value: datetime | str | None) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
