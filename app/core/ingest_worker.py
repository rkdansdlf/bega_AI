"""Lease-based worker for durable AI ingestion runs."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

from scripts.ingest_from_kbo import (
    IngestExecutionResult,
    ManualBaseballDataRequiredError,
    ingest,
)

from .ingest_runs import IngestRunMode, IngestRunRecord, IngestTableResult


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

    @staticmethod
    def _default_owner() -> str:
        return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:12]}"

    async def run_once(self) -> bool:
        """Process one queued run and return whether work was claimed."""

        run = await self.store.claim_next(self.owner)
        if run is None:
            return False

        heartbeat_task = asyncio.create_task(self._heartbeat_loop(run))
        try:
            result = await self._execute(run)
        except ManualBaseballDataRequiredError as exc:
            await self.store.finish_manual_data_required(
                run.run_id,
                self.owner,
                exc.contract,
            )
        except asyncio.CancelledError:
            raise
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
        else:
            await self.store.finish_success(
                run.run_id,
                self.owner,
                result.tables,
                result.watermarks,
            )
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
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

    async def _heartbeat_loop(self, run: IngestRunRecord) -> None:
        interval = max(1.0, self.lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            if not await self.store.heartbeat(run.run_id, self.owner):
                logger.error("Ingestion run lease lost run_id=%s", run.run_id)
                return

    async def _execute(self, run: IngestRunRecord) -> IngestExecutionResult:
        table_results: dict[str, IngestTableResult] = {}
        explicit_since = self._parse_since(run.request.since)
        for source_table in run.request.tables:
            since = explicit_since
            if since is None and run.request.mode is IngestRunMode.INCREMENTAL:
                since = await self.store.get_watermark(source_table)
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
            )
            table_results.update(partial.tables)
        return IngestExecutionResult(tables=table_results)

    @staticmethod
    def _parse_since(value: datetime | str | None) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
