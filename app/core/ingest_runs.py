"""Durable AI ingestion run domain types and state transitions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID


class IngestRunMode(str, Enum):
    """Supported ingestion execution modes."""

    FULL = "FULL"
    INCREMENTAL = "INCREMENTAL"


class IngestRunStatus(str, Enum):
    """Persisted ingestion run states."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    MANUAL_BASEBALL_DATA_REQUIRED = "MANUAL_BASEBALL_DATA_REQUIRED"


def _normalize_iso_timestamp(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    normalized = value.strip()
    if not normalized:
        return None
    parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    return parsed.isoformat()


@dataclass(frozen=True)
class IngestRunRequest:
    """Normalized work request persisted for a durable ingestion run."""

    tables: tuple[str, ...]
    season_year: int | None = None
    mode: IngestRunMode | str = IngestRunMode.INCREMENTAL
    trigger_source: str = "MANUAL_API"
    since: datetime | str | None = None

    def normalized(self) -> IngestRunRequest:
        tables = tuple(
            sorted({table.strip().lower() for table in self.tables if table.strip()})
        )
        if not tables:
            raise ValueError("at least one ingestion table is required")
        if "rag_chunks" in tables:
            raise ValueError("rag_chunks cannot be used as an ingestion source")

        mode = (
            self.mode
            if isinstance(self.mode, IngestRunMode)
            else IngestRunMode(str(self.mode).strip().upper())
        )
        trigger_source = self.trigger_source.strip().upper()
        if not trigger_source:
            raise ValueError("trigger_source is required")
        if self.season_year is not None and self.season_year < 1:
            raise ValueError("season_year must be positive")

        return IngestRunRequest(
            tables=tables,
            season_year=self.season_year,
            mode=mode,
            trigger_source=trigger_source,
            since=_normalize_iso_timestamp(self.since),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical work identity without submission metadata."""

        normalized = self.normalized()
        return {
            "mode": normalized.mode.value,
            "season_year": normalized.season_year,
            "since": normalized.since,
            "tables": list(normalized.tables),
        }


@dataclass(frozen=True)
class IngestTableResult:
    """Sanitized counts and watermark produced for one source table."""

    source_table: str
    written_chunks: int
    source_rows: int
    reused_embeddings: int
    embedded_chunks: int
    max_updated_at: datetime | None = None


@dataclass(frozen=True)
class IngestRunRecord:
    """Persisted ingestion run state used by the API and worker."""

    run_id: UUID
    request_key: str
    request: IngestRunRequest
    status: IngestRunStatus
    requested_at: datetime
    started_at: datetime | None = None
    heartbeat_at: datetime | None = None
    finished_at: datetime | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    recovery_attempts: int = 0
    error_code: str | None = None
    error_message: str | None = None
    table_summary: Mapping[str, Any] = field(default_factory=dict)


def build_request_key(request: IngestRunRequest) -> str:
    """Hash a canonical request identity for active-run deduplication."""

    payload = json.dumps(
        request.normalized().to_payload(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_ALLOWED_TRANSITIONS: Mapping[IngestRunStatus, frozenset[IngestRunStatus]] = {
    IngestRunStatus.QUEUED: frozenset({IngestRunStatus.RUNNING}),
    IngestRunStatus.RUNNING: frozenset(
        {
            IngestRunStatus.QUEUED,
            IngestRunStatus.SUCCEEDED,
            IngestRunStatus.FAILED,
            IngestRunStatus.MANUAL_BASEBALL_DATA_REQUIRED,
        }
    ),
    IngestRunStatus.SUCCEEDED: frozenset(),
    IngestRunStatus.FAILED: frozenset(),
    IngestRunStatus.MANUAL_BASEBALL_DATA_REQUIRED: frozenset(),
}


def ensure_transition(
    current: IngestRunStatus,
    target: IngestRunStatus,
) -> None:
    """Raise when a persisted run transition is not legal."""

    if target not in _ALLOWED_TRANSITIONS[current]:
        raise ValueError(
            f"illegal ingest run transition: {current.value} -> {target.value}"
        )
