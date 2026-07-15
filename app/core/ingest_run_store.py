"""PostgreSQL persistence for durable AI ingestion runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from .ingest_runs import (
    IngestRunRecord,
    IngestRunRequest,
    IngestRunStatus,
    IngestTableResult,
    build_request_key,
)


_RUN_COLUMNS = (
    "run_id",
    "request_key",
    "trigger_source",
    "status",
    "request_payload",
    "requested_at",
    "started_at",
    "heartbeat_at",
    "finished_at",
    "lease_owner",
    "lease_expires_at",
    "recovery_attempts",
    "error_code",
    "error_message",
    "table_summary",
)
_RUN_SELECT = ", ".join(_RUN_COLUMNS)


def _json_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _row_mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    return dict(zip(_RUN_COLUMNS, row, strict=True))


def _row_to_record(row: Any) -> IngestRunRecord:
    values = _row_mapping(row)
    payload = _json_mapping(values["request_payload"])
    request = IngestRunRequest(
        tables=tuple(payload.get("tables") or ()),
        season_year=payload.get("season_year"),
        mode=payload.get("mode", "INCREMENTAL"),
        trigger_source=str(values["trigger_source"]),
        since=payload.get("since"),
    ).normalized()
    return IngestRunRecord(
        run_id=values["run_id"],
        request_key=str(values["request_key"]),
        request=request,
        status=IngestRunStatus(str(values["status"])),
        requested_at=values["requested_at"],
        started_at=values.get("started_at"),
        heartbeat_at=values.get("heartbeat_at"),
        finished_at=values.get("finished_at"),
        lease_owner=values.get("lease_owner"),
        lease_expires_at=values.get("lease_expires_at"),
        recovery_attempts=int(values.get("recovery_attempts") or 0),
        error_code=values.get("error_code"),
        error_message=values.get("error_message"),
        table_summary=_json_mapping(values.get("table_summary")),
    )


def _table_result_payload(result: IngestTableResult) -> dict[str, Any]:
    return {
        "source_rows": result.source_rows,
        "written_chunks": result.written_chunks,
        "reused_embeddings": result.reused_embeddings,
        "embedded_chunks": result.embedded_chunks,
        "max_updated_at": (
            result.max_updated_at.isoformat() if result.max_updated_at else None
        ),
    }


def _clean_error_message(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized[:1000]


class IngestRunStore:
    """Transactional run queue, lease, and per-table watermark store."""

    def __init__(
        self,
        pool: Any,
        *,
        lease_seconds: int = 120,
        max_recovery_attempts: int = 1,
    ) -> None:
        self.pool = pool
        self.lease_seconds = max(1, int(lease_seconds))
        self.max_recovery_attempts = max(0, int(max_recovery_attempts))

    async def create_or_get_active(
        self,
        request: IngestRunRequest,
    ) -> tuple[IngestRunRecord, bool]:
        normalized = request.normalized()
        request_key = build_request_key(normalized)
        run_id = uuid4()
        async with self.pool.connection() as conn:
            async with conn.transaction():
                inserted = await (
                    await conn.execute(
                        f"""
                        INSERT INTO ai_ingest_runs (
                            run_id, request_key, trigger_source, status, request_payload
                        )
                        VALUES (%s, %s, %s, 'QUEUED', %s::jsonb)
                        ON CONFLICT DO NOTHING
                        RETURNING {_RUN_SELECT}
                        """,
                        (
                            run_id,
                            request_key,
                            normalized.trigger_source,
                            json.dumps(normalized.to_payload(), sort_keys=True),
                        ),
                    )
                ).fetchone()
                if inserted is not None:
                    return _row_to_record(inserted), False

                existing = await (
                    await conn.execute(
                        f"""
                        SELECT {_RUN_SELECT}
                        FROM ai_ingest_runs
                        WHERE request_key = %s
                          AND status IN ('QUEUED', 'RUNNING')
                        ORDER BY requested_at
                        LIMIT 1
                        """,
                        (request_key,),
                    )
                ).fetchone()
                if existing is None:
                    raise RuntimeError("active ingest run disappeared during deduplication")
                return _row_to_record(existing), True

    async def get(self, run_id: UUID) -> IngestRunRecord | None:
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    f"SELECT {_RUN_SELECT} FROM ai_ingest_runs WHERE run_id = %s",
                    (run_id,),
                )
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    async def claim_next(self, owner: str) -> IngestRunRecord | None:
        if not owner.strip():
            raise ValueError("ingest lease owner is required")
        async with self.pool.connection() as conn:
            async with conn.transaction():
                selected = await (
                    await conn.execute(
                        """
                        SELECT run_id
                        FROM ai_ingest_runs
                        WHERE status = 'QUEUED'
                        ORDER BY requested_at
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                        """
                    )
                ).fetchone()
                if selected is None:
                    return None
                selected_run_id = (
                    selected["run_id"] if isinstance(selected, Mapping) else selected[0]
                )
                claimed = await (
                    await conn.execute(
                        f"""
                        UPDATE ai_ingest_runs
                        SET status = 'RUNNING',
                            started_at = COALESCE(started_at, now()),
                            heartbeat_at = now(),
                            lease_owner = %s,
                            lease_expires_at = now() + make_interval(secs => %s),
                            updated_at = now()
                        WHERE run_id = %s
                          AND status = 'QUEUED'
                        RETURNING {_RUN_SELECT}
                        """,
                        (owner, self.lease_seconds, selected_run_id),
                    )
                ).fetchone()
                return _row_to_record(claimed) if claimed is not None else None

    async def heartbeat(self, run_id: UUID, owner: str) -> bool:
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    UPDATE ai_ingest_runs
                    SET heartbeat_at = now(),
                        lease_expires_at = now() + make_interval(secs => %s),
                        updated_at = now()
                    WHERE run_id = %s
                      AND status = 'RUNNING'
                      AND lease_owner = %s
                    RETURNING run_id
                    """,
                    (self.lease_seconds, run_id, owner),
                )
            ).fetchone()
        return row is not None

    async def finish_success(
        self,
        run_id: UUID,
        owner: str,
        table_results: Mapping[str, IngestTableResult],
        watermarks: Mapping[str, datetime],
        scope_key: str,
    ) -> None:
        summary = {
            source_table: _table_result_payload(result)
            for source_table, result in sorted(table_results.items())
        }
        async with self.pool.connection() as conn:
            async with conn.transaction():
                finished = await (
                    await conn.execute(
                        """
                        UPDATE ai_ingest_runs
                        SET status = 'SUCCEEDED',
                            table_summary = %s::jsonb,
                            finished_at = now(),
                            heartbeat_at = now(),
                            lease_expires_at = NULL,
                            updated_at = now()
                        WHERE run_id = %s
                          AND status = 'RUNNING'
                          AND lease_owner = %s
                        RETURNING run_id
                        """,
                        (json.dumps(summary, sort_keys=True), run_id, owner),
                    )
                ).fetchone()
                self._require_owned_run(finished, run_id)
                for source_table, watermark in sorted(watermarks.items()):
                    if source_table not in table_results:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO ai_ingest_watermarks (
                            source_table, scope_key, last_successful_updated_at, last_run_id, updated_at
                        )
                        VALUES (%s, %s, %s, %s, now())
                        ON CONFLICT (source_table, scope_key) DO UPDATE
                        SET last_run_id = CASE
                                WHEN ai_ingest_watermarks.last_successful_updated_at IS NULL
                                  OR EXCLUDED.last_successful_updated_at
                                     > ai_ingest_watermarks.last_successful_updated_at
                                THEN EXCLUDED.last_run_id
                                ELSE ai_ingest_watermarks.last_run_id
                            END,
                            last_successful_updated_at = GREATEST(
                                ai_ingest_watermarks.last_successful_updated_at,
                                EXCLUDED.last_successful_updated_at
                            ),
                            updated_at = now()
                        """,
                        (source_table, scope_key, watermark, run_id),
                    )

    async def finish_failed(
        self,
        run_id: UUID,
        owner: str,
        *,
        error_code: str,
        error_message: str | None,
    ) -> None:
        await self._finish_terminal(
            run_id,
            owner,
            status=IngestRunStatus.FAILED,
            error_code=error_code[:96],
            error_message=_clean_error_message(error_message),
            table_summary={},
        )

    async def finish_manual_data_required(
        self,
        run_id: UUID,
        owner: str,
        contract: Mapping[str, Any],
    ) -> None:
        allowed_keys = {
            "code",
            "scope",
            "entity",
            "range",
            "missing_fields",
            "import_source",
            "operator_message",
            "message",
            "blocking",
        }
        sanitized = {key: contract[key] for key in allowed_keys if key in contract}
        message = sanitized.get("operator_message") or sanitized.get("message")
        await self._finish_terminal(
            run_id,
            owner,
            status=IngestRunStatus.MANUAL_BASEBALL_DATA_REQUIRED,
            error_code="MANUAL_BASEBALL_DATA_REQUIRED",
            error_message=_clean_error_message(str(message)) if message else None,
            table_summary={"error_contract": sanitized},
        )

    async def _finish_terminal(
        self,
        run_id: UUID,
        owner: str,
        *,
        status: IngestRunStatus,
        error_code: str,
        error_message: str | None,
        table_summary: Mapping[str, Any],
    ) -> None:
        async with self.pool.connection() as conn:
            async with conn.transaction():
                finished = await (
                    await conn.execute(
                        """
                        UPDATE ai_ingest_runs
                        SET status = %s,
                            error_code = %s,
                            error_message = %s,
                            table_summary = %s::jsonb,
                            finished_at = now(),
                            heartbeat_at = now(),
                            lease_expires_at = NULL,
                            updated_at = now()
                        WHERE run_id = %s
                          AND status = 'RUNNING'
                          AND lease_owner = %s
                        RETURNING run_id
                        """,
                        (
                            status.value,
                            error_code,
                            error_message,
                            json.dumps(table_summary, sort_keys=True),
                            run_id,
                            owner,
                        ),
                    )
                ).fetchone()
                self._require_owned_run(finished, run_id)

    async def recover_expired(self) -> tuple[int, int]:
        async with self.pool.connection() as conn:
            async with conn.transaction():
                recovered = await (
                    await conn.execute(
                        """
                        UPDATE ai_ingest_runs
                        SET status = 'QUEUED',
                            recovery_attempts = recovery_attempts + 1,
                            lease_owner = NULL,
                            lease_expires_at = NULL,
                            heartbeat_at = NULL,
                            updated_at = now()
                        WHERE status = 'RUNNING'
                          AND lease_expires_at < now()
                          AND recovery_attempts < %s
                        RETURNING run_id
                        """,
                        (self.max_recovery_attempts,),
                    )
                ).fetchall()
                failed = await (
                    await conn.execute(
                        """
                        UPDATE ai_ingest_runs
                        SET status = 'FAILED',
                            error_code = 'INGEST_LEASE_EXPIRED',
                            error_message = 'Ingestion worker lease expired after recovery limit.',
                            finished_at = now(),
                            lease_owner = NULL,
                            lease_expires_at = NULL,
                            updated_at = now()
                        WHERE status = 'RUNNING'
                          AND lease_expires_at < now()
                          AND recovery_attempts >= %s
                        RETURNING run_id
                        """,
                        (self.max_recovery_attempts,),
                    )
                ).fetchall()
        return len(recovered), len(failed)

    async def get_watermark(
        self,
        source_table: str,
        scope_key: str,
    ) -> datetime | None:
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    SELECT last_successful_updated_at
                    FROM ai_ingest_watermarks
                    WHERE source_table = %s
                      AND scope_key = %s
                    """,
                    (source_table, scope_key),
                )
            ).fetchone()
        if row is None:
            return None
        return row["last_successful_updated_at"] if isinstance(row, Mapping) else row[0]

    async def advance_watermark(
        self,
        source_table: str,
        scope_key: str,
        watermark: datetime,
        run_id: UUID,
    ) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO ai_ingest_watermarks (
                    source_table, scope_key, last_successful_updated_at, last_run_id, updated_at
                )
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (source_table, scope_key) DO UPDATE
                SET last_run_id = CASE
                        WHEN ai_ingest_watermarks.last_successful_updated_at IS NULL
                          OR EXCLUDED.last_successful_updated_at
                             > ai_ingest_watermarks.last_successful_updated_at
                        THEN EXCLUDED.last_run_id
                        ELSE ai_ingest_watermarks.last_run_id
                    END,
                    last_successful_updated_at = GREATEST(
                        ai_ingest_watermarks.last_successful_updated_at,
                        EXCLUDED.last_successful_updated_at
                    ),
                    updated_at = now()
                """,
                (source_table, scope_key, watermark, run_id),
            )

    @staticmethod
    def _require_owned_run(row: Any, run_id: UUID) -> None:
        if row is None:
            raise RuntimeError(f"ingest run lease is not owned: {run_id}")
