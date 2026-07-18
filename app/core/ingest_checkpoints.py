"""Typed cursor primitives and persistence for durable ingest checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Any, Literal
from uuid import UUID


CURSOR_VERSION = 1
CursorScalarType = Literal[
    "integer",
    "decimal",
    "date",
    "datetime",
    "datetime_naive",
    "uuid",
    "text",
    "boolean",
]


class IngestCheckpointError(RuntimeError):
    code = "INGEST_CHECKPOINT_ERROR"


class IngestCheckpointCursorUnavailableError(IngestCheckpointError):
    code = "INGEST_CHECKPOINT_CURSOR_UNAVAILABLE"


class IngestCheckpointIncompatibleError(IngestCheckpointError):
    code = "INGEST_CHECKPOINT_INCOMPATIBLE"


class IngestCheckpointCursorTypeError(IngestCheckpointError):
    code = "INGEST_CHECKPOINT_CURSOR_TYPE_UNSUPPORTED"


class IngestCheckpointOrderError(IngestCheckpointError):
    code = "INGEST_CHECKPOINT_ORDER_VIOLATION"


class IngestCheckpointStaleCleanupError(IngestCheckpointError):
    code = "INGEST_CHECKPOINT_STALE_CLEANUP_UNSUPPORTED"


class IngestCheckpointMissingFieldError(IngestCheckpointError):
    def __init__(self, missing_fields: Sequence[str]) -> None:
        self.missing_fields = tuple(sorted(set(missing_fields)))
        super().__init__("checkpoint cursor fields are missing")


@dataclass(frozen=True)
class CheckpointOrderField:
    name: str
    scalar_type: CursorScalarType


@dataclass(frozen=True)
class CheckpointOrder:
    source_table: str
    fields: tuple[CheckpointOrderField, ...]
    query_version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))

    @property
    def signature(self) -> str:
        payload = {
            "cursor_version": CURSOR_VERSION,
            "source_table": self.source_table,
            "query_version": self.query_version,
            "fields": [
                {"name": field.name, "type": field.scalar_type, "direction": "asc"}
                for field in self.fields
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckpointCursor:
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))


def _normalize_cursor_value(kind: CursorScalarType, value: Any) -> Any:
    if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if kind == "decimal" and isinstance(value, Decimal):
        if value.is_finite():
            return value
        raise IngestCheckpointCursorTypeError("decimal cursor value must be finite")
    if kind == "datetime" and isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if kind == "datetime_naive" and isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        raise IngestCheckpointCursorTypeError(
            "datetime_naive cursor value must not include a timezone offset"
        )
    if kind == "date" and isinstance(value, date) and not isinstance(value, datetime):
        return value
    if kind == "uuid" and isinstance(value, UUID):
        return value
    if kind == "text" and isinstance(value, str):
        return value
    if kind == "boolean" and isinstance(value, bool):
        return value
    raise IngestCheckpointCursorTypeError(
        f"unsupported {kind} cursor value type: {type(value).__name__}"
    )


def _json_cursor_value(kind: CursorScalarType, value: Any) -> Any:
    normalized = _normalize_cursor_value(kind, value)
    if kind in {"decimal", "date", "datetime", "datetime_naive", "uuid"}:
        return normalized.isoformat() if hasattr(normalized, "isoformat") else str(normalized)
    return normalized


def _python_cursor_value(kind: CursorScalarType, value: Any) -> Any:
    try:
        if kind == "integer":
            decoded = value if isinstance(value, int) and not isinstance(value, bool) else None
        elif kind == "decimal":
            decoded = Decimal(str(value))
        elif kind == "date":
            decoded = date.fromisoformat(str(value))
        elif kind == "datetime":
            decoded = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        elif kind == "datetime_naive":
            decoded = datetime.fromisoformat(str(value))
        elif kind == "uuid":
            decoded = UUID(str(value))
        elif kind == "text":
            decoded = value if isinstance(value, str) else None
        else:
            decoded = value if isinstance(value, bool) else None
        if decoded is None:
            raise ValueError("invalid cursor scalar")
        return _normalize_cursor_value(kind, decoded)
    except (TypeError, ValueError, InvalidOperation) as exc:
        raise IngestCheckpointCursorTypeError(
            f"invalid stored {kind} cursor value"
        ) from exc


def encode_cursor(order: CheckpointOrder, cursor: CheckpointCursor) -> dict[str, Any]:
    if len(order.fields) != len(cursor.values):
        raise IngestCheckpointIncompatibleError("cursor arity mismatch")
    return {
        "values": [
            {
                "field": field.name,
                "type": field.scalar_type,
                "value": _json_cursor_value(field.scalar_type, value),
            }
            for field, value in zip(order.fields, cursor.values, strict=True)
        ]
    }


def decode_cursor(order: CheckpointOrder, payload: Mapping[str, Any]) -> CheckpointCursor:
    if not isinstance(payload, Mapping):
        raise IngestCheckpointIncompatibleError("stored cursor payload is not a mapping")
    items = payload.get("values")
    if not isinstance(items, list) or len(items) != len(order.fields):
        raise IngestCheckpointIncompatibleError("stored cursor arity mismatch")
    values = []
    for field, item in zip(order.fields, items, strict=True):
        if (
            not isinstance(item, Mapping)
            or item.get("field") != field.name
            or item.get("type") != field.scalar_type
        ):
            raise IngestCheckpointIncompatibleError("stored cursor field mismatch")
        values.append(_python_cursor_value(field.scalar_type, item.get("value")))
    return CheckpointCursor(tuple(values))


def cursor_from_row(order: CheckpointOrder, row: Mapping[str, Any]) -> CheckpointCursor:
    missing = [field.name for field in order.fields if row.get(field.name) is None]
    if missing:
        raise IngestCheckpointMissingFieldError(missing)
    return CheckpointCursor(
        tuple(
            _normalize_cursor_value(field.scalar_type, row[field.name])
            for field in order.fields
        )
    )


def ensure_cursor_advances(
    order: CheckpointOrder,
    previous: CheckpointCursor,
    current: CheckpointCursor,
) -> None:
    previous_values = _normalized_cursor_values(order, previous)
    current_values = _normalized_cursor_values(order, current)
    if current_values <= previous_values:
        raise IngestCheckpointOrderError("checkpoint cursor did not advance")


def _normalized_cursor_values(
    order: CheckpointOrder,
    cursor: CheckpointCursor,
) -> tuple[Any, ...]:
    if len(order.fields) != len(cursor.values):
        raise IngestCheckpointIncompatibleError("cursor arity mismatch")
    return tuple(
        _normalize_cursor_value(field.scalar_type, value)
        for field, value in zip(order.fields, cursor.values, strict=True)
    )


@dataclass(frozen=True)
class IngestCheckpoint:
    run_id: UUID
    source_table: str
    scope_key: str
    cursor_version: int
    cursor_signature: str
    cursor: CheckpointCursor | None
    committed_batches: int = 0
    source_rows: int = 0
    written_chunks: int = 0
    reused_embeddings: int = 0
    embedded_chunks: int = 0
    max_updated_at: datetime | None = None
    source_updated_before: datetime | None = None
    completed: bool = False
    completed_at: datetime | None = None


_CHECKPOINT_COLUMNS = (
    "run_id",
    "source_table",
    "scope_key",
    "cursor_version",
    "cursor_signature",
    "cursor_payload",
    "committed_batches",
    "source_rows",
    "written_chunks",
    "reused_embeddings",
    "embedded_chunks",
    "max_updated_at",
    "source_updated_before",
    "completed",
    "completed_at",
)
_CHECKPOINT_SELECT = ", ".join(_CHECKPOINT_COLUMNS)


def _checkpoint_row_mapping(row: Any) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    try:
        return dict(zip(_CHECKPOINT_COLUMNS, row, strict=True))
    except (TypeError, ValueError) as exc:
        raise IngestCheckpointIncompatibleError(
            "stored checkpoint row does not match the checkpoint schema"
        ) from exc


def _stored_cursor_payload(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise IngestCheckpointIncompatibleError(
                "stored checkpoint cursor payload is invalid JSON"
            ) from exc
    if not isinstance(value, Mapping):
        raise IngestCheckpointIncompatibleError(
            "stored checkpoint cursor payload is not a mapping"
        )
    return value


def _stored_counter(values: Mapping[str, Any], name: str) -> int:
    value = values[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise IngestCheckpointIncompatibleError(
            f"stored checkpoint {name} is not a nonnegative integer"
        )
    return value


def _checkpoint_datetime(value: Any, name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise IngestCheckpointIncompatibleError(
            f"stored checkpoint {name} is not a datetime"
        )
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _checkpoint_from_row(order: CheckpointOrder, row: Any) -> IngestCheckpoint:
    values = _checkpoint_row_mapping(row)
    try:
        run_id = values["run_id"]
        source_table = values["source_table"]
        scope_key = values["scope_key"]
        cursor_version = values["cursor_version"]
        cursor_signature = values["cursor_signature"]
        completed = values["completed"]
        if not isinstance(run_id, UUID):
            raise IngestCheckpointIncompatibleError(
                "stored checkpoint run_id is not a UUID"
            )
        if not isinstance(source_table, str) or not source_table.strip():
            raise IngestCheckpointIncompatibleError(
                "stored checkpoint source_table is invalid"
            )
        if not isinstance(scope_key, str) or not scope_key.strip():
            raise IngestCheckpointIncompatibleError(
                "stored checkpoint scope_key is invalid"
            )
        if not isinstance(cursor_version, int) or isinstance(cursor_version, bool):
            raise IngestCheckpointIncompatibleError(
                "stored checkpoint cursor_version is invalid"
            )
        if not isinstance(cursor_signature, str) or not cursor_signature:
            raise IngestCheckpointIncompatibleError(
                "stored checkpoint cursor_signature is invalid"
            )
        if not isinstance(completed, bool):
            raise IngestCheckpointIncompatibleError(
                "stored checkpoint completed flag is invalid"
            )

        payload = values["cursor_payload"]
        try:
            cursor = (
                decode_cursor(order, _stored_cursor_payload(payload))
                if payload is not None
                else None
            )
        except IngestCheckpointError as exc:
            raise IngestCheckpointIncompatibleError(
                "stored checkpoint cursor payload is incompatible"
            ) from exc

        committed_batches = _stored_counter(values, "committed_batches")
        source_rows = _stored_counter(values, "source_rows")
        written_chunks = _stored_counter(values, "written_chunks")
        reused_embeddings = _stored_counter(values, "reused_embeddings")
        embedded_chunks = _stored_counter(values, "embedded_chunks")
        max_updated_at = _checkpoint_datetime(
            values["max_updated_at"],
            "max_updated_at",
        )
        source_updated_before = _checkpoint_datetime(
            values["source_updated_before"],
            "source_updated_before",
        )
        completed_at = _checkpoint_datetime(values["completed_at"], "completed_at")
        if source_rows > 0 and cursor is None:
            raise IngestCheckpointIncompatibleError(
                "stored checkpoint has source rows without a cursor"
            )
        if completed != (completed_at is not None):
            raise IngestCheckpointIncompatibleError(
                "stored checkpoint completion state is inconsistent"
            )

        return IngestCheckpoint(
            run_id=run_id,
            source_table=source_table,
            scope_key=scope_key,
            cursor_version=cursor_version,
            cursor_signature=cursor_signature,
            cursor=cursor,
            committed_batches=committed_batches,
            source_rows=source_rows,
            written_chunks=written_chunks,
            reused_embeddings=reused_embeddings,
            embedded_chunks=embedded_chunks,
            max_updated_at=max_updated_at,
            source_updated_before=source_updated_before,
            completed=completed,
            completed_at=completed_at,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IngestCheckpointIncompatibleError(
            "stored checkpoint row contains invalid values"
        ) from exc


def _maximum_updated_at(
    persisted: datetime | None,
    candidate: datetime | None,
) -> datetime | None:
    persisted = _checkpoint_datetime(persisted, "max_updated_at")
    candidate = _checkpoint_datetime(candidate, "max_updated_at")
    if persisted is None:
        return candidate
    if candidate is None:
        return persisted
    return max(persisted, candidate)


def _checkpoint_has_progress(checkpoint: IngestCheckpoint) -> bool:
    return bool(
        checkpoint.cursor is not None
        or checkpoint.committed_batches
        or checkpoint.source_rows
        or checkpoint.written_chunks
        or checkpoint.reused_embeddings
        or checkpoint.embedded_chunks
        or checkpoint.max_updated_at is not None
    )


def _durable_source_updated_before(
    persisted: IngestCheckpoint | None,
    candidate: datetime | None,
) -> datetime | None:
    candidate = _checkpoint_datetime(candidate, "source_updated_before")
    if persisted is None or persisted.source_updated_before is None:
        return candidate
    if candidate != persisted.source_updated_before:
        raise IngestCheckpointIncompatibleError(
            "persisted source update cutoff is immutable"
        )
    return persisted.source_updated_before


class IngestCheckpointRepository:
    """Synchronous destination-cursor repository for checkpoint progress."""

    def __init__(self, order: CheckpointOrder) -> None:
        self.order = order

    def load(
        self,
        db_cursor: Any,
        *,
        run_id: UUID,
        source_table: str,
        for_update: bool = False,
    ) -> IngestCheckpoint | None:
        lock_clause = " FOR UPDATE" if for_update else ""
        db_cursor.execute(
            f"""
            SELECT {_CHECKPOINT_SELECT}
            FROM ai_ingest_checkpoints
            WHERE run_id = %s AND source_table = %s{lock_clause}
            """,
            (run_id, source_table),
        )
        row = db_cursor.fetchone()
        return _checkpoint_from_row(self.order, row) if row is not None else None

    def advance(
        self,
        db_cursor: Any,
        *,
        run_id: UUID,
        source_table: str,
        scope_key: str,
        current: IngestCheckpoint | None,
        next_cursor: CheckpointCursor,
        source_rows_delta: int,
        written_chunks_delta: int,
        reused_embeddings_delta: int,
        embedded_chunks_delta: int,
        max_updated_at: datetime | None,
        source_updated_before: datetime | None,
    ) -> IngestCheckpoint:
        deltas = (
            source_rows_delta,
            written_chunks_delta,
            reused_embeddings_delta,
            embedded_chunks_delta,
        )
        if any(delta < 0 for delta in deltas):
            raise ValueError("checkpoint progress deltas must be nonnegative")

        persisted = self.load(
            db_cursor,
            run_id=run_id,
            source_table=source_table,
            for_update=True,
        )
        self._require_expected(
            persisted,
            current,
            run_id=run_id,
            source_table=source_table,
            scope_key=scope_key,
        )
        if persisted is not None and persisted.cursor is not None:
            ensure_cursor_advances(self.order, persisted.cursor, next_cursor)

        committed_batches = (persisted.committed_batches if persisted else 0) + 1
        source_rows = (persisted.source_rows if persisted else 0) + source_rows_delta
        written_chunks = (
            (persisted.written_chunks if persisted else 0) + written_chunks_delta
        )
        reused_embeddings = (
            (persisted.reused_embeddings if persisted else 0)
            + reused_embeddings_delta
        )
        embedded_chunks = (
            (persisted.embedded_chunks if persisted else 0) + embedded_chunks_delta
        )
        durable_max_updated_at = _maximum_updated_at(
            persisted.max_updated_at if persisted else None,
            max_updated_at,
        )
        durable_source_updated_before = _durable_source_updated_before(
            persisted,
            source_updated_before,
        )
        encoded_cursor = encode_cursor(self.order, next_cursor)
        durable_cursor = decode_cursor(self.order, encoded_cursor)
        cursor_payload = json.dumps(
            encoded_cursor,
            sort_keys=True,
            separators=(",", ":"),
        )

        if persisted is None:
            db_cursor.execute(
                f"""
                INSERT INTO ai_ingest_checkpoints (
                    run_id,
                    source_table,
                    scope_key,
                    cursor_version,
                    cursor_signature,
                    cursor_payload,
                    committed_batches,
                    source_rows,
                    written_chunks,
                    reused_embeddings,
                    embedded_chunks,
                    max_updated_at,
                    source_updated_before,
                    completed,
                    completed_at,
                    updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s::jsonb,
                    %s, %s, %s, %s, %s, %s, %s,
                    false, NULL, clock_timestamp()
                )
                ON CONFLICT (run_id, source_table) DO NOTHING
                RETURNING {_CHECKPOINT_SELECT}
                """,
                (
                    run_id,
                    source_table,
                    scope_key,
                    CURSOR_VERSION,
                    self.order.signature,
                    cursor_payload,
                    committed_batches,
                    source_rows,
                    written_chunks,
                    reused_embeddings,
                    embedded_chunks,
                    durable_max_updated_at,
                    durable_source_updated_before,
                ),
            )
        else:
            db_cursor.execute(
                f"""
                UPDATE ai_ingest_checkpoints
                SET cursor_payload = %s::jsonb,
                    committed_batches = %s,
                    source_rows = %s,
                    written_chunks = %s,
                    reused_embeddings = %s,
                    embedded_chunks = %s,
                    max_updated_at = %s,
                    source_updated_before = COALESCE(source_updated_before, %s),
                    completed = false,
                    completed_at = NULL,
                    updated_at = clock_timestamp()
                WHERE run_id = %s AND source_table = %s
                RETURNING {_CHECKPOINT_SELECT}
                """,
                (
                    cursor_payload,
                    committed_batches,
                    source_rows,
                    written_chunks,
                    reused_embeddings,
                    embedded_chunks,
                    durable_max_updated_at,
                    durable_source_updated_before,
                    run_id,
                    source_table,
                ),
            )

        updated = self._returned_checkpoint(db_cursor)
        self._require_compatible(
            updated,
            run_id=run_id,
            source_table=source_table,
            scope_key=scope_key,
        )
        if (
            updated.cursor != durable_cursor
            or updated.committed_batches != committed_batches
            or updated.source_rows != source_rows
            or updated.written_chunks != written_chunks
            or updated.reused_embeddings != reused_embeddings
            or updated.embedded_chunks != embedded_chunks
            or updated.max_updated_at != durable_max_updated_at
            or updated.source_updated_before != durable_source_updated_before
            or updated.completed
            or updated.completed_at is not None
        ):
            raise IngestCheckpointIncompatibleError(
                "checkpoint advance returned unexpected durable state"
            )
        return updated

    def complete(
        self,
        db_cursor: Any,
        *,
        run_id: UUID,
        source_table: str,
        scope_key: str,
        current: IngestCheckpoint | None,
        source_updated_before: datetime | None,
    ) -> IngestCheckpoint:
        persisted = self.load(
            db_cursor,
            run_id=run_id,
            source_table=source_table,
            for_update=True,
        )
        self._require_expected(
            persisted,
            current,
            run_id=run_id,
            source_table=source_table,
            scope_key=scope_key,
        )
        durable_source_updated_before = _durable_source_updated_before(
            persisted,
            source_updated_before,
        )

        if persisted is None:
            db_cursor.execute(
                f"""
                INSERT INTO ai_ingest_checkpoints (
                    run_id,
                    source_table,
                    scope_key,
                    cursor_version,
                    cursor_signature,
                    cursor_payload,
                    committed_batches,
                    source_rows,
                    written_chunks,
                    reused_embeddings,
                    embedded_chunks,
                    max_updated_at,
                    source_updated_before,
                    completed,
                    completed_at,
                    updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, NULL,
                    0, 0, 0, 0, 0, NULL, %s,
                    true, clock_timestamp(), clock_timestamp()
                )
                ON CONFLICT (run_id, source_table) DO NOTHING
                RETURNING {_CHECKPOINT_SELECT}
                """,
                (
                    run_id,
                    source_table,
                    scope_key,
                    CURSOR_VERSION,
                    self.order.signature,
                    durable_source_updated_before,
                ),
            )
        else:
            db_cursor.execute(
                f"""
                UPDATE ai_ingest_checkpoints
                SET source_updated_before = COALESCE(source_updated_before, %s),
                    completed = true,
                    completed_at = clock_timestamp(),
                    updated_at = clock_timestamp()
                WHERE run_id = %s AND source_table = %s
                RETURNING {_CHECKPOINT_SELECT}
                """,
                (durable_source_updated_before, run_id, source_table),
            )

        completed = self._returned_checkpoint(db_cursor)
        self._require_compatible(
            completed,
            run_id=run_id,
            source_table=source_table,
            scope_key=scope_key,
        )
        expected = persisted or IngestCheckpoint(
            run_id=run_id,
            source_table=source_table,
            scope_key=scope_key,
            cursor_version=CURSOR_VERSION,
            cursor_signature=self.order.signature,
            cursor=None,
            source_updated_before=durable_source_updated_before,
        )
        if (
            completed.cursor != expected.cursor
            or completed.committed_batches != expected.committed_batches
            or completed.source_rows != expected.source_rows
            or completed.written_chunks != expected.written_chunks
            or completed.reused_embeddings != expected.reused_embeddings
            or completed.embedded_chunks != expected.embedded_chunks
            or completed.max_updated_at != expected.max_updated_at
            or completed.source_updated_before != durable_source_updated_before
            or not completed.completed
            or completed.completed_at is None
        ):
            raise IngestCheckpointIncompatibleError(
                "checkpoint completion returned unexpected durable state"
            )
        return completed

    def _require_expected(
        self,
        persisted: IngestCheckpoint | None,
        expected: IngestCheckpoint | None,
        *,
        run_id: UUID,
        source_table: str,
        scope_key: str,
    ) -> None:
        if persisted is not None:
            self._require_compatible(
                persisted,
                run_id=run_id,
                source_table=source_table,
                scope_key=scope_key,
            )
        if persisted != expected:
            raise IngestCheckpointIncompatibleError(
                "persisted checkpoint changed after session start"
            )

    def _require_compatible(
        self,
        stored: IngestCheckpoint,
        *,
        run_id: UUID,
        source_table: str,
        scope_key: str,
    ) -> None:
        if (
            stored.run_id != run_id
            or stored.source_table != source_table
            or stored.scope_key != scope_key
            or stored.cursor_version != CURSOR_VERSION
            or stored.cursor_signature != self.order.signature
        ):
            raise IngestCheckpointIncompatibleError(
                "stored checkpoint is incompatible with the ingest session"
            )

    def _returned_checkpoint(self, db_cursor: Any) -> IngestCheckpoint:
        row = db_cursor.fetchone()
        if row is None:
            raise IngestCheckpointIncompatibleError(
                "checkpoint mutation returned no durable row"
            )
        return _checkpoint_from_row(self.order, row)


class IngestCheckpointSession:
    """Tracks the checkpoint state durably observed by one transaction."""

    def __init__(
        self,
        repository: IngestCheckpointRepository,
        *,
        run_id: UUID,
        source_table: str,
        scope_key: str,
        initial: IngestCheckpoint | None,
        requires_source_updated_before: bool,
    ) -> None:
        self.repository = repository
        self.run_id = run_id
        self.source_table = source_table
        self.scope_key = scope_key
        self.initial = initial
        self.current = initial
        self.requires_source_updated_before = requires_source_updated_before
        self._source_updated_before = (
            initial.source_updated_before if initial is not None else None
        )

    @classmethod
    def start(
        cls,
        db_cursor: Any,
        *,
        run_id: UUID,
        source_table: str,
        scope_key: str,
        order: CheckpointOrder,
        requires_source_updated_before: bool = False,
    ) -> IngestCheckpointSession:
        if order.source_table != source_table:
            raise IngestCheckpointIncompatibleError(
                "checkpoint order does not match the source table"
            )
        repository = IngestCheckpointRepository(order)
        initial = repository.load(
            db_cursor,
            run_id=run_id,
            source_table=source_table,
        )
        if initial is not None:
            repository._require_compatible(
                initial,
                run_id=run_id,
                source_table=source_table,
                scope_key=scope_key,
            )
            if (
                requires_source_updated_before
                and not initial.completed
                and initial.source_updated_before is None
                and _checkpoint_has_progress(initial)
            ):
                raise IngestCheckpointIncompatibleError(
                    "progressed checkpoint is missing required source update cutoff"
                )
        return cls(
            repository,
            run_id=run_id,
            source_table=source_table,
            scope_key=scope_key,
            initial=initial,
            requires_source_updated_before=requires_source_updated_before,
        )

    @property
    def resumed(self) -> bool:
        return self.initial is not None

    @property
    def completed(self) -> bool:
        return bool(self.current and self.current.completed)

    @property
    def source_updated_before(self) -> datetime | None:
        if self.current is not None and self.current.source_updated_before is not None:
            return self.current.source_updated_before
        return self._source_updated_before

    def bind_source_updated_before(self, value: datetime) -> None:
        normalized = _checkpoint_datetime(value, "source_updated_before")
        if normalized is None:
            raise IngestCheckpointIncompatibleError(
                "source update cutoff must be a timestamp"
            )
        if (
            self.source_updated_before is not None
            and self.source_updated_before != normalized
        ):
            raise IngestCheckpointIncompatibleError(
                "persisted source update cutoff is immutable"
            )
        self._source_updated_before = normalized

    def _require_source_updated_before(self) -> None:
        if self.requires_source_updated_before and self.source_updated_before is None:
            raise IngestCheckpointIncompatibleError(
                "checkpoint requires a source update cutoff"
            )

    def advance(
        self,
        db_cursor: Any,
        *,
        next_cursor: CheckpointCursor,
        source_rows_delta: int,
        written_chunks_delta: int,
        reused_embeddings_delta: int,
        embedded_chunks_delta: int,
        max_updated_at: datetime | None,
    ) -> IngestCheckpoint:
        self._require_source_updated_before()
        updated = self.repository.advance(
            db_cursor,
            run_id=self.run_id,
            source_table=self.source_table,
            scope_key=self.scope_key,
            current=self.current,
            next_cursor=next_cursor,
            source_rows_delta=source_rows_delta,
            written_chunks_delta=written_chunks_delta,
            reused_embeddings_delta=reused_embeddings_delta,
            embedded_chunks_delta=embedded_chunks_delta,
            max_updated_at=max_updated_at,
            source_updated_before=self.source_updated_before,
        )
        self.current = updated
        return updated

    def complete(self, db_cursor: Any) -> IngestCheckpoint:
        self._require_source_updated_before()
        completed = self.repository.complete(
            db_cursor,
            run_id=self.run_id,
            source_table=self.source_table,
            scope_key=self.scope_key,
            current=self.current,
            source_updated_before=self.source_updated_before,
        )
        self.current = completed
        return completed
