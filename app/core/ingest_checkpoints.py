"""Typed cursor primitives for durable ingest checkpoints."""

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
    "integer", "decimal", "date", "datetime", "uuid", "text", "boolean"
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
    if kind in {"decimal", "date", "datetime", "uuid"}:
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
    completed: bool = False
    completed_at: datetime | None = None
