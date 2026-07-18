from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from app.core.ingest_checkpoints import (
    CheckpointCursor,
    CheckpointOrder,
    CheckpointOrderField,
    IngestCheckpointIncompatibleError,
    IngestCheckpointCursorTypeError,
    IngestCheckpointMissingFieldError,
    IngestCheckpointOrderError,
    decode_cursor,
    encode_cursor,
    cursor_from_row,
    ensure_cursor_advances,
)


ORDER = CheckpointOrder(
    source_table="game",
    fields=(
        CheckpointOrderField("game_date", "date"),
        CheckpointOrderField("game_id", "text"),
    ),
    query_version="1",
)


@pytest.mark.parametrize(
    ("scalar_type", "value"),
    [
        ("integer", 42),
        ("decimal", Decimal("3.140")),
        ("date", date(2026, 7, 18)),
        ("datetime", datetime(2026, 7, 18, 4, 0, tzinfo=UTC)),
        ("uuid", UUID("44444444-4444-4444-8444-444444444444")),
        ("text", "20260718LGKT"),
        ("boolean", True),
    ],
)
def test_cursor_codec_round_trips_supported_types(scalar_type, value):
    order = CheckpointOrder(
        source_table="source",
        fields=(CheckpointOrderField("key", scalar_type),),
        query_version="1",
    )
    cursor = CheckpointCursor((value,))

    assert decode_cursor(order, encode_cursor(order, cursor)) == cursor


def test_signature_changes_with_query_version():
    changed = CheckpointOrder(ORDER.source_table, ORDER.fields, query_version="2")
    assert changed.signature != ORDER.signature


def test_decode_rejects_field_or_type_mismatch():
    payload = encode_cursor(ORDER, CheckpointCursor((date(2026, 7, 18), "g1")))
    payload["values"][1]["field"] = "other"
    with pytest.raises(IngestCheckpointIncompatibleError):
        decode_cursor(ORDER, payload)


def test_cursor_from_row_reports_missing_and_null_fields():
    with pytest.raises(IngestCheckpointMissingFieldError) as raised:
        cursor_from_row(ORDER, {"game_date": date(2026, 7, 18), "game_id": None})
    assert raised.value.missing_fields == ("game_id",)


def test_cursor_must_advance_strictly():
    first = CheckpointCursor((date(2026, 7, 18), "g1"))
    with pytest.raises(IngestCheckpointOrderError):
        ensure_cursor_advances(ORDER, first, first)


def test_integer_cursor_rejects_boolean_value():
    order = CheckpointOrder(
        source_table="source",
        fields=(CheckpointOrderField("key", "integer"),),
    )

    with pytest.raises(IngestCheckpointCursorTypeError):
        encode_cursor(order, CheckpointCursor((True,)))


def test_decode_rejects_malformed_scalar_value():
    payload = {"values": [{"field": "game_date", "type": "date", "value": "bad-date"}, {"field": "game_id", "type": "text", "value": "g1"}]}

    with pytest.raises(IngestCheckpointCursorTypeError):
        decode_cursor(ORDER, payload)


def test_naive_datetime_is_normalized_to_utc():
    order = CheckpointOrder(
        source_table="source",
        fields=(CheckpointOrderField("key", "datetime"),),
    )
    cursor = cursor_from_row(order, {"key": datetime(2026, 7, 18, 4, 0)})

    assert cursor == CheckpointCursor((datetime(2026, 7, 18, 4, 0, tzinfo=UTC),))
