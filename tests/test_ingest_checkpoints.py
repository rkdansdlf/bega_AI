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


def test_cursor_rejects_decreasing_value():
    previous = CheckpointCursor((date(2026, 7, 18), "g2"))
    current = CheckpointCursor((date(2026, 7, 18), "g1"))

    with pytest.raises(IngestCheckpointOrderError):
        ensure_cursor_advances(ORDER, previous, current)


def test_cursor_accepts_tie_breaking_advancement():
    previous = CheckpointCursor((date(2026, 7, 18), "g1"))
    current = CheckpointCursor((date(2026, 7, 18), "g2"))

    ensure_cursor_advances(ORDER, previous, current)


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        (
            CheckpointCursor((date(2026, 7, 18),)),
            CheckpointCursor((date(2026, 7, 18), "g2")),
        ),
        (
            CheckpointCursor((date(2026, 7, 18), "g1")),
            CheckpointCursor((date(2026, 7, 18),)),
        ),
    ],
)
def test_cursor_comparison_rejects_arity_mismatch(previous, current):
    with pytest.raises(IngestCheckpointIncompatibleError):
        ensure_cursor_advances(ORDER, previous, current)


def test_cursor_comparison_rejects_type_invalid_value():
    previous = CheckpointCursor(("2026-07-18", "g1"))
    current = CheckpointCursor((date(2026, 7, 18), "g2"))

    with pytest.raises(IngestCheckpointCursorTypeError):
        ensure_cursor_advances(ORDER, previous, current)


def test_integer_cursor_rejects_boolean_value():
    order = CheckpointOrder(
        source_table="source",
        fields=(CheckpointOrderField("key", "integer"),),
    )

    with pytest.raises(IngestCheckpointCursorTypeError):
        encode_cursor(order, CheckpointCursor((True,)))


def test_decode_rejects_malformed_scalar_value():
    payload = {
        "values": [
            {"field": "game_date", "type": "date", "value": "bad-date"},
            {"field": "game_id", "type": "text", "value": "g1"},
        ]
    }

    with pytest.raises(IngestCheckpointCursorTypeError):
        decode_cursor(ORDER, payload)


@pytest.mark.parametrize("payload", [None, [], "invalid"])
def test_decode_rejects_non_mapping_payload(payload):
    with pytest.raises(IngestCheckpointIncompatibleError):
        decode_cursor(ORDER, payload)


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_decimal_cursor_rejects_non_finite_values(value):
    order = CheckpointOrder(
        source_table="source",
        fields=(CheckpointOrderField("key", "decimal"),),
    )

    with pytest.raises(IngestCheckpointCursorTypeError):
        encode_cursor(order, CheckpointCursor((value,)))

    payload = {"values": [{"field": "key", "type": "decimal", "value": str(value)}]}
    with pytest.raises(IngestCheckpointCursorTypeError):
        decode_cursor(order, payload)


def test_order_fields_are_immutable_after_input_list_mutation():
    fields = [CheckpointOrderField("game_date", "date")]
    order = CheckpointOrder("game", fields)
    signature = order.signature

    fields.append(CheckpointOrderField("game_id", "text"))

    assert order.fields == (CheckpointOrderField("game_date", "date"),)
    assert order.signature == signature


def test_cursor_values_are_immutable_after_input_list_mutation():
    values = [date(2026, 7, 18), "g1"]
    cursor = CheckpointCursor(values)

    values[1] = "g2"

    assert cursor.values == (date(2026, 7, 18), "g1")


def test_naive_datetime_is_normalized_to_utc():
    order = CheckpointOrder(
        source_table="source",
        fields=(CheckpointOrderField("key", "datetime"),),
    )
    cursor = cursor_from_row(order, {"key": datetime(2026, 7, 18, 4, 0)})

    assert cursor == CheckpointCursor((datetime(2026, 7, 18, 4, 0, tzinfo=UTC),))
