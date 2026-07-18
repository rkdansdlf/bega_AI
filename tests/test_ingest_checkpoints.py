from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from app.core.ingest_checkpoints import (
    CURSOR_VERSION,
    CheckpointCursor,
    CheckpointOrder,
    CheckpointOrderField,
    IngestCheckpoint,
    IngestCheckpointIncompatibleError,
    IngestCheckpointCursorTypeError,
    IngestCheckpointCursorUnavailableError,
    IngestCheckpointMissingFieldError,
    IngestCheckpointOrderError,
    IngestCheckpointRepository,
    IngestCheckpointSession,
    decode_cursor,
    encode_cursor,
    cursor_from_row,
    ensure_cursor_advances,
)
import scripts.ingest_from_kbo as ingest_script


ORDER = CheckpointOrder(
    source_table="game",
    fields=(
        CheckpointOrderField("game_date", "date"),
        CheckpointOrderField("game_id", "text"),
    ),
    query_version="1",
)
RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
SCOPE_KEY = "season:2026"
NOW = datetime(2026, 7, 18, 4, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 7, 18, 5, 0, tzinfo=UTC)


def _checkpoint_row(
    *,
    run_id=RUN_ID,
    source_table="game",
    scope_key=SCOPE_KEY,
    cursor_version=CURSOR_VERSION,
    cursor_signature=None,
    cursor=CheckpointCursor((date(2026, 7, 18), "g1")),
    committed_batches=0,
    source_rows=0,
    written_chunks=0,
    reused_embeddings=0,
    embedded_chunks=0,
    max_updated_at=None,
    source_updated_before=None,
    completed=False,
    completed_at=None,
):
    return (
        run_id,
        source_table,
        scope_key,
        cursor_version,
        cursor_signature or ORDER.signature,
        encode_cursor(ORDER, cursor) if cursor is not None else None,
        committed_batches,
        source_rows,
        written_chunks,
        reused_embeddings,
        embedded_chunks,
        max_updated_at,
        source_updated_before,
        completed,
        completed_at or (NOW if completed else None),
    )


def _checkpoint_mapping(**overrides):
    columns = (
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
    values = dict(zip(columns, _checkpoint_row(), strict=True))
    values.update(overrides)
    return values


class _RecordingCursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.executed = []

    def execute(self, statement, params=()):
        self.executed.append((" ".join(statement.split()), params))
        return self

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None


class FakeCatalogCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params):
        self.executed.append((query, params))

    def fetchall(self):
        return self.rows


class FakeCatalogConnection:
    def __init__(self, rows):
        self.catalog_cursor = FakeCatalogCursor(rows)

    def cursor(self):
        return self.catalog_cursor


def test_resolve_checkpoint_order_maps_catalog_types_in_primary_key_order():
    conn = FakeCatalogConnection(
        [("sequence_id", "bigint", True), ("event_uuid", "uuid", True)]
    )

    order = ingest_script.resolve_checkpoint_order(conn, "plain_table", {})

    assert order == CheckpointOrder(
        "plain_table",
        (
            CheckpointOrderField("sequence_id", "integer"),
            CheckpointOrderField("event_uuid", "uuid"),
        ),
    )
    query, params = conn.catalog_cursor.executed[0]
    assert "format_type(a.atttypid, a.atttypmod)" in query
    assert "ORDER BY array_position(i.indkey, a.attnum)" in query
    assert params == ("plain_table", "public")


def test_resolve_checkpoint_order_distinguishes_postgres_timestamp_types():
    conn = FakeCatalogConnection(
        [
            ("local_time", "timestamp without time zone", True),
            ("absolute_time", "timestamp with time zone", True),
        ]
    )

    order = ingest_script.resolve_checkpoint_order(conn, "event", {})

    assert order.fields == (
        CheckpointOrderField("local_time", "datetime_naive"),
        CheckpointOrderField("absolute_time", "datetime"),
    )


@pytest.mark.parametrize(
    ("pg_type", "expected"),
    [
        ("timestamp(0) without time zone", "datetime_naive"),
        ("timestamp(3) without time zone", "datetime_naive"),
        ("timestamp(6) without time zone", "datetime_naive"),
        ("timestamp(0) with time zone", "datetime"),
        ("timestamp(3) with time zone", "datetime"),
        ("timestamp(6) with time zone", "datetime"),
    ],
)
def test_resolve_checkpoint_order_accepts_valid_timestamp_precision_forms(
    pg_type,
    expected,
):
    conn = FakeCatalogConnection([("occurred_at", pg_type, True)])

    order = ingest_script.resolve_checkpoint_order(conn, "event", {})

    assert order.fields == (CheckpointOrderField("occurred_at", expected),)


@pytest.mark.parametrize(
    "pg_type",
    [
        "timestamp() with time zone",
        "timestamp(7) with time zone",
        "timestamp(10) without time zone",
        "timestamp(-1) without time zone",
        "timestamp(03) with time zone",
        "timestamp(3) with local time zone",
        "timestamp(3)",
    ],
)
def test_resolve_checkpoint_order_rejects_invalid_timestamp_precision_forms(pg_type):
    conn = FakeCatalogConnection([("occurred_at", pg_type, True)])

    with pytest.raises(IngestCheckpointCursorTypeError):
        ingest_script.resolve_checkpoint_order(conn, "event", {})


def test_resolve_checkpoint_order_rejects_float_primary_key():
    conn = FakeCatalogConnection([("score", "double precision", True)])

    with pytest.raises(IngestCheckpointCursorTypeError) as raised:
        ingest_script.resolve_checkpoint_order(conn, "plain_table", {})

    assert raised.value.code == "INGEST_CHECKPOINT_CURSOR_TYPE_UNSUPPORTED"


def test_resolve_checkpoint_order_rejects_missing_primary_key():
    conn = FakeCatalogConnection([])

    with pytest.raises(IngestCheckpointCursorUnavailableError):
        ingest_script.resolve_checkpoint_order(conn, "plain_table", {})


def test_resolve_checkpoint_order_rejects_nullable_primary_key_field():
    conn = FakeCatalogConnection([("entity_id", "text", False)])

    with pytest.raises(IngestCheckpointCursorUnavailableError):
        ingest_script.resolve_checkpoint_order(conn, "plain_table", {})


@pytest.mark.parametrize(
    ("rows", "expected_error"),
    [
        ([], IngestCheckpointCursorUnavailableError),
        (
            [("score", "double precision", True)],
            IngestCheckpointCursorTypeError,
        ),
        (
            [("entity_id", "text", False)],
            IngestCheckpointCursorUnavailableError,
        ),
    ],
)
def test_generic_profile_cannot_bypass_catalog_with_configured_order(
    rows, expected_error
):
    conn = FakeCatalogConnection(rows)
    profile = {"checkpoint_order": (("configured_id", "integer"),)}

    with pytest.raises(expected_error):
        ingest_script.resolve_checkpoint_order(conn, "plain_table", profile)

    assert len(conn.catalog_cursor.executed) == 1


@pytest.mark.parametrize(
    ("scalar_type", "value"),
    [
        ("integer", 42),
        ("decimal", Decimal("3.140")),
        ("date", date(2026, 7, 18)),
        ("datetime", datetime(2026, 7, 18, 4, 0, tzinfo=UTC)),
        ("datetime_naive", datetime(2026, 7, 18, 4, 0)),
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


def test_datetime_naive_has_distinct_signature_from_legacy_datetime():
    legacy = CheckpointOrder(
        "event",
        (CheckpointOrderField("occurred_at", "datetime"),),
    )
    wall_clock = CheckpointOrder(
        "event",
        (CheckpointOrderField("occurred_at", "datetime_naive"),),
    )

    assert wall_clock.signature != legacy.signature


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


def test_datetime_naive_round_trip_preserves_wall_clock_without_offset():
    order = CheckpointOrder(
        source_table="event",
        fields=(CheckpointOrderField("occurred_at", "datetime_naive"),),
    )
    value = datetime(2026, 7, 18, 4, 0, 1, 123456)

    payload = encode_cursor(order, CheckpointCursor((value,)))
    decoded = decode_cursor(order, payload)

    assert payload["values"][0]["value"] == "2026-07-18T04:00:01.123456"
    assert decoded == CheckpointCursor((value,))
    assert decoded.values[0].tzinfo is None


def test_datetime_naive_rejects_aware_runtime_value():
    order = CheckpointOrder(
        source_table="event",
        fields=(CheckpointOrderField("occurred_at", "datetime_naive"),),
    )

    with pytest.raises(IngestCheckpointCursorTypeError):
        encode_cursor(
            order,
            CheckpointCursor((datetime(2026, 7, 18, 4, 0, tzinfo=UTC),)),
        )


@pytest.mark.parametrize(
    "stored_value",
    [
        "2026-07-18T04:00:00+00:00",
        "2026-07-18T13:00:00+09:00",
        "2026-07-18T04:00:00Z",
    ],
)
def test_datetime_naive_rejects_offset_bearing_stored_value(stored_value):
    order = CheckpointOrder(
        source_table="event",
        fields=(CheckpointOrderField("occurred_at", "datetime_naive"),),
    )
    payload = {
        "values": [
            {
                "field": "occurred_at",
                "type": "datetime_naive",
                "value": stored_value,
            }
        ]
    }

    with pytest.raises(IngestCheckpointCursorTypeError):
        decode_cursor(order, payload)


def test_legacy_datetime_signature_is_incompatible_with_datetime_naive_order():
    legacy = CheckpointOrder(
        "event",
        (CheckpointOrderField("occurred_at", "datetime"),),
    )
    wall_clock = CheckpointOrder(
        "event",
        (CheckpointOrderField("occurred_at", "datetime_naive"),),
    )
    cursor = _RecordingCursor(
        rows=[
            _checkpoint_row(
                source_table="event",
                cursor_signature=legacy.signature,
                cursor=None,
            )
        ]
    )

    with pytest.raises(IngestCheckpointIncompatibleError):
        IngestCheckpointSession.start(
            cursor,
            run_id=RUN_ID,
            source_table="event",
            scope_key=SCOPE_KEY,
            order=wall_clock,
        )


def test_repository_load_uses_explicit_columns_and_decodes_typed_cursor():
    cursor = _RecordingCursor(rows=[_checkpoint_row(source_updated_before=CUTOFF)])
    repository = IngestCheckpointRepository(ORDER)

    loaded = repository.load(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        for_update=True,
    )

    assert loaded == IngestCheckpoint(
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        cursor_version=CURSOR_VERSION,
        cursor_signature=ORDER.signature,
        cursor=CheckpointCursor((date(2026, 7, 18), "g1")),
        source_updated_before=CUTOFF,
    )
    sql, params = cursor.executed[0]
    assert "SELECT *" not in sql
    assert "cursor_payload" in sql
    assert "source_updated_before" in sql
    assert "completed_at" in sql
    assert "FOR UPDATE" in sql
    assert params == (RUN_ID, "game")


def test_advance_locks_identity_and_returns_monotonic_progress():
    cursor = _RecordingCursor(
        rows=[
            None,
            None,
            _checkpoint_row(
                cursor=CheckpointCursor((date(2026, 7, 18), "g2")),
                source_rows=2,
                written_chunks=3,
                reused_embeddings=1,
                embedded_chunks=2,
                committed_batches=1,
                max_updated_at=NOW,
                source_updated_before=CUTOFF,
            ),
        ]
    )
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
        requires_source_updated_before=True,
    )
    session.bind_source_updated_before(CUTOFF)

    updated = session.advance(
        cursor,
        next_cursor=CheckpointCursor((date(2026, 7, 18), "g2")),
        source_rows_delta=2,
        written_chunks_delta=3,
        reused_embeddings_delta=1,
        embedded_chunks_delta=2,
        max_updated_at=NOW,
    )

    load_sql, lock_sql, insert_sql = (statement for statement, _ in cursor.executed)
    assert "FOR UPDATE" not in load_sql
    assert "FOR UPDATE" in lock_sql
    assert "WHERE run_id = %s AND source_table = %s" in lock_sql
    assert "INSERT INTO ai_ingest_checkpoints" in insert_sql
    assert updated.source_rows == 2
    assert updated.written_chunks == 3
    assert updated.committed_batches == 1
    assert updated.max_updated_at == NOW
    assert updated.source_updated_before == CUTOFF
    assert session.current == updated
    insert_params = cursor.executed[2][1]
    assert insert_params[:5] == (
        RUN_ID,
        "game",
        SCOPE_KEY,
        CURSOR_VERSION,
        ORDER.signature,
    )
    assert insert_params[6:] == (1, 2, 3, 1, 2, NOW, CUTOFF)


def test_start_reuses_persisted_source_cutoff_without_rebinding():
    cursor = _RecordingCursor(
        rows=[
            _checkpoint_row(
                committed_batches=1,
                source_rows=1,
                source_updated_before=CUTOFF,
            )
        ]
    )

    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
        requires_source_updated_before=True,
    )

    assert session.source_updated_before == CUTOFF
    with pytest.raises(IngestCheckpointIncompatibleError):
        session.bind_source_updated_before(CUTOFF + timedelta(seconds=1))


def test_start_rejects_progressed_incomplete_checkpoint_missing_required_cutoff():
    cursor = _RecordingCursor(
        rows=[_checkpoint_row(committed_batches=1, source_rows=1)]
    )

    with pytest.raises(IngestCheckpointIncompatibleError, match="source update cutoff"):
        IngestCheckpointSession.start(
            cursor,
            run_id=RUN_ID,
            source_table="game",
            scope_key=SCOPE_KEY,
            order=ORDER,
            requires_source_updated_before=True,
        )


def test_start_allows_completed_legacy_checkpoint_without_required_cutoff():
    cursor = _RecordingCursor(
        rows=[
            _checkpoint_row(
                committed_batches=1,
                source_rows=1,
                completed=True,
            )
        ]
    )

    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
        requires_source_updated_before=True,
    )

    assert session.completed is True
    assert session.source_updated_before is None


@pytest.mark.parametrize(
    "stored_overrides",
    [
        {"scope_key": "other"},
        {"cursor_version": CURSOR_VERSION + 1},
        {"cursor_signature": "other"},
        {"run_id": UUID("22222222-2222-4222-8222-222222222222")},
        {"source_table": "other"},
    ],
)
def test_start_rejects_identity_scope_version_or_signature_mismatch(
    stored_overrides,
):
    cursor = _RecordingCursor(rows=[_checkpoint_row(**stored_overrides)])

    with pytest.raises(IngestCheckpointIncompatibleError):
        IngestCheckpointSession.start(
            cursor,
            run_id=RUN_ID,
            source_table="game",
            scope_key=SCOPE_KEY,
            order=ORDER,
        )


def test_start_reports_resumed_and_completed_state_from_persisted_row():
    cursor = _RecordingCursor(rows=[_checkpoint_row(completed=True)])

    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )

    assert session.resumed is True
    assert session.completed is True


def test_advance_rejects_stale_persisted_state_before_mutating():
    initial_row = _checkpoint_row(
        committed_batches=1,
        source_rows=1,
        written_chunks=1,
    )
    stale_row = _checkpoint_row(
        committed_batches=1,
        source_rows=2,
        written_chunks=1,
    )
    cursor = _RecordingCursor(rows=[initial_row, stale_row])
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )
    initial = session.current

    with pytest.raises(IngestCheckpointIncompatibleError):
        session.advance(
            cursor,
            next_cursor=CheckpointCursor((date(2026, 7, 18), "g2")),
            source_rows_delta=1,
            written_chunks_delta=1,
            reused_embeddings_delta=0,
            embedded_chunks_delta=1,
            max_updated_at=NOW,
        )

    assert len(cursor.executed) == 2
    assert "FOR UPDATE" in cursor.executed[1][0]
    assert session.current == initial


def test_zero_output_advance_increments_batch_and_uses_absolute_counters():
    initial_row = _checkpoint_row(
        committed_batches=4,
        source_rows=7,
        written_chunks=5,
        reused_embeddings=2,
        embedded_chunks=3,
        max_updated_at=NOW,
    )
    returned_row = _checkpoint_row(
        cursor=CheckpointCursor((date(2026, 7, 18), "g2")),
        committed_batches=5,
        source_rows=9,
        written_chunks=5,
        reused_embeddings=2,
        embedded_chunks=3,
        max_updated_at=NOW,
    )
    cursor = _RecordingCursor(rows=[initial_row, initial_row, returned_row])
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )

    updated = session.advance(
        cursor,
        next_cursor=CheckpointCursor((date(2026, 7, 18), "g2")),
        source_rows_delta=2,
        written_chunks_delta=0,
        reused_embeddings_delta=0,
        embedded_chunks_delta=0,
        max_updated_at=None,
    )

    assert updated.committed_batches == 5
    assert updated.source_rows == 9
    assert updated.written_chunks == 5
    update_sql, update_params = cursor.executed[2]
    assert "committed_batches = %s" in update_sql
    assert "committed_batches +" not in update_sql
    assert update_params[1:7] == (5, 9, 5, 2, 3, NOW)
    assert update_params[-2:] == (RUN_ID, "game")


@pytest.mark.parametrize(
    "deltas",
    [
        (-1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 0, -1),
    ],
)
def test_advance_rejects_negative_deltas(deltas):
    cursor = _RecordingCursor(rows=[None])
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )

    with pytest.raises(ValueError, match="nonnegative"):
        session.advance(
            cursor,
            next_cursor=CheckpointCursor((date(2026, 7, 18), "g1")),
            source_rows_delta=deltas[0],
            written_chunks_delta=deltas[1],
            reused_embeddings_delta=deltas[2],
            embedded_chunks_delta=deltas[3],
            max_updated_at=None,
        )

    assert len(cursor.executed) == 1
    assert session.current is None


def test_advance_reopens_completed_checkpoint_and_clears_completion_time():
    completed_row = _checkpoint_row(
        committed_batches=1,
        source_rows=1,
        completed=True,
    )
    returned_row = _checkpoint_row(
        cursor=CheckpointCursor((date(2026, 7, 18), "g2")),
        committed_batches=2,
        source_rows=2,
    )
    cursor = _RecordingCursor(rows=[completed_row, completed_row, returned_row])
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )

    updated = session.advance(
        cursor,
        next_cursor=CheckpointCursor((date(2026, 7, 18), "g2")),
        source_rows_delta=1,
        written_chunks_delta=0,
        reused_embeddings_delta=0,
        embedded_chunks_delta=0,
        max_updated_at=None,
    )

    assert updated.completed is False
    assert session.completed is False
    update_sql = cursor.executed[2][0]
    assert "completed = false" in update_sql
    assert "completed_at = NULL" in update_sql


def test_advance_missing_returning_row_preserves_session_state():
    cursor = _RecordingCursor(rows=[None, None, None])
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )

    with pytest.raises(IngestCheckpointIncompatibleError):
        session.advance(
            cursor,
            next_cursor=CheckpointCursor((date(2026, 7, 18), "g1")),
            source_rows_delta=1,
            written_chunks_delta=0,
            reused_embeddings_delta=0,
            embedded_chunks_delta=0,
            max_updated_at=None,
        )

    assert session.current is None


def test_advance_accepts_canonical_return_for_naive_datetime_cursor():
    datetime_order = CheckpointOrder(
        source_table="event",
        fields=(CheckpointOrderField("occurred_at", "datetime"),),
    )
    naive_cursor = CheckpointCursor((datetime(2026, 7, 18, 4, 0),))
    canonical_cursor = CheckpointCursor((datetime(2026, 7, 18, 4, 0, tzinfo=UTC),))
    returned_row = (
        RUN_ID,
        "event",
        SCOPE_KEY,
        CURSOR_VERSION,
        datetime_order.signature,
        encode_cursor(datetime_order, canonical_cursor),
        1,
        1,
        0,
        0,
        0,
        None,
        None,
        False,
        None,
    )
    cursor = _RecordingCursor(rows=[None, None, returned_row])
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="event",
        scope_key=SCOPE_KEY,
        order=datetime_order,
    )

    updated = session.advance(
        cursor,
        next_cursor=naive_cursor,
        source_rows_delta=1,
        written_chunks_delta=0,
        reused_embeddings_delta=0,
        embedded_chunks_delta=0,
        max_updated_at=None,
    )

    assert updated.cursor == canonical_cursor


def test_complete_can_create_zero_row_checkpoint():
    cursor = _RecordingCursor(
        rows=[
            None,
            None,
            _checkpoint_row(
                cursor=None,
                source_updated_before=CUTOFF,
                completed=True,
            ),
        ]
    )
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
        requires_source_updated_before=True,
    )
    session.bind_source_updated_before(CUTOFF)

    completed = session.complete(cursor)

    assert completed.completed is True
    assert completed.cursor is None
    assert completed.source_rows == 0
    assert completed.committed_batches == 0
    assert completed.source_updated_before == CUTOFF
    assert session.resumed is False
    assert session.completed is True
    assert "FOR UPDATE" in cursor.executed[1][0]
    assert "INSERT INTO ai_ingest_checkpoints" in cursor.executed[2][0]
    assert CUTOFF in cursor.executed[2][1]


def test_complete_updates_only_completion_fields_and_never_increments_batches():
    initial_row = _checkpoint_row(
        committed_batches=4,
        source_rows=7,
        written_chunks=5,
        reused_embeddings=2,
        embedded_chunks=3,
        max_updated_at=NOW,
    )
    completed_row = _checkpoint_row(
        committed_batches=4,
        source_rows=7,
        written_chunks=5,
        reused_embeddings=2,
        embedded_chunks=3,
        max_updated_at=NOW,
        completed=True,
    )
    cursor = _RecordingCursor(rows=[initial_row, initial_row, completed_row])
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )

    completed = session.complete(cursor)

    assert completed.committed_batches == 4
    update_sql, params = cursor.executed[2]
    set_clause = update_sql.split("SET", 1)[1].split("WHERE", 1)[0]
    assert "completed = true" in set_clause
    assert "completed_at = clock_timestamp()" in set_clause
    assert "updated_at = clock_timestamp()" in set_clause
    assert "committed_batches" not in set_clause
    assert "source_rows" not in set_clause
    assert params == (None, RUN_ID, "game")


def test_complete_rejects_stale_or_missing_returning_state_without_mutating():
    initial_row = _checkpoint_row(committed_batches=1, source_rows=1)
    stale_row = _checkpoint_row(committed_batches=2, source_rows=1)
    stale_cursor = _RecordingCursor(rows=[initial_row, stale_row])
    stale_session = IngestCheckpointSession.start(
        stale_cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )
    stale_initial = stale_session.current

    with pytest.raises(IngestCheckpointIncompatibleError):
        stale_session.complete(stale_cursor)

    assert stale_session.current == stale_initial
    assert len(stale_cursor.executed) == 2

    missing_cursor = _RecordingCursor(rows=[initial_row, initial_row, None])
    missing_session = IngestCheckpointSession.start(
        missing_cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )
    missing_initial = missing_session.current

    with pytest.raises(IngestCheckpointIncompatibleError):
        missing_session.complete(missing_cursor)

    assert missing_session.current == missing_initial


def test_advance_insert_uses_conflict_fence_and_rejects_lost_insert_race():
    cursor = _RecordingCursor(rows=[None, None, None])
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )

    with pytest.raises(IngestCheckpointIncompatibleError, match="no durable row"):
        session.advance(
            cursor,
            next_cursor=CheckpointCursor((date(2026, 7, 18), "g1")),
            source_rows_delta=1,
            written_chunks_delta=0,
            reused_embeddings_delta=0,
            embedded_chunks_delta=0,
            max_updated_at=None,
        )

    insert_sql = cursor.executed[2][0]
    assert "ON CONFLICT (run_id, source_table) DO NOTHING" in insert_sql
    assert "RETURNING" in insert_sql
    assert session.current is None


def test_complete_insert_uses_conflict_fence_and_rejects_lost_insert_race():
    cursor = _RecordingCursor(rows=[None, None, None])
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )

    with pytest.raises(IngestCheckpointIncompatibleError, match="no durable row"):
        session.complete(cursor)

    insert_sql = cursor.executed[2][0]
    assert "ON CONFLICT (run_id, source_table) DO NOTHING" in insert_sql
    assert "RETURNING" in insert_sql
    assert session.current is None


@pytest.mark.parametrize(
    "counter_name",
    [
        "committed_batches",
        "source_rows",
        "written_chunks",
        "reused_embeddings",
        "embedded_chunks",
    ],
)
def test_load_rejects_negative_counter_mapping(counter_name):
    cursor = _RecordingCursor(rows=[_checkpoint_mapping(**{counter_name: -1})])

    with pytest.raises(IngestCheckpointIncompatibleError):
        IngestCheckpointRepository(ORDER).load(
            cursor,
            run_id=RUN_ID,
            source_table="game",
        )


def test_load_rejects_negative_counter_tuple():
    row = list(_checkpoint_row())
    row[6] = -1
    cursor = _RecordingCursor(rows=[tuple(row)])

    with pytest.raises(IngestCheckpointIncompatibleError):
        IngestCheckpointRepository(ORDER).load(
            cursor,
            run_id=RUN_ID,
            source_table="game",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"cursor_payload": None, "source_rows": 1},
        {"completed": True, "completed_at": None},
        {"completed": False, "completed_at": NOW},
        {"run_id": str(RUN_ID)},
        {"source_table": ""},
        {"scope_key": ""},
    ],
)
def test_load_rejects_invalid_mapping_row_invariants(overrides):
    cursor = _RecordingCursor(rows=[_checkpoint_mapping(**overrides)])

    with pytest.raises(IngestCheckpointIncompatibleError):
        IngestCheckpointRepository(ORDER).load(
            cursor,
            run_id=RUN_ID,
            source_table="game",
        )


def test_load_wraps_invalid_stored_cursor_type_as_incompatible():
    payload = {
        "values": [
            {"field": "game_date", "type": "date", "value": "bad-date"},
            {"field": "game_id", "type": "text", "value": "g1"},
        ]
    }
    cursor = _RecordingCursor(
        rows=[_checkpoint_mapping(cursor_payload=payload, source_rows=1)]
    )

    with pytest.raises(IngestCheckpointIncompatibleError) as raised:
        IngestCheckpointRepository(ORDER).load(
            cursor,
            run_id=RUN_ID,
            source_table="game",
        )

    assert isinstance(raised.value.__cause__, IngestCheckpointCursorTypeError)


@pytest.mark.parametrize("rows", [[], [_checkpoint_row()]])
def test_start_rejects_order_source_mismatch_before_database_load(rows):
    cursor = _RecordingCursor(rows=rows)

    with pytest.raises(IngestCheckpointIncompatibleError):
        IngestCheckpointSession.start(
            cursor,
            run_id=RUN_ID,
            source_table="other",
            scope_key=SCOPE_KEY,
            order=ORDER,
        )

    assert cursor.executed == []


@pytest.mark.parametrize(
    ("persisted", "candidate", "expected"),
    [
        (NOW, NOW - timedelta(hours=1), NOW),
        (NOW, NOW + timedelta(hours=1), NOW + timedelta(hours=1)),
        (
            NOW.replace(tzinfo=None),
            NOW + timedelta(hours=1),
            NOW + timedelta(hours=1),
        ),
        (
            NOW,
            (NOW + timedelta(hours=1)).replace(tzinfo=None),
            NOW + timedelta(hours=1),
        ),
    ],
)
def test_advance_max_updated_at_is_monotonic_and_canonical(
    persisted,
    candidate,
    expected,
):
    initial_row = _checkpoint_row(
        committed_batches=1,
        source_rows=1,
        max_updated_at=persisted,
    )
    returned_row = _checkpoint_row(
        cursor=CheckpointCursor((date(2026, 7, 18), "g2")),
        committed_batches=2,
        source_rows=2,
        max_updated_at=expected,
    )
    cursor = _RecordingCursor(rows=[initial_row, initial_row, returned_row])
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )

    updated = session.advance(
        cursor,
        next_cursor=CheckpointCursor((date(2026, 7, 18), "g2")),
        source_rows_delta=1,
        written_chunks_delta=0,
        reused_embeddings_delta=0,
        embedded_chunks_delta=0,
        max_updated_at=candidate,
    )

    assert updated.max_updated_at == expected
    assert updated.max_updated_at.tzinfo is UTC
    assert cursor.executed[2][1][6] == expected
