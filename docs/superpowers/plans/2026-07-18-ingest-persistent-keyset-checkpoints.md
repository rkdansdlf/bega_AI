# Persistent Ingest Keyset Checkpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume every checkpoint-capable internal database ingestion table strictly after its last atomically committed source key while preserving success-only watermarks and static-document behavior.

**Architecture:** Add an additive per-run/per-table checkpoint table and a focused typed-cursor module. Custom queries use an explicit ascending output-key registry, generic tables use real PostgreSQL primary keys, and the synchronous destination transaction commits chunk mutations and checkpoint progress together. The worker keeps cumulative run summaries but records Prometheus table counters from current-attempt deltas.

**Tech Stack:** Python 3.14, FastAPI lifespan, psycopg 3, PostgreSQL JSONB and row-value keyset comparisons, Prometheus client, pytest.

## Global Constraints

- Baseball data comes only from the internal database, trusted internal sync paths, static project documents, or operator-provided manual data.
- Never add crawling, scraping, web-search repair, browser automation, or external baseball API calls.
- Missing required trusted-source cursor fields surface `MANUAL_BASEBALL_DATA_REQUIRED` with `operator_manual_data`; never synthesize a cursor.
- Static `source_file` profiles remain atomic and uncheckpointed.
- Only leased database-table ingestion writes checkpoints; non-leased CLI behavior remains unchanged.
- Checkpointed calls require `limit=None` and `row_stale_cleanup="off"`.
- Every checkpoint order is ascending, non-null, stable, and unique.
- `rag_chunks` mutations and checkpoint advancement share one destination transaction and one lease fence.
- The successful table watermark advances only after the full run reaches `SUCCEEDED`.
- Preserve checkpoint rows after every terminal state; do not add retention automation.
- Do not use a shared or production database, external embedding call, or external baseball source in tests.
- Use `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python` for every Python and pytest command in this worktree.
- Run each RED test before production code, confirm the expected failure, then implement only enough for GREEN.

---

## File Responsibility Map

- `app/db/migrations/004_ai_ingest_checkpoints.sql`: additive checkpoint schema, constraints, and retention index.
- `app/db/schema_contract.py`: managed-mode required checkpoint columns and index.
- `app/deps.py`: ordered application of migrations 003 and 004 in compatibility mode.
- `app/core/ingest_checkpoints.py`: typed order/cursor models, codecs, errors, durable record mapping, and synchronous transactional repository/session.
- `scripts/ingest_from_kbo.py`: explicit custom-query cursor registry, generic PK discovery, keyset SQL, one-row lookahead, and batch integration.
- `app/core/ingest_runs.py`: cumulative checkpoint metadata plus current-attempt metric deltas on `IngestTableResult`.
- `app/core/ingest_run_store.py`: sanitized cumulative checkpoint fields in terminal table summaries.
- `app/core/ingest_sources.py`: one bounded source-table metric-label normalizer shared by worker and script.
- `app/core/ingest_worker.py`: scope propagation and attempt-delta metric accounting.
- `app/observability/metrics.py`: bounded checkpoint event counter.
- `docs/data-sync-orchestration-runbook.md`: schema, resume, errors, metrics, rollback, and residual-risk operations.
- `tests/test_ingest_checkpoints.py`: cursor and repository unit tests.
- `tests/test_ingest_checkpoint_integration.py`: transaction, crash-boundary, lookahead, completion, and resume behavior.
- Existing migration, query, worker, run-store, metrics, and policy tests: regression coverage at their current paths.

---

### Task 1: Add the durable checkpoint schema and startup contract

**Files:**
- Create: `app/db/migrations/004_ai_ingest_checkpoints.sql`
- Modify: `app/db/schema_contract.py`
- Modify: `app/deps.py`
- Modify: `tests/test_ai_schema_migrations.py`
- Modify: `tests/test_db_schema_contract.py`
- Modify: `tests/test_schema_startup_mode.py`

**Interfaces:**
- Consumes: migration 003's `ai_ingest_runs(run_id)` primary key.
- Produces: `ai_ingest_checkpoints` and `idx_ai_ingest_checkpoints_updated_at`; `deps.INGEST_ORCHESTRATION_MIGRATION_SQLS: tuple[str, ...]` ordered as 003 then 004.

- [ ] **Step 1: Write failing migration and schema-contract tests**

Add these assertions:

```python
def test_ingest_checkpoint_migration_defines_durable_progress_table():
    sql = (MIGRATION_DIR / "004_ai_ingest_checkpoints.sql").read_text(
        encoding="utf-8"
    )

    assert "CREATE TABLE IF NOT EXISTS ai_ingest_checkpoints" in sql
    assert "PRIMARY KEY (run_id, source_table)" in sql
    assert "cursor_signature varchar(64) NOT NULL" in sql
    assert "cursor_payload jsonb" in sql
    assert "source_rows = 0 OR cursor_payload IS NOT NULL" in sql
    assert "idx_ai_ingest_checkpoints_updated_at" in sql


def test_complete_contract_includes_ingest_checkpoints():
    assert set(REQUIRED_COLUMNS["ai_ingest_checkpoints"]) == {
        "run_id", "source_table", "scope_key", "cursor_version",
        "cursor_signature", "cursor_payload", "committed_batches",
        "source_rows", "written_chunks", "reused_embeddings",
        "embedded_chunks", "max_updated_at", "source_updated_before",
        "completed", "completed_at", "created_at", "updated_at",
    }
    assert "idx_ai_ingest_checkpoints_updated_at" in REQUIRED_INDEXES
```

Change the compatibility test to require two ordered executions:

```python
def test_auto_schema_compatibility_ensures_ingest_orchestration_tables():
    pool = _RecordingPool()

    asyncio.run(deps._ensure_ingest_orchestration_schema(pool))

    executed = pool.connection_instance.executed
    assert len(executed) == 2
    assert "CREATE TABLE IF NOT EXISTS ai_ingest_runs" in executed[0]
    assert "CREATE TABLE IF NOT EXISTS ai_ingest_checkpoints" in executed[1]
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_ai_schema_migrations.py \
  tests/test_db_schema_contract.py \
  tests/test_schema_startup_mode.py::test_auto_schema_compatibility_ensures_ingest_orchestration_tables -q
```

Expected: failure because migration 004 and `REQUIRED_COLUMNS["ai_ingest_checkpoints"]` do not exist and compatibility mode executes only migration 003.

- [ ] **Step 3: Add the migration and ordered startup implementation**

Create `004_ai_ingest_checkpoints.sql` with:

```sql
CREATE TABLE IF NOT EXISTS ai_ingest_checkpoints (
    run_id uuid NOT NULL REFERENCES ai_ingest_runs(run_id),
    source_table varchar(128) NOT NULL,
    scope_key varchar(64) NOT NULL,
    cursor_version integer NOT NULL,
    cursor_signature varchar(64) NOT NULL,
    cursor_payload jsonb,
    committed_batches bigint NOT NULL DEFAULT 0,
    source_rows bigint NOT NULL DEFAULT 0,
    written_chunks bigint NOT NULL DEFAULT 0,
    reused_embeddings bigint NOT NULL DEFAULT 0,
    embedded_chunks bigint NOT NULL DEFAULT 0,
    max_updated_at timestamptz,
    source_updated_before timestamptz,
    completed boolean NOT NULL DEFAULT false,
    completed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, source_table),
    CONSTRAINT ck_ai_ingest_checkpoint_counts_nonnegative CHECK (
        committed_batches >= 0
        AND source_rows >= 0
        AND written_chunks >= 0
        AND reused_embeddings >= 0
        AND embedded_chunks >= 0
    ),
    CONSTRAINT ck_ai_ingest_checkpoint_cursor_present CHECK (
        source_rows = 0 OR cursor_payload IS NOT NULL
    ),
    CONSTRAINT ck_ai_ingest_checkpoint_completion_time CHECK (
        (completed = false AND completed_at IS NULL)
        OR (completed = true AND completed_at IS NOT NULL)
    )
);

ALTER TABLE ai_ingest_checkpoints
    ADD COLUMN IF NOT EXISTS source_updated_before timestamptz;

CREATE INDEX IF NOT EXISTS idx_ai_ingest_checkpoints_updated_at
    ON ai_ingest_checkpoints (updated_at);
```

Add the exact column tuple and index name in `schema_contract.py`. Replace the single migration constant in `deps.py` with:

```python
INGEST_ORCHESTRATION_MIGRATION_SQLS = tuple(
    (
        Path(__file__).resolve().parent / "db" / "migrations" / filename
    ).read_text(encoding="utf-8")
    for filename in (
        "003_ai_ingest_orchestration.sql",
        "004_ai_ingest_checkpoints.sql",
    )
)
```

Execute the tuple in order on one compatibility connection:

```python
async def _ensure_ingest_orchestration_schema(pool) -> None:
    """Provision the durable queue and checkpoints in compatibility mode."""

    async with pool.connection() as conn:
        for migration_sql in INGEST_ORCHESTRATION_MIGRATION_SQLS:
            await conn.execute(migration_sql)
    logger.info("[Lifespan] AI ingest orchestration tables ensured")
```

- [ ] **Step 4: Run focused schema tests and verify GREEN**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 5: Commit the schema task**

```bash
git add app/db/migrations/004_ai_ingest_checkpoints.sql app/db/schema_contract.py app/deps.py tests/test_ai_schema_migrations.py tests/test_db_schema_contract.py tests/test_schema_startup_mode.py
git commit -m "feat: persist ingest batch checkpoints"
```

---

### Task 2: Implement typed checkpoint cursors and compatibility errors

**Files:**
- Create: `app/core/ingest_checkpoints.py`
- Create: `tests/test_ingest_checkpoints.py`

**Interfaces:**
- Produces: `CURSOR_VERSION`, `CheckpointOrderField`, `CheckpointOrder`, `CheckpointCursor`, `IngestCheckpoint`, `IngestCheckpointError`, `IngestCheckpointCursorUnavailableError`, `IngestCheckpointIncompatibleError`, `IngestCheckpointCursorTypeError`, `IngestCheckpointOrderError`, `IngestCheckpointStaleCleanupError`, `IngestCheckpointMissingFieldError`, `encode_cursor`, `decode_cursor`, `cursor_from_row`, and `ensure_cursor_advances`.
- Consumes: only standard-library types and sanitized mappings; no settings, network, or FastAPI dependencies.

- [ ] **Step 1: Write failing codec, signature, and ordering tests**

Create `tests/test_ingest_checkpoints.py` with these core cases:

```python
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID

import pytest

from app.core.ingest_checkpoints import (
    CheckpointCursor,
    CheckpointOrder,
    CheckpointOrderField,
    IngestCheckpointIncompatibleError,
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
```

- [ ] **Step 2: Run the new test file and verify RED**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_checkpoints.py -q
```

Expected: collection fails because `app.core.ingest_checkpoints` does not exist.

- [ ] **Step 3: Add the typed cursor domain implementation**

Create the module with explicit error codes and frozen dataclasses:

```python
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
    "integer", "decimal", "date", "datetime", "datetime_naive",
    "uuid", "text", "boolean"
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
```

Implement canonical scalar normalization with `datetime` checked before `date`,
aware timestamps normalized as instants, `datetime_naive` wall-clock fields kept
offset-free, and boolean rejected from the integer branch:

```python
def _normalize_cursor_value(kind: CursorScalarType, value: Any) -> Any:
    if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if kind == "decimal" and isinstance(value, Decimal):
        return value
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
    items = payload.get("values")
    if not isinstance(items, list) or len(items) != len(order.fields):
        raise IngestCheckpointIncompatibleError("stored cursor arity mismatch")
    values = []
    for field, item in zip(order.fields, items, strict=True):
        if not isinstance(item, Mapping) or item.get("field") != field.name or item.get("type") != field.scalar_type:
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
    del order
    if current.values <= previous.values:
        raise IngestCheckpointOrderError("checkpoint cursor did not advance")
```

Define the durable record used by later tasks:

```python
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
```

- [ ] **Step 4: Run the codec tests and verify GREEN**

Run the command from Step 2.

Expected: all tests pass.

- [ ] **Step 5: Commit the typed cursor task**

```bash
git add app/core/ingest_checkpoints.py tests/test_ingest_checkpoints.py
git commit -m "feat: define typed ingest checkpoint cursors"
```

---

### Task 3: Add explicit custom orders and keyset source queries

**Files:**
- Modify: `scripts/ingest_from_kbo.py`
- Modify: `tests/test_ingest_query.py`
- Modify: `tests/test_ingest_checkpoints.py`

**Interfaces:**
- Consumes: `CheckpointOrder`, `CheckpointOrderField`, `CheckpointCursor`, and cursor errors from Task 2.
- Produces: `resolve_checkpoint_order(conn, table_name, profile) -> CheckpointOrder`, and an extended `build_select_query(..., checkpoint_order=None, resume_cursor=None)`.

- [ ] **Step 1: Write failing profile coverage, generic PK, and custom keyset tests**

Add tests:

```python
def test_every_custom_database_profile_declares_checkpoint_order():
    missing = [
        table
        for table, profile in TABLE_PROFILES.items()
        if "source_file" not in profile
        and "select_sql" in profile
        and not profile.get("checkpoint_order")
    ]
    assert missing == []


def test_custom_checkpoint_query_wraps_output_aliases(sample_since):
    profile = TABLE_PROFILES["game"]
    order = resolve_checkpoint_order(None, "game", profile)
    cursor = CheckpointCursor((41,))

    query, params = build_select_query(
        table="game",
        profile=profile,
        pk_columns=["id"],
        limit=None,
        season_year=2026,
        since=sample_since,
        checkpoint_order=order,
        resume_cursor=cursor,
    )

    assert "WITH checkpoint_source AS" in query
    assert 'ROW("id") > ROW(%s)' in query
    assert query.rstrip().endswith('ORDER BY "id" ASC')
    assert params == (2026, sample_since, 41)


def test_generic_checkpoint_query_uses_composite_primary_key():
    order = CheckpointOrder(
        "plain_table",
        (
            CheckpointOrderField("season", "integer"),
            CheckpointOrderField("entity_id", "text"),
        ),
    )
    query, params = build_select_query(
        table="plain_table",
        profile={"season_filter_column": None, "since_filter_column": None},
        pk_columns=["season", "entity_id"],
        limit=None,
        season_year=None,
        since=None,
        checkpoint_order=order,
        resume_cursor=CheckpointCursor((2025, "P100")),
    )

    rendered = str(query)
    assert "ROW" in rendered and "season" in rendered and "entity_id" in rendered
    assert params == (2025, "P100")
```

Use a fake catalog connection to assert `resolve_checkpoint_order` maps
`bigint` to `integer`, `uuid` to `uuid`, and rejects a float primary key with
`INGEST_CHECKPOINT_CURSOR_TYPE_UNSUPPORTED`.

- [ ] **Step 2: Run query and cursor tests and verify RED**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_query.py tests/test_ingest_checkpoints.py -q
```

Expected: failures because profiles lack `checkpoint_order`, the resolver is undefined, and `build_select_query` has no checkpoint arguments.

- [ ] **Step 3: Add the explicit order registry and PostgreSQL type mapping**

Attach this exact registry to database profiles after `TABLE_PROFILES` is built:

```python
CUSTOM_CHECKPOINT_ORDERS = {
    "player_season_batting": (("id", "integer"),),
    "player_season_pitching": (("id", "integer"),),
    "game": (("id", "integer"),),
    "game_flow_summary": (("game_id", "text"),),
    "game_batting_stats": (("id", "integer"),),
    "game_pitching_stats": (("id", "integer"),),
    "game_inning_scores": (("id", "integer"),),
    "game_lineups": (("id", "integer"),),
    "game_metadata": (("game_id", "text"),),
    "team_history": (("id", "integer"),),
    "awards": (("id", "integer"),),
    "player_movements": (("id", "integer"),),
    "team_franchises": (("id", "integer"),),
    "player_basic": (("player_id", "text"),),
    "team_name_mapping": (("full_name", "text"),),
    "team_profiles": (("id", "integer"),),
    "team_season_batting": (("id", "integer"),),
    "team_season_pitching": (("id", "integer"),),
    "stat_rankings": (("id", "integer"),),
    "game_summary": (("id", "integer"),),
}

for source_table, checkpoint_order in CUSTOM_CHECKPOINT_ORDERS.items():
    TABLE_PROFILES[source_table]["checkpoint_order"] = checkpoint_order
    TABLE_PROFILES[source_table]["checkpoint_query_version"] = "1"
```

Implement generic PK discovery with `format_type(a.atttypid, a.atttypmod)` and
this closed mapping:

```python
def _postgres_type_to_cursor_type(pg_type: str) -> CursorScalarType:
    normalized = pg_type.lower()
    if normalized in {"smallint", "integer", "bigint"}:
        return "integer"
    if normalized.startswith(("numeric", "decimal")):
        return "decimal"
    if normalized == "date":
        return "date"
    if re.fullmatch(r"timestamp(?:\([0-6]\))? with time zone", normalized):
        return "datetime"
    if re.fullmatch(r"timestamp(?:\([0-6]\))? without time zone", normalized):
        return "datetime_naive"
    if normalized == "uuid":
        return "uuid"
    if normalized in {"text", "character", "character varying"} or normalized.startswith(("character(", "character varying(")):
        return "text"
    if normalized == "boolean":
        return "boolean"
    raise IngestCheckpointCursorTypeError(f"unsupported primary-key type: {pg_type}")
```

`resolve_checkpoint_order` uses the explicit profile tuple for custom SQL and the
catalog result for generic tables. An empty catalog result raises
`IngestCheckpointCursorUnavailableError`.

- [ ] **Step 4: Extend query construction for canonical keyset mode**

Keep the existing path byte-for-byte compatible when `checkpoint_order is None`.
When an order is supplied:

```python
def _quoted_checkpoint_fields(order: CheckpointOrder) -> str:
    for field in order.fields:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field.name) is None:
            raise IngestCheckpointCursorUnavailableError("unsafe cursor field")
    return ", ".join(f'"{field.name}"' for field in order.fields)
```

For custom SQL, strip only the last top-level `ORDER BY`, apply existing filters
inside the base query, wrap it as `checkpoint_source`, append the row comparison
when `resume_cursor` exists, and always append canonical ascending order. For the
generic psycopg `sql.SQL` path, add the same row comparison to `where_parts` and
replace its order with `checkpoint_order.fields`. Append resume values after the
existing season/since/date params.

- [ ] **Step 5: Run query and cursor tests and verify GREEN**

Run the command from Step 2.

Expected: all selected tests pass, including all pre-existing query-shape tests.

- [ ] **Step 6: Commit the keyset query task**

```bash
git add scripts/ingest_from_kbo.py tests/test_ingest_query.py tests/test_ingest_checkpoints.py
git commit -m "feat: query ingest sources with stable keysets"
```

---

### Task 4: Implement the synchronous transactional checkpoint repository

**Files:**
- Modify: `app/core/ingest_checkpoints.py`
- Modify: `tests/test_ingest_checkpoints.py`

**Interfaces:**
- Consumes: Task 2 domain models and a synchronous psycopg-style destination cursor.
- Produces: `IngestCheckpointRepository.load`, `advance`, and `complete`, plus `IngestCheckpointSession` with durable current state and `resumed` flag.

- [ ] **Step 1: Write failing repository tests**

Add a recording cursor that scripts `fetchone()` values and records SQL. Cover:

```python
def test_advance_locks_identity_and_returns_monotonic_progress():
    cursor = _RecordingCursor(
        rows=[
            None,
            None,
            _checkpoint_row(
                source_rows=2,
                written_chunks=3,
                committed_batches=1,
            ),
        ]
    )
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
        written_chunks_delta=3,
        reused_embeddings_delta=1,
        embedded_chunks_delta=2,
        max_updated_at=NOW,
    )

    sql = " ".join(statement for statement, _ in cursor.executed)
    assert "FOR UPDATE" in sql
    assert "INSERT INTO ai_ingest_checkpoints" in sql
    assert updated.source_rows == 2
    assert updated.written_chunks == 3
    assert updated.committed_batches == 1


def test_start_rejects_scope_or_signature_mismatch():
    cursor = _RecordingCursor(rows=[_checkpoint_row(scope_key="other")])
    with pytest.raises(IngestCheckpointIncompatibleError):
        IngestCheckpointSession.start(
            cursor,
            run_id=RUN_ID,
            source_table="game",
            scope_key=SCOPE_KEY,
            order=ORDER,
        )


def test_complete_can_create_zero_row_checkpoint():
    cursor = _RecordingCursor(
        rows=[None, None, _checkpoint_row(completed=True)]
    )
    session = IngestCheckpointSession.start(
        cursor,
        run_id=RUN_ID,
        source_table="game",
        scope_key=SCOPE_KEY,
        order=ORDER,
    )
    completed = session.complete(cursor)
    assert completed.completed is True
    assert completed.cursor is None
    assert completed.source_rows == 0
```

Also test that an advance whose database row differs from the session's expected
cursor raises `IngestCheckpointIncompatibleError`, and that `complete` never
increments `committed_batches`.

- [ ] **Step 2: Run repository tests and verify RED**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_checkpoints.py -q
```

Expected: failures because repository and session interfaces are undefined.

- [ ] **Step 3: Implement load, advance, and complete under a row lock**

Use one explicit column list for row mapping. `load(..., for_update=True)` appends
`FOR UPDATE`. `IngestCheckpointSession.start` performs an unlocked load and
validates:

```python
stored.scope_key == scope_key
stored.cursor_version == CURSOR_VERSION
stored.cursor_signature == order.signature
```

`advance` performs a locked reload, verifies it equals the session's current
durable cursor and counters, then inserts or updates absolute monotonic values.
The update sets `completed=false`, `completed_at=NULL`, and
`updated_at=clock_timestamp()`. It increments `committed_batches` by exactly one
for each data/progress commit, including a zero-output source batch.

`complete` performs the same locked compatibility check, inserts a zero-row row
when needed, and updates only `completed`, `completed_at`, and `updated_at`; it
does not increment batch or data counters. Both mutations use `RETURNING` and
replace `session.current` only after a returned row is decoded successfully.

The session exposes these exact call signatures:

- `IngestCheckpointSession.start(db_cursor: Any, *, run_id: UUID, source_table: str, scope_key: str, order: CheckpointOrder) -> IngestCheckpointSession`
- `session.advance(db_cursor: Any, *, next_cursor: CheckpointCursor, source_rows_delta: int, written_chunks_delta: int, reused_embeddings_delta: int, embedded_chunks_delta: int, max_updated_at: datetime | None) -> IngestCheckpoint`
- `session.complete(db_cursor: Any) -> IngestCheckpoint`

Its state properties are:

```python
@property
def resumed(self) -> bool:
    return self.initial is not None

@property
def completed(self) -> bool:
    return bool(self.current and self.current.completed)
```

- [ ] **Step 4: Run repository tests and verify GREEN**

Run the command from Step 2.

Expected: all tests pass.

- [ ] **Step 5: Commit the repository task**

```bash
git add app/core/ingest_checkpoints.py tests/test_ingest_checkpoints.py
git commit -m "feat: transact durable ingest checkpoint progress"
```

---

### Task 5: Integrate checkpoint commits, lookahead, and recovery into ingestion

**Files:**
- Modify: `app/core/ingest_runs.py`
- Modify: `scripts/ingest_from_kbo.py`
- Modify: `tests/test_ingest_results.py`
- Create: `tests/test_ingest_checkpoint_integration.py`

**Interfaces:**
- Consumes: Task 3 order/query functions and Task 4 session.
- Produces: `iter_checkpoint_rows`, checkpoint-aware `flush_chunks`, `ingest_table(..., checkpoint_run_id=None, checkpoint_scope_key=None)`, and `ingest(..., checkpoint_scope_key=None)`.

- [ ] **Step 1: Write failing lookahead and unsafe-option tests**

Create integration tests using in-memory recording source and destination
connections:

```python
def test_lookahead_rejects_duplicate_before_boundary_is_yielded():
    rows = [
        {"id": 1, "content": "first"},
        {"id": 1, "content": "duplicate"},
    ]
    order = CheckpointOrder("source", (CheckpointOrderField("id", "integer"),))

    iterator = iter_checkpoint_rows(rows, order=order, previous=None)
    with pytest.raises(IngestCheckpointOrderError):
        next(iterator)


@pytest.mark.parametrize("row_stale_cleanup", ["dry-run", "apply"])
def test_checkpointed_ingest_rejects_stale_cleanup(row_stale_cleanup):
    with pytest.raises(IngestCheckpointError) as raised:
        ingest(
            tables=["game"],
            lease_run_id=RUN_ID,
            lease_owner="worker-1",
            checkpoint_scope_key=SCOPE_KEY,
            row_stale_cleanup=row_stale_cleanup,
            **OPTIONS,
        )
    assert raised.value.code == "INGEST_CHECKPOINT_STALE_CLEANUP_UNSUPPORTED"
```

Add a leased `limit=1` case that raises
`INGEST_CHECKPOINT_CURSOR_UNAVAILABLE` before opening connections, and a
lease-without-scope case that raises `ValueError`.

- [ ] **Step 2: Run the new integration tests and verify RED**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_checkpoint_integration.py -q
```

Expected: collection or assertion failures because the iterator, scope argument,
and checkpoint guards do not exist.

- [ ] **Step 3: Implement one-row lookahead and manual cursor-field mapping**

Implement a generator with one pending row:

```python
def iter_checkpoint_rows(rows, *, order, previous):
    pending = None
    last_cursor = previous
    for raw_row in rows:
        row = dict(raw_row)
        try:
            cursor = cursor_from_row(order, row)
        except IngestCheckpointMissingFieldError as exc:
            _raise_manual_source_schema(order.source_table, exc.missing_fields)
        if pending is not None:
            ensure_cursor_advances(order, pending[1], cursor)
            yield pending
        elif last_cursor is not None:
            ensure_cursor_advances(order, last_cursor, cursor)
        pending = (row, cursor)
    if pending is not None:
        yield pending
```

Feed this iterator from the existing `fetchmany` loop without materializing the
whole source table. Keep exactly one raw row pending between fetched batches.

- [ ] **Step 4: Write failing atomic batch and completed-resume tests**

Define `_EventConnection`, `_EventCursor`, and `_EventCheckpointSession` test
doubles whose only side effects append `rag_chunks.executemany`,
`checkpoint.upsert`, `checkpoint.complete`, and `destination.commit` to one
shared `events: list[str]`. Define `_run_fake_checkpoint_ingest` to install those
doubles with `monkeypatch`, call `ingest_table`, and return its result plus the
shared event list. Add cases that assert:

```python
def test_chunk_write_and_checkpoint_precede_one_commit():
    result, events = _run_fake_checkpoint_ingest(rows=[{"id": 1}, {"id": 2}])
    assert events.index("rag_chunks.executemany") < events.index("checkpoint.upsert")
    assert events.index("checkpoint.upsert") < events.index("destination.commit")
    assert events.count("destination.commit") == 2  # data progress + completion
    assert result.source_rows == 2


def test_zero_output_rows_still_advance_checkpoint():
    result, events = _run_fake_checkpoint_ingest(
        rows=[{"id": 1}], render_payloads=[]
    )
    assert "rag_chunks.executemany" not in events
    assert "checkpoint.upsert" in events
    assert result.source_rows == 1
    assert result.written_chunks == 0


def test_completed_checkpoint_skips_source_select_and_returns_zero_attempt_delta():
    result, events = _run_fake_checkpoint_ingest(completed_checkpoint=True)
    assert "source.execute" not in events
    assert result.source_rows == 10
    assert result.attempt_source_rows == 0
    assert result.attempt_written_chunks == 0
```

Add two crash tests: failure before destination commit leaves the scripted durable
checkpoint unchanged, while a restart after a scripted committed cursor passes
that cursor into `build_select_query` and returns cumulative prior plus new
counts. Add a last-data-commit crash case whose resumed empty suffix writes only
the completion marker.

- [ ] **Step 5: Make `flush_chunks` commit progress atomically**

Add optional arguments without changing non-checkpoint callers:

```python
def flush_chunks(
    cur,
    settings,
    buffer,
    *,
    max_concurrency,
    commit_interval,
    stats,
    skip_embedding,
    lease_guard=None,
    checkpoint_session=None,
    checkpoint_cursor=None,
    checkpoint_source_rows=0,
    checkpoint_max_updated_at=None,
) -> int:
```

Capture reused/embedded counters before processing. Do not return early for an
empty buffer when `checkpoint_source_rows > 0`. After optional chunk writes and
before `cur.connection.commit()`:

```python
checkpoint_session.advance(
    cur,
    next_cursor=checkpoint_cursor,
    source_rows_delta=checkpoint_source_rows,
    written_chunks_delta=flushed,
    reused_embeddings_delta=int(stats.get("embedding_reused", 0)) - reused_before,
    embedded_chunks_delta=int(stats.get("reembedded_count", 0)) - embedded_before,
    max_updated_at=checkpoint_max_updated_at,
)
```

Call `lease_guard(True)` for checkpoint-only progress as well. If any operation
raises, do not call `commit`; closing or rolling back the connection preserves the
previous durable checkpoint and chunks.

- [ ] **Step 6: Make `ingest_table` load, resume, flush, and complete sessions**

For leased non-static tables, `ingest` passes `lease_run_id` as
`checkpoint_run_id` and passes `checkpoint_scope_key` into `ingest_table`. Then:

1. Resolve the checkpoint order.
2. Load a session from the destination cursor using run ID, table, scope, order.
3. Return a durable result immediately when `session.completed`.
4. Build the canonical query with `session.current.cursor` when incomplete.
5. Accumulate source-row count, last safe cursor, and batch max update time.
6. Flush when chunk count reaches `embed_batch_size` or pending source rows reach
   `read_batch_size`, so filtered-only runs also make bounded progress.
7. Flush the final pending batch.
8. Fence, call `session.complete(write_cur)`, and commit once.

Build the result through one helper:

```python
def _checkpoint_result(session, *, attempt_source_rows, attempt_written_chunks):
    checkpoint = session.current
    return IngestTableResult(
        source_table=checkpoint.source_table,
        written_chunks=checkpoint.written_chunks,
        source_rows=checkpoint.source_rows,
        reused_embeddings=checkpoint.reused_embeddings,
        embedded_chunks=checkpoint.embedded_chunks,
        max_updated_at=checkpoint.max_updated_at,
        checkpoint_resumed=session.resumed,
        checkpoint_committed_batches=checkpoint.committed_batches,
        checkpoint_completed=checkpoint.completed,
        attempt_source_rows=attempt_source_rows,
        attempt_written_chunks=attempt_written_chunks,
    )
```

Static and non-leased paths continue returning the existing result defaults.
Before constructing checkpoint-aware results, append these backward-compatible
fields to `IngestTableResult` in `app/core/ingest_runs.py`:

```python
checkpoint_resumed: bool = False
checkpoint_committed_batches: int = 0
checkpoint_completed: bool = False
attempt_source_rows: int | None = None
attempt_written_chunks: int | None = None
```

- [ ] **Step 7: Run focused integration and existing ingestion tests**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_ingest_checkpoint_integration.py \
  tests/test_ingest_results.py \
  tests/test_ingest_query.py \
  tests/test_ingest_parallel_engine.py \
  tests/test_game_flow_summary_ingest.py -q
```

Expected: all selected tests pass with no external calls.

- [ ] **Step 8: Commit the ingestion integration task**

```bash
git add app/core/ingest_runs.py scripts/ingest_from_kbo.py tests/test_ingest_results.py tests/test_ingest_checkpoint_integration.py
git commit -m "feat: resume ingest batches from durable keysets"
```

---

### Task 6: Propagate scope, summaries, attempt metrics, and checkpoint events

**Files:**
- Modify: `app/core/ingest_run_store.py`
- Modify: `app/core/ingest_sources.py`
- Modify: `app/core/ingest_worker.py`
- Modify: `app/observability/metrics.py`
- Modify: `scripts/ingest_from_kbo.py`
- Modify: `tests/test_ingest_worker.py`
- Modify: `tests/test_ingest_run_store.py`
- Modify: `tests/test_observability_metrics.py`

**Interfaces:**
- Consumes: checkpoint-aware ingestion results from Task 5.
- Produces: cumulative terminal summary fields, attempt-only existing metric increments, and `AI_INGEST_CHECKPOINT_EVENTS_TOTAL`.

- [ ] **Step 1: Write failing result, worker, summary, and label tests**

Use the checkpoint-aware `IngestTableResult` produced by Task 5:

```python
result = IngestTableResult(
    "game", 10, 20, 3, 7, NOW,
    checkpoint_resumed=True,
    checkpoint_committed_batches=4,
    checkpoint_completed=True,
    attempt_source_rows=2,
    attempt_written_chunks=1,
)
assert result.attempt_source_rows == 2
```

In the worker execution test assert:

```python
assert calls[0]["checkpoint_scope_key"] == build_watermark_scope_key(REQUEST)
```

Add a metric test where cumulative values are 20/10 but attempt values are 2/1,
then assert the existing source-row and written-chunk counters increase by 2 and
1. Add a completed-skip case with zero attempt deltas and assert no increase.

Add run-store payload assertions:

```python
assert payload["checkpoint"] == {
    "resumed": True,
    "committed_batches": 4,
    "completed": True,
}
assert "attempt_source_rows" not in payload
```

Add observability assertions:

```python
assert AI_INGEST_CHECKPOINT_EVENTS_TOTAL._labelnames == (
    "source_table", "result"
)
assert normalize_ingest_source_table("game") == "game"
assert normalize_ingest_source_table("untrusted") == "other"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_ingest_worker.py \
  tests/test_ingest_run_store.py \
  tests/test_observability_metrics.py -q
```

Expected: failures because result fields, scope kwarg, metric normalizer, checkpoint
counter, and checkpoint summary are absent.

- [ ] **Step 3: Add cumulative checkpoint fields to terminal summaries**

In `_table_result_payload`, retain existing cumulative fields and add only:

```python
"checkpoint": {
    "resumed": result.checkpoint_resumed,
    "committed_batches": result.checkpoint_committed_batches,
    "completed": result.checkpoint_completed,
}
```

- [ ] **Step 4: Add bounded metrics and attempt-delta accounting**

Define and export:

```python
AI_INGEST_CHECKPOINT_EVENTS_TOTAL = Counter(
    "ai_ingest_checkpoint_events_total",
    "Durable ingestion checkpoint lifecycle events.",
    ["source_table", "result"],
)
```

In `ingest_sources.py` add:

```python
def normalize_ingest_source_table(value: object) -> str:
    normalized = str(value or "").strip()
    return normalized if normalized in TRUSTED_INGEST_SOURCE_SET else "other"
```

Use it in the worker and script. Worker table metrics use:

```python
written = (
    table_result.written_chunks
    if table_result.attempt_written_chunks is None
    else table_result.attempt_written_chunks
)
rows = (
    table_result.source_rows
    if table_result.attempt_source_rows is None
    else table_result.attempt_source_rows
)
```

The script increments checkpoint events exactly once for `created`, `advanced`,
`completed`, and `resumed`; compatibility errors increment `incompatible`, other
checkpoint rejections increment `rejected`. Never label by run ID, owner, cursor,
or error message.

- [ ] **Step 5: Propagate scope from worker to synchronous ingestion**

Add to the existing `asyncio.to_thread` call:

```python
checkpoint_scope_key=scope_key,
```

Require the scope only when both lease arguments are present. Static sources may
receive the scope but do not create a checkpoint.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the command from Step 2, then:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_ingest_checkpoint_integration.py \
  tests/test_ingest_results.py \
  tests/test_ingest_query.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit worker and observability integration**

```bash
git add app/core/ingest_run_store.py app/core/ingest_sources.py app/core/ingest_worker.py app/observability/metrics.py scripts/ingest_from_kbo.py tests/test_ingest_worker.py tests/test_ingest_run_store.py tests/test_observability_metrics.py
git commit -m "feat: expose ingest checkpoint recovery metrics"
```

---

### Task 7: Document operations and run the release verification gate

**Files:**
- Modify: `docs/data-sync-orchestration-runbook.md`
- Modify: `docs/superpowers/plans/2026-07-18-ingest-persistent-keyset-checkpoints.md`
- Test: all focused and full AI tests.

**Interfaces:**
- Consumes: all behavior from Tasks 1-6.
- Produces: operator migration, resume, alert, rollback, and residual-risk guidance plus fresh verification evidence.

- [ ] **Step 1: Add runbook assertions before editing the runbook**

Add or extend a documentation contract test in `tests/test_ai_schema_migrations.py`:

```python
def test_data_sync_runbook_documents_persistent_checkpoint_operations():
    text = (ROOT / "docs" / "data-sync-orchestration-runbook.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "004_ai_ingest_checkpoints.sql",
        "ai_ingest_checkpoints",
        "INGEST_CHECKPOINT_INCOMPATIBLE",
        "INGEST_CHECKPOINT_CURSOR_UNAVAILABLE",
        "ai_ingest_checkpoint_events_total",
        "row_stale_cleanup",
        "rollback",
    ):
        assert required in text
```

- [ ] **Step 2: Run the runbook contract test and verify RED**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ai_schema_migrations.py::test_data_sync_runbook_documents_persistent_checkpoint_operations -q
```

Expected: failure because the runbook still states that persistent batch
checkpointing is absent.

- [ ] **Step 3: Update the runbook with exact operator behavior**

Document:

- startup applies 003 then 004;
- one row per `(run_id, source_table)` is retained after terminal completion;
- resume uses a typed ascending keyset and never an offset;
- chunks and cursor commit together under the lease fence;
- static documents restart atomically;
- success-only watermarks remain unchanged;
- the five checkpoint failure contracts and `MANUAL_BASEBALL_DATA_REQUIRED`;
- `row_stale_cleanup` must be `off` for checkpointed runs;
- `ai_ingest_checkpoint_events_total{source_table,result}` meanings;
- rollback keeps 004 and checkpoint rows while older code ignores them;
- no automatic retention job exists;
- no external baseball source is used;
- lack of a disposable live PostgreSQL smoke test is a residual risk unless the
  separately approved local-only smoke is run.

- [ ] **Step 4: Run focused checkpoint regression tests**

Run:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest \
  tests/test_ai_schema_migrations.py \
  tests/test_db_schema_contract.py \
  tests/test_schema_startup_mode.py \
  tests/test_ingest_checkpoints.py \
  tests/test_ingest_checkpoint_integration.py \
  tests/test_ingest_query.py \
  tests/test_ingest_results.py \
  tests/test_ingest_worker.py \
  tests/test_ingest_run_store.py \
  tests/test_observability_metrics.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Run static, policy, contract, and full-suite verification**

Run each command and retain its exit code and summary:

```bash
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m compileall -q app scripts tests
python3 /Users/mac/project/KBO_platform/scripts/validate_baseball_data_policy.py
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python scripts/export_openapi_contract.py --check
/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/ -q
git diff --check
```

Expected: exit code 0 for every command, zero policy violations, current OpenAPI
artifacts, and zero pytest failures. Existing documented dependency deprecation
warnings and environment-dependent operator-data skips may remain.

- [ ] **Step 6: Record actual verification evidence in this plan**

Append a `## Verification Evidence` section containing the exact date, command,
exit code, pass/skip/fail counts, and any residual risk. Do not write success
claims before the commands in Step 5 finish.

- [ ] **Step 7: Commit the runbook and verification record**

```bash
git add docs/data-sync-orchestration-runbook.md docs/superpowers/plans/2026-07-18-ingest-persistent-keyset-checkpoints.md tests/test_ai_schema_migrations.py
git commit -m "docs: operate persistent ingest checkpoints"
```

---

## Final Review Checklist

- [ ] Compare every approved design acceptance criterion to Tasks 1-7.
- [ ] Confirm no external baseball domain, HTTP client, crawler, scraper, browser automation, or search repair was added.
- [ ] Confirm static profiles never create checkpoint rows.
- [ ] Confirm all custom database profiles have explicit unique cursor orders.
- [ ] Confirm generic tables require real PostgreSQL primary keys and supported types.
- [ ] Confirm duplicate cursor lookahead fails before an unsafe boundary commit.
- [ ] Confirm checkpoint-only zero-output progress is fenced and committed.
- [ ] Confirm completed-table recovery performs no source data SELECT.
- [ ] Confirm cumulative summaries and attempt-only metrics are distinct.
- [ ] Confirm terminal watermarks still advance only in `finish_success`.
- [ ] Confirm rollback preserves migration 004 and checkpoint audit rows.
- [ ] Confirm no uncommitted scratch reports or generated artifacts remain.
- [ ] Do not push, open a PR, merge to `feature`, delete the worktree, or delete the branch without the user's explicit finishing instruction.

## Verification Evidence

Date: 2026-07-18

- RED documentation contract: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ai_schema_migrations.py::test_data_sync_runbook_documents_persistent_checkpoint_operations -q` exited 1 as expected before the runbook update: 0 passed, 1 failed, 0 skipped, 0 warnings.
- GREEN documentation contract: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ai_schema_migrations.py::test_data_sync_runbook_documents_persistent_checkpoint_operations -q` exited 0: 1 passed, 0 failed, 0 skipped, 0 warnings.
- Focused checkpoint regression: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ai_schema_migrations.py tests/test_db_schema_contract.py tests/test_schema_startup_mode.py tests/test_ingest_checkpoints.py tests/test_ingest_checkpoint_integration.py tests/test_ingest_query.py tests/test_ingest_results.py tests/test_ingest_worker.py tests/test_ingest_run_store.py tests/test_observability_metrics.py -q` exited 0: 214 passed, 0 failed, 0 skipped, 17 warnings (`datetime.utcnow()` deprecation in checkpoint integration).
- Static compile: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m compileall -q app scripts tests` exited 0 with no output (non-test gate; pass/skip/fail counts not applicable).
- Baseball data policy: `python3 /Users/mac/project/KBO_platform/scripts/validate_baseball_data_policy.py` exited 0: `External baseball data policy OK` (zero violations; non-test gate).
- OpenAPI contract: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python scripts/export_openapi_contract.py --check` exited 0: artifacts current, 0 failures; one existing `google.generativeai` FutureWarning.
- Full AI suite: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/ -q` exited 0: 1878 passed, 5 skipped, 0 failed, 25 warnings. Skips require unavailable local operator-data migration, validation report, or handoff reports; warnings include the `google.generativeai` FutureWarning, checkpoint-path `datetime.utcnow()` deprecation, deprecated HTTP 422 alias, and Pydantic model field deprecations.
- Whitespace check: `git diff --check` exited 0 with no output (non-test gate; pass/skip/fail counts not applicable).

Residual risk: no separately approved local-only disposable live PostgreSQL smoke was run. Therefore real PostgreSQL DDL/transaction behavior remains unverified outside the fake-connection coverage; no shared or production database was contacted.

## Post-Review Verification Evidence

Date: 2026-07-18

- Managed migration contract RED: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ai_schema_migrations.py::test_managed_migration_script_applies_ingest_checkpoint_migration_in_order -q` exited 1: 0 passed, 1 failed, 0 skipped, 0 warnings. The script did not yet include migration 004.
- Managed migration contract GREEN: the same command exited 0: 1 passed, 0 failed, 0 skipped, 0 warnings. Relevant schema/startup regression `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ai_schema_migrations.py tests/test_db_schema_contract.py tests/test_schema_startup_mode.py -q` exited 0: 20 passed, 0 failed, 0 skipped, 0 warnings.
- Strengthened runbook contract RED: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ai_schema_migrations.py::test_data_sync_runbook_documents_persistent_checkpoint_operations -q` exited 1: 0 passed, 1 failed, 0 skipped, 0 warnings. It lacked the exact reviewed operational contract statements.
- Strengthened runbook contract GREEN: the same command exited 0: 1 passed, 0 failed, 0 skipped, 0 warnings.
- Focused checkpoint regression: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ai_schema_migrations.py tests/test_db_schema_contract.py tests/test_schema_startup_mode.py tests/test_ingest_checkpoints.py tests/test_ingest_checkpoint_integration.py tests/test_ingest_query.py tests/test_ingest_results.py tests/test_ingest_worker.py tests/test_ingest_run_store.py tests/test_observability_metrics.py -q` exited 0: 215 passed, 0 failed, 0 skipped, 17 `datetime.utcnow()` deprecation warnings.
- Static compile: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m compileall -q app scripts tests` exited 0 with no output (non-test gate; pass/skip/fail counts not applicable).
- Baseball data policy: `python3 /Users/mac/project/KBO_platform/scripts/validate_baseball_data_policy.py` exited 0: `External baseball data policy OK` (zero violations; non-test gate).
- OpenAPI contract: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python scripts/export_openapi_contract.py --check` exited 0: artifacts current, 0 failures; one existing `google.generativeai` FutureWarning.
- Full AI suite: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/ -q` exited 0: 1879 passed, 5 skipped, 0 failed, 25 warnings. The skips require unavailable local operator-data migration, validation report, or handoff reports.
- Whitespace check: `git diff --check` exited 0 with no output (non-test gate; pass/skip/fail counts not applicable).

Final residual risk: no separately approved local-only disposable live PostgreSQL smoke was run. Real PostgreSQL DDL/transaction behavior therefore remains unverified outside fake-connection coverage; no shared or production database was contacted.

## Final-Review Fix Verification Evidence

Date: 2026-07-18

- Focused RED: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_checkpoints.py tests/test_ingest_query.py tests/test_ingest_checkpoint_integration.py tests/test_ingest_results.py -q -k 'team_profiles_checkpoint_uses_unique_id or custom_checkpoint_registry_is_exact or resolve_checkpoint_order_distinguishes or datetime_naive or invalid_checkpoint_scope or invalid_scope_before or direct_checkpoint_ingest_strips_scope or leased_database_ingest_passes_checkpoint_identity'` exited 1 before the production fix: 12 failed, 6 passed, 139 deselected, 1 existing `datetime.utcnow()` deprecation warning. Failures exercised the non-unique `team_profiles.team_id` cursor, collapsed PostgreSQL timestamp subtypes, missing naive-datetime codec/query support, and unnormalized or overlong checkpoint scopes.
- Focused GREEN: the same focused command exited 0 after the production fix: 18 passed, 0 failed, 139 deselected, 1 existing `datetime.utcnow()` deprecation warning.
- Checkpoint/query/worker/schema regression: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_checkpoints.py tests/test_ingest_checkpoint_integration.py tests/test_ingest_query.py tests/test_ingest_results.py tests/test_ingest_worker.py tests/test_ai_schema_migrations.py tests/test_ai_schema_rehearsal_script.py tests/test_db_schema_contract.py tests/test_rag_storage_schema.py tests/test_schema_startup_mode.py tests/test_validate_ai_runtime_schema.py -q` exited 0: 222 passed, 0 failed, 0 skipped, 18 existing `datetime.utcnow()` deprecation warnings.
- Code and tests were committed as `681b337` (`fix: preserve ingest checkpoint boundaries`). The `team_profiles` registry now orders by integer primary key `id`; `timestamp without time zone` uses the distinct `datetime_naive` signature/payload type and binds a naive Python `datetime`; old `datetime` signatures fail closed as incompatible; checkpoint scopes are stripped and validated against the 64-character schema limit before runtime or connection setup.
- Static compile: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m compileall -q app scripts tests` exited 0 with no output (non-test gate; pass/skip/fail counts not applicable).
- Baseball data policy: `python3 /Users/mac/project/KBO_platform/scripts/validate_baseball_data_policy.py` exited 0: `External baseball data policy OK` (zero violations; non-test gate).
- OpenAPI contract: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python scripts/export_openapi_contract.py --check` exited 0: artifacts current, 0 failures; one existing `google.generativeai` FutureWarning.
- Full AI suite: `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/ -q` exited 0: 1895 passed, 5 skipped, 0 failed, 26 warnings in 329.24 seconds. Skips require unavailable local operator-data migration, validation report, or handoff reports; warnings are the existing `google.generativeai`, checkpoint-path `datetime.utcnow()`, deprecated HTTP 422 alias, and Pydantic model-field deprecations.
- Whitespace check: `git diff --check` exited 0 with no output before this addendum (non-test gate; pass/skip/fail counts not applicable).

Residual risk: no live PostgreSQL test was run, and no shared or production database or network source was contacted. Unit coverage proves that `datetime_naive` JSON round trips without an offset, rejects aware or offset-bearing values, remains signature-distinct from `datetime`, and reaches the query parameter tuple as a naive Python `datetime`; it does not exercise psycopg binding against PostgreSQL sessions configured with different `TimeZone` settings. Text keyset ordering likewise remains dependent on the source PostgreSQL collation; query predicates and ascending order use the same database expressions, but no live locale/collation matrix was executed.

## Frozen Source Window Final-Fix Verification Evidence

Date: 2026-07-18

- Focused schema/repository/query/precision RED exited 1: 18 failed, 7 passed, 124 deselected. Focused fake recovery RED exited 1: 6 failed, 29 deselected. Same-filter watermark RED exited 1: 2 failed, 36 deselected. Each failure matched the missing fixed-cutoff or precision-qualified timestamp behavior.
- Focused GREEN: schema/repository/query/precision selection exited 0 with 27 passed and 123 deselected; fake recovery selection exited 0 with 6 passed and 29 deselected. Documentation contract moved from 2 failed to 2 passed after the design and runbook update.
- Expanded checkpoint/schema/query/worker regression `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/test_ingest_checkpoints.py tests/test_ingest_checkpoint_integration.py tests/test_ingest_query.py tests/test_ingest_results.py tests/test_ingest_worker.py tests/test_ingest_run_store.py tests/test_observability_metrics.py tests/test_ai_schema_migrations.py tests/test_ai_schema_rehearsal_script.py tests/test_db_schema_contract.py tests/test_rag_storage_schema.py tests/test_schema_startup_mode.py tests/test_validate_ai_runtime_schema.py -q` exited 0: 272 passed, 0 failed, 25 existing `datetime.utcnow()` warnings in 21.12 seconds.
- Code, schema, tests, and behavior documentation were committed as `1a062b6` (`fix: freeze checkpoint source update windows`). Migration 004 now upgrades old and fresh checkpoint tables with `source_updated_before`; update-filtered runs sample and immutably persist one source-clock cutoff; recovery reuses it; progressed legacy rows without it fail closed; and custom/generic queries apply the matching upper bound before resume.
- Static compile `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m compileall -q app scripts tests` exited 0 with no output.
- Baseball data policy `python3 /Users/mac/project/KBO_platform/scripts/validate_baseball_data_policy.py` exited 0: `External baseball data policy OK`.
- OpenAPI contract `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python scripts/export_openapi_contract.py --check` exited 0: `AI OpenAPI artifacts are current`; one existing `google.generativeai` FutureWarning.
- Full AI suite `/Users/mac/project/KBO_platform/bega_AI/.venv/bin/python -m pytest tests/ -q` exited 0: 1921 passed, 5 skipped, 0 failed, 33 warnings in 358.47 seconds. Skips require unavailable local operator-data migration, validation, or handoff artifacts. Warnings are existing dependency/API deprecations.
- Whitespace gate `git diff --check HEAD^ HEAD` exited 0 with no output before this addendum.

Residual risk: no live PostgreSQL smoke or locale/collation matrix was run. Fake coverage cannot verify psycopg binding or migration execution against a real server. Text keyset comparisons remain source-collation dependent, and the lower watermark's inclusive comparison deliberately retains a safe duplicate window through the fixed cutoff. No shared/production database, network source, or external baseball-data source was contacted.
