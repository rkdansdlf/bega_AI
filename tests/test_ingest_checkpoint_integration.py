from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from app.core.ingest_checkpoints import (
    CheckpointCursor,
    CheckpointOrder,
    CheckpointOrderField,
    IngestCheckpointCursorUnavailableError,
    IngestCheckpointOrderError,
    IngestCheckpointStaleCleanupError,
)
from scripts import ingest_from_kbo as module


RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
SCOPE_KEY = "season:2026"
OPTIONS = {
    "source_db_url": "postgresql://internal-source",
    "limit": None,
    "embed_batch_size": 2,
    "read_batch_size": 2,
    "season_year": 2026,
    "use_legacy_renderer": False,
    "since": None,
    "skip_embedding": True,
    "max_concurrency": 1,
    "commit_interval": 500,
    "parallel_engine": "thread",
    "workers": 1,
}
ORDER = CheckpointOrder(
    "teams",
    (CheckpointOrderField("id", "integer"),),
)


@dataclass
class _CheckpointRecord:
    source_table: str = "teams"
    cursor: CheckpointCursor | None = None
    committed_batches: int = 0
    source_rows: int = 0
    written_chunks: int = 0
    reused_embeddings: int = 0
    embedded_chunks: int = 0
    max_updated_at: Any = None
    completed: bool = False


@dataclass
class _CheckpointState:
    durable: _CheckpointRecord | None = None
    pending: _CheckpointRecord | None = None
    durable_chunk_writes: int = 0
    pending_chunk_writes: int = 0


class _EventCheckpointSession:
    events: list[str]
    state: _CheckpointState
    fail_advance: bool

    def __init__(self) -> None:
        self.initial = deepcopy(self.state.durable)
        self.current = deepcopy(self.state.durable)

    @classmethod
    def start(cls, _cursor, **_kwargs):
        return cls()

    @property
    def resumed(self) -> bool:
        return self.initial is not None

    @property
    def completed(self) -> bool:
        return bool(self.current and self.current.completed)

    def advance(
        self,
        _cursor,
        *,
        next_cursor,
        source_rows_delta,
        written_chunks_delta,
        reused_embeddings_delta,
        embedded_chunks_delta,
        max_updated_at,
    ):
        self.events.append("checkpoint.upsert")
        if self.fail_advance:
            raise RuntimeError("scripted checkpoint failure")
        current = self.current or _CheckpointRecord()
        durable_max = current.max_updated_at
        if durable_max is None or (
            max_updated_at is not None and max_updated_at > durable_max
        ):
            durable_max = max_updated_at
        self.current = replace(
            current,
            cursor=next_cursor,
            committed_batches=current.committed_batches + 1,
            source_rows=current.source_rows + source_rows_delta,
            written_chunks=current.written_chunks + written_chunks_delta,
            reused_embeddings=current.reused_embeddings + reused_embeddings_delta,
            embedded_chunks=current.embedded_chunks + embedded_chunks_delta,
            max_updated_at=durable_max,
            completed=False,
        )
        self.state.pending = deepcopy(self.current)
        return self.current

    def complete(self, _cursor):
        self.events.append("checkpoint.complete")
        current = self.current or _CheckpointRecord()
        self.current = replace(current, completed=True)
        self.state.pending = deepcopy(self.current)
        return self.current


class _EventCursor:
    def __init__(
        self,
        connection,
        *,
        rows: list[dict[str, Any]] | None = None,
        source: bool = False,
    ) -> None:
        self.connection = connection
        self.rows = list(rows or [])
        self.source = source
        self.description = [SimpleNamespace(name="id")] if source else []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc, traceback
        if exc_type is not None and not self.source:
            self.connection.state.pending = None
            self.connection.state.pending_chunk_writes = 0
        return False

    def execute(self, _query, _params=None):
        if self.source:
            self.connection.events.append("source.execute")

    def fetchmany(self, size):
        batch = self.rows[:size]
        del self.rows[:size]
        return batch

    def executemany(self, _query, data):
        self.connection.events.append("rag_chunks.executemany")
        self.connection.state.pending_chunk_writes += len(data)


class _EventConnection:
    def __init__(
        self,
        events,
        state,
        *,
        rows=None,
        source=False,
        fail_commits=(),
    ) -> None:
        self.events = events
        self.state = state
        self.rows = list(rows or [])
        self.source = source
        self.fail_commits = set(fail_commits)
        self.commit_count = 0

    def cursor(self, *_args, **_kwargs):
        return _EventCursor(
            self,
            rows=self.rows if self.source else None,
            source=self.source,
        )

    def commit(self):
        self.commit_count += 1
        self.events.append("destination.commit")
        if self.commit_count in self.fail_commits:
            raise RuntimeError("scripted destination commit failure")
        if self.state.pending is not None:
            self.state.durable = deepcopy(self.state.pending)
            self.state.pending = None
        self.state.durable_chunk_writes += self.state.pending_chunk_writes
        self.state.pending_chunk_writes = 0


def _payload_for_task(task):
    table_name, row, source_row_id, _legacy, _today = task
    return {
        "table": table_name,
        "source_row_id": source_row_id,
        "title": f"row {row['id']}",
        "content": f"content {row['id']}",
        "season_year": None,
        "season_id": None,
        "league_type_code": None,
        "team_id": None,
        "player_id": None,
        "meta": dict(row),
    }


def _run_fake_checkpoint_ingest(
    monkeypatch,
    *,
    rows=(),
    render_payloads=None,
    completed_checkpoint=False,
    state=None,
    events=None,
    fail_advance=False,
    fail_commits=(),
    captured_resumes=None,
    lease_calls=None,
    read_batch_size=2,
):
    events = [] if events is None else events
    if state is None:
        durable = None
        if completed_checkpoint:
            durable = _CheckpointRecord(
                cursor=CheckpointCursor((10,)),
                committed_batches=4,
                source_rows=10,
                written_chunks=7,
                reused_embeddings=2,
                embedded_chunks=5,
                completed=True,
            )
        state = _CheckpointState(durable=durable)
    captured_resumes = [] if captured_resumes is None else captured_resumes
    lease_calls = [] if lease_calls is None else lease_calls
    source = _EventConnection(events, state, rows=list(rows), source=True)
    destination = _EventConnection(
        events,
        state,
        fail_commits=fail_commits,
    )

    class _ConfiguredSession(_EventCheckpointSession):
        pass

    _ConfiguredSession.events = events
    _ConfiguredSession.state = state
    _ConfiguredSession.fail_advance = fail_advance

    def _build_select_query(*_args, **kwargs):
        captured_resumes.append(kwargs.get("resume_cursor"))
        return "SELECT id FROM teams", ()

    def _prepare(tasks, *, parallel_engine, workers):
        del workers
        if render_payloads is not None:
            return [[dict(payload) for payload in render_payloads] for _ in tasks], parallel_engine
        return [[_payload_for_task(task)] for task in tasks], parallel_engine

    monkeypatch.setattr(module, "IngestCheckpointSession", _ConfiguredSession)
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(module, "resolve_primary_key_columns", lambda *_args: ["id"])
    monkeypatch.setattr(module, "resolve_checkpoint_order", lambda *_args: ORDER)
    monkeypatch.setattr(module, "build_select_query", _build_select_query)
    monkeypatch.setattr(
        module,
        "build_source_row_id",
        lambda row, *_args: f"id={row['id']}",
    )
    monkeypatch.setattr(module, "_prepare_rows_for_engine", _prepare)
    monkeypatch.setattr(module, "resolve_embedding_model", lambda _settings: "test")
    monkeypatch.setattr(module, "resolve_embedding_version", lambda _settings: "1")
    monkeypatch.setattr(module, "scan_sensitive_content", lambda _value: [])
    monkeypatch.setattr(module, "is_search_worthy_content", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        module,
        "build_chunk_storage_fields",
        lambda **_kwargs: {"content_hash": "hash"},
    )
    monkeypatch.setattr(module, "build_upsert_tuple", lambda **_kwargs: ("row",))
    monkeypatch.setattr(module, "soft_deactivate_missing_parts", lambda *_args, **_kwargs: 0)

    result = module.ingest_table(
        source,
        destination,
        "teams",
        limit=None,
        embed_batch_size=2,
        read_batch_size=read_batch_size,
        season_year=2026,
        since=None,
        use_legacy_renderer=False,
        skip_embedding=True,
        max_concurrency=1,
        commit_interval=500,
        parallel_engine="thread",
        workers=1,
        row_stale_cleanup="off",
        stats={},
        lease_guard=lambda write=False: lease_calls.append(write),
        checkpoint_run_id=RUN_ID,
        checkpoint_scope_key=SCOPE_KEY,
    )
    return result, events


def test_lookahead_rejects_duplicate_before_boundary_is_yielded():
    rows = [
        {"id": 1, "content": "first"},
        {"id": 1, "content": "duplicate"},
    ]
    order = CheckpointOrder(
        "source",
        (CheckpointOrderField("id", "integer"),),
    )

    iterator = module.iter_checkpoint_rows(rows, order=order, previous=None)

    with pytest.raises(IngestCheckpointOrderError):
        next(iterator)


def test_lookahead_maps_missing_cursor_field_to_manual_contract():
    order = CheckpointOrder(
        "game",
        (CheckpointOrderField("id", "integer"),),
    )

    iterator = module.iter_checkpoint_rows(
        [{"content": "missing id"}],
        order=order,
        previous=None,
    )

    with pytest.raises(module.ManualBaseballDataRequiredError) as raised:
        next(iterator)

    assert raised.value.contract["code"] == "MANUAL_BASEBALL_DATA_REQUIRED"
    assert raised.value.contract["entity"] == "game"
    assert raised.value.contract["missing_fields"] == ["id"]
    assert raised.value.contract["import_source"] == "operator_manual_data"


def test_lookahead_rejects_non_advancing_resume_row_before_yielding():
    order = CheckpointOrder(
        "source",
        (CheckpointOrderField("id", "integer"),),
    )
    iterator = module.iter_checkpoint_rows(
        [{"id": 1}],
        order=order,
        previous=CheckpointCursor((1,)),
    )

    with pytest.raises(IngestCheckpointOrderError):
        next(iterator)


@pytest.mark.parametrize("row_stale_cleanup", ["dry-run", "apply"])
def test_checkpointed_ingest_rejects_stale_cleanup(row_stale_cleanup):
    with pytest.raises(IngestCheckpointStaleCleanupError) as raised:
        module.ingest(
            tables=["game"],
            lease_run_id=RUN_ID,
            lease_owner="worker-1",
            checkpoint_scope_key=SCOPE_KEY,
            row_stale_cleanup=row_stale_cleanup,
            **OPTIONS,
        )

    assert raised.value.code == "INGEST_CHECKPOINT_STALE_CLEANUP_UNSUPPORTED"


def test_checkpointed_ingest_rejects_limit_before_opening_connections(monkeypatch):
    monkeypatch.setattr(module, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        module.psycopg,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("connections must not be opened")
        ),
    )

    with pytest.raises(IngestCheckpointCursorUnavailableError) as raised:
        module.ingest(
            tables=["game"],
            lease_run_id=RUN_ID,
            lease_owner="worker-1",
            checkpoint_scope_key=SCOPE_KEY,
            row_stale_cleanup="off",
            **{**OPTIONS, "limit": 1},
        )

    assert raised.value.code == "INGEST_CHECKPOINT_CURSOR_UNAVAILABLE"


def test_leased_ingest_requires_checkpoint_scope_before_opening_connections(
    monkeypatch,
):
    monkeypatch.setattr(module, "_require_psycopg", lambda: None)
    monkeypatch.setattr(
        module.psycopg,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("connections must not be opened")
        ),
    )

    with pytest.raises(ValueError, match="checkpoint_scope_key"):
        module.ingest(
            tables=["game"],
            lease_run_id=RUN_ID,
            lease_owner="worker-1",
            row_stale_cleanup="off",
            **OPTIONS,
        )


def test_leased_ingest_rejects_blank_owner_before_runtime_or_connections(
    monkeypatch,
):
    def _unexpected(*_args, **_kwargs):
        raise AssertionError("runtime and connections must not be opened")

    monkeypatch.setattr(module, "_require_psycopg", _unexpected)
    monkeypatch.setattr(module, "get_settings", _unexpected)
    monkeypatch.setattr(module.psycopg, "connect", _unexpected)

    with pytest.raises(ValueError, match="lease_run_id and lease_owner"):
        module.ingest(
            tables=["game"],
            lease_run_id=RUN_ID,
            lease_owner="   ",
            checkpoint_scope_key=SCOPE_KEY,
            row_stale_cleanup="off",
            **OPTIONS,
        )


def test_chunk_write_and_checkpoint_precede_one_commit(monkeypatch):
    result, events = _run_fake_checkpoint_ingest(
        monkeypatch,
        rows=[{"id": 1}, {"id": 2}],
    )

    assert events.index("rag_chunks.executemany") < events.index("checkpoint.upsert")
    assert events.index("checkpoint.upsert") < events.index("destination.commit")
    assert events.count("destination.commit") == 2
    assert result.source_rows == 2


def test_zero_output_rows_still_advance_checkpoint(monkeypatch):
    lease_calls: list[bool] = []
    result, events = _run_fake_checkpoint_ingest(
        monkeypatch,
        rows=[{"id": 1}],
        render_payloads=[],
        lease_calls=lease_calls,
    )

    assert "rag_chunks.executemany" not in events
    assert "checkpoint.upsert" in events
    assert result.source_rows == 1
    assert result.written_chunks == 0
    assert lease_calls.count(True) == 2


def test_completed_checkpoint_skips_source_select_and_returns_zero_attempt_delta(
    monkeypatch,
):
    result, events = _run_fake_checkpoint_ingest(
        monkeypatch,
        completed_checkpoint=True,
    )

    assert "source.execute" not in events
    assert result.source_rows == 10
    assert result.attempt_source_rows == 0
    assert result.attempt_written_chunks == 0


def test_failure_before_destination_commit_preserves_durable_state(monkeypatch):
    initial = _CheckpointRecord(
        cursor=CheckpointCursor((0,)),
        committed_batches=2,
        source_rows=4,
        written_chunks=3,
    )
    state = _CheckpointState(durable=deepcopy(initial), durable_chunk_writes=3)
    events: list[str] = []

    with pytest.raises(RuntimeError, match="scripted checkpoint failure"):
        _run_fake_checkpoint_ingest(
            monkeypatch,
            rows=[{"id": 1}],
            state=state,
            events=events,
            fail_advance=True,
        )

    assert state.durable == initial
    assert state.durable_chunk_writes == 3
    assert "checkpoint.upsert" in events
    assert "destination.commit" not in events


def test_restart_resumes_after_committed_cursor_and_returns_cumulative_counts(
    monkeypatch,
):
    state = _CheckpointState(
        durable=_CheckpointRecord(
            cursor=CheckpointCursor((1,)),
            committed_batches=1,
            source_rows=1,
            written_chunks=1,
        ),
        durable_chunk_writes=1,
    )
    captured_resumes: list[CheckpointCursor | None] = []

    result, _events = _run_fake_checkpoint_ingest(
        monkeypatch,
        rows=[{"id": 2}],
        state=state,
        captured_resumes=captured_resumes,
    )

    assert captured_resumes == [CheckpointCursor((1,))]
    assert result.source_rows == 2
    assert result.written_chunks == 2
    assert result.attempt_source_rows == 1
    assert result.attempt_written_chunks == 1
    assert result.checkpoint_resumed is True
    assert result.checkpoint_committed_batches == 2
    assert result.checkpoint_completed is True


def test_last_data_commit_crash_resumes_with_completion_only(monkeypatch):
    state = _CheckpointState()
    first_events: list[str] = []

    with pytest.raises(RuntimeError, match="scripted destination commit failure"):
        _run_fake_checkpoint_ingest(
            monkeypatch,
            rows=[{"id": 1}],
            state=state,
            events=first_events,
            fail_commits={2},
        )

    assert state.durable == _CheckpointRecord(
        cursor=CheckpointCursor((1,)),
        committed_batches=1,
        source_rows=1,
        written_chunks=1,
    )
    assert first_events[-2:] == ["checkpoint.complete", "destination.commit"]

    second_events: list[str] = []
    result, second_events = _run_fake_checkpoint_ingest(
        monkeypatch,
        rows=[],
        state=state,
        events=second_events,
    )

    assert second_events == [
        "source.execute",
        "checkpoint.complete",
        "destination.commit",
    ]
    assert result.source_rows == 1
    assert result.attempt_source_rows == 0
    assert result.checkpoint_completed is True


def test_ingest_lookahead_spans_fetch_batches_before_checkpointing(monkeypatch):
    events: list[str] = []

    with pytest.raises(IngestCheckpointOrderError):
        _run_fake_checkpoint_ingest(
            monkeypatch,
            rows=[{"id": 1}, {"id": 1}],
            events=events,
            read_batch_size=1,
        )

    assert "rag_chunks.executemany" not in events
    assert "checkpoint.upsert" not in events
    assert "destination.commit" not in events
