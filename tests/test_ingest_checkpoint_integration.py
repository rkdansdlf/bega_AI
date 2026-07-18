from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from app.core.ingest_checkpoints import (
    CheckpointCursor,
    CheckpointOrder,
    CheckpointOrderField,
    IngestCheckpointCursorUnavailableError,
    IngestCheckpointIncompatibleError,
    IngestCheckpointOrderError,
    IngestCheckpointStaleCleanupError,
)
from scripts import ingest_from_kbo as module


RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
SCOPE_KEY = "season:2026"
CUTOFF = datetime(2026, 7, 18, 4, 0, tzinfo=UTC)
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
    source_updated_before: Any = None
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
    source_table: str = "teams"

    def __init__(self, *, requires_source_updated_before: bool) -> None:
        self.initial = deepcopy(self.state.durable)
        self.current = deepcopy(self.state.durable)
        self.requires_source_updated_before = requires_source_updated_before
        if (
            self.current is not None
            and self.current.source_updated_before is None
            and (
                self.current.cursor is not None
                or self.current.committed_batches > 0
                or self.current.source_rows > 0
            )
            and requires_source_updated_before
        ):
            raise IngestCheckpointIncompatibleError(
                "progressed checkpoint is missing source update cutoff"
            )

    @classmethod
    def start(cls, _cursor, **kwargs):
        return cls(
            requires_source_updated_before=kwargs.get(
                "requires_source_updated_before",
                False,
            )
        )

    @property
    def resumed(self) -> bool:
        return self.initial is not None

    @property
    def completed(self) -> bool:
        return bool(self.current and self.current.completed)

    @property
    def source_updated_before(self):
        if self.current is not None and self.current.source_updated_before is not None:
            return self.current.source_updated_before
        return getattr(self, "_source_updated_before", None)

    def bind_source_updated_before(self, value):
        current = self.source_updated_before
        if current is not None and current != value:
            raise IngestCheckpointIncompatibleError("source update cutoff changed")
        self._source_updated_before = value

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
        current = self.current or _CheckpointRecord(source_table=self.source_table)
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
            source_updated_before=self.source_updated_before,
            completed=False,
        )
        self.state.pending = deepcopy(self.current)
        return self.current

    def complete(self, _cursor):
        self.events.append("checkpoint.complete")
        current = self.current or _CheckpointRecord(source_table=self.source_table)
        self.current = replace(
            current,
            source_updated_before=self.source_updated_before,
            completed=True,
        )
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
        self._one = None

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
            if "clock_timestamp()" in str(_query):
                self.connection.events.append("source.clock")
                self._one = {
                    "source_updated_before": self.connection.source_clock_values.pop(0)
                }
                return
            self.connection.events.append("source.execute")
            resume = self.connection.selected_resume
            cutoff = self.connection.selected_cutoff
            since = self.connection.selected_since
            self.rows = [
                row
                for row in self.rows
                if (resume is None or row["id"] > resume.values[0])
                and (
                    cutoff is None
                    or row.get("updated_at") is None
                    or row["updated_at"] <= cutoff
                )
                and (
                    since is None
                    or (
                        row.get("updated_at") is not None
                        and row["updated_at"] >= since
                    )
                )
            ]

    def fetchone(self):
        value = self._one
        self._one = None
        return value

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
        source_clock_values=(),
    ) -> None:
        self.events = events
        self.state = state
        self.rows = list(rows or [])
        self.source = source
        self.fail_commits = set(fail_commits)
        self.commit_count = 0
        self.source_clock_values = list(source_clock_values)
        self.selected_resume = None
        self.selected_cutoff = None
        self.selected_since = None

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
    table_name="teams",
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
    observe_metrics=False,
    start_error=None,
    checkpoint_scope_key=SCOPE_KEY,
    captured_scopes=None,
    captured_cutoffs=None,
    source_clock_values=None,
    since=None,
):
    events = [] if events is None else events
    if state is None:
        durable = None
        if completed_checkpoint:
            durable = _CheckpointRecord(
                source_table=table_name,
                cursor=CheckpointCursor((10,)),
                committed_batches=4,
                source_rows=10,
                written_chunks=7,
                reused_embeddings=2,
                embedded_chunks=5,
                source_updated_before=CUTOFF,
                completed=True,
            )
        state = _CheckpointState(durable=durable)
    captured_resumes = [] if captured_resumes is None else captured_resumes
    lease_calls = [] if lease_calls is None else lease_calls
    captured_scopes = [] if captured_scopes is None else captured_scopes
    captured_cutoffs = [] if captured_cutoffs is None else captured_cutoffs
    source = _EventConnection(
        events,
        state,
        rows=list(rows),
        source=True,
        source_clock_values=(
            [CUTOFF] if source_clock_values is None else source_clock_values
        ),
    )
    destination = _EventConnection(
        events,
        state,
        fail_commits=fail_commits,
    )

    class _ConfiguredSession(_EventCheckpointSession):
        @classmethod
        def start(cls, cursor, **kwargs):
            if start_error is not None:
                raise start_error
            captured_scopes.append(kwargs["scope_key"])
            return super().start(cursor, **kwargs)

    _ConfiguredSession.events = events
    _ConfiguredSession.state = state
    _ConfiguredSession.fail_advance = fail_advance
    _ConfiguredSession.source_table = table_name

    def _build_select_query(*_args, **kwargs):
        captured_resumes.append(kwargs.get("resume_cursor"))
        cutoff = kwargs.get("source_updated_before")
        captured_cutoffs.append(cutoff)
        source.selected_resume = kwargs.get("resume_cursor")
        source.selected_cutoff = cutoff
        source.selected_since = kwargs.get("since")
        return "SELECT id FROM teams", ()

    def _prepare(tasks, *, parallel_engine, workers):
        del workers
        if render_payloads is not None:
            return [[dict(payload) for payload in render_payloads] for _ in tasks], parallel_engine
        return [[_payload_for_task(task)] for task in tasks], parallel_engine

    monkeypatch.setattr(module, "IngestCheckpointSession", _ConfiguredSession)
    if observe_metrics:
        monkeypatch.setattr(
            module,
            "_record_checkpoint_event",
            lambda source_table, result: events.append(
                f"metric.checkpoint:{source_table}:{result}"
            ),
        )
        monkeypatch.setattr(
            module,
            "_record_checkpoint_batch_metrics",
            lambda source_table, source_rows, written_chunks: events.append(
                f"metric.batch:{source_table}:{source_rows}:{written_chunks}"
            ),
        )
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(module, "resolve_primary_key_columns", lambda *_args: ["id"])
    monkeypatch.setattr(
        module,
        "validate_required_source_columns",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "resolve_checkpoint_order",
        lambda _connection, name, _profile: CheckpointOrder(
            name,
            (CheckpointOrderField("id", "integer"),),
        ),
    )
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
        table_name,
        limit=None,
        embed_batch_size=2,
        read_batch_size=read_batch_size,
        season_year=2026,
        since=since,
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
        checkpoint_scope_key=checkpoint_scope_key,
    )
    return result, events


def _read_metric_value(name: str, labels: dict[str, str]) -> float:
    prometheus_client = pytest.importorskip("prometheus_client")
    for metric in prometheus_client.REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == name and sample.labels == labels:
                return sample.value
    return 0.0


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


def test_checkpointed_ingest_records_typed_rejection_once_before_connections(
    monkeypatch,
):
    events: list[str] = []
    monkeypatch.setattr(
        module,
        "_record_checkpoint_event",
        lambda source_table, result: events.append(
            f"metric.checkpoint:{source_table}:{result}"
        ),
    )

    with pytest.raises(IngestCheckpointCursorUnavailableError):
        module.ingest(
            tables=["game"],
            lease_run_id=RUN_ID,
            lease_owner="worker-1",
            checkpoint_scope_key=SCOPE_KEY,
            row_stale_cleanup="off",
            **{**OPTIONS, "limit": 1},
        )

    assert events == ["metric.checkpoint:game:rejected"]


def test_incompatible_checkpoint_start_records_only_incompatible(monkeypatch):
    events: list[str] = []

    with pytest.raises(IngestCheckpointIncompatibleError):
        _run_fake_checkpoint_ingest(
            monkeypatch,
            rows=[],
            events=events,
            start_error=IngestCheckpointIncompatibleError("stored state changed"),
            observe_metrics=True,
        )

    assert events == ["metric.checkpoint:teams:incompatible"]


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


@pytest.mark.parametrize("scope_key", ["   ", "x" * 65])
def test_leased_ingest_rejects_invalid_checkpoint_scope_before_runtime_or_connections(
    monkeypatch,
    scope_key,
):
    def _unexpected(*_args, **_kwargs):
        raise AssertionError("runtime and connections must not be opened")

    monkeypatch.setattr(module, "_require_psycopg", _unexpected)
    monkeypatch.setattr(module, "get_settings", _unexpected)
    monkeypatch.setattr(module.psycopg, "connect", _unexpected)

    with pytest.raises(ValueError, match="checkpoint_scope_key"):
        module.ingest(
            tables=["game"],
            lease_run_id=RUN_ID,
            lease_owner="worker-1",
            checkpoint_scope_key=scope_key,
            row_stale_cleanup="off",
            **OPTIONS,
        )


@pytest.mark.parametrize("scope_key", ["   ", "x" * 65])
def test_direct_checkpoint_ingest_rejects_invalid_scope_before_runtime_or_cursors(
    monkeypatch,
    scope_key,
):
    def _unexpected(*_args, **_kwargs):
        raise AssertionError("runtime and cursors must not be opened")

    monkeypatch.setattr(module, "get_settings", _unexpected)

    with pytest.raises(ValueError, match="checkpoint_scope_key"):
        module.ingest_table(
            SimpleNamespace(cursor=_unexpected),
            SimpleNamespace(cursor=_unexpected),
            "teams",
            limit=None,
            embed_batch_size=2,
            read_batch_size=2,
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
            lease_guard=lambda _write=False: None,
            checkpoint_run_id=RUN_ID,
            checkpoint_scope_key=scope_key,
        )


def test_direct_checkpoint_ingest_strips_scope_before_session_start(monkeypatch):
    captured_scopes: list[str] = []

    _run_fake_checkpoint_ingest(
        monkeypatch,
        rows=[],
        checkpoint_scope_key=f"  {SCOPE_KEY}  ",
        captured_scopes=captured_scopes,
    )

    assert captured_scopes == [SCOPE_KEY]


def test_chunk_write_and_checkpoint_precede_one_commit(monkeypatch):
    result, events = _run_fake_checkpoint_ingest(
        monkeypatch,
        rows=[{"id": 1}, {"id": 2}],
    )

    assert events.index("rag_chunks.executemany") < events.index("checkpoint.upsert")
    assert events.index("checkpoint.upsert") < events.index("destination.commit")
    assert events.count("destination.commit") == 2
    assert result.source_rows == 2


def test_new_checkpoint_events_and_batch_metrics_follow_durable_commits(monkeypatch):
    result, events = _run_fake_checkpoint_ingest(
        monkeypatch,
        rows=[{"id": 1}, {"id": 2}],
        observe_metrics=True,
    )

    first_commit = events.index("destination.commit")
    created = "metric.checkpoint:teams:created"
    advanced = "metric.checkpoint:teams:advanced"
    completed = "metric.checkpoint:teams:completed"
    batch = "metric.batch:teams:2:2"
    assert first_commit < events.index(created)
    assert first_commit < events.index(batch)
    assert first_commit < events.index(advanced)
    assert events.index("destination.commit", first_commit + 1) < events.index(completed)
    assert events.count(created) == 1
    assert events.count(advanced) == 1
    assert events.count(completed) == 1
    assert events.count(batch) == 1
    assert result.attempt_source_rows == 2
    assert result.attempt_written_chunks == 2


def test_new_zero_row_checkpoint_is_created_only_after_completion_commit(monkeypatch):
    _result, events = _run_fake_checkpoint_ingest(
        monkeypatch,
        rows=[],
        observe_metrics=True,
    )

    created = "metric.checkpoint:teams:created"
    completed = "metric.checkpoint:teams:completed"
    assert events.index("destination.commit") < events.index(created)
    assert events.index("destination.commit") < events.index(completed)
    assert events.count(created) == 1
    assert events.count(completed) == 1
    assert not any(event.startswith("metric.batch:") for event in events)
    assert "metric.checkpoint:teams:advanced" not in events


def test_failed_zero_row_completion_commit_emits_no_created_or_completed(monkeypatch):
    events: list[str] = []

    with pytest.raises(RuntimeError, match="scripted destination commit failure"):
        _run_fake_checkpoint_ingest(
            monkeypatch,
            rows=[],
            events=events,
            fail_commits={1},
            observe_metrics=True,
        )

    assert "destination.commit" in events
    assert not any(event.startswith("metric.checkpoint:") for event in events)
    assert not any(event.startswith("metric.batch:") for event in events)


def test_failed_first_advance_emits_no_durable_lifecycle_or_batch_metrics(monkeypatch):
    events: list[str] = []

    with pytest.raises(RuntimeError, match="scripted checkpoint failure"):
        _run_fake_checkpoint_ingest(
            monkeypatch,
            rows=[{"id": 1}],
            events=events,
            fail_advance=True,
            observe_metrics=True,
        )

    assert not any(event.startswith("metric.checkpoint:") for event in events)
    assert not any(event.startswith("metric.batch:") for event in events)


def test_failed_first_advance_commit_emits_no_created_advanced_or_batch(monkeypatch):
    events: list[str] = []

    with pytest.raises(RuntimeError, match="scripted destination commit failure"):
        _run_fake_checkpoint_ingest(
            monkeypatch,
            rows=[{"id": 1}],
            events=events,
            fail_commits={1},
            observe_metrics=True,
        )

    assert "destination.commit" in events
    assert not any(event.startswith("metric.checkpoint:") for event in events)
    assert not any(event.startswith("metric.batch:") for event in events)


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
        observe_metrics=True,
    )

    assert "source.execute" not in events
    assert events == ["metric.checkpoint:teams:resumed"]
    assert result.source_rows == 10
    assert result.attempt_source_rows == 0
    assert result.attempt_written_chunks == 0


def test_failure_before_destination_commit_preserves_durable_state(monkeypatch):
    initial = _CheckpointRecord(
        cursor=CheckpointCursor((0,)),
        committed_batches=2,
        source_rows=4,
        written_chunks=3,
        source_updated_before=CUTOFF,
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
            source_updated_before=CUTOFF,
        ),
        durable_chunk_writes=1,
    )
    captured_resumes: list[CheckpointCursor | None] = []

    result, events = _run_fake_checkpoint_ingest(
        monkeypatch,
        rows=[{"id": 2}],
        state=state,
        captured_resumes=captured_resumes,
        observe_metrics=True,
    )

    assert captured_resumes == [CheckpointCursor((1,))]
    assert events.count("metric.checkpoint:teams:resumed") == 1
    assert "metric.checkpoint:teams:created" not in events
    assert events.count("metric.checkpoint:teams:advanced") == 1
    assert events.count("metric.checkpoint:teams:completed") == 1
    assert events.count("metric.batch:teams:1:1") == 1
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
            observe_metrics=True,
        )

    assert state.durable == _CheckpointRecord(
        cursor=CheckpointCursor((1,)),
        committed_batches=1,
        source_rows=1,
        written_chunks=1,
    )
    assert first_events[-2:] == ["checkpoint.complete", "destination.commit"]
    assert first_events.count("metric.checkpoint:teams:created") == 1
    assert first_events.count("metric.checkpoint:teams:advanced") == 1
    assert first_events.count("metric.batch:teams:1:1") == 1
    assert "metric.checkpoint:teams:completed" not in first_events

    second_events: list[str] = []
    result, second_events = _run_fake_checkpoint_ingest(
        monkeypatch,
        rows=[],
        state=state,
        events=second_events,
        observe_metrics=True,
    )

    assert second_events == [
        "metric.checkpoint:teams:resumed",
        "source.execute",
        "checkpoint.complete",
        "destination.commit",
        "metric.checkpoint:teams:completed",
    ]
    assert result.source_rows == 1
    assert result.attempt_source_rows == 0
    assert result.checkpoint_completed is True
    assert not any(event.startswith("metric.batch:") for event in second_events)


def test_first_committed_batch_is_counted_when_later_batch_commit_fails(monkeypatch):
    state = _CheckpointState()
    first_events: list[str] = []

    with pytest.raises(RuntimeError, match="scripted destination commit failure"):
        _run_fake_checkpoint_ingest(
            monkeypatch,
            rows=[{"id": 1}, {"id": 2}],
            state=state,
            events=first_events,
            fail_commits={2},
            read_batch_size=1,
            observe_metrics=True,
        )

    assert first_events.count("metric.batch:teams:1:1") == 1
    assert first_events.count("metric.checkpoint:teams:created") == 1
    assert first_events.count("metric.checkpoint:teams:advanced") == 1
    assert state.durable is not None
    assert state.durable.cursor == CheckpointCursor((1,))

    retry_events: list[str] = []
    result, retry_events = _run_fake_checkpoint_ingest(
        monkeypatch,
        rows=[{"id": 2}],
        state=state,
        events=retry_events,
        read_batch_size=1,
        observe_metrics=True,
    )

    assert retry_events.count("metric.batch:teams:1:1") == 1
    assert retry_events.count("metric.checkpoint:teams:resumed") == 1
    assert "metric.checkpoint:teams:created" not in retry_events
    assert result.source_rows == 2


def test_checkpoint_ingest_increments_existing_counters_at_batch_commit(monkeypatch):
    rows_before = _read_metric_value(
        "ai_ingest_table_source_rows_total", {"source_table": "teams"}
    )
    chunks_before = _read_metric_value(
        "ai_ingest_table_written_chunks_total", {"source_table": "teams"}
    )

    _run_fake_checkpoint_ingest(
        monkeypatch,
        rows=[{"id": 1}, {"id": 2}],
    )

    assert _read_metric_value(
        "ai_ingest_table_source_rows_total", {"source_table": "teams"}
    ) - rows_before == 2
    assert _read_metric_value(
        "ai_ingest_table_written_chunks_total", {"source_table": "teams"}
    ) - chunks_before == 2


def test_kbo_seasons_full_checkpoint_skips_source_cutoff_sampling(monkeypatch):
    captured_cutoffs: list[datetime | None] = []

    result, events = _run_fake_checkpoint_ingest(
        monkeypatch,
        table_name="kbo_seasons",
        rows=[],
        captured_cutoffs=captured_cutoffs,
        source_clock_values=[CUTOFF],
    )

    assert "source.clock" not in events
    assert "source.execute" in events
    assert captured_cutoffs == [None]
    assert result.attempt_source_rows == 0


def test_game_full_checkpoint_keeps_null_updates_and_excludes_late_updates(
    monkeypatch,
):
    captured_cutoffs: list[datetime | None] = []

    result, events = _run_fake_checkpoint_ingest(
        monkeypatch,
        table_name="game",
        rows=[
            {"id": 1, "updated_at": None},
            {"id": 2, "updated_at": CUTOFF + timedelta(minutes=1)},
        ],
        captured_cutoffs=captured_cutoffs,
        source_clock_values=[CUTOFF],
    )

    assert events.count("source.clock") == 1
    assert captured_cutoffs == [CUTOFF]
    assert result.attempt_source_rows == 1


def test_completed_progressed_checkpoint_without_cutoff_fails_before_source_query(
    monkeypatch,
):
    state = _CheckpointState(
        durable=_CheckpointRecord(
            source_table="game",
            cursor=CheckpointCursor((1,)),
            committed_batches=1,
            source_rows=1,
            completed=True,
        )
    )
    events: list[str] = []

    with pytest.raises(IngestCheckpointIncompatibleError):
        _run_fake_checkpoint_ingest(
            monkeypatch,
            table_name="game",
            rows=[],
            state=state,
            events=events,
            observe_metrics=True,
            source_clock_values=[CUTOFF],
        )

    assert events == ["metric.checkpoint:game:incompatible"]


def test_recovery_excludes_row_updated_behind_cursor_after_fixed_cutoff(monkeypatch):
    state = _CheckpointState(
        durable=_CheckpointRecord(
            source_table="game",
            cursor=CheckpointCursor((1,)),
            committed_batches=1,
            source_rows=1,
            written_chunks=1,
            max_updated_at=CUTOFF - timedelta(hours=1),
            source_updated_before=CUTOFF,
        )
    )
    captured_cutoffs: list[datetime | None] = []

    result, events = _run_fake_checkpoint_ingest(
        monkeypatch,
        table_name="game",
        rows=[
            {"id": 1, "updated_at": CUTOFF + timedelta(minutes=1)},
            {"id": 2, "updated_at": CUTOFF - timedelta(minutes=1)},
        ],
        state=state,
        captured_cutoffs=captured_cutoffs,
    )

    assert "source.clock" not in events
    assert captured_cutoffs == [CUTOFF]
    assert result.source_rows == 2
    assert result.attempt_source_rows == 1


def test_suffix_update_after_cutoff_cannot_advance_current_watermark(monkeypatch):
    previous_max = CUTOFF - timedelta(hours=1)
    state = _CheckpointState(
        durable=_CheckpointRecord(
            source_table="game",
            cursor=CheckpointCursor((1,)),
            committed_batches=1,
            source_rows=1,
            written_chunks=1,
            max_updated_at=previous_max,
            source_updated_before=CUTOFF,
        )
    )

    result, _events = _run_fake_checkpoint_ingest(
        monkeypatch,
        table_name="game",
        rows=[{"id": 2, "updated_at": CUTOFF + timedelta(minutes=2)}],
        state=state,
    )

    assert result.attempt_source_rows == 0
    assert result.max_updated_at == previous_max


def test_crash_restart_reuses_persisted_cutoff_without_resampling(monkeypatch):
    state = _CheckpointState()
    first_events: list[str] = []

    with pytest.raises(RuntimeError, match="scripted destination commit failure"):
        _run_fake_checkpoint_ingest(
            monkeypatch,
            table_name="game",
            rows=[{"id": 1, "updated_at": CUTOFF - timedelta(minutes=1)}],
            state=state,
            events=first_events,
            fail_commits={2},
            source_clock_values=[CUTOFF],
        )

    assert first_events.count("source.clock") == 1
    assert state.durable is not None
    assert state.durable.source_updated_before == CUTOFF

    second_events: list[str] = []
    captured_cutoffs: list[datetime | None] = []
    _run_fake_checkpoint_ingest(
        monkeypatch,
        table_name="game",
        rows=[],
        state=state,
        events=second_events,
        source_clock_values=[CUTOFF + timedelta(hours=1)],
        captured_cutoffs=captured_cutoffs,
    )

    assert "source.clock" not in second_events
    assert captured_cutoffs == [CUTOFF]


def test_next_normal_run_sees_updates_deferred_by_prior_cutoff(monkeypatch):
    previous_watermark = CUTOFF - timedelta(hours=1)
    deferred = CUTOFF + timedelta(minutes=1)

    result, events = _run_fake_checkpoint_ingest(
        monkeypatch,
        table_name="game",
        rows=[{"id": 1, "updated_at": deferred}],
        state=_CheckpointState(),
        since=previous_watermark,
        source_clock_values=[CUTOFF + timedelta(hours=1)],
    )

    assert events.count("source.clock") == 1
    assert result.attempt_source_rows == 1
    assert result.max_updated_at == deferred


def test_zero_row_completion_persists_sampled_source_cutoff(monkeypatch):
    state = _CheckpointState()

    result, events = _run_fake_checkpoint_ingest(
        monkeypatch,
        table_name="game",
        rows=[],
        state=state,
        source_clock_values=[CUTOFF],
    )

    assert events.index("source.clock") < events.index("source.execute")
    assert state.durable is not None
    assert state.durable.completed is True
    assert state.durable.source_updated_before == CUTOFF
    assert result.attempt_source_rows == 0


def test_progressed_incomplete_checkpoint_without_cutoff_fails_closed(monkeypatch):
    state = _CheckpointState(
        durable=_CheckpointRecord(
            source_table="game",
            cursor=CheckpointCursor((1,)),
            committed_batches=1,
            source_rows=1,
        )
    )

    with pytest.raises(IngestCheckpointIncompatibleError):
        _run_fake_checkpoint_ingest(
            monkeypatch,
            table_name="game",
            rows=[],
            state=state,
            source_clock_values=[CUTOFF],
        )


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
