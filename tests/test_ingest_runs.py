from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.ingest_runs import (
    IngestRunMode,
    IngestRunRequest,
    IngestRunStatus,
    build_request_key,
    ensure_transition,
)


def test_request_key_is_stable_for_equivalent_table_order():
    left = IngestRunRequest(tables=("game", "teams"), season_year=2026)
    right = IngestRunRequest(tables=("teams", "game"), season_year=2026)

    assert build_request_key(left) == build_request_key(right)


def test_request_key_uses_work_identity_not_trigger_source():
    scheduled = IngestRunRequest(
        tables=("game",),
        trigger_source="backend_scheduled",
    )
    manual = IngestRunRequest(
        tables=("game",),
        trigger_source="manual_api",
    )

    assert build_request_key(scheduled) == build_request_key(manual)


def test_request_normalization_deduplicates_tables_and_preserves_iso_since():
    request = IngestRunRequest(
        tables=(" Teams ", "game", "teams"),
        mode="incremental",
        trigger_source="backend_scheduled",
        since=datetime(2026, 7, 15, 4, 30, tzinfo=UTC),
    ).normalized()

    assert request.tables == ("game", "teams")
    assert request.mode is IngestRunMode.INCREMENTAL
    assert request.trigger_source == "BACKEND_SCHEDULED"
    assert request.to_payload()["since"] == "2026-07-15T04:30:00+00:00"


@pytest.mark.parametrize("tables", [(), (" ",), ("rag_chunks",)])
def test_request_normalization_rejects_empty_or_managed_target(tables):
    with pytest.raises(ValueError):
        IngestRunRequest(tables=tables).normalized()


def test_terminal_run_cannot_transition_back_to_running():
    with pytest.raises(ValueError, match="illegal ingest run transition"):
        ensure_transition(IngestRunStatus.SUCCEEDED, IngestRunStatus.RUNNING)


def test_expired_running_lease_can_requeue():
    ensure_transition(IngestRunStatus.RUNNING, IngestRunStatus.QUEUED)
