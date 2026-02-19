from datetime import datetime, timedelta, timezone

from app.routers.coach import _determine_cache_gate, _should_generate_from_gate


def test_completed_cache_hit_even_if_old():
    old_time = datetime.now(timezone.utc) - timedelta(days=30)
    gate = _determine_cache_gate(
        status="COMPLETED",
        has_cached_json=True,
        updated_at=old_time,
    )
    assert gate == "HIT"
    assert _should_generate_from_gate(gate) is False


def test_failed_cache_is_locked():
    gate = _determine_cache_gate(
        status="FAILED",
        has_cached_json=False,
        updated_at=datetime.now(timezone.utc),
    )
    assert gate == "FAILED_LOCKED"
    assert _should_generate_from_gate(gate) is False


def test_pending_fresh_waits_without_generation():
    fresh_time = datetime.now(timezone.utc) - timedelta(seconds=120)
    gate = _determine_cache_gate(
        status="PENDING",
        has_cached_json=False,
        updated_at=fresh_time,
        pending_stale_seconds=300,
    )
    assert gate == "PENDING_WAIT"
    assert _should_generate_from_gate(gate) is False


def test_pending_stale_takeover_allows_generation():
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=301)
    gate = _determine_cache_gate(
        status="PENDING",
        has_cached_json=False,
        updated_at=stale_time,
        pending_stale_seconds=300,
    )
    assert gate == "PENDING_STALE_TAKEOVER"
    assert _should_generate_from_gate(gate) is True


def test_missing_row_becomes_miss_generate():
    gate = _determine_cache_gate(
        status=None,
        has_cached_json=False,
        updated_at=None,
    )
    assert gate == "MISS_GENERATE"
    assert _should_generate_from_gate(gate) is True


def test_gate_to_generation_matrix():
    matrix = {
        "HIT": False,
        "PENDING_WAIT": False,
        "FAILED_LOCKED": False,
        "PENDING_STALE_TAKEOVER": True,
        "MISS_GENERATE": True,
    }
    for gate, expected in matrix.items():
        assert _should_generate_from_gate(gate) is expected
