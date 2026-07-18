"""Regression tests for the auto-brief warm-path operations smoke."""

import asyncio
from types import SimpleNamespace

from scripts import smoke_coach_auto_brief_ops as ops_smoke


def test_warm_path_smoke_accepts_router_hit_cache_state(monkeypatch):
    async def fake_call_analyze(**_kwargs):
        return {
            "status": "skipped",
            "reason": "cache_hit",
            "meta": {
                "cache_state": "HIT",
                "cached": True,
            },
        }

    monkeypatch.setattr(ops_smoke, "call_analyze", fake_call_analyze)

    result = asyncio.run(
        ops_smoke.run_warm_path_smoke(
            target=SimpleNamespace(game_id="game-1", cache_key="cache-1"),
            base_url="http://127.0.0.1:8001",
            internal_api_key="test-token",
            timeout_seconds=1.0,
        )
    )

    assert result.ok is True
    assert result.status == "skipped"
    assert result.cache_state == "HIT"
    assert result.cached is True
