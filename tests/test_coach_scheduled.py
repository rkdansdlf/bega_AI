"""
Coach 기능 테스트 모듈.

Fast Path, 캐싱, 응답 검증기 등을 테스트합니다.
"""

import pytest
import json
import asyncio
import logging
import os
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from pydantic import ValidationError

from tests.coach_test_support import (
    _build_game_evidence,
    _collect_sse_text,
    _extract_sse_meta_events,
    _install_coach_endpoint_cache_hit,
)

class TestAttachScheduledWinProbability:
    def _tool_results(self, home=(7, 3, 0), away=(5, 5, 0)):
        def block(record):
            wins, losses, draws = record
            return {
                "recent": {
                    "summary": {
                        "wins": wins,
                        "losses": losses,
                        "draws": draws,
                        "run_diff": wins - losses,
                    }
                }
            }

        return {"home": block(home), "away": block(away)}

    def test_uses_payload_value_when_valid(self):
        from app.routers.coach import _attach_scheduled_win_probability

        meta = {}
        _attach_scheduled_win_probability(
            meta,
            game_status_bucket="SCHEDULED",
            tool_results=self._tool_results(),
            response_payload={"win_probability_home": 0.6123},
        )
        assert meta["win_probability_home"] == 0.612

    def test_computes_when_payload_missing_and_data_sufficient(self):
        from app.routers.coach import _attach_scheduled_win_probability

        meta = {}
        _attach_scheduled_win_probability(
            meta,
            game_status_bucket="SCHEDULED",
            tool_results=self._tool_results(),
            response_payload=None,
        )
        assert "win_probability_home" in meta
        assert 0.30 <= meta["win_probability_home"] <= 0.75

    def test_no_key_when_data_insufficient(self):
        from app.routers.coach import _attach_scheduled_win_probability

        meta = {}
        _attach_scheduled_win_probability(
            meta,
            game_status_bucket="SCHEDULED",
            tool_results={"home": {}, "away": {}},
            response_payload={"win_probability_home": 1.5},  # 범위 밖 → 무시
        )
        assert "win_probability_home" not in meta

    def test_no_key_when_not_scheduled(self):
        from app.routers.coach import _attach_scheduled_win_probability

        meta = {}
        _attach_scheduled_win_probability(
            meta,
            game_status_bucket="COMPLETED",
            tool_results=self._tool_results(),
            response_payload={"win_probability_home": 0.55},
        )
        assert "win_probability_home" not in meta


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
