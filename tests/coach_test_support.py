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


async def _collect_sse_text(response) -> str:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode("utf-8"))
        elif isinstance(chunk, dict):
            event_name = chunk.get("event")
            data = chunk.get("data")
            if event_name:
                chunks.append(f"event: {event_name}\n")
            chunks.append(f"data: {data}\n\n")
        else:
            chunks.append(str(chunk))
    return "".join(chunks)


def _extract_sse_meta_events(sse_text: str) -> list[dict]:
    events: list[dict] = []
    event_name = "message"
    data_lines: list[str] = []
    for line in sse_text.splitlines():
        if line == "":
            if event_name == "meta" and data_lines:
                events.append(json.loads("\n".join(data_lines)))
            event_name = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if event_name == "meta" and data_lines:
        events.append(json.loads("\n".join(data_lines)))
    return events


def _install_coach_endpoint_cache_hit(monkeypatch, evidence, cached_payload=None):
    from app.routers import coach

    class _Agent:
        def _convert_team_id_to_name(self, team_id):
            return {
                "LG": "LG 트윈스",
                "KT": "KT 위즈",
            }.get(team_id, str(team_id))

    async def _fake_resolve_target_year(payload, pool):
        return (2026, "test")

    async def _fake_collect_game_evidence(*args, **kwargs):
        return evidence

    async def _fake_claim_cache_generation(**kwargs):
        return (
            "HIT",
            cached_payload
            or {
                "headline": "테스트 브리핑",
                "sentiment": "neutral",
                "key_metrics": [],
                "analysis": {
                    "summary": "캐시된 분석입니다.",
                    "strengths": [],
                    "weaknesses": [],
                    "risks": [],
                },
                "detailed_markdown": "캐시된 분석입니다.",
                "coach_note": "캐시된 코치 노트입니다.",
            },
            None,
            None,
            1,
        )

    monkeypatch.setattr(coach, "get_connection_pool", lambda: object())
    monkeypatch.setattr(coach, "_resolve_target_year", _fake_resolve_target_year)
    monkeypatch.setattr(coach, "_collect_game_evidence", _fake_collect_game_evidence)
    monkeypatch.setattr(coach, "resolve_coach_openrouter_models", lambda primary, fallback: ["openrouter/free"])
    monkeypatch.setattr(coach, "_find_missing_focus_sections", lambda response, focus: [])
    monkeypatch.setattr(coach, "_build_cache_lease_owner", lambda: "test-owner")
    monkeypatch.setattr(coach, "_claim_cache_generation", _fake_claim_cache_generation)
    return _Agent()


def _build_game_evidence(**overrides):
    from app.routers.coach import GameEvidence

    base = dict(
        season_year=2025,
        home_team_code="LG",
        away_team_code="KT",
        home_team_name="LG 트윈스",
        away_team_name="KT 위즈",
        game_id="202503120001",
        season_id=20250,
        game_date="2025-03-12",
        game_status="SCHEDULED",
        game_status_bucket="SCHEDULED",
        league_type_code=0,
        stage_label="REGULAR",
        round_display="정규시즌",
        home_pitcher="임찬규",
        away_pitcher="쿠에바스",
        lineup_announced=True,
        home_lineup=["홍창기", "박해민"],
        away_lineup=["강백호", "로하스"],
        summary_items=["결승타 홍창기"],
        stadium_name="잠실",
        start_time="18:30",
        weather="맑음",
    )
    base.update(overrides)
    return GameEvidence(**base)
