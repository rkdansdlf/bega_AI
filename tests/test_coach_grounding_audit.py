from pathlib import Path

import httpx

from app.routers.coach import EvidenceSeriesState, GameEvidence, assess_game_evidence
from scripts.coach_grounding_audit import (
    BackendMatchMeta,
    build_league_context,
    build_report_payload,
    build_diagnosis_summary,
    call_coach_analyze,
    compare_backend_meta,
    fetch_backend_match_meta,
    parse_sse_stream,
    render_markdown_report,
    resolve_default_internal_api_key,
    select_validation_samples,
    validate_capture,
    write_report_files,
)


def _build_evidence(**overrides):
    base = dict(
        game_id="20251011LGKT001",
        season_id=20255,
        season_year=2025,
        game_date="2025-10-11",
        game_status="SCHEDULED",
        game_status_bucket="SCHEDULED",
        home_team_code="LG",
        away_team_code="KT",
        home_team_name="LG 트윈스",
        away_team_name="KT 위즈",
        league_type_code=0,
        stage_label="REGULAR",
        round_display="정규시즌",
        home_pitcher="임찬규",
        away_pitcher="쿠에바스",
        lineup_announced=True,
        home_lineup=["홍창기"],
        away_lineup=["강백호"],
        summary_items=["summary"],
        stadium_name="잠실",
        start_time="18:30",
        weather="맑음",
        series_state=None,
    )
    base.update(overrides)
    return GameEvidence(**base)


def test_assess_game_evidence_grounded_when_all_game_sources_present():
    assessment = assess_game_evidence(_build_evidence())

    assert assessment.expected_data_quality == "grounded"
    assert assessment.root_causes == []
    assert assessment.metadata_present is True
    assert assessment.summary_present is True
    assert assessment.lineup_announced is True


def test_assess_game_evidence_partial_when_announced_data_missing():
    assessment = assess_game_evidence(
        _build_evidence(
            home_pitcher=None,
            lineup_announced=False,
            summary_items=[],
            stadium_name=None,
            start_time=None,
            weather=None,
        )
    )

    assert assessment.expected_data_quality == "partial"
    assert assessment.root_causes == [
        "missing_starters",
        "missing_lineups",
        "missing_summary",
        "missing_metadata",
    ]


def test_select_validation_samples_keeps_same_matchup_pair():
    diagnosis = [
        {
            "game_id": "g1",
            "game_date": "2025-03-01",
            "season_year": 2025,
            "home_team_id": "LG",
            "away_team_id": "KT",
            "stage_label": "REGULAR",
            "expected_data_quality": "partial",
        },
        {
            "game_id": "g2",
            "game_date": "2025-03-02",
            "season_year": 2025,
            "home_team_id": "KT",
            "away_team_id": "LG",
            "stage_label": "REGULAR",
            "expected_data_quality": "partial",
        },
        {
            "game_id": "g3",
            "game_date": "2025-03-03",
            "season_year": 2025,
            "home_team_id": "SSG",
            "away_team_id": "NC",
            "stage_label": "KS",
            "expected_data_quality": "grounded",
        },
        {
            "game_id": "g4",
            "game_date": "2025-03-04",
            "season_year": 2025,
            "home_team_id": "KIA",
            "away_team_id": "두산",
            "stage_label": "PO",
            "expected_data_quality": "partial",
        },
        {
            "game_id": "g5",
            "game_date": "2025-03-05",
            "season_year": 2025,
            "home_team_id": "한화",
            "away_team_id": "롯데",
            "stage_label": "REGULAR",
            "expected_data_quality": "grounded",
        },
    ]

    selected = select_validation_samples(diagnosis, limit=4)
    selected_ids = [item["game_id"] for item in selected]

    assert len(selected) == 4
    assert "g1" in selected_ids
    assert "g2" in selected_ids
    assert "g3" in selected_ids
    assert "g4" in selected_ids or "g5" in selected_ids


def test_compare_backend_meta_detects_expected_mismatch():
    record = {
        "stage_label": "PO",
        "series_game_no": 3,
    }
    backend_meta = BackendMatchMeta(
        game_id="g1",
        season_id=20254,
        league_type="REGULAR",
        post_season_series="KS",
        series_game_no=4,
        home_pitcher="임찬규",
        away_pitcher="쿠에바스",
        game_status="SCHEDULED",
    )

    failures = compare_backend_meta(record, backend_meta)

    assert any("leagueType mismatch" in item for item in failures)
    assert any("postSeasonSeries mismatch" in item for item in failures)
    assert any("seriesGameNo mismatch" in item for item in failures)


def test_build_league_context_prefers_backend_stage_metadata():
    record = {
        "season_id": 20251,
        "season_year": 2025,
        "league_type_code": 0,
        "stage_label": "REGULAR",
        "series_game_no": None,
        "game_date": "2025-03-08",
        "lineup_announced": True,
    }
    backend_meta = BackendMatchMeta(
        game_id="g1",
        season_id=260,
        league_type="PRE",
        post_season_series=None,
        series_game_no=None,
        home_pitcher="반즈",
        away_pitcher="윤영철",
        game_status="COMPLETED",
    )

    context = build_league_context(record, backend_meta)

    assert context["season"] == 260
    assert context["league_type"] == "PRE"
    assert context["league_type_code"] == 1
    assert context["stage_label"] == "PRE"


class _FakeResponse:
    def __init__(self, lines):
        self.status_code = 200
        self._lines = lines

    def iter_lines(self):
        for line in self._lines:
            yield line


def test_parse_sse_stream_extracts_meta_and_done():
    response = _FakeResponse(
        [
            "event: message",
            'data: {"delta":"hello"}',
            "event: meta",
            'data: {"generation_mode":"deterministic_auto","data_quality":"grounded","used_evidence":["game"],"structured_response":{"headline":"ok","sentiment":"neutral","key_metrics":[],"analysis":{"strengths":[],"weaknesses":[],"risks":[]},"detailed_markdown":"","coach_note":"ok"}}',
            "event: done",
            "data: [DONE]",
        ]
    )

    parsed = parse_sse_stream(response)

    assert parsed["done_seen"] is True
    assert parsed["answer"] == "hello"
    assert parsed["meta"]["generation_mode"] == "deterministic_auto"
    assert parsed["structured_response"]["headline"] == "ok"


def test_fetch_backend_match_meta_falls_back_when_detail_endpoint_fails():
    record = {
        "game_id": "g1",
        "game_date": "2025-03-08",
        "game_status": "SCHEDULED",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/matches":
            return httpx.Response(
                200,
                json=[
                    {
                        "gameId": "g1",
                        "seasonId": 20255,
                        "leagueType": "REGULAR",
                        "postSeasonSeries": None,
                        "seriesGameNo": None,
                        "homePitcher": {"name": "임찬규"},
                        "awayPitcher": {"name": "쿠에바스"},
                    }
                ],
            )
        if request.url.path == "/api/matches/g1":
            return httpx.Response(500, json={"message": "server error"})
        raise AssertionError(f"unexpected path: {request.url.path}")

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://testserver"
    )
    try:
        backend_meta = fetch_backend_match_meta(client, "http://testserver", record)
    finally:
        client.close()

    assert backend_meta.home_pitcher == "임찬규"
    assert backend_meta.away_pitcher == "쿠에바스"
    assert backend_meta.game_status == "SCHEDULED"
    assert backend_meta.detail_status_code == 500
    assert "match_detail_failed" in str(backend_meta.detail_error)


def test_call_coach_analyze_returns_clean_error_payload_for_non_200():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='{"detail":"Unauthorized"}')

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://testserver"
    )
    try:
        capture = call_coach_analyze(
            client,
            "http://testserver",
            {"home_team_id": "LG", "request_mode": "auto_brief"},
            {},
        )
    finally:
        client.close()

    assert capture["status_code"] == 401
    assert capture["done_seen"] is False
    assert "Unauthorized" in str(capture["error_payload"])


def test_call_coach_analyze_returns_timeout_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://testserver"
    )
    try:
        capture = call_coach_analyze(
            client,
            "http://testserver",
            {"home_team_id": "LG", "request_mode": "auto_brief"},
            {},
        )
    finally:
        client.close()

    assert capture["status_code"] == 0
    assert capture["done_seen"] is False
    assert "coach_request_timeout" in str(capture["error_payload"])


def test_validate_capture_soft_warns_on_grounding_reason_codes_after_fallback():
    record = {
        "expected_data_quality": "grounded",
        "lineup_announced": True,
        "home_pitcher_present": True,
        "away_pitcher_present": True,
        "root_causes": [],
        "stage_label": "REGULAR",
    }
    capture = {
        "status_code": 200,
        "done_seen": True,
        "meta": {
            "generation_mode": "evidence_fallback",
            "data_quality": "grounded",
            "used_evidence": ["game"],
            "grounding_warnings": ["근거 밖 수치 감지"],
            "grounding_reasons": ["unsupported_numeric_claim"],
            "supported_fact_count": 8,
        },
        "structured_response": {
            "headline": "ok",
            "sentiment": "neutral",
            "key_metrics": [],
            "analysis": {"strengths": [], "weaknesses": [], "risks": []},
            "detailed_markdown": "",
            "coach_note": "",
        },
    }

    hard_failures, soft_warnings = validate_capture(
        record=record,
        request_mode="manual_detail",
        capture=capture,
    )

    assert hard_failures == []
    assert any("unsupported numeric claim" in item for item in soft_warnings)
    assert "manual_detail fallback 발생" in soft_warnings


def test_validate_capture_hard_fails_on_grounding_reason_codes_without_fallback():
    record = {
        "expected_data_quality": "grounded",
        "lineup_announced": True,
        "home_pitcher_present": True,
        "away_pitcher_present": True,
        "root_causes": [],
        "stage_label": "REGULAR",
    }
    capture = {
        "status_code": 200,
        "done_seen": True,
        "meta": {
            "generation_mode": "llm_manual",
            "data_quality": "grounded",
            "used_evidence": ["game"],
            "grounding_warnings": ["근거 밖 엔티티 감지"],
            "grounding_reasons": ["unsupported_entity_name"],
            "supported_fact_count": 8,
        },
        "structured_response": {
            "headline": "ok",
            "sentiment": "neutral",
            "key_metrics": [],
            "analysis": {"strengths": [], "weaknesses": [], "risks": []},
            "detailed_markdown": "",
            "coach_note": "",
        },
    }

    hard_failures, soft_warnings = validate_capture(
        record=record,
        request_mode="manual_detail",
        capture=capture,
    )

    assert any("unsupported entity name" in item for item in hard_failures)
    assert "manual_detail fallback 발생" not in soft_warnings


def test_resolve_default_internal_api_key_reads_local_env_files(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("AI_INTERNAL_TOKEN", raising=False)
    (tmp_path / ".env.prod").write_text(
        "AI_INTERNAL_TOKEN=file-token\n", encoding="utf-8"
    )

    assert resolve_default_internal_api_key(tmp_path) == "file-token"


def test_resolve_default_internal_api_key_prefers_env_prod_over_env(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("AI_INTERNAL_TOKEN", raising=False)
    (tmp_path / ".env").write_text("AI_INTERNAL_TOKEN=dev-token\n", encoding="utf-8")
    (tmp_path / ".env.prod").write_text(
        "AI_INTERNAL_TOKEN=prod-token\n", encoding="utf-8"
    )

    assert resolve_default_internal_api_key(tmp_path) == "prod-token"


def test_write_report_files_creates_timestamped_and_latest_outputs(tmp_path: Path):
    report = build_report_payload(
        command="diagnose",
        options={"command": "diagnose"},
        diagnosis=[
            {
                "game_id": "g1",
                "game_date": "2025-03-01",
                "season_year": 2025,
                "home_team_id": "LG",
                "away_team_id": "KT",
                "stage_label": "REGULAR",
                "expected_data_quality": "grounded",
                "root_causes": [],
            }
        ],
        validation=None,
    )

    paths = write_report_files(report, tmp_path)

    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
    assert Path(paths["latest_json"]).exists()
    assert Path(paths["latest_markdown"]).exists()


def test_build_diagnosis_summary_counts_series_state_partial_flags():
    summary = build_diagnosis_summary(
        [
            {
                "game_id": "g1",
                "stage_label": "KS",
                "expected_data_quality": "partial",
                "root_causes": [],
                "series_state_partial": True,
                "series_state_hint_mismatch": False,
            },
            {
                "game_id": "g2",
                "stage_label": "PO",
                "expected_data_quality": "grounded",
                "root_causes": [],
                "series_state_partial": False,
                "series_state_hint_mismatch": False,
            },
        ]
    )

    assert summary["series_state_partial_count"] == 1
    assert summary["series_state_hint_mismatch_count"] == 0


def test_render_markdown_report_includes_series_diagnostics():
    report = build_report_payload(
        command="all",
        options={"command": "all"},
        diagnosis=[
            {
                "game_id": "g1",
                "game_date": "2025-10-31",
                "season_year": 2025,
                "home_team_id": "LG",
                "away_team_id": "HH",
                "stage_label": "KS",
                "expected_data_quality": "partial",
                "root_causes": [],
                "series_state_partial": True,
                "series_state_hint_mismatch": False,
            }
        ],
        validation={
            "targets": ["g1"],
            "results": [
                {
                    "diagnosis": {
                        "game_id": "g1",
                        "away_team_id": "HH",
                        "home_team_id": "LG",
                    },
                    "hard_failures": [],
                    "soft_warnings": [],
                    "series_game_no_mismatch": True,
                    "ok": True,
                }
            ],
            "summary": {
                "total_targets": 1,
                "hard_failure_count": 0,
                "soft_warning_count": 0,
                "passed_targets": 1,
                "failed_targets": 0,
                "series_game_no_mismatch_count": 1,
            },
        },
    )

    markdown = render_markdown_report(report)

    assert "시리즈 진단" in markdown
    assert "partial 1" in markdown
    assert "validation_seriesGameNo_mismatch 1" in markdown
