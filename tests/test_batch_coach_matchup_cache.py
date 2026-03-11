import asyncio

from scripts import batch_coach_matchup_cache as batch_module
from scripts.batch_coach_matchup_cache import (
    _classify_meta_result,
    _extract_error_message,
    _filter_targets_by_cache_state,
    _league_codes_for_type,
    _normalize_cache_state_label,
    _build_analyze_payload,
    _build_cache_verification_result,
    _is_retryable_failure_reason,
    _postseason_mismatch_error_message,
    _sort_targets,
    _build_terminal_cache_result_if_available,
    collect_cache_verification_results,
    load_failed_cache_keys,
    MatchupTarget,
    call_analyze_with_deadline,
    call_analyze_with_retries,
    parse_cache_state_filter,
    parse_status_bucket_filter,
    parse_target_order,
)


def test_extract_error_message_reads_error_field() -> None:
    payload = '{"error": "rate limit", "message": "fallback"}'
    assert _extract_error_message(payload) == "rate limit"


def test_extract_error_message_ignores_empty_error_field() -> None:
    assert _extract_error_message('{"error": "", "message": "   "}') is None


def test_extract_error_message_ignores_empty_payload() -> None:
    assert _extract_error_message("   ") is None


def test_extract_error_message_uses_message_when_error_empty() -> None:
    payload = '{"error": "", "message": "detail available"}'
    assert _extract_error_message(payload) == "detail available"


def test_extract_error_message_fallback_to_raw_non_json() -> None:
    assert _extract_error_message("service unavailable") == "service unavailable"


def test_league_codes_for_type_maps_preseason_and_postseason() -> None:
    assert _league_codes_for_type("PRE") == (1,)
    assert _league_codes_for_type("POST") == (2, 3, 4, 5)


def test_classify_meta_result_accepts_evidence_fallback_generation() -> None:
    status, reason = _classify_meta_result(
        {
            "validation_status": "fallback",
            "generation_mode": "evidence_fallback",
            "cached": False,
            "in_progress": False,
            "cache_state": "MISS_GENERATE",
        }
    )

    assert status == "generated"
    assert reason == "generated_fallback"


def test_is_retryable_failure_reason_handles_transient_categories() -> None:
    assert _is_retryable_failure_reason('http_429:{"detail":"Rate limit"}') is True
    assert _is_retryable_failure_reason("ReadTimeout") is True
    assert _is_retryable_failure_reason("ReadError") is True
    assert (
        _is_retryable_failure_reason("Server disconnected without sending a response.")
        is True
    )
    assert (
        _is_retryable_failure_reason(
            "peer closed connection without sending complete message body"
        )
        is True
    )
    assert _is_retryable_failure_reason("coach_internal_error") is True
    assert _is_retryable_failure_reason("target_wall_timeout") is True
    assert _is_retryable_failure_reason("failed_locked") is True
    assert _is_retryable_failure_reason("validation_fallback") is False


def test_parse_target_order_and_status_bucket_filter() -> None:
    assert parse_target_order("DESC") == "desc"
    assert parse_status_bucket_filter("completed") == "COMPLETED"
    assert parse_cache_state_filter("unresolved") == "UNRESOLVED"


def test_sort_targets_descending_uses_latest_game_first() -> None:
    older = MatchupTarget(
        cache_key="older",
        game_id="2025050101",
        season_id=1,
        season_year=2025,
        game_date="2025-05-01",
        game_type="REGULAR",
        home_team_id="KIA",
        away_team_id="LG",
        league_type_code=0,
        stage_label="REGULAR",
        series_game_no=None,
        game_status_bucket="COMPLETED",
        starter_signature="older",
        lineup_signature="older",
        request_focus=["matchup"],
        request_mode="manual_detail",
        question_override=None,
    )
    newer = MatchupTarget(
        cache_key="newer",
        game_id="2025050201",
        season_id=1,
        season_year=2025,
        game_date="2025-05-02",
        game_type="REGULAR",
        home_team_id="KIA",
        away_team_id="LG",
        league_type_code=0,
        stage_label="REGULAR",
        series_game_no=None,
        game_status_bucket="COMPLETED",
        starter_signature="newer",
        lineup_signature="newer",
        request_focus=["matchup"],
        request_mode="manual_detail",
        question_override=None,
    )

    ordered = _sort_targets([older, newer], order="desc")

    assert [item.cache_key for item in ordered] == ["newer", "older"]


def test_build_analyze_payload_includes_postseason_stage_context() -> None:
    target = MatchupTarget(
        cache_key="post",
        game_id="20251031LGHH0",
        season_id=264,
        season_year=2025,
        game_date="2025-10-31",
        game_type="POST",
        home_team_id="HH",
        away_team_id="LG",
        league_type_code=5,
        stage_label="KS",
        series_game_no=5,
        game_status_bucket="COMPLETED",
        starter_signature="s",
        lineup_signature="l",
        request_focus=["matchup", "recent_form"],
        request_mode="manual_detail",
        question_override=None,
    )

    payload = _build_analyze_payload(target)

    assert payload["league_context"]["league_type"] == "POST"
    assert payload["league_context"]["league_type_code"] == 5
    assert payload["league_context"]["stage_label"] == "KS"
    assert payload["league_context"]["round"] == "KS"
    assert payload["league_context"]["series_game_no"] == 5
    assert payload["league_context"]["game_no"] == 5


def test_normalize_cache_state_label_maps_missing_and_pending_rows() -> None:
    assert _normalize_cache_state_label(None) == "MISSING"
    assert _normalize_cache_state_label(("completed", {}, None)) == "COMPLETED"
    assert _normalize_cache_state_label(("failed", None, "x")) == "FAILED"
    assert _normalize_cache_state_label(("failed_locked", None, "x")) == "PENDING"


def test_filter_targets_by_cache_state_returns_only_unresolved(monkeypatch) -> None:
    completed = MatchupTarget(
        cache_key="completed",
        game_id="2025032201",
        season_id=260,
        season_year=2025,
        game_date="2025-03-22",
        game_type="REGULAR",
        home_team_id="KT",
        away_team_id="HH",
        league_type_code=0,
        stage_label="REGULAR",
        series_game_no=None,
        game_status_bucket="COMPLETED",
        starter_signature="s1",
        lineup_signature="l1",
        request_focus=["matchup"],
        request_mode="manual_detail",
        question_override=None,
    )
    missing = MatchupTarget(
        cache_key="missing",
        game_id="2025032202",
        season_id=260,
        season_year=2025,
        game_date="2025-03-22",
        game_type="REGULAR",
        home_team_id="LG",
        away_team_id="NC",
        league_type_code=0,
        stage_label="REGULAR",
        series_game_no=None,
        game_status_bucket="COMPLETED",
        starter_signature="s2",
        lineup_signature="l2",
        request_focus=["matchup"],
        request_mode="manual_detail",
        question_override=None,
    )
    pending = MatchupTarget(
        cache_key="pending",
        game_id="2025032203",
        season_id=260,
        season_year=2025,
        game_date="2025-03-22",
        game_type="REGULAR",
        home_team_id="SS",
        away_team_id="KIA",
        league_type_code=0,
        stage_label="REGULAR",
        series_game_no=None,
        game_status_bucket="COMPLETED",
        starter_signature="s3",
        lineup_signature="l3",
        request_focus=["matchup"],
        request_mode="manual_detail",
        question_override=None,
    )

    monkeypatch.setattr(
        batch_module,
        "_fetch_cache_rows",
        lambda keys: {
            "completed": ("COMPLETED", {"meta": {}}, None),
            "pending": ("FAILED_LOCKED", None, None),
        },
    )

    filtered = _filter_targets_by_cache_state(
        [completed, missing, pending], "UNRESOLVED"
    )

    assert [target.cache_key for target in filtered] == ["missing", "pending"]


def test_postseason_mismatch_error_message_includes_count_and_samples() -> None:
    class _Mismatch:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

    message = _postseason_mismatch_error_message(
        [_Mismatch("g1"), _Mismatch("g2"), _Mismatch("g3"), _Mismatch("g4")]
    )

    assert "count=4" in message
    assert "g1, g2, g3" in message
    assert "repair_postseason_season_ids.py" in message


def test_load_failed_cache_keys_filters_failed_entries(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        """
        {
          "details": [
            {"cache_key": "a", "status": "failed"},
            {"cache_key": "b", "status": "generated"},
            {"cache_key": "c", "status": "FAILED"},
            {"cache_key": "", "status": "failed"}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    assert load_failed_cache_keys(str(report_path)) == {"a", "c"}


def test_build_cache_verification_result_marks_completed_row_as_cache_hit() -> None:
    target = MatchupTarget(
        cache_key="cache-hit",
        game_id="20250310SSHH0",
        season_id=260,
        season_year=2025,
        game_date="2025-03-10",
        game_type="PRE",
        home_team_id="SS",
        away_team_id="HH",
        league_type_code=1,
        stage_label="PRE",
        series_game_no=None,
        game_status_bucket="COMPLETED",
        starter_signature="s",
        lineup_signature="l",
        request_focus=["matchup"],
        request_mode="manual_detail",
        question_override=None,
    )

    result = _build_cache_verification_result(
        target,
        (
            "COMPLETED",
            {
                "meta": {
                    "validation_status": "fallback",
                    "generation_mode": "evidence_fallback",
                }
            },
            None,
        ),
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "cache_hit"
    assert result["meta"]["cached"] is True
    assert result["meta"]["generation_mode"] == "evidence_fallback"


def test_collect_cache_verification_results_marks_missing_rows_failed(
    monkeypatch,
) -> None:
    target = MatchupTarget(
        cache_key="missing",
        game_id="20250310SSHH0",
        season_id=260,
        season_year=2025,
        game_date="2025-03-10",
        game_type="PRE",
        home_team_id="SS",
        away_team_id="HH",
        league_type_code=1,
        stage_label="PRE",
        series_game_no=None,
        game_status_bucket="COMPLETED",
        starter_signature="s",
        lineup_signature="l",
        request_focus=["matchup"],
        request_mode="manual_detail",
        question_override=None,
    )

    monkeypatch.setattr(batch_module, "_fetch_cache_rows", lambda keys: {})

    results = collect_cache_verification_results([target])

    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert results[0]["reason"] == "missing_cache_row"


def test_build_terminal_cache_result_if_available_uses_completed_row(
    monkeypatch,
) -> None:
    target = MatchupTarget(
        cache_key="completed",
        game_id="20250310SSHH0",
        season_id=260,
        season_year=2025,
        game_date="2025-03-10",
        game_type="PRE",
        home_team_id="SS",
        away_team_id="HH",
        league_type_code=1,
        stage_label="PRE",
        series_game_no=None,
        game_status_bucket="COMPLETED",
        starter_signature="s",
        lineup_signature="l",
        request_focus=["matchup"],
        request_mode="manual_detail",
        question_override=None,
    )

    monkeypatch.setattr(
        batch_module,
        "_fetch_cache_state",
        lambda cache_key: (
            "COMPLETED",
            {"meta": {"validation_status": "success"}},
            None,
        ),
    )

    result = _build_terminal_cache_result_if_available(target)

    assert result is not None
    assert result["status"] == "skipped"
    assert result["reason"] == "cache_hit"


def test_call_analyze_with_deadline_returns_failed_on_wall_timeout(
    monkeypatch,
) -> None:
    target = MatchupTarget(
        cache_key="timeout",
        game_id="20250310SSHH0",
        season_id=260,
        season_year=2025,
        game_date="2025-03-10",
        game_type="PRE",
        home_team_id="SS",
        away_team_id="HH",
        league_type_code=1,
        stage_label="PRE",
        series_game_no=None,
        game_status_bucket="COMPLETED",
        starter_signature="s",
        lineup_signature="l",
        request_focus=["matchup"],
        request_mode="manual_detail",
        question_override=None,
    )

    async def _stuck_call(**kwargs):
        await asyncio.sleep(0.05)
        return {"status": "generated", "reason": "generated", "meta": {}}

    monkeypatch.setattr(batch_module, "call_analyze", _stuck_call)

    result = asyncio.run(
        call_analyze_with_deadline(
            client=None,
            base_url="http://127.0.0.1:18080/api/ai",
            target=target,
            timeout_seconds=0.01,
        )
    )

    assert result["status"] == "failed"
    assert result["reason"] == "target_wall_timeout"
    assert result["meta"]["timeout_seconds"] == 0.01


def test_call_analyze_with_retries_succeeds_after_retry(monkeypatch) -> None:
    target = MatchupTarget(
        cache_key="retry-success",
        game_id="20250310SSHH0",
        season_id=260,
        season_year=2025,
        game_date="2025-03-10",
        game_type="PRE",
        home_team_id="SS",
        away_team_id="HH",
        league_type_code=1,
        stage_label="PRE",
        series_game_no=None,
        game_status_bucket="COMPLETED",
        starter_signature="s",
        lineup_signature="l",
        request_focus=["matchup"],
        request_mode="manual_detail",
        question_override=None,
    )
    calls = {"count": 0}

    async def _fake_call(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "failed", "reason": "ReadTimeout", "meta": {}}
        return {"status": "generated", "reason": "generated", "meta": {}}

    async def _fast_sleep(seconds):
        return None

    monkeypatch.setattr(batch_module, "call_analyze_with_deadline", _fake_call)
    monkeypatch.setattr(
        batch_module, "_build_terminal_cache_result_if_available", lambda target: None
    )
    monkeypatch.setattr(batch_module.asyncio, "sleep", _fast_sleep)

    result = asyncio.run(
        call_analyze_with_retries(
            client=None,
            base_url="http://127.0.0.1:8001/ai",
            target=target,
            timeout_seconds=10.0,
            max_attempts=3,
            retry_backoff_seconds=0.1,
        )
    )

    assert calls["count"] == 2
    assert result["status"] == "generated"
    assert result["meta"]["attempt"] == 2


def test_call_analyze_with_retries_clears_locked_cache_before_retry(
    monkeypatch,
) -> None:
    target = MatchupTarget(
        cache_key="retry-locked",
        game_id="20250310SSHH0",
        season_id=260,
        season_year=2025,
        game_date="2025-03-10",
        game_type="PRE",
        home_team_id="SS",
        away_team_id="HH",
        league_type_code=1,
        stage_label="PRE",
        series_game_no=None,
        game_status_bucket="COMPLETED",
        starter_signature="s",
        lineup_signature="l",
        request_focus=["matchup"],
        request_mode="manual_detail",
        question_override=None,
    )
    calls = {"count": 0, "deleted": 0}

    async def _fake_call(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "failed", "reason": "failed_locked", "meta": {}}
        return {"status": "generated", "reason": "generated", "meta": {}}

    async def _fast_sleep(seconds):
        return None

    monkeypatch.setattr(batch_module, "call_analyze_with_deadline", _fake_call)
    monkeypatch.setattr(
        batch_module, "_build_terminal_cache_result_if_available", lambda target: None
    )
    monkeypatch.setattr(batch_module, "force_rebuild_delete", lambda keys: 1)
    monkeypatch.setattr(batch_module.asyncio, "sleep", _fast_sleep)

    result = asyncio.run(
        call_analyze_with_retries(
            client=None,
            base_url="http://127.0.0.1:8001/ai",
            target=target,
            timeout_seconds=10.0,
            max_attempts=2,
            retry_backoff_seconds=0.1,
        )
    )

    assert calls["count"] == 2
    assert result["status"] == "generated"
    assert result["meta"]["attempt"] == 2
