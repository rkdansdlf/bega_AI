import asyncio
from datetime import datetime, timedelta, timezone

from scripts import batch_coach_matchup_cache as batch_module
from scripts.batch_coach_matchup_cache import (
    _classify_meta_result,
    _collect_retryable_replay_targets,
    _extract_error_message,
    _filter_targets_by_cache_state,
    _league_codes_for_type,
    _normalize_cache_state_label,
    _build_analyze_payload,
    _build_cache_verification_result,
    _is_retryable_failure_reason,
    _is_retryable_failure_result,
    _normalize_recovered_pending_result,
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
    summarize_matchup_results,
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
    assert payload["starter_signature"] == "s"
    assert payload["lineup_signature"] == "l"
    assert payload["expected_cache_key"] == "post"


def test_normalize_cache_state_label_maps_missing_and_pending_rows() -> None:
    assert _normalize_cache_state_label(None) == "MISSING"
    assert (
        _normalize_cache_state_label(("completed", {}, None, None, 0, None, None, None))
        == "COMPLETED"
    )
    assert (
        _normalize_cache_state_label(("failed", None, "x", None, 0, None, None, None))
        == "FAILED"
    )
    assert (
        _normalize_cache_state_label(
            ("failed_locked", None, "x", None, 0, None, None, None)
        )
        == "PENDING"
    )


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
            "completed": ("COMPLETED", {"_meta": {}}, None, None, 1, None, None, None),
            "pending": ("FAILED_LOCKED", None, None, None, 1, None, None, None),
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
                "_meta": {
                    "validation_status": "fallback",
                    "generation_mode": "evidence_fallback",
                }
            },
            None,
            None,
            1,
            None,
            None,
            None,
        ),
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "cache_hit"
    assert result["meta"]["cached"] is True
    assert result["meta"]["generation_mode"] == "evidence_fallback"


def test_build_cache_verification_result_marks_retryable_failed_row() -> None:
    target = MatchupTarget(
        cache_key="retryable-failed",
        game_id="20250310SSHH1",
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
        ("FAILED", None, "empty_response", "empty_response", 1, None, None, None),
    )

    assert result["status"] == "failed"
    assert result["reason"] == "empty_response"
    assert result["meta"]["retryable_failure"] is True
    assert result["meta"]["failure_class"] == "retryable_failed"


def test_build_cache_verification_result_marks_pending_wait_row() -> None:
    target = MatchupTarget(
        cache_key="pending-row",
        game_id="20250310SSHH2",
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
        ("PENDING", None, None, None, 2, None, None, None),
    )

    assert result["status"] == "in_progress"
    assert result["reason"] == "pending_wait"
    assert (
        result["meta"]["recheck_after_seconds"] == batch_module.PENDING_RECHECK_SECONDS
    )


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
    assert results[0]["meta"]["cache_row_missing"] is True


def test_collect_cache_verification_results_uses_resolved_cache_key_from_previous_result(
    monkeypatch,
) -> None:
    target = MatchupTarget(
        cache_key="expected-key",
        game_id="20250708SSNC0",
        season_id=260,
        season_year=2025,
        game_date="2025-07-08",
        game_type="REGULAR",
        home_team_id="SS",
        away_team_id="NC",
        league_type_code=0,
        stage_label="REGULAR",
        series_game_no=None,
        game_status_bucket="COMPLETED",
        starter_signature="starter:same",
        lineup_signature="lineup:same",
        request_focus=["matchup"],
        request_mode="manual_detail",
        question_override=None,
    )

    monkeypatch.setattr(
        batch_module,
        "_fetch_cache_rows",
        lambda keys: {
            "resolved-key": batch_module.CacheLookupResult(
                cache_key="resolved-key",
                row=(
                    "COMPLETED",
                    {
                        "_meta": {
                            "validation_status": "success",
                            "cache_key": "resolved-key",
                        }
                    },
                    None,
                    None,
                    1,
                    None,
                    None,
                    None,
                ),
            )
        },
    )

    results = collect_cache_verification_results(
        [target],
        previous_results={
            "expected-key": {
                "cache_key": "expected-key",
                "meta": {"resolved_cache_key": "resolved-key"},
            }
        },
    )

    assert results[0]["status"] == "skipped"
    assert results[0]["reason"] == "cache_hit"
    assert results[0]["meta"]["resolved_cache_key"] == "resolved-key"
    assert results[0]["meta"]["expected_cache_key"] == "expected-key"
    assert results[0]["meta"]["cache_key_mismatch"] is True


def test_build_cache_verification_result_marks_db_unavailable_retryable() -> None:
    target = MatchupTarget(
        cache_key="db-down",
        game_id="20250708OBLT0",
        season_id=260,
        season_year=2025,
        game_date="2025-07-08",
        game_type="REGULAR",
        home_team_id="OB",
        away_team_id="LT",
        league_type_code=0,
        stage_label="REGULAR",
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
        batch_module.CacheLookupResult(
            cache_key="db-down",
            row=None,
            unavailable=True,
            error_message="server closed the connection unexpectedly",
        ),
    )

    assert result["status"] == "failed"
    assert result["reason"] == "db_unavailable"
    assert result["meta"]["retryable_failure"] is True
    assert result["meta"]["db_unavailable"] is True


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
            {"_meta": {"validation_status": "success"}},
            None,
            None,
            1,
            None,
            None,
            None,
        ),
    )

    result = _build_terminal_cache_result_if_available(target)

    assert result is not None
    assert result["status"] == "skipped"
    assert result["reason"] == "cache_hit"


def test_is_retryable_failure_result_uses_failure_class() -> None:
    assert (
        _is_retryable_failure_result(
            {
                "status": "failed",
                "reason": "empty_response",
                "meta": {"failure_class": "retryable_failed"},
            }
        )
        is True
    )


def test_normalize_recovered_pending_result_promotes_cache_hit() -> None:
    item = _normalize_recovered_pending_result(
        {
            "status": "skipped",
            "reason": "cache_hit",
            "meta": {"generation_mode": "evidence_fallback"},
            "cache_key": "k",
        }
    )

    assert item["status"] == "generated"
    assert item["reason"] == "generated_fallback"
    assert item["meta"]["recovered_from"] == "pending_wait"


def test_normalize_recovered_pending_result_keeps_in_progress_unrecovered() -> None:
    item = _normalize_recovered_pending_result(
        {
            "status": "in_progress",
            "reason": "pending_wait",
            "meta": {},
            "cache_key": "k2",
        }
    )

    assert item["status"] == "in_progress"
    assert "recovered_from" not in item["meta"]


def test_normalize_recovered_pending_result_promotes_missing_row_to_retryable() -> None:
    item = _normalize_recovered_pending_result(
        {
            "status": "failed",
            "reason": "missing_cache_row",
            "meta": {},
            "cache_key": "k3",
        }
    )

    assert item["status"] == "failed"
    assert item["reason"] == "missing_cache_row_after_pending"
    assert item["meta"]["recovered_from"] == "pending_wait"
    assert item["meta"]["failure_class"] == "retryable_failed"
    assert item["meta"]["retryable_failure"] is True
    assert item["meta"]["pending_to_missing"] is True


def test_normalize_recovered_pending_result_promotes_db_unavailable_to_retryable() -> (
    None
):
    item = _normalize_recovered_pending_result(
        {
            "status": "failed",
            "reason": "db_unavailable",
            "meta": {},
            "cache_key": "kdb",
        }
    )

    assert item["reason"] == "db_unavailable_after_pending"
    assert item["meta"]["retryable_failure"] is True
    assert item["meta"]["pending_to_db_unavailable"] is True


def test_summarize_matchup_results_uses_resolved_count_for_quality_rates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        batch_module,
        "collect_matchup_integrity_metrics",
        lambda years, aliases: (0, 0),
    )

    summary = summarize_matchup_results(
        results=[
            {
                "cache_key": "g1",
                "status": "generated",
                "reason": "generated",
                "home_team_id": "LG",
                "away_team_id": "KIA",
                "meta": {
                    "generation_mode": "llm_manual",
                    "focus_section_missing": True,
                },
            },
            {
                "cache_key": "g2",
                "status": "generated",
                "reason": "generated_fallback",
                "home_team_id": "KT",
                "away_team_id": "LT",
                "meta": {
                    "generation_mode": "evidence_fallback",
                    "focus_section_missing": True,
                },
            },
            {
                "cache_key": "g3",
                "status": "skipped",
                "reason": "cache_hit",
                "home_team_id": "SS",
                "away_team_id": "DB",
                "meta": {
                    "generation_mode": "evidence_fallback",
                    "focus_section_missing": True,
                },
            },
            {
                "cache_key": "g4",
                "status": "skipped",
                "reason": "cache_hit",
                "home_team_id": "HH",
                "away_team_id": "NC",
                "meta": {
                    "generation_mode": "llm_manual",
                    "focus_section_missing": False,
                },
            },
        ],
        years=[2025],
        league_type="REGULAR",
        focus=["matchup", "recent_form"],
    )

    assert summary["success"] == 2
    assert summary["skipped"] == 2
    assert summary["llm_manual_count"] == 2
    assert summary["evidence_fallback_count"] == 2
    assert summary["llm_manual_rate"] == 0.5
    assert summary["fallback_rate"] == 0.5
    assert summary["focus_section_missing_count"] == 3
    assert summary["focus_section_missing_rate"] == 0.75


def test_collect_retryable_replay_targets_includes_pending_to_missing() -> None:
    target = MatchupTarget(
        cache_key="k3",
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
    item = _normalize_recovered_pending_result(
        {
            "status": "failed",
            "reason": "missing_cache_row",
            "meta": {},
            "cache_key": "k3",
        }
    )

    replay_targets = _collect_retryable_replay_targets([target], {"k3": item})

    assert [candidate.cache_key for candidate in replay_targets] == ["k3"]


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


def test_call_analyze_with_deadline_returns_in_progress_when_pending_row_exists(
    monkeypatch,
) -> None:
    target = MatchupTarget(
        cache_key="timeout-pending",
        game_id="20250310SSHH1",
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
    monkeypatch.setattr(
        batch_module, "_build_terminal_cache_result_if_available", lambda target: None
    )
    monkeypatch.setattr(
        batch_module,
        "_build_in_progress_cache_result_if_available",
        lambda target: {
            "cache_key": target.cache_key,
            "game_id": target.game_id,
            "home_team_id": target.home_team_id,
            "away_team_id": target.away_team_id,
            "year": target.season_year,
            "game_type": target.game_type,
            "status": "in_progress",
            "reason": "pending_wait",
            "meta": {"cache_state": "PENDING", "in_progress": True},
        },
    )

    result = asyncio.run(
        call_analyze_with_deadline(
            client=None,
            base_url="http://127.0.0.1:18080/api/ai",
            target=target,
            timeout_seconds=0.01,
        )
    )

    assert result["status"] == "in_progress"
    assert result["reason"] == "pending_wait"
    assert result["meta"]["timeout_seconds"] == 0.01
    assert result["meta"]["timed_out_while_pending"] is True
    assert result["meta"]["recovered_from"] == "target_wall_timeout"


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
        batch_module,
        "_build_terminal_cache_result_if_available",
        lambda *args, **kwargs: None,
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
        batch_module,
        "_build_terminal_cache_result_if_available",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        batch_module,
        "force_rebuild_delete",
        lambda keys: {
            "deleted_cache_rows": 1,
            "unsafe_force_rebuild_blocked_count": 0,
            "stale_pending_takeover_count": 0,
            "missing_cache_row_count": 0,
        },
    )
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


def test_build_cache_verification_result_keeps_stream_cancelled_retryable() -> None:
    target = MatchupTarget(
        cache_key="retry-stream-cancelled",
        game_id="20250310SSHH2",
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

    item = _build_cache_verification_result(
        target,
        (
            "FAILED",
            None,
            "client_stream_cancelled",
            "stream_cancelled",
            7,
            datetime.now(timezone.utc),
            None,
            None,
        ),
    )

    assert item["status"] == "failed"
    assert item["meta"]["retryable_failure"] is True
    assert item["meta"]["failure_class"] == "retryable_failed"


def test_force_rebuild_delete_blocks_active_pending_but_deletes_terminal(
    monkeypatch,
) -> None:
    fresh_time = datetime.now(timezone.utc) - timedelta(seconds=5)

    class _FakeResult:
        def __init__(self, rows=None, rowcount=0):
            self._rows = rows or []
            self.rowcount = rowcount

        def fetchall(self):
            return list(self._rows)

    class _FakeConn:
        def __init__(self):
            self.deleted_keys = None

        def execute(self, sql, params):
            if "SELECT cache_key, status" in sql:
                return _FakeResult(
                    rows=[
                        (
                            "pending-key",
                            "PENDING",
                            fresh_time,
                            fresh_time + timedelta(seconds=90),
                            fresh_time,
                        ),
                        ("completed-key", "COMPLETED", fresh_time, None, None),
                    ]
                )
            if sql.startswith("DELETE FROM coach_analysis_cache"):
                self.deleted_keys = params[0]
                return _FakeResult(rowcount=1)
            raise AssertionError(sql)

        def commit(self):
            return None

    class _Ctx:
        def __init__(self, conn):
            self._conn = conn

        def __enter__(self):
            return self._conn

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakePool:
        def __init__(self, conn):
            self._conn = conn

        def connection(self):
            return _Ctx(self._conn)

    conn = _FakeConn()
    monkeypatch.setattr(batch_module, "get_connection_pool", lambda: _FakePool(conn))

    stats = batch_module.force_rebuild_delete(
        ["pending-key", "completed-key", "missing-key"]
    )

    assert stats["deleted_cache_rows"] == 1
    assert stats["unsafe_force_rebuild_blocked_count"] == 1
    assert stats["missing_cache_row_count"] == 1
    assert conn.deleted_keys == ["completed-key"]
