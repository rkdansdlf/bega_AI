from scripts.coach_backfill_audit import (
    BackfillCapture,
    _dedupe_messages,
    _manual_baseball_data_required_rows,
    _starter_announcement_pending_rows,
    assess_quality,
    collect_missing_data,
    collect_response_notes,
    dedupe_official_game_ids,
    effective_request_interval_seconds,
    focus_for_record,
    is_retryable_capture,
    summarize_results,
    resolve_default_internal_api_key,
    select_records,
    starter_announcement_due_at_kst,
)


def _record(**overrides):
    base = {
        "game_id": "20260322LGSS0",
        "game_date": "2026-03-22",
        "game_status_bucket": "COMPLETED",
        "home_team_id": "SS",
        "away_team_id": "LG",
        "stage_label": "REGULAR",
        "expected_data_quality": "grounded",
        "root_causes": [],
        "lineup_announced": True,
        "home_pitcher_present": True,
        "away_pitcher_present": True,
    }
    base.update(overrides)
    return base


def _structured(markdown=None):
    return {
        "headline": "LG 트윈스 승리, 데이터 기반 리뷰",
        "coach_note": "경기 흐름을 실데이터 중심으로 요약했습니다.",
        "key_metrics": [{"label": "득점", "value": "5-3"}],
        "analysis": {"swing_factors": ["초반 득점"]},
        "detailed_markdown": markdown
        or "\n".join(
            [
                "## 최근 전력",
                "최근 경기 흐름은 확인 가능한 표본 기준으로 해석합니다.",
                "## 불펜 상태",
                "후반 운영은 실점 흐름과 등판 기록 중심으로 봅니다.",
                "## 선발 투수",
                "선발 맞대결은 확인된 선발 정보 기준으로 평가합니다.",
                "## 상대 전적",
                "맞대결 표본은 제한적으로 반영합니다.",
                "## 타격 생산성",
                "타격 생산성은 확인 가능한 지표만 사용합니다.",
            ]
        ),
    }


def _capture(**overrides):
    base = {
        "status_code": 200,
        "elapsed_seconds": 0.1,
        "response_headers": {},
        "done_seen": True,
        "event_sequence": ["message", "meta", "done"],
        "message_text": "",
        "meta": {
            "data_quality": "grounded",
            "generation_mode": "evidence_fallback",
            "cache_state": "COMPLETED",
            "cached": False,
            "game_status_bucket": "COMPLETED",
            "focus_section_missing": False,
            "missing_focus_sections": [],
            "grounding_reasons": [],
        },
        "structured_response": _structured(),
        "error_payload": None,
        "sample_response": "",
    }
    base.update(overrides)
    return BackfillCapture(**base)


def test_focus_for_record_uses_completed_review_focuses():
    assert focus_for_record(_record()) == [
        "recent_form",
        "bullpen",
        "starter",
        "matchup",
        "batting",
    ]


def test_focus_for_record_uses_scheduled_prediction_focuses():
    assert focus_for_record(_record(game_status_bucket="SCHEDULED")) == [
        "recent_form",
        "bullpen",
        "starter",
    ]


def test_select_records_filters_orders_and_offsets():
    records = [
        _record(game_id="g1", game_date="2026-03-01"),
        _record(game_id="g2", game_date="2026-03-02", game_status_bucket="SCHEDULED"),
        _record(game_id="g3", game_date="2026-03-03"),
    ]

    selected = select_records(
        records,
        status_bucket="COMPLETED",
        offset=1,
        limit=1,
        order="asc",
    )

    assert [item["game_id"] for item in selected] == ["g3"]


def test_dedupe_official_game_ids_prefers_kbo_game_id_alias():
    records = [
        _record(
            game_id="20260403LGKH0",
            game_date="2026-04-03",
            home_team_id="KH",
            away_team_id="LG",
        ),
        _record(
            game_id="20260403LGWO0",
            game_date="2026-04-03",
            home_team_id="KH",
            away_team_id="LG",
        ),
        _record(
            game_id="20260403OBSSG0",
            game_date="2026-04-03",
            home_team_id="SSG",
            away_team_id="DB",
        ),
        _record(
            game_id="20260403OBSK0",
            game_date="2026-04-03",
            home_team_id="SSG",
            away_team_id="DB",
        ),
    ]

    assert [item["game_id"] for item in dedupe_official_game_ids(records)] == [
        "20260403LGWO0",
        "20260403OBSK0",
    ]


def test_collect_missing_data_keeps_only_actual_missing_data_reasons():
    rows = collect_missing_data(
        _record(root_causes=["missing_lineups", "missing_summary"]),
        {
            "grounding_reasons": [
                "missing_lineups",
                "focus_data_unavailable",
                "empty_response",
            ]
        },
    )

    assert rows == [
        {
            "source": "diagnosis",
            "code": "missing_lineups",
            "label": "라인업 정보 부족",
        },
        {
            "source": "diagnosis",
            "code": "missing_summary",
            "label": "경기 요약 정보 부족",
        },
        {
            "source": "response",
            "code": "missing_lineups",
            "label": "라인업 정보 부족",
        },
    ]


def test_collect_response_notes_separates_non_missing_grounding_reasons():
    rows = collect_response_notes(
        {
            "grounding_reasons": [
                "missing_lineups",
                "focus_data_unavailable",
                "empty_response",
                "focus_data_unavailable",
            ]
        }
    )

    assert rows == [
        {
            "source": "response",
            "code": "focus_data_unavailable",
            "label": "요청 focus 직접 근거 제한",
        },
        {
            "source": "response",
            "code": "empty_response",
            "label": "LLM 빈 응답 후 보수 생성",
        },
    ]


def test_manual_baseball_data_required_rows_surface_missing_starters_after_announcement(
    monkeypatch,
):
    monkeypatch.setenv("COACH_AUDIT_NOW_KST", "2026-04-23T19:00:00+09:00")
    record = _record(
        game_status_bucket="SCHEDULED",
        game_id="20260424KTSK0",
        game_date="2026-04-24",
        home_team_id="SSG",
        away_team_id="KT",
        root_causes=["missing_starters"],
        home_pitcher_present=False,
        away_pitcher_present=False,
    )
    rows = _manual_baseball_data_required_rows(
        [
            {
                "target": {
                    "game_id": record["game_id"],
                    "game_date": record["game_date"],
                    "game_status_bucket": record["game_status_bucket"],
                    "home_team_id": record["home_team_id"],
                    "away_team_id": record["away_team_id"],
                },
                "diagnosis": record,
                "missing_data": [
                    {
                        "source": "diagnosis",
                        "code": "missing_starters",
                        "label": "선발 투수 정보 부족",
                    }
                ],
            }
        ]
    )

    assert rows == [
        {
            "game_id": "20260424KTSK0",
            "game_date": "2026-04-24",
            "status_bucket": "SCHEDULED",
            "away_team_id": "KT",
            "home_team_id": "SSG",
            "contract_code": "MANUAL_BASEBALL_DATA_REQUIRED",
            "missing_code": "missing_starters",
            "required_fields": "game.home_pitcher|game.away_pitcher",
            "home_pitcher": "",
            "away_pitcher": "",
            "operator_message": (
                "다음 야구 데이터가 필요합니다: 선발 투수 정보 "
                "(game.home_pitcher, game.away_pitcher)"
            ),
        }
    ]


def test_manual_baseball_data_required_rows_include_response_contract_items():
    record = _record(
        game_status_bucket="SCHEDULED",
        game_id="20260423HHLG0",
        game_date="2026-04-23",
        home_team_id="LG",
        away_team_id="HH",
        root_causes=[],
    )
    rows = _manual_baseball_data_required_rows(
        [
            {
                "target": {
                    "game_id": record["game_id"],
                    "game_date": record["game_date"],
                    "game_status_bucket": record["game_status_bucket"],
                    "home_team_id": record["home_team_id"],
                    "away_team_id": record["away_team_id"],
                },
                "diagnosis": record,
                "missing_data": [],
                "response": {
                    "meta": {
                        "manual_data_request": {
                            "code": "MANUAL_BASEBALL_DATA_REQUIRED",
                            "missingItems": [
                                {
                                    "key": "game_status",
                                    "label": "경기 상태",
                                    "expected_format": "SCHEDULED, COMPLETED",
                                },
                                {
                                    "key": "final_score",
                                    "label": "최종 점수",
                                    "expected_format": "home_score, away_score",
                                },
                            ],
                            "operatorMessage": "경기 상태와 최종 점수가 필요합니다.",
                        }
                    }
                },
                "cache_probe_response": {
                    "meta": {
                        "manual_data_request": {
                            "code": "MANUAL_BASEBALL_DATA_REQUIRED",
                            "missingItems": [
                                {"key": "game_status"},
                                {"key": "final_score"},
                            ],
                            "operatorMessage": "경기 상태와 최종 점수가 필요합니다.",
                        }
                    }
                },
            }
        ]
    )

    assert rows == [
        {
            "game_id": "20260423HHLG0",
            "game_date": "2026-04-23",
            "status_bucket": "SCHEDULED",
            "away_team_id": "HH",
            "home_team_id": "LG",
            "contract_code": "MANUAL_BASEBALL_DATA_REQUIRED",
            "missing_code": "game_status|final_score",
            "required_fields": "game.game_status|game.home_score|game.away_score",
            "home_pitcher": "",
            "away_pitcher": "",
            "operator_message": "경기 상태와 최종 점수가 필요합니다.",
        }
    ]


def test_missing_starters_before_official_announcement_are_pending(monkeypatch):
    monkeypatch.setenv("COACH_AUDIT_NOW_KST", "2026-04-23T17:00:00+09:00")
    record = _record(
        game_status_bucket="SCHEDULED",
        game_id="20260424KTSK0",
        game_date="2026-04-24",
        home_team_id="SSG",
        away_team_id="KT",
        root_causes=["missing_starters"],
        home_pitcher_present=False,
        away_pitcher_present=False,
    )
    result = {
        "target": {
            "game_id": record["game_id"],
            "game_date": record["game_date"],
            "game_status_bucket": record["game_status_bucket"],
            "home_team_id": record["home_team_id"],
            "away_team_id": record["away_team_id"],
        },
        "diagnosis": record,
        "missing_data": collect_missing_data(record, None),
    }

    assert result["missing_data"] == [
        {
            "source": "diagnosis",
            "code": "starter_announcement_pending",
            "label": "공식 선발 발표 대기",
        }
    ]
    assert _manual_baseball_data_required_rows([result]) == []
    pending_rows = _starter_announcement_pending_rows([result])
    assert pending_rows[0]["status_code"] == "starter_announcement_pending"
    assert pending_rows[0]["expected_announcement_at_kst"] == ("2026-04-23T18:00+09:00")


def test_explicit_starter_announcement_pending_rows_are_reported(monkeypatch):
    monkeypatch.setenv("COACH_AUDIT_NOW_KST", "2026-04-27T15:00:00+09:00")
    record = _record(
        game_status_bucket="SCHEDULED",
        game_id="20260429LGKT0",
        game_date="2026-04-29",
        home_team_id="KT",
        away_team_id="LG",
        root_causes=["starter_announcement_pending"],
        home_pitcher_present=False,
        away_pitcher_present=False,
    )
    result = {
        "target": {
            "game_id": record["game_id"],
            "game_date": record["game_date"],
            "game_status_bucket": record["game_status_bucket"],
            "home_team_id": record["home_team_id"],
            "away_team_id": record["away_team_id"],
        },
        "diagnosis": record,
        "missing_data": collect_missing_data(record, None),
    }

    assert result["missing_data"] == [
        {
            "source": "diagnosis",
            "code": "starter_announcement_pending",
            "label": "공식 선발 발표 대기",
        }
    ]
    assert _starter_announcement_pending_rows([result])[0]["game_id"] == (
        "20260429LGKT0"
    )


def test_starter_announcement_due_at_kst_uses_previous_day_18():
    assert (
        starter_announcement_due_at_kst("2026-04-24").isoformat(timespec="minutes")
        == "2026-04-23T18:00+09:00"
    )


def test_assess_quality_rejects_banned_template_phrase():
    structured = _structured(
        markdown="\n".join(
            [
                "## 최근 전력",
                "A가 근소 우세지만 격차는 크지 않습니다.",
                "## 불펜 상태",
                "후반 운영은 보수적으로 봅니다.",
                "## 선발 투수",
                "선발 정보 기준으로 봅니다.",
                "## 상대 전적",
                "상대 전적 표본을 제한적으로 반영합니다.",
                "## 타격 생산성",
                "타격 지표만 사용합니다.",
            ]
        )
    )
    hard, soft = assess_quality(
        _record(),
        _capture(structured_response=structured),
    )

    assert any("금지 문구 포함" in item for item in hard)
    assert soft == []


def test_assess_quality_rejects_scheduled_result_language():
    record = _record(
        game_status_bucket="SCHEDULED",
        expected_data_quality="partial",
        root_causes=["missing_lineups"],
        lineup_announced=False,
    )
    structured = _structured(
        markdown="\n".join(
            [
                "## 최근 전력",
                "예정 경기 흐름을 보수적으로 봅니다.",
                "## 불펜 상태",
                "후반 운영 데이터가 제한적입니다.",
                "## 선발 투수",
                "선발 정보가 확정되지 않았습니다.",
                "결승타 가능성을 단정합니다.",
            ]
        )
    )
    capture = _capture(
        meta={
            "data_quality": "partial",
            "generation_mode": "evidence_fallback",
            "cache_state": "COMPLETED",
            "cached": False,
            "game_status_bucket": "SCHEDULED",
            "focus_section_missing": False,
            "missing_focus_sections": [],
            "grounding_reasons": ["missing_lineups"],
        },
        structured_response=structured,
    )

    hard, _soft = assess_quality(record, capture)

    assert any("예정 경기 결과 확정 표현 포함" in item for item in hard)


def test_assess_quality_rejects_scheduled_wpa_and_language_jargon():
    record = _record(game_status_bucket="SCHEDULED", expected_data_quality="grounded")
    structured = _structured(
        markdown="\n".join(
            [
                "## 최근 전력",
                "예정 경기 흐름을 보수적으로 봅니다.",
                "## 불펜 상태",
                "WPA/PA와 오프ensive 흐름을 봅니다.",
                "## 선발 투수",
                "선발 정보 기준으로 봅니다.",
            ]
        )
    )
    capture = _capture(
        meta={**_capture().meta, "game_status_bucket": "SCHEDULED"},
        structured_response=structured,
    )

    hard, _soft = assess_quality(record, capture)

    assert any("예정 경기 사용자 문구에 지표/언어 잔여 포함" in item for item in hard)


def test_assess_quality_rejects_scheduled_review_headers_and_starter_conflict():
    record = _record(
        game_status_bucket="SCHEDULED",
        expected_data_quality="grounded",
        home_pitcher_present=True,
        away_pitcher_present=True,
    )
    structured = _structured(
        markdown="\n".join(
            [
                "## 최근 전력",
                "예정 경기 흐름을 보수적으로 봅니다.",
                "## 불펜 상태",
                "후반 운영 데이터가 제한적입니다.",
                "## 선발 투수",
                "선발 정보가 확정되지 않아 선발 맞대결 평가는 제한됩니다.",
                "## 결과 진단",
                "SSG 랜더스의 우위 확정.",
            ]
        )
    )
    capture = _capture(
        meta={**_capture().meta, "game_status_bucket": "SCHEDULED"},
        structured_response=structured,
    )

    hard, _soft = assess_quality(record, capture)

    assert any("예정 경기 결과 확정 표현 포함" in item for item in hard)
    assert any("예정 경기 선발 확정 상태와 충돌" in item for item in hard)


def test_assess_quality_requires_cache_hit_for_probe():
    hard, _soft = assess_quality(
        _record(),
        _capture(meta={**_capture().meta, "cache_state": "COMPLETED"}),
        cache_hit_probe=True,
    )

    assert any("재호출 cache_state가 HIT가 아님" in item for item in hard)


def test_summarize_results_separates_transport_failures_from_content_failures():
    record = _record()
    result = {
        "target": {
            "game_id": record["game_id"],
            "game_date": record["game_date"],
            "game_status_bucket": record["game_status_bucket"],
        },
        "diagnosis": record,
        "response": {
            "status_code": 0,
            "meta": {},
        },
        "cache_probe_response": None,
        "missing_data": [],
        "response_notes": [],
        "quality": {
            "ok": False,
            "hard_failures": ["HTTP 실패 status=0"],
            "soft_warnings": [],
        },
    }

    summary = summarize_results([record], [result])

    assert summary["failed_targets"] == 1
    assert summary["hard_failure_count"] == 1
    assert summary["transport_failed_targets"] == 1
    assert summary["transport_failure_count"] == 1
    assert summary["content_failed_targets"] == 0
    assert summary["content_hard_failure_count"] == 0
    assert summary["content_passed_targets"] == 1


def test_summarize_results_treats_rate_limit_as_transport_failure():
    record = _record()
    result = {
        "target": {
            "game_id": record["game_id"],
            "game_date": record["game_date"],
            "game_status_bucket": record["game_status_bucket"],
        },
        "diagnosis": record,
        "response": {
            "status_code": 429,
            "meta": {},
        },
        "cache_probe_response": None,
        "missing_data": [],
        "response_notes": [],
        "quality": {
            "ok": False,
            "hard_failures": ["HTTP 실패 status=429"],
            "soft_warnings": [],
        },
    }

    summary = summarize_results([record], [result])

    assert summary["transport_failed_targets"] == 1
    assert summary["transport_failure_count"] == 1
    assert summary["content_failed_targets"] == 0
    assert summary["content_hard_failure_count"] == 0


def test_assess_quality_treats_retryable_sse_error_as_transport_failure():
    capture = _capture(
        event_sequence=["status", "meta", "error", "done"],
        error_payload={
            "code": "oci_mapping_degraded",
            "message": "팀 매핑 조회가 일시적으로 불안정했습니다.",
        },
        meta={
            **_capture().meta,
            "cache_state": "FAILED_RETRY_WAIT",
            "cache_error_code": "oci_mapping_degraded",
            "retryable_failure": True,
        },
        structured_response=None,
    )

    hard, _soft = assess_quality(_record(game_status_bucket="SCHEDULED"), capture)

    assert is_retryable_capture(capture) is True
    assert len(hard) == 1
    assert hard[0].startswith("SSE transport error 이벤트 감지:")


def test_summarize_results_treats_retryable_sse_error_as_transport_failure():
    record = _record()
    result = {
        "target": {
            "game_id": record["game_id"],
            "game_date": record["game_date"],
            "game_status_bucket": record["game_status_bucket"],
        },
        "diagnosis": record,
        "response": {
            "status_code": 200,
            "meta": {"cache_state": "FAILED_RETRY_WAIT"},
        },
        "cache_probe_response": None,
        "missing_data": [],
        "response_notes": [],
        "quality": {
            "ok": False,
            "hard_failures": [
                "SSE transport error 이벤트 감지: {'code': 'oci_mapping_degraded'}"
            ],
            "soft_warnings": [],
        },
    }

    summary = summarize_results([record], [result])

    assert summary["transport_failed_targets"] == 1
    assert summary["transport_failure_count"] == 1
    assert summary["content_failed_targets"] == 0
    assert summary["content_hard_failure_count"] == 0


def test_assess_quality_rejects_status_bucket_mismatch():
    hard, _soft = assess_quality(
        _record(game_status_bucket="COMPLETED"),
        _capture(meta={**_capture().meta, "game_status_bucket": "UNKNOWN"}),
    )

    assert any("응답 경기 상태 불일치" in item for item in hard)


def test_dedupe_messages_preserves_order():
    assert _dedupe_messages(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_effective_request_interval_clamps_cache_hit_verification():
    assert effective_request_interval_seconds(0.2, verify_cache_hit=True) == 3.0


def test_effective_request_interval_keeps_non_probe_interval():
    assert effective_request_interval_seconds(0.2, verify_cache_hit=False) == 0.2


def test_effective_request_interval_keeps_higher_probe_interval():
    assert effective_request_interval_seconds(4.0, verify_cache_hit=True) == 4.0


def test_resolve_default_internal_api_key_prefers_service_local_env(
    monkeypatch, tmp_path
):
    from scripts import coach_backfill_audit

    service_root = tmp_path / "bega_AI"
    service_root.mkdir()
    (service_root / ".env").write_text(
        "AI_INTERNAL_TOKEN=service-token\n", encoding="utf-8"
    )
    (tmp_path / ".env.prod").write_text(
        "AI_INTERNAL_TOKEN=root-token\n", encoding="utf-8"
    )
    monkeypatch.delenv("AI_INTERNAL_TOKEN", raising=False)
    monkeypatch.setattr(coach_backfill_audit, "PROJECT_ROOT", service_root)

    assert resolve_default_internal_api_key() == "service-token"


def test_grounding_disclaimer_accepts_negative_confirmation_phrase():
    from app.core.coach_grounding import has_grounding_disclaimer

    assert has_grounding_disclaimer("선발 정보가 확정되지 않아 평가는 제한됩니다.")
