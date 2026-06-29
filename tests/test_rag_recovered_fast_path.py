from __future__ import annotations

import asyncio

from datetime import date
from types import SimpleNamespace
from typing import Any, Mapping

from app.core.rag import (
    RAGPipeline,
    _build_future_event_pending_result,
    _build_static_kbo_faq_result,
)


def _entity_filter(**overrides):
    values = {"stat_leader": None}
    values.update(overrides)
    return SimpleNamespace(**values)


class _OperatorCursor:
    def __init__(self, rows: list[Mapping[str, Any]]) -> None:
        self.rows = rows

    async def __aenter__(self) -> "_OperatorCursor":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        del query, params

    async def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self.rows)


class _OperatorConn:
    def __init__(self, rows: list[Mapping[str, Any]]) -> None:
        self.rows = rows

    def cursor(self, *args, **kwargs) -> _OperatorCursor:
        del args, kwargs
        return _OperatorCursor(self.rows)


def _operator_pipeline(enabled: bool, rows: list[Mapping[str, Any]]) -> RAGPipeline:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = SimpleNamespace(operator_data_fast_path_enabled=enabled)
    pipeline.connection = _OperatorConn(rows)
    pipeline._pool = None
    return pipeline


def test_recovered_leaderboard_questions_force_agent_fast_path() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)

    assert pipeline._should_force_agent_fast_path(
        "2026년 홈런왕은 누구야?", _entity_filter()
    )
    assert pipeline._should_force_agent_fast_path(
        "2026년 최다안타 1위는 누구야?", _entity_filter()
    )
    assert pipeline._should_force_agent_fast_path(
        "타율 1위는 누구야?",
        _entity_filter(stat_leader={"stat_name": "avg", "position": "batting"}),
    )


def test_recovered_lg_questions_force_agent_fast_path() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)

    assert pipeline._should_force_agent_fast_path(
        "LG 트윈스 2026 시즌 현재 순위와 승률 흐름 알려줘.", _entity_filter()
    )
    assert pipeline._should_force_agent_fast_path(
        "LG 상대 전적 보면 누구만 만나면 유독 꼬이는지 알려줘.", _entity_filter()
    )
    assert pipeline._should_force_agent_fast_path(
        "LG 수비 실책 때문에 날린 경기들 정리해줘.", _entity_filter()
    )
    assert pipeline._should_force_agent_fast_path(
        "2026년 김도영를 같은 포지션 선수와 비교하면 어때?", _entity_filter()
    )


def test_balanced_team_analysis_questions_force_agent_fast_path() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    team_filter = _entity_filter(team_id="LG")

    fast_path_questions = [
        "LG 트윈스 타선이 터지는 패턴과 막히는 패턴을 비교해줘.",
        "LG 트윈스 최근 10경기 득점 흐름을 팬 눈높이로 정리해줘.",
        "LG 트윈스 홈 경기와 원정 경기 흐름 차이가 있는지 정리해줘.",
        "LG 트윈스 가을야구 가능성을 현재 데이터 기준으로 봐줘.",
        "2026년 LG 트윈스 팀 타율과 타선 흐름 알려줘.",
        "2026년 LG 트윈스 팀 평균자책점과 마운드 강점 알려줘.",
        "2026년 LG 트윈스 정규시즌 순위와 승패 흐름 정리해줘.",
        "2026년 LG 트윈스 홈런 생산력은 리그에서 어느 정도였어?",
        "2026년 LG 트윈스 도루와 주루 지표는 강점이었어?",
        "2026년 LG 트윈스 선발진과 불펜 중 어디가 더 안정적이었어?",
        "2026년 LG 트윈스 경기에서 선발투수가 무너진 대표 경기를 찾아줘.",
    ]

    for question in fast_path_questions:
        assert pipeline._should_force_agent_fast_path(question, team_filter), question


def test_player_lookup_questions_force_agent_fast_path() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)

    assert pipeline._should_force_agent_fast_path(
        "2026년 양의지 시즌 성적 핵심만 알려줘.",
        _entity_filter(player_name="양의지"),
    )
    assert pipeline._should_force_agent_fast_path(
        "2026년 구자욱는 타자야 투수야? 주요 기록도 같이 알려줘.",
        _entity_filter(player_name="구자욱"),
    )


def test_false_player_extraction_does_not_force_strict_single_query() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = SimpleNamespace(retrieval_single_query_for_strict_entity=True)

    assert not pipeline._should_use_single_query_retrieval(
        query="KBO 리그는 어떤 리그야?",
        search_strategy={"is_ranking_query": False},
        entity_filter=_entity_filter(player_name="리그야"),
        final_filters={},
    )
    assert pipeline._should_use_single_query_retrieval(
        query="2026년 양의지 시즌 성적 핵심만 알려줘.",
        search_strategy={"is_ranking_query": False},
        entity_filter=_entity_filter(player_name="양의지"),
        final_filters={"season_year": 2026},
    )


def test_regular_rag_question_does_not_force_agent_fast_path() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)

    assert not pipeline._should_force_agent_fast_path(
        "KBO 비디오 판독 규정 알려줘.", _entity_filter()
    )
    assert not pipeline._should_force_agent_fast_path(
        "KBO 리그는 어떤 리그야?", _entity_filter(player_name="리그야")
    )


def test_static_kbo_faq_answers_generic_league_question_without_retrieval() -> None:
    result = _build_static_kbo_faq_result("KBO 리그는 어떤 리그야?")

    assert result is not None
    assert result["planner_mode"] == "fast_path"
    assert result["strategy"] == "static_kbo_faq"
    assert "한국의 최상위 프로야구 리그" in result["answer"]


def test_static_kbo_faq_answers_win_rate_and_tie_rules() -> None:
    ranking = _build_static_kbo_faq_result("KBO 는 어떻게 순위를 정해")
    win_rate = _build_static_kbo_faq_result("KBO 승률은 어떻게 계산해?")
    tie = _build_static_kbo_faq_result("KBO 무승부는 순위에 어떻게 반영돼?")

    assert ranking is not None
    assert "승률을 먼저 봅니다" in ranking["answer"]
    assert win_rate is not None
    assert "승수 / (승수 + 패수)" in win_rate["answer"]
    assert tie is not None
    assert "무승부는 승률 계산의 분모에서 제외" in tie["answer"]


def test_live_kbo_question_surfaces_manual_data_contract() -> None:
    result = _build_static_kbo_faq_result("오늘 KBO 경기 일정 알려줘.")
    opening_day = _build_static_kbo_faq_result("2026 KBO 개막일은 언제야?")
    vague_score = _build_static_kbo_faq_result("경기 스코어를 알려줘.")
    rotation = _build_static_kbo_faq_result("팀별 선발 로테이션은 어떻게 돼?")
    game_save = _build_static_kbo_faq_result("세이브는 누가 올렸어?")
    unsupported_leader = _build_static_kbo_faq_result("포수 도루저지율 1위는 누구야?")
    pitch_type = _build_static_kbo_faq_result("구종별 성적은 어디서 봐?")
    ticket = _build_static_kbo_faq_result("야구장 티켓은 어디서 예매해?")
    parking = _build_static_kbo_faq_result("야구장 주차는 어떻게 해?")
    chant = _build_static_kbo_faq_result("팀별 응원가는 뭐야?")
    late_game = _build_static_kbo_faq_result("오늘 경기 후반 집중력이 더 좋은 팀은?")
    runner_count = _build_static_kbo_faq_result("주자는 몇 명이야?")
    next_pitcher = _build_static_kbo_faq_result("다음 투수는 누구야?")
    foreign_news = _build_static_kbo_faq_result("외국인 선수 교체 소식 있어?")
    best_player = _build_static_kbo_faq_result("오늘 최고의 선수는 누구야?")
    my_team = _build_static_kbo_faq_result("오늘 우리 팀 이겨?")
    ace = _build_static_kbo_faq_result("이 팀의 에이스는 누구야?")
    team_compare = _build_static_kbo_faq_result("두 팀 전력 비교해줘.")
    named_team_compare = _build_static_kbo_faq_result("LG와 KT를 비교해줘?")
    spaced_team_compare = _build_static_kbo_faq_result("LG 와 KT 를 비교해줘")
    injury_list = _build_static_kbo_faq_result("부상자 명단은 어디서 봐?")
    most_titles = _build_static_kbo_faq_result("KBO 우승이 가장 많은 팀은?")
    leadership = _build_static_kbo_faq_result(
        "양현종, 김광현, 최형우를 팀 상징성까지 가진 베테랑으로 놓으면 숫자 밖 리더십을 어떻게 읽어야 해?"
    )

    assert result is not None
    assert result["strategy"] == "manual_baseball_data_required"
    assert result["fallback_reason"] == "manual_baseball_data_required"
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in result["answer"]
    assert opening_day is not None
    assert opening_day["strategy"] == "manual_baseball_data_required"
    assert vague_score is not None
    assert vague_score["strategy"] == "manual_baseball_data_required"
    assert rotation is not None
    assert rotation["strategy"] == "manual_baseball_data_required"
    assert game_save is not None
    assert game_save["strategy"] == "manual_baseball_data_required"
    assert unsupported_leader is not None
    assert unsupported_leader["strategy"] == "manual_baseball_data_required"
    assert pitch_type is not None
    assert pitch_type["strategy"] == "manual_baseball_data_required"
    assert ticket is not None
    assert ticket["strategy"] == "manual_baseball_data_required"
    assert parking is not None
    assert parking["strategy"] == "manual_baseball_data_required"
    assert chant is not None
    assert chant["strategy"] == "manual_baseball_data_required"
    assert late_game is not None
    assert late_game["strategy"] == "manual_baseball_data_required"
    assert runner_count is not None
    assert runner_count["strategy"] == "manual_baseball_data_required"
    assert next_pitcher is not None
    assert next_pitcher["strategy"] == "manual_baseball_data_required"
    assert foreign_news is not None
    assert foreign_news["strategy"] == "manual_baseball_data_required"
    assert best_player is not None
    assert best_player["strategy"] == "manual_baseball_data_required"
    assert my_team is not None
    assert my_team["strategy"] == "clarification_required"
    assert ace is not None
    assert ace["strategy"] == "clarification_required"
    assert team_compare is not None
    assert team_compare["strategy"] == "manual_baseball_data_required"
    assert named_team_compare is None
    assert spaced_team_compare is None
    assert injury_list is not None
    assert injury_list["strategy"] == "manual_baseball_data_required"
    assert most_titles is not None
    assert most_titles["strategy"] == "manual_baseball_data_required"
    assert leadership is not None
    assert leadership["strategy"] == "manual_baseball_data_required"


def test_operator_data_fast_path_flag_off_keeps_manual_contract() -> None:
    pipeline = _operator_pipeline(
        False,
        [
            {
                "queue_id": "ODQ-0001",
                "game_date": "2026-06-05",
                "game_id": "20260605LGKT0",
                "home_team": "LG",
                "away_team": "KT",
                "stadium_name": "잠실",
                "start_time": "18:30",
                "game_status": "scheduled",
                "source_checked_at": "2026-06-05",
                "confidence": 0.95,
            }
        ],
    )

    result = asyncio.run(
        pipeline._build_operator_or_static_kbo_result("오늘 KBO 경기 일정 알려줘.")
    )

    assert result is not None
    assert result["strategy"] == "manual_baseball_data_required"
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in result["answer"]


def test_operator_data_fast_path_flag_on_returns_operator_answer() -> None:
    pipeline = _operator_pipeline(
        True,
        [
            {
                "queue_id": "ODQ-0001",
                "game_date": "2026-06-05",
                "game_id": "20260605LGKT0",
                "home_team": "LG",
                "away_team": "KT",
                "stadium_name": "잠실",
                "start_time": "18:30",
                "game_status": "scheduled",
                "source_checked_at": "2026-06-05",
                "confidence": 0.95,
            }
        ],
    )

    result = asyncio.run(
        pipeline._build_operator_or_static_kbo_result("오늘 KBO 경기 일정 알려줘.")
    )

    assert result is not None
    assert result["strategy"] == "operator_data_fast_path"
    assert result["source_tier"] == "operator_data"
    assert result["operator_data_partial"] is True
    assert "확인된 항목만" in result["answer"]


def test_operator_data_fast_path_without_rows_keeps_manual_contract() -> None:
    pipeline = _operator_pipeline(True, [])

    result = asyncio.run(
        pipeline._build_operator_or_static_kbo_result("오늘 KBO 경기 일정 알려줘.")
    )

    assert result is not None
    assert result["strategy"] == "manual_baseball_data_required"
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in result["answer"]


def test_operator_data_fast_path_underspecified_rows_keep_manual_contract() -> None:
    pipeline = _operator_pipeline(
        True,
        [
            {
                "queue_id": "ODQ-0030",
                "game_id": "20260605LGKT0",
                "team_code": "LG",
                "player_name": "홍길동",
                "position": "CF",
                "batting_order": 1,
                "notes": {
                    "source_type": "manual_lineup",
                    "queue_id": "ODQ-0030",
                    "is_verified": True,
                    "confidence": 0.9,
                },
                "game_date": "2026-06-05",
                "home_team": "LG",
                "away_team": "KT",
                "season_year": 2026,
                "roster_event_type": "부상",
                "effective_date": "2026-06-04",
                "status_text": "엔트리 제외",
                "source_checked_at": "2026-06-05",
                "confidence": 0.9,
            }
        ],
    )

    lineup = asyncio.run(
        pipeline._build_operator_or_static_kbo_result("LG 라인업 알려줘.")
    )
    roster = asyncio.run(
        pipeline._build_operator_or_static_kbo_result("LG 부상자 명단은 어디서 봐?")
    )

    assert lineup is not None
    assert lineup["strategy"] == "manual_baseball_data_required"
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in lineup["answer"]
    assert roster is not None
    assert roster["strategy"] == "manual_baseball_data_required"
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in roster["answer"]


def test_operator_data_fast_path_malformed_lineup_notes_keep_manual_contract() -> None:
    pipeline = _operator_pipeline(
        True,
        [
            {
                "queue_id": "ODQ-0030",
                "game_id": "20260605LGKT0",
                "team_code": "LG",
                "player_name": "홍길동",
                "position": "P",
                "batting_order": 1,
                "notes": "manual_lineup malformed metadata",
                "game_date": "2026-06-05",
                "home_team": "LG",
                "away_team": "KT",
            }
        ],
    )

    result = asyncio.run(
        pipeline._build_operator_or_static_kbo_result("2026-06-05 LG 라인업 알려줘.")
    )

    assert result is not None
    assert result["strategy"] == "manual_baseball_data_required"
    assert "MANUAL_BASEBALL_DATA_REQUIRED" in result["answer"]


def test_db_answerable_remaining_failures_do_not_trigger_manual_gate() -> None:
    assert _build_static_kbo_faq_result("2026년 득점 1위는 누구야?") is None
    assert _build_static_kbo_faq_result("2026년 팀별 홈런 수 알려줘.") is None
    assert _build_static_kbo_faq_result("2026년 최근 10경기 성적 알려줘.") is None
    assert _build_static_kbo_faq_result("2026년 홈 승률과 원정 승률 알려줘.") is None
    assert _build_static_kbo_faq_result("LG와 KT를 비교해줘?") is None
    assert _build_static_kbo_faq_result("SSG와 KIA를 비교해줘?") is None
    assert _build_static_kbo_faq_result("도루가 많은 팀은 어디야?") is None
    assert _build_static_kbo_faq_result("실책이 적은 팀은 어디야?") is None
    assert _build_static_kbo_faq_result("팀별 ERA는?") is None
    assert _build_static_kbo_faq_result("팀별 경기당 평균 득점은?") is None
    assert _build_static_kbo_faq_result("팀별 경기당 평균 실점은?") is None
    assert _build_static_kbo_faq_result("최근 흐름 비교해줘.") is None
    assert _build_static_kbo_faq_result("타선 비교해줘.") is None
    assert _build_static_kbo_faq_result("수비 비교해줘.") is None
    assert _build_static_kbo_faq_result("주루 비교해줘.") is None
    assert _build_static_kbo_faq_result("가장 기복이 큰 팀은?") is None
    assert _build_static_kbo_faq_result("불펜 비교해줘.") is None
    assert _build_static_kbo_faq_result("선발진 비교해줘.") is None


def test_db_fast_path_candidates_with_insufficient_db_basis_stay_manual() -> None:
    player_fielding = _build_static_kbo_faq_result("실책이 적은 선수는 누구야?")
    stolen_base_season = _build_static_kbo_faq_result("도루가 많은 시즌이야?")
    home_run_season = _build_static_kbo_faq_result("홈런이 많은 시즌이야?")

    assert player_fielding is not None
    assert player_fielding["strategy"] == "manual_baseball_data_required"
    assert stolen_base_season is not None
    assert stolen_base_season["strategy"] == "manual_baseball_data_required"
    assert home_run_season is not None
    assert home_run_season["strategy"] == "manual_baseball_data_required"


def test_ambiguous_static_questions_return_clarification_not_manual() -> None:
    result = _build_static_kbo_faq_result("이 팀 최근 10경기 성적 알려줘.")

    assert result is not None
    assert result["strategy"] == "clarification_required"
    assert result["fallback_reason"] == "clarification_required"
    assert "팀명을 같이 알려주세요" in result["answer"]
    assert "MANUAL_BASEBALL_DATA_REQUIRED" not in result["answer"]


def test_pending_season_events_are_not_manual_data_failures() -> None:
    all_star = _build_future_event_pending_result(
        "2026 KBO 올스타전은 언제야?",
        today=date(2026, 5, 31),
    )
    all_star_mvp = _build_future_event_pending_result(
        "2026 올스타전 MVP는 누구야?",
        today=date(2026, 5, 31),
    )
    postseason = _build_future_event_pending_result(
        "2026 한국시리즈 우승팀은 어디야?",
        today=date(2026, 5, 31),
    )
    postseason_rule = _build_static_kbo_faq_result("KBO 포스트시즌은 어떻게 진행돼?")
    korean_series_rule = _build_static_kbo_faq_result("KBO 한국시리즈는 몇 경기야?")
    game_mvp = _build_future_event_pending_result(
        "경기 MVP는 누구야?",
        today=date(2026, 5, 31),
    )
    past_all_star = _build_future_event_pending_result(
        "2025 KBO 올스타전 MVP는 누구야?",
        today=date(2026, 5, 31),
    )
    metric_term = _build_future_event_pending_result(
        "WAR 5면 올스타급이야?",
        today=date(2026, 5, 31),
    )

    assert all_star is not None
    assert all_star["strategy"] == "future_event_pending"
    assert all_star["fallback_reason"] is None
    assert "MANUAL_BASEBALL_DATA_REQUIRED" not in all_star["answer"]
    assert "아직 진행 전" in all_star["answer"]
    assert all_star_mvp is not None
    assert all_star_mvp["strategy"] == "future_event_pending"
    assert postseason is not None
    assert postseason["strategy"] == "future_event_pending"
    assert postseason_rule is not None
    assert postseason_rule["strategy"] == "static_kbo_faq"
    assert "토너먼트" in postseason_rule["answer"]
    assert korean_series_rule is not None
    assert korean_series_rule["strategy"] == "static_kbo_faq"
    assert "7전 4선승제" in korean_series_rule["answer"]
    assert game_mvp is None
    assert past_all_star is None
    assert metric_term is None


def test_db_answerable_schedule_and_rank_queries_bypass_manual_contract() -> None:
    dated_schedule = _build_static_kbo_faq_result(
        "2026년 4월 30일 KBO 경기 일정 알려줘."
    )
    range_schedule = _build_static_kbo_faq_result(
        "2026년 4월 28일부터 4월 30일까지 KBO 경기표 보여줘."
    )
    broadcast = _build_static_kbo_faq_result("2026년 4월 30일 KBO 중계는 어디서 봐?")
    first_place = _build_static_kbo_faq_result("1위 팀은 어디야?")
    last_place = _build_static_kbo_faq_result("최하위 팀은 어디야?")
    standings = _build_static_kbo_faq_result("팀별 승패는 어떻게 돼?")
    standings_by_date = _build_static_kbo_faq_result(
        "2026년 4월 30일 기준 KBO 순위는 어떻게 돼?"
    )

    assert dated_schedule is None
    assert range_schedule is None
    assert first_place is None
    assert last_place is None
    assert standings is None
    assert standings_by_date is None
    assert broadcast is not None
    assert broadcast["strategy"] == "manual_baseball_data_required"


def test_static_chatbot_meta_questions_do_not_use_retrieval() -> None:
    feature = _build_static_kbo_faq_result("KBO 챗봇 기능을 추천해줘?")
    schedule_capability = _build_static_kbo_faq_result("챗봇이 일정만 알려줄 수 있어?")
    notification = _build_static_kbo_faq_result("경기 알림 설정은 할 수 있어?")
    generated = _build_static_kbo_faq_result("KBO 관련 질문을 10개 더 만들어줘?")
    faq = _build_static_kbo_faq_result("KBO 초보자 FAQ를 만들어줘?")
    data_source = _build_static_kbo_faq_result("KBO 선수 데이터는 어디서 가져와?")
    api_design = _build_static_kbo_faq_result("KBO API를 어떻게 설계해?")
    phrase = _build_static_kbo_faq_result("승리 축하 메시지 만들어줘.")
    home_run_phrase = _build_static_kbo_faq_result("홈런 나왔을 때 쓸 문구는?")
    rain_tip = _build_static_kbo_faq_result("비 오는 날 직관 팁 알려줘.")
    cheering_manners = _build_static_kbo_faq_result("야구장 응원 예절은 뭐야?")
    guidance_example = _build_static_kbo_faq_result("직관 안내 예시를 만들어줘?")

    assert feature is not None
    assert feature["strategy"] == "static_chatbot_meta"
    assert schedule_capability is not None
    assert schedule_capability["strategy"] == "static_chatbot_meta"
    assert notification is not None
    assert notification["strategy"] == "static_chatbot_meta"
    assert generated is not None
    assert generated["strategy"] == "static_chatbot_meta"
    assert faq is not None
    assert faq["strategy"] == "static_chatbot_meta"
    assert data_source is not None
    assert data_source["strategy"] == "static_chatbot_meta"
    assert api_design is not None
    assert api_design["strategy"] == "static_chatbot_meta"
    assert phrase is not None
    assert phrase["strategy"] == "static_chatbot_meta"
    assert home_run_phrase is not None
    assert home_run_phrase["strategy"] == "static_chatbot_meta"
    assert rain_tip is not None
    assert rain_tip["strategy"] == "static_chatbot_meta"
    assert cheering_manners is not None
    assert cheering_manners["strategy"] == "static_chatbot_meta"
    assert guidance_example is not None
    assert guidance_example["strategy"] == "static_chatbot_meta"


def test_static_baseball_explainer_terms_do_not_use_retrieval() -> None:
    result = _build_static_kbo_faq_result("필승조는 무엇이야?")
    rules = _build_static_kbo_faq_result("야구 규칙을 쉽게 알려줘.")
    defense = _build_static_kbo_faq_result("수비율은 어떻게 봐?")
    replay = _build_static_kbo_faq_result("2026 KBO 비디오판독은 어떻게 돼?")
    roster = _build_static_kbo_faq_result("2026 KBO 등록 인원은 몇 명이야?")
    road_trip = _build_static_kbo_faq_result("원정 연전은 얼마나 힘들어?")
    obstruction = _build_static_kbo_faq_result("체크 상황 주루 방해는 어떻게 처리돼?")
    shift_penalty = _build_static_kbo_faq_result("수비 시프트 위반 시 페널티는 뭐야?")
    stolen_base_success = _build_static_kbo_faq_result("도루 성공률은 어떻게 봐?")

    assert result is not None
    assert result["intent"] == "baseball_explainer"
    assert result["grounding_mode"] == "static_baseball_explainer"
    assert "핵심 불펜" in result["answer"]
    assert rules is not None
    assert "베이스를 돌아 득점" in rules["answer"]
    assert defense is not None
    assert "수비 기회" in defense["answer"]
    assert stolen_base_success is not None
    assert stolen_base_success["strategy"] == "static_baseball_explainer"
    assert "도루 / (도루 + 도루 실패)" in stolen_base_success["answer"]
    assert replay is not None
    assert "영상으로 다시 확인" in replay["answer"]
    assert roster is not None
    assert roster["strategy"] == "manual_baseball_data_required"
    assert road_trip is not None
    assert road_trip["intent"] == "baseball_explainer"
    assert obstruction is not None
    assert "정상적인 주루" in obstruction["answer"]
    assert shift_penalty is not None
    assert shift_penalty["strategy"] == "manual_baseball_data_required"


def test_static_team_profile_questions_do_not_use_retrieval() -> None:
    team_profile = _build_static_kbo_faq_result("LG 트윈스는 어떤 팀이야?")
    stadiums = _build_static_kbo_faq_result("각 팀의 홈구장은 어디야?")
    managers = _build_static_kbo_faq_result("각 팀의 감독은 누구야?")
    current_rank = _build_static_kbo_faq_result("현재 KBO 순위는 어떻게 돼?")
    colors = _build_static_kbo_faq_result("각 팀의 팀 컬러는 뭐야?")

    assert team_profile is not None
    assert team_profile["intent"] == "team_profile"
    assert "LG 트윈스는 KBO 리그의 10개 구단" in team_profile["answer"]
    assert stadiums is not None
    assert "잠실야구장" in stadiums["answer"]
    assert managers is not None
    assert managers["strategy"] == "manual_baseball_data_required"
    assert current_rank is None
    assert colors is not None
    assert colors["intent"] == "team_profile"
