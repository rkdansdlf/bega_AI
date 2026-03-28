from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..core.entity_extractor import extract_entities_from_query, extract_team
from .tool_caller import ToolCall


class ChatIntent(str, Enum):
    TEAM_ANALYSIS = "team_analysis"
    SCHEDULE_LOOKUP = "schedule_lookup"
    PLAYER_LOOKUP = "player_lookup"
    LEADERBOARD_LOOKUP = "leaderboard_lookup"
    REGULATION_LOOKUP = "regulation_lookup"
    SEASON_RESULT_LOOKUP = "season_result_lookup"
    SEASON_STANDING_LOOKUP_BY_RANK = "season_standing_lookup_by_rank"
    AMBIGUOUS_SUPERLATIVE = "ambiguous_superlative"
    FOLLOWUP_CONTINUATION = "followup_continuation"
    LOW_DATA_RECOVERY = "low_data_recovery"
    BASEBALL_EXPLAINER = "baseball_explainer"
    LATEST_INFO = "latest_info"
    LONG_TAIL_ENTITY = "long_tail_entity"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentDecision:
    intent: ChatIntent
    planner_mode: str = "llm"
    tool_calls: list[ToolCall] = field(default_factory=list)
    subject_type: Optional[str] = None
    season_year: Optional[int] = None
    metric_policy: Optional[str] = None
    confidence: float = 0.0
    analysis: str = ""
    resolved_intent: Optional[ChatIntent] = None
    followup_source_question: Optional[str] = None
    direct_answer: Optional[str] = None
    grounding_mode: str = "structured_kbo"
    source_tier: str = "db"
    fallback_reason: Optional[str] = None


class ChatIntentRouter:
    def __init__(
        self,
        *,
        resolve_reference_year=None,
        detect_team_alias=None,
        resolve_award_query_type=None,
        build_team_tool_calls=None,
        fast_path_enabled: bool,
        fast_path_scope: str,
    ) -> None:
        self._resolve_reference_year = resolve_reference_year
        self._detect_team_alias = detect_team_alias
        self._resolve_award_query_type = resolve_award_query_type
        self._build_team_tool_calls = build_team_tool_calls
        self._fast_path_enabled = fast_path_enabled
        self._fast_path_scope = fast_path_scope

    def bind(self, agent: Any) -> "_BoundChatIntentRouter":
        return _BoundChatIntentRouter(router=self, agent=agent)

    def resolve(
        self,
        query: str,
        entity_filter: Any,
        context: Optional[dict[str, Any]] = None,
        *,
        agent: Any = None,
    ) -> IntentDecision:
        resolve_reference_year = (
            agent._resolve_reference_year
            if agent is not None
            else self._resolve_reference_year
        )
        detect_team_alias = (
            agent._detect_team_alias_from_query
            if agent is not None
            else self._detect_team_alias
        )
        resolve_award_query_type = (
            agent._resolve_award_query_type
            if agent is not None
            else self._resolve_award_query_type
        )
        build_team_tool_calls = (
            agent._build_team_fast_path_tool_calls
            if agent is not None
            else self._build_team_tool_calls
        )

        query_lower = query.lower()
        season_year = resolve_reference_year(query, entity_filter)
        if self._is_vague_followup_query(query_lower):
            previous_query = self._find_previous_user_question(context, query)
            if previous_query:
                previous_entity_filter = extract_entities_from_query(previous_query)
                delegated = self.resolve(
                    previous_query,
                    previous_entity_filter,
                    None,
                    agent=agent,
                )
                delegated_intent = delegated.resolved_intent or delegated.intent
                if delegated_intent != ChatIntent.UNKNOWN:
                    return IntentDecision(
                        intent=ChatIntent.FOLLOWUP_CONTINUATION,
                        planner_mode=delegated.planner_mode,
                        tool_calls=delegated.tool_calls,
                        subject_type=delegated.subject_type,
                        season_year=delegated.season_year,
                        metric_policy=delegated.metric_policy,
                        confidence=delegated.confidence,
                        analysis=f"후속 발화로 판단되어 직전 질문 '{previous_query}'의 의도를 이어받습니다.",
                        resolved_intent=delegated_intent,
                        followup_source_question=previous_query,
                        direct_answer=delegated.direct_answer,
                        grounding_mode=delegated.grounding_mode,
                        source_tier=delegated.source_tier,
                        fallback_reason=delegated.fallback_reason,
                    )
            return IntentDecision(
                intent=ChatIntent.FOLLOWUP_CONTINUATION,
                planner_mode="fast_path",
                tool_calls=[],
                subject_type="clarification",
                season_year=season_year,
                metric_policy="clarification_only",
                confidence=0.95,
                analysis="맥락 없는 후속 발화로 판단되어 clarification 응답을 반환합니다.",
                direct_answer="어떤 질문을 이어서 답하면 되는지 한 번만 더 적어주세요.",
                grounding_mode="unsupported",
                source_tier="none",
                fallback_reason="missing_followup_context",
            )

        team_name = (
            getattr(entity_filter, "team_id", None)
            or extract_team(query)
            or detect_team_alias(query)
        )
        player_name = (
            getattr(entity_filter, "player_name", None)
            or getattr(entity_filter, "person_name", None)
            or getattr(entity_filter, "name", None)
        )
        game_id_match = re.search(r"\b\d{8}[A-Z]{4}\d\b", query)
        extracted_date = self._extract_date(query, query_lower)
        explicit_metric = self._resolve_leaderboard_spec(query_lower)

        comparison_tool_calls = self._build_player_comparison_tool_calls(
            query=query,
            query_lower=query_lower,
            season_year=season_year,
        )
        if comparison_tool_calls:
            comparison_type = str(
                comparison_tool_calls[0].parameters.get("comparison_type") or "season"
            )
            metric_policy = (
                "player_comparison_career"
                if comparison_type == "career"
                else "player_comparison_season"
            )
            return IntentDecision(
                intent=ChatIntent.PLAYER_LOOKUP,
                planner_mode="fast_path",
                tool_calls=comparison_tool_calls,
                subject_type="player_comparison",
                season_year=season_year,
                metric_policy=metric_policy,
                confidence=0.97,
                analysis="두 선수 비교 질의로 판단되어 compare_players fast-path를 사용합니다.",
            )

        schedule_bundle_tool_calls = self._build_schedule_bundle_tool_calls(
            query_lower=query_lower,
            extracted_date=extracted_date,
            game_id=game_id_match.group(0) if game_id_match else None,
            team_name=team_name,
        )
        if schedule_bundle_tool_calls:
            return IntentDecision(
                intent=ChatIntent.SCHEDULE_LOOKUP,
                planner_mode="fast_path_bundle",
                tool_calls=schedule_bundle_tool_calls,
                subject_type="schedule_bundle",
                season_year=season_year,
                metric_policy="schedule_lineup_box_score_bundle",
                confidence=0.97,
                analysis="일정/라인업/박스스코어 복합 질의로 판단되어 deterministic bundle fast-path를 사용합니다.",
            )

        team_bundle_tool_calls = self._build_team_analysis_bundle_tool_calls(
            query_lower=query_lower,
            team_name=team_name,
            season_year=season_year,
        )
        if team_bundle_tool_calls:
            return IntentDecision(
                intent=ChatIntent.TEAM_ANALYSIS,
                planner_mode="fast_path_bundle",
                tool_calls=team_bundle_tool_calls,
                subject_type="team_bundle",
                season_year=season_year,
                metric_policy="team_recent_bullpen_bundle",
                confidence=0.97,
                analysis=f"{team_name} 팀의 최근 흐름/불펜 복합 질의로 판단되어 deterministic bundle fast-path를 사용합니다.",
            )

        player_bundle_tool_calls = self._build_player_validation_bundle_tool_calls(
            query_lower=query_lower,
            player_name=player_name,
            season_year=season_year,
            explicit_metric=explicit_metric,
        )
        if player_bundle_tool_calls:
            position = self._resolve_player_lookup_position(query_lower)
            return IntentDecision(
                intent=ChatIntent.PLAYER_LOOKUP,
                planner_mode="fast_path_bundle",
                tool_calls=player_bundle_tool_calls,
                subject_type="player_bundle",
                season_year=season_year,
                metric_policy=f"player_validation_bundle_{position}",
                confidence=0.96,
                analysis="선수 검증/기록/리더보드가 함께 필요한 복합 질의로 판단되어 deterministic bundle fast-path를 사용합니다.",
            )

        if self._is_game_flow_narrative_query(query_lower) and (
            game_id_match or extracted_date
        ):
            parameters: dict[str, Any] = {}
            if game_id_match:
                parameters["game_id"] = game_id_match.group(0)
            elif extracted_date:
                parameters["date"] = extracted_date
            return IntentDecision(
                intent=ChatIntent.SCHEDULE_LOOKUP,
                planner_mode="fast_path",
                tool_calls=[ToolCall("get_game_box_score", parameters)],
                subject_type="game_flow",
                season_year=season_year,
                metric_policy="game_flow_lookup",
                confidence=0.95,
                analysis="경기 흐름 질문으로 판단되어 박스스코어 기반 game-flow fast-path를 사용합니다.",
            )

        if self._is_team_analysis_query(query_lower, team_name):
            return IntentDecision(
                intent=ChatIntent.TEAM_ANALYSIS,
                planner_mode="fast_path",
                tool_calls=build_team_tool_calls(
                    query,
                    team_name,
                    season_year,
                ),
                subject_type="team",
                season_year=season_year,
                metric_policy="team_fast_path",
                confidence=0.96,
                analysis=f"{team_name} 팀 분석 질문으로 판단되어 fast-path를 사용합니다.",
            )

        if self._is_regulation_query(query_lower):
            return IntentDecision(
                intent=ChatIntent.REGULATION_LOOKUP,
                planner_mode="fast_path",
                tool_calls=[ToolCall("search_regulations", {"query": query})],
                subject_type="regulation",
                season_year=season_year,
                metric_policy="regulation_lookup",
                confidence=0.95,
                analysis="규정성 질문으로 판단되어 규정 검색 fast-path를 사용합니다.",
            )

        award_type = resolve_award_query_type(query, entity_filter)
        if award_type:
            parameters: dict[str, Any] = {"year": season_year}
            if award_type != "any":
                parameters["award_type"] = award_type
            return IntentDecision(
                intent=ChatIntent.SEASON_RESULT_LOOKUP,
                planner_mode="fast_path",
                tool_calls=[ToolCall("get_award_winners", parameters)],
                subject_type="award",
                season_year=season_year,
                metric_policy="award_lookup",
                confidence=0.95,
                analysis="수상 질문으로 판단되어 awards fast-path를 사용합니다.",
            )

        if self._is_runner_up_query(query_lower):
            return IntentDecision(
                intent=ChatIntent.SEASON_RESULT_LOOKUP,
                planner_mode="fast_path",
                tool_calls=[
                    ToolCall("get_korean_series_winner", {"year": season_year})
                ],
                subject_type="runner_up",
                season_year=season_year,
                metric_policy="runner_up_lookup",
                confidence=0.95,
                analysis="준우승 질문으로 판단되어 한국시리즈 결과 fast-path를 사용합니다.",
            )

        if self._is_champion_query(query_lower):
            return IntentDecision(
                intent=ChatIntent.SEASON_RESULT_LOOKUP,
                planner_mode="fast_path",
                tool_calls=[
                    ToolCall("get_korean_series_winner", {"year": season_year})
                ],
                subject_type="champion",
                season_year=season_year,
                metric_policy="champion_lookup",
                confidence=0.95,
                analysis="우승팀 질문으로 판단되어 한국시리즈 우승팀 fast-path를 사용합니다.",
            )

        rank_lookup = self._extract_rank_lookup(query_lower)
        if rank_lookup:
            rank, season_phase = rank_lookup
            return IntentDecision(
                intent=ChatIntent.SEASON_STANDING_LOOKUP_BY_RANK,
                planner_mode="fast_path",
                tool_calls=[
                    ToolCall(
                        "get_team_by_rank",
                        {
                            "year": season_year,
                            "rank": rank,
                            "season_phase": season_phase,
                        },
                    )
                ],
                subject_type="season_rank_team",
                season_year=season_year,
                metric_policy=f"{season_phase}_rank_lookup",
                confidence=0.95,
                analysis=f"{season_year}년 {season_phase} 순위 역질의로 판단되어 순위 기반 팀 조회 fast-path를 사용합니다.",
            )

        if self._is_team_metric_query(query_lower, team_name, player_name):
            return IntentDecision(
                intent=ChatIntent.TEAM_ANALYSIS,
                planner_mode="fast_path",
                tool_calls=build_team_tool_calls(
                    query,
                    team_name,
                    season_year,
                ),
                subject_type="team_metric",
                season_year=season_year,
                metric_policy="team_metric_lookup",
                confidence=0.96,
                analysis=f"{team_name} 팀 지표 질문으로 판단되어 team fast-path를 사용합니다.",
            )

        if explicit_metric and not player_name:
            position, stat_name = explicit_metric
            return IntentDecision(
                intent=ChatIntent.LEADERBOARD_LOOKUP,
                planner_mode="fast_path",
                tool_calls=[
                    ToolCall(
                        "get_leaderboard",
                        {
                            "stat_name": stat_name,
                            "year": season_year,
                            "position": position,
                            "limit": 5,
                        },
                    )
                ],
                subject_type="pitcher" if position == "pitching" else "batter",
                season_year=season_year,
                metric_policy=f"{position}_{stat_name}",
                confidence=0.94,
                analysis="리더보드 질문으로 판단되어 leaderboard fast-path를 사용합니다.",
            )

        if self._is_ambiguous_superlative_query(query_lower):
            subject_type = self._resolve_subject_type(query_lower)
            if subject_type == "pitcher":
                return IntentDecision(
                    intent=ChatIntent.AMBIGUOUS_SUPERLATIVE,
                    planner_mode="fast_path",
                    tool_calls=[
                        ToolCall(
                            "get_leaderboard",
                            {
                                "stat_name": "era",
                                "year": season_year,
                                "position": "pitching",
                                "limit": 5,
                            },
                        )
                    ],
                    subject_type="pitcher",
                    season_year=season_year,
                    metric_policy="pitcher_quick3",
                    confidence=0.93,
                    analysis="모호한 최고 투수 질문으로 판단되어 quick-3 leaderboard fast-path를 사용합니다.",
                )
            if subject_type == "batter":
                return IntentDecision(
                    intent=ChatIntent.AMBIGUOUS_SUPERLATIVE,
                    planner_mode="fast_path",
                    tool_calls=[
                        ToolCall(
                            "get_leaderboard",
                            {
                                "stat_name": "ops",
                                "year": season_year,
                                "position": "batting",
                                "limit": 5,
                            },
                        )
                    ],
                    subject_type="batter",
                    season_year=season_year,
                    metric_policy="batter_quick3",
                    confidence=0.93,
                    analysis="모호한 최고 타자 질문으로 판단되어 quick-3 leaderboard fast-path를 사용합니다.",
                )

        if self._is_box_score_query(query_lower) and (game_id_match or extracted_date):
            parameters: dict[str, Any] = {}
            if game_id_match:
                parameters["game_id"] = game_id_match.group(0)
            elif extracted_date:
                parameters["date"] = extracted_date
            return IntentDecision(
                intent=ChatIntent.SCHEDULE_LOOKUP,
                planner_mode="fast_path",
                tool_calls=[ToolCall("get_game_box_score", parameters)],
                subject_type="game_box_score",
                season_year=season_year,
                metric_policy="box_score_lookup",
                confidence=0.94,
                analysis="박스스코어/이닝 질문으로 판단되어 경기 조회 fast-path를 사용합니다.",
            )

        if extracted_date and not self._is_game_flow_narrative_query(query_lower):
            parameters = {"date": extracted_date}
            if team_name:
                parameters["team"] = team_name
            return IntentDecision(
                intent=ChatIntent.SCHEDULE_LOOKUP,
                planner_mode="fast_path",
                tool_calls=[ToolCall("get_games_by_date", parameters)],
                subject_type="schedule",
                season_year=season_year,
                metric_policy="games_by_date",
                confidence=0.94,
                analysis="날짜/일정 질문으로 판단되어 경기 일정 fast-path를 사용합니다.",
            )

        if team_name and self._is_team_last_game_query(query_lower):
            return IntentDecision(
                intent=ChatIntent.SEASON_RESULT_LOOKUP,
                planner_mode="fast_path",
                tool_calls=[
                    ToolCall(
                        "get_team_last_game",
                        {"team_name": team_name, "year": season_year},
                    )
                ],
                subject_type="team_last_game",
                season_year=season_year,
                metric_policy="team_last_game",
                confidence=0.94,
                analysis="팀 마지막 경기 질문으로 판단되어 마지막 경기 fast-path를 사용합니다.",
            )

        if team_name and self._is_team_rank_query(query_lower):
            return IntentDecision(
                intent=ChatIntent.SEASON_RESULT_LOOKUP,
                planner_mode="fast_path",
                tool_calls=[
                    ToolCall(
                        "get_team_rank", {"team_name": team_name, "year": season_year}
                    )
                ],
                subject_type="team_rank",
                season_year=season_year,
                metric_policy="team_rank",
                confidence=0.94,
                analysis="팀 순위 질문으로 판단되어 팀 순위 fast-path를 사용합니다.",
            )

        if self._is_latest_info_query(query_lower):
            return IntentDecision(
                intent=ChatIntent.LATEST_INFO,
                planner_mode="fast_path",
                tool_calls=[
                    ToolCall("search_latest_baseball", {"query": query, "limit": 5})
                ],
                subject_type="latest_info",
                season_year=season_year,
                metric_policy="latest_web_search",
                confidence=0.9,
                analysis="최신성 질문으로 판단되어 최신 야구 정보 검색 fast-path를 사용합니다.",
                grounding_mode="latest_info",
                source_tier="web",
                fallback_reason="direct_latest_request",
            )

        if self._is_long_tail_entity_query(query_lower):
            return IntentDecision(
                intent=ChatIntent.LONG_TAIL_ENTITY,
                planner_mode="fast_path",
                tool_calls=[ToolCall("search_documents", {"query": query, "limit": 5})],
                subject_type="long_tail_entity",
                season_year=season_year,
                metric_policy="document_long_tail",
                confidence=0.91,
                analysis="구단/구장/팬문화/역사 계열 질문으로 판단되어 문서 검색 fast-path를 사용합니다.",
                grounding_mode="long_tail_entity",
                source_tier="docs",
            )

        if self._is_baseball_explainer_query(query_lower):
            return IntentDecision(
                intent=ChatIntent.BASEBALL_EXPLAINER,
                planner_mode="fast_path",
                tool_calls=[ToolCall("search_documents", {"query": query, "limit": 5})],
                subject_type="baseball_explainer",
                season_year=season_year,
                metric_policy="document_explainer",
                confidence=0.9,
                analysis="설명형 야구 질문으로 판단되어 문서 검색 fast-path를 사용합니다.",
                grounding_mode="baseball_explainer",
                source_tier="docs",
            )

        if player_name and self._is_player_lookup_query(query_lower):
            position = self._resolve_player_lookup_position(query_lower)
            return IntentDecision(
                intent=ChatIntent.PLAYER_LOOKUP,
                planner_mode="fast_path",
                tool_calls=[
                    ToolCall(
                        "get_player_stats",
                        {
                            "player_name": player_name,
                            "year": season_year,
                            "position": position,
                        },
                    )
                ],
                subject_type="player",
                season_year=season_year,
                metric_policy="player_stats",
                confidence=0.93,
                analysis="선수 기록 질문으로 판단되어 player stats fast-path를 사용합니다.",
            )

        if not self._is_baseball_domain_query(query_lower, team_name, player_name):
            return IntentDecision(
                intent=ChatIntent.UNSUPPORTED,
                planner_mode="fast_path",
                tool_calls=[],
                subject_type="unsupported",
                season_year=season_year,
                metric_policy="out_of_domain",
                confidence=0.82,
                analysis="야구 도메인과 무관한 질문으로 판단되어 범위를 안내합니다.",
                direct_answer="야구 관련 질문이라면 KBO 기록, 규정, 구단/선수 이야기, 구장/팬문화, 최신 소식까지 도와드릴 수 있습니다. 질문을 야구 쪽으로 조금만 좁혀주세요.",
                grounding_mode="unsupported",
                source_tier="none",
                fallback_reason="out_of_domain",
            )

        return IntentDecision(
            intent=ChatIntent.UNKNOWN,
            planner_mode="llm",
            tool_calls=[],
            season_year=season_year,
            confidence=0.0,
            analysis="기존 LLM planner 경로를 유지합니다.",
        )

    def _is_latest_info_query(self, query_lower: str) -> bool:
        temporal_tokens = [
            "오늘",
            "지금",
            "현재",
            "실시간",
            "방금",
            "최신",
            "속보",
            "금일",
            "최근",
            "요즘",
        ]
        latest_subject_tokens = [
            "선발",
            "라인업",
            "엔트리",
            "등록",
            "말소",
            "부상",
            "트레이드",
            "루머",
            "이슈",
            "소식",
            "현황",
            "상황",
            "점수",
            "중계",
            "맞대결",
            "활약",
            "경기",
            "일정",
        ]
        return any(token in query_lower for token in temporal_tokens) and any(
            token in query_lower for token in latest_subject_tokens
        )

    def _is_baseball_explainer_query(self, query_lower: str) -> bool:
        explainer_subjects = [
            "abs",
            "war",
            "wrc+",
            "ops",
            "whip",
            "fip",
            "babip",
            "qs",
            "세이버",
            "지표",
            "전술",
            "전략",
            "포지션",
            "수비 시프트",
            "플래툰",
            "번트",
            "히트앤런",
            "체인지업",
            "슬라이더",
            "커브",
            "직구",
            "피치클락",
            "파크팩터",
            "태그업",
            "인필드 플라이",
            "필승조",
            "포수 리드",
            "체크 스윙",
            "야구 상식",
            "야구 용어",
            "용어",
            "구종",
        ]
        explainer_verbs = [
            "뜻",
            "의미",
            "왜",
            "설명",
            "해설",
            "어떻게",
            "기준",
            "차이",
            "뭐야",
        ]
        return any(token in query_lower for token in explainer_subjects) and any(
            token in query_lower for token in explainer_verbs
        )

    def _is_long_tail_entity_query(self, query_lower: str) -> bool:
        return any(
            token in query_lower
            for token in [
                "마스코트",
                "응원가",
                "별명",
                "엠블럼",
                "유니폼",
                "팬 문화",
                "응원 문화",
                "라이벌",
                "홈구장",
                "구장 분위기",
                "구단 역사",
                "전통",
                "사직",
                "잠실",
                "대구",
                "광주",
                "인천",
                "수원",
                "창원",
                "대전",
            ]
        )

    def _is_baseball_domain_query(
        self,
        query_lower: str,
        team_name: Optional[str],
        player_name: Optional[str],
    ) -> bool:
        if team_name or player_name:
            return True
        baseball_tokens = [
            "야구",
            "kbo",
            "프로야구",
            "투수",
            "타자",
            "홈런",
            "타율",
            "방어율",
            "ops",
            "war",
            "wrc+",
            "whip",
            "라인업",
            "선발",
            "불펜",
            "포수",
            "내야수",
            "외야수",
            "스트라이크",
            "볼넷",
            "삼진",
            "구장",
            "응원가",
        ]
        return any(token in query_lower for token in baseball_tokens)

    def _extract_date(self, query: str, query_lower: str) -> Optional[str]:
        if "오늘" in query_lower and any(
            keyword in query_lower
            for keyword in ["경기", "일정", "중계", "몇 시", "몇시"]
        ):
            return datetime.now().strftime("%Y-%m-%d")
        for pattern in [
            r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일",
            r"(\d{4})-(\d{1,2})-(\d{1,2})",
        ]:
            match = re.search(pattern, query)
            if match:
                year_str, month_str, day_str = match.groups()
                return f"{year_str}-{month_str.zfill(2)}-{day_str.zfill(2)}"
        return None

    def _resolve_subject_type(self, query_lower: str) -> Optional[str]:
        if any(token in query_lower for token in ["투수", "선발", "불펜", "마무리"]):
            return "pitcher"
        if any(token in query_lower for token in ["타자", "타선", "타격"]):
            return "batter"
        return None

    def _extract_rank_lookup(self, query_lower: str) -> Optional[tuple[int, str]]:
        if self._is_runner_up_query(query_lower):
            return None
        if "팀" not in query_lower and "구단" not in query_lower:
            return None
        if any(
            token in query_lower
            for token in [
                "투수",
                "타자",
                "홈런",
                "ops",
                "era",
                "탈삼진",
                "타점",
                "선수",
            ]
        ):
            return None

        match = re.search(r"(?<!\d)(10|[1-9])\s*(등|위)", query_lower)
        if not match:
            return None

        return int(match.group(1)), "regular"

    def _resolve_leaderboard_spec(self, query_lower: str) -> Optional[tuple[str, str]]:
        stat_specs = [
            (("평균자책", "방어율", "era"), ("pitching", "era")),
            (("whip",), ("pitching", "whip")),
            (("탈삼진", "삼진", "strikeout", "strikeouts"), ("pitching", "strikeouts")),
            (("다승", "wins", "승수"), ("pitching", "wins")),
            (("세이브", "saves", "save"), ("pitching", "saves")),
            (("홀드", "holds", "hold"), ("pitching", "holds")),
            (("ops",), ("batting", "ops")),
            (("홈런", "home_runs", "home runs"), ("batting", "home_runs")),
            (("타점", "rbi"), ("batting", "rbi")),
            (("타율", "avg"), ("batting", "avg")),
        ]
        for tokens, spec in stat_specs:
            if any(token in query_lower for token in tokens):
                return spec
        return None

    def _resolve_player_lookup_position(self, query_lower: str) -> str:
        pitching_stat_tokens = [
            "투수",
            "평균자책",
            "평균 자책",
            "평균자책점",
            "방어율",
            "era",
            "whip",
            "세이브",
            "save",
            "홀드",
            "hold",
            "탈삼진",
            "삼진",
            "다승",
            "wins",
        ]
        batting_stat_tokens = [
            "타자",
            "타율",
            "ops",
            "홈런",
            "타점",
            "타석",
            "장타",
            "출루",
            "볼넷",
            "출루율",
            "장타율",
            "안타",
        ]
        if any(token in query_lower for token in pitching_stat_tokens):
            return "pitching"
        if any(token in query_lower for token in batting_stat_tokens):
            return "batting"
        return "both"

    def _build_team_analysis_bundle_tool_calls(
        self,
        *,
        query_lower: str,
        team_name: Optional[str],
        season_year: int,
    ) -> list[ToolCall]:
        if not team_name:
            return []
        if not self._is_team_analysis_query(query_lower, team_name):
            return []

        has_recent_scope = any(
            token in query_lower
            for token in [
                "최근",
                "5경기",
                "10경기",
                "흐름",
                "페이스",
                "폼",
                "요즘",
                "살아난",
                "식은",
                "올라오는",
                "내려가는",
            ]
        )
        has_bullpen_scope = any(
            token in query_lower for token in ["불펜", "필승조", "갈아", "퍼진"]
        )
        if not (has_recent_scope and has_bullpen_scope):
            return []

        recent_limit = 10 if "10경기" in query_lower else 5
        return [
            ToolCall(
                "get_team_summary",
                {"team_name": team_name, "year": season_year},
            ),
            ToolCall(
                "get_team_advanced_metrics",
                {"team_name": team_name, "year": season_year},
            ),
            ToolCall(
                "get_recent_games_by_team",
                {
                    "team_name": team_name,
                    "year": season_year,
                    "limit": recent_limit,
                },
            ),
        ]

    def _build_schedule_bundle_tool_calls(
        self,
        *,
        query_lower: str,
        extracted_date: Optional[str],
        game_id: Optional[str],
        team_name: Optional[str],
    ) -> list[ToolCall]:
        if not extracted_date and not game_id:
            return []

        wants_schedule = any(
            token in query_lower for token in ["경기", "일정", "몇 시", "몇시"]
        )
        wants_lineup = any(
            token in query_lower for token in ["라인업", "선발", "스타팅", "타순"]
        )
        wants_box_score = self._is_box_score_query(query_lower) or self._is_game_flow_narrative_query(query_lower) or any(
            token in query_lower for token in ["스코어", "점수", "결과"]
        )

        if not wants_lineup or not (wants_schedule or wants_box_score):
            return []

        tool_calls: list[ToolCall] = []
        if extracted_date:
            schedule_params: dict[str, Any] = {"date": extracted_date}
            if team_name:
                schedule_params["team"] = team_name
            tool_calls.append(ToolCall("get_games_by_date", schedule_params))

        lineup_params: dict[str, Any] = {}
        if game_id:
            lineup_params["game_id"] = game_id
        elif extracted_date:
            lineup_params["date"] = extracted_date
            if team_name:
                lineup_params["team_name"] = team_name
        tool_calls.append(ToolCall("get_game_lineup", lineup_params))

        box_score_params: dict[str, Any] = {}
        if game_id:
            box_score_params["game_id"] = game_id
        elif extracted_date:
            box_score_params["date"] = extracted_date
        tool_calls.append(ToolCall("get_game_box_score", box_score_params))
        return tool_calls

    def _build_player_validation_bundle_tool_calls(
        self,
        *,
        query_lower: str,
        player_name: Optional[str],
        season_year: int,
        explicit_metric: Optional[tuple[str, str]],
    ) -> list[ToolCall]:
        if not player_name or not explicit_metric:
            return []
        if not self._is_player_lookup_query(query_lower):
            return []
        if not any(
            token in query_lower for token in ["순위", "몇 위", "몇위", "랭킹", "리더보드"]
        ):
            return []

        position, stat_name = explicit_metric
        return [
            ToolCall(
                "validate_player",
                {"player_name": player_name, "year": season_year},
            ),
            ToolCall(
                "get_player_stats",
                {
                    "player_name": player_name,
                    "year": season_year,
                    "position": position,
                },
            ),
            ToolCall(
                "get_leaderboard",
                {
                    "stat_name": stat_name,
                    "year": season_year,
                    "position": position,
                    "limit": 5,
                },
            ),
        ]

    def _build_player_comparison_tool_calls(
        self,
        *,
        query: str,
        query_lower: str,
        season_year: int,
    ) -> list[ToolCall]:
        if not self._is_player_comparison_query(query_lower):
            return []

        compared_players = self._extract_player_comparison_names(query)
        if not compared_players:
            return []

        player1, player2 = compared_players
        comparison_type = (
            "career"
            if any(token in query_lower for token in ["통산", "커리어", "career"])
            else "season"
        )
        parameters: dict[str, Any] = {
            "player1": player1,
            "player2": player2,
            "comparison_type": comparison_type,
            "position": self._resolve_player_lookup_position(query_lower),
        }
        if comparison_type == "season":
            parameters["year"] = season_year
        return [ToolCall("compare_players", parameters)]

    def _normalize_player_candidate(self, raw_name: str) -> str:
        cleaned = re.sub(r"\s+", "", str(raw_name or "").strip())
        cleaned = re.sub(r"(선수|타자|투수)$", "", cleaned)
        cleaned = re.sub(r"(은|는|이|가|을|를)$", "", cleaned)
        return cleaned

    def _extract_player_comparison_names(
        self, query: str
    ) -> Optional[tuple[str, str]]:
        normalized_query = query.replace("VS", "vs").replace("Vs", "vs")
        patterns = [
            r"([가-힣A-Za-z]{2,20})\s*(?:선수)?\s*(?:와|과|랑|이랑)\s*([가-힣A-Za-z]{2,20})(?:선수)?",
            r"([가-힣A-Za-z]{2,20})\s*vs\.?\s*([가-힣A-Za-z]{2,20})",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized_query, flags=re.IGNORECASE)
            if not match:
                continue
            candidate1 = self._normalize_player_candidate(match.group(1))
            candidate2 = self._normalize_player_candidate(match.group(2))
            if (
                len(candidate1) < 2
                or len(candidate2) < 2
                or candidate1 == candidate2
                or extract_team(candidate1)
                or extract_team(candidate2)
            ):
                continue
            return candidate1, candidate2
        return None

    def _is_team_metric_query(
        self,
        query_lower: str,
        team_name: Optional[str],
        player_name: Optional[str],
    ) -> bool:
        del player_name
        if not team_name:
            return False
        if self._is_regulation_query(query_lower):
            return False
        if any(token in query_lower for token in ["팀 내", "팀내"]):
            return False
        if any(
            token in query_lower
            for token in [
                "1위",
                "2위",
                "3위",
                "상위",
                "리더",
                "누가",
                "누구",
                "최고",
                "제일",
                "가장",
            ]
        ):
            return False

        team_context_tokens = ["팀", "구단"]
        team_metric_tokens = [
            "타율",
            "ops",
            "평균자책",
            "평균 자책",
            "평균자책점",
            "방어율",
            "era",
            "홈런",
            "타점",
        ]
        return any(token in query_lower for token in team_context_tokens) and any(
            token in query_lower for token in team_metric_tokens
        )

    def _is_team_analysis_query(
        self, query_lower: str, team_name: Optional[str]
    ) -> bool:
        if not self._fast_path_enabled or self._fast_path_scope != "team":
            return False
        if not team_name:
            return False
        if self._is_regulation_query(query_lower):
            return False
        if any(
            keyword in query_lower
            for keyword in ["언제", "날짜", "몇 시", "몇시", "예매", "중계"]
        ):
            return False
        team_analysis_keywords = [
            "분석",
            "요약",
            "장단점",
            "리스크",
            "흐름",
            "추이",
            "전력",
            "강점",
            "약점",
            "진단",
            "시즌",
            "폼",
            "페이스",
            "가을야구",
            "플레이오프",
            "플옵",
            "상태",
            "패턴",
            "꼬이",
            "흔들",
            "침묵",
            "터질",
            "실책",
            "승패",
            "갈아",
            "퍼진",
            "타선",
            "득점",
            "선발",
            "선발진",
            "불펜",
            "수비",
            "상대 전적",
            "상대전적",
            "5경기",
            "10경기",
            "필승조",
            "살아난",
            "식은",
            "올라오는",
            "내려가는",
            "믿고",
        ]
        if any(keyword in query_lower for keyword in team_analysis_keywords):
            return True
        if any(
            phrase in query_lower
            for phrase in [
                "어때",
                "가능성",
                "괜찮아",
                "문제",
                "보이니",
                "냉정",
                "현실적으로",
            ]
        ):
            return True
        return any(marker in query_lower for marker in ["최근", "요즘", "폼"]) and any(
            marker in query_lower
            for marker in [
                "득점",
                "타선",
                "선발",
                "선발진",
                "불펜",
                "수비",
                "상대 전적",
                "상대전적",
            ]
        )

    def _is_regulation_query(self, query_lower: str) -> bool:
        return any(
            keyword in query_lower
            for keyword in [
                "규정",
                "규칙",
                "제도",
                "판정",
                "스트라이크존",
                "fa",
                "보상선수",
                "보상 선수",
                "등록일수",
                "선수 등록",
                "드래프트",
                "엔트리",
                "로스터",
                "외국인 선수",
                "외국인선수",
                "부상자",
                "il",
                "육성선수",
                "육성 선수",
                "군보류",
                "임의해지",
                "특수 신분",
            ]
        )

    def _is_runner_up_query(self, query_lower: str) -> bool:
        return any(
            keyword in query_lower
            for keyword in [
                "준우승",
                "코시 준우승",
                "한국시리즈 준우승",
                "한국시리즈 2등",
            ]
        )

    def _is_champion_query(self, query_lower: str) -> bool:
        champion_keywords = [
            "우승팀",
            "챔피언",
            "한국시리즈 우승",
            "코시 우승",
            "우승한 팀",
        ]
        champion_future_keywords = ["가능성", "할까", "할 수", "예상", "예측", "후보"]
        return any(keyword in query_lower for keyword in champion_keywords) and not any(
            keyword in query_lower for keyword in champion_future_keywords
        )

    def _is_vague_followup_query(self, query_lower: str) -> bool:
        normalized = re.sub(r"\s+", "", query_lower.strip())
        return normalized in {
            "그냥답해줘",
            "계속",
            "이어서",
            "한줄로",
            "짧게",
            "결론만",
        }

    def _find_previous_user_question(
        self,
        context: Optional[dict[str, Any]],
        current_query: str,
    ) -> Optional[str]:
        if not isinstance(context, dict):
            return None

        messages = context.get("history")
        if not isinstance(messages, list):
            messages = context.get("messages")
        if not isinstance(messages, list):
            return None

        current_normalized = current_query.strip()
        for item in reversed(messages):
            if not isinstance(item, dict):
                continue
            role = item.get("role") or item.get("sender")
            content = item.get("content") or item.get("text")
            if role != "user" or not isinstance(content, str):
                continue
            normalized = content.strip()
            if not normalized or normalized == current_normalized:
                continue
            if self._is_vague_followup_query(normalized.lower()):
                continue
            return normalized
        return None

    def _is_ambiguous_superlative_query(self, query_lower: str) -> bool:
        if self._resolve_subject_type(query_lower) is None:
            return False
        if self._resolve_leaderboard_spec(query_lower):
            return False
        return any(
            token in query_lower
            for token in ["최고", "제일", "가장", "누가", "누구야", "잘한"]
        )

    def _is_box_score_query(self, query_lower: str) -> bool:
        return any(
            keyword in query_lower
            for keyword in [
                "박스스코어",
                "box score",
                "이닝별",
                "이닝별 득점",
                "몇 점",
                "7회",
                "8회",
                "9회",
                "연장",
            ]
        )

    def _is_game_flow_narrative_query(self, query_lower: str) -> bool:
        return any(
            keyword in query_lower
            for keyword in [
                "경기 흐름",
                "흐름 요약",
                "승부처",
                "언제 갈렸어",
                "언제 갈렸",
                "역전",
                "동점 흐름",
                "초중후반 득점",
                "득점 양상",
            ]
        )

    def _is_team_rank_query(self, query_lower: str) -> bool:
        return any(
            keyword in query_lower
            for keyword in ["순위", "몇 위", "몇위", "승률", "승패"]
        )

    def _is_team_last_game_query(self, query_lower: str) -> bool:
        return any(
            keyword in query_lower
            for keyword in ["마지막 경기", "최근 경기 언제", "최종전"]
        )

    def _is_player_comparison_query(self, query_lower: str) -> bool:
        if any(
            keyword in query_lower
            for keyword in ["승부", "이길까", "맞대결", "대결", "상대전적"]
        ):
            return False
        return any(
            keyword in query_lower
            for keyword in [
                "비교",
                "차이",
                "우위",
                "누가 더",
                "더 낫",
                "더 잘",
                "vs",
            ]
        )

    def _is_player_lookup_query(self, query_lower: str) -> bool:
        return any(
            keyword in query_lower
            for keyword in [
                "성적",
                "기록",
                "타율",
                "ops",
                "홈런",
                "타점",
                "평균자책",
                "평균 자책",
                "평균자책점",
                "방어율",
                "era",
                "whip",
                "세이브",
                "홀드",
                "탈삼진",
                "다승",
                "war",
            ]
        )


class _BoundChatIntentRouter:
    def __init__(self, *, router: ChatIntentRouter, agent: Any) -> None:
        self._router = router
        self._agent = agent

    def resolve(
        self,
        query: str,
        entity_filter: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> IntentDecision:
        return self._router.resolve(
            query,
            entity_filter,
            context,
            agent=self._agent,
        )
