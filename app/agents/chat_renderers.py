from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

from ..core.renderers.baseball import render_game_flow_summary
from .chat_intent_router import ChatIntent, IntentDecision
from .tool_caller import ToolResult

if TYPE_CHECKING:
    from .baseball_agent import BaseballStatisticsAgent


METRIC_LABELS = {
    "era": "평균자책점(ERA)",
    "avg": "타율",
    "battingavg": "타율",
    "ops": "OPS",
    "hr": "홈런",
    "homeruns": "홈런",
    "home_runs": "홈런",
    "rbi": "타점",
    "wins": "다승",
    "win": "다승",
    "whip": "WHIP",
    "saves": "세이브",
    "save": "세이브",
    "holds": "홀드",
    "hold": "홀드",
    "strikeouts": "탈삼진",
    "so": "탈삼진",
    "war": "WAR",
}

GAME_FLOW_NARRATIVE_KEYWORDS = (
    "경기 흐름",
    "흐름 요약",
    "승부처",
    "언제 갈렸어",
    "언제 갈렸",
    "역전",
    "동점 흐름",
    "초중후반 득점",
    "득점 양상",
    "이닝별 득점",
)


class ChatRendererRegistry:
    def __init__(self, agent: "BaseballStatisticsAgent" | None = None) -> None:
        self._default_agent = agent

    def bind(self, agent: "BaseballStatisticsAgent") -> "_BoundChatRendererRegistry":
        return _BoundChatRendererRegistry(registry=self, agent=agent)

    def _resolve_agent(
        self,
        agent: "BaseballStatisticsAgent" | None,
    ) -> "BaseballStatisticsAgent":
        resolved_agent = agent or self._default_agent
        if resolved_agent is None:
            raise RuntimeError("ChatRendererRegistry requires a bound agent.")
        return resolved_agent

    def render_reference(
        self,
        query: str,
        tool_results: list[ToolResult],
        decision: IntentDecision,
        *,
        agent: "BaseballStatisticsAgent" | None = None,
    ) -> Optional[str]:
        resolved_agent = self._resolve_agent(agent)
        for result in tool_results:
            if not result.success or not isinstance(result.data, dict):
                continue
            data = result.data
            if data.get("found") is False:
                continue
            if "metrics" in data and "team_name" in data:
                answer = self.render_team_metric(query, data, agent=resolved_agent)
                if answer:
                    return answer
            if "leaderboard" in data:
                answer = self.render_leaderboard(
                    query,
                    data,
                    decision,
                    agent=resolved_agent,
                )
                if answer:
                    return answer
            if "batting_stats" in data or "pitching_stats" in data:
                answer = resolved_agent._build_player_stats_chat_answer(data)
                if answer:
                    return answer
            if "awards" in data:
                answer = resolved_agent._build_award_chat_answer(data)
                if answer:
                    return answer
            if "regulations" in data:
                answer = resolved_agent._build_regulation_chat_answer(query, data)
                if answer:
                    return answer
            if "games" in data:
                answer = self.render_game_flow(query, data, agent=resolved_agent)
                if answer:
                    return answer
            if "games" in data and "date" in data:
                answer = resolved_agent._build_games_by_date_chat_answer(data)
                if answer:
                    return answer
            if "winner_team_name" in data and "series_type" in data:
                answer = resolved_agent._build_korean_series_winner_chat_answer(data)
                if answer:
                    return answer
            if "last_game_date" in data or "final_date" in data:
                answer = resolved_agent._build_team_last_game_date_chat_answer(data)
                if answer:
                    return answer
        return None

    def render_team_metric(
        self,
        query: str,
        data: dict[str, Any],
        *,
        agent: "BaseballStatisticsAgent" | None = None,
    ) -> Optional[str]:
        resolved_agent = self._resolve_agent(agent)
        query_lower = query.lower()
        if not any(token in query_lower for token in ["팀", "구단"]):
            return None
        if not any(
            token in query_lower
            for token in [
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
        ):
            return None

        metrics = data.get("metrics") or {}
        batting = metrics.get("batting") or {}
        pitching = metrics.get("pitching") or {}
        rankings = data.get("rankings") or {}
        return resolved_agent._build_team_metric_fast_path_answer(
            query,
            data.get("team_name"),
            data.get("year"),
            batting,
            pitching,
            rankings,
        )

    def render_leaderboard(
        self,
        query: str,
        data: dict[str, Any],
        decision: Optional[IntentDecision] = None,
        *,
        agent: "BaseballStatisticsAgent" | None = None,
    ) -> Optional[str]:
        resolved_agent = self._resolve_agent(agent)
        leaderboard = data.get("leaderboard") or data.get("leaders") or []
        if not leaderboard:
            return None
        year = resolved_agent._format_deterministic_metric(data.get("year"))
        raw_stat_name = resolved_agent._format_deterministic_metric(data.get("stat_name"))
        stat_name = self._metric_label(raw_stat_name)
        season_label = f"{year}년" if year != "확인 불가" else "해당 시즌"
        top_entry = leaderboard[0]
        top_player = self._player_label(top_entry, agent=resolved_agent)
        if decision and decision.intent == ChatIntent.AMBIGUOUS_SUPERLATIVE:
            if decision.subject_type == "pitcher":
                return self._render_pitcher_superlative(
                    season_label,
                    top_player,
                    top_entry,
                    agent=resolved_agent,
                )
            if decision.subject_type == "batter":
                return self._render_batter_superlative(
                    season_label,
                    top_player,
                    top_entry,
                    agent=resolved_agent,
                )
        lines = [f"{season_label} {stat_name} 기준으로 보면 상위권은 이렇게 보입니다."]
        for index, entry in enumerate(leaderboard[:3], start=1):
            player_name = self._player_label(entry, agent=resolved_agent)
            stat_value = resolved_agent._format_deterministic_metric(
                resolved_agent._extract_leaderboard_value(entry, raw_stat_name)
            )
            lines.append(
                f"{index}위는 {player_name}이고, {stat_name}은 {stat_value}입니다."
            )
        return "\n\n".join(lines[:4])

    def render_game_flow(
        self,
        query: str,
        data: dict[str, Any],
        *,
        agent: "BaseballStatisticsAgent" | None = None,
    ) -> Optional[str]:
        resolved_agent = self._resolve_agent(agent)
        query_lower = query.lower()
        if not any(keyword in query_lower for keyword in GAME_FLOW_NARRATIVE_KEYWORDS):
            return None

        games = data.get("games") or []
        if not isinstance(games, list):
            return None

        box_score_games = [
            game
            for game in games
            if isinstance(game, dict) and isinstance(game.get("box_score"), dict)
        ]
        if not box_score_games:
            return None

        team_code = resolved_agent._detect_team_alias_from_query(query)
        if team_code:
            filtered_games = [
                game
                for game in box_score_games
                if team_code
                in {
                    game.get("home_team_code"),
                    game.get("away_team_code"),
                    game.get("home_team"),
                    game.get("away_team"),
                }
            ]
            if filtered_games:
                box_score_games = filtered_games

        if len(box_score_games) != 1:
            game_summaries = []
            for game in box_score_games[:3]:
                away_team = resolved_agent._format_team_display_name(
                    game.get("away_team_code")
                    or game.get("away_team")
                    or game.get("away_team_name")
                )
                home_team = resolved_agent._format_team_display_name(
                    game.get("home_team_code")
                    or game.get("home_team")
                    or game.get("home_team_name")
                )
                away_score = resolved_agent._format_deterministic_metric(
                    game.get("away_score")
                )
                home_score = resolved_agent._format_deterministic_metric(
                    game.get("home_score")
                )
                game_summaries.append(
                    f"{away_team} {away_score}-{home_score} {home_team}"
                )

            date_label = resolved_agent._format_deterministic_metric(
                data.get("date") or (data.get("query_params") or {}).get("date")
            )
            if date_label == "확인 불가":
                date_label = "그 날짜"
            return (
                f"{date_label}에는 경기 흐름을 볼 대상이 {len(box_score_games)}경기라 한 경기로 바로 못 좁히겠습니다.\n\n"
                f"{', '.join(game_summaries)} 중 어떤 경기를 볼지 팀명이나 game_id를 같이 알려주세요."
            )

        row = self._build_game_flow_row_from_box_score(
            box_score_games[0],
            agent=resolved_agent,
        )
        if row is None:
            return None

        rendered = render_game_flow_summary(row, today_str=row.get("game_date"))
        return self._normalize_game_flow_answer(rendered, agent=resolved_agent)

    def render_low_data(
        self,
        query: str,
        tool_results: list[ToolResult],
        decision: IntentDecision,
        *,
        agent: "BaseballStatisticsAgent" | None = None,
    ) -> str:
        resolved_agent = self._resolve_agent(agent)
        del tool_results
        if decision.intent == ChatIntent.REGULATION_LOOKUP:
            return (
                "지금 조회로는 질문에 딱 맞는 규정 조항이 충분히 안 잡혔습니다.\n\n"
                "괜히 뭉뚱그려 설명해서 헷갈리게 하진 않겠습니다.\n\n"
                "`FA 보상선수 규정`, `등록일수`, `1군 엔트리`처럼 주제를 한 번만 더 좁혀주시면 바로 정리해드릴게요."
            )
        if decision.subject_type == "team_last_game":
            return (
                "현재 연결된 자료에서는 마지막 경기 날짜를 확정하지 못했습니다.\n\n"
                "확인되지 않은 날짜를 추정해서 말하지는 않겠습니다."
            )
        if decision.intent == ChatIntent.SCHEDULE_LOOKUP:
            return (
                "현재 연결된 일정 자료에서는 질문에 맞는 경기 정보를 확인하지 못했습니다.\n\n"
                "없는 경기를 있는 것처럼 말하지는 않겠습니다."
            )
        if decision.subject_type == "team_rank":
            return (
                "현재 연결된 순위 자료에서는 해당 팀의 시즌 순위를 확정하지 못했습니다.\n\n"
                "확인되지 않은 순위를 추정해서 답하지는 않겠습니다."
            )
        if decision.intent == ChatIntent.AMBIGUOUS_SUPERLATIVE:
            role_label = "투수" if decision.subject_type == "pitcher" else "타자"
            return (
                f"현재 연결된 자료만으로는 특정 {role_label} 한 명을 단정하기 어렵습니다.\n\n"
                f"확인되지 않은 비교 결과를 추정해서 답하지는 않겠습니다."
            )
        if decision.intent == ChatIntent.LEADERBOARD_LOOKUP:
            return (
                "현재 연결된 통계 자료에서는 해당 시즌 리더보드를 확정하지 못했습니다.\n\n"
                "빈칸을 추정으로 메우지 않고 확인된 결과만 말씀드리겠습니다."
            )
        if decision.intent == ChatIntent.PLAYER_LOOKUP:
            return (
                "현재 연결된 통계 자료에서는 요청한 선수 기록을 정확히 매칭하지 못했습니다.\n\n"
                "없는 기록을 지어내지는 않겠습니다."
            )
        if decision.intent == ChatIntent.TEAM_ANALYSIS:
            query_lower = query.lower()
            metric_label = None
            if "ops" in query_lower:
                metric_label = "팀 OPS"
            elif "타율" in query_lower:
                metric_label = "팀 타율"
            elif any(
                token in query_lower
                for token in ["평균자책", "평균 자책", "평균자책점", "방어율", "era"]
            ):
                metric_label = "팀 평균자책점"
            elif "홈런" in query_lower:
                metric_label = "팀 홈런"
            elif "타점" in query_lower:
                metric_label = "팀 타점"

            if metric_label:
                team_name = resolved_agent._detect_team_alias_from_query(query)
                team_label = (
                    resolved_agent._format_team_display_name(team_name)
                    if team_name
                    else "해당 팀"
                )
                return (
                    f"현재 연결된 자료에서는 {team_label} {metric_label}를 직접 확인하지 못했습니다.\n\n"
                    "확인되지 않은 팀 지표를 추정해서 답하지는 않겠습니다."
                )
            return (
                "현재 연결된 자료만으로는 팀 흐름을 단정하기 어렵습니다.\n\n"
                "확인된 범위에서만 말씀드리겠습니다."
            )
        query_lower = query.lower()
        metric_label = None
        if "ops" in query_lower:
            metric_label = "팀 OPS"
        elif "타율" in query_lower:
            metric_label = "팀 타율"
        elif any(
            token in query_lower
            for token in ["평균자책", "평균 자책", "평균자책점", "방어율", "era"]
        ):
            metric_label = "팀 평균자책점"
        elif "홈런" in query_lower:
            metric_label = "팀 홈런"
        elif "타점" in query_lower:
            metric_label = "팀 타점"

        if metric_label and any(token in query_lower for token in ["팀", "구단"]):
            team_name = resolved_agent._detect_team_alias_from_query(query)
            team_label = (
                resolved_agent._format_team_display_name(team_name)
                if team_name
                else "해당 팀"
            )
            return (
                f"현재 연결된 자료에서는 {team_label} {metric_label}를 직접 확인하지 못했습니다.\n\n"
                "확인되지 않은 팀 지표를 추정해서 답하지는 않겠습니다."
            )
        return (
            "현재 연결된 자료만으로는 질문에 대해 확인된 답을 만들지 못했습니다.\n\n"
            "추정하지 않고 확인된 범위에서만 말씀드리겠습니다."
        )

    def _player_label(
        self,
        entry: dict[str, Any],
        *,
        agent: "BaseballStatisticsAgent",
    ) -> str:
        player_name = agent._format_deterministic_metric(
            entry.get("player_name") or entry.get("name")
        )
        team_name = agent._format_team_display_name(
            entry.get("team_name") or entry.get("team")
        )
        if team_name != "확인 불가":
            return f"{player_name}({team_name})"
        return player_name

    def _metric_label(self, raw_name: Any) -> str:
        stat_key = re.sub(r"[^a-z0-9가-힣_]+", "", str(raw_name).lower())
        return METRIC_LABELS.get(stat_key, str(raw_name))

    def _build_game_flow_row_from_box_score(
        self,
        game: dict[str, Any],
        *,
        agent: "BaseballStatisticsAgent",
    ) -> Optional[dict[str, Any]]:
        box_score = game.get("box_score")
        if not isinstance(box_score, dict):
            return None

        inning_lines = []
        for inning in range(1, 13):
            away_runs = box_score.get(f"away_{inning}")
            home_runs = box_score.get(f"home_{inning}")
            if away_runs in (None, "") and home_runs in (None, ""):
                continue
            inning_lines.append(
                {
                    "inning": inning,
                    "away_runs": int(away_runs or 0),
                    "home_runs": int(home_runs or 0),
                    "is_extra": inning > 9,
                }
            )

        return {
            "game_id": game.get("game_id"),
            "game_date": game.get("game_date"),
            "home_team": game.get("home_team_code") or game.get("home_team"),
            "away_team": game.get("away_team_code") or game.get("away_team"),
            "home_team_name": agent._format_team_display_name(
                game.get("home_team_code")
                or game.get("home_team")
                or game.get("home_team_name")
            ),
            "away_team_name": agent._format_team_display_name(
                game.get("away_team_code")
                or game.get("away_team")
                or game.get("away_team_name")
            ),
            "home_score": game.get("home_score"),
            "away_score": game.get("away_score"),
            "winning_team": game.get("winning_team_code") or game.get("winning_team"),
            "inning_lines_json": inning_lines,
            "source_table": "game_box_score",
            "source_row_id": (
                f"game_id={game.get('game_id')}" if game.get("game_id") else ""
            ),
        }

    def _normalize_game_flow_answer(
        self,
        rendered: str,
        *,
        agent: "BaseballStatisticsAgent",
    ) -> str:
        paragraphs = []
        for raw_line in rendered.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped == "[상세]":
                continue
            if stripped.startswith("[META]") or stripped.startswith("[출처]"):
                continue
            for prefix in ("[TL;DR] ", "[핵심 문장] "):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :].strip()
                    break
            if stripped.startswith("- "):
                stripped = stripped[2:].strip()
            if stripped:
                paragraphs.append(stripped)
        return agent._normalize_chatbot_answer_text("\n\n".join(paragraphs))

    def _render_pitcher_superlative(
        self,
        season_label: str,
        top_player: str,
        top_entry: dict[str, Any],
        *,
        agent: "BaseballStatisticsAgent",
    ) -> str:
        basis_parts = []
        era = agent._format_deterministic_metric(top_entry.get("era"))
        whip = agent._format_deterministic_metric(top_entry.get("whip"))
        strikeouts = agent._format_deterministic_metric(top_entry.get("strikeouts"))
        innings_pitched = agent._format_deterministic_metric(
            top_entry.get("innings_pitched")
        )
        if era != "확인 불가":
            basis_parts.append(f"평균자책점(ERA) {era}")
        if whip != "확인 불가":
            basis_parts.append(f"WHIP {whip}")
        if strikeouts != "확인 불가":
            basis_parts.append(f"탈삼진 {strikeouts}개")
        if len(basis_parts) < 3 and innings_pitched != "확인 불가":
            basis_parts.append(f"이닝 {innings_pitched}")
        basis_text = ", ".join(basis_parts[:3]) or "핵심 투수 지표"
        detail = f"지금 답은 {basis_text}를 같이 본 1차 판단입니다."
        if innings_pitched != "확인 불가":
            detail = f"{detail} 이닝 소화는 {innings_pitched} 수준이라 표본도 너무 가볍진 않습니다."
        return (
            f"{season_label} 최고 투수를 빠르게 한 명만 꼽자면 {top_player}입니다.\n\n"
            f"{detail}\n\n"
            "WAR이나 시즌 기여도까지 같이 보면 순서는 달라질 수 있지만, 빠르게 답하면 지금은 이 쪽이 가장 설득력 있습니다."
        )

    def _render_batter_superlative(
        self,
        season_label: str,
        top_player: str,
        top_entry: dict[str, Any],
        *,
        agent: "BaseballStatisticsAgent",
    ) -> str:
        basis_parts = []
        ops = agent._format_deterministic_metric(top_entry.get("ops"))
        home_runs = agent._format_deterministic_metric(top_entry.get("home_runs"))
        rbi = agent._format_deterministic_metric(top_entry.get("rbi"))
        avg = agent._format_deterministic_metric(top_entry.get("avg"))
        if ops != "확인 불가":
            basis_parts.append(f"OPS {ops}")
        if home_runs != "확인 불가":
            basis_parts.append(f"홈런 {home_runs}개")
        if rbi != "확인 불가":
            basis_parts.append(f"타점 {rbi}개")
        if len(basis_parts) < 3 and avg != "확인 불가":
            basis_parts.append(f"타율 {avg}")
        basis_text = ", ".join(basis_parts[:3]) or "핵심 타격 지표"
        return (
            f"{season_label} 최고 타자를 빠르게 한 명만 꼽자면 {top_player}입니다.\n\n"
            f"지금 답은 {basis_text}를 같이 본 1차 판단입니다.\n\n"
            "WAR이나 득점 생산력을 더 강하게 보면 순서는 달라질 수 있지만, 빠르게 답하면 지금은 이 선수가 가장 먼저 올라옵니다."
        )


class _BoundChatRendererRegistry:
    def __init__(
        self,
        *,
        registry: ChatRendererRegistry,
        agent: "BaseballStatisticsAgent",
    ) -> None:
        self._registry = registry
        self._agent = agent

    def render_reference(
        self,
        query: str,
        tool_results: list[ToolResult],
        decision: IntentDecision,
    ) -> Optional[str]:
        return self._registry.render_reference(
            query,
            tool_results,
            decision,
            agent=self._agent,
        )

    def render_leaderboard(
        self,
        query: str,
        data: dict[str, Any],
        decision: Optional[IntentDecision] = None,
    ) -> Optional[str]:
        return self._registry.render_leaderboard(
            query,
            data,
            decision,
            agent=self._agent,
        )

    def render_low_data(
        self,
        query: str,
        tool_results: list[ToolResult],
        decision: IntentDecision,
    ) -> str:
        return self._registry.render_low_data(
            query,
            tool_results,
            decision,
            agent=self._agent,
        )
