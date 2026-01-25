"""
의도별로 다양한 컨텍스트 포맷을 제공하는 모듈입니다.

질문의 의도(intent)에 따라 검색된 데이터를 가장 적절한 형식으로
포맷팅하여 LLM이 더 나은 답변을 생성할 수 있도록 도와줍니다.
"""

from typing import Dict, List, Any, Optional
import logging
from . import kbo_metrics

logger = logging.getLogger(__name__)

class ContextFormatter:
    """질문 의도에 따른 컨텍스트 포맷터 클래스"""
    
    def __init__(self):
        self.MIN_IP_SP = 70
        self.MIN_IP_RP = 30  
        self.MIN_PA_BATTER = 100
    
    def format_context(
        self,
        processed_data: Dict[str, Any],
        intent: str,
        query: str,
        entity_filter,
        year: int
    ) -> str:
        """
        질문 의도에 따라 최적화된 컨텍스트 형식을 생성하여 반환합니다.

        인수:
            processed_data: _process_and_enrich_docs에서 처리된 데이터
            intent: 질문 의도 (통계조회, 선수프로필, 비교분석, 설명형 등)
            query: 사용자의 원본 질문
            entity_filter: 질문에서 추출된 엔티티 정보 (연도, 팀, 선수명 등)
            year: 분석 대상 연도
        """
        logger.info(f"[ContextFormatter] Formatting context for intent: {intent}")

        # 수상 관련 질문
        if intent == "award_lookup" or entity_filter.award_type:
            return self._format_award_context(processed_data, query, entity_filter, year)

        # 경기 상세 질문
        if intent == "game_detail" or entity_filter.game_date:
            return self._format_game_detail(processed_data, query, entity_filter, year)

        # 선수 이동/이적 질문
        if intent == "movement_lookup" or entity_filter.movement_type:
            return self._format_player_movement(processed_data, query, entity_filter, year)

        # 기존 의도들
        if intent == "stats_lookup" or entity_filter.stat_type:
            return self._format_statistical_ranking(processed_data, query, entity_filter, year)
        elif intent == "player_profile" or entity_filter.player_name:
            return self._format_player_profile(processed_data, query, entity_filter, year)
        elif intent == "team_analysis" or entity_filter.team_id:
            return self._format_team_analysis(processed_data, query, entity_filter, year)
        elif intent == "comparison":
            return self._format_comparison_table(processed_data, query, entity_filter, year)
        elif intent == "explanatory":
            return self._format_explanatory_content(processed_data, query, entity_filter, year)
        else:
            # 기본값: 통계 순위 형식 사용
            return self._format_statistical_ranking(processed_data, query, entity_filter, year)
    
    def _format_statistical_ranking(
        self, 
        processed_data: Dict[str, Any], 
        query: str, 
        entity_filter, 
        year: int
    ) -> str:
        """통계 지표 기반 랭킹 형식으로 포맷합니다 (기존 방식 개선)."""
        context_parts = []
        
        sp_pitchers = [p for p in processed_data["pitchers"] if p["role"] == "SP"]
        rp_pitchers = [p for p in processed_data["pitchers"] if p["role"] == "RP"]
        batters = processed_data["batters"]
        
        # 요청된 통계 지표에 따라 정렬 순서 조정
        if entity_filter.stat_type:
            sp_pitchers = self._sort_by_requested_stat(sp_pitchers, entity_filter.stat_type, "pitcher")
            rp_pitchers = self._sort_by_requested_stat(rp_pitchers, entity_filter.stat_type, "pitcher")
            batters = self._sort_by_requested_stat(batters, entity_filter.stat_type, "batter")
        
        if sp_pitchers:
            header = kbo_metrics.scope_header(year, len(set(p['team'] for p in sp_pitchers)), "SP", self.MIN_IP_SP)
            context_parts.append(f"### 선발 투수 성적\n{header}\n")
            for i, p in enumerate(sp_pitchers[:10], 1):  # 상위 10명으로 제한
                line = self._format_pitcher_line(p, i, entity_filter.stat_type)
                context_parts.append(f"{i}. {line}")
        
        if rp_pitchers:
            header = kbo_metrics.scope_header(year, len(set(p['team'] for p in rp_pitchers)), "RP", self.MIN_IP_RP)
            context_parts.append(f"\n### 불펜 투수 성적\n{header}\n")
            for i, p in enumerate(rp_pitchers[:10], 1):
                line = self._format_pitcher_line(p, i, entity_filter.stat_type)
                context_parts.append(f"{i}. {line}")

        if batters:
            header = kbo_metrics.scope_header(year, len(set(b['team'] for b in batters)), "BAT", self.MIN_PA_BATTER)
            context_parts.append(f"\n### 타자 성적\n{header}\n")
            for i, b in enumerate(batters[:15], 1):  # 상위 15명으로 제한
                line = self._format_batter_line(b, i, entity_filter.stat_type)
                context_parts.append(f"{i}. {line}")
        
        if not context_parts:
            context_parts.append("요청하신 조건에 해당하는 데이터를 찾을 수 없습니다.")
        
        return "\n".join(context_parts)
    
    def _format_player_profile(
        self, 
        processed_data: Dict[str, Any], 
        query: str, 
        entity_filter, 
        year: int
    ) -> str:
        """특정 선수에 대한 상세 프로필 형식으로 포맷합니다."""
        context_parts = []
        
        target_player = entity_filter.player_name
        if not target_player:
            return self._format_statistical_ranking(processed_data, query, entity_filter, year)
        
        # 해당 선수 찾기
        all_players = processed_data["pitchers"] + processed_data["batters"]
        player_data = None
        for player in all_players:
            if target_player in player["name"] or player["name"] in target_player:
                player_data = player
                break
        
        if not player_data:
            context_parts.append(f"### {target_player} 선수 정보")
            context_parts.append(f"'{target_player}' 선수의 {year}년 상세 기록 데이터를 찾을 수 없습니다.")
            context_parts.append("")
            context_parts.append("**가능한 원인:**")
            context_parts.append("- 선수명 오타 또는 다른 표기법")  
            context_parts.append("- 해당 연도에 최소 출전 기준 미달 (타자 100타석, 투수 30이닝 미만)")
            context_parts.append("- 다른 연도 또는 다른 리그 소속")
            return "\n".join(context_parts)
        
        # 선수 기본 정보
        context_parts.append(f"### {player_data['name']} ({player_data['team']}) - {year}년 성적")
        
        # 투수인지 타자인지 구분하여 상세 정보 제공
        if player_data in processed_data["pitchers"]:
            context_parts.append(f"**포지션**: {player_data['role']} (투수)")
            context_parts.append(f"**이닝**: {kbo_metrics.format_ip(player_data['ip'])} IP")
            context_parts.append(f"**방어율**: {player_data['era']:.2f}")
            context_parts.append(f"**WHIP**: {kbo_metrics.describe_metric_ko('WHIP', player_data['whip'])}")
            context_parts.append(f"**ERA-**: {kbo_metrics.describe_metric_ko('ERA-', player_data['era_minus'], 0)}")
            context_parts.append(f"**FIP-**: {kbo_metrics.describe_metric_ko('FIP-', player_data['fip_minus'], 0)}")
            context_parts.append(f"**K-BB%**: {kbo_metrics.describe_metric_ko('K-BB%', player_data['kbb_pct'], 1)}%")
        else:
            context_parts.append(f"**포지션**: 타자")
            context_parts.append(f"**타석수**: {player_data['pa']} PA")
            
            # 기본 타격 지표 (타율은 항상 우선 표시)
            if player_data.get('avg'):
                context_parts.append(f"**타율(AVG)**: {kbo_metrics.describe_metric_ko('AVG', player_data['avg'], 3)}")
            
            # 출루율/장타율
            if player_data.get('obp'):
                context_parts.append(f"**출루율(OBP)**: {kbo_metrics.describe_metric_ko('OBP', player_data['obp'], 3)}")
            if player_data.get('slg'):
                context_parts.append(f"**장타율(SLG)**: {kbo_metrics.describe_metric_ko('SLG', player_data['slg'], 3)}")
                
            # 종합 지표
            if player_data.get('ops'):
                context_parts.append(f"**OPS**: {kbo_metrics.describe_metric_ko('OPS', player_data['ops'], 3)}")
            if player_data.get('ops_plus'):
                context_parts.append(f"**OPS+**: {kbo_metrics.describe_metric_ko('OPS+', player_data['ops_plus'], 0)}")
            if player_data.get('wrc_plus'):
                context_parts.append(f"**wRC+**: {kbo_metrics.describe_metric_ko('WRC+', player_data['wrc_plus'], 0)}")
            if player_data.get('war'):
                context_parts.append(f"**WAR**: {kbo_metrics.describe_metric_ko('WAR', player_data['war'], 2)}")
            
            # 기록 상세
            counting_stats = []
            if player_data.get('home_runs'):
                counting_stats.append(f"홈런 {int(player_data['home_runs'])}개")
            if player_data.get('rbi'):
                counting_stats.append(f"타점 {int(player_data['rbi'])}개")
            if player_data.get('steals'):
                counting_stats.append(f"도루 {int(player_data['steals'])}개")
            if counting_stats:
                context_parts.append(f"**주요 기록**: {', '.join(counting_stats)}")
        
        return "\n".join(context_parts)
    
    def _format_team_analysis(
        self, 
        processed_data: Dict[str, Any], 
        query: str, 
        entity_filter, 
        year: int
    ) -> str:
        """팀 단위 분석 형식으로 포맷합니다."""
        target_team = entity_filter.team_id
        if not target_team:
            return self._format_statistical_ranking(processed_data, query, entity_filter, year)
        
        context_parts = []
        context_parts.append(f"### {self._get_full_team_name(target_team)} {year}년 주요 선수")
        
        # 팀 소속 선수들만 필터링
        team_pitchers = [p for p in processed_data["pitchers"] if p["team"] == self._get_full_team_name(target_team)]
        team_batters = [b for b in processed_data["batters"] if b["team"] == self._get_full_team_name(target_team)]
        
        if team_pitchers:
            context_parts.append(f"\n**주요 투수 ({len(team_pitchers)}명)**")
            for i, p in enumerate(team_pitchers[:5], 1):
                line = f"{p['name']} - ERA {p['era']:.2f}, WHIP {p['whip']:.2f}, {kbo_metrics.format_ip(p['ip'])} IP"
                context_parts.append(f"{i}. {line}")
        
        if team_batters:
            context_parts.append(f"\n**주요 타자 ({len(team_batters)}명)**")
            for i, b in enumerate(team_batters[:5], 1):
                ops_str = f"OPS {b['ops']:.3f}" if b.get('ops') else ""
                hr_rbi = []
                if b.get('home_runs'):
                    hr_rbi.append(f"HR {int(b['home_runs'])}")
                if b.get('rbi'):
                    hr_rbi.append(f"RBI {int(b['rbi'])}")
                counting = f", {'/'.join(hr_rbi)}" if hr_rbi else ""
                line = f"{b['name']} - {ops_str}{counting}"
                context_parts.append(f"{i}. {line}")
        
        if not team_pitchers and not team_batters:
            context_parts.append(f"{target_team} 소속 선수 데이터를 찾을 수 없습니다.")
        
        return "\n".join(context_parts)
    
    def _format_comparison_table(
        self, 
        processed_data: Dict[str, Any], 
        query: str, 
        entity_filter, 
        year: int
    ) -> str:
        """비교 분석을 위한 테이블 형식으로 포맷합니다."""
        context_parts = []
        context_parts.append(f"### {year}년 선수 비교 분석")
        
        # 상위 선수들을 표 형태로 정리
        if processed_data["pitchers"]:
            context_parts.append("\n**투수 비교표**")
            context_parts.append("| 순위 | 선수명 | 팀 | ERA | WHIP | IP | ERA- | FIP- |")
            context_parts.append("|------|-------|-----|-----|------|-----|------|------|")
            
            for i, p in enumerate(processed_data["pitchers"][:8], 1):
                row = f"| {i} | {p['name']} | {p['team']} | {p['era']:.2f} | {p['whip']:.2f} | {kbo_metrics.format_ip(p['ip'])} | {p['era_minus']:.0f} | {p['fip_minus']:.0f} |"
                context_parts.append(row)
        
        if processed_data["batters"]:
            context_parts.append("\n**타자 비교표**") 
            context_parts.append("| 순위 | 선수명 | 팀 | OPS | wRC+ | HR | RBI | WAR |")
            context_parts.append("|------|-------|-----|-----|------|-----|-----|-----|")
            
            for i, b in enumerate(processed_data["batters"][:10], 1):
                ops_val = f"{b['ops']:.3f}" if b.get('ops') else "-"
                wrc_val = f"{b['wrc_plus']:.0f}" if b.get('wrc_plus') else "-"
                hr_val = str(int(b['home_runs'])) if b.get('home_runs') else "-"
                rbi_val = str(int(b['rbi'])) if b.get('rbi') else "-"
                war_val = f"{b['war']:.1f}" if b.get('war') else "-"
                row = f"| {i} | {b['name']} | {b['team']} | {ops_val} | {wrc_val} | {hr_val} | {rbi_val} | {war_val} |"
                context_parts.append(row)
        
        return "\n".join(context_parts)
    
    def _format_explanatory_content(
        self, 
        processed_data: Dict[str, Any], 
        query: str, 
        entity_filter, 
        year: int
    ) -> str:
        """설명형 질문을 위한 자유로운 텍스트 포맷입니다."""
        # 설명형 질문의 경우 원본 검색 컨텍스트를 그대로 사용
        context_parts = []
        context_parts.append(f"### {year}년 KBO 리그 관련 정보")
        
        # 간단한 요약 정보만 제공
        if processed_data["pitchers"]:
            context_parts.append(f"\n**투수 데이터**: {len(processed_data['pitchers'])}명의 투수 정보")
        if processed_data["batters"]:
            context_parts.append(f"**타자 데이터**: {len(processed_data['batters'])}명의 타자 정보")
        
        context_parts.append("\n상세한 통계나 분석이 필요하시면 구체적인 질문을 해주세요.")
        return "\n".join(context_parts)
    
    def _sort_by_requested_stat(self, players: List[Dict], stat_type: str, position: str) -> List[Dict]:
        """요청된 통계 지표에 따라 선수를 정렬합니다."""
        if not stat_type or not players:
            return players
        
        # 기본 정렬 키 매핑
        sort_mapping = {
            # 투수 지표 (낮을수록 좋음)
            "era": ("era", False),
            "whip": ("whip", False),
            # 타자 지표 (높을수록 좋음)  
            "ops": ("ops", True),
            "home_runs": ("home_runs", True),
            "rbi": ("rbi", True),
            "war": ("war", True),
            "wrc_plus": ("wrc_plus", True),
        }
        
        if stat_type in sort_mapping:
            key, reverse = sort_mapping[stat_type]
            try:
                return sorted(players, key=lambda p: p.get(key, 0 if reverse else 999), reverse=reverse)
            except (KeyError, TypeError):
                pass
        
        return players  # 정렬 실패 시 원래 순서 유지
    
    def _format_pitcher_line(self, pitcher: Dict, rank: int, focus_stat: Optional[str] = None) -> str:
        """투수 한 줄 요약을 포맷합니다."""
        base = f"{pitcher['name']}({pitcher['team']})"
        
        if focus_stat == "era":
            return f"{base} — **ERA {pitcher['era']:.2f}**, WHIP {pitcher['whip']:.2f}, {kbo_metrics.format_ip(pitcher['ip'])} IP"
        elif focus_stat == "whip":
            return f"{base} — **WHIP {pitcher['whip']:.2f}**, ERA {pitcher['era']:.2f}, {kbo_metrics.format_ip(pitcher['ip'])} IP"
        else:
            # 기본 종합 정보
            return f"{base} — ERA {pitcher['era']:.2f}, WHIP {pitcher['whip']:.2f}, ERA- {pitcher['era_minus']:.0f}, {kbo_metrics.format_ip(pitcher['ip'])} IP"
    
    def _format_batter_line(self, batter: Dict, rank: int, focus_stat: Optional[str] = None) -> str:
        """타자 한 줄 요약을 포맷합니다."""
        base = f"{batter['name']}({batter['team']})"
        
        if focus_stat == "ops" and batter.get('ops'):
            return f"{base} — **OPS {batter['ops']:.3f}**, wRC+ {batter.get('wrc_plus', 0):.0f}, {batter['pa']} PA"
        elif focus_stat == "home_runs" and batter.get('home_runs'):
            return f"{base} — **HR {int(batter['home_runs'])}**, RBI {int(batter.get('rbi', 0))}, OPS {batter.get('ops', 0):.3f}"
        elif focus_stat == "war" and batter.get('war'):
            return f"{base} — **WAR {batter['war']:.1f}**, wRC+ {batter.get('wrc_plus', 0):.0f}, OPS {batter.get('ops', 0):.3f}"
        else:
            # 기본 종합 정보
            metrics = []
            if batter.get('ops'):
                metrics.append(f"OPS {batter['ops']:.3f}")
            if batter.get('wrc_plus'):
                metrics.append(f"wRC+ {batter['wrc_plus']:.0f}")
            if batter.get('home_runs'):
                metrics.append(f"HR {int(batter['home_runs'])}")
            return f"{base} — {', '.join(metrics)}, {batter['pa']} PA"
    
    def _get_full_team_name(self, team_id: str) -> str:
        """팀 ID를 전체 팀명으로 변환합니다."""
        team_names = {
            "KIA": "KIA 타이거즈", "LG": "LG 트윈스", "두산": "두산 베어스",
            "롯데": "롯데 자이언츠", "삼성": "삼성 라이온즈", "키움": "키움 히어로즈",
            "한화": "한화 이글스", "KT": "KT 위즈", "NC": "NC 다이노스", "SSG": "SSG 랜더스"
        }
        return team_names.get(team_id, team_id)

    def _format_award_context(
        self,
        processed_data: Dict[str, Any],
        query: str,
        entity_filter,
        year: int
    ) -> str:
        """수상 기록 형식으로 포맷합니다."""
        context_parts = []

        award_type_display = {
            "mvp": "MVP",
            "rookie": "신인왕",
            "golden_glove": "골든글러브",
            "batting_title": "타격왕",
            "hr_leader": "홈런왕",
            "rbi_leader": "타점왕",
            "sb_leader": "도루왕",
            "wins_leader": "다승왕",
            "era_leader": "방어율왕",
            "saves_leader": "세이브왕",
            "so_leader": "탈삼진왕",
        }

        # 수상 유형 표시
        if entity_filter.award_type and entity_filter.award_type != "any":
            award_name = award_type_display.get(entity_filter.award_type, entity_filter.award_type)
            context_parts.append(f"### {year}년 KBO {award_name}")
        else:
            context_parts.append(f"### {year}년 KBO 수상 기록")

        # 수상 데이터 추출 (processed_data에서 awards 관련 문서 찾기)
        awards = processed_data.get("awards", [])

        if not awards:
            # raw_docs에서 수상 관련 데이터 추출 시도
            raw_docs = processed_data.get("raw_docs", [])
            for doc in raw_docs:
                if doc.get("source_table") == "awards":
                    awards.append(doc)

        if awards:
            # 수상 유형별로 그룹화
            award_groups: Dict[str, List] = {}
            for award in awards:
                award_type = award.get("award_type", "기타")
                if award_type not in award_groups:
                    award_groups[award_type] = []
                award_groups[award_type].append(award)

            for award_type, award_list in award_groups.items():
                display_name = award_type_display.get(award_type.lower(), award_type)
                context_parts.append(f"\n**{display_name}**")
                for award in award_list:
                    player = award.get("player_name", "알 수 없음")
                    team = award.get("team_name") or award.get("team", "")
                    position = award.get("position", "")
                    line = f"- {player}"
                    if team:
                        line += f" ({team})"
                    if position:
                        line += f" - {position}"
                    context_parts.append(line)
        else:
            context_parts.append(f"\n{year}년 수상 기록을 찾을 수 없습니다.")

        return "\n".join(context_parts)

    def _format_game_detail(
        self,
        processed_data: Dict[str, Any],
        query: str,
        entity_filter,
        year: int
    ) -> str:
        """경기 상세 정보 형식으로 포맷합니다."""
        context_parts = []

        game_date = entity_filter.game_date
        if game_date:
            context_parts.append(f"### {game_date} 경기 정보")
        else:
            context_parts.append(f"### {year}년 경기 정보")

        # 경기 데이터 추출
        games = processed_data.get("games", [])
        if not games:
            raw_docs = processed_data.get("raw_docs", [])
            for doc in raw_docs:
                if doc.get("source_table") in ["game", "game_metadata", "game_inning_scores"]:
                    games.append(doc)

        if games:
            # 경기별로 그룹화
            seen_games = set()
            for game in games:
                game_id = game.get("game_id")
                if game_id and game_id not in seen_games:
                    seen_games.add(game_id)

                    home_team = game.get("home_team_name") or game.get("home_team", "")
                    away_team = game.get("away_team_name") or game.get("away_team", "")
                    home_score = game.get("home_score", "?")
                    away_score = game.get("away_score", "?")
                    stadium = game.get("stadium_name") or game.get("stadium", "")
                    date = game.get("game_date", "")

                    context_parts.append(f"\n**{date} {away_team} @ {home_team}**")
                    context_parts.append(f"- 스코어: {away_team} {away_score} - {home_score} {home_team}")
                    if stadium:
                        context_parts.append(f"- 구장: {stadium}")

                    # 관중, 경기시간 등 추가 정보
                    attendance = game.get("attendance") or game.get("crowd")
                    if attendance:
                        context_parts.append(f"- 관중: {attendance:,}명" if isinstance(attendance, int) else f"- 관중: {attendance}")

                    game_time = game.get("game_duration") or game.get("game_time")
                    if game_time:
                        context_parts.append(f"- 경기 시간: {game_time}")
        else:
            if game_date:
                context_parts.append(f"\n{game_date} 경기 정보를 찾을 수 없습니다.")
            else:
                context_parts.append(f"\n경기 정보를 찾을 수 없습니다.")

        return "\n".join(context_parts)

    def _format_player_movement(
        self,
        processed_data: Dict[str, Any],
        query: str,
        entity_filter,
        year: int
    ) -> str:
        """선수 이동 기록 형식으로 포맷합니다."""
        context_parts = []

        movement_type_display = {
            "fa": "FA (자유계약)",
            "trade": "트레이드",
            "draft": "드래프트",
            "foreign": "외국인 선수",
            "release": "방출",
            "retirement": "은퇴",
            "military": "군보류",
            "return": "복귀",
        }

        # 이동 유형 표시
        if entity_filter.movement_type and entity_filter.movement_type != "any":
            movement_name = movement_type_display.get(entity_filter.movement_type, entity_filter.movement_type)
            context_parts.append(f"### {year}년 {movement_name} 기록")
        else:
            context_parts.append(f"### {year}년 선수 이동 기록")

        # 이동 데이터 추출
        movements = processed_data.get("movements", [])
        if not movements:
            raw_docs = processed_data.get("raw_docs", [])
            for doc in raw_docs:
                if doc.get("source_table") == "player_movements":
                    movements.append(doc)

        if movements:
            # 이동 유형별로 그룹화
            movement_groups: Dict[str, List] = {}
            for movement in movements:
                m_type = movement.get("section") or movement.get("movement_type", "기타")
                if m_type not in movement_groups:
                    movement_groups[m_type] = []
                movement_groups[m_type].append(movement)

            for m_type, m_list in movement_groups.items():
                display_name = movement_type_display.get(m_type.lower(), m_type)
                context_parts.append(f"\n**{display_name}**")
                for m in m_list:
                    player = m.get("player_name", "알 수 없음")
                    team = m.get("team_name") or m.get("team_code", "")
                    date = m.get("date", "")
                    remarks = m.get("remarks", "")

                    line = f"- {player}"
                    if team:
                        line += f" → {team}"
                    if date:
                        line += f" ({date})"
                    context_parts.append(line)
                    if remarks:
                        context_parts.append(f"  {remarks}")
        else:
            context_parts.append(f"\n{year}년 선수 이동 기록을 찾을 수 없습니다.")

        return "\n".join(context_parts)