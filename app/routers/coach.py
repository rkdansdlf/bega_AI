"""
'The Coach' 기능과 관련된 API 엔드포인트를 정의합니다.

Fast Path 최적화:
- 도구 계획 LLM 호출을 건너뛰고 focus 영역에 따라 직접 도구 호출
- 병렬 도구 실행으로 대기 시간 단축
- Coach 전용 컨텍스트 포맷팅
"""

import logging
import json
import asyncio
import uuid
import hashlib
from time import perf_counter
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, model_validator

from psycopg_pool import ConnectionPool

from ..deps import (
    get_agent,
    get_db_connection,
    get_connection_pool,
    get_coach_llm_generator,
)
from ..agents.baseball_agent import BaseballStatisticsAgent
from ..core.prompts import COACH_PROMPT_V2
from ..core.coach_validator import (
    parse_coach_response,
    CoachResponse,
    _create_fallback_response,
)
from ..core.ratelimit import rate_limit_dependency
from ..tools.database_query import DatabaseQueryTool
from ..tools.team_code_resolver import CANONICAL_CODES, TeamCodeResolver

logger = logging.getLogger(__name__)

# 빈 응답 시 재시도 횟수
MAX_RETRY_ON_EMPTY = 2
COACH_CACHE_SCHEMA_VERSION = "v2"
COACH_CACHE_PROMPT_VERSION = "v4_dual"
COACH_MODEL_NAME = "solar-pro-3"
COACH_YEAR_MIN = 1982


def _normalize_cached_response(cached_data: dict) -> dict:
    """
    레거시 캐시 데이터를 현재 스키마에 맞게 정규화합니다.

    CoachResponse 검증기를 통과시켜 자동으로 변환:
    - status: "주의" → "warning", "양호" → "good", "위험" → "danger"
    - area: "불펜" → "bullpen", "선발" → "starter", "타격" → "batting"
    - coach_note: 150자 초과 시 자동 truncate

    Args:
        cached_data: 캐시에서 읽은 원본 JSON 데이터

    Returns:
        정규화된 데이터 (실패 시 원본 반환)
    """
    if not cached_data:
        return cached_data

    try:
        # CoachResponse 검증기를 통과시켜 자동 정규화
        response = CoachResponse(**cached_data)
        normalized = response.model_dump()
        logger.debug("[Coach Cache] Normalized legacy data")
        return normalized
    except Exception as e:
        logger.warning(f"[Coach Cache] Failed to normalize legacy data: {e}")
        return cached_data


def _parse_explicit_year(value: Any) -> Optional[int]:
    """명시적 year 필드(예: season_year)를 정수로 파싱합니다."""
    if value is None or isinstance(value, bool):
        return None
    try:
        year = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if 1000 <= year <= 9999:
        return year
    return None


def _parse_year_from_date_like(value: Any) -> Optional[int]:
    """YYYY-MM-DD 또는 YYYY... 형태 문자열에서 연도를 추출합니다."""
    if value is None:
        return None
    text = str(value).strip()
    if len(text) < 4 or not text[:4].isdigit():
        return None
    return int(text[:4])


def _is_valid_analysis_year(year: int) -> bool:
    return COACH_YEAR_MIN <= year <= datetime.now().year + 1


def _resolve_year_from_season_id(pool: ConnectionPool, season_id: Any) -> Optional[int]:
    if season_id is None:
        return None
    try:
        normalized_season_id = int(str(season_id).strip())
    except (TypeError, ValueError):
        return None

    try:
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT season_year FROM kbo_seasons WHERE season_id = %s LIMIT 1",
                (normalized_season_id,),
            ).fetchone()
        if not row:
            return None
        season_year = int(row[0])
        return season_year
    except Exception as exc:
        logger.warning("[Coach Router] Failed to resolve season_id=%s: %s", season_id, exc)
        return None


def _resolve_year_from_game_context(
    pool: ConnectionPool, game_id: Optional[str], game_date: Any
) -> Optional[int]:
    explicit_game_year = _parse_year_from_date_like(game_date)
    if explicit_game_year is not None:
        return explicit_game_year

    if game_id:
        try:
            with pool.connection() as conn:
                row = conn.execute(
                    "SELECT game_date FROM game WHERE game_id = %s LIMIT 1",
                    (game_id,),
                ).fetchone()
            if row and row[0]:
                game_date_obj = row[0]
                if hasattr(game_date_obj, "year"):
                    return int(game_date_obj.year)
                return _parse_year_from_date_like(game_date_obj)
        except Exception as exc:
            logger.warning("[Coach Router] Failed to resolve game_id=%s: %s", game_id, exc)

        fallback_year = _parse_year_from_date_like(game_id)
        if fallback_year is not None:
            return fallback_year

    return None


def _resolve_target_year(payload: "AnalyzeRequest", pool: ConnectionPool) -> tuple[int, str]:
    league_context = payload.league_context or {}

    if "season_year" in league_context:
        parsed_year = _parse_explicit_year(league_context.get("season_year"))
        if parsed_year is None or not _is_valid_analysis_year(parsed_year):
            raise HTTPException(
                status_code=400,
                detail="invalid_season_year_for_analysis",
            )
        return parsed_year, "league_context.season_year"

    season_id = league_context.get("season")
    season_year = _resolve_year_from_season_id(pool, season_id)
    if season_year is not None and _is_valid_analysis_year(season_year):
        return season_year, "league_context.season->kbo_seasons"

    game_year = _resolve_year_from_game_context(
        pool,
        payload.game_id,
        league_context.get("game_date"),
    )
    if game_year is not None and _is_valid_analysis_year(game_year):
        return game_year, "game_date"

    raise HTTPException(
        status_code=400,
        detail="unable_to_resolve_analysis_year",
    )


router = APIRouter(prefix="/coach", tags=["coach"])


# ============================================================
# Fast Path Helper Functions
# ============================================================


def _build_coach_query(
    team_name: str,
    focus: List[str],
    opponent_name: Optional[str] = None,
    league_context: Optional[Dict[str, Any]] = None,
) -> str:
    """focus 영역에 따라 Coach 질문을 구성합니다."""
    focus_text = ", ".join(focus) if focus else "종합적인 전력"

    if opponent_name:
        query = f"{team_name}와 {opponent_name}의 {focus_text}에 대해 냉철하고 다각적인 비교 분석을 수행해줘."
    else:
        query = (
            f"{team_name}의 {focus_text}에 대해 냉철하고 다각적인 분석을 수행해줘."
        )

    # 리그 컨텍스트 반영
    if league_context:
        season = league_context.get("season")
        league_type = league_context.get("league_type")
        if league_type == "POST":
            round_name = league_context.get("round", "포스트시즌")
            game_no = league_context.get("game_no")
            query += f" 특히 {season}년 {round_name} {game_no}차전임을 감안하여 분석해줘."
        elif league_type == "REGULAR":
            home_ctx = league_context.get("home", {})
            away_ctx = league_context.get("away", {})
            if home_ctx and away_ctx:
                home_rank = home_ctx.get("rank")
                away_rank = away_ctx.get("rank")
                if home_rank is not None and away_rank is not None:
                    rank_diff = abs(int(home_rank) - int(away_rank))
                    if rank_diff <= 2:
                        query += " 두 팀의 순위 경쟁이 치열한 상황이야."

    if "batting" in focus or not focus:
        if opponent_name:
            query += (
                " 양 팀의 타격 생산성(OPS, wRC+)과 주요 타자들의 최근 클러치 능력을 진단해줘."
            )
        else:
            query += (
                " 타격 생산성(OPS, wRC+)과 주요 타자들의 최근 클러치 능력을 진단해줘."
            )

    if "bullpen" in focus:
        query += " 불펜진의 하이 레버리지 상황 처리 능력과 과부하 지표를 분석해줘."

    if "recent_form" in focus or not focus:
        query += " 최근 10경기 승패 트렌드와 득실점 마진을 보고 팀의 상승세/하락세를 진단해줘."

    if "starter" in focus:
        query += " 선발 로테이션의 이닝 소화력과 QS 비율, 구속 변화를 분석해줘."

    if "matchup" in focus:
        query += " 주요 라이벌 팀들과의 상대 전적(승률, 득실 등)을 비교 분석해줘."

    return query


async def _execute_coach_tools_parallel(
    pool: ConnectionPool,
    home_team_id: str,
    year: int,
    focus: List[str],
    away_team_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Coach에 필요한 도구들을 병렬로 실행합니다.
    홈팀과 원정팀 데이터를 모두 조회합니다.
    """
    loop = asyncio.get_event_loop()

    def get_team_data(team_code: str):
        """특정 팀의 모든 데이터 조회"""
        results = {}
        with pool.connection() as conn:
            db_query = DatabaseQueryTool(conn)
            results["summary"] = db_query.get_team_summary(team_code, year)
            results["advanced"] = db_query.get_team_advanced_metrics(team_code, year)
            if "recent_form" in focus or not focus:
                results["recent"] = db_query.get_team_recent_form(team_code, year)
            if "matchup" in focus and away_team_id:
                # 상대 전적은 홈팀 기준 한번만 조회해도 됨
                pass
        return results

    def get_matchup_stats_sync(team1: str, team2: str):
        with pool.connection() as conn:
            from app.tools.game_query import GameQueryTool
            game_query = GameQueryTool(conn)
            return game_query.get_head_to_head(team1, team2, year)

    # 병렬 실행 태스크 준비
    tasks = []
    
    # 1. 홈팀 데이터
    tasks.append(loop.run_in_executor(None, get_team_data, home_team_id))
    
    # 2. 원정팀 데이터 (있을 경우)
    if away_team_id:
        tasks.append(loop.run_in_executor(None, get_team_data, away_team_id))
        
    # 3. 상대 전적 (Matchup focus일 경우)
    if "matchup" in focus and away_team_id:
        tasks.append(loop.run_in_executor(None, get_matchup_stats_sync, home_team_id, away_team_id))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    tool_results = {
        "home": {},
        "away": {},
        "matchup": {},
        "error": None
    }

    # 홈팀 결과 처리
    if isinstance(results[0], Exception):
        tool_results["error"] = str(results[0])
        tool_results["home"] = {"error": str(results[0])}
    else:
        tool_results["home"] = results[0]

    # 원정팀 결과 처리
    if away_team_id:
        if isinstance(results[1], Exception):
            tool_results["away"] = {"error": str(results[1])}
        else:
            tool_results["away"] = results[1]
            
        # 상대 전적 처리
        if "matchup" in focus:
            if len(tasks) > 2 and isinstance(results[2], Exception):
                tool_results["matchup"] = {"error": str(results[2])}
            elif len(tasks) > 2:
                tool_results["matchup"] = results[2]

    # 레거시 구조 호환성 유지 (단일 팀 분석 요청 시)
    if not away_team_id:
        tool_results["team_summary"] = tool_results["home"].get("summary", {})
        tool_results["advanced_metrics"] = tool_results["home"].get("advanced", {})
        tool_results["recent_form"] = tool_results["home"].get("recent", {})

    return tool_results


def _safe_float(value: Any, default: float = 0.0) -> float:
    """None-safe float conversion for formatting."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """None-safe int conversion for formatting."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _remove_duplicate_json_start(text: str) -> str:
    """
    LLM 스트리밍 중 발생하는 JSON 시작 부분 중복을 제거합니다.

    일부 LLM은 스트리밍 중 "content restart" 현상으로 동일한 필드를
    두 번 출력하는 경우가 있습니다. 이 함수는 중복을 감지하고 제거합니다.

    예시:
        입력: '{"headline": "A",\\n"headline": "A", "sentiment": ...'
        출력: '{"headline": "A", "sentiment": ...'

    Args:
        text: LLM의 원시 출력 텍스트

    Returns:
        중복이 제거된 텍스트
    """
    import re

    if not text or "{" not in text:
        return text

    # headline 필드 패턴 찾기
    headline_pattern = r'"headline"\s*:\s*"[^"]*"'
    matches = list(re.finditer(headline_pattern, text))

    if len(matches) < 2:
        return text

    # 두 개 이상의 headline 필드가 있으면 중복
    first_match = matches[0]
    second_match = matches[1]

    # 두 headline 값이 동일한지 확인
    first_value = text[first_match.start() : first_match.end()]
    second_value = text[second_match.start() : second_match.end()]

    if first_value == second_value:
        # 중복 발견 - 두 번째 headline 이후 내용만 유지
        logger.warning(
            "[Coach] Duplicate JSON start detected, removing first occurrence"
        )

        # { + 두 번째 headline부터의 내용
        brace_pos = text.index("{")
        clean_text = text[brace_pos : brace_pos + 1]  # '{'

        # 두 번째 headline 이후 내용
        after_second = text[second_match.start() :]
        clean_text += after_second

        return clean_text

    return text


def _format_team_stats(team_data: Dict[str, Any], team_role: str = "Home") -> str:
    """단일 팀 통계 포맷팅 헬퍼"""
    parts = []
    
    summary = team_data.get("summary", {})
    advanced = team_data.get("advanced", {})
    team_name = summary.get("team_name", "Unknown")
    
    parts.append(f"### [{team_role}] {team_name}")
    
    # 핵심 지표
    if advanced.get("metrics"):
        batting = advanced["metrics"].get("batting", {})
        pitching = advanced["metrics"].get("pitching", {})
        rankings = advanced.get("rankings", {})
        
        parts.append("| 지표 | 수치 | 순위 |")
        parts.append("|------|------|------|")
        if batting.get("ops"):
            parts.append(f"| OPS | {_safe_float(batting['ops']):.3f} | {rankings.get('batting_ops', '-')}|")
        if pitching.get("avg_era"):
            parts.append(f"| ERA | {_safe_float(pitching['avg_era']):.2f} | {pitching.get('era_rank', '-')}|")
        parts.append("")

    # 불펜
    fatigue = advanced.get("fatigue_index", {})
    if fatigue:
        parts.append(f"- **불펜 비중**: {fatigue.get('bullpen_share', '-')}")
        parts.append(f"- **피로도 순위**: {fatigue.get('bullpen_load_rank', '-')}")
        parts.append("")

    # 주요 선수 (간략화)
    top_batters = summary.get("top_batters", [])[:3]
    if top_batters:
        parts.append("**주요 타자**:")
        for b in top_batters:
             parts.append(f"- {b['player_name']}: OPS {_safe_float(b.get('ops')):.3f}, {b.get('home_runs')}HR")
    
    top_pitchers = summary.get("top_pitchers", [])[:3]
    if top_pitchers:
        parts.append("**주요 투수**:")
        for p in top_pitchers:
            parts.append(f"- {p['player_name']}: ERA {_safe_float(p.get('era')):.2f}, {p.get('wins')}승")
    
    # 최근 폼 — DB schema: summary={wins,losses,draws,run_diff}, games=[{result:"Win"/"Loss"/"Draw", score:"5:3", run_diff, date, opponent}]
    recent = team_data.get("recent", {})
    if recent and recent.get("found"):
        parts.append("**최근 경기 흐름**:")
        r_summary = recent.get("summary", {})
        r_games = recent.get("games", [])
        wins = r_summary.get("wins", 0)
        losses = r_summary.get("losses", 0)
        draws = r_summary.get("draws", 0)
        parts.append(f"- 최근 {len(r_games)}경기: {wins}승 {losses}패{f' {draws}무' if draws else ''}")
        run_diff = r_summary.get("run_diff")
        if run_diff is not None:
            parts.append(f"- 득실 마진: {'+' if run_diff >= 0 else ''}{run_diff}")
        win_rate = r_summary.get("win_rate")
        if win_rate is not None:
            parts.append(f"- 승률: {win_rate:.3f}")
        parts.append("")

    parts.append("")
    return "\n".join(parts)


def _format_coach_context(
    tool_results: Dict[str, Any],
    focus: List[str],
    game_context: Optional[str] = None,
    league_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Coach 전용 컨텍스트를 포맷합니다.
    듀얼 팀 데이터 지원.
    """
    parts = []

    # 1. 리그/경기 컨텍스트
    if league_context:
        season = league_context.get("season")
        league_type = league_context.get("league_type")
        parts.append(f"## 🏟️ {season} 시즌 컨텍스트")
        
        if league_type == "POST":
            parts.append(f"**{league_context.get('round')} {league_context.get('game_no')}차전**")
        else:
            home = league_context.get("home", {})
            away = league_context.get("away", {})
            parts.append(f"- **Home**: {home.get('rank')}위 ({home.get('gamesBehind')} GB)")
            parts.append(f"- **Away**: {away.get('rank')}위 ({away.get('gamesBehind')} GB)")
        parts.append("")

    # 2. 경기 별 모드 안내
    if game_context:
        parts.append("## ⚠️ 특정 경기 분석 모드")
        parts.append(f"**분석 대상**: {game_context}")
        parts.append("")

    # 3. 팀별 데이터
    if tool_results.get("home"):
        parts.append(_format_team_stats(tool_results["home"], "Home"))
        
    if tool_results.get("away"):
        parts.append(_format_team_stats(tool_results["away"], "Away"))

    # 4. 상대 전적
    matchup = tool_results.get("matchup", {})
    if matchup and matchup.get("games"):
        parts.append("### ⚔️ 맞대결 전적")
        summary = matchup.get("summary", {})
        t1 = matchup.get("team1", "팀1")
        t2 = matchup.get("team2", "팀2")
        parts.append(
            f"- {t1} {summary.get('team1_wins', 0)}승 / "
            f"{t2} {summary.get('team2_wins', 0)}승 / "
            f"{summary.get('draws', 0)}무"
        )
        parts.append("| 날짜 | 스코어 | 결과 |")
        parts.append("|------|--------|------|")
        for g in matchup.get("games", [])[:3]:
            game_date = g.get("game_date", "")
            if hasattr(game_date, "strftime"):
                game_date = game_date.strftime("%Y-%m-%d")
            score = f"{g.get('home_score', 0)}:{g.get('away_score', 0)}"
            result_val = g.get("game_result", "")
            parts.append(f"| {game_date} | {score} | {result_val} |")
        parts.append("")

    return "\n".join(parts)


class AnalyzeRequest(BaseModel):
    team_id: Optional[str] = None  # deprecated — use home_team_id
    home_team_id: Optional[str] = None
    away_team_id: Optional[str] = None
    league_context: Optional[Dict[str, Any]] = None
    focus: List[str] = []
    game_id: Optional[str] = None
    question_override: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def backfill_home_team_id(cls, values: Any) -> Any:
        """team_id만 보내는 기존 호출을 home_team_id로 매핑"""
        if isinstance(values, dict):
            if not values.get("home_team_id") and values.get("team_id"):
                values["home_team_id"] = values["team_id"]
        return values


@router.post("/analyze")
async def analyze_team(
    payload: AnalyzeRequest,
    agent: BaseballStatisticsAgent = Depends(get_agent),
    _: None = Depends(rate_limit_dependency),
):
    """
    특정 팀(들)에 대한 심층 분석을 요청합니다. 'The Coach' 페르소나가 적용됩니다.
    """
    from sse_starlette.sse import EventSourceResponse

    # 하위 호환성은 model_validator에서 처리됨
    if not payload.home_team_id:
        raise HTTPException(status_code=400, detail="home_team_id 또는 team_id가 필요합니다.")

    try:
        request_id = uuid.uuid4().hex[:8]
        pool = get_connection_pool()
        team_resolver = TeamCodeResolver()

        home_team_canonical = team_resolver.resolve_canonical(payload.home_team_id)
        away_team_canonical = (
            team_resolver.resolve_canonical(payload.away_team_id)
            if payload.away_team_id
            else None
        )
        if home_team_canonical not in CANONICAL_CODES:
            raise HTTPException(
                status_code=400,
                detail="unsupported_team_for_regular_analysis",
            )
        if away_team_canonical and away_team_canonical not in CANONICAL_CODES:
            raise HTTPException(
                status_code=400,
                detail="unsupported_team_for_regular_analysis",
            )

        home_name = agent._convert_team_id_to_name(payload.home_team_id)
        away_name = agent._convert_team_id_to_name(payload.away_team_id) if payload.away_team_id else None
        
        if payload.question_override:
            query = payload.question_override
        else:
            query = _build_coach_query(
                home_name, 
                payload.focus, 
                opponent_name=away_name,
                league_context=payload.league_context
            )

        year, resolve_source = _resolve_target_year(payload, pool)
        if not _is_valid_analysis_year(year):
            raise HTTPException(status_code=400, detail="analysis_year_out_of_range")

        # Cache Key 생성
        game_type = str((payload.league_context or {}).get("league_type") or "UNKNOWN").upper()
        cache_key_payload = {
            "schema": COACH_CACHE_SCHEMA_VERSION,
            "prompt_version": COACH_CACHE_PROMPT_VERSION,
            "team_code_canonical": home_team_canonical,
            "away_team_code_canonical": away_team_canonical or "",
            "year": year,
            "game_type": game_type,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_key_payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

        logger.info(
            "[Coach Router] Analyzing %s vs %s (year=%d): %s... (CacheKey: %s) input_season=%s resolved_year=%d resolve_source=%s",
            home_name,
            away_name or "Single",
            year,
            query[:100],
            cache_key,
            (payload.league_context or {}).get("season"),
            year,
            resolve_source,
        )

        async def event_generator():
            try:
                total_start = perf_counter()
                
                # Phase 1: 시작
                yield {
                    "event": "status",
                    "data": json.dumps({"message": "양 팀 전력 분석 중..."}, ensure_ascii=False),
                }
                # Phase 0: 캐시 확인
                CACHE_TTL_HOURS = 168
                cached_data = None
                
                with pool.connection() as conn:
                    row = conn.execute(
                        """
                        INSERT INTO coach_analysis_cache (cache_key, team_id, year, prompt_version, model_name, status)
                        VALUES (%s, %s, %s, %s, %s, 'PENDING')
                        ON CONFLICT (cache_key) DO UPDATE
                            SET cache_key = coach_analysis_cache.cache_key
                        RETURNING status, response_json, (xmax = 0) AS inserted,
                                  (updated_at > now() - interval '7 days') AS is_valid
                        """,
                        (
                            cache_key,
                            home_team_canonical,
                            year,
                            COACH_CACHE_PROMPT_VERSION,
                            COACH_MODEL_NAME,
                        ),
                    ).fetchone()
                    conn.commit()

                    if row:
                        status, cached_json, was_inserted, is_valid = row
                        if status == "COMPLETED" and cached_json and is_valid:
                            cached_data = cached_json
                            logger.info("[Coach] Cache HIT for %s", cache_key)
                        elif status == "COMPLETED" and cached_json and not is_valid:
                            conn.execute(
                                "UPDATE coach_analysis_cache SET status = 'PENDING', updated_at = now() WHERE cache_key = %s",
                                (cache_key,),
                            )
                            conn.commit()
                            logger.info("[Coach] Cache EXPIRED, recomputing")

                if cached_data:
                    cached_data = _normalize_cached_response(cached_data)
                    yield {
                        "event": "status",
                        "data": json.dumps({"message": "분석 데이터를 불러옵니다..."}, ensure_ascii=False),
                    }
                    json_str = json.dumps(cached_data, ensure_ascii=False, indent=2)
                    yield {
                        "event": "message",
                        "data": json.dumps({"delta": json_str}, ensure_ascii=False),
                    }
                    yield {
                        "event": "meta",
                        "data": json.dumps({
                            "validation_status": "success",
                            "structured_response": cached_data,
                            "fast_path": True,
                            "cached": True,
                        }, ensure_ascii=False),
                    }
                    yield {"event": "done", "data": "[DONE]"}
                    return

                # 도구 실행
                yield {
                    "event": "tool_start",
                    "data": json.dumps({"tool": "parallel_fetch_team_data"}, ensure_ascii=False),
                }

                tool_results = await _execute_coach_tools_parallel(
                    pool, payload.home_team_id, year, payload.focus, payload.away_team_id
                )
                
                yield {
                    "event": "tool_result",
                    "data": json.dumps({
                        "tool": "parallel_fetch_team_data",
                        "success": True,
                        "message": "데이터 조회 완료"
                    }, ensure_ascii=False),
                }

                # Phase 2: 컨텍스트 포맷팅
                game_context = payload.question_override if payload.question_override else None
                # Game info fetching can be added here if needed, consistent with tool_results usage

                context = _format_coach_context(
                    tool_results, 
                    payload.focus, 
                    game_context,
                    payload.league_context
                )

                # 데이터 무결성 검사 (간소화)
                # 홈팀 데이터가 충분한지 확인
                home_data = tool_results.get("home", {})
                has_home_data = bool(home_data.get("summary")) or bool(home_data.get("advanced"))
                
                if not has_home_data:
                    logger.warning("[Coach] Data validation failed - skipping LLM call")
                    with pool.connection() as conn:
                        conn.execute(
                            "UPDATE coach_analysis_cache SET status = 'FAILED', error_message = %s, updated_at = now() WHERE cache_key = %s",
                            ("Data insufficient", cache_key),
                        )
                        conn.commit()
                        
                    fallback_response = json.dumps({
                         "headline": f"{home_name} 데이터를 확인할 수 없습니다",
                         "sentiment": "neutral",
                         "key_metrics": [],
                         "analysis": {"strengths": [], "weaknesses": [], "risks": []},
                         "detailed_markdown": "## 데이터 부족\n\n데이터를 조회할 수 없습니다.",
                         "coach_note": "잠시 후 다시 시도해주세요."
                    }, ensure_ascii=False)
                    
                    yield {"event": "message", "data": json.dumps({"delta": fallback_response}, ensure_ascii=False)}
                    yield {"event": "done", "data": "[DONE]"}
                    return

                # Phase 3: LLM 호출
                yield {
                    "event": "status",
                    "data": json.dumps({"message": "AI 코치가 분석 리포트 작성 중..."}, ensure_ascii=False),
                }

                coach_prompt = COACH_PROMPT_V2.format(question=query, context=context)
                messages = [{"role": "user", "content": coach_prompt}]
                
                coach_llm = get_coach_llm_generator()
                response_chunks = []
                
                async for chunk in coach_llm(messages):
                    response_chunks.append(chunk)
                    yield {
                        "event": "message",
                        "data": json.dumps({"delta": chunk}, ensure_ascii=False),
                    }
                full_response = "".join(response_chunks)
                full_response = _remove_duplicate_json_start(full_response)

                # Phase 4: 검증 및 저장
                parsed_response, parse_error = parse_coach_response(full_response)
                
                with pool.connection() as conn:
                    if parsed_response:
                        conn.execute(
                            "UPDATE coach_analysis_cache SET status = 'COMPLETED', response_json = %s, updated_at = now() WHERE cache_key = %s",
                            (json.dumps(parsed_response.model_dump(), ensure_ascii=False), cache_key),
                        )
                    else:
                        conn.execute(
                            "UPDATE coach_analysis_cache SET status = 'FAILED', error_message = %s, updated_at = now() WHERE cache_key = %s",
                            (parse_error or "Validation failed", cache_key),
                        )
                    conn.commit()
                
                meta_payload = {
                     "verified": True,
                     "fast_path": True,
                     "validation_status": "success" if parsed_response else "fallback"
                }
                if parsed_response:
                    meta_payload["structured_response"] = parsed_response.model_dump()
                
                yield {
                    "event": "meta",
                    "data": json.dumps(meta_payload, ensure_ascii=False),
                }

                yield {"event": "done", "data": "[DONE]"}

            except Exception as e:
                logger.error(f"[Coach Streaming Error] {e}")
                # Cache fail logic
                try:
                    pool = get_connection_pool()
                    with pool.connection() as conn:
                        conn.execute(
                            "UPDATE coach_analysis_cache SET status = 'FAILED', error_message = %s, updated_at = now() WHERE cache_key = %s",
                            (str(e), cache_key),
                        )
                        conn.commit()
                except:
                    pass
                    
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)}, ensure_ascii=False),
                }

        return EventSourceResponse(
            event_generator(),
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"[Coach Router] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Legacy endpoint (기존 방식 유지, 필요 시 사용)
# ============================================================


@router.post("/analyze-legacy")
async def analyze_team_legacy(
    payload: AnalyzeRequest,
    agent: BaseballStatisticsAgent = Depends(get_agent),
    _: None = Depends(rate_limit_dependency),
):
    """
    기존 방식의 Coach 분석 (전체 에이전트 파이프라인 사용).
    Fast Path에 문제가 있을 경우 대안으로 사용.
    """
    from sse_starlette.sse import EventSourceResponse

    try:
        primary_team_id = payload.home_team_id or payload.team_id
        if not primary_team_id:
            raise HTTPException(status_code=400, detail="home_team_id 또는 team_id가 필요합니다.")

        team_name = agent._convert_team_id_to_name(primary_team_id)

        if payload.question_override:
            query = payload.question_override
        else:
            query = _build_coach_query(team_name, payload.focus)

        logger.info(f"[Coach Router Legacy] Analyzing for {team_name}")

        context_data = {"persona": "coach", "team_id": primary_team_id}

        async def event_generator():
            try:
                async for event in agent.process_query_stream(
                    query, context=context_data
                ):
                    if event["type"] == "status":
                        yield {
                            "event": "status",
                            "data": json.dumps(
                                {"message": event["message"]}, ensure_ascii=False
                            ),
                        }
                    elif event["type"] == "tool_start":
                        yield {
                            "event": "tool_start",
                            "data": json.dumps(
                                {"tool": event["tool"]}, ensure_ascii=False
                            ),
                        }
                    elif event["type"] == "tool_result":
                        yield {
                            "event": "tool_result",
                            "data": json.dumps(
                                {
                                    "tool": event["tool"],
                                    "success": event["success"],
                                    "message": event["message"],
                                },
                                ensure_ascii=False,
                            ),
                        }
                    elif event["type"] == "answer_chunk":
                        yield {
                            "event": "message",
                            "data": json.dumps(
                                {"delta": event["content"]}, ensure_ascii=False
                            ),
                        }
                    elif event["type"] == "metadata":
                        meta_payload = {
                            "tool_calls": [
                                tc.to_dict() for tc in event["data"]["tool_calls"]
                            ],
                            "verified": event["data"]["verified"],
                            "data_sources": event["data"]["data_sources"],
                        }
                        yield {
                            "event": "meta",
                            "data": json.dumps(meta_payload, ensure_ascii=False),
                        }

                yield {"event": "done", "data": "[DONE]"}
            except Exception as e:
                logger.error(f"[Coach Legacy Streaming Error] {e}")
                import traceback

                logger.error(traceback.format_exc())
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)}, ensure_ascii=False),
                }

        return EventSourceResponse(
            event_generator(),
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"[Coach Router Legacy] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
