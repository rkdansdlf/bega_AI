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
from pydantic import BaseModel

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

logger = logging.getLogger(__name__)

# 빈 응답 시 재시도 횟수
MAX_RETRY_ON_EMPTY = 2


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


router = APIRouter(prefix="/coach", tags=["coach"])


# ============================================================
# Fast Path Helper Functions
# ============================================================


def _build_coach_query(team_name: str, focus: List[str]) -> str:
    """focus 영역에 따라 Coach 질문을 구성합니다."""
    focus_text = ", ".join(focus) if focus else "종합적인 전력"

    query = f"{team_name}의 {focus_text}에 대해 냉철하고 다각적인 분석을 수행해줘."

    if "batting" in focus or not focus:
        query += (
            " 팀의 타격 생산성(OPS, wRC+)과 주요 타자들의 최근 클러치 능력을 진단해줘."
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
    pool: ConnectionPool, team_code: str, year: int, focus: List[str]
) -> Dict[str, Any]:
    """
    Coach에 필요한 도구들을 병렬로 실행합니다.

    LLM 도구 계획 호출을 건너뛰고 focus에 따라 직접 도구를 호출합니다.

    [P1 Fix] 각 병렬 작업이 별도의 connection을 사용하도록 수정.
    psycopg connection은 스레드 세이프하지 않으므로 pool에서 별도 connection을 빌립니다.
    """
    loop = asyncio.get_event_loop()

    def get_team_summary_sync():
        """별도 connection으로 팀 요약 조회"""
        with pool.connection() as conn:
            db_query = DatabaseQueryTool(conn)
            return db_query.get_team_summary(team_code, year)

    def get_team_advanced_metrics_sync():
        """별도 connection으로 팀 고급 지표 조회"""
        with pool.connection() as conn:
            db_query = DatabaseQueryTool(conn)
            return db_query.get_team_advanced_metrics(team_code, year)

    def get_team_recent_form_sync():
        """별도 connection으로 최근 성적 조회"""
        with pool.connection() as conn:
            db_query = DatabaseQueryTool(conn)
            # recent_form이 focus에 있거나 focus가 없으면 기본 조회
            if "recent_form" in focus or not focus:
                return db_query.get_team_recent_form(team_name, year)
            return {}

    def get_team_matchup_stats_sync():
        """별도 connection으로 상대 전적 조회"""
        with pool.connection() as conn:
            db_query = DatabaseQueryTool(conn)
            if "matchup" in focus:
                return db_query.get_team_matchup_stats(team_name, year)
            return {}

    # 팀 이름이 필요하므로 먼저 코드로 이름을 변환하거나, 쿼리 도구 내부에서 처리하도록 해야 함.
    # DatabaseQueryTool의 메서드들은 team_name(str)을 인자로 받으므로, team_code가 아닌 team_name을 넘겨야 함.
    # 하지만 현재 함수 인자는 team_code임.
    # _execute_coach_tools_parallel 의 인자로 team_code가 들어오지만,
    # DatabaseQueryTool.get_team_summary 등은 team_name을 받음 (내부에서 get_team_code 호출).
    # 따라서 team_code를 그대로 넘겨도 get_team_code가 처리해줄 것임.
    # (DatabaseQueryTool.get_team_code는 입력값을 그대로 반환하거나 매핑된 값을 반환)
    # 안전을 위해 team_name 변수를 정의
    team_name = team_code  # DatabaseQueryTool handles code/name flexible

    # 병렬 실행 (각각 별도 connection 사용)
    tasks = [
        loop.run_in_executor(None, get_team_summary_sync),
        loop.run_in_executor(None, get_team_advanced_metrics_sync),
        loop.run_in_executor(None, get_team_recent_form_sync),
        loop.run_in_executor(None, get_team_matchup_stats_sync),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    tool_results = {
        "team_summary": (
            results[0]
            if not isinstance(results[0], Exception)
            else {"error": str(results[0])}
        ),
        "advanced_metrics": (
            results[1]
            if not isinstance(results[1], Exception)
            else {"error": str(results[1])}
        ),
        "recent_form": (
            results[2]
            if not isinstance(results[2], Exception)
            else {"error": str(results[2])}
        ),
        "matchup_stats": (
            results[3]
            if not isinstance(results[3], Exception)
            else {"error": str(results[3])}
        ),
    }

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


def _format_coach_context(
    tool_results: Dict[str, Any], focus: List[str], game_context: Optional[str] = None
) -> str:
    """
    Coach 전용 컨텍스트를 포맷합니다.

    리그 평균 비교, 불펜 부담 지표 등을 명확하게 표시합니다.

    Args:
        tool_results: 도구 실행 결과
        focus: 분석 초점 영역
        game_context: 특정 경기 분석 시 경기 정보 (optional)
    """
    parts = []

    # 경기별 분석 모드일 경우 안내 추가
    if game_context:
        parts.append("## ⚠️ 특정 경기 분석 모드")
        parts.append(
            "이 분석은 특정 경기에 대한 것입니다. 아래 팀 시즌 통계는 **참고용**입니다."
        )
        parts.append(f"**분석 대상**: {game_context}")
        parts.append("")
        parts.append("분석 시 다음에 집중하세요:")
        parts.append("- 해당 경기의 승패 요인")
        parts.append("- 경기에서 활약한/부진한 개별 선수")
        parts.append("- 시즌 통계는 맥락 제공용으로만 간략히 언급")
        parts.append("")

    team_summary = tool_results.get("team_summary", {})
    advanced = tool_results.get("advanced_metrics", {})

    team_name = team_summary.get("team_name") or advanced.get(
        "team_name", "알 수 없는 팀"
    )
    year = team_summary.get("year") or advanced.get("year", datetime.now().year)

    parts.append(f"## {team_name} {year}시즌 분석 데이터\n")

    # 1. 핵심 지표 요약 테이블
    if advanced.get("metrics"):
        parts.append("### 핵심 지표")
        parts.append("| 지표 | 팀 수치 | 리그 순위 |")
        parts.append("|------|---------|----------|")

        batting = advanced["metrics"].get("batting", {})
        pitching = advanced["metrics"].get("pitching", {})
        rankings = advanced.get("rankings", {})

        if batting.get("ops") is not None:
            parts.append(
                f"| 팀 OPS | {_safe_float(batting['ops']):.3f} | {rankings.get('batting_ops', 'N/A')} |"
            )
        if batting.get("avg") is not None:
            parts.append(
                f"| 팀 타율 | {_safe_float(batting['avg']):.3f} | {rankings.get('batting_avg', 'N/A')} |"
            )
        if pitching.get("avg_era") is not None:
            parts.append(
                f"| 팀 평균 ERA | {_safe_float(pitching['avg_era']):.2f} | {pitching.get('era_rank', 'N/A')} |"
            )
        if pitching.get("qs_rate"):
            parts.append(f"| QS 비율 | {pitching['qs_rate']} | - |")
        parts.append("")

    # 2. 불펜 과부하 지표 (Coach의 핵심 분석 포인트)
    fatigue = advanced.get("fatigue_index", {})
    league_avg = advanced.get("league_averages", {})

    if fatigue or "bullpen" in focus:
        parts.append("### 불펜 부담 지표 (핵심)")
        parts.append(f"- **팀 불펜 비중**: {fatigue.get('bullpen_share', 'N/A')}")
        parts.append(
            f"- **리그 평균 불펜 비중**: {league_avg.get('bullpen_share', 'N/A')}"
        )
        parts.append(f"- **불펜 부담 순위**: {fatigue.get('bullpen_load_rank', 'N/A')}")

        # 리그 평균 대비 차이 (원시 데이터만, LLM이 판단하도록)
        try:
            team_share = float(fatigue.get("bullpen_share", "0").replace("%", ""))
            league_share = float(league_avg.get("bullpen_share", "0").replace("%", ""))
            diff = team_share - league_share
            parts.append(f"- **리그 평균 대비 차이**: {diff:+.1f}%p")
        except (ValueError, TypeError):
            pass
        parts.append("")

    # 3. 주요 타자 정보 (역할 포함)
    top_batters = team_summary.get("top_batters", [])
    if top_batters and ("batting" in focus or not focus):
        parts.append("### 주요 타자 (OPS 상위)")
        parts.append("| 선수 | 역할 | 타율 | OBP | SLG | OPS | HR | RBI |")
        parts.append("|------|------|------|-----|-----|-----|-----|-----|")
        role_kr = {"regular": "주전", "platoon": "준주전", "bench": "벤치"}
        for b in top_batters[:8]:
            role = role_kr.get(b.get("role", ""), "")
            parts.append(
                f"| {b.get('player_name', 'N/A')} | {role} | "
                f"{_safe_float(b.get('avg')):.3f} | {_safe_float(b.get('obp')):.3f} | "
                f"{_safe_float(b.get('slg')):.3f} | {_safe_float(b.get('ops')):.3f} | "
                f"{_safe_int(b.get('home_runs'))} | {_safe_int(b.get('rbi'))} |"
            )
        parts.append("")

    # 4. 주요 투수 정보 (역할 포함)
    top_pitchers = team_summary.get("top_pitchers", [])
    if top_pitchers and ("starter" in focus or "bullpen" in focus or not focus):
        parts.append("### 주요 투수 (ERA 상위)")
        parts.append("| 선수 | 역할 | ERA | WHIP | 승 | 패 | SV | HLD | 이닝 |")
        parts.append("|------|------|-----|------|-----|-----|-----|-----|------|")
        role_kr = {
            "starter": "선발",
            "closer": "마무리",
            "setup": "셋업",
            "middle_reliever": "중계",
        }
        for p in top_pitchers[:8]:
            role = role_kr.get(p.get("role", ""), "")
            parts.append(
                f"| {p.get('player_name', 'N/A')} | {role} | "
                f"{_safe_float(p.get('era')):.2f} | {_safe_float(p.get('whip')):.2f} | "
                f"{_safe_int(p.get('wins'))} | {_safe_int(p.get('losses'))} | "
                f"{_safe_int(p.get('saves'))} | {_safe_int(p.get('holds'))} | "
                f"{_safe_float(p.get('innings_pitched')):.1f} |"
            )
        parts.append("")

    # 5. 리그 평균 참고 데이터
    if league_avg:
        parts.append("### 리그 평균 (참고)")
        if league_avg.get("era") is not None:
            parts.append(f"- 리그 평균 ERA: {_safe_float(league_avg['era']):.2f}")
        if league_avg.get("bullpen_share"):
            parts.append(f"- 리그 평균 불펜 비중: {league_avg['bullpen_share']}")
        parts.append("")

    return "\n".join(parts) + _format_extended_context(tool_results)


def _format_extended_context(tool_results: Dict[str, Any]) -> str:
    """추가된 데이터(최근 성적, 상대 전적)를 포맷팅합니다."""
    parts = []

    # 6. 최근 경기 성적 (New)
    recent = tool_results.get("recent_form", {})
    if recent.get("found"):
        parts.append("### 최근 10경기 성적")
        summary = recent.get("summary", {})
        parts.append(
            f"- **전적**: {summary.get('wins')}승 {summary.get('losses')}패 {summary.get('draws')}무 (승률 {summary.get('win_rate')})"
        )
        parts.append(f"- **득실마진**: {summary.get('run_diff'):+d}점")

        parts.append("| 날짜 | 상대 | 결과 | 스코어 | 득실 |")
        parts.append("|------|------|------|--------|------|")
        for g in recent.get("games", [])[:5]:  # 최근 5경기만 상세 표시
            parts.append(
                f"| {g['date']} | {g['opponent']} | {g['result']} | {g['score']} | {g['run_diff']:+d} |"
            )
        parts.append("")

    # 7. 상대 전적 (New)
    matchup = tool_results.get("matchup_stats", {})
    if matchup.get("found"):
        parts.append("### 주요 상대 전적 (승률순)")
        parts.append("| 상대팀 | 경기수 | 승 | 패 | 무 | 승률 |")
        parts.append("|--------|--------|----|----|----|------|")

        # 승률 높은 순 3팀, 낮은 순 3팀 표시 or 전체 표시
        # 여기서는 상위/하위 3개씩 보여주는 대신 전체 리스트 중 승률 순 정렬된거 상위 5개만 예시로
        matchups = matchup.get("matchups", {})
        sorted_opps = sorted(
            matchups.items(), key=lambda x: x[1]["win_rate"], reverse=True
        )

        for opp_name, data in sorted_opps:
            parts.append(
                f"| {opp_name} | {data['games']} | {data['wins']} | {data['losses']} | {data['draws']} | {data['win_rate']:.3f} |"
            )
        parts.append("")

    return "\n" + "\n".join(parts)


class AnalyzeRequest(BaseModel):
    team_id: str
    focus: List[str] = []  # 예: ["bullpen", "recent_form", "matchup"]
    game_id: Optional[str] = None
    question_override: Optional[str] = None


@router.post("/analyze")
async def analyze_team(
    payload: AnalyzeRequest,
    agent: BaseballStatisticsAgent = Depends(get_agent),
    _: None = Depends(rate_limit_dependency),
):
    """
    특정 팀에 대한 심층 분석을 요청합니다. 'The Coach' 페르소나가 적용됩니다.

    Fast Path 최적화:
    - 도구 계획 LLM 호출 생략 (직접 도구 호출)
    - 병렬 도구 실행
    - 1회의 LLM 호출만 사용 (답변 생성)

    스트리밍(SSE) 응답을 지원합니다.
    """
    from sse_starlette.sse import EventSourceResponse
    import psycopg
    import hashlib  # Added for cache key generation

    try:
        request_id = uuid.uuid4().hex[:8]
        team_name = agent._convert_team_id_to_name(payload.team_id)
        if payload.question_override:
            query = payload.question_override
        else:
            query = _build_coach_query(team_name, payload.focus)

        # 연도 결정 로직 (Pre-season 고려)
        now = datetime.now()
        target_year = now.year
        pre_season_notice = None

        # 1~3월은 시즌 시작 전이므로 작년 데이터 기준 분석
        if now.month <= 3:
            target_year = now.year - 1
            pre_season_notice = f"NOTICE: 현재 {now.year}년 시즌 개막 전이므로, {target_year}년 시즌 데이터를 바탕으로 분석합니다."

        year = target_year

        # Cache Key 생성
        # Key 구성: team_id + year + focus(sorted) + query_override(optional) + game_id(optional) + model_version
        cache_components = [
            payload.team_id,
            str(year),
            ",".join(sorted(payload.focus)),
            payload.question_override or "",
            payload.game_id or "",
            "v3_prompt",  # 프롬프트 버전 (변경 시 업데이트 필요)
        ]
        cache_key = hashlib.sha256("|".join(cache_components).encode()).hexdigest()

        logger.info(
            "[Coach Router] Analyzing for %s (%s, year=%d): %s... (CacheKey: %s)",
            team_name,
            request_id,
            year,
            query[:100],
            cache_key,
        )

        async def event_generator():
            try:
                total_start = perf_counter()
                # ============================================================
                # Phase 1: 데이터 수집 (병렬 도구 실행)
                # ============================================================
                phase1_start = perf_counter()
                yield {
                    "event": "status",
                    "data": json.dumps(
                        {"message": "상대팀 정보 몰래 캐는 중..."},
                        ensure_ascii=False,
                    ),
                }

                # [P1 Fix] DB 연결 풀 사용 - 병렬 실행 시 각 작업이 별도 connection 사용
                pool = get_connection_pool()

                # ============================================================
                # Phase 0: 캐시 확인 (Coach Caching) - Race Condition 방지
                # ============================================================
                # Atomic upsert: INSERT or get existing row in single query
                # This prevents race conditions where two requests both see "no cache"
                # 캐시 TTL 설정 (7일)
                CACHE_TTL_HOURS = 168

                cached_data = None
                should_compute = False
                with pool.connection() as conn:
                    row = conn.execute(
                        """
                        INSERT INTO coach_analysis_cache (cache_key, team_id, year, prompt_version, model_name, status)
                        VALUES (%s, %s, %s, %s, %s, 'PENDING')
                        ON CONFLICT (cache_key) DO UPDATE
                            SET cache_key = coach_analysis_cache.cache_key  -- no-op update to trigger RETURNING
                        RETURNING status, response_json, (xmax = 0) AS inserted,
                                  (updated_at > now() - interval '7 days') AS is_valid
                        """,
                        (cache_key, payload.team_id, year, "v3_prompt", "solar-pro-3"),
                    ).fetchone()
                    conn.commit()

                    if row:
                        status, cached_json, was_inserted, is_valid = row
                        if status == "COMPLETED" and cached_json and is_valid:
                            # 7일 이내 유효한 캐시
                            cached_data = cached_json
                            logger.info("[Coach] Cache HIT for %s", cache_key)
                        elif status == "COMPLETED" and cached_json and not is_valid:
                            # 캐시 만료 - PENDING으로 변경 후 재계산
                            conn.execute(
                                "UPDATE coach_analysis_cache SET status = 'PENDING', updated_at = now() WHERE cache_key = %s",
                                (cache_key,),
                            )
                            conn.commit()
                            should_compute = True
                            logger.info(
                                "[Coach] Cache EXPIRED (>%dh), recomputing for %s",
                                CACHE_TTL_HOURS,
                                cache_key,
                            )
                        elif was_inserted:
                            # We successfully inserted PENDING - we should compute
                            should_compute = True
                            logger.info(
                                "[Coach] Cache MISS, inserted PENDING for %s", cache_key
                            )
                        elif status == "PENDING":
                            # Another request is computing - proceed anyway (duplicate work but reliable)
                            # In production, could implement polling/waiting here
                            should_compute = True
                            logger.warning(
                                "[Coach] Cache PENDING by another request for %s, proceeding anyway",
                                cache_key,
                            )

                if cached_data:
                    # 레거시 캐시 데이터 정규화 (한글 status/area → 영어)
                    cached_data = _normalize_cached_response(cached_data)

                    yield {
                        "event": "status",
                        "data": json.dumps(
                            {"message": "저장된 분석 결과를 불러옵니다..."},
                            ensure_ascii=False,
                        ),
                    }
                    # 캐시된 JSON을 텍스트 스트림처럼 전송 (프론트엔드 호환성)
                    json_str = json.dumps(cached_data, ensure_ascii=False, indent=2)
                    yield {
                        "event": "message",
                        "data": json.dumps({"delta": json_str}, ensure_ascii=False),
                    }
                    yield {
                        "event": "meta",
                        "data": json.dumps(
                            {
                                "validation_status": "success",
                                "structured_response": cached_data,
                                "fast_path": True,
                                "cached": True,
                                "verified": True,
                            },
                            ensure_ascii=False,
                        ),
                    }
                    yield {"event": "done", "data": "[DONE]"}
                    return

                # 도구 실행 시작 알림
                yield {
                    "event": "tool_start",
                    "data": json.dumps(
                        {"tool": "get_team_summary"}, ensure_ascii=False
                    ),
                }
                yield {
                    "event": "tool_start",
                    "data": json.dumps(
                        {"tool": "get_team_advanced_metrics"}, ensure_ascii=False
                    ),
                }

                # 병렬 도구 실행 (각 도구가 별도 connection 사용)
                tool_results = await _execute_coach_tools_parallel(
                    pool, payload.team_id, year, payload.focus
                )
                phase1_end = perf_counter()
                logger.info(
                    "[Coach Timing] %s phase1_tools=%.3fs",
                    request_id,
                    phase1_end - phase1_start,
                )

                # 도구 결과 알림
                team_summary_result = tool_results.get("team_summary", {})
                advanced_metrics_result = tool_results.get("advanced_metrics", {})
                yield {
                    "event": "tool_result",
                    "data": json.dumps(
                        {
                            "tool": "get_team_summary",
                            "success": team_summary_result.get("found", False),
                            "message": "팀 요약 데이터 조회 완료",
                        },
                        ensure_ascii=False,
                    ),
                }
                yield {
                    "event": "tool_result",
                    "data": json.dumps(
                        {
                            "tool": "get_team_advanced_metrics",
                            "success": advanced_metrics_result.get("found", False),
                            "message": "팀 고급 지표 조회 완료",
                        },
                        ensure_ascii=False,
                    ),
                }

                # ============================================================
                # Phase 2: 컨텍스트 포맷팅 (경기별 분석 모드 지원)
                # ============================================================
                phase2_start = perf_counter()
                yield {
                    "event": "status",
                    "data": json.dumps(
                        {"message": "숫자들이랑 씨름하는 중..."}, ensure_ascii=False
                    ),
                }

                # question_override가 있으면 경기별 분석 모드로 처리
                game_context = (
                    payload.question_override if payload.question_override else None
                )

                # game_id가 있으면 경기 세부 정보 가져오기
                if payload.game_id:
                    with pool.connection() as conn:
                        db_query = DatabaseQueryTool(conn)
                        game_info = db_query.get_game_info(payload.game_id)
                        if game_info.get("found"):
                            game_summary_text = f"{game_info['date']} {game_info['home_team_name']} vs {game_info['away_team_name']} (@{game_info['stadium']})"
                            if game_info.get("home_score") is not None:
                                game_summary_text += f" [스코어 {game_info['home_score']}:{game_info['away_score']}]"

                            # 만약 기존 game_context(question_override)가 있다면 병합, 없으면 생성
                            if game_context:
                                game_context = f"{game_summary_text}\n(사용자 질문: {game_context})"
                            else:
                                game_context = game_summary_text

                context = _format_coach_context(
                    tool_results, payload.focus, game_context
                )

                # [Pre-season Notice] 컨텍스트에 추가
                if pre_season_notice:
                    context = f"## 중요 알림\n{pre_season_notice}\n\n" + context

                phase2_end = perf_counter()
                logger.info(
                    "[Coach Timing] %s phase2_context=%.3fs context_chars=%d",
                    request_id,
                    phase2_end - phase2_start,
                    len(context),
                )

                # ============================================================
                # [PATCH] 데이터 무결성 게이트 - LLM 호출 전 검증
                # ============================================================
                team_summary = tool_results.get("team_summary", {})
                advanced_metrics = tool_results.get("advanced_metrics", {})

                # 핵심 데이터가 없으면 LLM 호출 스킵
                has_batters = len(team_summary.get("top_batters", [])) > 0
                has_pitchers = len(team_summary.get("top_pitchers", [])) > 0
                has_metrics = bool(advanced_metrics.get("metrics"))

                if not has_batters and not has_pitchers and not has_metrics:
                    logger.warning("[Coach] Data validation failed - skipping LLM call")

                    # [Cache Fail] 데이터 부족으로 실패 처리
                    with pool.connection() as conn:
                        conn.execute(
                            "UPDATE coach_analysis_cache SET status = 'FAILED', error_message = %s, updated_at = now() WHERE cache_key = %s",
                            ("Data insufficient", cache_key),
                        )
                        conn.commit()

                    # 즉시 안전한 응답 반환
                    fallback_response = json.dumps(
                        {
                            "headline": f"{team_name} 데이터를 현재 확인할 수 없습니다",
                            "sentiment": "neutral",
                            "key_metrics": [],
                            "analysis": {
                                "strengths": [],
                                "weaknesses": [],
                                "risks": [],
                            },
                            "detailed_markdown": "## 데이터 부족\n\n현재 해당 팀의 시즌 데이터가 DB에 없거나 조회에 실패했습니다.",
                            "coach_note": "데이터가 확보되면 다시 분석을 요청해 주세요.",
                        },
                        ensure_ascii=False,
                    )
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {"delta": fallback_response}, ensure_ascii=False
                        ),
                    }
                    yield {
                        "event": "meta",
                        "data": json.dumps(
                            {
                                "validation_status": "data_insufficient",
                                "fast_path": True,
                                "llm_skipped": True,
                            },
                            ensure_ascii=False,
                        ),
                    }
                    yield {"event": "done", "data": "[DONE]"}
                    return

                # ============================================================
                # Phase 3: LLM 답변 생성 (1회 호출)
                # ============================================================
                phase3_start = perf_counter()
                yield {
                    "event": "status",
                    "data": json.dumps(
                        {"message": "AI 코치가 작전판에 열심히 낙서 중..."},
                        ensure_ascii=False,
                    ),
                }

                # [P2 Fix] COACH_PROMPT_V2 사용 (JSON 스키마 기반)
                coach_prompt = COACH_PROMPT_V2.format(question=query, context=context)
                messages = [{"role": "user", "content": coach_prompt}]
                logger.info(
                    "[Coach Timing] %s prompt_chars=%d",
                    request_id,
                    len(coach_prompt),
                )

                # [근본 해결책 #1] Coach 전용 LLM generator 사용 (설정 기반)
                # Gemini/OpenRouter 자동 선택 + 폴백 지원
                coach_llm = get_coach_llm_generator()

                # LLM 스트리밍 답변 생성 + 전체 응답 수집
                # max_tokens는 config.py의 coach_max_output_tokens 사용 (기본 2000)
                # Use list + join for O(n) performance instead of string concatenation O(n²)
                response_chunks = []
                first_chunk_at = None
                async for chunk in coach_llm(messages):
                    if first_chunk_at is None:
                        first_chunk_at = perf_counter()
                        logger.info(
                            "[Coach Timing] %s ttft=%.3fs",
                            request_id,
                            first_chunk_at - phase3_start,
                        )
                    response_chunks.append(chunk)
                    yield {
                        "event": "message",
                        "data": json.dumps({"delta": chunk}, ensure_ascii=False),
                    }
                full_response = "".join(response_chunks)
                phase3_end = perf_counter()
                logger.info(
                    "[Coach Timing] %s phase3_stream=%.3fs response_chars=%d",
                    request_id,
                    phase3_end - phase3_start,
                    len(full_response),
                )

                # [Phase 7 Fix] LLM 스트리밍 중복 출력 제거
                full_response = _remove_duplicate_json_start(full_response)

                # ============================================================
                # Phase 4: 응답 검증 및 메타데이터
                # ============================================================
                phase4_start = perf_counter()
                meta_payload = {
                    "tool_calls": [
                        {
                            "tool_name": "get_team_summary",
                            "parameters": {"team_name": payload.team_id, "year": year},
                        },
                        {
                            "tool_name": "get_team_advanced_metrics",
                            "parameters": {"team_name": payload.team_id, "year": year},
                        },
                    ],
                    "verified": True,
                    "data_sources": [
                        "player_season_batting",
                        "player_season_pitching",
                        "teams",
                    ],
                    "fast_path": True,  # Fast Path 사용 표시
                }

                # [P2 Fix] 응답 완료 후 JSON 파싱 시도
                parsed_response, parse_error = parse_coach_response(full_response)

                # [Cache Update] 결과 저장 or 실패 처리
                with pool.connection() as conn:
                    if parsed_response:
                        conn.execute(
                            "UPDATE coach_analysis_cache SET status = 'COMPLETED', response_json = %s, updated_at = now() WHERE cache_key = %s",
                            (
                                json.dumps(
                                    parsed_response.model_dump(), ensure_ascii=False
                                ),
                                cache_key,
                            ),
                        )
                        conn.commit()

                        meta_payload["validation_status"] = "success"
                        meta_payload["structured_response"] = (
                            parsed_response.model_dump()
                        )
                        logger.info(
                            "[Coach Router] Response validated and CACHED successfully"
                        )
                    else:
                        # 파싱 실패 - 캐시를 FAILED로 마킹 (재요청 시 재시도하도록)
                        error_reason = parse_error or "Validation failed"
                        conn.execute(
                            "UPDATE coach_analysis_cache SET status = 'FAILED', error_message = %s, updated_at = now() WHERE cache_key = %s",
                            (error_reason, cache_key),
                        )
                        conn.commit()

                        # 클라이언트에는 fallback 응답 제공 (UX 유지)
                        fallback = _create_fallback_response(
                            "JSON 파싱 실패", full_response
                        )
                        meta_payload["validation_status"] = "fallback"
                        meta_payload["structured_response"] = fallback.model_dump()
                        logger.warning(
                            "[Coach Router] Validation failed, serving fallback response"
                        )

                phase4_end = perf_counter()
                logger.info(
                    "[Coach Timing] %s phase4_validate=%.3fs",
                    request_id,
                    phase4_end - phase4_start,
                )
                total_end = perf_counter()
                logger.info(
                    "[Coach Timing] %s total=%.3fs",
                    request_id,
                    total_end - total_start,
                )

                yield {
                    "event": "meta",
                    "data": json.dumps(meta_payload, ensure_ascii=False),
                }

                yield {"event": "done", "data": "[DONE]"}

            except Exception as e:
                logger.error(f"[Coach Streaming Error] {e}")
                # [Cache Fail] 에러 발생 시 실패 처리
                try:
                    pool = get_connection_pool()
                    with pool.connection() as conn:
                        conn.execute(
                            "UPDATE coach_analysis_cache SET status = 'FAILED', error_message = %s, updated_at = now() WHERE cache_key = %s",
                            (str(e), cache_key),
                        )
                        conn.commit()
                except Exception as db_e:
                    logger.error(
                        f"[Coach Cache Error] Failed to update failure status: {db_e}"
                    )

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
        team_name = agent._convert_team_id_to_name(payload.team_id)

        if payload.question_override:
            query = payload.question_override
        else:
            query = _build_coach_query(team_name, payload.focus)

        logger.info(f"[Coach Router Legacy] Analyzing for {team_name}")

        context_data = {"persona": "coach", "team_id": payload.team_id}

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
