#!/usr/bin/env python3
"""
2025 시즌 전체 팀 Coach 분석 배치 캐싱 스크립트.

모든 KBO 10개 팀에 대해 Coach 분석을 미리 생성하고 캐시에 저장합니다.
이를 통해 사용자 요청 시 빠른 응답이 가능합니다.

Usage:
    cd AI
    source .venv/bin/activate
    python scripts/batch_coach_cache.py
"""

import asyncio
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_connection_pool, get_coach_llm_generator
from app.tools.database_query import DatabaseQueryTool
from app.core.prompts import COACH_PROMPT_V2
from app.core.coach_validator import parse_coach_response
from app.routers.coach import _remove_duplicate_json_start

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# KBO 10개 팀
TEAMS = ["KIA", "LG", "SSG", "NC", "두산", "KT", "롯데", "삼성", "한화", "키움"]
SEASON_YEAR = 2025
PROMPT_VERSION = "v3_prompt"
MODEL_NAME = "solar-pro-3"


def _safe_float(value, default: float = 0.0) -> float:
    """None-safe float conversion for formatting."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """None-safe int conversion for formatting."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _format_coach_context_for_batch(tool_results: dict) -> str:
    """배치 작업용 컨텍스트 포맷 (간소화)."""
    parts = []

    team_summary = tool_results.get("team_summary", {})
    advanced = tool_results.get("advanced_metrics", {})

    team_name = team_summary.get("team_name") or advanced.get(
        "team_name", "알 수 없는 팀"
    )
    year = team_summary.get("year") or advanced.get("year", SEASON_YEAR)

    parts.append(f"## {team_name} {year}시즌 분석 데이터\n")

    # 핵심 지표
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
        parts.append("")

    # 불펜 부담 지표
    fatigue = advanced.get("fatigue_index", {})
    league_avg = advanced.get("league_averages", {})

    if fatigue:
        parts.append("### 불펜 부담 지표")
        parts.append(f"- **팀 불펜 비중**: {fatigue.get('bullpen_share', 'N/A')}")
        parts.append(
            f"- **리그 평균 불펜 비중**: {league_avg.get('bullpen_share', 'N/A')}"
        )
        try:
            team_share = float(fatigue.get("bullpen_share", "0").replace("%", ""))
            league_share = float(league_avg.get("bullpen_share", "0").replace("%", ""))
            diff = team_share - league_share
            parts.append(f"- **리그 평균 대비 차이**: {diff:+.1f}%p")
        except (ValueError, TypeError):
            pass
        parts.append("")

    # 주요 타자
    top_batters = team_summary.get("top_batters", [])
    if top_batters:
        parts.append("### 주요 타자 (OPS 상위)")
        parts.append("| 선수 | 역할 | 타율 | OPS | HR | RBI |")
        parts.append("|------|------|------|-----|-----|-----|")
        role_kr = {"regular": "주전", "platoon": "준주전", "bench": "벤치"}
        for b in top_batters[:8]:
            role = role_kr.get(b.get("role", ""), "")
            parts.append(
                f"| {b.get('player_name', 'N/A')} | {role} | "
                f"{_safe_float(b.get('avg')):.3f} | {_safe_float(b.get('ops')):.3f} | "
                f"{_safe_int(b.get('home_runs'))} | {_safe_int(b.get('rbi'))} |"
            )
        parts.append("")

    # 주요 투수
    top_pitchers = team_summary.get("top_pitchers", [])
    if top_pitchers:
        parts.append("### 주요 투수 (ERA 상위)")
        parts.append("| 선수 | 역할 | ERA | WHIP | 승 | 패 | SV | HLD |")
        parts.append("|------|------|-----|------|-----|-----|-----|-----|")
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
                f"{_safe_int(p.get('saves'))} | {_safe_int(p.get('holds'))} |"
            )
        parts.append("")

    return "\n".join(parts)


async def generate_and_cache_team(pool, team_id: str, year: int) -> dict:
    """단일 팀 Coach 분석 생성 및 캐싱."""

    # 1. 캐시 키 생성
    cache_components = [team_id, str(year), "", "", PROMPT_VERSION]
    cache_key = hashlib.sha256("|".join(cache_components).encode()).hexdigest()

    logger.info(f"[{team_id}] Cache key: {cache_key[:16]}...")

    # 2. 기존 캐시 확인
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT status, response_json FROM coach_analysis_cache WHERE cache_key = %s",
            (cache_key,),
        ).fetchone()

        if row:
            status, cached_json = row
            if status == "COMPLETED" and cached_json:
                logger.info(f"[{team_id}] Already cached, skipping")
                return {
                    "team": team_id,
                    "status": "skipped",
                    "reason": "already cached",
                }

        # PENDING 상태로 삽입/업데이트
        conn.execute(
            """
            INSERT INTO coach_analysis_cache (cache_key, team_id, year, prompt_version, model_name, status)
            VALUES (%s, %s, %s, %s, %s, 'PENDING')
            ON CONFLICT (cache_key) DO UPDATE SET status = 'PENDING', updated_at = now()
        """,
            (cache_key, team_id, year, PROMPT_VERSION, MODEL_NAME),
        )
        conn.commit()

    # 3. 도구 실행 (팀 요약 + 고급 지표)
    logger.info(f"[{team_id}] Fetching team data...")

    tool_results = {}
    with pool.connection() as conn:
        db_query = DatabaseQueryTool(conn)
        tool_results["team_summary"] = db_query.get_team_summary(team_id, year)
        tool_results["advanced_metrics"] = db_query.get_team_advanced_metrics(
            team_id, year
        )

    # 데이터 확인
    team_summary = tool_results.get("team_summary", {})
    if not team_summary.get("found"):
        logger.warning(f"[{team_id}] No data found for {year}")
        with pool.connection() as conn:
            conn.execute(
                """
                UPDATE coach_analysis_cache
                SET status = 'FAILED', error_message = %s, updated_at = now()
                WHERE cache_key = %s
            """,
                (f"No data found for {team_id} {year}", cache_key),
            )
            conn.commit()
        return {"team": team_id, "status": "failed", "reason": "no data"}

    # 4. 컨텍스트 포맷팅
    context = _format_coach_context_for_batch(tool_results)

    # 5. LLM 호출
    logger.info(f"[{team_id}] Calling LLM...")

    coach_llm = get_coach_llm_generator()
    question = f"{team_id} {year}시즌 종합 분석"
    messages = [
        {
            "role": "user",
            "content": COACH_PROMPT_V2.format(question=question, context=context),
        }
    ]

    try:
        chunks = []
        async for chunk in coach_llm(messages):
            chunks.append(chunk)
        full_response = "".join(chunks)

        logger.info(f"[{team_id}] LLM response received: {len(full_response)} chars")

    except Exception as e:
        logger.error(f"[{team_id}] LLM call failed: {e}")
        with pool.connection() as conn:
            conn.execute(
                """
                UPDATE coach_analysis_cache
                SET status = 'FAILED', error_message = %s, updated_at = now()
                WHERE cache_key = %s
            """,
                (str(e), cache_key),
            )
            conn.commit()
        return {"team": team_id, "status": "failed", "reason": f"LLM error: {e}"}

    # 6. 중복 JSON 시작 제거 + 파싱 및 저장
    full_response = _remove_duplicate_json_start(full_response)
    response, error = parse_coach_response(full_response)

    with pool.connection() as conn:
        if response:
            response_json = json.dumps(response.model_dump(), ensure_ascii=False)
            conn.execute(
                """
                UPDATE coach_analysis_cache
                SET status = 'COMPLETED', response_json = %s, error_message = NULL, updated_at = now()
                WHERE cache_key = %s
            """,
                (response_json, cache_key),
            )
            conn.commit()
            logger.info(f"[{team_id}] Successfully cached")
            return {
                "team": team_id,
                "status": "success",
                "headline": response.headline[:50],
            }
        else:
            error_msg = error or "JSON parsing failed"
            conn.execute(
                """
                UPDATE coach_analysis_cache
                SET status = 'FAILED', error_message = %s, updated_at = now()
                WHERE cache_key = %s
            """,
                (error_msg, cache_key),
            )
            conn.commit()
            logger.warning(f"[{team_id}] Parsing failed: {error_msg}")
            return {"team": team_id, "status": "failed", "reason": error_msg}


async def main():
    """메인 배치 루프."""
    print("=" * 60)
    print(f"Coach 배치 캐싱 시작: {SEASON_YEAR} 시즌")
    print(f"대상 팀: {len(TEAMS)}개 - {', '.join(TEAMS)}")
    print(f"프롬프트 버전: {PROMPT_VERSION}")
    print("=" * 60)

    pool = get_connection_pool()

    results = {"success": 0, "skipped": 0, "failed": 0, "details": []}
    start_time = datetime.now()

    for i, team in enumerate(TEAMS, 1):
        print(f"\n[{i}/{len(TEAMS)}] {team} 처리 중...")
        try:
            result = await generate_and_cache_team(pool, team, SEASON_YEAR)
            results[result["status"]] = results.get(result["status"], 0) + 1
            results["details"].append(result)

            status_icon = {"success": "✓", "skipped": "⊘", "failed": "✗"}.get(
                result["status"], "?"
            )
            reason = result.get("headline") or result.get("reason", "")
            print(f"  {status_icon} {result['status']}: {reason}")

        except Exception as e:
            results["failed"] += 1
            results["details"].append(
                {"team": team, "status": "failed", "reason": str(e)}
            )
            print(f"  ✗ FAILED: {e}")

        # Rate limiting (OpenRouter API 보호)
        if i < len(TEAMS):
            await asyncio.sleep(3)  # 3초 대기

    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 60)
    print("완료 요약")
    print("=" * 60)
    print(f"성공: {results['success']}개")
    print(f"스킵: {results['skipped']}개 (이미 캐시됨)")
    print(f"실패: {results['failed']}개")
    print(f"소요 시간: {elapsed:.1f}초")

    if results["failed"] > 0:
        print("\n실패한 팀:")
        for detail in results["details"]:
            if detail["status"] == "failed":
                print(f"  - {detail['team']}: {detail.get('reason', 'unknown')}")

    print("\n캐시 확인 명령:")
    print(
        f'  psql $SUPABASE_DB_URL -c "SELECT team_id, status, updated_at FROM coach_analysis_cache WHERE year = {SEASON_YEAR} ORDER BY team_id"'
    )


if __name__ == "__main__":
    asyncio.run(main())
