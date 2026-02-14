#!/usr/bin/env python3
"""
만료된 Coach 분석 캐시 정리 스크립트.

기본 정책:
- 7일 이상 지난 캐시를 정리
- 기본 안전모드로 전역 삭제를 차단
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_connection_pool
from app.tools.team_code_resolver import CANONICAL_CODES, TeamCodeResolver

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 7


def parse_years_csv(raw_years: Optional[str]) -> Optional[List[int]]:
    if raw_years is None:
        return None
    years = []
    for token in str(raw_years).split(","):
        value = token.strip()
        if not value:
            continue
        years.append(int(value))
    deduped = sorted(set(years))
    return deduped


def parse_teams_csv(raw_teams: Optional[str], resolver: TeamCodeResolver) -> Optional[List[str]]:
    if raw_teams is None:
        return None

    teams: List[str] = []
    for token in str(raw_teams).split(","):
        value = token.strip()
        if not value:
            continue
        canonical = resolver.resolve_canonical(value)
        if canonical in CANONICAL_CODES and canonical not in teams:
            teams.append(canonical)
        else:
            logger.warning("Ignoring unsupported team input for cleanup: %s -> %s", value, canonical)

    if not teams:
        raise ValueError("no valid canonical teams were resolved from --teams")
    return teams


def build_where_clause(
    retention_days: int,
    years: Optional[List[int]],
    teams: Optional[List[str]],
) -> tuple[str, List[Any]]:
    clauses = ["updated_at < now() - make_interval(days => %s)"]
    params: List[Any] = [retention_days]

    if years:
        clauses.append("year = ANY(%s)")
        params.append(years)

    if teams:
        clauses.append("UPPER(team_id) = ANY(%s)")
        params.append(teams)

    where_clause = " AND ".join(clauses)
    return where_clause, params


def cleanup_expired_cache(
    retention_days: int,
    dry_run: bool = False,
    years: Optional[List[int]] = None,
    teams: Optional[List[str]] = None,
    allow_global: bool = False,
) -> dict:
    """
    만료된 캐시 항목을 정리합니다.

    Args:
        retention_days: 보관 기간 (일)
        dry_run: True면 삭제 없이 대상만 확인
        years: 정리 대상 연도 필터
        teams: 정리 대상 팀 코드 필터(캐노니컬)
        allow_global: True면 전역 정리 허용

    Returns:
        정리 결과 요약
    """
    if not allow_global and not years and not teams:
        raise ValueError(
            "global cleanup is blocked by default; provide --years/--teams or pass --allow-global"
        )

    pool = get_connection_pool()
    where_clause, where_params = build_where_clause(retention_days, years, teams)
    scope = {
        "retention_days": retention_days,
        "years": years or [],
        "teams": teams or [],
        "allow_global": allow_global,
    }

    with pool.connection() as conn:
        select_sql = f"""
            SELECT cache_key, team_id, year, status, updated_at
            FROM coach_analysis_cache
            WHERE {where_clause}
            ORDER BY updated_at
        """
        expired_rows = conn.execute(select_sql, tuple(where_params)).fetchall()
        expired_count = len(expired_rows)

        if expired_count == 0:
            logger.info("삭제할 만료 캐시 없음")
            return {"deleted": 0, "would_delete": 0, "dry_run": dry_run, "scope": scope}

        logger.info(
            "만료 캐시 %d개 발견 (>%d일, years=%s, teams=%s)",
            expired_count,
            retention_days,
            years or "ALL",
            teams or "ALL",
        )

        for row in expired_rows[:10]:
            cache_key, team_id, year, status, updated_at = row
            logger.info(
                "  - %s (%s) | %s | 마지막 업데이트: %s",
                team_id,
                year,
                status,
                updated_at,
            )
        if expired_count > 10:
            logger.info("  ... 외 %d개", expired_count - 10)

        if dry_run:
            logger.info("[DRY-RUN] 실제 삭제 없이 종료")
            return {
                "deleted": 0,
                "would_delete": expired_count,
                "dry_run": True,
                "scope": scope,
            }

        delete_sql = f"""
            DELETE FROM coach_analysis_cache
            WHERE {where_clause}
        """
        deleted = conn.execute(delete_sql, tuple(where_params)).rowcount
        conn.commit()
        logger.info("삭제 완료: %d개 항목", deleted)
        return {
            "deleted": deleted,
            "would_delete": expired_count,
            "dry_run": False,
            "scope": scope,
        }


def write_output(path: str, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Coach 캐시 정리 스크립트")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help=f"보관 기간 (일, 기본값: {DEFAULT_RETENTION_DAYS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="삭제 없이 대상만 확인",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="정리 대상 연도 목록 (예: 2025 또는 2024,2025)",
    )
    parser.add_argument(
        "--teams",
        default=None,
        help="정리 대상 팀 목록 (예: SSG,DB 또는 legacy 별칭 입력 가능)",
    )
    parser.add_argument(
        "--allow-global",
        action="store_true",
        help="years/teams 없이 전역 정리를 허용합니다.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="결과 JSON 저장 경로",
    )
    args = parser.parse_args()

    resolver = TeamCodeResolver()
    years = parse_years_csv(args.years)
    teams = parse_teams_csv(args.teams, resolver)

    print("=" * 50)
    print("Coach 캐시 정리")
    print("=" * 50)
    print(f"보관 기간: {args.days}일")
    print(f"연도 필터: {years if years else 'ALL'}")
    print(f"팀 필터: {teams if teams else 'ALL'}")
    print(f"전역 허용: {args.allow_global}")
    print(f"모드: {'DRY-RUN (확인만)' if args.dry_run else '실제 삭제'}")
    print("=" * 50)

    start = datetime.now()
    try:
        result = cleanup_expired_cache(
            retention_days=args.days,
            dry_run=args.dry_run,
            years=years,
            teams=teams,
            allow_global=args.allow_global,
        )
        status = "PASS"
        fatal_error = None
        exit_code = 0
    except Exception as exc:
        result = {"deleted": 0, "would_delete": 0, "dry_run": args.dry_run}
        status = "FAIL"
        fatal_error = str(exc)
        exit_code = 1
        logger.error("cleanup failed: %s", exc)

    elapsed = (datetime.now() - start).total_seconds()
    payload = {
        "status": status,
        "fatal_error": fatal_error,
        "retention_days": args.days,
        "years": years or [],
        "teams": teams or [],
        "allow_global": args.allow_global,
        "dry_run": args.dry_run,
        "result": result,
        "runtime_seconds": round(elapsed, 3),
        "generated_at": datetime.now().isoformat(),
    }

    print("\n결과:")
    if args.dry_run:
        print(f"  삭제 예정: {result.get('would_delete', 0)}개")
    else:
        print(f"  삭제됨: {result.get('deleted', 0)}개")
    if fatal_error:
        print(f"  오류: {fatal_error}")
    print(f"  소요 시간: {elapsed:.2f}초")

    if args.output:
        write_output(args.output, payload)
        print(f"  결과 JSON: {args.output}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
