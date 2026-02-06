#!/usr/bin/env python3
"""
만료된 Coach 분석 캐시 정리 스크립트.

7일 이상 오래된 캐시 항목을 삭제합니다.
주 1회 cron job으로 실행 권장.

Usage:
    cd AI
    source .venv/bin/activate
    python scripts/cleanup_expired_cache.py

    # Dry-run 모드 (삭제 없이 확인만)
    python scripts/cleanup_expired_cache.py --dry-run

    # 커스텀 TTL (예: 3일)
    python scripts/cleanup_expired_cache.py --days 3
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_connection_pool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 14


def cleanup_expired_cache(retention_days: int, dry_run: bool = False) -> dict:
    """
    만료된 캐시 항목을 정리합니다.

    Args:
        retention_days: 보관 기간 (일)
        dry_run: True면 삭제 없이 대상만 확인

    Returns:
        정리 결과 요약
    """
    pool = get_connection_pool()

    with pool.connection() as conn:
        # 1. 삭제 대상 확인
        expired_rows = conn.execute(
            """
            SELECT cache_key, team_id, year, status, updated_at
            FROM coach_analysis_cache
            WHERE updated_at < now() - make_interval(days => %s)
            ORDER BY updated_at
            """,
            (retention_days,),
        ).fetchall()

        expired_count = len(expired_rows)

        if expired_count == 0:
            logger.info("삭제할 만료 캐시 없음")
            return {"deleted": 0, "dry_run": dry_run}

        logger.info(f"만료 캐시 {expired_count}개 발견 (>{retention_days}일)")

        # 상세 로그
        for row in expired_rows[:10]:  # 최대 10개만 출력
            cache_key, team_id, year, status, updated_at = row
            logger.info(
                f"  - {team_id} ({year}) | {status} | 마지막 업데이트: {updated_at}"
            )

        if expired_count > 10:
            logger.info(f"  ... 외 {expired_count - 10}개")

        # 2. 삭제 실행 (dry_run이 아닐 때만)
        if dry_run:
            logger.info("[DRY-RUN] 실제 삭제 없이 종료")
            return {"deleted": 0, "would_delete": expired_count, "dry_run": True}

        deleted = conn.execute(
            """
            DELETE FROM coach_analysis_cache
            WHERE updated_at < now() - make_interval(days => %s)
            """,
            (retention_days,),
        ).rowcount
        conn.commit()

        logger.info(f"삭제 완료: {deleted}개 항목")

        return {"deleted": deleted, "dry_run": False}


def main():
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
    args = parser.parse_args()

    print("=" * 50)
    print("Coach 캐시 정리")
    print("=" * 50)
    print(f"보관 기간: {args.days}일")
    print(f"모드: {'DRY-RUN (확인만)' if args.dry_run else '실제 삭제'}")
    print("=" * 50)

    start = datetime.now()
    result = cleanup_expired_cache(args.days, args.dry_run)
    elapsed = (datetime.now() - start).total_seconds()

    print("\n결과:")
    if args.dry_run:
        print(f"  삭제 예정: {result.get('would_delete', 0)}개")
    else:
        print(f"  삭제됨: {result['deleted']}개")
    print(f"  소요 시간: {elapsed:.2f}초")


if __name__ == "__main__":
    main()
