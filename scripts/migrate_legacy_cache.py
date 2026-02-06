#!/usr/bin/env python3
"""
레거시 Coach 캐시 데이터 마이그레이션 스크립트.

기존 캐시의 한글 status/area 값을 영어로 변환합니다.
이를 통해 읽기 시점의 정규화 오버헤드를 제거합니다.

변환 매핑:
- status: "주의" → "warning", "양호" → "good", "위험" → "danger"
- area: "불펜" → "bullpen", "선발" → "starter", "타격" → "batting"

Usage:
    cd AI
    source .venv/bin/activate
    python scripts/migrate_legacy_cache.py

    # Dry-run 모드
    python scripts/migrate_legacy_cache.py --dry-run
"""

import argparse
import json
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

# 정규화 매핑
STATUS_MAP = {
    "주의": "warning",
    "양호": "good",
    "위험": "danger",
}

AREA_MAP = {
    "불펜": "bullpen",
    "선발": "starter",
    "타격": "batting",
    "수비": "defense",
    "주루": "baserunning",
}


def normalize_response_json(data: dict) -> tuple[dict, bool]:
    """
    캐시된 JSON 데이터를 정규화합니다.

    Returns:
        (정규화된 데이터, 변경 여부)
    """
    changed = False

    # key_metrics 배열의 status/area 변환
    if "key_metrics" in data and isinstance(data["key_metrics"], list):
        for metric in data["key_metrics"]:
            if "status" in metric and metric["status"] in STATUS_MAP:
                metric["status"] = STATUS_MAP[metric["status"]]
                changed = True
            if "area" in metric and metric["area"] in AREA_MAP:
                metric["area"] = AREA_MAP[metric["area"]]
                changed = True

    # analysis 객체의 status 변환 (있는 경우)
    if "analysis" in data and isinstance(data["analysis"], dict):
        for key in ["strengths", "weaknesses", "risks"]:
            if key in data["analysis"] and isinstance(data["analysis"][key], list):
                for item in data["analysis"][key]:
                    if isinstance(item, dict):
                        if "status" in item and item["status"] in STATUS_MAP:
                            item["status"] = STATUS_MAP[item["status"]]
                            changed = True
                        if "area" in item and item["area"] in AREA_MAP:
                            item["area"] = AREA_MAP[item["area"]]
                            changed = True

    return data, changed


def migrate_legacy_cache(dry_run: bool = False) -> dict:
    """
    레거시 캐시 데이터를 마이그레이션합니다.

    Args:
        dry_run: True면 변환 없이 대상만 확인

    Returns:
        마이그레이션 결과 요약
    """
    pool = get_connection_pool()

    stats = {
        "total": 0,
        "legacy_found": 0,
        "migrated": 0,
        "errors": 0,
        "dry_run": dry_run,
    }

    with pool.connection() as conn:
        # 1. 모든 COMPLETED 캐시 조회
        rows = conn.execute(
            """
            SELECT cache_key, team_id, year, response_json
            FROM coach_analysis_cache
            WHERE status = 'COMPLETED' AND response_json IS NOT NULL
            """
        ).fetchall()

        stats["total"] = len(rows)
        logger.info(f"전체 캐시 항목: {stats['total']}개")

        for row in rows:
            cache_key, team_id, year, response_json = row

            try:
                # JSON 파싱
                if isinstance(response_json, str):
                    data = json.loads(response_json)
                else:
                    data = response_json

                # 정규화
                normalized, changed = normalize_response_json(data)

                if changed:
                    stats["legacy_found"] += 1
                    logger.info(f"레거시 발견: {team_id} ({year})")

                    if not dry_run:
                        # DB 업데이트
                        conn.execute(
                            """
                            UPDATE coach_analysis_cache
                            SET response_json = %s, updated_at = now()
                            WHERE cache_key = %s
                            """,
                            (json.dumps(normalized, ensure_ascii=False), cache_key),
                        )
                        stats["migrated"] += 1

            except Exception as e:
                logger.error(f"오류 ({team_id} {year}): {e}")
                stats["errors"] += 1

        if not dry_run:
            conn.commit()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Coach 캐시 레거시 마이그레이션")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="변환 없이 대상만 확인",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Coach 캐시 레거시 마이그레이션")
    print("=" * 60)
    print(f"모드: {'DRY-RUN (확인만)' if args.dry_run else '실제 마이그레이션'}")
    print()
    print("변환 매핑:")
    print("  status: 주의→warning, 양호→good, 위험→danger")
    print("  area: 불펜→bullpen, 선발→starter, 타격→batting")
    print("=" * 60)

    start = datetime.now()
    result = migrate_legacy_cache(args.dry_run)
    elapsed = (datetime.now() - start).total_seconds()

    print("\n결과:")
    print(f"  전체 캐시: {result['total']}개")
    print(f"  레거시 발견: {result['legacy_found']}개")
    if args.dry_run:
        print(f"  마이그레이션 예정: {result['legacy_found']}개")
    else:
        print(f"  마이그레이션 완료: {result['migrated']}개")
    print(f"  오류: {result['errors']}개")
    print(f"  소요 시간: {elapsed:.2f}초")

    if result["legacy_found"] == 0:
        print("\n레거시 데이터 없음 - 마이그레이션 불필요")


if __name__ == "__main__":
    main()
