"""
실패한 청크만 재시도하는 스크립트

마이그레이션 중 실패한 청크 ID들을 migration_progress.json에서 읽어서
다시 마이그레이션을 시도합니다.

사용법:
    python scripts/retry_failed_chunks.py --service-account-key path/to/key.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List


def retry_failed_chunks(
    service_account_key_path: str,
    source_db_url: str,
    dry_run: bool = False,
    skip_storage: bool = False,
):
    """실패한 청크들을 재시도합니다."""

    # 진행 파일 읽기
    progress_file = Path(__file__).parent / "migration_progress.json"

    if not progress_file.exists():
        print("❌ migration_progress.json 파일을 찾을 수 없습니다.")
        print("마이그레이션을 먼저 실행하세요.")
        return

    with open(progress_file, "r") as f:
        progress = json.load(f)

    failed_ids = progress.get("failed_ids", [])

    if not failed_ids:
        print("✓ 실패한 청크가 없습니다!")
        return

    print(f"\n{'=' * 60}")
    print(f"실패한 청크 재시도")
    print(f"{'=' * 60}")
    print(f"실패한 청크 개수: {len(failed_ids)}")
    print(f"{'=' * 60}\n")

    if dry_run:
        print("⚠️  Dry Run 모드: 실제 데이터는 변경되지 않습니다.\n")

    try:
        from migrate_to_firebase import FirebaseMigration
    except ImportError as exc:
        raise RuntimeError(
            "migrate_to_firebase 모듈을 찾을 수 없습니다. 스크립트 경로와 의존성을 확인하세요."
        ) from exc

    # Firebase 마이그레이션 객체 생성
    migration = FirebaseMigration(
        service_account_key_path,
        source_db_url,
        batch_size=1,  # 한 번에 하나씩 처리
        dry_run=dry_run,
        skip_storage=skip_storage,
    )

    # 실패한 청크들을 다시 시도
    success_count = 0
    still_failed_ids = []

    print("재시도 중...\n")

    for chunk_id in failed_ids:
        print(f"청크 ID {chunk_id} 재시도 중...", end=" ")

        # Source DB에서 청크 데이터 가져오기
        try:
            with migration.pg_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        id, season_year, season_id, league_type_code,
                        team_id, player_id, source_table, source_row_id,
                        title, content, embedding, meta, created_at
                    FROM rag_chunks
                    WHERE id = %s
                """,
                    (chunk_id,),
                )

                row = cur.fetchone()

                if not row:
                    print("❌ 데이터 없음")
                    still_failed_ids.append(chunk_id)
                    continue

                # dict로 변환
                chunk = {
                    "id": row[0],
                    "season_year": row[1],
                    "season_id": row[2],
                    "league_type_code": row[3],
                    "team_id": row[4],
                    "player_id": row[5],
                    "source_table": row[6],
                    "source_row_id": row[7],
                    "title": row[8],
                    "content": row[9],
                    "embedding": row[10],
                    "meta": row[11],
                    "created_at": row[12],
                }

                # 마이그레이션 시도
                if migration.migrate_chunk(chunk):
                    print("✓ 성공")
                    success_count += 1
                else:
                    print("❌ 실패")
                    still_failed_ids.append(chunk_id)

        except Exception as e:
            print(f"❌ 오류: {e}")
            still_failed_ids.append(chunk_id)

    # 결과 출력
    print(f"\n{'=' * 60}")
    print(f"재시도 완료!")
    print(f"{'=' * 60}")
    print(f"성공: {success_count}/{len(failed_ids)}")
    print(f"여전히 실패: {len(still_failed_ids)}")

    if still_failed_ids:
        print(f"\n여전히 실패한 청크 ID:")
        print(still_failed_ids[:20])  # 처음 20개만 표시
        if len(still_failed_ids) > 20:
            print(f"... 외 {len(still_failed_ids) - 20}개")

    print(f"{'=' * 60}\n")

    # progress 파일 업데이트
    if not dry_run:
        progress["failed_ids"] = still_failed_ids
        progress["migrated_count"] += success_count

        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        print(f"✓ 진행 상태 업데이트 완료")

        if not still_failed_ids:
            print(f"✓ 모든 청크가 성공적으로 마이그레이션되었습니다!")
            print(f"진행 파일을 삭제해도 됩니다: rm {progress_file}")

    migration.close()


def main():
    parser = argparse.ArgumentParser(description="실패한 청크 재시도")
    parser.add_argument(
        "--service-account-key",
        required=True,
        help="Firebase 서비스 계정 키 JSON 파일 경로",
    )
    parser.add_argument(
        "--source-db-url",
        default="",
        help="Source PostgreSQL 연결 URL (기본값: 환경변수 POSTGRES_DB_URL)",
    )
    parser.add_argument(
        "--supabase-url",
        default="",
        help="[Deprecated] --source-db-url 사용 권장",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run 모드 (실제 데이터 변경 없음)"
    )
    parser.add_argument(
        "--skip-storage",
        action="store_true",
        help="Firebase Storage 업로드 스킵 (Firestore에만 저장)",
    )

    args = parser.parse_args()

    source_db_url = args.source_db_url.strip()
    if not source_db_url and args.supabase_url.strip():
        print("[WARN] --supabase-url is deprecated. Use --source-db-url instead.")
        source_db_url = args.supabase_url.strip()

    if not source_db_url:
        source_db_url = os.getenv("POSTGRES_DB_URL", "").strip()

    if not source_db_url:
        legacy_env = os.getenv("SUPABASE_DB_URL", "").strip()
        if legacy_env:
            print("[WARN] SUPABASE_DB_URL is deprecated. Use POSTGRES_DB_URL instead.")
            source_db_url = legacy_env

    if not source_db_url:
        print("오류: Source DB URL이 필요합니다. --source-db-url 또는 환경변수 POSTGRES_DB_URL을 설정하세요.")
        sys.exit(1)

    retry_failed_chunks(
        service_account_key_path=args.service_account_key,
        source_db_url=source_db_url,
        dry_run=args.dry_run,
        skip_storage=args.skip_storage,
    )


if __name__ == "__main__":
    main()
