"""
Supabase rag_chunks 테이블을 Firebase로 마이그레이션하는 스크립트

주요 기능:
1. Supabase에서 rag_chunks 데이터를 배치로 읽기
2. Firebase Firestore에 임베딩 벡터 저장
3. Firebase Storage에 텍스트 컨텐츠 저장 (옵션)
4. Supabase에 Firebase 참조 추가
5. 중단 후 재개 가능 (progress 추적)

사용법:
    python scripts/migrate_to_firebase.py --service-account-key path/to/key.json

옵션:
    --batch-size: 배치 크기 (기본: 100)
    --dry-run: 실제 마이그레이션 없이 테스트
    --skip-storage: Storage 업로드 스킵 (Firestore만 사용)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore, storage
import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm


class FirebaseMigration:
    def __init__(
        self,
        service_account_key_path: str,
        supabase_db_url: str,
        batch_size: int = 100,
        dry_run: bool = False,
        skip_storage: bool = False
    ):
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.skip_storage = skip_storage

        # Firebase 초기화
        cred = credentials.Certificate(service_account_key_path)
        if not firebase_admin._apps:
            # 서비스 계정 키에서 프로젝트 ID 읽기
            with open(service_account_key_path, 'r') as f:
                service_account_data = json.load(f)
                project_id = service_account_data.get('project_id')
                storage_bucket = f"{project_id}.firebasestorage.app"

            firebase_admin.initialize_app(cred, {
                'storageBucket': storage_bucket
            })

        # Firestore 클라이언트 (데이터베이스 이름: chatbot)
        self.db_firestore = firestore.client(database_id='chatbot')
        self.bucket = storage.bucket() if not skip_storage else None

        # Supabase (PostgreSQL) 연결
        self.pg_conn = psycopg2.connect(supabase_db_url)

        # 진행 상태 추적
        self.progress_file = Path(__file__).parent / 'migration_progress.json'
        self.progress = self._load_progress()

    def _load_progress(self) -> Dict:
        """이전 마이그레이션 진행 상태 로드"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'last_id': 0, 'migrated_count': 0, 'total_count': 0}

    def _save_progress(self):
        """마이그레이션 진행 상태 저장"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def get_total_count(self) -> int:
        """전체 마이그레이션 대상 개수 확인"""
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*)
                FROM rag_chunks
                WHERE id > %s
            """, (self.progress['last_id'],))
            count = cur.fetchone()[0]
        return count

    def fetch_batch(self, last_id: int) -> List[Dict[str, Any]]:
        """Supabase에서 배치 데이터 가져오기"""
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    id, season_year, season_id, league_type_code,
                    team_id, player_id, source_table, source_row_id,
                    title, content, embedding, meta, created_at
                FROM rag_chunks
                WHERE id > %s
                ORDER BY id
                LIMIT %s
            """, (last_id, self.batch_size))

            rows = cur.fetchall()
            return [dict(row) for row in rows]

    def migrate_chunk(self, chunk: Dict[str, Any]) -> bool:
        """단일 청크를 Firebase로 마이그레이션"""
        try:
            chunk_id = chunk['id']
            content = chunk.get('content', '')
            embedding = chunk.get('embedding')

            # A. Firebase Storage에 텍스트 저장 (옵션)
            storage_path = None
            if not self.skip_storage and content and self.bucket:
                blob_path = f"rag_chunks/{chunk['source_table']}/{chunk_id}.txt"
                blob = self.bucket.blob(blob_path)

                if not self.dry_run:
                    blob.upload_from_string(content, content_type='text/plain; charset=utf-8')

                storage_path = blob_path

            # B. Firebase Firestore에 벡터 및 메타데이터 저장
            doc_ref = self.db_firestore.collection('rag_chunks').document(str(chunk_id))

            firestore_data = {
                'id': chunk_id,
                'seasonYear': chunk.get('season_year'),
                'seasonId': chunk.get('season_id'),
                'leagueTypeCode': chunk.get('league_type_code'),
                'teamId': chunk.get('team_id'),
                'playerId': chunk.get('player_id'),
                'sourceTable': chunk['source_table'],
                'sourceRowId': chunk['source_row_id'],
                'title': chunk.get('title'),
                'meta': chunk.get('meta', {}),
                'createdAt': chunk.get('created_at'),
                'migratedAt': firestore.SERVER_TIMESTAMP,
            }

            # 임베딩 벡터 저장 (Firestore는 배열로 저장)
            if embedding:
                # PostgreSQL의 vector 타입은 문자열로 반환됨: "[0.1, 0.2, ...]"
                if isinstance(embedding, str):
                    embedding = json.loads(embedding.replace('[', '[').replace(']', ']'))
                firestore_data['embedding'] = embedding

            # Storage 경로 저장
            if storage_path:
                firestore_data['storagePath'] = storage_path
                firestore_data['contentLength'] = len(content)
            else:
                # Storage를 사용하지 않는 경우 content를 Firestore에 저장
                firestore_data['content'] = content

            if not self.dry_run:
                doc_ref.set(firestore_data)

            # C. Supabase 업데이트 (firebase_ref 추가, 옵션으로 content/embedding 삭제)
            # 주의: 공간 절약이 목적이라면 content와 embedding을 NULL로 설정
            # 하지만 안전을 위해 먼저 firebase_ref만 추가하고, 검증 후 삭제 권장
            if not self.dry_run:
                with self.pg_conn.cursor() as cur:
                    cur.execute("""
                        UPDATE rag_chunks
                        SET meta = meta || %s::jsonb
                        WHERE id = %s
                    """, (json.dumps({
                        'firebase_doc_id': str(chunk_id),
                        'firebase_migrated_at': datetime.now().isoformat(),
                        'firebase_storage_path': storage_path
                    }), chunk_id))
                    self.pg_conn.commit()

            return True

        except Exception as e:
            print(f"\n오류 발생 (chunk_id={chunk.get('id')}): {e}")
            return False

    def run(self):
        """마이그레이션 실행"""
        print(f"\n{'='*60}")
        print(f"Firebase 마이그레이션 시작")
        print(f"{'='*60}")
        print(f"배치 크기: {self.batch_size}")
        print(f"Dry Run: {self.dry_run}")
        print(f"Storage 스킵: {self.skip_storage}")
        print(f"마지막 처리 ID: {self.progress['last_id']}")
        print(f"이미 처리된 개수: {self.progress['migrated_count']}")

        # 전체 개수 확인
        total_count = self.get_total_count()
        self.progress['total_count'] = total_count

        if total_count == 0:
            print("\n마이그레이션할 데이터가 없습니다.")
            return

        print(f"남은 처리 개수: {total_count}")

        if self.dry_run:
            print("\n⚠️  Dry Run 모드: 실제 데이터는 변경되지 않습니다.\n")

        # 배치 처리
        with tqdm(total=total_count, desc="마이그레이션 진행") as pbar:
            last_id = self.progress['last_id']

            while True:
                # 배치 가져오기
                batch = self.fetch_batch(last_id)

                if not batch:
                    break

                # 배치 처리
                success_count = 0
                for chunk in batch:
                    if self.migrate_chunk(chunk):
                        success_count += 1
                        pbar.update(1)

                    last_id = chunk['id']

                # 진행 상태 저장
                self.progress['last_id'] = last_id
                self.progress['migrated_count'] += success_count
                self._save_progress()

                # 배치 크기보다 적게 가져왔으면 종료
                if len(batch) < self.batch_size:
                    break

        print(f"\n{'='*60}")
        print(f"마이그레이션 완료!")
        print(f"총 처리: {self.progress['migrated_count']}")
        print(f"{'='*60}\n")

        # 진행 파일 삭제 (완료 시)
        if not self.dry_run and self.progress_file.exists():
            self.progress_file.unlink()

    def cleanup_supabase(self, confirm: bool = False):
        """
        Supabase에서 content와 embedding 삭제 (공간 절약)

        ⚠️ 주의: Firebase 마이그레이션이 완전히 검증된 후에만 실행하세요!
        """
        if not confirm:
            print("⚠️  cleanup_supabase는 confirm=True로 명시적으로 호출해야 합니다.")
            return

        print("\n⚠️  Supabase에서 content와 embedding을 삭제합니다...")
        print("이 작업은 되돌릴 수 없습니다. 계속하려면 'YES'를 입력하세요: ", end='')

        response = input().strip()
        if response != 'YES':
            print("취소되었습니다.")
            return

        with self.pg_conn.cursor() as cur:
            # Firebase 마이그레이션이 완료된 항목만 삭제
            cur.execute("""
                UPDATE rag_chunks
                SET
                    content = NULL,
                    embedding = NULL
                WHERE meta ? 'firebase_doc_id'
            """)

            affected = cur.rowcount
            self.pg_conn.commit()

            print(f"✓ {affected}개 행의 content와 embedding을 삭제했습니다.")
            print("VACUUM FULL을 실행하여 디스크 공간을 회수하세요:")
            print("  psql $SUPABASE_DB_URL -c 'VACUUM FULL rag_chunks;'")

    def close(self):
        """리소스 정리"""
        self.pg_conn.close()


def main():
    parser = argparse.ArgumentParser(description='Supabase → Firebase 마이그레이션')
    parser.add_argument(
        '--service-account-key',
        required=True,
        help='Firebase 서비스 계정 키 JSON 파일 경로'
    )
    parser.add_argument(
        '--supabase-url',
        default=os.getenv('SUPABASE_DB_URL'),
        help='Supabase PostgreSQL 연결 URL (기본값: 환경변수 SUPABASE_DB_URL)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='배치 크기 (기본값: 100)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run 모드 (실제 데이터 변경 없음)'
    )
    parser.add_argument(
        '--skip-storage',
        action='store_true',
        help='Firebase Storage 업로드 스킵 (Firestore에만 저장)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='마이그레이션 후 Supabase에서 content/embedding 삭제 (주의!)'
    )

    args = parser.parse_args()

    if not args.supabase_url:
        print("오류: Supabase DB URL이 필요합니다. --supabase-url 또는 환경변수 SUPABASE_DB_URL을 설정하세요.")
        sys.exit(1)

    # 마이그레이션 실행
    migration = FirebaseMigration(
        service_account_key_path=args.service_account_key,
        supabase_db_url=args.supabase_url,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        skip_storage=args.skip_storage
    )

    try:
        migration.run()

        if args.cleanup and not args.dry_run:
            migration.cleanup_supabase(confirm=True)

    finally:
        migration.close()


if __name__ == '__main__':
    main()
