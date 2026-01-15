#!/usr/bin/env python3
"""
BEGA AI 챗봇 종합 진단 스크립트
실행: python test_diagnosis.py
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class BEGADiagnostics:
    def __init__(self):
        self.db_url = os.getenv("OCI_DB_URL")
        if not self.db_url:
            raise ValueError("OCI_DB_URL이 .env 파일에 없습니다!")
        
        self.conn = psycopg2.connect(self.db_url)
        self.results = []
    
    def print_header(self, title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    def print_result(self, test_name, passed, message):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {test_name}")
        if message:
            print(f"     → {message}")
        self.results.append((test_name, passed))
    
    def test_1_database_connection(self):
        """테스트 1: 데이터베이스 연결"""
        self.print_header("TEST 1: 데이터베이스 연결")
        
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            self.print_result(
                "DB 연결",
                True,
                f"PostgreSQL 연결 성공: {version[:50]}..."
            )
            cur.close()
        except Exception as e:
            self.conn.rollback()
            self.print_result("DB 연결", False, str(e))
    
    def test_2_rag_chunks_table(self):
        """테스트 2: rag_chunks 테이블 존재 및 데이터"""
        self.print_header("TEST 2: RAG Chunks 테이블")
        
        try:
            cur = self.conn.cursor()
            
            # 테이블 존재 확인
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'rag_chunks'
                )
            """)
            exists = cur.fetchone()[0]
            self.print_result("rag_chunks 테이블 존재", exists, None)
            
            if exists:
                # 총 데이터 개수
                cur.execute("SELECT COUNT(*) FROM rag_chunks")
                total_count = cur.fetchone()[0]
                self.print_result(
                    "총 청크 개수",
                    total_count > 0,
                    f"{total_count:,}개"
                )
                
                # 소스 테이블별 분포
                cur.execute("""
                    SELECT source_table, COUNT(*) as cnt
                    FROM rag_chunks
                    GROUP BY source_table
                    ORDER BY cnt DESC
                """)
                print("\n     [소스 테이블별 분포]")
                for row in cur.fetchall():
                    print(f"     - {row[0]}: {row[1]:,}개")
                
                # 임베딩 생성 확인
                cur.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(embedding) as with_embedding
                    FROM rag_chunks
                """)
                result = cur.fetchone()
                embedding_rate = (result[1] / result[0] * 100) if result[0] > 0 else 0
                self.print_result(
                    "임베딩 생성률",
                    embedding_rate > 90,
                    f"{embedding_rate:.1f}% ({result[1]:,}/{result[0]:,})"
                )
            
            cur.close()
        except Exception as e:
            self.conn.rollback()
            self.print_result("rag_chunks 테이블", False, str(e))
    
    def test_3_player_data(self):
        """테스트 3: 선수 데이터"""
        self.print_header("TEST 3: 선수 데이터")
        
        try:
            cur = self.conn.cursor()
            
            # player_basic 테이블
            cur.execute("SELECT COUNT(*) FROM player_basic")
            player_count = cur.fetchone()[0]
            self.print_result(
                "선수 기본 정보",
                player_count > 0,
                f"{player_count:,}명"
            )
            
            # 2025년 타격 데이터
            cur.execute("""
                SELECT COUNT(*) FROM player_season_batting 
                WHERE season = 2025
            """)
            batting_2025 = cur.fetchone()[0]
            self.print_result(
                "2025년 타격 데이터",
                batting_2025 > 0,
                f"{batting_2025:,}건"
            )
            
            # 2025년 투구 데이터
            cur.execute("""
                SELECT COUNT(*) FROM player_season_pitching 
                WHERE season = 2025
            """)
            pitching_2025 = cur.fetchone()[0]
            self.print_result(
                "2025년 투구 데이터",
                pitching_2025 > 0,
                f"{pitching_2025:,}건"
            )
            
            cur.close()
        except Exception as e:
            self.conn.rollback()
            self.print_result("선수 데이터", False, str(e))
    
    def test_4_kim_doyoung_search(self):
        """테스트 4: 김도영 선수 검색"""
        self.print_header("TEST 4: 김도영 선수 검색")
        
        try:
            cur = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # RAG chunks에서 검색
            cur.execute("""
                SELECT COUNT(*) FROM rag_chunks
                WHERE content ILIKE '%김도영%'
            """)
            rag_count = cur.fetchone()['count']
            self.print_result(
                "RAG에서 김도영 청크",
                rag_count > 0,
                f"{rag_count}개 발견"
            )
            
            # player_basic에서 검색
            cur.execute("""
                SELECT player_id, name, team, position
                FROM player_basic
                WHERE name LIKE '%김도영%'
            """)
            players = cur.fetchall()
            
            if players:
                print("\n     [발견된 선수]")
                for p in players:
                    print(f"     - {p['name']} ({p['player_id']}) | {p['team']} | {p['position']}")
                
                # 첫 번째 선수의 2025년 성적
                player_id = players[0]['player_id']
                cur.execute("""
                    SELECT 
                        psb.season,
                        t.team_name,
                        psb.avg,
                        psb.ops,
                        psb.home_runs,
                        psb.rbi,
                        psb.games
                    FROM player_season_batting psb
                    LEFT JOIN teams t ON psb.team_code = t.team_id
                    WHERE psb.player_id = %s
                    AND psb.season = 2025
                    AND psb.league = '정규시즌'
                    LIMIT 1
                """, (player_id,))
                
                stats = cur.fetchone()
                if stats:
                    print(f"\n     [2025년 성적]")
                    print(f"     - 소속: {stats['team_name']}")
                    print(f"     - 타율: {stats['avg']}")
                    print(f"     - OPS: {stats['ops']}")
                    print(f"     - 홈런: {stats['home_runs']}개")
                    print(f"     - 타점: {stats['rbi']}")
                    print(f"     - 경기: {stats['games']}경기")
                    self.print_result("2025년 성적 조회", True, "성공")
                else:
                    self.print_result("2025년 성적 조회", False, "데이터 없음")
            else:
                self.print_result("김도영 선수 찾기", False, "선수 정보 없음")
            
            cur.close()
        except Exception as e:
            self.conn.rollback()
            self.print_result("김도영 검색", False, str(e))
    
    def test_5_vector_search(self):
        """테스트 5: 벡터 검색 (임베딩)"""
        self.print_header("TEST 5: 벡터 검색 기능")
        
        try:
            cur = self.conn.cursor()
            
            # pgvector 확장 확인
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                )
            """)
            has_vector = cur.fetchone()[0]
            self.print_result("pgvector 확장", has_vector, None)
            
            # 벡터 인덱스 확인
            cur.execute("""
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'rag_chunks'
                AND indexname LIKE '%embedding%'
            """)
            indexes = cur.fetchall()
            self.print_result(
                "임베딩 인덱스",
                len(indexes) > 0,
                f"{len(indexes)}개 인덱스 발견"
            )
            
            cur.close()
        except Exception as e:
            self.conn.rollback()
            self.print_result("벡터 검색", False, str(e))
    
    def test_6_full_text_search(self):
        """테스트 6: Full-text 검색"""
        self.print_header("TEST 6: Full-text 검색")
        
        try:
            cur = self.conn.cursor()
            
            # tsvector 컬럼 확인
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns
                WHERE table_name = 'rag_chunks'
                AND column_name = 'content_tsv'
            """)
            has_tsv = cur.fetchone() is not None
            self.print_result("content_tsv 컬럼", has_tsv, None)
            
            # GIN 인덱스 확인
            cur.execute("""
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'rag_chunks'
                AND indexname LIKE '%tsv%'
            """)
            tsv_indexes = cur.fetchall()
            self.print_result(
                "Full-text 인덱스",
                len(tsv_indexes) > 0,
                f"{len(tsv_indexes)}개 발견"
            )
            
            # 실제 검색 테스트
            cur.execute("""
                SELECT COUNT(*) FROM rag_chunks
                WHERE content_tsv @@ plainto_tsquery('simple', '김도영')
            """)
            fts_count = cur.fetchone()[0]
            self.print_result(
                "Full-text 검색 테스트",
                fts_count > 0,
                f"'김도영' 검색 결과: {fts_count}건"
            )
            
            cur.close()
        except Exception as e:
            self.conn.rollback()
            self.print_result("Full-text 검색", False, str(e))
    
    def test_7_regulations_data(self):
        """테스트 7: 규정 문서 데이터"""
        self.print_header("TEST 7: KBO 규정 데이터")
        
        try:
            cur = self.conn.cursor()
            
            cur.execute("""
                SELECT COUNT(*) FROM rag_chunks
                WHERE source_table = 'kbo_regulations'
            """)
            reg_count = cur.fetchone()[0]
            self.print_result(
                "규정 문서 청크",
                reg_count > 0,
                f"{reg_count}개"
            )
            
            # 타이브레이크 검색 테스트
            cur.execute("""
                SELECT title FROM rag_chunks
                WHERE source_table = 'kbo_regulations'
                AND content ILIKE '%타이브레이크%'
                LIMIT 1
            """)
            result = cur.fetchone()
            self.print_result(
                "규정 검색 테스트",
                result is not None,
                f"타이브레이크 관련: {result[0] if result else 'N/A'}"
            )
            
            cur.close()
        except Exception as e:
            self.conn.rollback()
            self.print_result("규정 데이터", False, str(e))
    
    def test_8_api_endpoints(self):
        """테스트 8: API 엔드포인트"""
        self.print_header("TEST 8: API 엔드포인트")
        
        try:
            import requests
            
            base_url = "http://localhost:8001"
            
            # Health check
            try:
                r = requests.get(f"{base_url}/health", timeout=5)
                self.print_result(
                    "Health check",
                    r.status_code == 200,
                    f"Status: {r.status_code}"
                )
            except Exception as e:
                self.print_result("Health check", False, f"연결 실패: {e}")
            
            # Chat completion
            try:
                r = requests.post(
                    f"{base_url}/chat/completion",
                    json={"question": "안녕하세요"},
                    timeout=10
                )
                self.print_result(
                    "Chat completion",
                    r.status_code == 200,
                    f"Status: {r.status_code}"
                )
            except Exception as e:
                self.print_result("Chat completion", False, f"연결 실패: {e}")
            
        except ImportError:
            self.print_result("API 테스트", False, "requests 라이브러리 필요")
    
    def print_summary(self):
        """테스트 결과 요약"""
        self.print_header("테스트 결과 요약")
        
        total = len(self.results)
        passed = sum(1 for _, p in self.results if p)
        failed = total - passed
        
        print(f"\n총 테스트: {total}개")
        print(f"✅ 성공: {passed}개")
        print(f"❌ 실패: {failed}개")
        print(f"성공률: {passed/total*100:.1f}%\n")
        
        if failed > 0:
            print("실패한 테스트:")
            for name, passed in self.results:
                if not passed:
                    print(f"  - {name}")
        
        print("\n" + "="*60 + "\n")
    
    def run_all(self):
        """모든 테스트 실행"""
        print("\n" + "="*60)
        print("  BEGA AI 챗봇 종합 진단 시작")
        print("="*60)
        
        self.test_1_database_connection()
        self.test_2_rag_chunks_table()
        self.test_3_player_data()
        self.test_4_kim_doyoung_search()
        self.test_5_vector_search()
        self.test_6_full_text_search()
        self.test_7_regulations_data()
        self.test_8_api_endpoints()
        
        self.print_summary()
        
        self.conn.close()
        
        # 종료 코드 반환
        failed = sum(1 for _, p in self.results if not p)
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        diagnostics = BEGADiagnostics()
        exit_code = diagnostics.run_all()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
        sys.exit(1)