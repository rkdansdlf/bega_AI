"""
데이터베이스 관리 파일
MySQL 연결 및 쿼리 실행을 담당합니다.
"""

import mysql.connector
from mysql.connector import Error
import json
import datetime
from typing import Optional

from config import settings


class DatabaseManager:
    """MySQL 데이터베이스 관리 클래스"""
    
    def __init__(self):
        """데이터베이스 설정 초기화"""
        self.config = {
            'host': settings.MYSQL_HOST,
            'user': settings.MYSQL_USER,
            'password': settings.MYSQL_PASSWORD,
            'database': settings.MYSQL_DATABASE,
            'port': settings.MYSQL_PORT
        }
    
    def _default_serializer(self, obj):
        """
        날짜 객체를 JSON 직렬화 가능하게 변환
        
        Args:
            obj: 직렬화할 객체
            
        Returns:
            ISO 형식 문자열
        """
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )
    
    def get_connection(self):
        """
        데이터베이스 연결 생성
        
        Returns:
            MySQL connection 객체 또는 None
        """
        try:
            connection = mysql.connector.connect(**self.config)
            return connection
        except Error as e:
            print(f"[ERROR] DB 연결 실패: {e}")
            return None
    
    def execute_query(self, query: str) -> str:
        """
        SQL 쿼리 실행 및 결과 반환
        
        Args:
            query: 실행할 SQL SELECT 쿼리
            
        Returns:
            JSON 형식의 쿼리 결과 또는 에러 메시지
        """
        connection = None
        
        try:
            print(f"[DB] 쿼리 실행 시도: {query[:100]}...")
            
            connection = self.get_connection()
            
            if not connection:
                return "DATABASE ERROR: 데이터베이스 연결에 실패했습니다."
            
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            result_data = cursor.fetchall()
            cursor.close()
            
            print(f"[DB] 쿼리 성공: {len(result_data)}개의 결과 반환")
            
            return json.dumps(
                result_data, 
                ensure_ascii=False, 
                indent=2, 
                default=self._default_serializer
            )
            
        except mysql.connector.Error as e:
            error_msg = f"DATABASE ERROR: 쿼리 실행 중 오류가 발생했습니다. 오류: {e}"
            print(f"[ERROR] {error_msg}")
            return error_msg
        
        except Exception as e:
            error_msg = f"DATABASE ERROR: 예상치 못한 오류가 발생했습니다. 오류: {e}"
            print(f"[ERROR] {error_msg}")
            return error_msg
        
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    def test_connection(self) -> bool:
        """
        데이터베이스 연결 테스트
        
        Returns:
            연결 성공 여부
        """
        try:
            connection = self.get_connection()
            
            if connection and connection.is_connected():
                connection.close()
                print("[DB] 연결 테스트 성공")
                return True
            else:
                print("[DB] 연결 테스트 실패")
                return False
                
        except Exception as e:
            print(f"[ERROR] DB 연결 테스트 실패: {e}")
            return False
    
    def get_table_info(self) -> Optional[dict]:
        """
        game 테이블 정보 조회 (디버깅용)
        
        Returns:
            테이블 정보 딕셔너리 또는 None
        """
        connection = None
        
        try:
            connection = self.get_connection()
            
            if not connection:
                return None
            
            cursor = connection.cursor(dictionary=True)
            
            # 테이블 구조 확인
            cursor.execute("DESCRIBE game")
            columns = cursor.fetchall()
            
            # 레코드 수 확인
            cursor.execute("SELECT COUNT(*) as count FROM game")
            count = cursor.fetchone()
            
            cursor.close()
            
            return {
                "columns": columns,
                "record_count": count['count']
            }
            
        except Exception as e:
            print(f"[ERROR] 테이블 정보 조회 실패: {e}")
            return None
        
        finally:
            if connection and connection.is_connected():
                connection.close()


# 싱글톤 인스턴스 생성
db_manager = DatabaseManager()


# 테스트 코드
if __name__ == "__main__":
    print("=" * 50)
    print("데이터베이스 연결 테스트")
    print("=" * 50)
    
    # 연결 테스트
    if db_manager.test_connection():
        print("✅ 데이터베이스 연결 성공")
        
        # 테이블 정보 출력
        table_info = db_manager.get_table_info()
        if table_info:
            print(f"\n📊 game 테이블 정보:")
            print(f"레코드 수: {table_info['record_count']}")
            print(f"컬럼 수: {len(table_info['columns'])}")
            print("\n컬럼 목록:")
            for col in table_info['columns']:
                print(f"  - {col['Field']} ({col['Type']})")
    else:
        print("❌ 데이터베이스 연결 실패")
    
    print("=" * 50)