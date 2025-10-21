"""
설정 관리 파일
환경변수와 설정값을 중앙에서 관리합니다.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # ========================================
    # API 설정
    # ========================================
    MODEL_NAME: str = "gemini-2.5-flash"
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GEMINI_API_KEY: str = "AIzaSyDg4RCRH63rTBSHsF5UOfo7VTeleG3yX7g"
    
    # ========================================
    # MySQL 설정
    # ========================================
    MYSQL_HOST: str = "localhost"
    MYSQL_USER: str = "user1"
    MYSQL_PASSWORD: str = "1234"
    MYSQL_DATABASE: str = "api_test_data"
    MYSQL_PORT: int = 3306
    
    # ========================================
    # 서버 설정
    # ========================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    API_RELOAD: bool = True
    
    # ========================================
    # CORS 설정
    # ========================================
    CORS_ORIGINS: list = ["http://localhost:5173", "http://127.0.0.1:5173"]  # 실제 운영시 특정 도메인만 허용
    
    class Config:
        """Pydantic 설정"""
        env_file = ".env"  # .env 파일에서 환경변수 로드
        env_file_encoding = "utf-8"


# 싱글톤 인스턴스 생성
settings = Settings()


# 설정 확인 함수
def print_settings():
    """현재 설정 출력 (디버깅용)"""
    print("=" * 50)
    print("현재 설정:")
    print("=" * 50)
    print(f"MODEL: {settings.MODEL_NAME}")
    print(f"DB HOST: {settings.MYSQL_HOST}")
    print(f"DB NAME: {settings.MYSQL_DATABASE}")
    print(f"API PORT: {settings.API_PORT}")
    print("=" * 50)


if __name__ == "__main__":
    print_settings()