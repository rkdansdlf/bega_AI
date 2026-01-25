"""FastAPI 애플리케이션의 생성 및 구성을 담당하는 메인 모듈입니다.

이 파일은 FastAPI 앱 인스턴스를 생성하고, 미들웨어, 라우터, 생명주기(lifespan) 이벤트 등
애플리케이션의 핵심 구성 요소를 설정하는 팩토리 함수를 포함합니다.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .deps import lifespan
from .routers import chat_stream, search, ingest, vision, coach, moderation


def create_app() -> FastAPI:
    """FastAPI 애플리케이션을 생성하고 구성합니다.

    Returns:
        구성된 FastAPI 애플리케이션 인스턴스.
    """
    settings = get_settings()  # 애플리케이션 설정 로드

    # Sentry Init
    import sentry_sdk

    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
        )

    # FastAPI 앱 인스턴스 생성 및 기본 설정
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,  # 애플리케이션 시작/종료 시 이벤트 처리
    )

    # CORS(Cross-Origin Resource Sharing) 미들웨어 추가
    # 다른 도메인에서의 요청을 허용하기 위한 설정입니다.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,  # 허용할 출처 목록
        allow_credentials=True,  # 자격 증명(쿠키 등) 허용
        allow_methods=["*"],  # 모든 HTTP 메소드 허용
        allow_headers=["*"],  # 모든 HTTP 헤더 허용
    )

    # API 라우터 등록
    # 각 라우터는 특정 기능 그룹(채팅, 검색, 데이터 수집)에 대한 엔드포인트를 정의합니다.
    app.include_router(chat_stream.router)
    app.include_router(search.router)
    app.include_router(ingest.router)
    app.include_router(vision.router)
    app.include_router(coach.router)
    app.include_router(moderation.router)

    @app.get("/health", tags=["system"])
    async def health():
        """애플리케이션의 상태를 확인하는 헬스 체크 엔드포인트."""
        return {"status": "ok"}

    return app


app = create_app()
