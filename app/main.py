"""FastAPI 애플리케이션 인스턴스를 생성하는 팩토리 함수 모듈."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .deps import lifespan
from .routers import chat_stream, search, ingest


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_stream.router)
    app.include_router(search.router)
    app.include_router(ingest.router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
