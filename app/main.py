"""FastAPI 애플리케이션의 생성 및 구성을 담당하는 메인 모듈입니다.

이 파일은 FastAPI 앱 인스턴스를 생성하고, 미들웨어, 라우터, 생명주기(lifespan) 이벤트 등
애플리케이션의 핵심 구성 요소를 설정하는 팩토리 함수를 포함합니다.
"""

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from .config import get_settings
from .deprecation import LegacyApiDeprecationMiddleware
from .deps import lifespan
from .internal_auth import require_ai_internal_token
from .routers import (
    chat_stream,
    search,
    ingest,
    vision,
    coach,
    coach_auto_brief_ops,
    moderation,
    release_decision,
)
from .streaming.http_errors import install_ai_stream_http_error_handler


def _include_internal_router(
    app: FastAPI,
    router: APIRouter,
    *,
    prefix: str = "",
    deprecated: bool | None = None,
) -> None:
    """내부 업무 라우터를 공통 토큰 인증과 함께 등록합니다."""
    app.include_router(
        router,
        prefix=prefix,
        dependencies=[Depends(require_ai_internal_token)],
        deprecated=deprecated,
    )


def create_app() -> FastAPI:
    """FastAPI 애플리케이션을 생성하고 구성합니다.

    Returns:
        구성된 FastAPI 애플리케이션 인스턴스.
    """
    settings = get_settings()  # 애플리케이션 설정 로드

    # [Security] 운영 환경에서 내부 토큰 오설정(미설정/공개 기본값) 시 기동 거부.
    settings.validate_internal_token_security()

    # Sentry Init
    import sentry_sdk

    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
        )

    # FastAPI 앱 인스턴스 생성 및 기본 설정
    api_docs_enabled = settings.api_docs_enabled
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,  # 애플리케이션 시작/종료 시 이벤트 처리
        docs_url="/docs" if api_docs_enabled else None,
        redoc_url="/redoc" if api_docs_enabled else None,
        openapi_url="/openapi.json" if api_docs_enabled else None,
    )
    install_ai_stream_http_error_handler(app)
    app.add_middleware(LegacyApiDeprecationMiddleware)

    # 브라우저의 FastAPI 직접 호출은 로컬 개발 또는 명시적 opt-in에서만 허용합니다.
    if settings.browser_direct_access_enabled and settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # API 라우터 등록
    # 각 라우터는 특정 기능 그룹(채팅, 검색, 데이터 수집)에 대한 엔드포인트를 정의합니다.
    _include_internal_router(app, chat_stream.router)
    _include_internal_router(app, search.router)
    _include_internal_router(app, ingest.router)
    _include_internal_router(app, vision.router, deprecated=True)
    _include_internal_router(app, vision.router, prefix="/ai")
    _include_internal_router(app, coach.router, deprecated=True)
    _include_internal_router(app, coach.router, prefix="/ai")
    _include_internal_router(app, coach_auto_brief_ops.router)
    _include_internal_router(app, moderation.router)
    _include_internal_router(app, release_decision.router)

    if settings.metrics_enabled:
        # Prometheus /metrics 엔드포인트 마운트
        # /ai/metrics와 /metrics 양쪽 모두 노출하여 운영자/스크레이퍼 호환성 확보
        try:
            from app.observability.http_metrics import install_http_metrics
            from app.observability.metrics import metrics_asgi_app

            install_http_metrics(app)
            app.mount("/metrics", metrics_asgi_app())
            app.mount("/ai/metrics", metrics_asgi_app())
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning(
                "[Observability] failed to mount /metrics endpoint: %s", exc
            )

    def _custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        security_schemes = openapi_schema.setdefault("components", {}).setdefault(
            "securitySchemes", {}
        )
        security_schemes.setdefault(
            "InternalApiKey",
            {
                "type": "apiKey",
                "in": "header",
                "name": "X-Internal-Api-Key",
                "description": (
                    "AI 내부 호출용 키. Authorization Bearer 토큰을 사용할 수 있습니다."
                ),
            },
        )

        public_paths = {"/health"}
        for path, operations in openapi_schema.get("paths", {}).items():
            if path not in public_paths:
                for operation in operations.values():
                    if not isinstance(operation, dict):
                        continue
                    security = operation.setdefault("security", [])
                    if {"InternalApiKey": []} not in security:
                        security.append({"InternalApiKey": []})

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = _custom_openapi

    @app.get("/health", tags=["system"])
    async def health():
        """애플리케이션의 상태를 확인하는 헬스 체크 엔드포인트."""
        return {"status": "ok"}

    return app


from .core.logging_config import configure_logging

# 로깅 설정 초기화
configure_logging()

app = create_app()
