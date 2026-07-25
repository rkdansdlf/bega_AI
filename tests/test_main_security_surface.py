from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.config import get_settings
from app.internal_auth import require_ai_internal_token


def _route_paths(app) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def _has_cors_middleware(app) -> bool:
    return any(middleware.cls is CORSMiddleware for middleware in app.user_middleware)


def test_create_app_disables_docs_and_metrics_by_default_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "a-strong-prod-token")
    monkeypatch.setenv("CORS_ORIGINS", "https://www.begabaseball.xyz")
    monkeypatch.delenv("AI_DOCS_ENABLED", raising=False)
    monkeypatch.delenv("AI_METRICS_ENABLED", raising=False)
    get_settings.cache_clear()

    from app.main import create_app

    app = create_app()
    paths = _route_paths(app)

    assert "/docs" not in paths
    assert "/redoc" not in paths
    assert "/openapi.json" not in paths
    assert "/metrics" not in paths
    assert "/ai/metrics" not in paths
    assert "/health" in paths
    assert _has_cors_middleware(app) is False


def test_create_app_keeps_metrics_when_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "a-strong-prod-token")
    monkeypatch.setenv("CORS_ORIGINS", "https://www.begabaseball.xyz")
    monkeypatch.setenv("AI_DOCS_ENABLED", "false")
    monkeypatch.setenv("AI_METRICS_ENABLED", "true")
    get_settings.cache_clear()

    from app.main import create_app

    app = create_app()
    paths = _route_paths(app)

    assert "/docs" not in paths
    assert "/metrics" in paths
    assert "/ai/metrics" in paths


def test_create_app_keeps_cors_for_local_direct_development(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "local-test-token")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5176")
    monkeypatch.delenv("AI_DIRECT_BROWSER_ACCESS_ENABLED", raising=False)
    get_settings.cache_clear()

    from app.main import create_app

    app = create_app()

    assert _has_cors_middleware(app) is True


def test_internal_router_registration_protects_routes_without_endpoint_dependency(
    monkeypatch,
):
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "local-test-token")
    get_settings.cache_clear()

    from app.main import _include_internal_router

    app = FastAPI()
    router = APIRouter(prefix="/probe")

    @router.get("/unprotected-at-endpoint")
    async def unprotected_at_endpoint():
        return {"ok": True}

    _include_internal_router(app, router)

    client = TestClient(app)
    assert client.get("/probe/unprotected-at-endpoint").status_code == 401
    assert (
        client.get(
            "/probe/unprotected-at-endpoint",
            headers={"X-Internal-Api-Key": "local-test-token"},
        ).status_code
        == 200
    )


def test_create_app_registers_every_business_router_with_internal_auth(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "local-test-token")
    get_settings.cache_clear()

    import app.main as main_module

    original_include_router = FastAPI.include_router
    registrations = []
    direct_registrations = []
    include_router_depth = 0
    direct_registration_depth = 0

    def record_router_registration(app, router, *args, **kwargs):
        nonlocal include_router_depth
        registrations.append((router, args, kwargs))
        include_router_depth += 1
        try:
            return original_include_router(app, router, *args, **kwargs)
        finally:
            include_router_depth -= 1

    monkeypatch.setattr(
        FastAPI,
        "include_router",
        record_router_registration,
    )

    direct_route_methods = (
        "api_route",
        "get",
        "post",
        "put",
        "patch",
        "delete",
        "options",
        "head",
        "trace",
        "websocket",
        "websocket_route",
        "add_api_route",
        "add_api_websocket_route",
    )
    for method_name in direct_route_methods:
        original_method = getattr(FastAPI, method_name)

        def record_direct_registration(
            app,
            path,
            *args,
            _method_name=method_name,
            _original_method=original_method,
            **kwargs,
        ):
            nonlocal direct_registration_depth
            is_outer_direct_registration = (
                include_router_depth == 0 and direct_registration_depth == 0
            )
            if is_outer_direct_registration:
                direct_registrations.append((_method_name, path))
            direct_registration_depth += 1
            try:
                return _original_method(app, path, *args, **kwargs)
            finally:
                direct_registration_depth -= 1

        monkeypatch.setattr(
            FastAPI,
            method_name,
            record_direct_registration,
        )

    app = main_module.create_app()

    assert [router for router, _args, _kwargs in registrations] == [
        main_module.chat_stream.router,
        main_module.search.router,
        main_module.ingest.router,
        main_module.vision.router,
        main_module.vision.router,
        main_module.coach.router,
        main_module.coach.router,
        main_module.coach_auto_brief_ops.router,
        main_module.moderation.router,
        main_module.release_decision.router,
    ]
    assert all(
        any(
            getattr(dependency, "dependency", None) is require_ai_internal_token
            for dependency in kwargs.get("dependencies", [])
        )
        for _router, _args, kwargs in registrations
    )
    assert direct_registrations == [("get", "/health")]

    assert TestClient(app).get("/health").status_code == 200


def test_openapi_marks_business_operations_as_internal(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "local-test-token")
    monkeypatch.setenv("AI_DOCS_ENABLED", "true")
    get_settings.cache_clear()

    from app.main import create_app

    schema = create_app().openapi()

    assert schema["paths"]["/ai/chat/completion"]["post"]["security"] == [
        {"InternalApiKey": []}
    ]
    assert "security" not in schema["paths"]["/health"]["get"]
    assert schema["components"]["securitySchemes"]["InternalApiKey"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-Internal-Api-Key",
        "description": (
            "AI 내부 호출용 키. Authorization Bearer 토큰을 사용할 수 있습니다."
        ),
    }


def test_deps_reexports_canonical_internal_auth_dependency():
    from app.deps import require_ai_internal_token as deps_auth

    assert deps_auth is require_ai_internal_token
