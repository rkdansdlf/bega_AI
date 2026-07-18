from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from app.config import get_settings
from app.internal_auth import require_ai_internal_token


def _route_paths(app) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def _has_cors_middleware(app) -> bool:
    return any(middleware.cls is CORSMiddleware for middleware in app.user_middleware)


def _route_has_internal_auth(route: APIRoute) -> bool:
    return any(
        dependency.call is require_ai_internal_token
        for dependency in route.dependant.dependencies
    )


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


def test_internal_router_registration_protects_routes_without_endpoint_dependency():
    from app.main import _include_internal_router

    app = FastAPI()
    router = APIRouter(prefix="/probe")

    @router.get("/unprotected-at-endpoint")
    async def unprotected_at_endpoint():
        return {"ok": True}

    _include_internal_router(app, router)

    route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute)
        and route.path == "/probe/unprotected-at-endpoint"
    )
    assert _route_has_internal_auth(route)


def test_create_app_protects_every_business_api_route(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "local-test-token")
    get_settings.cache_clear()

    from app.main import create_app

    app = create_app()
    business_routes = [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path != "/health"
    ]

    assert business_routes
    assert all(_route_has_internal_auth(route) for route in business_routes)

    health_route = next(
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == "/health"
    )
    assert _route_has_internal_auth(health_route) is False


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
