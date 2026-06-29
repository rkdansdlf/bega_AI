from app.config import get_settings


def _route_paths(app) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


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
